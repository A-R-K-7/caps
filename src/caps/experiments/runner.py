"""
Experiment Runner — the main execution loop.

Runs a configured experiment:
    dataset → planner → CAPS → receiver → evaluator → update → database

Supports:
    - Multiple policies (baselines + CAPS)
    - Multiple seeds
    - Checkpointing and resumption
    - Budget enforcement
    - Progress display
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from caps.agents.evaluator import EvaluatorAgent
from caps.agents.planner import PlannerAgent
from caps.agents.receiver import ReceiverAgent, ReceiverOutput
from caps.bandits.linucb import LinUCBPolicy
from caps.communication.baselines import (
    EmbeddingSimilarityPolicy,
    FullContextPolicy,
    LLMPlannerPolicy,
    RandomPolicy,
    StaticHeuristicPolicy,
)
from caps.communication.ciu import CIU
from caps.communication.policy import CommunicationPolicy
from caps.communication.state import Action, CommunicationState
from caps.config import ExperimentConfig
from caps.datasets.generator import generate_dataset, save_dataset
from caps.embeddings.local import LocalEmbeddingProvider
from caps.environments.synthetic import SyntheticEnvironment
from caps.evaluation.metrics import compute_all_metrics
from caps.evaluation.reward import RewardFunction
from caps.llm import LLMProvider, LLMResponse
from caps.llm.budget import BudgetManager
from caps.storage.database import ExperimentDB

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Main experiment runner.

    Manages the full MDP loop for CAPS experiments.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        llm_provider: LLMProvider,
    ):
        self.config = config
        self.llm = llm_provider

        # Initialize components
        self.embedder = LocalEmbeddingProvider(
            model_name=config.embedding_model,
            cache_path=config.cache.embedding_cache_path if config.cache.enabled else None,
        )
        self.planner = PlannerAgent(
            llm_provider=llm_provider,
            embedding_provider=self.embedder,
            model=config.models.planner,
            temperature=config.temperature,
            seed=config.seed,
        )
        self.receiver = ReceiverAgent(
            llm_provider=llm_provider,
            model=config.models.receiver,
            temperature=config.temperature,
            seed=config.seed,
        )
        self.evaluator = EvaluatorAgent(
            receiver_agent=self.receiver,
            max_counterfactuals=config.evaluator.max_counterfactuals_per_task,
            strategy=config.evaluator.counterfactual_strategy,
            seed=config.seed,
        )
        self.reward_fn = RewardFunction(
            lambda_cost=config.reward.lambda_cost,
            w_tokens=config.reward.weights_tokens,
            w_latency=config.reward.weights_latency,
            w_clarification=config.reward.weights_clarification,
            w_retries=config.reward.weights_retries,
        )
        self.budget = BudgetManager(
            max_daily_tokens=config.budget.max_daily_tokens,
            max_requests_per_run=config.budget.max_requests_per_run,
            max_tokens_per_task=config.budget.max_tokens_per_task,
        )
        self.db = ExperimentDB(config.database_path)
        
        if self.config.dataset.name.startswith("swe_bench"):
            from caps.environments.swe_bench import SWEBenchEnvironment
            self.env = SWEBenchEnvironment(
                llm_provider=self.llm,
                model_name=self.config.models.evaluator
            )
        else:
            from caps.environments.synthetic import SyntheticEnvironment
            self.env = SyntheticEnvironment()

    def build_policies(self) -> Dict[str, CommunicationPolicy]:
        """Construct all policies to evaluate."""
        policies: Dict[str, CommunicationPolicy] = {}

        # CAPS LinUCB
        policies["CAPS"] = LinUCBPolicy(
            alpha=self.config.policy.alpha,
            selection_strategy=self.config.policy.selection_strategy,
            epsilon=self.config.policy.epsilon,
        )
        
        # Baselines
        policies["LLMPlanner"] = LLMPlannerPolicy(
            llm_provider=self.planner.llm,
            model=self.config.models.planner
        )
        policies["Heuristic"] = StaticHeuristicPolicy()
        policies["Random"] = RandomPolicy(seed=self.config.seed)
        policies["FullContext"] = FullContextPolicy()

        return policies

    def run(self) -> Dict[str, Dict[str, float]]:
        """
        Run the full experiment across all policies and seeds.

        Returns:
            Dict mapping policy_name → aggregate metrics.
        """
        # Generate or load dataset
        if self.config.dataset.name.startswith("swe_bench"):
            from caps.datasets.swe_bench import generate_swebench_dataset
            tasks = generate_swebench_dataset(
                num_tasks=self.config.dataset.test_size,
                seed=self.config.dataset.seed,
            )
        else:
            tasks = generate_dataset(
                families=self.config.dataset.families,
                size=(
                    self.config.dataset.train_size
                    + self.config.dataset.validation_size
                    + self.config.dataset.test_size
                ),
                seed=self.config.dataset.seed,
            )

        # Save dataset
        dataset_path = f"data/synthetic/{self.config.dataset.name}.jsonl"
        save_dataset(tasks, dataset_path)

        self.env.load_tasks(tasks)

        policies = self.build_policies()
        all_results: Dict[str, Dict[str, float]] = {}

        for policy_name, policy in policies.items():
            logger.info(f"Running policy: {policy_name}")
            experiment_id = f"{self.config.name}_{policy_name}_{self.config.seed}"

            existing_episodes = self.db.count_episodes(experiment_id)
            
            if existing_episodes > 0:
                logger.info(f"Resuming experiment {experiment_id} from episode {existing_episodes}/{self.env.num_tasks}")
                self.db.update_experiment(experiment_id, status="running")
            else:
                # Clean up any empty/aborted records
                self.db.delete_experiment(experiment_id)

                # Create experiment record
                self.db.create_experiment(
                    experiment_id=experiment_id,
                    name=f"{self.config.name} / {policy_name}",
                    config_yaml=json.dumps(self.config.model_dump(), default=str),
                    seed=self.config.seed,
                    policy_name=policy_name,
                    planner_model=self.config.models.planner,
                    receiver_model=self.config.models.receiver,
                    dataset_name=self.config.dataset.name,
                    total_episodes=self.env.num_tasks,
                )

            # Reset budget for this run
            self.budget.reset_run()

            # Run episodes
            results = self._run_policy(
                policy=policy,
                policy_name=policy_name,
                experiment_id=experiment_id,
                start_idx=existing_episodes,
            )

            all_results[policy_name] = results

            # Update experiment status
            self.db.update_experiment(
                experiment_id=experiment_id,
                status="completed",
                completed_episodes=self.env.num_tasks,
            )

            # Store aggregate metrics
            for metric_name, metric_value in results.items():
                self.db.add_metric(experiment_id, metric_name, metric_value)

        return all_results

    def _run_policy(
        self,
        policy: CommunicationPolicy,
        policy_name: str,
        experiment_id: str,
        start_idx: int = 0,
    ) -> Dict[str, float]:
        """Run all episodes for a single policy."""
        successes: List[bool] = []
        token_counts: List[int] = []
        clarifications: List[bool] = []
        retries: List[int] = []
        rewards: List[float] = []

        for task_idx in range(start_idx, self.env.num_tasks):
            # Reset problem context to allow API key switching on the first request if rate limited
            for agent in [self.planner, self.receiver, self.evaluator]:
                if hasattr(agent, "llm") and hasattr(agent.llm, "reset_problem_context"):
                    agent.llm.reset_problem_context()

            # Reset environment
            task_data = self.env.reset(task_idx)
            task_desc = self.env.get_task_description()
            task_meta = self.env.task_metadata()
            cius = self.env.get_cius()

            # Budget check
            estimated_tokens = sum(c.estimate_tokens() for c in cius)
            can_proceed, reason = self.budget.can_make_request(
                estimated_tokens, task_meta["task_id"]
            )
            if not can_proceed:
                logger.warning(f"Budget exceeded: {reason}. Stopping.")
                break

            start_time = time.time()

            # 1. Build communication state
            state = self.planner.build_state(
                task=task_desc,
                cius=cius,
                token_budget=self.config.budget.max_tokens_per_message,
                receiver_role=task_meta.get("receiver_role", "general"),
                task_category=task_meta.get("category", "unknown"),
                difficulty=task_meta.get("difficulty", "medium"),
                receiver_model=self.config.models.receiver,
            )

            # 2. Policy selects CIUs
            action = policy.select(state)
            ciu_map = {c.id: c for c in cius}
            selected_cius = [ciu_map[cid] for cid in action.selected_ciu_ids if cid in ciu_map]

            # 3. Receiver executes
            receiver_output = self.receiver.execute(
                task=task_desc,
                selected_cius=selected_cius,
                receiver_role=task_meta.get("receiver_role", "general"),
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # 4. Evaluate task success (deterministic)
            success = self.env.evaluate(receiver_output.answer)

            # 5. Calculate reward
            clarification = receiver_output.status == "needs_clarification"
            reward = self.reward_fn.calculate(
                task_success=success,
                state=state,
                action=action,
                latency_ms=elapsed_ms,
                clarification_turns=1 if clarification else 0,
            )

            # 6. Evaluation (VOI)
            voi_scores = {}
            if policy_name == "CAPS":
                if self.config.evaluator.counterfactual:
                    # O(N) Counterfactual evaluation
                    voi_scores = self.evaluator.evaluate_voi(
                        task_desc=task_desc,
                        selected_cius=selected_cius,
                        original_success=success,
                        env=self.env,
                        receiver_role=task_meta.get("receiver_role", "general"),
                    )
                else:
                    # O(1) Self-reflection evaluation
                    voi_scores = self.evaluator.evaluate_self_reflection(
                        selected_cius=selected_cius,
                        used_ciu_ids=receiver_output.used_ciu_ids,
                        original_success=success,
                    )

            # 7. Update policy (learning!)
            policy.update(state, action, reward, voi_scores=voi_scores)

            # 8. Record usage
            tokens_used = sum(c.estimate_tokens() for c in selected_cius)
            self.budget.record_usage(tokens_used, tokens_used // 2, tokens_used,
                                      task_meta["task_id"])

            # 8. Store episode
            episode_id = f"{experiment_id}_ep{task_idx:04d}"
            self.db.add_episode(
                experiment_id=experiment_id,
                episode_id=episode_id,
                task_id=task_meta["task_id"],
                seed=self.config.seed,
                task_category=task_meta.get("category"),
                planner_model=self.config.models.planner,
                receiver_model=self.config.models.receiver,
                available_ciu_ids=[c.id for c in cius],
                selected_ciu_ids=action.selected_ciu_ids,
                num_available_cius=len(cius),
                num_selected_cius=len(selected_cius),
                total_tokens=tokens_used,
                latency_ms=elapsed_ms,
                task_success=success,
                receiver_status=receiver_output.status,
                clarification_turns=1 if clarification else 0,
                reward=reward,
                voi_scores=voi_scores,
                policy_version=task_idx,
            )

            # Accumulate
            successes.append(success)
            token_counts.append(tokens_used)
            clarifications.append(clarification)
            retries.append(0)
            rewards.append(reward)

            # Progress display
            if self.config.verbose or task_idx % 10 == 0:
                budget_remaining = self.budget.remaining()
                logger.info(
                    f"Episode {task_idx + 1:03d}/{self.env.num_tasks} | "
                    f"Policy: {policy_name} | "
                    f"Task: {task_meta['task_id']} | "
                    f"Selected: {len(selected_cius)}/{len(cius)} CIUs | "
                    f"Tokens: {tokens_used} | "
                    f"Success: {'YES' if success else 'NO'} | "
                    f"Reward: {reward:.3f} | "
                    f"Budget: {budget_remaining['daily_tokens_remaining']}"
                )

        # Compute aggregate metrics
        return compute_all_metrics(
            successes=successes,
            token_counts=token_counts,
            clarifications=clarifications,
            retries=retries,
        )
