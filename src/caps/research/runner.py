"""
Research Runner — paired train/validation/frozen-test experiment loop.

Design principles:
- Completely separate from ExperimentRunner and the existing caps_experiments.db.
- Uses synthetic_v2 tasks with hidden ground-truth labels.
- Runs 7 methods × N budget fractions × N seeds over the full split.
- Test episodes are frozen (no LinUCB update).
- Checkpoints LinUCB state after every training episode.
- Writes CSV instrumentation via ResearchLogger.

Methods:
    FullContext-Unbounded  — sends all CIUs (no budget limit)
    FullContext-Budgeted   — sends CIUs in order up to budget
    Random                 — budgeted random selection
    Heuristic              — type-priority budgeted selection
    LLMPlanner             — LLM or deterministic proxy selection
    CAPS-NoVOI             — LinUCB, global reward only
    CAPS-VOI               — LinUCB + VOI signal from synthetic_v2 labels
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from caps.agents.evaluator import EvaluatorAgent
from caps.agents.planner import PlannerAgent
from caps.agents.receiver import ReceiverAgent
from caps.bandits.linucb import LinUCBPolicy
from caps.communication.baselines import (
    BudgetedFullContextPolicy,
    CAPSNoVOIPolicy,
    EmbeddingSimilarityPolicy,
    FullContextPolicy,
    LLMPlannerPolicy,
    RandomPolicy,
    StaticHeuristicPolicy,
)
from caps.communication.policy import CommunicationPolicy
from caps.communication.state import Action, CommunicationState
from caps.datasets.synthetic_v2 import (
    create_splits,
    dataset_hash,
    generate_synthetic_v2_dataset,
    save_jsonl,
    save_splits,
)
from caps.embeddings.local import LocalEmbeddingProvider
from caps.environments.synthetic_v2 import SyntheticV2Environment
from caps.evaluation.reward import RewardFunction
from caps.llm import LLMProvider
from caps.llm.budget import BudgetManager
from caps.research.instrumentation import (
    DecisionRow,
    EpisodeRow,
    ResearchLogger,
    UpdateRow,
)

logger = logging.getLogger(__name__)

# Default budget fractions for the pilot (full: add 0.75, 0.15)
DEFAULT_BUDGET_FRACTIONS = [1.0, 0.5, 0.3]

# Default number of tasks in the dataset
DEFAULT_DATASET_SIZE = 35   # 21 train / 7 val / 7 test


class ResearchRunner:
    """
    Main research pipeline runner for the synthetic_v2 benchmark.

    Usage::

        runner = ResearchRunner(config=cfg, llm_provider=llm)
        runner.run()

    All output goes to:
        research/results/<run_id>_episodes.csv
        research/results/<run_id>_decisions.csv
        research/results/<run_id>_updates.csv
        research/checkpoints/<run_id>_<method>_<seed>_policy.jsonl
        research/manifests/<run_id>_manifest.json
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        # Dataset
        dataset_size: int = DEFAULT_DATASET_SIZE,
        dataset_seed: int = 1,
        # Budget
        budget_fractions: Optional[List[float]] = None,
        max_tokens_per_task: int = 8_000,
        # Policy
        linucb_alpha: float = 1.0,
        # Seeds
        seeds: Optional[List[int]] = None,
        # Methods (all 7 if None)
        methods: Optional[List[str]] = None,
        # Paths
        results_dir: str = "research/results",
        checkpoint_dir: str = "research/checkpoints",
        manifest_dir: str = "research/manifests",
        data_dir: str = "data/research",
        embedding_model: str = "all-MiniLM-L6-v2",
        # Misc
        verbose: bool = False,
        resume_run_id: Optional[str] = None,
    ):
        self.llm = llm_provider
        self.dataset_size = dataset_size
        self.dataset_seed = dataset_seed
        self.budget_fractions = budget_fractions or DEFAULT_BUDGET_FRACTIONS
        self.max_tokens_per_task = max_tokens_per_task
        self.linucb_alpha = linucb_alpha
        self.seeds = seeds or [1, 2, 3]
        self.methods = methods or [
            "FullContext-Unbounded",
            "FullContext-Budgeted",
            "Random",
            "Heuristic",
            "LLMPlanner",
            "CAPS-NoVOI",
            "CAPS-VOI",
            "CAPS-OracleVOI",
        ]
        self.results_dir = results_dir
        self.checkpoint_dir = checkpoint_dir
        self.manifest_dir = manifest_dir
        self.data_dir = data_dir
        self.verbose = verbose

        # Create output directories
        for d in [results_dir, checkpoint_dir, manifest_dir, data_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Shared components
        self.embedder = LocalEmbeddingProvider(
            model_name=embedding_model,
            cache_path=None,  # no cache for research run to keep evidence clean
        )
        self.planner = PlannerAgent(
            llm_provider=llm_provider,
            embedding_provider=self.embedder,
        )
        self.reward_fn = RewardFunction(
            lambda_cost=0.5,
            w_tokens=0.4,
            w_latency=0.2,
            w_clarification=0.2,
            w_retries=0.2,
        )

        if resume_run_id:
            self._run_id = resume_run_id
        else:
            self._run_id = f"research_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the full research experiment.

        Returns:
            Summary dict with run_id, method results, and file paths.
        """
        # 1. Generate dataset
        logger.info("Generating synthetic_v2 dataset (size=%d, seed=%d)…",
                    self.dataset_size, self.dataset_seed)
        tasks = generate_synthetic_v2_dataset(
            size=self.dataset_size,
            seed=self.dataset_seed,
        )
        d_hash = dataset_hash(tasks)

        # 2. Create & persist splits
        splits = create_splits(tasks, seed=self.dataset_seed)
        splits_path = f"{self.data_dir}/splits_{self.dataset_seed}.json"
        save_splits(splits, splits_path)

        # 3. Persist full dataset (with hidden labels — for internal audit)
        dataset_path = f"{self.data_dir}/synthetic_v2_full.jsonl"
        save_jsonl(tasks, dataset_path, visible_only=False)

        train_ids = splits["splits"]["train"]
        val_ids = splits["splits"]["validation"]
        test_ids = splits["splits"]["test"]

        task_by_id = {t["task_id"]: t for t in tasks}

        logger.info(
            "Dataset split: train=%d  val=%d  test=%d  hash=%s…",
            len(train_ids), len(val_ids), len(test_ids), d_hash[:12],
        )

        # 4. Run methods × budget fractions × seeds
        all_results: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []

        # completed_splits maps experiment_id -> set of completed split names
        # A split is "complete" if it has >= the expected number of episodes for that split
        split_sizes = {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
        }
        completed_splits: Dict[str, set] = {}  # key -> {"train", "validation", "test"}
        completed_keys = set()
        if self._run_id:
            ep_path = Path(self.results_dir) / f"{self._run_id}_episodes.csv"
            if ep_path.exists():
                import csv
                # Track (experiment_id, split) -> count
                split_counts: Dict[str, Dict[str, int]] = {}
                with ep_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        eid = row["experiment_id"]
                        sp = row.get("split", "")
                        if eid not in split_counts:
                            split_counts[eid] = {}
                        split_counts[eid][sp] = split_counts[eid].get(sp, 0) + 1
                for eid, sp_counts in split_counts.items():
                    done_splits = set()
                    for sp, expected in split_sizes.items():
                        if sp_counts.get(sp, 0) >= expected:
                            done_splits.add(sp)
                    completed_splits[eid] = done_splits
                    if done_splits == {"train", "validation", "test"}:
                        completed_keys.add(eid)
                logger.info(
                    "Resuming run %s: %d fully completed keys, %d partially done.",
                    self._run_id, len(completed_keys),
                    sum(1 for s in completed_splits.values() if s and s != {"train", "validation", "test"}),
                )

        with ResearchLogger(self._run_id, self.results_dir, append=True) as rlog:
            for seed in self.seeds:
                for budget_frac in self.budget_fractions:
                    # Compute absolute token budget (based on average task token count)
                    avg_tokens = _estimate_avg_tokens(tasks)
                    budget_tokens = max(
                        100, int(avg_tokens * budget_frac)
                    )

                    for method_name in self.methods:
                        key = f"{method_name}_bfrac{budget_frac}_seed{seed}"

                        if key in completed_keys:
                            logger.info("Skipping already completed: %s", key)
                            continue

                        # Determine which splits still need to run
                        already_done_splits = completed_splits.get(key, set())
                        logger.info("Running: %s (skipping splits: %s)", key, already_done_splits)

                        result = self._run_method(
                            method_name=method_name,
                            task_by_id=task_by_id,
                            train_ids=train_ids,
                            val_ids=val_ids,
                            test_ids=test_ids,
                            budget_frac=budget_frac,
                            budget_tokens=budget_tokens,
                            seed=seed,
                            rlog=rlog,
                            skip_splits=already_done_splits,
                        )
                        all_results[key] = result
                        summary_rows.append({
                            "method": method_name,
                            "budget_fraction": budget_frac,
                            "budget_tokens": budget_tokens,
                            "seed": seed,
                            **result,
                        })

        # 5. Persist manifest
        manifest = {
            "run_id": self._run_id,
            "dataset_size": self.dataset_size,
            "dataset_seed": self.dataset_seed,
            "dataset_hash": d_hash,
            "splits_path": str(splits_path),
            "methods": self.methods,
            "seeds": self.seeds,
            "budget_fractions": self.budget_fractions,
            "results": all_results,
        }
        manifest_path = f"{self.manifest_dir}/{self._run_id}_manifest.json"
        Path(manifest_path).write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        logger.info("Run complete. Manifest: %s", manifest_path)
        logger.info("Episodes CSV: %s/%s_episodes.csv", self.results_dir, self._run_id)

        return {
            "run_id": self._run_id,
            "summary_rows": summary_rows,
            "manifest_path": manifest_path,
            "episodes_csv": f"{self.results_dir}/{self._run_id}_episodes.csv",
        }

    # ------------------------------------------------------------------
    # Per-method loop
    # ------------------------------------------------------------------

    def _run_method(
        self,
        method_name: str,
        task_by_id: Dict[str, Any],
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        budget_frac: float,
        budget_tokens: int,
        seed: int,
        rlog: ResearchLogger,
        skip_splits: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Run a single (method, budget, seed) combination across all splits.

        Args:
            skip_splits: Set of split names to skip (already completed on a prior run).

        Returns aggregate metrics for test split.
        """
        if skip_splits is None:
            skip_splits = set()

        policy = _make_policy(
            method_name=method_name,
            llm_provider=self.llm,
            seed=seed,
            alpha=self.linucb_alpha,
        )

        env = SyntheticV2Environment()
        receiver = ReceiverAgent(llm_provider=self.llm, model=self.llm.default_model)

        split_results: Dict[str, Dict[str, float]] = {}

        # Checkpoint path for CAPS variants
        ckpt_path: Optional[Path] = None
        if method_name in ("CAPS-NoVOI", "CAPS-VOI", "CAPS-OracleVOI"):
            ckpt_name = f"{self._run_id}_{method_name.replace('-','_')}_seed{seed}_bfrac{budget_frac}_policy.jsonl"
            ckpt_path = Path(self.checkpoint_dir) / ckpt_name
            # Only truncate checkpoint if starting fresh (not resuming a partial)
            if not skip_splits:
                ckpt_path.open("w").close()  # truncate / create
            else:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt_path.touch()  # ensure file exists for appending

        for split_name, task_ids in [
            ("train", train_ids),
            ("validation", val_ids),
            ("test", test_ids),
        ]:
            if split_name in skip_splits:
                logger.info("  Skipping already-completed split: %s/%s", method_name, split_name)
                # Fill in placeholder split_results so test aggregation works
                split_results[split_name] = {"n": 0, "success_rate": 0.0, "mean_tokens": 0.0,
                                              "mean_reward": 0.0, "mean_precision": 0.0,
                                              "mean_recall": 0.0, "mean_misleading_rate": 0.0}
                continue
            is_frozen = split_name == "test"
            tasks_in_split = [task_by_id[tid] for tid in task_ids if tid in task_by_id]
            env.load_tasks(tasks_in_split)

            successes: List[bool] = []
            token_counts: List[int] = []
            rewards: List[float] = []
            precisions: List[float] = []
            recalls: List[float] = []
            misleading_rates: List[float] = []

            episode_counter = 0

            for task_idx in range(env.num_tasks):
                # Reset LLM problem context so key-rotation fires on 429
                # (not mid-problem wait) between tasks
                self.llm.reset_problem_context()

                task_data = env.reset(task_idx)
                task_desc = env.get_task_description()
                task_meta = env.task_metadata()
                cius = env.get_cius()

                if not cius:
                    continue

                # --- Build state ---
                state = self.planner.build_state(
                    task=task_desc,
                    cius=cius,
                    token_budget=budget_tokens,
                    receiver_role=task_meta.get("receiver_role", "general"),
                    task_category=task_meta.get("category", "unknown"),
                    difficulty=task_meta.get("difficulty", "medium"),
                )

                # --- Select CIUs ---
                t0 = time.time()
                action = policy.select(state)
                ciu_map = {c.id: c for c in cius}
                selected_cius = [
                    ciu_map[cid] for cid in action.selected_ciu_ids if cid in ciu_map
                ]

                # --- Receiver executes ---
                receiver_output = receiver.execute(
                    task=task_desc,
                    selected_cius=selected_cius,
                    receiver_role=task_meta.get("receiver_role", "general"),
                )
                elapsed_ms = (time.time() - t0) * 1000.0

                # --- Evaluate ---
                success = env.evaluate(receiver_output.answer)
                sq = env.selection_quality(action.selected_ciu_ids)

                clarification = receiver_output.status == "needs_clarification"
                reward = self.reward_fn.calculate(
                    task_success=success,
                    state=state,
                    action=action,
                    latency_ms=elapsed_ms,
                    clarification_turns=1 if clarification else 0,
                )

                # --- VOI (CAPS-VOI and CAPS-OracleVOI only) ---
                voi_scores: Dict[str, float] = {}
                if method_name == "CAPS-OracleVOI":
                    # Oracle condition using hidden ground-truth labels
                    voi_scores = _derive_synthetic_voi(
                        task_data=task_data,
                        selected_ids=action.selected_ciu_ids,
                    )
                elif method_name == "CAPS-VOI":
                    # True non-leaking mechanism: Receiver self-reported usage
                    # +1.0 for used, -0.5 for selected but not used.
                    for cid in action.selected_ciu_ids:
                        if cid in receiver_output.used_ciu_ids:
                            voi_scores[cid] = 1.0
                        else:
                            voi_scores[cid] = -0.5

                # --- Update policy (train split only, not frozen on test) ---
                episode_counter += 1
                if not is_frozen:
                    policy.update(state, action, reward, voi_scores=voi_scores)

                    # Checkpoint CAPS arms after every train update
                    if ckpt_path is not None and hasattr(policy, "get_state_dict"):
                        inner = getattr(policy, "_inner", policy)
                        with ckpt_path.open("a", encoding="utf-8") as ckf:
                            ckf.write(json.dumps({
                                "episode": episode_counter,
                                "split": split_name,
                                "state_dict": inner.get_state_dict() if hasattr(inner, "get_state_dict") else {},
                            }) + "\n")

                # --- Metrics ---
                available_tokens = sum(c.estimate_tokens() for c in cius)
                selected_tokens = sum(c.estimate_tokens() for c in selected_cius)
                compression = selected_tokens / available_tokens if available_tokens > 0 else 0.0

                api_prompt = getattr(receiver_output.llm_usage, "prompt_tokens", 0) if receiver_output.llm_usage else 0
                api_completion = getattr(receiver_output.llm_usage, "completion_tokens", 0) if receiver_output.llm_usage else 0
                if hasattr(action, "llm_usage") and action.llm_usage:
                    api_prompt += getattr(action.llm_usage, "prompt_tokens", 0)
                    api_completion += getattr(action.llm_usage, "completion_tokens", 0)

                episode_id = (
                    f"{self._run_id}_{method_name}_{seed}_bfrac{budget_frac}_{split_name}_ep{task_idx:04d}"
                )

                # Log episode row
                rlog.log_episode(EpisodeRow(
                    run_id=self._run_id,
                    experiment_id=f"{method_name}_bfrac{budget_frac}_seed{seed}",
                    episode_id=episode_id,
                    task_id=task_meta["task_id"],
                    method=method_name,
                    seed=seed,
                    split=split_name,
                    budget_fraction=budget_frac,
                    budget_tokens=budget_tokens,
                    policy_frozen=is_frozen,
                    category=task_meta.get("category", ""),
                    difficulty=task_meta.get("difficulty", ""),
                    receiver_role=task_meta.get("receiver_role", ""),
                    num_available=len(cius),
                    num_selected=len(selected_cius),
                    available_tokens=available_tokens,
                    selected_tokens=selected_tokens,
                    compression=compression,
                    api_prompt_tokens=api_prompt,
                    api_completion_tokens=api_completion,
                    task_success=success,
                    reward=reward,
                    latency_ms=elapsed_ms,
                    clarification_turns=1 if clarification else 0,
                    precision=sq["precision"],
                    recall=sq["recall"],
                    misleading_rate=sq["misleading_rate"],
                    voi_scores_json=json.dumps(voi_scores),
                ))

                # Log decision rows
                dec_rows = ResearchLogger.build_decision_rows(
                    run_id=self._run_id,
                    episode_id=episode_id,
                    state=state,
                    available_cius=cius,
                    selected_ids=action.selected_ciu_ids,
                    ciu_scores=action.ciu_scores,
                    budget_tokens=budget_tokens,
                    policy=_get_linucb_policy(policy),
                    policy_version=episode_counter,
                )
                rlog.log_decisions(dec_rows)
                # Flush after every episode so results are visible in real-time
                rlog.flush()

                # Log update rows (only when policy was updated)
                if not is_frozen and hasattr(policy, "_arms") or (
                    not is_frozen and hasattr(getattr(policy, "_inner", None), "_arms")
                ):
                    inner = getattr(policy, "_inner", policy)
                    upd_rows = ResearchLogger.build_update_rows(
                        run_id=self._run_id,
                        episode_id=episode_id,
                        policy=inner,
                        selected_ids=action.selected_ciu_ids,
                        selected_cius=selected_cius,
                        reward=reward,
                        voi_scores=voi_scores,
                        policy_version=episode_counter,
                    )
                    for u in upd_rows:
                        rlog.log_update(u)

                # Accumulate for aggregate metrics
                successes.append(success)
                token_counts.append(selected_tokens)
                rewards.append(reward)
                precisions.append(sq["precision"])
                recalls.append(sq["recall"])
                misleading_rates.append(sq["misleading_rate"])

                if self.verbose or task_idx % 5 == 0:
                    logger.info(
                        "  [%s/%s/%s] ep=%d | success=%s | tokens=%d/%d | "
                        "prec=%.2f recall=%.2f | reward=%.3f",
                        method_name, split_name, f"bfrac{budget_frac}",
                        task_idx + 1,
                        "Y" if success else "N",
                        selected_tokens, budget_tokens,
                        sq["precision"], sq["recall"],
                        reward,
                    )

            # Aggregate for this split
            n = len(successes)
            split_results[split_name] = {
                "n": n,
                "success_rate": sum(successes) / n if n else 0.0,
                "mean_tokens": sum(token_counts) / n if n else 0.0,
                "mean_reward": sum(rewards) / n if n else 0.0,
                "mean_precision": sum(precisions) / n if n else 0.0,
                "mean_recall": sum(recalls) / n if n else 0.0,
                "mean_misleading_rate": sum(misleading_rates) / n if n else 0.0,
            }
            rlog.flush()

        # Return test-split aggregates as the primary result
        test_r = split_results.get("test", {})
        return {
            "test_success_rate": test_r.get("success_rate", 0.0),
            "test_mean_tokens": test_r.get("mean_tokens", 0.0),
            "test_mean_reward": test_r.get("mean_reward", 0.0),
            "test_mean_precision": test_r.get("mean_precision", 0.0),
            "test_mean_recall": test_r.get("mean_recall", 0.0),
            "test_mean_misleading_rate": test_r.get("mean_misleading_rate", 0.0),
            "splits": split_results,
        }

    @property
    def run_id(self) -> str:
        return self._run_id


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_policy(
    method_name: str,
    llm_provider: LLMProvider,
    seed: int,
    alpha: float = 1.0,
) -> CommunicationPolicy:
    """Instantiate the correct policy object for a given method name."""
    if method_name == "FullContext-Unbounded":
        return FullContextPolicy()
    elif method_name == "FullContext-Budgeted":
        return BudgetedFullContextPolicy()
    elif method_name == "Random":
        return RandomPolicy(seed=seed)
    elif method_name == "Heuristic":
        return StaticHeuristicPolicy()
    elif method_name == "LLMPlanner":
        # In mock mode, fall back to StaticHeuristicPolicy as a deterministic proxy.
        # The mock LLM returns an empty list of selected CIUs, which gives no evidence.
        # Using a deterministic proxy (same visible CIUs, same budget) is the correct
        # research choice: document that mock LLMPlanner = Heuristic proxy.
        from caps.llm.mock_client import MockLLMProvider
        if isinstance(llm_provider, MockLLMProvider):
            return StaticHeuristicPolicy()  # deterministic proxy, documented
        return LLMPlannerPolicy(llm_provider=llm_provider, model=llm_provider.default_model)
    elif method_name == "CAPS-NoVOI":
        return CAPSNoVOIPolicy(alpha=alpha)
    elif method_name in ("CAPS-VOI", "CAPS-OracleVOI"):
        return LinUCBPolicy(alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method_name!r}")


def _get_linucb_policy(policy: CommunicationPolicy) -> Optional[Any]:
    """Return the underlying LinUCBPolicy if available."""
    if isinstance(policy, LinUCBPolicy):
        return policy
    inner = getattr(policy, "_inner", None)
    if isinstance(inner, LinUCBPolicy):
        return inner
    return None


def _estimate_avg_tokens(tasks: List[Dict[str, Any]]) -> int:
    """Estimate average total token count per task."""
    if not tasks:
        return 1000
    total = sum(
        sum(c.get("token_count", len(c.get("content", "")) // 4)
            for c in t.get("cius", []))
        for t in tasks
    )
    return total // len(tasks)


def _derive_synthetic_voi(
    task_data: Dict[str, Any],
    selected_ids: List[str],
) -> Dict[str, float]:
    """
    Derive VOI scores from hidden synthetic_v2 ground-truth labels.

    Called ONLY for CAPS-VOI. The ground-truth necessity / relevance fields
    are available in the raw task dict and provide a perfect oracle signal
    in the synthetic setting.

    Returns: dict mapping ciu_id → voi (0.0 – 1.0)
    """
    gt = task_data.get("ground_truth", {})
    necessary_set = set(gt.get("necessary_cius", []))
    useful_set = set(gt.get("useful_cius", []))
    misleading_set = set(gt.get("misleading_cius", []))

    voi: Dict[str, float] = {}
    for ciu in task_data.get("cius", []):
        cid = ciu["id"]
        if cid not in selected_ids:
            continue
        if cid in misleading_set:
            voi[cid] = -0.5
        elif cid in necessary_set:
            voi[cid] = 1.0
        elif cid in useful_set:
            voi[cid] = 0.6
        else:
            # redundant or irrelevant
            relevance = ciu.get("relevance", 0.0)
            voi[cid] = max(0.0, float(relevance) * 0.4)
    return voi
