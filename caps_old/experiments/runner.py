import os
import json
import logging
from typing import List, Dict, Any
from caps.benchmark.swe_bench import SWEBenchEnv
from caps.engine.env import CAPSEnv
from caps.utils.llm import BaseLLMClient
from caps.evaluator.oracle import OracleEvaluator, CounterfactualEvaluator
from caps.policy.baselines import (
    AlwaysSendEverythingPolicy,
    RandomPolicy,
    TokenBudgetGreedyPolicy,
    TopKPolicy,
    EmbeddingSimilarityPolicy,
    HeuristicPolicy
)
from caps.policy.reflection import ReflectionPolicy
from caps.policy.bandit import MOLinUCBPolicy
from configs.settings import settings
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, llm_client: BaseLLMClient, num_tasks=100, seeds=None, use_counterfactual_voi=True):
        self.llm = llm_client
        self.num_tasks = num_tasks
        self.seeds = seeds or [42, 43, 44, 45, 46]
        self.use_counterfactual_voi = use_counterfactual_voi
        self.env = SWEBenchEnv(subset=num_tasks)
        self.caps_engine = CAPSEnv(self.llm, temperature=0.0) # Seed will be overridden per run
        self.oracle = OracleEvaluator(self.caps_engine.receiver, self.env)
        self.counterfactual = CounterfactualEvaluator(self.caps_engine.receiver, self.env)
        
    def get_policies(self):
        return {
            "AlwaysSendEverything": AlwaysSendEverythingPolicy(),
            "Random": RandomPolicy(),
            "TopK": TopKPolicy(k=5),
            "TokenBudgetGreedy": TokenBudgetGreedyPolicy(),
            "EmbeddingSimilarity": EmbeddingSimilarityPolicy(),
            "Heuristic": HeuristicPolicy(),
            "Reflection": ReflectionPolicy(self.llm),
            "CAPS_MOLinUCB": MOLinUCBPolicy()
        }
        
    def run(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("CAPS_Phase6_Evaluation")
        
        for policy_name, policy in self.get_policies().items():
            for seed in self.seeds:
                with mlflow.start_run(run_name=f"{policy_name}_seed_{seed}"):
                    mlflow.log_params({"policy": policy_name, "seed": seed, "num_tasks": self.num_tasks})
                    
                    # Ensure deterministic seed for the LLM inside CAPSEnv
                    self.caps_engine.planner.seed = seed
                    self.caps_engine.receiver.seed = seed
                    self.caps_engine.evaluator.seed = seed
                    
                    total_reward = 0
                    successes = 0
                    total_tokens = 0
                    total_clarifications = 0
                    
                    for task_idx in range(self.num_tasks):
                        task_id = f"task_{task_idx}"
                        task_desc = self.env.reset(task_idx)
                        raw_context = json.dumps(self.env.metrics())
                        
                        # Step through the MDP
                        transition = self.caps_engine.step(task_desc, raw_context, policy)
                        
                        # Evaluate objective task success
                        success = self.env.evaluate(transition.result.answer)
                        if success:
                            successes += 1
                            
                        # Correct the reward with actual success
                        reward = self.caps_engine.reward_engine.calculate_reward(
                            transition.state, transition.action, transition.result, success
                        )
                        transition.reward = reward
                        
                        selected_cius = [c for c in transition.available_cius if c.id in transition.action.selected_ciu_ids]
                        
                        oracle_results = None
                        # If successful, compute counterfactuals and oracle boundaries
                        if success:
                            oracle_cius = self.oracle.find_minimal_subset(task_desc, transition.available_cius)
                            oracle_results = self.oracle.compute_metrics(selected_cius, oracle_cius)
                            
                            if self.use_counterfactual_voi:
                                cf_voi = self.counterfactual.evaluate(task_desc, selected_cius, success)
                                transition.voi_scores.update(cf_voi)
                                
                        # Re-update policy with actual reward and refined VOI
                        policy.update(transition.state, transition.action, reward, voi_scores=transition.voi_scores)
                        
                        # Accumulate metrics for MLFlow
                        tokens_used = sum(c.token_count for c in selected_cius)
                        total_tokens += tokens_used
                        total_reward += reward
                        if transition.result.asked_clarification:
                            total_clarifications += 1
                                
                        # Add to Replay Buffer
                        self.caps_engine.replay_buffer.add_transition(
                            task=task_desc,
                            task_id=task_id,
                            policy_name=policy_name,
                            available_ciu_ids=[ciu.id for ciu in transition.available_cius],
                            token_budget=transition.state.token_budget,
                            selected_ciu_ids=transition.action.selected_ciu_ids,
                            asked_clarification=transition.result.asked_clarification,
                            receiver_answer=transition.result.answer,
                            task_success=success,
                            env_metrics=self.env.metrics(),
                            voi_scores=transition.voi_scores,
                            oracle_results=oracle_results,
                            reward=reward
                        )
                        
                    mlflow.log_metrics({
                        "avg_reward": total_reward / max(1, self.num_tasks),
                        "success_rate": successes / max(1, self.num_tasks),
                        "avg_tokens": total_tokens / max(1, self.num_tasks),
                        "avg_clarifications": total_clarifications / max(1, self.num_tasks)
                    })
                    
                    logger.info(f"Finished {policy_name} seed {seed}: SR={successes/self.num_tasks:.2f}")

if __name__ == "__main__":
    from caps.utils.llm import GeminiLLMClient
    
    # Instantiate the LLM client using real Gemini API and Cache
    llm = GeminiLLMClient(use_cache=True)
    
    # We reduce num_tasks and seeds for a quick test run. 
    # For full evaluation, increase to 100 tasks and 5 seeds.
    runner = ExperimentRunner(llm_client=llm, num_tasks=5, seeds=[42])
    
    logger.info("Starting CAPS Experiment Runner...")
    runner.run()
    logger.info("Experiment Runner Finished.")

