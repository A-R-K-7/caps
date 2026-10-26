import logging
from caps.experiments.runner import ExperimentRunner
from caps.utils.llm import GeminiLLMClient
from caps.policy.baselines import RandomPolicy, HeuristicPolicy
from caps.policy.bandit import MOLinUCBPolicy
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblationRunner:
    def __init__(self, num_tasks=50, seeds=None):
        self.llm = GeminiLLMClient(use_cache=True)
        self.seeds = seeds or [42]
        self.num_tasks = num_tasks
        
    def run_all(self):
        ablations = [
            # 1. Full CAPS
            {"name": "CAPS_Full", "policy": MOLinUCBPolicy(), "use_counterfactual_voi": True, "enable_learning": True},
            
            # 2. Disable Learning (we simulate this by resetting policy or wrapping it to ignore updates)
            # Or by using Random/Heuristic which don't learn.
            
            # 3. Disable VOI estimation (Self-Reflection and Counterfactual)
            {"name": "CAPS_NoVOI", "policy": MOLinUCBPolicy(), "use_counterfactual_voi": False, "enable_learning": True},
            
            # 4. Replace Contextual Bandit with Random
            {"name": "CAPS_Random", "policy": RandomPolicy(), "use_counterfactual_voi": False, "enable_learning": False},
            
            # 5. Replace learned policy with heuristic
            {"name": "CAPS_Heuristic", "policy": HeuristicPolicy(), "use_counterfactual_voi": False, "enable_learning": False},
        ]
        
        for ablation in ablations:
            runner = ExperimentRunner(
                llm_client=self.llm,
                num_tasks=self.num_tasks,
                seeds=self.seeds,
                use_counterfactual_voi=ablation["use_counterfactual_voi"]
            )
            
            # Override get_policies to only run this ablation
            runner.get_policies = lambda ab=ablation: {ab["name"]: ab["policy"]}
            
            logger.info(f"Starting ablation study: {ablation['name']}")
            runner.run()

if __name__ == "__main__":
    AblationRunner().run_all()
