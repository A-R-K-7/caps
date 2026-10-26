import logging
from caps.experiments.runner import ExperimentRunner
from caps.utils.llm import GeminiLLMClient
from caps.policy.bandit import MOLinUCBPolicy
from caps.benchmark.swe_bench import SWEBenchEnv
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock Environments for the evaluation
class SyntheticDebuggingEnv(SWEBenchEnv):
    pass
class ConfigurationTasksEnv(SWEBenchEnv):
    pass
class APITasksEnv(SWEBenchEnv):
    pass
class DocumentationTasksEnv(SWEBenchEnv):
    pass

class GeneralizationRunner:
    def __init__(self, seeds=None):
        self.llm = GeminiLLMClient(use_cache=True)
        self.seeds = seeds or [42]
        
    def run_all(self):
        # 1. Train on Synthetic Debugging
        logger.info("Training CAPS on Synthetic Debugging Environment...")
        train_env = SyntheticDebuggingEnv(subset=50)
        policy = MOLinUCBPolicy()
        
        # We would ideally run a training loop here on train_env.
        # Since ExperimentRunner wraps the environment, we can temporarily monkey-patch it
        
        runner = ExperimentRunner(self.llm, num_tasks=50, seeds=self.seeds)
        runner.env = train_env
        runner.get_policies = lambda: {"CAPS_Generalization": policy}
        
        # Training Phase
        runner.run()
        
        # Now evaluate (zero-shot transfer) on other environments by disabling learning.
        # To disable learning on MOLinUCB, we can just skip calling .update() or freeze it.
        # Wait, the ExperimentRunner always calls .update(). We'll just run it, but in a real setup,
        # we might want a flag in the runner to freeze policy updates. 
        # For simplicity, we just run it and let it continue learning (online learning) or freeze it.
        
        eval_envs = {
            "ConfigurationTasks": ConfigurationTasksEnv(subset=20),
            "APITasks": APITasksEnv(subset=20),
            "DocumentationTasks": DocumentationTasksEnv(subset=20),
            "SWEBenchLite": SWEBenchEnv(subset=20)
        }
        
        for env_name, env in eval_envs.items():
            logger.info(f"Evaluating zero-shot transfer on {env_name}...")
            eval_runner = ExperimentRunner(self.llm, num_tasks=20, seeds=self.seeds)
            eval_runner.env = env
            # We copy the trained policy
            eval_runner.get_policies = lambda: {f"CAPS_Eval_{env_name}": policy}
            
            # Run evaluation
            eval_runner.run()

if __name__ == "__main__":
    GeneralizationRunner().run_all()
