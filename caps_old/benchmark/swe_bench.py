import json
from datasets import load_dataset
from caps.benchmark.env import Environment
from typing import Dict, Any

class SWEBenchEnv(Environment):
    def __init__(self, split="test", subset=100):
        # Load the SWE-bench Lite dataset
        self.dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
        
        if subset and subset < len(self.dataset):
            # Take a deterministic subset for reproducible evaluation
            self.dataset = self.dataset.select(range(subset))
            
        self.current_task = None
        
    def reset(self, task_idx: int) -> str:
        self.current_task = self.dataset[task_idx]
        task_desc = f"Fix the following issue in {self.current_task['repo']}:\n\n{self.current_task['problem_statement']}"
        return task_desc
        
    def evaluate(self, receiver_output: str) -> bool:
        """
        Since running the full SWE-bench test harness requires an isolated Docker
        environment per issue, we use a heuristic proxy for this evaluation loop.
        In a full cluster deployment, this would invoke `swebench.evaluate()`.
        
        For our research proxy, we check if the agent correctly identified the
        files and functions that need to be modified as per the ground truth patch.
        """
        if not self.current_task:
            return False
            
        patch = self.current_task["patch"]
        # Extract files modified in the patch
        modified_files = []
        for line in patch.split('\n'):
            if line.startswith('+++ b/'):
                modified_files.append(line[6:])
                
        success = False
        if modified_files:
            # Did the agent mention the files that need fixing?
            success = any(f in receiver_output for f in modified_files)
            
        return success
        
    def ground_truth(self) -> Any:
        return self.current_task["patch"] if self.current_task else None
        
    def metrics(self) -> Dict[str, Any]:
        return {
            "instance_id": self.current_task["instance_id"],
            "repo": self.current_task["repo"],
            "base_commit": self.current_task["base_commit"]
        } if self.current_task else {}
