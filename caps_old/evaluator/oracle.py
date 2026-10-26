from caps.receiver.agent import ReceiverAgent
from caps.benchmark.env import Environment
from caps.ciu.models import CIU
from typing import List, Dict

class CounterfactualEvaluator:
    """
    Evaluates the causal contribution of each CIU by re-running the receiver
    with one CIU removed at a time.
    """
    def __init__(self, receiver: ReceiverAgent, env: Environment):
        self.receiver = receiver
        self.env = env
        
    def evaluate(self, task: str, selected_cius: List[CIU], baseline_success: bool) -> Dict[str, float]:
        """
        Returns a dictionary mapping ciu_id -> Value of Information (VOI).
        VOI = baseline_success - counterfactual_success
        """
        voi_scores = {}
        for ciu in selected_cius:
            # Remove this CIU
            subset = [c for c in selected_cius if c.id != ciu.id]
            result = self.receiver.execute(task, subset)
            cf_success = self.env.evaluate(result.answer)
            
            # If baseline succeeded but counterfactual failed, VOI is 1.0 (crucial)
            # If baseline failed but counterfactual succeeded, VOI is -1.0 (harmful)
            voi_scores[ciu.id] = float(baseline_success) - float(cf_success)
            
        return voi_scores

class OracleEvaluator:
    """
    Finds the theoretical minimum communication required to succeed.
    """
    def __init__(self, receiver: ReceiverAgent, env: Environment):
        self.receiver = receiver
        self.env = env
        
    def find_minimal_subset(self, task: str, all_cius: List[CIU]) -> List[CIU]:
        """
        Iteratively removes CIUs to find the smallest subset that still succeeds.
        Returns the minimal subset, or empty if it fails even with all.
        """
        # Step 1: Check if it succeeds with all
        result = self.receiver.execute(task, all_cius)
        if not self.env.evaluate(result.answer):
            return [] # Fails even with all
            
        # Step 2: Iterative removal (greedy backward search)
        current_subset = all_cius.copy()
        
        while len(current_subset) > 1:
            removed_any = False
            
            # Try removing each CIU one by one
            for ciu in current_subset:
                candidate = [c for c in current_subset if c.id != ciu.id]
                res = self.receiver.execute(task, candidate)
                if self.env.evaluate(res.answer):
                    current_subset = candidate
                    removed_any = True
                    break # Restart loop with smaller subset
                    
            if not removed_any:
                break # Cannot remove any more without failing
                
    def compute_metrics(self, caps_selected_cius: List[CIU], oracle_cius: List[CIU]) -> Dict[str, float]:
        """
        Computes the Oracle Token Gap, Oracle Precision, Oracle Recall, and Distance to Oracle.
        """
        if not oracle_cius:
            # If there is no successful oracle subset, metrics might not be fully meaningful, 
            # but we can return zeros or N/As.
            return {
                "oracle_token_gap": 0.0,
                "oracle_precision": 0.0,
                "oracle_recall": 0.0,
                "oracle_efficiency": 0.0
            }
            
        caps_ids = {c.id for c in caps_selected_cius}
        oracle_ids = {c.id for c in oracle_cius}
        
        true_positives = len(caps_ids.intersection(oracle_ids))
        
        precision = true_positives / len(caps_ids) if caps_ids else 0.0
        recall = true_positives / len(oracle_ids) if oracle_ids else 0.0
        
        caps_tokens = sum(c.token_count for c in caps_selected_cius)
        oracle_tokens = sum(c.token_count for c in oracle_cius)
        
        token_gap = caps_tokens - oracle_tokens
        efficiency = oracle_tokens / caps_tokens if caps_tokens > 0 else 0.0
        
        return {
            "oracle_token_gap": token_gap,
            "oracle_precision": precision,
            "oracle_recall": recall,
            "oracle_efficiency": efficiency,
            "oracle_tokens": oracle_tokens,
            "caps_tokens": caps_tokens
        }
