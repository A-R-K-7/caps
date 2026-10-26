from caps.policy.models import PlannerState, Action
from caps.receiver.agent import ReceiverResult
from configs.settings import settings

class RewardEngine:
    """
    Calculates the reward for the reinforcement learning policy.
    Implements reward shaping to prevent policy collapse into trivial states.
    """
    def __init__(self):
        self.lambda_cost = settings.reward_lambda
        
    def calculate_reward(self, 
                         state: PlannerState, 
                         action: Action, 
                         result: ReceiverResult, 
                         task_success: bool) -> float:
        """
        reward = Task Success 
               + Shaping Milestones 
               - lambda * Communication Cost
        """
        # 1. Base Task Success (Ground truth provided by objective environment)
        reward = 1.0 if task_success else 0.0
        
        # 2. Shaping: Penalize clarification requests heavily to encourage self-sufficiency
        if result.asked_clarification:
            reward -= 0.5
            
        # 3. Communication Cost: Fraction of token budget used
        used_tokens = 0
        for ciu in state.available_cius:
            if ciu.id in action.selected_ciu_ids:
                used_tokens += ciu.token_count
                
        cost_penalty = self.lambda_cost * (used_tokens / max(1, state.token_budget))
        reward -= cost_penalty
        
        return reward
