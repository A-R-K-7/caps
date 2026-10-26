import json
from caps.engine.env import Policy
from caps.policy.models import PlannerState, Action
from caps.utils.llm import BaseLLMClient
from pydantic import BaseModel, Field

class ReflectionSchema(BaseModel):
    selected_ciu_ids: list[str] = Field(description="List of CIU IDs to send")

class ReflectionPolicy(Policy):
    """
    Uses an LLM to reflect on previous failures and select CIUs.
    """
    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.past_trajectories = [] # List of past states/actions/rewards for reflection

    def select(self, state: PlannerState) -> Action:
        # Prompt the LLM with the task and available CIUs, and past reflections if any
        available_str = "\n".join([f"ID: {c.id} | Type: {c.type} | Tokens: {c.token_count}" for c in state.available_cius])
        
        prompt = (
            "You are a communication planner. Your goal is to select the minimum set of information units (CIUs) "
            "needed to solve a task. Stay within the token budget.\n\n"
            f"Token Budget: {state.token_budget}\n"
            f"Available CIUs:\n{available_str}\n\n"
        )
        
        if self.past_trajectories:
            prompt += "Past failed attempts (Reflect on these to avoid repeating mistakes):\n"
            for t in self.past_trajectories[-3:]: # last 3
                prompt += f"- Selected: {t['action']}. Reward received: {t['reward']}\n"
                
        prompt += "\nOutput JSON containing 'selected_ciu_ids'."
        
        system_instruction = "Select the best subset of CIUs."
        
        try:
            resp_text, _ = self.llm.generate_content(
                system_instruction=system_instruction,
                prompt=prompt,
                response_schema=ReflectionSchema
            )
            parsed = json.loads(resp_text)
            selected_ids = parsed.get("selected_ciu_ids", [])
        except Exception:
            # Fallback
            selected_ids = [c.id for c in state.available_cius[:5]]
            
        scores = {ciu.id: 1.0 if ciu.id in selected_ids else 0.0 for ciu in state.available_cius}
        
        # Enforce budget
        final_ids = []
        current_tokens = 0
        for ciu in state.available_cius:
            if ciu.id in selected_ids and current_tokens + ciu.token_count <= state.token_budget:
                final_ids.append(ciu.id)
                current_tokens += ciu.token_count
                
        return Action(selected_ciu_ids=final_ids, ciu_scores=scores)

    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        # Store for reflection in the next step of the same task
        self.past_trajectories.append({
            "action": action.selected_ciu_ids,
            "reward": reward
        })
