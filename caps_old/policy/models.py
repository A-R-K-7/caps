from typing import List, Dict, Optional
from pydantic import BaseModel
from caps.ciu.models import CIU

class PlannerState(BaseModel):
    """
    The state observed by the policy before selecting an action.
    """
    task_embedding: List[float]
    receiver_embedding: Optional[List[float]] = None
    planner_confidence: float
    token_budget: int
    latency_budget: float
    available_cius: List[CIU]
    # Future enhancement: cross-CIU similarity matrix or dependency graph

class Action(BaseModel):
    """
    The action taken by the policy.
    """
    selected_ciu_ids: List[str]
    ciu_scores: Dict[str, float]
