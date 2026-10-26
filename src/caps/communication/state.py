"""
Communication State — the observation space for the CAPS policy.

The state captures everything the policy needs to make a CIU selection decision:
    - Task features
    - Receiver features
    - Budget features
    - Per-CIU features
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from caps.communication.ciu import CIU


class CommunicationState(BaseModel):
    """
    Full state observed by the communication policy.

    This state is constructed BEFORE the policy selects CIUs.
    The policy uses this to decide which CIUs to transmit.
    """

    # --- Task features ---
    task_embedding: List[float] = Field(default_factory=list)
    task_category: str = "unknown"
    task_difficulty: str = "medium"
    task_length: int = 0  # character length of task description
    planner_confidence: float = 1.0

    # --- Receiver features ---
    receiver_role: str = "general"
    receiver_capability: List[float] = Field(default_factory=list)
    receiver_model: str = ""
    receiver_historical_success: float = 0.5

    # --- Budget features ---
    token_budget: int = 4000
    latency_budget_ms: float = 15000.0
    current_communication_cost: int = 0
    previous_turns: int = 0

    # --- Available CIUs ---
    available_cius: List[CIU] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class Action(BaseModel):
    """
    The action taken by the communication policy.

    Attributes:
        selected_ciu_ids: IDs of CIUs chosen for transmission.
        ciu_scores: Policy scores for each CIU (for logging/analysis).
    """

    selected_ciu_ids: List[str] = Field(default_factory=list)
    ciu_scores: Dict[str, float] = Field(default_factory=dict)
    llm_usage: Optional[Any] = None
