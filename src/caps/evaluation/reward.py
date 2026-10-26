"""
Reward Function.

R = R_task - λ * C

where C = weighted normalized cost:
    w1 * normalized_tokens
    + w2 * normalized_latency
    + w3 * normalized_clarification_turns
    + w4 * normalized_retries

All cost components are normalized to [0, 1] so no single
component dominates due to raw unit differences.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from caps.communication.ciu import CIU
from caps.communication.state import Action, CommunicationState


class RewardFunction:
    """
    Configurable reward function for CAPS.

    Supports binary and continuous task success rewards.
    Communication cost is a weighted sum of normalized components.
    """

    def __init__(
        self,
        lambda_cost: float = 0.5,
        w_tokens: float = 0.4,
        w_latency: float = 0.2,
        w_clarification: float = 0.2,
        w_retries: float = 0.2,
    ):
        self.lambda_cost = lambda_cost
        self.w_tokens = w_tokens
        self.w_latency = w_latency
        self.w_clarification = w_clarification
        self.w_retries = w_retries

    def calculate(
        self,
        task_success: bool,
        state: CommunicationState,
        action: Action,
        latency_ms: float = 0.0,
        clarification_turns: int = 0,
        retry_count: int = 0,
        max_latency_ms: float = 15000.0,
        max_clarification: int = 5,
        max_retries: int = 5,
    ) -> float:
        """
        Calculate the reward.

        Args:
            task_success: Binary success indicator.
            state: Communication state (contains token budget and CIUs).
            action: Action taken (contains selected CIU IDs).
            latency_ms: Communication latency.
            clarification_turns: Number of clarification rounds.
            retry_count: Number of retries.

        Returns:
            Reward value R = R_task - λ * C
        """
        # Task reward (binary for now)
        r_task = 1.0 if task_success else 0.0

        # Communication cost components (normalized to [0, 1])
        used_tokens = sum(
            c.estimate_tokens()
            for c in state.available_cius
            if c.id in action.selected_ciu_ids
        )
        max_tokens = max(state.token_budget, 1)

        norm_tokens = min(used_tokens / max_tokens, 1.0)
        norm_latency = min(latency_ms / max(max_latency_ms, 1.0), 1.0)
        norm_clarification = min(clarification_turns / max(max_clarification, 1), 1.0)
        norm_retries = min(retry_count / max(max_retries, 1), 1.0)

        # Weighted cost
        cost = (
            self.w_tokens * norm_tokens
            + self.w_latency * norm_latency
            + self.w_clarification * norm_clarification
            + self.w_retries * norm_retries
        )

        reward = r_task - self.lambda_cost * cost
        return reward

    def cost_breakdown(
        self,
        state: CommunicationState,
        action: Action,
        latency_ms: float = 0.0,
        clarification_turns: int = 0,
        retry_count: int = 0,
    ) -> Dict[str, float]:
        """Return detailed cost breakdown for analysis."""
        used_tokens = sum(
            c.estimate_tokens()
            for c in state.available_cius
            if c.id in action.selected_ciu_ids
        )
        return {
            "tokens_used": used_tokens,
            "token_budget": state.token_budget,
            "latency_ms": latency_ms,
            "clarification_turns": clarification_turns,
            "retry_count": retry_count,
            "num_cius_selected": len(action.selected_ciu_ids),
            "num_cius_available": len(state.available_cius),
        }
