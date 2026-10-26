"""
Per-CIU LinUCB — the core CAPS learning algorithm.

Mathematical formulation (§14):

    For each CIU arm i, we maintain:
        A_i  ∈ R^{d×d}   (precision matrix, initialized to I)
        b_i  ∈ R^{d×1}   (reward-weighted feature sum)

    Parameter estimate:
        θ_i = A_i^{-1} b_i

    Upper confidence bound:
        UCB_i = x^T θ_i + α √(x^T A_i^{-1} x)

    where:
        x = context feature vector (task + receiver + budget features)
        α = exploration parameter

    Selection:
        Score each CIU by UCB_i, then select greedily under token budget.

    Update (after observing reward r for CIU i):
        A_i ← A_i + x x^T
        b_i ← b_i + r x

The key research property: each CIU type learns an independent
value function conditioned on the context, allowing CAPS to
discover which information is useful for which task/receiver contexts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from caps.communication.ciu import CIU
from caps.communication.policy import CommunicationPolicy
from caps.communication.selector import select_cius
from caps.communication.state import Action, CommunicationState

logger = logging.getLogger(__name__)


class LinUCBPolicy(CommunicationPolicy):
    """
    Per-CIU contextual LinUCB policy.

    Mode A (§15): Each CIU type has an independent contextual reward model.
    This is the primary CAPS policy.

    Attributes:
        alpha: Exploration parameter (higher = more exploration).
        feature_dim: Dimension of the context feature vector.
        selection_strategy: How to select CIUs from scores (greedy_budget, top_k, etc.).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        selection_strategy: str = "greedy_budget",
        epsilon: float = 0.0,
    ):
        self.alpha = alpha
        self.selection_strategy = selection_strategy
        self.epsilon = epsilon  # epsilon-greedy on top of UCB
        self.feature_dim: Optional[int] = None

        # Per-CIU-type arms: type_name → (A, b)
        self._arms: Dict[str, Dict[str, np.ndarray]] = {}
        self._episode_count = 0

    def _get_or_create_arm(self, ciu_type: str, dim: int) -> Dict[str, np.ndarray]:
        """Get or initialize the A, b matrices for a CIU type."""
        if ciu_type not in self._arms:
            self._arms[ciu_type] = {
                "A": np.eye(dim, dtype=np.float64),
                "b": np.zeros((dim, 1), dtype=np.float64),
            }
        return self._arms[ciu_type]

    def _build_context(self, state: CommunicationState, ciu: CIU) -> np.ndarray:
        """
        Build the context feature vector x for a (state, CIU) pair.

        Features:
            - Task embedding (full dimension)
            - Budget features (normalized)
            - CIU scalar features
            - Semantic similarity between task and CIU
        """
        # Task embedding — use full dimension
        dim = len(state.task_embedding) if state.task_embedding else 384
        task_emb = state.task_embedding if state.task_embedding else [0.0] * dim

        # Budget features (normalized)
        budget_feats = [
            min(state.token_budget / 10000.0, 1.0),
            state.planner_confidence,
            min(state.previous_turns / 5.0, 1.0),
        ]

        # CIU features
        ciu_feats = [
            min(ciu.estimate_tokens() / 2000.0, 1.0),
            ciu.priority,
            ciu.historical_utility,
            ciu.historical_selection_rate,
            ciu.uncertainty,
        ]

        # Cross-CIU feature: Semantic similarity
        sim = 0.0
        if state.task_embedding and ciu.embedding:
            vec1 = np.array(state.task_embedding)
            vec2 = np.array(ciu.embedding)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(vec1, vec2) / (norm1 * norm2))

        # We omit the raw 384-D task_embedding to prevent the curse of dimensionality
        # in LinUCB with a small number of episodes. The `sim` feature captures semantic relevance.
        x = np.array(budget_feats + ciu_feats + [sim], dtype=np.float64)
        return x.reshape(-1, 1)

    def select(self, state: CommunicationState) -> Action:
        self._episode_count += 1

        if not state.available_cius:
            return Action(selected_ciu_ids=[], ciu_scores={})

        # Build context and compute UCB for each CIU
        scores: Dict[str, float] = {}

        for ciu in state.available_cius:
            x = self._build_context(state, ciu)

            # Initialize feature dim on first call
            if self.feature_dim is None:
                self.feature_dim = x.shape[0]

            arm = self._get_or_create_arm(ciu.type, self.feature_dim)
            A = arm["A"]
            b = arm["b"]

            # θ = A^{-1} b
            try:
                A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            theta = A_inv @ b

            # UCB = x^T θ + α √(x^T A^{-1} x)
            exploitation = float((x.T @ theta).item())
            exploration = self.alpha * float(np.sqrt((x.T @ A_inv @ x).item()))
            ucb = exploitation + exploration

            scores[ciu.id] = ucb

        # Epsilon-greedy perturbation
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            for cid in scores:
                scores[cid] = np.random.random()

        # Select CIUs under budget
        selected = select_cius(
            state.available_cius,
            scores,
            state.token_budget,
            strategy=self.selection_strategy,
        )

        return Action(selected_ciu_ids=selected, ciu_scores=scores)

    def update(
        self,
        state: CommunicationState,
        action: Action,
        reward: float,
        voi_scores: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """
        Update per-CIU arms using observed rewards.

        For each selected CIU:
            - If VOI scores are available, use per-CIU VOI as the reward signal.
            - Otherwise, use the global task reward for all selected CIUs.
        """
        if not action.selected_ciu_ids:
            return

        for ciu in state.available_cius:
            if ciu.id not in action.selected_ciu_ids:
                continue

            x = self._build_context(state, ciu)
            if self.feature_dim is None:
                self.feature_dim = x.shape[0]

            arm = self._get_or_create_arm(ciu.type, self.feature_dim)

            # Determine reward signal for this CIU
            if voi_scores and ciu.id in voi_scores:
                # Combine VOI signal with the global reward (which contains token penalties)
                ciu_reward = (voi_scores[ciu.id] + reward) / 2.0
            else:
                ciu_reward = reward

            # LinUCB update
            arm["A"] = arm["A"] + x @ x.T
            arm["b"] = arm["b"] + ciu_reward * x

    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize for checkpointing."""
        state = {
            "alpha": self.alpha,
            "feature_dim": self.feature_dim,
            "episode_count": self._episode_count,
            "epsilon": self.epsilon,
            "selection_strategy": self.selection_strategy,
            "arms": {},
        }
        for arm_name, arm_data in self._arms.items():
            state["arms"][arm_name] = {
                "A": arm_data["A"].tolist(),
                "b": arm_data["b"].tolist(),
            }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore from checkpoint."""
        self.alpha = state_dict.get("alpha", self.alpha)
        self.feature_dim = state_dict.get("feature_dim", self.feature_dim)
        self._episode_count = state_dict.get("episode_count", 0)
        self.epsilon = state_dict.get("epsilon", self.epsilon)
        self.selection_strategy = state_dict.get("selection_strategy", self.selection_strategy)
        self._arms = {}
        for arm_name, arm_data in state_dict.get("arms", {}).items():
            self._arms[arm_name] = {
                "A": np.array(arm_data["A"], dtype=np.float64),
                "b": np.array(arm_data["b"], dtype=np.float64),
            }

    @property
    def name(self) -> str:
        return f"LinUCB(α={self.alpha})"
