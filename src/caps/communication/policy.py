"""
Communication Policy — abstract base class.

Every CIU selection strategy (baselines and CAPS) implements this interface.
The policy MUST run locally — never use an LLM for the bandit decision.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from caps.communication.state import Action, CommunicationState


class CommunicationPolicy(abc.ABC):
    """
    Abstract base for communication policies.

    select() chooses which CIUs to transmit.
    update() learns from observed rewards.
    """

    @abc.abstractmethod
    def select(self, state: CommunicationState) -> Action:
        """Select CIUs to transmit given the current state."""
        ...

    @abc.abstractmethod
    def update(
        self,
        state: CommunicationState,
        action: Action,
        reward: float,
        voi_scores: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """Update the policy from observed reward and CIU utilities."""
        ...

    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize policy state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore policy state from checkpoint."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
