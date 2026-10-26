"""
Communication Evaluator Agent.

Estimates the Value of Information (VOI) of transmitted CIUs
via Counterfactual Replay.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from caps.agents.receiver import ReceiverAgent
from caps.communication.ciu import CIU
from caps.environments.synthetic import SyntheticEnvironment

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Evaluates the true utility of selected CIUs by running
    counterfactual experiments (removing one CIU at a time
    and observing the change in task success).
    """

    def __init__(
        self,
        receiver_agent: ReceiverAgent,
        max_counterfactuals: int = 3,
        strategy: str = "random",
        seed: int = 42,
    ):
        self.receiver = receiver_agent
        self.max_counterfactuals = max_counterfactuals
        self.strategy = strategy
        self._rng = random.Random(seed)

    def evaluate_voi(
        self,
        task_desc: str,
        selected_cius: List[CIU],
        original_success: bool,
        env: SyntheticEnvironment,
        receiver_role: str = "general",
    ) -> Dict[str, float]:
        """
        Estimate VOI for a subset of the selected CIUs.

        VOI assignment:
        +1.0 : Removing CIU caused a previously successful task to fail (Critical)
        -1.0 : Removing CIU caused a previously failed task to succeed (Harmful)
         0.0 : Removing CIU did not change the outcome (Neutral/Redundant)
        """
        voi_scores: Dict[str, float] = {}

        if not selected_cius or self.max_counterfactuals <= 0:
            return voi_scores

        # Select which CIUs to test (to save budget)
        to_test = self._select_for_testing(selected_cius)

        for ciu in to_test:
            # Create counterfactual context (all selected EXCEPT this one)
            cf_cius = [c for c in selected_cius if c.id != ciu.id]

            # Replay receiver
            cf_output = self.receiver.execute(
                task=task_desc,
                selected_cius=cf_cius,
                receiver_role=receiver_role,
            )

            # Evaluate counterfactual success
            cf_success = env.evaluate(cf_output.answer)

            # Assign VOI
            if original_success and not cf_success:
                voi = 1.0  # Critical for success
            elif not original_success and cf_success:
                voi = -1.0  # Harmful (distracting context)
            else:
                voi = 0.0  # Neutral or redundant

            voi_scores[ciu.id] = voi
            logger.debug(f"Counterfactual for {ciu.id}: orig={original_success}, cf={cf_success} -> VOI={voi}")

        return voi_scores

    def _select_for_testing(self, selected_cius: List[CIU]) -> List[CIU]:
        """Select a subset of CIUs to perform counterfactual testing on."""
        if len(selected_cius) <= self.max_counterfactuals:
            return selected_cius

        if self.strategy == "random":
            return self._rng.sample(selected_cius, self.max_counterfactuals)
        
        # Future: could implement "uncertainty" strategy (pick CIUs with highest policy uncertainty)
        return self._rng.sample(selected_cius, self.max_counterfactuals)

    def evaluate_self_reflection(
        self,
        selected_cius: List[CIU],
        used_ciu_ids: List[str],
        original_success: bool,
    ) -> Dict[str, float]:
        """
        Assign VOI based on LLM self-reflection attribution in O(1) time.
        
        VOI assignment:
        If task succeeded:
          +1.0 : CIU was selected and LLM reported using it.
           0.0 : CIU was selected but LLM reported it was unused.
        If task failed:
          -1.0 : CIU was selected and LLM reported using it (potentially harmful/distracting).
           0.0 : CIU was selected but unused.
        """
        voi_scores: Dict[str, float] = {}

        for ciu in selected_cius:
            if original_success:
                voi_scores[ciu.id] = 1.0 if ciu.id in used_ciu_ids else 0.0
            else:
                # Neutral penalty instead of a harsh -1.0 so we don't discourage complex CIU selection
                voi_scores[ciu.id] = 0.0

        return voi_scores
