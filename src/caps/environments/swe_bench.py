"""
SWE-Bench Environment for CAPS.

Uses an LLM judge to evaluate the receiver's generated patch
against the gold patch from the SWE-Bench dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from caps.environments.synthetic import SyntheticEnvironment
from caps.llm import LLMProvider

logger = logging.getLogger(__name__)


class SWEBenchEnvironment(SyntheticEnvironment):
    """
    Environment for SWE-Bench tasks.
    
    Inherits task loading from SyntheticEnvironment, but overrides evaluate()
    to use an LLM judge to compare the generated patch with the gold patch.
    """

    def __init__(self, llm_provider: LLMProvider, model_name: str = "llama-3.1-8b-instant"):
        super().__init__()
        self.llm = llm_provider
        self.model_name = model_name

    def evaluate(self, receiver_output: str) -> bool:
        """
        Evaluate using LLM as a judge.
        Compares the receiver's output patch to the ground truth patch.
        """
        if self._current_task is None:
            return False
            
        gt_patch = self.ground_truth().get("patch", "")
        if not gt_patch:
            # If no ground truth patch, we can't evaluate correctly.
            return False

        prompt = (
            f"You are an expert software engineer evaluating a proposed patch for an issue.\n\n"
            f"Ground Truth Patch:\n{gt_patch}\n\n"
            f"Proposed Patch by Agent:\n{receiver_output}\n\n"
            f"Does the proposed patch solve the issue in the same logical way as the ground truth patch? "
            f"It does not have to be exactly character-for-character identical, but it must make the same logical code changes.\n"
            f"Respond with exactly 'YES' or 'NO'."
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a strict code evaluator.",
                model=self.model_name,
                temperature=0.0,
            )
            
            result = response.text.strip().upper()
            return "YES" in result
            
        except Exception as e:
            logger.error(f"Error during SWE-Bench LLM evaluation: {e}")
            return False
