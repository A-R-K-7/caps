"""
Synthetic Task Environment.

Provides deterministic evaluation of receiver outputs against ground truth.
Does NOT use an LLM judge for synthetic tasks — uses keyword/pattern matching.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from caps.communication.ciu import CIU


class SyntheticEnvironment:
    """
    Environment for the synthetic benchmark.

    Evaluation is deterministic: checks if receiver output contains
    the answer keyword or matches the answer pattern.
    Ground truth is used ONLY during evaluation, never in prompts.
    """

    def __init__(self):
        self._current_task: Optional[Dict[str, Any]] = None
        self._tasks: List[Dict[str, Any]] = []

    def load_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Load a list of task dictionaries."""
        self._tasks = tasks

    def reset(self, task_idx: int) -> Dict[str, Any]:
        """
        Reset to a specific task.
        Returns the task dict (without ground truth — caller must strip it).
        """
        if task_idx >= len(self._tasks):
            raise IndexError(f"Task index {task_idx} out of range (have {len(self._tasks)})")
        self._current_task = self._tasks[task_idx]
        return self._current_task

    def get_task_description(self) -> str:
        """Return the task description string."""
        if self._current_task is None:
            return ""
        return self._current_task.get("task", "")

    def get_cius(self) -> List[CIU]:
        """Build CIU objects from the current task's CIU data."""
        if self._current_task is None:
            return []
        cius = []
        for ciu_data in self._current_task.get("cius", []):
            cius.append(CIU(
                id=ciu_data["id"],
                type=ciu_data["type"],
                content=ciu_data["content"],
                token_count=ciu_data.get("token_count", len(ciu_data["content"]) // 4),
                source="benchmark",
            ))
        return cius

    def evaluate(self, receiver_output: str) -> bool:
        """
        Deterministic evaluation: does the receiver output contain
        the answer keyword or match the answer pattern?

        This is the ground-truth evaluator (NOT an LLM judge).
        """
        if self._current_task is None:
            return False

        output_lower = receiver_output.lower()

        # Check keyword
        keyword = self._current_task.get("answer_keyword", "")
        if keyword and keyword.lower() in output_lower:
            return True

        # Check regex pattern
        pattern = self._current_task.get("answer_pattern", "")
        if pattern:
            try:
                if re.search(pattern, output_lower, re.IGNORECASE):
                    return True
            except re.error:
                pass

        return False

    def ground_truth(self) -> Dict[str, Any]:
        """Return ground truth (only for evaluation/analysis, never for prompts)."""
        if self._current_task is None:
            return {}
        return self._current_task.get("ground_truth", {})

    def get_necessary_ciu_ids(self) -> List[str]:
        """Return the IDs of CIUs that are necessary for task success."""
        gt = self.ground_truth()
        return gt.get("necessary_cius", [])

    def task_metadata(self) -> Dict[str, Any]:
        """Return task metadata for logging."""
        if self._current_task is None:
            return {}
        return {
            "task_id": self._current_task.get("task_id", ""),
            "category": self._current_task.get("category", ""),
            "difficulty": self._current_task.get("difficulty", ""),
            "receiver_role": self._current_task.get("receiver_role", ""),
            "num_cius": len(self._current_task.get("cius", [])),
        }

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)
