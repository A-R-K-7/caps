"""
Synthetic v2 Environment — deterministic hidden-label evaluator.

Ground-truth labels (information_role, necessity, etc.) are stored in
``synthetic_v2.py`` task dicts but are NEVER exposed to agents, policies,
or LLM prompts. This module is the only place that reads them.

Selection quality metrics (precision, recall, misleading rate) are computed
here using the hidden labels and returned alongside the Boolean success signal.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from caps.communication.ciu import CIU
from caps.datasets.synthetic_v2 import HIDDEN_FIELDS, visible_ciu


class SyntheticV2Environment:
    """
    Environment for the synthetic_v2 benchmark.

    Key properties:
    - Evaluation is fully deterministic (no LLM judge).
    - Hidden benchmark labels never leave this class.
    - ``get_cius()`` strips all hidden fields before returning CIU objects.
    - ``selection_quality()`` uses hidden labels to compute precision/recall.
    """

    def __init__(self) -> None:
        self._tasks: List[Dict[str, Any]] = []
        self._current_task: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def load_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Load a list of task dicts (may contain hidden fields — that is fine)."""
        self._tasks = tasks

    def filter_to_ids(self, task_ids: List[str]) -> None:
        """Restrict the loaded task list to the given IDs (for split slicing)."""
        id_set = set(task_ids)
        self._tasks = [t for t in self._tasks if t["task_id"] in id_set]

    def reset(self, task_idx: int) -> Dict[str, Any]:
        """Set the current task by index. Returns the full task dict (with hidden fields)."""
        if task_idx >= len(self._tasks):
            raise IndexError(
                f"Task index {task_idx} out of range (have {len(self._tasks)})"
            )
        self._current_task = self._tasks[task_idx]
        return self._current_task

    # ------------------------------------------------------------------
    # Accessors (hidden-field-safe)
    # ------------------------------------------------------------------

    def get_task_description(self) -> str:
        """Return the task description string (safe to pass to agents)."""
        if self._current_task is None:
            return ""
        return self._current_task.get("task", "")

    def get_cius(self) -> List[CIU]:
        """
        Build CIU objects stripped of all hidden benchmark fields.

        These objects are safe to pass to planner, receiver, and policy.
        """
        if self._current_task is None:
            return []
        cius: List[CIU] = []
        for raw in self._current_task.get("cius", []):
            clean = visible_ciu(raw)
            cius.append(
                CIU(
                    id=clean["id"],
                    type=clean["type"],
                    content=clean["content"],
                    token_count=clean.get("token_count", len(clean["content"]) // 4),
                    source=clean.get("source", "synthetic_v2"),
                )
            )
        return cius

    def task_metadata(self) -> Dict[str, Any]:
        """Return task metadata for logging (no hidden fields)."""
        if self._current_task is None:
            return {}
        return {
            "task_id": self._current_task.get("task_id", ""),
            "category": self._current_task.get("category", ""),
            "difficulty": self._current_task.get("difficulty", ""),
            "receiver_role": self._current_task.get("receiver_role", ""),
            "num_cius": len(self._current_task.get("cius", [])),
            "benchmark_version": "synthetic_v2",
        }

    # ------------------------------------------------------------------
    # Evaluation (uses hidden fields — internal only)
    # ------------------------------------------------------------------

    def evaluate(self, receiver_output: str) -> bool:
        """
        Deterministic success check.

        Returns True if the receiver output contains the answer keyword
        (``answer_key``) or a substring of the answer text (``answer_text``).
        No LLM is involved.
        """
        if self._current_task is None:
            return False

        output_lower = receiver_output.lower()

        # Primary: answer_key keyword (exact word)
        key = self._current_task.get("answer_key", "")
        if key and _keyword_match(key, output_lower):
            return True

        # Secondary: answer_text fragment (first meaningful chunk)
        answer_text = self._current_task.get("answer_text", "")
        if answer_text:
            # Take the first meaningful phrase (up to the first period/comma)
            snippet = re.split(r"[,.]", answer_text)[0].strip().lower()
            if len(snippet) >= 8 and snippet in output_lower:
                return True

        return False

    def selection_quality(
        self,
        selected_ids: List[str],
    ) -> Dict[str, float]:
        """
        Compute hidden-label selection quality for the current task.

        Returns:
            precision        — fraction of selected CIUs that are useful
                               (role in {necessary, useful, conditional})
            recall           — fraction of necessary CIUs that were selected
            misleading_rate  — fraction of selected CIUs that are misleading
            compression      — selected_tokens / total_tokens
        """
        if self._current_task is None:
            return {"precision": 0.0, "recall": 0.0, "misleading_rate": 0.0, "compression": 0.0}

        selected_set: Set[str] = set(selected_ids)
        gt = self._current_task.get("ground_truth", {})
        necessary_set: Set[str] = set(gt.get("necessary_cius", []))
        useful_set: Set[str] = set(gt.get("useful_cius", []))
        misleading_set: Set[str] = set(gt.get("misleading_cius", []))

        # Token counts from raw CIU data (hidden fields present)
        total_tokens = sum(
            c.get("token_count", len(c.get("content", "")) // 4)
            for c in self._current_task.get("cius", [])
        )
        selected_tokens = sum(
            c.get("token_count", len(c.get("content", "")) // 4)
            for c in self._current_task.get("cius", [])
            if c["id"] in selected_set
        )

        precision = (
            len(selected_set & useful_set) / len(selected_set)
            if selected_set
            else 0.0
        )
        recall = (
            len(selected_set & necessary_set) / len(necessary_set)
            if necessary_set
            else 1.0
        )
        misleading_rate = (
            len(selected_set & misleading_set) / len(selected_set)
            if selected_set
            else 0.0
        )
        compression = selected_tokens / total_tokens if total_tokens > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "misleading_rate": misleading_rate,
            "compression": compression,
        }

    def ground_truth_necessary_ids(self) -> List[str]:
        """Return necessary CIU IDs for the current task (internal use only)."""
        if self._current_task is None:
            return []
        return self._current_task.get("ground_truth", {}).get("necessary_cius", [])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_tasks(self) -> int:
        return len(self._tasks)


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _keyword_match(keyword: str, text: str) -> bool:
    """Case-insensitive word-boundary match for keyword in text."""
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    if keyword_lower in text_lower:
        return True
    if "_" in keyword_lower:
        pattern = r"\b" + re.escape(keyword_lower).replace(r"\_", r"[\s\-_]") + r"\b"
    else:
        pattern = r"\b" + re.escape(keyword_lower) + r"\b"
    try:
        return bool(re.search(pattern, text_lower))
    except re.error:
        return keyword_lower in text_lower
