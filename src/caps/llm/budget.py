"""
Budget Manager — API quota tracking and enforcement.

Tracks token usage per experiment and daily totals.
Prevents exceeding configured budgets before making requests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Cumulative usage for a single experiment or daily window."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    start_time: float = field(default_factory=time.time)


class BudgetManager:
    """
    Enforces token and request budgets.

    Before every LLM request, callers must check can_make_request().
    After every LLM response, callers must call record_usage().

    Configuration:
        max_daily_tokens: Total tokens allowed per day.
        max_requests_per_run: Max requests in a single experiment run.
        max_tokens_per_task: Max tokens per individual task.
    """

    def __init__(
        self,
        max_daily_tokens: int = 450_000,
        max_requests_per_run: int = 500,
        max_tokens_per_task: int = 10_000,
    ):
        self.max_daily_tokens = max_daily_tokens
        self.max_requests_per_run = max_requests_per_run
        self.max_tokens_per_task = max_tokens_per_task

        self._daily = UsageRecord()
        self._run = UsageRecord()
        self._task_usage: Dict[str, int] = {}  # task_id → tokens

    def can_make_request(
        self,
        estimated_tokens: int = 0,
        task_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Check whether a request fits within all configured budgets.

        Returns:
            (allowed, reason): allowed is True if the request can proceed.
            If False, reason explains which budget would be exceeded.
        """
        # Daily token budget
        if self._daily.total_tokens + estimated_tokens > self.max_daily_tokens:
            return False, (
                f"Daily token budget exceeded: "
                f"{self._daily.total_tokens}/{self.max_daily_tokens} used, "
                f"request needs ~{estimated_tokens}"
            )

        # Run request budget
        if self._run.requests >= self.max_requests_per_run:
            return False, (
                f"Run request limit reached: "
                f"{self._run.requests}/{self.max_requests_per_run}"
            )

        # Per-task token budget
        if task_id is not None:
            task_used = self._task_usage.get(task_id, 0)
            if task_used + estimated_tokens > self.max_tokens_per_task:
                return False, (
                    f"Task {task_id} token budget exceeded: "
                    f"{task_used}/{self.max_tokens_per_task} used"
                )

        return True, "OK"

    def reserve(self, estimated_tokens: int, task_id: Optional[str] = None) -> bool:
        """
        Reserve tokens before making a request.
        Returns True if reservation succeeded.
        """
        allowed, reason = self.can_make_request(estimated_tokens, task_id)
        if not allowed:
            logger.warning(f"Budget reservation denied: {reason}")
            return False
        return True

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        task_id: Optional[str] = None,
    ) -> None:
        """Record actual token usage after a successful request."""
        self._daily.total_tokens += total_tokens
        self._daily.prompt_tokens += prompt_tokens
        self._daily.completion_tokens += completion_tokens
        self._daily.requests += 1

        self._run.total_tokens += total_tokens
        self._run.prompt_tokens += prompt_tokens
        self._run.completion_tokens += completion_tokens
        self._run.requests += 1

        if task_id is not None:
            self._task_usage[task_id] = self._task_usage.get(task_id, 0) + total_tokens

    def remaining(self) -> Dict[str, int]:
        """Return remaining budget across all dimensions."""
        return {
            "daily_tokens_remaining": max(0, self.max_daily_tokens - self._daily.total_tokens),
            "daily_tokens_used": self._daily.total_tokens,
            "run_requests_remaining": max(0, self.max_requests_per_run - self._run.requests),
            "run_requests_used": self._run.requests,
            "run_tokens_used": self._run.total_tokens,
        }

    def reset_daily(self) -> None:
        """Reset the daily budget counter."""
        self._daily = UsageRecord()

    def reset_run(self) -> None:
        """Reset the run budget counter."""
        self._run = UsageRecord()
        self._task_usage.clear()

    def warning_level(self, warning_percent: int = 20) -> Optional[str]:
        """
        Check if budget is approaching limits.
        Returns a warning message or None.
        """
        remaining = self.remaining()
        daily_pct = (remaining["daily_tokens_remaining"] / max(1, self.max_daily_tokens)) * 100

        if daily_pct <= 0:
            return "CRITICAL: Daily token budget exhausted."
        elif daily_pct <= warning_percent:
            return (
                f"WARNING: Only {daily_pct:.0f}% of daily token budget remaining "
                f"({remaining['daily_tokens_remaining']} tokens)."
            )
        return None
