"""
LLM Provider abstraction layer.

All LLM interactions in CAPS go through LLMProvider.generate().
No other module should directly import the Groq SDK.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ErrorCategory(str, Enum):
    """Categorized LLM call errors for structured error handling."""
    RATE_LIMIT = "RATE_LIMIT"
    AUTH_ERROR = "AUTH_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    TASK_FAILURE = "TASK_FAILURE"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass
class TokenUsage:
    """Actual token usage from API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    is_estimated: bool = False  # True if usage was estimated, not from API


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    token_usage: TokenUsage
    model: str
    latency_ms: float
    cached: bool = False
    raw_response: Any = None
    error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None


class LLMCallRecord:
    """Record of an LLM call for logging/tracking."""

    def __init__(
        self,
        call_id: str,
        timestamp: float,
        experiment_id: str,
        task_id: str,
        agent_role: str,
        model: str,
        prompt_hash: str,
        prompt_version: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        retry_count: int = 0,
        cached: bool = False,
    ):
        self.call_id = call_id
        self.timestamp = timestamp
        self.experiment_id = experiment_id
        self.task_id = task_id
        self.agent_role = agent_role
        self.model = model
        self.prompt_hash = prompt_hash
        self.prompt_version = prompt_version
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.success = success
        self.error_type = error_type
        self.retry_count = retry_count
        self.cached = cached


class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.

    Every LLM interaction in CAPS must go through this interface.
    Implementations: GroqProvider, MockLLMProvider.
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_schema: Optional[Type[BaseModel]] = None,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt / main content.
            system_prompt: System instruction for the model.
            model: Model identifier (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_schema: Optional Pydantic model for structured JSON output.
            seed: Random seed for reproducibility (best-effort).

        Returns:
            LLMResponse with text, token usage, latency, etc.
        """
        ...

    @abc.abstractmethod
    def list_models(self) -> List[str]:
        """List available model IDs from the provider."""
        ...

    @abc.abstractmethod
    def validate_model(self, model_id: str) -> bool:
        """Check if a model ID is available."""
        ...

    @abc.abstractmethod
    def reset_problem_context(self) -> None:
        """Reset the context of the current problem/task for the provider."""
        ...
