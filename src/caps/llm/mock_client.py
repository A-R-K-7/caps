"""
Mock LLM Provider for unit and integration tests.

Produces deterministic responses without any API calls.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from caps.llm import LLMProvider, LLMResponse, TokenUsage


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM provider.

    Supports:
    - Fixed response mapping (prompt_hash → response)
    - Schema-aware default responses
    - Simulated token usage and latency
    - Call tracking for test assertions
    """

    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "Mock LLM response.",
        latency_ms: float = 10.0,
        token_multiplier: float = 0.25,  # tokens ≈ chars * multiplier
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.latency_ms = latency_ms
        self.token_multiplier = token_multiplier
        self.call_log: List[Dict[str, Any]] = []

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
        start = time.time()
        model = model or "mock-model"

        # Look up by prompt hash or exact prompt
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        text = self.responses.get(prompt_hash)
        if text is None:
            text = self.responses.get(prompt, None)

        # If no stored response, generate a schema-aware default
        if text is None:
            if response_schema is not None:
                text = self._generate_schema_response(response_schema)
            else:
                text = self.default_response

        prompt_tokens = int(len(prompt) * self.token_multiplier)
        completion_tokens = int(len(text) * self.token_multiplier)

        # Track call
        self.call_log.append({
            "prompt": prompt[:200],
            "system_prompt": system_prompt[:200],
            "model": model,
            "response": text[:200],
            "prompt_hash": prompt_hash,
        })

        elapsed = (time.time() - start) * 1000 + self.latency_ms
        return LLMResponse(
            text=text,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                is_estimated=True,
            ),
            model=model,
            latency_ms=elapsed,
            cached=False,
        )

    def _generate_schema_response(self, schema: Type[BaseModel]) -> str:
        """Generate a minimal valid JSON response matching the Pydantic schema."""
        try:
            json_schema = schema.model_json_schema()
            return json.dumps(self._build_default_from_schema(json_schema))
        except Exception:
            return "{}"

    def _build_default_from_schema(self, schema: Dict[str, Any]) -> Any:
        """Recursively build a default object from a JSON schema."""
        schema_type = schema.get("type", "object")

        if schema_type == "object":
            props = schema.get("properties", {})
            obj = {}
            for key, prop in props.items():
                # Resolve $ref / $defs if present
                if "$ref" in prop:
                    ref_path = prop["$ref"].split("/")[-1]
                    defs = schema.get("$defs", {})
                    prop = defs.get(ref_path, prop)
                obj[key] = self._build_default_from_schema(prop)
            return obj
        elif schema_type == "array":
            items = schema.get("items", {})
            return [self._build_default_from_schema(items)]
        elif schema_type == "string":
            return schema.get("default", "mock_value")
        elif schema_type == "number" or schema_type == "integer":
            return schema.get("default", 0)
        elif schema_type == "boolean":
            return schema.get("default", False)
        else:
            return None

    def list_models(self) -> List[str]:
        return ["mock-model"]

    def validate_model(self, model_id: str) -> bool:
        return True

    def reset(self) -> None:
        """Clear call log."""
        self.call_log.clear()

    def reset_problem_context(self) -> None:
        """No-op for mock provider."""
        pass

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def num_calls(self) -> int:
        return len(self.call_log)
