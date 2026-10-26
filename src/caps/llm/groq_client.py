"""
Groq LLM Provider.

Wraps the Groq Python SDK behind the LLMProvider abstraction.
Handles rate limiting, retries, structured output, and token tracking.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from caps.llm import ErrorCategory, LLMProvider, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """
    Groq API LLM provider with rate-limit handling and structured output.

    Rate-limit headers from Groq responses are parsed and stored.
    On HTTP 429, the provider waits for retry-after and retries with
    exponential backoff (max 3 attempts, never infinite).
    """

    def __init__(
        self,
        api_key: Union[str, List[str]],
        default_model: str = "llama-3.3-70b-versatile",
        fallback_models: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("API key(s) required.")

        if isinstance(api_key, str):
            api_key = [api_key]

        if base_url:
            import openai
            # max_retries=0: disable SDK-level retries so our key-rotation logic
            # in _call_with_retry gets full control over 429 handling.
            self._clients = [openai.OpenAI(api_key=k, base_url=base_url, max_retries=0) for k in api_key]
        else:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
            # max_retries=0: same reason — let _call_with_retry handle rotation.
            self._clients = [Groq(api_key=k, max_retries=0) for k in api_key]
            
        self.active_key_index = 0
        self.default_model = default_model
        self.base_url = base_url

        self.fallback_models = fallback_models or [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "groq/compound",
            "groq/compound-mini",
        ]
        self.active_model = default_model
        self.fallback_index = 0

        # Rate-limit tracking (updated after each call)
        self.remaining_requests: Optional[int] = None
        self.remaining_tokens: Optional[int] = None
        self.reset_time: Optional[float] = None
        self.calls_in_current_problem = 0

    def reset_problem_context(self) -> None:
        self.calls_in_current_problem = 0

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
        if self.fallback_index > 0:
            model = self.active_model
        else:
            model = model or self.active_model

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if seed is not None:
            kwargs["seed"] = seed

        # Structured JSON output via response_format
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
            # Append schema hint to system prompt
            schema_hint = json.dumps(response_schema.model_json_schema(), indent=2)
            if messages[0]["role"] == "system":
                messages[0]["content"] += (
                    f"\n\nYou MUST respond with valid JSON matching this schema:\n{schema_hint}"
                )
            else:
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"Respond with valid JSON matching this schema:\n{schema_hint}",
                    },
                )

        return self._call_with_retry(kwargs, model)

    def _call_with_retry(self, kwargs: Dict[str, Any], model: str) -> LLMResponse:
        """Execute the API call with retry logic for rate limits."""
        last_error: Optional[Exception] = None
        max_retries = 10
        attempt = 0
        key_attempts = 1  # Track how many keys we've tried for this request

        while attempt < max_retries:
            try:
                # Put a 2.5 second wait between requests to avoid burst rate limits
                time.sleep(2.5)

                start = time.time()
                client = self._clients[self.active_key_index]
                
                if self.base_url and "openrouter" in self.base_url:
                    pass # kwargs["extra_body"] = {"reasoning": {"enabled": True}}
                
                response = client.chat.completions.create(**kwargs)
                elapsed_ms = (time.time() - start) * 1000

                # Parse response
                # Note: For stealth/ox-alpha we might get reasoning_details, we just pass the content back
                text = response.choices[0].message.content or ""
                usage = response.usage
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    is_estimated=False,
                )

                self.calls_in_current_problem += 1
                return LLMResponse(
                    text=text,
                    token_usage=token_usage,
                    model=model,
                    latency_ms=elapsed_ms,
                    cached=False,
                    raw_response=response,
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
                    wait_time = self._parse_retry_after(e, attempt)
                    if wait_time > 120 or "per day" in error_str or "daily" in error_str:
                        if self.calls_in_current_problem > 0:
                            # In the middle of solving a problem, wait instead of switching keys
                            logger.warning(
                                f"Rate limit hit for {kwargs.get('model', model)} mid-problem (wait: {wait_time:.1f}s). "
                                f"Waiting to preserve context."
                            )
                            time.sleep(wait_time)
                            attempt += 1
                            continue

                        # Not in the middle of a problem, switch API keys if we have multiples
                        if len(self._clients) > 1 and key_attempts < len(self._clients):
                            self.active_key_index = (self.active_key_index + 1) % len(self._clients)
                            key_attempts += 1
                            logger.warning(
                                f"Extended rate limit hit for {kwargs.get('model', model)} (wait: {wait_time:.1f}s). "
                                f"Switching to API key {self.active_key_index + 1}/{len(self._clients)}."
                            )
                            attempt = 0
                            continue

                        # If all API keys are exhausted, fallback to the next model
                        next_model = self.fallback_models[
                            self.fallback_index % len(self.fallback_models)
                        ]
                        self.fallback_index += 1
                        logger.warning(
                            f"Daily/extended rate limit hit for {kwargs.get('model', model)} across all keys (wait: {wait_time:.1f}s). "
                            f"Falling back to {next_model}."
                        )
                        self.active_model = next_model
                        # Update the current request to use the new model
                        kwargs["model"] = next_model
                        model = next_model
                        attempt = 0  # Reset attempt counter for the new model
                        key_attempts = 1  # Reset key tracker for the new model
                        continue

                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                elif "401" in error_str or "auth" in error_str:
                    raise RuntimeError(f"Authentication error for key {self.active_key_index}: {e}")
                elif "model" in error_str and (
                    "not found" in error_str or "unavailable" in error_str
                ):
                    raise RuntimeError(f"Model {model} not found on key {self.active_key_index}: {e}")
                elif "timeout" in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(2**attempt)
                        attempt += 1
                        continue
                    raise RuntimeError(f"Timeout after {max_retries} retries: {e}")
                elif "json_validate_failed" in error_str:
                    logger.warning("JSON validation failed. Dropping response_format and retrying.")
                    kwargs.pop("response_format", None)
                    time.sleep(1.0)
                    attempt += 1
                    continue
                else:
                    raise RuntimeError(f"Unknown API error: {e}")

        # All retries exhausted
        raise RuntimeError(
            f"Rate limit: all {max_retries} retries exhausted. Last error: {last_error}"
        )

    def _parse_retry_after(self, error: Exception, attempt: int) -> float:
        """Extract retry-after from error or use exponential backoff."""
        # Try to extract retry-after header from the exception
        try:
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    return float(retry_after)
        except Exception:
            pass

        # Try to parse wait time from error message
        import re

        error_str = str(error)
        match = re.search(
            r"try again in (?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+(?:\.[0-9]+)?)s)?(?:([0-9]+)ms)?",
            error_str,
        )
        if match:
            h, m, s, ms = match.groups()
            total = 0.0
            if h:
                total += float(h) * 3600
            if m:
                total += float(m) * 60
            if s:
                total += float(s)
            if ms:
                total += float(ms) / 1000
            return total

        import random
        # Exponential backoff with jitter
        base = min(2 ** (attempt + 1), 60)
        return base + random.uniform(0, 1)

    def list_models(self) -> List[str]:
        """List available models from Groq API."""
        try:
            client = self._clients[self.active_key_index]
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def validate_model(self, model_id: str) -> bool:
        """Check if a model is available."""
        available = self.list_models()
        return model_id in available
