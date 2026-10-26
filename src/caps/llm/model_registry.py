"""
Model Registry — validates and displays available Groq models.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from caps.llm import LLMProvider

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Validates configured models against the provider's available models.
    Supports the `caps models` CLI command.
    """

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._cached_models: Optional[List[str]] = None

    def available_models(self, refresh: bool = False) -> List[str]:
        """Fetch and cache the list of available models."""
        if self._cached_models is None or refresh:
            self._cached_models = self.provider.list_models()
        return self._cached_models

    def validate_config(self, models_config: Dict[str, str]) -> Dict[str, dict]:
        """
        Check each configured model against available models.

        Returns:
            Dict mapping role → {"model": str, "available": bool}
        """
        available = set(self.available_models())
        result = {}
        for role, model_id in models_config.items():
            result[role] = {
                "model": model_id,
                "available": model_id in available or model_id == "mock",
            }
        return result

    def display(self, models_config: Dict[str, str]) -> str:
        """Format model status for CLI display."""
        validation = self.validate_config(models_config)
        lines = ["Model Configuration:"]
        for role, info in validation.items():
            status = "✓" if info["available"] else "✗ NOT FOUND"
            lines.append(f"  {role:12s} → {info['model']:40s} [{status}]")

        available = self.available_models()
        if available:
            lines.append(f"\nAvailable models ({len(available)}):")
            for m in sorted(available):
                lines.append(f"  • {m}")
        else:
            lines.append("\n  (Could not fetch model list — check API key)")

        return "\n".join(lines)
