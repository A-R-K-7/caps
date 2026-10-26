"""
Candidate Information Unit (CIU) — the fundamental data unit in CAPS.

Every piece of information that could be communicated from the planner
to the receiver is represented as a CIU.
CIUs must be independently selectable.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CIU(BaseModel):
    """
    Candidate Information Unit.

    Attributes:
        id: Unique identifier.
        type: Information type (e.g., task_description, stack_trace, source_code).
        content: The actual information content.
        source: Origin of the information (e.g., benchmark, planner).
        token_count: Number of tokens in content.
        embedding: Semantic embedding vector (populated by EmbeddingProvider).
        priority: Optional priority score (used by heuristic baselines).
        historical_utility: Running average of observed utility.
        historical_selection_rate: Fraction of episodes this CIU was selected.
        uncertainty: Current uncertainty in utility estimate.
        metadata: Arbitrary extra metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str
    content: str
    source: str = "benchmark"
    token_count: int = 0
    embedding: Optional[List[float]] = None
    priority: float = 0.5
    historical_utility: float = 0.0
    historical_selection_rate: float = 0.0
    uncertainty: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def estimate_tokens(self) -> int:
        """Estimate token count from content length (1 token ≈ 4 chars)."""
        if self.token_count > 0:
            return self.token_count
        return max(1, len(self.content) // 4)

    class Config:
        arbitrary_types_allowed = True
