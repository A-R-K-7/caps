"""
Embedding Provider abstraction.
"""

from __future__ import annotations

import abc
from typing import List


class EmbeddingProvider(abc.ABC):
    """Abstract base for embedding providers. Always runs locally."""

    @abc.abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""
        ...

    @abc.abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
