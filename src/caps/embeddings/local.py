"""
Local embedding provider using sentence-transformers.
Includes content-hash caching to avoid redundant computations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from caps.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local sentence-transformers embedding provider with SQLite caching.

    Default model: all-MiniLM-L6-v2 (384-dim, fast, small).
    Cache key: sha256(text) → embedding vector.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_path: Optional[str] = "data/embedding_cache.db",
    ):
        self.model_name = model_name
        self._model = None  # Lazy load
        self._dim: Optional[int] = None
        self._cache_path = cache_path
        self._cache: Dict[str, List[float]] = {}

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initialize SQLite cache table."""
        if not self._cache_path:
            return
        with sqlite3.connect(self._cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_cached(self, text: str) -> Optional[List[float]]:
        """Check in-memory cache, then SQLite cache."""
        h = self._hash(text)
        if h in self._cache:
            return self._cache[h]
        if self._cache_path:
            try:
                with sqlite3.connect(self._cache_path) as conn:
                    row = conn.execute(
                        "SELECT embedding FROM embedding_cache WHERE content_hash = ? AND model = ?",
                        (h, self.model_name),
                    ).fetchone()
                    if row:
                        emb = json.loads(row[0])
                        self._cache[h] = emb
                        return emb
            except Exception:
                pass
        return None

    def _put_cached(self, text: str, embedding: List[float]) -> None:
        """Store in both in-memory and SQLite cache."""
        h = self._hash(text)
        self._cache[h] = embedding
        if self._cache_path:
            try:
                with sqlite3.connect(self._cache_path) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO embedding_cache (content_hash, model, embedding) VALUES (?, ?, ?)",
                        (h, self.model_name, json.dumps(embedding)),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Embedding cache write error: {e}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts, using cache where possible."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            model = self._load_model()
            embeddings = model.encode(uncached_texts, show_progress_bar=False)
            for idx, text, emb in zip(uncached_indices, uncached_texts, embeddings):
                emb_list = emb.tolist()
                results[idx] = emb_list
                self._put_cached(text, emb_list)

        return results  # type: ignore

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed([text])[0]

    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dim is None:
            self._load_model()
        return self._dim  # type: ignore
