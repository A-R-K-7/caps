"""
LLM Response Cache.

Content-hash based caching of LLM responses in SQLite.
Cache key = hash(model, prompt, system_prompt, temperature, max_tokens, seed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from caps.llm import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class LLMCache:
    """
    SQLite-backed LLM response cache.

    Caching must be explicitly enabled in config.
    For learning experiments, caching behavior must be made explicit
    to prevent contaminating exploration.
    """

    def __init__(self, db_path: str = "data/llm_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def make_key(
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> str:
        """Deterministic cache key from request parameters."""
        key_data = json.dumps({
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int] = None,
    ) -> Optional[LLMResponse]:
        """Look up a cached response. Returns None on miss."""
        key = self.make_key(model, prompt, system_prompt, temperature, max_tokens, seed)
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT response_text, prompt_tokens, completion_tokens, total_tokens "
                    "FROM llm_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row:
                    return LLMResponse(
                        text=row[0],
                        token_usage=TokenUsage(
                            prompt_tokens=row[1],
                            completion_tokens=row[2],
                            total_tokens=row[3],
                            is_estimated=False,
                        ),
                        model=model,
                        latency_ms=0,
                        cached=True,
                    )
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def put(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        response: LLMResponse,
        seed: Optional[int] = None,
    ) -> None:
        """Store a response in the cache."""
        key = self.make_key(model, prompt, system_prompt, temperature, max_tokens, seed)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache "
                    "(cache_key, model, response_text, prompt_tokens, completion_tokens, total_tokens) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        model,
                        response.text,
                        response.token_usage.prompt_tokens,
                        response.token_usage.completion_tokens,
                        response.token_usage.total_tokens,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> None:
        """Clear all cached responses."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()

    def stats(self) -> dict:
        """Return cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
            return {"entries": count, "path": self.db_path}
