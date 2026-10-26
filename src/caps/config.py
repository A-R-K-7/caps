"""
CAPS Configuration System.

Configuration precedence (highest to lowest):
    CLI arguments → environment variables → YAML config → safe defaults

All experiment parameters are centralized here for reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------

class ModelsConfig(BaseModel):
    """LLM model identifiers for each agent role."""
    planner: str = "llama-3.1-8b-instant"
    receiver: str = "llama-3.1-8b-instant"
    evaluator: str = "llama-3.1-8b-instant"


class PolicyConfig(BaseModel):
    """Communication policy hyperparameters."""
    algorithm: str = "linucb"
    alpha: float = 1.0
    selection_strategy: str = "greedy_budget"
    epsilon: float = 0.1  # for epsilon-greedy exploration


class BudgetConfig(BaseModel):
    """Token / request budget constraints."""
    max_daily_tokens: int = 450_000
    max_requests_per_run: int = 500
    max_tokens_per_task: int = 10_000
    max_tokens_per_message: int = 4_000


class RewardConfig(BaseModel):
    """Reward function weights."""
    lambda_cost: float = 0.5
    weights_tokens: float = 0.4
    weights_latency: float = 0.2
    weights_clarification: float = 0.2
    weights_retries: float = 0.2


class EvaluatorConfig(BaseModel):
    """Communication evaluator settings."""
    counterfactual: bool = True
    max_counterfactuals_per_task: int = 3
    counterfactual_strategy: str = "uncertainty"  # uncertainty | top_utility | random | clarification_related
    judge_enabled: bool = False


class DatasetConfig(BaseModel):
    """Synthetic dataset generation parameters."""
    name: str = "synthetic_v1"
    train_size: int = 100
    validation_size: int = 50
    test_size: int = 50
    seed: int = 42
    families: List[str] = Field(
        default_factory=lambda: [
            "debugging",
            "configuration",
            "data_transformation",
            "code_repair",
            "documentation",
        ]
    )


class CacheConfig(BaseModel):
    """LLM and embedding cache settings."""
    enabled: bool = True
    llm_cache_path: str = "data/llm_cache.db"
    embedding_cache_path: str = "data/embedding_cache.db"


class CheckpointConfig(BaseModel):
    """Experiment checkpointing."""
    enabled: bool = True
    every_n_episodes: int = 10
    directory: str = "experiments/checkpoints"


class QuotaConfig(BaseModel):
    """API quota management."""
    warning_percent: int = 20
    auto_pause: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""
    name: str = "caps_experiment"
    seed: int = 42
    seeds: List[int] = Field(default_factory=lambda: [42, 123, 456, 789])
    temperature: float = 0.0

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)

    # Paths
    database_path: str = "data/experiments.db"
    results_dir: str = "experiments/results"
    embedding_model: str = "all-MiniLM-L6-v2"
    verbose: bool = False

    @field_validator("seeds", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> List[int]:
        if isinstance(v, int):
            return [v]
        return v


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base*."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    yaml_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """
    Build an ExperimentConfig with the correct precedence:
        CLI arguments → environment variables → YAML config → safe defaults
    """
    raw: Dict[str, Any] = {}

    # 1. YAML config
    if yaml_path is not None:
        raw = load_yaml_config(yaml_path)

    # 2. Environment-variable overrides (selected keys)
    env_map = {
        "GROQ_API_KEY": None,  # handled separately, never stored in config
        "CAPS_PLANNER_MODEL": ("models", "planner"),
        "CAPS_RECEIVER_MODEL": ("models", "receiver"),
        "CAPS_EVALUATOR_MODEL": ("models", "evaluator"),
        "CAPS_SEED": ("seed",),
        "CAPS_TEMPERATURE": ("temperature",),
        "CAPS_VERBOSE": ("verbose",),
    }
    for env_key, config_path in env_map.items():
        val = os.environ.get(env_key)
        if val is not None and config_path is not None:
            # Navigate / create nested dicts
            d = raw
            for part in config_path[:-1]:
                d = d.setdefault(part, {})
            # Type coerce simple cases
            if config_path[-1] in ("seed",):
                val = int(val)
            elif config_path[-1] in ("temperature",):
                val = float(val)
            elif config_path[-1] in ("verbose",):
                val = val.lower() in ("true", "1", "yes")
            d[config_path[-1]] = val

    # 3. CLI overrides
    if cli_overrides:
        raw = _deep_merge(raw, cli_overrides)

    # Flatten nested dicts matching sub-model field names
    # (Pydantic handles this natively)
    return ExperimentConfig(**raw)


def get_groq_api_keys() -> List[str]:
    """Return the Groq API keys from the environment. Never store in config objects."""
    import json
    
    # Check for GROQ_API_KEYS array or comma-separated string
    keys_str = os.environ.get("GROQ_API_KEYS", "")
    if not keys_str:
        # Check manually in .env
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEYS="):
                    keys_str = line.split("=", 1)[1].strip()
                    break
                    
    if keys_str:
        try:
            parsed = json.loads(keys_str)
            if isinstance(parsed, list):
                return [str(k) for k in parsed]
        except json.JSONDecodeError:
            return [k.strip() for k in keys_str.split(",") if k.strip()]

    # Fallback to single GROQ_API_KEY
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    
    return [key] if key else []


def get_openrouter_api_keys() -> List[str]:
    """Return the OpenRouter API keys from the environment."""
    import json
    
    keys_str = os.environ.get("OPEN_ROUTER_API_KEY", "")
    if not keys_str:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPEN_ROUTER_API_KEY="):
                    keys_str = line.split("=", 1)[1].strip()
                    break
                    
    if keys_str:
        try:
            parsed = json.loads(keys_str)
            if isinstance(parsed, list):
                return [str(k) for k in parsed]
        except json.JSONDecodeError:
            return [k.strip() for k in keys_str.split(",") if k.strip()]

    return [keys_str] if keys_str else []
