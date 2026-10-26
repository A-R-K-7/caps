"""
Experiment Database — SQLite schema and ORM models.

Tables (§43):
    experiments    — top-level experiment records
    episodes       — per-task execution records
    llm_calls      — every LLM API call
    ciu_selections — which CIUs were selected per episode
    metrics        — computed metrics per experiment
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class ExperimentRecord(Base):
    """Top-level experiment record."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), unique=True, nullable=False)
    name = Column(String(256), nullable=False)
    config_yaml = Column(Text, nullable=True)
    seed = Column(Integer, nullable=False, default=42)
    policy_name = Column(String(128), nullable=False)
    planner_model = Column(String(128), nullable=True)
    receiver_model = Column(String(128), nullable=True)
    evaluator_model = Column(String(128), nullable=True)
    dataset_name = Column(String(128), nullable=True)
    status = Column(String(32), default="running")  # running | completed | failed | paused
    total_episodes = Column(Integer, default=0)
    completed_episodes = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class EpisodeRecord(Base):
    """Per-episode execution record (§44)."""
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    episode_id = Column(String(64), nullable=False)
    task_id = Column(String(128), nullable=False)
    seed = Column(Integer, nullable=False)
    task_category = Column(String(64), nullable=True)

    # Models
    planner_model = Column(String(128), nullable=True)
    receiver_model = Column(String(128), nullable=True)

    # CIU info
    available_ciu_ids = Column(Text, nullable=True)  # JSON list
    selected_ciu_ids = Column(Text, nullable=True)   # JSON list
    num_available_cius = Column(Integer, default=0)
    num_selected_cius = Column(Integer, default=0)

    # Token usage
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)

    # Timing
    latency_ms = Column(Float, default=0.0)

    # Results
    task_success = Column(Boolean, nullable=False)
    receiver_status = Column(String(32), nullable=True)
    clarification_turns = Column(Integer, default=0)
    retry_count = Column(Integer, default=0)

    # Reward
    reward = Column(Float, default=0.0)

    # VOI
    voi_scores = Column(Text, nullable=True)  # JSON dict

    # Policy state
    policy_version = Column(Integer, default=0)

    created_at = Column(DateTime, server_default=func.now())


class LLMCallRecord(Base):
    """LLM API call record (§45)."""
    __tablename__ = "llm_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    call_id = Column(String(64), nullable=False)
    timestamp = Column(Float, nullable=False)
    experiment_id = Column(String(64), nullable=True, index=True)
    task_id = Column(String(128), nullable=True)
    agent_role = Column(String(32), nullable=False)  # planner | receiver | evaluator
    model = Column(String(128), nullable=False)
    prompt_hash = Column(String(64), nullable=True)
    prompt_version = Column(String(32), nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True)
    error_type = Column(String(64), nullable=True)
    retry_count = Column(Integer, default=0)
    cached = Column(Boolean, default=False)


class MetricRecord(Base):
    """Computed metrics per experiment."""
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), nullable=False, index=True)
    metric_name = Column(String(128), nullable=False)
    metric_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class ExperimentDB:
    """
    Manages the SQLite experiment database.

    All experiment data is persisted here for reproducibility.
    """

    def __init__(self, db_path: str = "data/experiments.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine)

    def session(self) -> Session:
        return self._Session()

    # --- Experiments ---

    def create_experiment(self, **kwargs) -> ExperimentRecord:
        with self.session() as s:
            record = ExperimentRecord(**kwargs)
            s.add(record)
            s.commit()
            s.refresh(record)
            return record

    def update_experiment(self, experiment_id: str, **kwargs) -> None:
        with self.session() as s:
            s.query(ExperimentRecord).filter_by(
                experiment_id=experiment_id
            ).update(kwargs)
            s.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        with self.session() as s:
            return s.query(ExperimentRecord).filter_by(
                experiment_id=experiment_id
            ).first()

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment and all its associated records."""
        with self.session() as s:
            s.query(EpisodeRecord).filter_by(experiment_id=experiment_id).delete()
            s.query(MetricRecord).filter_by(experiment_id=experiment_id).delete()
            s.query(LLMCallRecord).filter_by(experiment_id=experiment_id).delete()
            s.query(ExperimentRecord).filter_by(experiment_id=experiment_id).delete()
            s.commit()

    # --- Episodes ---

    def add_episode(self, **kwargs) -> EpisodeRecord:
        # Serialize list/dict fields to JSON
        for key in ("available_ciu_ids", "selected_ciu_ids", "voi_scores"):
            if key in kwargs and not isinstance(kwargs[key], str):
                kwargs[key] = json.dumps(kwargs[key])

        with self.session() as s:
            record = EpisodeRecord(**kwargs)
            s.add(record)
            s.commit()
            s.refresh(record)
            return record

    def get_episodes(self, experiment_id: str) -> List[EpisodeRecord]:
        with self.session() as s:
            return s.query(EpisodeRecord).filter_by(
                experiment_id=experiment_id
            ).all()

    def count_episodes(self, experiment_id: str) -> int:
        with self.session() as s:
            return s.query(EpisodeRecord).filter_by(
                experiment_id=experiment_id
            ).count()

    # --- LLM Calls ---

    def add_llm_call(self, **kwargs) -> None:
        with self.session() as s:
            record = LLMCallRecord(**kwargs)
            s.add(record)
            s.commit()

    # --- Metrics ---

    def add_metric(self, experiment_id: str, name: str, value: float) -> None:
        with self.session() as s:
            record = MetricRecord(
                experiment_id=experiment_id,
                metric_name=name,
                metric_value=value,
            )
            s.add(record)
            s.commit()

    def get_metrics(self, experiment_id: str) -> Dict[str, float]:
        with self.session() as s:
            records = s.query(MetricRecord).filter_by(
                experiment_id=experiment_id
            ).all()
            return {r.metric_name: r.metric_value for r in records}

    # --- Utilities ---

    def get_all_experiments(self) -> List[ExperimentRecord]:
        with self.session() as s:
            return s.query(ExperimentRecord).order_by(
                ExperimentRecord.created_at.desc()
            ).all()

    def episodes_to_dicts(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Export episodes as plain dicts for analysis."""
        with self.session() as s:
            episodes = s.query(EpisodeRecord).filter_by(
                experiment_id=experiment_id
            ).all()
            results = []
            for ep in episodes:
                d = {
                    "experiment_id": ep.experiment_id,
                    "episode_id": ep.episode_id,
                    "task_id": ep.task_id,
                    "seed": ep.seed,
                    "task_category": ep.task_category,
                    "num_available_cius": ep.num_available_cius,
                    "num_selected_cius": ep.num_selected_cius,
                    "total_tokens": ep.total_tokens,
                    "latency_ms": ep.latency_ms,
                    "task_success": ep.task_success,
                    "clarification_turns": ep.clarification_turns,
                    "retry_count": ep.retry_count,
                    "reward": ep.reward,
                    "policy_version": ep.policy_version,
                }
                results.append(d)
            return results
