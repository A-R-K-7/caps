"""
Research Instrumentation — extended per-episode, per-decision, and per-update logging.

All data is written to CSV files under ``research/results/`` and is separate
from the main experiment database. This lets us do offline analysis without
touching the SQLite schema.

Row schemas
-----------
Episode row  — one row per (method, budget, seed, split, task)
Decision row — one row per (episode, CIU) with UCB decomposition
Update row   — one row per (episode, arm) after the LinUCB update
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Row data-classes
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRow:
    run_id: str
    experiment_id: str
    episode_id: str
    task_id: str
    method: str
    seed: int
    split: str                    # train | validation | test
    budget_fraction: float        # 1.0 | 0.75 | 0.5 | 0.3 | 0.15
    budget_tokens: int
    policy_frozen: bool

    # Task metadata
    category: str
    difficulty: str
    receiver_role: str

    # CIU counts / tokens
    num_available: int
    num_selected: int
    available_tokens: int
    selected_tokens: int
    compression: float            # selected / available tokens
    api_prompt_tokens: int        # actual LLM API prompt tokens
    api_completion_tokens: int    # actual LLM API output tokens

    # Results
    task_success: bool
    reward: float
    latency_ms: float
    clarification_turns: int

    # Selection quality (from hidden labels — never in prompts)
    precision: float
    recall: float
    misleading_rate: float

    # VOI (empty for non-CAPS methods)
    voi_scores_json: str = "{}"


@dataclass
class DecisionRow:
    run_id: str
    episode_id: str
    ciu_id: str
    ciu_type: str
    ciu_tokens: int
    selected: bool
    budget_remaining: int
    exploitation_score: float
    exploration_bonus: float
    ucb_score: float
    feature_vector_json: str
    policy_version: int


@dataclass
class UpdateRow:
    run_id: str
    episode_id: str
    ciu_type: str
    reward_signal: float
    update_type: str              # voi | global
    arm_norm_A: float             # Frobenius norm of A after update
    arm_norm_b: float             # L2 norm of b after update
    policy_version: int


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class ResearchLogger:
    """
    Writes research instrumentation to CSV files.

    Call ``log_episode``, ``log_decisions``, ``log_update`` during the run,
    then ``flush`` to ensure all rows are written.

    Files created:
        research/results/<run_id>_episodes.csv
        research/results/<run_id>_decisions.csv
        research/results/<run_id>_updates.csv
    """

    def __init__(self, run_id: str, results_dir: str = "research/results", append: bool = False) -> None:
        self.run_id = run_id
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._ep_path = self._dir / f"{run_id}_episodes.csv"
        self._dec_path = self._dir / f"{run_id}_decisions.csv"
        self._upd_path = self._dir / f"{run_id}_updates.csv"

        ep_fieldnames = [f.name for f in fields(EpisodeRow)]
        dec_fieldnames = [f.name for f in fields(DecisionRow)]
        upd_fieldnames = [f.name for f in fields(UpdateRow)]

        # On resume: detect and repair stale headers before opening in append mode.
        if append:
            self._repair_header(self._ep_path, ep_fieldnames)
            self._repair_header(self._dec_path, dec_fieldnames)
            self._repair_header(self._upd_path, upd_fieldnames)

        mode = "a" if append else "w"
        write_header = not append or not self._ep_path.exists()

        self._ep_file = self._ep_path.open(mode, newline="", encoding="utf-8")
        self._dec_file = self._dec_path.open(mode, newline="", encoding="utf-8")
        self._upd_file = self._upd_path.open(mode, newline="", encoding="utf-8")

        self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=ep_fieldnames)
        self._dec_writer = csv.DictWriter(self._dec_file, fieldnames=dec_fieldnames)
        self._upd_writer = csv.DictWriter(self._upd_file, fieldnames=upd_fieldnames)

        if write_header:
            self._ep_writer.writeheader()
            self._dec_writer.writeheader()
            self._upd_writer.writeheader()

    @staticmethod
    def _repair_header(path: Path, expected_fieldnames: list) -> None:
        """If the CSV exists but has a stale header, rewrite the header in-place."""
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                return  # empty file
            if existing_header == expected_fieldnames:
                return  # header is current, nothing to do
            # Header is stale — read all data rows and rewrite the file
            data_rows = [row for row in reader if row]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(expected_fieldnames)
            writer.writerows(data_rows)

    # ------------------------------------------------------------------

    def log_episode(self, row: EpisodeRow) -> None:
        self._ep_writer.writerow(asdict(row))

    def log_decisions(self, rows: List[DecisionRow]) -> None:
        for row in rows:
            self._dec_writer.writerow(asdict(row))

    def log_update(self, row: UpdateRow) -> None:
        self._upd_writer.writerow(asdict(row))

    def flush(self) -> None:
        self._ep_file.flush()
        self._dec_file.flush()
        self._upd_file.flush()

    def close(self) -> None:
        self._ep_file.close()
        self._dec_file.close()
        self._upd_file.close()

    def __enter__(self) -> "ResearchLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Convenience: build decision rows from LinUCB diagnostics

    @staticmethod
    def build_decision_rows(
        run_id: str,
        episode_id: str,
        state: Any,                         # CommunicationState
        available_cius: List[Any],          # List[CIU]
        selected_ids: List[str],
        ciu_scores: Dict[str, float],
        budget_tokens: int,
        policy: Any,                        # LinUCBPolicy or None
        policy_version: int,
    ) -> List[DecisionRow]:
        import numpy as np
        rows: List[DecisionRow] = []
        used = 0
        for ciu in available_cius:
            ucb = ciu_scores.get(ciu.id, 0.0)
            # Decompose UCB if we have the policy internals
            exploitation = 0.0
            exploration = 0.0
            feat_json = "[]"
            if (
                policy is not None
                and hasattr(policy, "_build_context")
                and hasattr(policy, "_arms")
                and hasattr(policy, "_get_or_create_arm")
            ):
                try:
                    import numpy as np
                    x = policy._build_context(state, ciu)
                    dim = x.shape[0]
                    arm = policy._get_or_create_arm(ciu.type, dim)
                    A = arm["A"]
                    try:
                        A_inv = np.linalg.solve(A, np.eye(dim))
                    except np.linalg.LinAlgError:
                        A_inv = np.linalg.pinv(A)
                    theta = A_inv @ arm["b"]
                    exploitation = float((x.T @ theta).item())
                    exploration = float(policy.alpha * np.sqrt((x.T @ A_inv @ x).item()))
                    feat_json = json.dumps([round(float(v), 4) for v in x.flatten()])
                except Exception:
                    pass
            tokens = ciu.estimate_tokens()
            rows.append(DecisionRow(
                run_id=run_id,
                episode_id=episode_id,
                ciu_id=ciu.id,
                ciu_type=ciu.type,
                ciu_tokens=tokens,
                selected=ciu.id in selected_ids,
                budget_remaining=max(0, budget_tokens - used),
                exploitation_score=exploitation,
                exploration_bonus=exploration,
                ucb_score=ucb,
                feature_vector_json=feat_json,
                policy_version=policy_version,
            ))
            if ciu.id in selected_ids:
                used += tokens
        return rows

    @staticmethod
    def build_update_rows(
        run_id: str,
        episode_id: str,
        policy: Any,                        # LinUCBPolicy with _arms
        selected_ids: List[str],
        selected_cius: List[Any],           # List[CIU]
        reward: float,
        voi_scores: Dict[str, float],
        policy_version: int,
    ) -> List[UpdateRow]:
        import numpy as np
        rows: List[UpdateRow] = []
        if not hasattr(policy, "_arms"):
            return rows
        seen_types: set = set()
        for ciu in selected_cius:
            if ciu.id not in selected_ids or ciu.type in seen_types:
                continue
            seen_types.add(ciu.type)
            arm = policy._arms.get(ciu.type)
            if arm is None:
                continue
            update_type = "voi" if (voi_scores and ciu.id in voi_scores) else "global"
            reward_signal = (
                (voi_scores[ciu.id] + reward) / 2.0
                if update_type == "voi"
                else reward
            )
            rows.append(UpdateRow(
                run_id=run_id,
                episode_id=episode_id,
                ciu_type=ciu.type,
                reward_signal=reward_signal,
                update_type=update_type,
                arm_norm_A=float(np.linalg.norm(arm["A"], "fro")),
                arm_norm_b=float(np.linalg.norm(arm["b"])),
                policy_version=policy_version,
            ))
        return rows
