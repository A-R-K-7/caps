"""
Research Validation — data quality and leakage checks.

Validates that:
1. No hidden benchmark labels appear in visible CIU content.
2. Budgets actually bind at restricted fractions (selected ≤ budget).
3. Train / validation / test splits are disjoint.
4. Test episodes are frozen (policy_frozen == True in episode CSV).
5. Pilot-DB (caps_experiments.db) is untouched.

Usage::

    from caps.research.validation import ResearchValidator
    v = ResearchValidator()
    report = v.validate_all(
        episode_csv="research/results/<run_id>_episodes.csv",
        splits_path="data/research/splits_1.json",
    )
    v.print_report(report)
    assert report["overall"] == "PASS"
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HIDDEN_FIELD_LABELS = [
    "information_role",
    "ground_truth_necessity",
    "ground_truth_sufficiency",
    "dependency_group",
    "source_agent",
    "benchmark_generator",     # string value of source_agent
]

PILOT_DB_PATH = "data/caps_experiments.db"


class ResearchValidator:
    """Runs a battery of validation checks on a completed research run."""

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def validate_all(
        self,
        episode_csv: str | Path,
        splits_path: str | Path,
        dataset_jsonl: Optional[str | Path] = None,
        pilot_db_path: str | Path = PILOT_DB_PATH,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {"checks": [], "overall": "PASS"}

        def add(name: str, status: str, msg: str) -> None:
            report["checks"].append({"name": name, "status": status, "message": msg})
            if status == "FAIL":
                report["overall"] = "FAIL"
            elif status == "WARN" and report["overall"] == "PASS":
                report["overall"] = "WARN"

        # 1. Episode CSV exists
        ep_path = Path(episode_csv)
        if not ep_path.exists():
            add("episode_csv_exists", "FAIL", f"Not found: {ep_path}")
            return report
        add("episode_csv_exists", "PASS", str(ep_path))

        rows = _load_episode_csv(ep_path)

        # 2. Splits are disjoint
        splits_path = Path(splits_path)
        if splits_path.exists():
            disjoint_ok, disjoint_msg = _check_splits_disjoint(splits_path)
            add("splits_disjoint", "PASS" if disjoint_ok else "FAIL", disjoint_msg)
        else:
            add("splits_disjoint", "WARN", f"Splits file not found: {splits_path}")

        # 3. Test episodes are frozen
        frozen_ok, frozen_msg = _check_test_frozen(rows)
        add("test_episodes_frozen", "PASS" if frozen_ok else "FAIL", frozen_msg)

        # 4. Budgets bind (selected_tokens <= budget_tokens for budgeted methods)
        budget_ok, budget_msg = _check_budgets_bind(rows)
        add("budgets_bind", "PASS" if budget_ok else "FAIL", budget_msg)

        # 5. Pilot DB untouched (check mtime hasn't changed significantly)
        pilot_ok, pilot_msg = _check_pilot_db_untouched(pilot_db_path)
        add("pilot_db_untouched", "PASS" if pilot_ok else "WARN", pilot_msg)

        # 6. Label leakage check (if dataset JSONL provided)
        if dataset_jsonl is not None and Path(dataset_jsonl).exists():
            leak_ok, leak_msg = _check_label_leakage(Path(dataset_jsonl))
            add("label_leakage", "PASS" if leak_ok else "FAIL", leak_msg)
        else:
            add("label_leakage", "WARN", "Dataset JSONL not provided; skipping leakage check")

        # 7. Episode count sanity
        if not rows:
            add("episode_count", "FAIL", "No episode rows found in CSV")
        else:
            n_test = sum(1 for r in rows if r.get("split") == "test")
            add(
                "episode_count",
                "PASS",
                f"Total rows: {len(rows)}, test rows: {n_test}",
            )

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        overall = report.get("overall", "?")
        symbol = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(overall, "?")
        color_start = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m"}.get(overall, "")
        color_end = "\033[0m"
        print(f"\n{color_start}{'='*60}")
        print(f"  Research Validation: {symbol} {overall}")
        print(f"{'='*60}{color_end}")
        for check in report.get("checks", []):
            sym = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}.get(check["status"], "?")
            c_s = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m"}.get(check["status"], "")
            print(f"  {c_s}{sym} [{check['status']}]{color_end} {check['name']}: {check['message']}")
        print()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _load_episode_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_splits_disjoint(splits_path: Path) -> Tuple[bool, str]:
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    train = set(splits.get("train", []))
    val = set(splits.get("validation", []))
    test = set(splits.get("test", []))

    overlaps = []
    if train & val:
        overlaps.append(f"train∩val={len(train & val)}")
    if train & test:
        overlaps.append(f"train∩test={len(train & test)}")
    if val & test:
        overlaps.append(f"val∩test={len(val & test)}")

    if overlaps:
        return False, "Overlap detected: " + ", ".join(overlaps)
    return True, (
        f"Splits are disjoint: train={len(train)}, val={len(val)}, test={len(test)}"
    )


def _check_test_frozen(rows: List[Dict[str, str]]) -> Tuple[bool, str]:
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        return True, "No test rows (vacuously true)"
    not_frozen = [r for r in test_rows if r.get("policy_frozen", "").lower() not in ("true", "1")]
    if not_frozen:
        return False, (
            f"{len(not_frozen)}/{len(test_rows)} test episodes have policy_frozen != True"
        )
    return True, f"All {len(test_rows)} test episodes are frozen"


def _check_budgets_bind(rows: List[Dict[str, str]]) -> Tuple[bool, str]:
    violations = 0
    unbounded_methods = {"FullContext-Unbounded"}
    total_checked = 0
    for row in rows:
        method = row.get("method", "")
        if method in unbounded_methods:
            continue
        try:
            selected = int(row.get("selected_tokens", 0))
            budget = int(row.get("budget_tokens", 0))
        except ValueError:
            continue
        total_checked += 1
        # Allow a small tolerance for token estimation rounding
        if selected > budget + 50:
            violations += 1

    if violations > 0:
        return False, f"{violations}/{total_checked} budgeted episodes exceeded budget"
    return True, f"All {total_checked} budgeted episodes respect budget constraint"


def _check_pilot_db_untouched(pilot_db_path: str | Path) -> Tuple[bool, str]:
    import time as _time
    p = Path(pilot_db_path)
    if not p.exists():
        return True, f"Pilot DB not found at {p} (may be expected)"
    mtime = p.stat().st_mtime
    now = _time.time()
    age_seconds = now - mtime
    if age_seconds < 30:
        return False, (
            f"Pilot DB modified {age_seconds:.0f}s ago — research run may have touched it"
        )
    return True, (
        f"Pilot DB last modified {age_seconds/3600:.1f}h ago — appears untouched"
    )


def _check_label_leakage(dataset_jsonl: Path) -> Tuple[bool, str]:
    """
    Check that hidden benchmark field strings do not appear in CIU content
    of the visible (exported) dataset. We check the full dataset and look
    for any CIU whose content contains literal hidden field names.
    """
    violations: List[str] = []
    with dataset_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            for ciu in task.get("cius", []):
                content = ciu.get("content", "").lower()
                for label in HIDDEN_FIELD_LABELS:
                    if label in content:
                        violations.append(
                            f"Task {task.get('task_id','?')} CIU {ciu.get('id','?')}: "
                            f"found hidden label '{label}' in content"
                        )

    if violations:
        return False, f"{len(violations)} leakage instance(s): {violations[:3]}"
    return True, "No hidden label strings found in CIU content"
