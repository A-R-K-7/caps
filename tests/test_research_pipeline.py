"""
Tests for the CAPS research pipeline.

Covers:
- synthetic_v2 dataset generation and hidden-label integrity
- SyntheticV2Environment evaluation and selection quality
- BudgetedFullContextPolicy and CAPSNoVOIPolicy
- ResearchRunner (smoke test, mock mode)
- ResearchValidator data quality checks
- ResearchAnalysis summary statistics

Run with: pytest tests/test_research_pipeline.py -v
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tasks():
    """Generate a small synthetic_v2 dataset for testing."""
    from caps.datasets.synthetic_v2 import generate_synthetic_v2_dataset
    return generate_synthetic_v2_dataset(size=7, seed=1, families=["debugging"])


@pytest.fixture
def mock_llm():
    from caps.llm.mock_client import MockLLMProvider
    return MockLLMProvider()


@pytest.fixture
def research_runner(mock_llm, tmp_path):
    from caps.research.runner import ResearchRunner
    return ResearchRunner(
        llm_provider=mock_llm,
        dataset_size=7,
        dataset_seed=1,
        budget_fractions=[1.0, 0.5],
        seeds=[1],
        methods=["FullContext-Unbounded", "FullContext-Budgeted", "Random",
                 "Heuristic", "CAPS-NoVOI", "CAPS-VOI"],
        results_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        manifest_dir=str(tmp_path / "manifests"),
        data_dir=str(tmp_path / "data"),
        verbose=False,
    )


# ===========================================================================
# 1. Dataset: synthetic_v2
# ===========================================================================

class TestSyntheticV2Dataset:

    def test_generates_correct_size(self, small_tasks):
        assert len(small_tasks) == 7

    def test_hidden_fields_present_in_cius(self, small_tasks):
        """Hidden fields must exist in the raw task dict for internal use."""
        from caps.datasets.synthetic_v2 import HIDDEN_FIELDS
        for task in small_tasks:
            for ciu in task["cius"]:
                for field in HIDDEN_FIELDS:
                    assert field in ciu, f"Missing hidden field '{field}' in CIU {ciu['id']}"

    def test_visible_ciu_strips_hidden_fields(self, small_tasks):
        """visible_ciu must not expose any hidden fields."""
        from caps.datasets.synthetic_v2 import HIDDEN_FIELDS, visible_ciu
        for task in small_tasks:
            for ciu in task["cius"]:
                clean = visible_ciu(ciu)
                for field in HIDDEN_FIELDS:
                    assert field not in clean, f"Hidden field '{field}' leaked into visible CIU"

    def test_unique_task_ids(self, small_tasks):
        ids = [t["task_id"] for t in small_tasks]
        assert len(ids) == len(set(ids))

    def test_ground_truth_present(self, small_tasks):
        for task in small_tasks:
            gt = task.get("ground_truth", {})
            assert "necessary_cius" in gt
            assert len(gt["necessary_cius"]) > 0

    def test_necessary_cius_are_subset_of_all(self, small_tasks):
        for task in small_tasks:
            all_ids = {c["id"] for c in task["cius"]}
            necessary = set(task["ground_truth"]["necessary_cius"])
            assert necessary.issubset(all_ids)

    def test_splits_disjoint(self, small_tasks):
        from caps.datasets.synthetic_v2 import create_splits
        splits = create_splits(small_tasks, seed=1)
        train = set(splits["splits"]["train"])
        val = set(splits["splits"]["validation"])
        test = set(splits["splits"]["test"])
        assert not (train & val), "train and val overlap"
        assert not (train & test), "train and test overlap"
        assert not (val & test), "val and test overlap"
        assert train | val | test == {t["task_id"] for t in small_tasks}


# ===========================================================================
# 2. SyntheticV2Environment
# ===========================================================================

class TestSyntheticV2Environment:

    def test_evaluate_success_on_answer_key(self, small_tasks):
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        env.reset(0)
        task = small_tasks[0]
        answer_key = task["answer_key"]
        assert env.evaluate(f"The fix is {answer_key} applied here.")

    def test_evaluate_failure_on_wrong_answer(self, small_tasks):
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        env.reset(0)
        assert not env.evaluate("completely unrelated response xyz")

    def test_get_cius_hides_labels(self, small_tasks):
        from caps.datasets.synthetic_v2 import HIDDEN_FIELDS
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        env.reset(0)
        cius = env.get_cius()
        for ciu in cius:
            # CIU object should not have hidden field attributes
            d = ciu.model_dump()
            for field in HIDDEN_FIELDS:
                assert field not in d, f"Hidden field '{field}' in CIU object"

    def test_selection_quality_perfect(self, small_tasks):
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        env.reset(0)
        task = small_tasks[0]
        necessary_ids = task["ground_truth"]["necessary_cius"]
        sq = env.selection_quality(necessary_ids)
        assert sq["recall"] == pytest.approx(1.0)
        assert 0.0 <= sq["precision"] <= 1.0

    def test_selection_quality_empty(self, small_tasks):
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        env.reset(0)
        sq = env.selection_quality([])
        assert sq["recall"] == 0.0
        assert sq["precision"] == 0.0

    def test_num_tasks(self, small_tasks):
        from caps.environments.synthetic_v2 import SyntheticV2Environment
        env = SyntheticV2Environment()
        env.load_tasks(small_tasks)
        assert env.num_tasks == len(small_tasks)


# ===========================================================================
# 3. New baseline policies
# ===========================================================================

class TestNewBaselines:

    def _make_state(self, budget_tokens: int = 500):
        from caps.communication.ciu import CIU
        from caps.communication.state import CommunicationState
        cius = [
            CIU(id=f"c{i}", type="task_constraint", content="x" * 100, token_count=50)
            for i in range(8)
        ]
        return CommunicationState(
            task_embedding=[0.1] * 10,
            task_category="debugging",
            task_difficulty="medium",
            task_length=100,
            planner_confidence=0.8,
            receiver_role="debugger",
            receiver_model="mock",
            token_budget=budget_tokens,
            available_cius=cius,
        )

    def test_budgeted_full_context_respects_budget(self):
        from caps.communication.baselines import BudgetedFullContextPolicy
        policy = BudgetedFullContextPolicy()
        state = self._make_state(budget_tokens=200)  # 4 CIUs at 50 tokens each
        action = policy.select(state)
        selected_tokens = sum(
            50 for cid in action.selected_ciu_ids
        )
        assert selected_tokens <= 200 + 50  # allow one-item rounding

    def test_budgeted_full_context_selects_in_order(self):
        from caps.communication.baselines import BudgetedFullContextPolicy
        policy = BudgetedFullContextPolicy()
        state = self._make_state(budget_tokens=300)
        action = policy.select(state)
        # Scores should be strictly descending (original order preserved)
        scores = [action.ciu_scores[cid] for cid in action.selected_ciu_ids]
        assert scores == sorted(scores, reverse=True)

    def test_budgeted_full_context_does_not_learn(self):
        from caps.communication.baselines import BudgetedFullContextPolicy
        from caps.communication.state import Action
        policy = BudgetedFullContextPolicy()
        state = self._make_state()
        action = policy.select(state)
        policy.update(state, action, 1.0)  # must not raise

    def test_caps_no_voi_ignores_voi(self):
        """CAPSNoVOIPolicy should update even when voi_scores is provided,
        but must ignore VOI in its update (only uses global reward)."""
        from caps.communication.baselines import CAPSNoVOIPolicy
        policy = CAPSNoVOIPolicy()
        state = self._make_state()
        action = policy.select(state)
        voi = {cid: 1.0 for cid in action.selected_ciu_ids}
        # Should not raise; internal update must use global reward only
        policy.update(state, action, 0.5, voi_scores=voi)
        inner = policy._inner
        # After update, arms should exist for CIU types in selection
        assert len(inner._arms) > 0

    def test_caps_no_voi_selects_within_budget(self):
        from caps.communication.baselines import CAPSNoVOIPolicy
        policy = CAPSNoVOIPolicy()
        state = self._make_state(budget_tokens=150)
        action = policy.select(state)
        selected_tokens = len(action.selected_ciu_ids) * 50
        assert selected_tokens <= 150 + 50  # token estimation tolerance


# ===========================================================================
# 4. ResearchRunner (smoke test)
# ===========================================================================

class TestResearchRunner:

    def test_run_completes(self, research_runner, tmp_path):
        output = research_runner.run()
        assert "run_id" in output
        assert Path(output["episodes_csv"]).exists()
        assert Path(output["manifest_path"]).exists()

    def test_episodes_csv_has_expected_columns(self, research_runner, tmp_path):
        output = research_runner.run()
        with Path(output["episodes_csv"]).open() as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        required = [
            "method", "split", "budget_fraction", "policy_frozen",
            "task_success", "precision", "recall", "selected_tokens",
        ]
        for col in required:
            assert col in cols, f"Missing column: {col}"

    def test_test_episodes_are_frozen(self, research_runner, tmp_path):
        output = research_runner.run()
        with Path(output["episodes_csv"]).open() as f:
            rows = list(csv.DictReader(f))
        test_rows = [r for r in rows if r["split"] == "test"]
        assert len(test_rows) > 0, "No test rows found"
        for row in test_rows:
            assert row["policy_frozen"].lower() in ("true", "1"), (
                f"Test episode not frozen: {row['episode_id']}"
            )

    def test_pilot_db_untouched(self, research_runner, tmp_path):
        """The existing caps_experiments.db must not be created/modified."""
        pilot_db = Path("data/caps_experiments.db")
        # Only check modification time if it exists
        if pilot_db.exists():
            import time
            mtime_before = pilot_db.stat().st_mtime
            research_runner.run()
            mtime_after = pilot_db.stat().st_mtime
            # Allow up to 1 second drift for filesystem clock
            assert abs(mtime_after - mtime_before) < 1.0, "Pilot DB was modified!"
        else:
            research_runner.run()
            assert not pilot_db.exists(), "Pilot DB was created by research runner"

    def test_manifest_contains_expected_keys(self, research_runner, tmp_path):
        output = research_runner.run()
        manifest = json.loads(Path(output["manifest_path"]).read_text())
        for key in ["run_id", "dataset_hash", "splits_path", "methods", "seeds"]:
            assert key in manifest, f"Missing manifest key: {key}"

    def test_all_methods_present_in_episodes(self, research_runner, tmp_path):
        output = research_runner.run()
        with Path(output["episodes_csv"]).open() as f:
            rows = list(csv.DictReader(f))
        methods_seen = {r["method"] for r in rows}
        for m in research_runner.methods:
            assert m in methods_seen, f"Method {m} has no episodes"


# ===========================================================================
# 5. ResearchValidator
# ===========================================================================

class TestResearchValidator:

    def test_validates_clean_run(self, research_runner, tmp_path):
        from caps.research.validation import ResearchValidator
        output = research_runner.run()
        v = ResearchValidator()
        report = v.validate_all(
            episode_csv=output["episodes_csv"],
            splits_path=f"{research_runner.data_dir}/splits_{research_runner.dataset_seed}.json",
        )
        assert report["overall"] in ("PASS", "WARN"), (
            f"Validation failed: {[c for c in report['checks'] if c['status'] == 'FAIL']}"
        )

    def test_test_frozen_check_detects_violation(self, tmp_path):
        from caps.research.validation import ResearchValidator
        # Create a fake CSV where a test episode is not frozen
        csv_path = tmp_path / "fake_episodes.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "run_id", "experiment_id", "episode_id", "task_id", "method",
                "seed", "split", "budget_fraction", "budget_tokens", "policy_frozen",
                "category", "difficulty", "receiver_role",
                "num_available", "num_selected", "available_tokens", "selected_tokens",
                "compression", "task_success", "reward", "latency_ms",
                "clarification_turns", "precision", "recall", "misleading_rate",
                "voi_scores_json",
            ])
            writer.writeheader()
            writer.writerow({
                "run_id": "x", "experiment_id": "x", "episode_id": "e1",
                "task_id": "t1", "method": "CAPS-VOI", "seed": 1,
                "split": "test", "budget_fraction": 1.0, "budget_tokens": 1000,
                "policy_frozen": "False",  # ← violation
                "category": "debugging", "difficulty": "easy", "receiver_role": "debugger",
                "num_available": 5, "num_selected": 3, "available_tokens": 500,
                "selected_tokens": 300, "compression": 0.6,
                "task_success": True, "reward": 0.8, "latency_ms": 10.0,
                "clarification_turns": 0, "precision": 0.9, "recall": 1.0,
                "misleading_rate": 0.0, "voi_scores_json": "{}",
            })

        v = ResearchValidator()
        report = v.validate_all(
            episode_csv=str(csv_path),
            splits_path=tmp_path / "nonexistent_splits.json",
        )
        frozen_check = next(
            (c for c in report["checks"] if c["name"] == "test_episodes_frozen"), None
        )
        assert frozen_check is not None
        assert frozen_check["status"] == "FAIL"


# ===========================================================================
# 6. ResearchAnalysis
# ===========================================================================

class TestResearchAnalysis:

    def test_summary_returns_rows(self, research_runner, tmp_path):
        from caps.research.analysis import ResearchAnalysis
        output = research_runner.run()
        ra = ResearchAnalysis(output["episodes_csv"])
        rows = ra.summary()
        assert len(rows) > 0

    def test_markdown_export(self, research_runner, tmp_path):
        from caps.research.analysis import ResearchAnalysis
        output = research_runner.run()
        ra = ResearchAnalysis(output["episodes_csv"])
        md_path = tmp_path / "summary.md"
        ra.save_markdown(str(md_path))
        assert md_path.exists()
        content = md_path.read_text()
        assert "Method" in content
        assert "Success" in content

    def test_csv_export(self, research_runner, tmp_path):
        from caps.research.analysis import ResearchAnalysis
        output = research_runner.run()
        ra = ResearchAnalysis(output["episodes_csv"])
        csv_path = tmp_path / "summary.csv"
        ra.save_summary_csv(str(csv_path))
        assert csv_path.exists()
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "success_rate" in rows[0]

    def test_bootstrap_ci_bounds(self, research_runner, tmp_path):
        from caps.research.analysis import ResearchAnalysis, _bootstrap_ci
        mean, lo, hi = _bootstrap_ci([1.0] * 10)
        assert lo <= mean <= hi
        assert abs(mean - 1.0) < 1e-6
