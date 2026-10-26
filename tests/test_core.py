"""
Unit tests for core CAPS components.

Tests cover: CIU, LinUCB, selector, reward, budget, metrics, config, environment.
All tests run without API calls using MockLLMProvider.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# CIU Tests
# ---------------------------------------------------------------------------

from caps.communication.ciu import CIU


class TestCIU:
    def test_creation(self):
        ciu = CIU(type="task", content="Fix the bug", token_count=10)
        assert ciu.type == "task"
        assert ciu.content == "Fix the bug"
        assert ciu.token_count == 10
        assert ciu.id is not None

    def test_estimate_tokens(self):
        ciu = CIU(type="task", content="A" * 400, token_count=0)
        assert ciu.estimate_tokens() == 100  # 400 chars / 4

    def test_token_count_takes_precedence(self):
        ciu = CIU(type="task", content="A" * 400, token_count=50)
        assert ciu.estimate_tokens() == 50

    def test_serialization(self):
        ciu = CIU(type="task", content="test", token_count=5, embedding=[1.0, 2.0])
        data = ciu.model_dump()
        assert data["type"] == "task"
        assert data["embedding"] == [1.0, 2.0]

    def test_default_fields(self):
        ciu = CIU(type="task", content="test")
        assert ciu.priority == 0.5
        assert ciu.historical_utility == 0.0
        assert ciu.uncertainty == 1.0
        assert ciu.source == "benchmark"


# ---------------------------------------------------------------------------
# LinUCB Tests
# ---------------------------------------------------------------------------

from caps.bandits.linucb import LinUCBPolicy
from caps.communication.state import Action, CommunicationState


def make_test_state(
    num_cius: int = 3,
    token_budget: int = 250,
    embed_dim: int = 32,
) -> CommunicationState:
    """Helper to create a test state."""
    cius = []
    for i in range(num_cius):
        cius.append(CIU(
            id=f"ciu_{i}",
            type=f"type_{i % 3}",
            content=f"Content {i}" * 20,
            token_count=100,
            embedding=[float(i) / 10] * embed_dim,
        ))
    return CommunicationState(
        task_embedding=[0.1] * embed_dim,
        planner_confidence=0.8,
        token_budget=token_budget,
        available_cius=cius,
    )


class TestLinUCB:
    def test_initialization(self):
        policy = LinUCBPolicy(alpha=1.0)
        assert policy.alpha == 1.0
        assert policy.feature_dim is None

    def test_select_respects_budget(self):
        policy = LinUCBPolicy(alpha=1.0)
        state = make_test_state(num_cius=5, token_budget=250)
        action = policy.select(state)
        # Budget is 250, each CIU is 100 tokens, so max 2
        assert len(action.selected_ciu_ids) <= 2
        assert len(action.ciu_scores) == 5

    def test_select_empty(self):
        policy = LinUCBPolicy()
        state = CommunicationState(available_cius=[])
        action = policy.select(state)
        assert action.selected_ciu_ids == []

    def test_update_changes_parameters(self):
        policy = LinUCBPolicy(alpha=1.0)
        state = make_test_state()
        action = policy.select(state)

        # Get initial arm state
        assert policy.feature_dim is not None

        # Update with reward
        policy.update(state, action, reward=1.0)

        # Arms corresponding to selected CIUs should have been updated
        selected_types = {ciu.type for ciu in state.available_cius if ciu.id in action.selected_ciu_ids}
        for arm_name, arm in policy._arms.items():
            if arm_name in selected_types:
                # A should no longer be identity (updated with x @ x.T)
                assert not np.allclose(arm["A"], np.eye(policy.feature_dim))

    def test_update_with_voi_scores(self):
        policy = LinUCBPolicy(alpha=1.0)
        state = make_test_state()
        action = policy.select(state)
    
        voi = {cid: 1.0 for cid in action.selected_ciu_ids}
        policy.update(state, action, reward=0.5, voi_scores=voi)
    
        # b vectors should be non-zero for selected arms
        selected_types = {ciu.type for ciu in state.available_cius if ciu.id in action.selected_ciu_ids}
        for arm_name, arm in policy._arms.items():
            if arm_name in selected_types:
                assert np.any(arm["b"] != 0)

    def test_learning_changes_selection(self):
        """The most important test: policy behavior CHANGES over episodes."""
        policy = LinUCBPolicy(alpha=0.5)
        state = make_test_state(num_cius=4, token_budget=200)

        # Record initial selection
        initial_action = policy.select(state)
        initial_scores = dict(initial_action.ciu_scores)

        # Simulate learning: reward type_0 CIUs, punish type_1
        for _ in range(20):
            action = policy.select(state)
            voi = {}
            for ciu in state.available_cius:
                if ciu.id in action.selected_ciu_ids:
                    voi[ciu.id] = 1.0 if ciu.type == "type_0" else -0.5
            policy.update(state, action, reward=0.5, voi_scores=voi)

        # After learning, scores should have changed
        final_action = policy.select(state)
        final_scores = dict(final_action.ciu_scores)
        assert initial_scores != final_scores

    def test_checkpoint(self):
        policy = LinUCBPolicy(alpha=1.5)
        state = make_test_state()
        policy.select(state)
        policy.update(state, policy.select(state), 1.0)

        # Save
        state_dict = policy.get_state_dict()
        assert "arms" in state_dict
        assert state_dict["alpha"] == 1.5

        # Restore
        new_policy = LinUCBPolicy()
        new_policy.load_state_dict(state_dict)
        assert new_policy.alpha == 1.5
        assert len(new_policy._arms) == len(policy._arms)

    def test_numerical_stability(self):
        """Test with extreme values."""
        policy = LinUCBPolicy(alpha=1.0)
        state = make_test_state(num_cius=2, token_budget=1000)

        # Run many updates
        for i in range(100):
            action = policy.select(state)
            policy.update(state, action, reward=float(i % 2))

        # Should not have NaN or Inf
        final_action = policy.select(state)
        for score in final_action.ciu_scores.values():
            assert np.isfinite(score)


# ---------------------------------------------------------------------------
# Selector Tests
# ---------------------------------------------------------------------------

from caps.communication.selector import (
    greedy_budget_select,
    random_select,
    select_cius,
    threshold_select,
    top_k_select,
)


class TestSelector:
    def test_greedy_budget(self):
        cius = [
            CIU(id="a", type="t", content="x", token_count=100),
            CIU(id="b", type="t", content="x", token_count=100),
            CIU(id="c", type="t", content="x", token_count=100),
        ]
        scores = {"a": 0.9, "b": 0.7, "c": 0.8}
        selected = greedy_budget_select(cius, scores, token_budget=250)
        assert len(selected) == 2
        assert "a" in selected
        assert "c" in selected  # 0.9, 0.8 before 0.7

    def test_budget_zero(self):
        cius = [CIU(id="a", type="t", content="x", token_count=100)]
        selected = greedy_budget_select(cius, {"a": 1.0}, token_budget=0)
        assert selected == []

    def test_no_duplicates(self):
        cius = [
            CIU(id="a", type="t", content="x", token_count=50),
            CIU(id="b", type="t", content="x", token_count=50),
        ]
        selected = greedy_budget_select(cius, {"a": 1.0, "b": 0.5}, token_budget=1000)
        assert len(selected) == len(set(selected))

    def test_random_respects_budget(self):
        cius = [CIU(id=f"c{i}", type="t", content="x", token_count=100) for i in range(10)]
        selected = random_select(cius, token_budget=250, seed=42)
        total_tokens = sum(c.token_count for c in cius if c.id in selected)
        assert total_tokens <= 250

    def test_top_k(self):
        cius = [CIU(id=f"c{i}", type="t", content="x", token_count=50) for i in range(5)]
        scores = {f"c{i}": float(i) for i in range(5)}
        selected = top_k_select(cius, scores, k=2, token_budget=1000)
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# Reward Tests
# ---------------------------------------------------------------------------

from caps.evaluation.reward import RewardFunction


class TestReward:
    def test_perfect_success_no_cost(self):
        rf = RewardFunction(lambda_cost=0.5)
        state = make_test_state(token_budget=1000)
        action = Action(selected_ciu_ids=[], ciu_scores={})
        r = rf.calculate(task_success=True, state=state, action=action)
        assert r == 1.0  # success with no cost

    def test_failure(self):
        rf = RewardFunction(lambda_cost=0.5)
        state = make_test_state()
        action = Action(selected_ciu_ids=["ciu_0"], ciu_scores={})
        r = rf.calculate(task_success=False, state=state, action=action)
        assert r < 0  # failure with cost

    def test_cost_normalization(self):
        rf = RewardFunction(lambda_cost=1.0, w_tokens=1.0, w_latency=0, w_clarification=0, w_retries=0)
        state = make_test_state(token_budget=200)
        action = Action(selected_ciu_ids=["ciu_0"], ciu_scores={})  # 100 tokens
        r = rf.calculate(task_success=True, state=state, action=action)
        # R = 1.0 - 1.0 * (100/200) = 0.5
        assert abs(r - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Budget Manager Tests
# ---------------------------------------------------------------------------

from caps.llm.budget import BudgetManager


class TestBudgetManager:
    def test_initial_budget(self):
        bm = BudgetManager(max_daily_tokens=1000)
        remaining = bm.remaining()
        assert remaining["daily_tokens_remaining"] == 1000

    def test_record_usage(self):
        bm = BudgetManager(max_daily_tokens=1000)
        bm.record_usage(100, 50, 150)
        remaining = bm.remaining()
        assert remaining["daily_tokens_remaining"] == 850
        assert remaining["daily_tokens_used"] == 150

    def test_budget_exceeded(self):
        bm = BudgetManager(max_daily_tokens=100)
        bm.record_usage(50, 50, 100)
        allowed, reason = bm.can_make_request(estimated_tokens=50)
        assert not allowed
        assert "exceeded" in reason.lower() or "budget" in reason.lower()

    def test_per_task_budget(self):
        bm = BudgetManager(max_tokens_per_task=100)
        bm.record_usage(50, 50, 100, task_id="task_1")
        allowed, _ = bm.can_make_request(50, task_id="task_1")
        assert not allowed

    def test_reset(self):
        bm = BudgetManager(max_daily_tokens=1000)
        bm.record_usage(500, 500, 1000)
        bm.reset_daily()
        assert bm.remaining()["daily_tokens_remaining"] == 1000


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

from caps.evaluation.metrics import (
    clarification_rate,
    information_precision,
    information_recall,
    task_success_rate,
    token_reduction,
)


class TestMetrics:
    def test_success_rate(self):
        assert task_success_rate([True, True, False]) == pytest.approx(2/3)
        assert task_success_rate([]) == 0.0

    def test_token_reduction(self):
        assert token_reduction(80, 100) == pytest.approx(0.2)
        assert token_reduction(100, 100) == pytest.approx(0.0)
        assert token_reduction(50, 0) == 0.0

    def test_information_precision(self):
        assert information_precision({"a", "b", "c"}, {"a", "b"}) == pytest.approx(2/3)
        assert information_precision(set(), {"a"}) == 0.0

    def test_information_recall(self):
        assert information_recall({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2/3)
        assert information_recall(set(), set()) == 1.0

    def test_clarification_rate(self):
        assert clarification_rate([True, False, True]) == pytest.approx(2/3)


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

from caps.config import ExperimentConfig, load_config


class TestConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.seed == 42
        assert cfg.models.planner == "llama-3.1-8b-instant"
        assert cfg.policy.algorithm == "linucb"

    def test_yaml_loading(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_exp\nseed: 123\n")
            f.flush()
            cfg = load_config(f.name)
            assert cfg.name == "test_exp"
            assert cfg.seed == 123
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Synthetic Environment Tests
# ---------------------------------------------------------------------------

from caps.environments.synthetic import SyntheticEnvironment
from caps.datasets.generator import generate_dataset


class TestSyntheticEnvironment:
    def test_load_and_evaluate(self):
        tasks = generate_dataset(size=5, seed=42)
        env = SyntheticEnvironment()
        env.load_tasks(tasks)
        assert env.num_tasks == 5

    def test_deterministic_evaluation(self):
        tasks = generate_dataset(families=["debugging"], size=5, seed=42)
        env = SyntheticEnvironment()
        env.load_tasks(tasks)

        env.reset(0)
        keyword = tasks[0].get("answer_keyword", "")
        if keyword:
            assert env.evaluate(f"The fix is to {keyword} before dividing")
            assert not env.evaluate("I have no idea what the problem is")

    def test_ground_truth_extraction(self):
        tasks = generate_dataset(size=3, seed=42)
        env = SyntheticEnvironment()
        env.load_tasks(tasks)
        env.reset(0)

        gt = env.ground_truth()
        assert "necessary_cius" in gt
        necessary = env.get_necessary_ciu_ids()
        assert len(necessary) > 0


# ---------------------------------------------------------------------------
# Dataset Generator Tests
# ---------------------------------------------------------------------------

class TestDatasetGenerator:
    def test_generation(self):
        tasks = generate_dataset(size=20, seed=42)
        assert len(tasks) == 20

    def test_deterministic(self):
        t1 = generate_dataset(size=10, seed=42)
        t2 = generate_dataset(size=10, seed=42)
        assert [t["task_id"] for t in t1] == [t["task_id"] for t in t2]

    def test_ground_truth_present(self):
        tasks = generate_dataset(size=5, seed=42)
        for task in tasks:
            assert "ground_truth" in task
            assert "necessary_cius" in task["ground_truth"]

    def test_cius_have_ids(self):
        tasks = generate_dataset(size=5, seed=42)
        for task in tasks:
            for ciu in task["cius"]:
                assert "id" in ciu
                assert "type" in ciu
                assert "content" in ciu

    def test_save_load(self):
        tasks = generate_dataset(size=5, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
            f.flush()
            from caps.datasets.generator import load_dataset
            loaded = load_dataset(f.name)
            assert len(loaded) == 5
        os.unlink(f.name)


# ---------------------------------------------------------------------------
# Mock LLM Provider Tests
# ---------------------------------------------------------------------------

from caps.llm.mock_client import MockLLMProvider


class TestMockLLMProvider:
    def test_default_response(self):
        provider = MockLLMProvider()
        resp = provider.generate("Hello")
        assert resp.text == "Mock LLM response."
        assert resp.token_usage.total_tokens > 0

    def test_custom_response(self):
        provider = MockLLMProvider(responses={"Hello": "World"})
        resp = provider.generate("Hello")
        assert resp.text == "World"

    def test_schema_response(self):
        from pydantic import BaseModel
        class TestSchema(BaseModel):
            name: str = ""
            value: int = 0

        provider = MockLLMProvider()
        resp = provider.generate("Test", response_schema=TestSchema)
        parsed = json.loads(resp.text)
        assert "name" in parsed
        assert "value" in parsed

    def test_call_tracking(self):
        provider = MockLLMProvider()
        provider.generate("one")
        provider.generate("two")
        assert provider.num_calls == 2

    def test_list_models(self):
        provider = MockLLMProvider()
        assert "mock-model" in provider.list_models()


# ---------------------------------------------------------------------------
# Database Tests
# ---------------------------------------------------------------------------

from caps.storage.database import ExperimentDB


class TestDatabase:
    def test_create_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = ExperimentDB(f.name)
            db.create_experiment(
                experiment_id="test_001",
                name="Test Experiment",
                seed=42,
                policy_name="CAPS",
            )
            exp = db.get_experiment("test_001")
            assert exp is not None
            assert exp.name == "Test Experiment"
            db.engine.dispose()
        os.unlink(f.name)

    def test_episode_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = ExperimentDB(f.name)
            db.create_experiment(
                experiment_id="test_002",
                name="Test",
                seed=42,
                policy_name="CAPS",
            )
            db.add_episode(
                experiment_id="test_002",
                episode_id="ep_001",
                task_id="task_001",
                seed=42,
                task_success=True,
                total_tokens=500,
                reward=0.75,
                selected_ciu_ids=["a", "b"],
                available_ciu_ids=["a", "b", "c"],
            )
            episodes = db.get_episodes("test_002")
            assert len(episodes) == 1
            assert episodes[0].task_success is True
            db.engine.dispose()
        os.unlink(f.name)

    def test_metrics(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = ExperimentDB(f.name)
            db.create_experiment(
                experiment_id="test_003",
                name="Test",
                seed=42,
                policy_name="CAPS",
            )
            db.add_metric("test_003", "success_rate", 0.85)
            db.add_metric("test_003", "mean_tokens", 150.0)
            metrics = db.get_metrics("test_003")
            assert metrics["success_rate"] == 0.85
            assert metrics["mean_tokens"] == 150.0
            db.engine.dispose()
        os.unlink(f.name)
