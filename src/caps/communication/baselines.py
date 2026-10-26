"""
Baseline communication policies.

These are fixed heuristic policies that do NOT learn from feedback.
They serve as experimental controls for comparing against CAPS.
"""

from __future__ import annotations

import random as rng
from typing import Any, Dict, List, Optional

import numpy as np

from caps.communication.ciu import CIU
from caps.communication.policy import CommunicationPolicy
from caps.communication.selector import greedy_budget_select, random_select
from caps.communication.state import Action, CommunicationState


class FullContextPolicy(CommunicationPolicy):
    """
    Baseline 1: Send every CIU (ignoring token budget).
    Measures maximum-information performance.
    """

    def select(self, state: CommunicationState) -> Action:
        ids = [c.id for c in state.available_cius]
        scores = {c.id: 1.0 for c in state.available_cius}
        return Action(selected_ciu_ids=ids, ciu_scores=scores)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass  # Does not learn


class RandomPolicy(CommunicationPolicy):
    """
    Baseline 2: Randomly select CIUs within token budget.
    Use multiple seeds for variance estimation.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._episode = 0

    def select(self, state: CommunicationState) -> Action:
        self._episode += 1
        selected = random_select(
            state.available_cius,
            state.token_budget,
            seed=self.seed + self._episode,
        )
        scores = {c.id: rng.Random(self.seed).random() for c in state.available_cius}
        return Action(selected_ciu_ids=selected, ciu_scores=scores)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass


class StaticHeuristicPolicy(CommunicationPolicy):
    """
    Baseline 3: Manually defined type-based priorities.
    task > stack_trace > source_code > tests > environment > docs > unrelated
    """

    TYPE_PRIORITIES = {
        "task_description": 1.0,
        "task": 1.0,
        "stack_trace": 0.9,
        "error_message": 0.85,
        "source_code": 0.8,
        "relevant_code": 0.8,
        "failing_test": 0.75,
        "tests": 0.7,
        "config_file": 0.65,
        "environment": 0.6,
        "dependency_versions": 0.55,
        "constraints": 0.5,
        "schema": 0.5,
        "sample_data": 0.45,
        "documentation": 0.4,
        "relevant_documentation": 0.4,
        "version": 0.3,
        "metadata": 0.2,
        "readme": 0.15,
        "irrelevant_code": 0.05,
        "irrelevant_documentation": 0.05,
        "irrelevant_logs": 0.05,
        "unrelated_files": 0.02,
        "license": 0.0,
    }
    DEFAULT_PRIORITY = 0.3

    def select(self, state: CommunicationState) -> Action:
        scores = {}
        for ciu in state.available_cius:
            scores[ciu.id] = self.TYPE_PRIORITIES.get(
                ciu.type, self.DEFAULT_PRIORITY
            )
        selected = greedy_budget_select(
            state.available_cius, scores, state.token_budget
        )
        return Action(selected_ciu_ids=selected, ciu_scores=scores)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass


class LLMPlannerPolicy(CommunicationPolicy):
    """
    Baseline 4: Ask the planner LLM to select CIUs directly.
    Represents the current heuristic LLM-agent paradigm.
    """

    def __init__(self, llm_provider, model: str = "llama-3.1-8b-instant"):
        self.llm = llm_provider
        self.model = model

    def select(self, state: CommunicationState) -> Action:
        import json

        ciu_list = "\n".join(
            f"- ID: {c.id} | Type: {c.type} | Tokens: {c.estimate_tokens()}"
            for c in state.available_cius
        )
        prompt = (
            f"You are selecting information to send to an agent that will solve a task.\n"
            f"Token budget: {state.token_budget}\n\n"
            f"Available information units:\n{ciu_list}\n\n"
            f"Select the most important CIUs. Respond with a JSON object:\n"
            f'{{"selected_ciu_ids": ["id1", "id2", ...]}}'
        )

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="Select the minimum set of information units needed.",
            model=self.model,
            temperature=0.0,
            max_tokens=800,
        )

        try:
            parsed = json.loads(response.text)
            selected = parsed["selected_ciu_ids"]
        except (json.JSONDecodeError, KeyError) as e:
            import logging
            log = logging.getLogger(__name__)
            log.error("fallback_triggered: True")
            log.error("fallback_reason: JSON decode error or missing key")
            log.error(f"original_error: {str(e)}")
            # Failure to plan means 0 CIUs selected, do NOT silently fall back to heuristic.
            selected = []

        # Enforce budget
        final = []
        used = 0
        for ciu in state.available_cius:
            if ciu.id in selected:
                tokens = ciu.estimate_tokens()
                if used + tokens <= state.token_budget:
                    final.append(ciu.id)
                    used += tokens

        scores = {c.id: (1.0 if c.id in final else 0.0) for c in state.available_cius}
        return Action(selected_ciu_ids=final, ciu_scores=scores, llm_usage=response.token_usage)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass  # LLM does not learn between episodes


class EmbeddingSimilarityPolicy(CommunicationPolicy):
    """
    Baseline: Score CIUs by cosine similarity to task embedding.
    Select highest-similarity CIUs within token budget.
    """

    def select(self, state: CommunicationState) -> Action:
        scores = {}
        if not state.task_embedding:
            # Fallback to equal scores
            scores = {c.id: 0.5 for c in state.available_cius}
        else:
            task_emb = np.array(state.task_embedding)
            task_norm = np.linalg.norm(task_emb)
            for ciu in state.available_cius:
                if ciu.embedding:
                    ciu_emb = np.array(ciu.embedding)
                    ciu_norm = np.linalg.norm(ciu_emb)
                    if task_norm > 0 and ciu_norm > 0:
                        scores[ciu.id] = float(
                            np.dot(task_emb, ciu_emb) / (task_norm * ciu_norm)
                        )
                    else:
                        scores[ciu.id] = 0.0
                else:
                    scores[ciu.id] = 0.0

        selected = greedy_budget_select(
            state.available_cius, scores, state.token_budget
        )
        return Action(selected_ciu_ids=selected, ciu_scores=scores)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass


class BudgetedFullContextPolicy(CommunicationPolicy):
    """
    Baseline: FullContext-Budgeted.

    Sends CIUs in their original order until the token budget is exhausted.
    This gives a fair budget-controlled comparison with CAPS — unlike
    FullContextPolicy (which ignores budget), this policy is subject to
    the same constraint but applies no learned or heuristic ranking.
    """

    def select(self, state: CommunicationState) -> Action:
        selected: list[str] = []
        used = 0
        scores: dict[str, float] = {}
        for i, ciu in enumerate(state.available_cius):
            # Uniform descending score based on original order
            scores[ciu.id] = 1.0 - (i / max(len(state.available_cius), 1))
            tokens = ciu.estimate_tokens()
            if used + tokens <= state.token_budget:
                selected.append(ciu.id)
                used += tokens
        return Action(selected_ciu_ids=selected, ciu_scores=scores)

    def update(self, state, action, reward, voi_scores=None, **kw):
        pass  # Does not learn


class CAPSNoVOIPolicy(CommunicationPolicy):
    """
    CAPS-NoVOI baseline.

    Wraps LinUCBPolicy but suppresses the VOI signal: updates only with
    the global task reward (no per-CIU VOI weighting). This isolates the
    contribution of VOI-guided updates from the LinUCB mechanism itself.
    """

    def __init__(self, alpha: float = 1.0, selection_strategy: str = "greedy_budget"):
        from caps.bandits.linucb import LinUCBPolicy
        self._inner = LinUCBPolicy(alpha=alpha, selection_strategy=selection_strategy)

    def select(self, state: CommunicationState) -> Action:
        return self._inner.select(state)

    def update(self, state, action, reward, voi_scores=None, **kw):
        # Force voi_scores=None to update only on global reward
        self._inner.update(state, action, reward, voi_scores=None)

    def get_state_dict(self) -> dict:
        return self._inner.get_state_dict()

    def load_state_dict(self, d: dict) -> None:
        self._inner.load_state_dict(d)
