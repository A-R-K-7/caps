"""
CIU Selector — budget-constrained selection strategies.

Given scored CIUs, selects a subset that fits within the token budget.
Strategies: greedy_budget, top_k, threshold, random.
"""

from __future__ import annotations

import random as rng
from typing import Dict, List, Tuple

from caps.communication.ciu import CIU


def greedy_budget_select(
    cius: List[CIU],
    scores: Dict[str, float],
    token_budget: int,
) -> List[str]:
    """
    Select CIUs greedily by score, respecting token budget.
    Returns list of selected CIU IDs.

    NOTE: The score threshold has been intentionally removed. Heuristic policies
    produce scores in [0, 1] where 0 means 'do not select', but LinUCB (used by
    CAPS-NoVOI and CAPS-VOI) produces UCB scores that can legitimately be
    negative during exploration. Filtering on score <= 0 would cause CAPS to
    transmit nothing after training with negative-reward episodes, which is
    an implementation artifact rather than a learned behaviour.
    """
    sorted_cius = sorted(cius, key=lambda c: scores.get(c.id, 0.0), reverse=True)
    selected = []
    used_tokens = 0
    for ciu in sorted_cius:
        tokens = ciu.estimate_tokens()
        if used_tokens + tokens <= token_budget:
            selected.append(ciu.id)
            used_tokens += tokens
    selected.reverse()
    return selected


def top_k_select(
    cius: List[CIU],
    scores: Dict[str, float],
    k: int,
    token_budget: int,
) -> List[str]:
    """Select top-k CIUs by score, respecting token budget."""
    sorted_cius = sorted(cius, key=lambda c: scores.get(c.id, 0.0), reverse=True)
    selected = []
    used_tokens = 0
    for ciu in sorted_cius[:k]:
        tokens = ciu.estimate_tokens()
        if used_tokens + tokens <= token_budget:
            selected.append(ciu.id)
            used_tokens += tokens
    selected.reverse()
    return selected


def threshold_select(
    cius: List[CIU],
    scores: Dict[str, float],
    threshold: float,
    token_budget: int,
) -> List[str]:
    """Select all CIUs with score above threshold, respecting token budget."""
    above = [c for c in cius if scores.get(c.id, 0.0) >= threshold]
    above.sort(key=lambda c: scores.get(c.id, 0.0), reverse=True)
    selected = []
    used_tokens = 0
    for ciu in above:
        tokens = ciu.estimate_tokens()
        if used_tokens + tokens <= token_budget:
            selected.append(ciu.id)
            used_tokens += tokens
    selected.reverse()
    return selected


def random_select(
    cius: List[CIU],
    token_budget: int,
    seed: int = 42,
) -> List[str]:
    """Randomly select CIUs within token budget."""
    r = rng.Random(seed)
    shuffled = list(cius)
    r.shuffle(shuffled)
    selected = []
    used_tokens = 0
    for ciu in shuffled:
        tokens = ciu.estimate_tokens()
        if used_tokens + tokens <= token_budget:
            selected.append(ciu.id)
            used_tokens += tokens
    return selected


def select_cius(
    cius: List[CIU],
    scores: Dict[str, float],
    token_budget: int,
    strategy: str = "greedy_budget",
    **kwargs,
) -> List[str]:
    """
    Dispatch to the appropriate selection strategy.

    Args:
        cius: Available CIUs.
        scores: Score per CIU ID.
        token_budget: Maximum total tokens.
        strategy: One of greedy_budget, top_k, threshold, random.

    Returns:
        List of selected CIU IDs.
    """
    if strategy == "greedy_budget":
        return greedy_budget_select(cius, scores, token_budget)
    elif strategy == "top_k":
        k = kwargs.get("k", 5)
        return top_k_select(cius, scores, k, token_budget)
    elif strategy == "threshold":
        thresh = kwargs.get("threshold", 0.5)
        return threshold_select(cius, scores, thresh, token_budget)
    elif strategy == "random":
        seed = kwargs.get("seed", 42)
        return random_select(cius, token_budget, seed)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
