"""
Evaluation Metrics — primary and secondary metrics for CAPS.

Primary (§58):
    - Task Success Rate
    - Communication Token Cost
    - Token Reduction (vs baseline)
    - Success Preservation (vs baseline)
    - Clarification Rate
    - Retry Rate

Secondary (§59):
    - Information Precision
    - Information Recall
    - Minimal Sufficient Context Distance
    - Policy Entropy
    - Exploration Rate
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import numpy as np


def task_success_rate(successes: List[bool]) -> float:
    """Fraction of tasks that succeeded."""
    if not successes:
        return 0.0
    return sum(successes) / len(successes)


def mean_communication_tokens(token_counts: List[int]) -> float:
    """Average total communication tokens across episodes."""
    if not token_counts:
        return 0.0
    return sum(token_counts) / len(token_counts)


def token_reduction(caps_tokens: float, baseline_tokens: float) -> float:
    """Token reduction relative to baseline: 1 - caps/baseline."""
    if baseline_tokens <= 0:
        return 0.0
    return 1.0 - (caps_tokens / baseline_tokens)


def success_preservation(caps_success: float, baseline_success: float) -> float:
    """Difference in success rate: caps - baseline."""
    return caps_success - baseline_success


def clarification_rate(clarifications: List[bool]) -> float:
    """Fraction of tasks that required clarification."""
    if not clarifications:
        return 0.0
    return sum(clarifications) / len(clarifications)


def retry_rate(retries: List[int]) -> float:
    """Fraction of tasks that required at least one retry."""
    if not retries:
        return 0.0
    return sum(1 for r in retries if r > 0) / len(retries)


# --- Secondary Metrics ---

def information_precision(
    transmitted_ids: Set[str],
    useful_ids: Set[str],
) -> float:
    """
    Fraction of transmitted CIUs that were useful.
    Useful = positive counterfactual utility.
    """
    if not transmitted_ids:
        return 0.0
    return len(transmitted_ids & useful_ids) / len(transmitted_ids)


def information_recall(
    transmitted_ids: Set[str],
    necessary_ids: Set[str],
) -> float:
    """
    Fraction of necessary CIUs that were transmitted.
    Necessary = identified through ground truth or ablation.
    """
    if not necessary_ids:
        return 1.0  # vacuously true
    return len(transmitted_ids & necessary_ids) / len(necessary_ids)


def minimal_sufficient_context_distance(
    selected_tokens: int,
    msc_tokens: int,
) -> int:
    """
    Distance between selected context and minimal sufficient context.
    MSC = smallest known subset that preserves task success.
    """
    return selected_tokens - msc_tokens


def policy_entropy(selection_probs: List[float]) -> float:
    """
    Shannon entropy of CIU selection probabilities.
    H(A|s) = -Σ p(a_i|s) log p(a_i|s)

    Higher entropy = more uncertain/exploratory policy.
    """
    if not selection_probs:
        return 0.0
    entropy = 0.0
    for p in selection_probs:
        if 0 < p < 1:
            entropy -= p * math.log2(p)
    return entropy


def compute_all_metrics(
    successes: List[bool],
    token_counts: List[int],
    clarifications: List[bool],
    retries: List[int],
    baseline_tokens: Optional[float] = None,
    baseline_success: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all primary metrics as a dictionary."""
    sr = task_success_rate(successes)
    mt = mean_communication_tokens(token_counts)

    metrics = {
        "success_rate": sr,
        "mean_tokens": mt,
        "median_tokens": float(np.median(token_counts)) if token_counts else 0.0,
        "std_tokens": float(np.std(token_counts)) if token_counts else 0.0,
        "clarification_rate": clarification_rate(clarifications),
        "retry_rate": retry_rate(retries),
        "total_episodes": len(successes),
    }

    if baseline_tokens is not None:
        metrics["token_reduction"] = token_reduction(mt, baseline_tokens)
    if baseline_success is not None:
        metrics["success_preservation"] = success_preservation(sr, baseline_success)

    return metrics
