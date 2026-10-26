"""
Statistical Analysis Module.

Performs rigorous statistical testing on experiment results.
- Bootstrap Confidence Intervals
- Wilcoxon Signed-Rank tests
- Effect size (Cohen's d)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np

# Suppress warnings from scipy/numpy for small sample sizes in synthetic data
warnings.filterwarnings("ignore", category=UserWarning)


def bootstrap_ci(
    data: List[float],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate bootstrap confidence intervals for the mean."""
    if not data:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    data_array = np.array(data)
    n = len(data_array)
    
    if n == 1:
        return (data[0], data[0])

    # Generate bootstrap samples
    samples = rng.choice(data_array, size=(num_bootstrap, n), replace=True)
    means = np.mean(samples, axis=1)

    # Compute percentiles
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100
    upper_pct = (1.0 - alpha / 2.0) * 100

    lower = float(np.percentile(means, lower_pct))
    upper = float(np.percentile(means, upper_pct))

    return lower, upper


def wilcoxon_signed_rank(
    data_a: List[float],
    data_b: List[float],
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test on paired data.
    Returns (statistic, p-value).
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        import logging
        logging.getLogger(__name__).warning("scipy not installed. Skipping Wilcoxon test.")
        return 0.0, 1.0

    if len(data_a) != len(data_b) or len(data_a) < 5:
        return 0.0, 1.0  # Not enough data or mismatched

    # Handle perfect ties which cause scipy to raise an exception
    diffs = [a - b for a, b in zip(data_a, data_b)]
    if all(d == 0 for d in diffs):
        return 0.0, 1.0

    try:
        stat, p_val = wilcoxon(data_a, data_b)
        return float(stat), float(p_val)
    except Exception:
        return 0.0, 1.0


def cohens_d(data_a: List[float], data_b: List[float]) -> float:
    """Calculate Cohen's d effect size for two paired samples."""
    if not data_a or not data_b or len(data_a) != len(data_b):
        return 0.0

    a = np.array(data_a)
    b = np.array(data_b)
    
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    
    if std_diff == 0:
        return 0.0
        
    return float(mean_diff / std_diff)


def compute_statistics(
    caps_results: Dict[str, List[float]],
    baseline_results: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Compute full statistical comparison between CAPS and a baseline.
    Requires paired lists (e.g., token usage per episode).
    """
    stats = {}
    
    for metric in ["rewards", "token_counts", "successes"]:
        if metric in caps_results and metric in baseline_results:
            caps_data = caps_results[metric]
            base_data = baseline_results[metric]
            
            # Means
            stats[f"{metric}_caps_mean"] = float(np.mean(caps_data)) if caps_data else 0.0
            stats[f"{metric}_base_mean"] = float(np.mean(base_data)) if base_data else 0.0
            
            # CIs
            ci_lower, ci_upper = bootstrap_ci(caps_data)
            stats[f"{metric}_caps_ci"] = (ci_lower, ci_upper)
            
            # Significance & Effect Size
            stat, pval = wilcoxon_signed_rank(caps_data, base_data)
            stats[f"{metric}_wilcoxon_p"] = pval
            stats[f"{metric}_cohens_d"] = cohens_d(caps_data, base_data)
            
    return stats
