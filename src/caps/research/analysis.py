"""
Research Analysis — summary statistics, confidence intervals, and reporting.

Reads episode CSV files produced by ResearchLogger and generates:
- Summary table (method × budget × seed → test metrics)
- Bootstrap confidence intervals for success rate and mean tokens
- Markdown table for paper / README

Usage::

    from caps.research.analysis import ResearchAnalysis
    ra = ResearchAnalysis("research/results/research_abc123_episodes.csv")
    ra.print_summary()
    ra.save_markdown("research/results/summary.md")
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean, lower_ci, upper_ci) via non-parametric bootstrap."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = boot_means[max(0, int(alpha * n_boot))]
    hi = boot_means[min(n_boot - 1, int((1 - alpha) * n_boot))]
    return sum(values) / n, lo, hi


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ResearchAnalysis:
    """
    Loads episode CSV(s) and computes summary statistics.

    Parameters
    ----------
    episode_csv : str | Path
        Path to the ``*_episodes.csv`` produced by ResearchLogger.
    split : str
        Which split to report (``"test"`` for primary claims).
    """

    def __init__(
        self,
        episode_csv: str | Path,
        split: str = "test",
    ) -> None:
        self.episode_csv = Path(episode_csv)
        self.split = split
        self._rows: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.episode_csv.exists():
            raise FileNotFoundError(f"Episode CSV not found: {self.episode_csv}")
        with self.episode_csv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append(row)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(self) -> List[Dict[str, Any]]:
        """
        Return a list of summary dicts, one per (method, budget_fraction, seed).

        Filters to ``self.split`` rows only.
        """
        grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
        for row in self._rows:
            if row.get("split") != self.split:
                continue
            key = (
                row["method"],
                float(row["budget_fraction"]),
                int(row["seed"]),
            )
            grouped[key].append(row)

        results = []
        for (method, bfrac, seed), rows in sorted(grouped.items()):
            successes = [r["task_success"].lower() in ("true", "1", "yes") for r in rows]
            tokens = [int(r["selected_tokens"]) for r in rows]
            rewards = [float(r["reward"]) for r in rows]
            precisions = [float(r["precision"]) for r in rows]
            recalls = [float(r["recall"]) for r in rows]

            n = len(rows)
            k = sum(successes)
            sr_mean, sr_lo, sr_hi = _bootstrap_ci([float(s) for s in successes])
            tok_mean, tok_lo, tok_hi = _bootstrap_ci(tokens)
            w_lo, w_hi = _wilson_ci(k, n)

            results.append({
                "method": method,
                "budget_fraction": bfrac,
                "seed": seed,
                "n": n,
                "success_rate": sr_mean,
                "success_ci_lo": sr_lo,
                "success_ci_hi": sr_hi,
                "wilson_ci_lo": w_lo,
                "wilson_ci_hi": w_hi,
                "mean_tokens": tok_mean,
                "tokens_ci_lo": tok_lo,
                "tokens_ci_hi": tok_hi,
                "mean_reward": sum(rewards) / n if n else 0.0,
                "mean_precision": sum(precisions) / n if n else 0.0,
                "mean_recall": sum(recalls) / n if n else 0.0,
            })
        return results

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_summary(self, budget_fraction: Optional[float] = None) -> None:
        """Print a human-readable summary table to stdout."""
        rows = self.summary()
        if budget_fraction is not None:
            rows = [r for r in rows if abs(r["budget_fraction"] - budget_fraction) < 1e-6]

        print(f"\n{'='*100}")
        print(f"{'Method':<28} {'Budget':>7} {'Seed':>5} {'N':>4} "
              f"{'Success%':>10} {'95% CI':>14} "
              f"{'Tokens':>8} {'Precision':>10} {'Recall':>8} {'Reward':>8}")
        print(f"{'-'*100}")
        for r in rows:
            print(
                f"{r['method']:<28} {r['budget_fraction']:>7.0%} "
                f"{r['seed']:>5d} {r['n']:>4d} "
                f"{r['success_rate']:>9.1%}  "
                f"[{r['success_ci_lo']:.2f},{r['success_ci_hi']:.2f}]"
                f"{r['mean_tokens']:>9.0f} "
                f"{r['mean_precision']:>10.3f} "
                f"{r['mean_recall']:>8.3f} "
                f"{r['mean_reward']:>8.3f}"
            )
        print(f"{'='*100}\n")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def save_markdown(self, path: str | Path, budget_fraction: Optional[float] = None) -> None:
        """Write a Markdown table of summary statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.summary()
        if budget_fraction is not None:
            rows = [r for r in rows if abs(r["budget_fraction"] - budget_fraction) < 1e-6]

        lines = [
            f"# CAPS Research Results (split=`{self.split}`)\n",
            "| Method | Budget | Seed | N | Success% | 95% CI | "
            "Mean Tokens | Precision | Recall | Mean Reward |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            lines.append(
                f"| {r['method']} "
                f"| {r['budget_fraction']:.0%} "
                f"| {r['seed']} "
                f"| {r['n']} "
                f"| {r['success_rate']:.1%} "
                f"| [{r['success_ci_lo']:.2f}, {r['success_ci_hi']:.2f}] "
                f"| {r['mean_tokens']:.0f} "
                f"| {r['mean_precision']:.3f} "
                f"| {r['mean_recall']:.3f} "
                f"| {r['mean_reward']:.3f} |"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_summary_csv(self, path: str | Path) -> None:
        """Write summary statistics to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.summary()
        if not rows:
            return

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
