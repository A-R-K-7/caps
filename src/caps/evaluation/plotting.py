"""
Plotting Module.

Generates publication-ready plots for CAPS evaluation.
Uses matplotlib and seaborn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    # Configure publication-style settings
    plt.style.use("seaborn-v0_8-paper")
    sns.set_palette("colorblind")
    sns.set_context("paper", font_scale=1.5)
    
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib/seaborn/pandas not installed. Plotting disabled.")


def plot_learning_curve(
    results_df: "pd.DataFrame",
    output_path: str,
    metric: str = "reward",
    window: int = 10,
) -> None:
    """Plot the learning curve (metric over episodes) for each policy."""
    if not HAS_MATPLOTLIB:
        return
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for policy in results_df["policy"].unique():
        policy_data = results_df[results_df["policy"] == policy].copy()
        policy_data.sort_values("episode", inplace=True)
        
        # Calculate moving average
        smoothed = policy_data[metric].rolling(window=window, min_periods=1).mean()
        
        plt.plot(
            policy_data["episode"],
            smoothed,
            label=policy,
            linewidth=2,
        )
        
    plt.title(f"{metric.capitalize()} Learning Curve (MA={window})")
    plt.xlabel("Episode")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_token_vs_success(
    summary_df: "pd.DataFrame",
    output_path: str,
) -> None:
    """Scatter plot of Token Cost vs Success Rate."""
    if not HAS_MATPLOTLIB:
        return
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(
        data=summary_df,
        x="mean_tokens",
        y="success_rate",
        hue="policy",
        s=200,
        alpha=0.8,
    )
    
    # Add labels to points
    for idx, row in summary_df.iterrows():
        plt.text(
            row["mean_tokens"] * 1.05,
            row["success_rate"],
            row["policy"],
            fontsize=12,
        )
        
    plt.title("Token Cost vs Task Success Rate")
    plt.xlabel("Mean Tokens per Episode")
    plt.ylabel("Success Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    db_manager: Any,
    experiment_id_prefix: str,
    output_dir: str = "experiments/results/plots",
) -> None:
    """Generate all standard evaluation plots from the database."""
    if not HAS_MATPLOTLIB:
        return
        
    # Extract data from DB
    episodes = []
    for exp in db_manager.get_all_experiments():
        if exp.experiment_id.startswith(experiment_id_prefix):
            for ep_dict in db_manager.episodes_to_dicts(exp.experiment_id):
                ep_dict["policy"] = exp.policy_name
                # Parse episode ID to get sequential number
                try:
                    ep_dict["episode"] = int(ep_dict["episode_id"].split("ep")[-1])
                except ValueError:
                    ep_dict["episode"] = 0
                episodes.append(ep_dict)
                
    if not episodes:
        logger.warning(f"No data found for prefix {experiment_id_prefix}")
        return
        
    df = pd.DataFrame(episodes)
    
    # Generate plots
    plot_learning_curve(df, f"{output_dir}/{experiment_id_prefix}_learning_reward.png", metric="reward")
    plot_learning_curve(df, f"{output_dir}/{experiment_id_prefix}_learning_tokens.png", metric="total_tokens")
    
    # Summary
    summary = df.groupby("policy").agg({
        "task_success": "mean",
        "total_tokens": "mean"
    }).rename(columns={"task_success": "success_rate", "total_tokens": "mean_tokens"}).reset_index()
    
    plot_token_vs_success(summary, f"{output_dir}/{experiment_id_prefix}_tradeoff.png")
