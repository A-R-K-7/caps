import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from caps.experiments.statistics import StatisticsEngine
import os

class ReportGenerator:
    def __init__(self, df: pd.DataFrame, output_dir: str = "docs/experiments"):
        self.df = df
        self.output_dir = output_dir
        self.stats = StatisticsEngine(df)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _save_plot(self, name: str):
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{name}.png")
        plt.savefig(f"{self.output_dir}/{name}.pdf")
        plt.savefig(f"{self.output_dir}/{name}.svg")
        plt.close()

    def generate_plots(self):
        # 1. Reward Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="policy", y="reward")
        plt.xticks(rotation=45, ha='right')
        plt.title("Reward Distribution by Policy")
        self._save_plot("reward_boxplot")
        
        # 2. Token Usage Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x="policy", y="tokens")
        plt.xticks(rotation=45, ha='right')
        plt.title("Token Usage by Policy")
        self._save_plot("tokens_boxplot")
        
        # 3. Success vs Episode (Learning Curve) - approximate by grouping
        if 'task_id' in self.df.columns:
            # Assuming task_id implies episode index roughly if sorted
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=self.df, x="task_id", y="reward", hue="policy")
            plt.title("Reward vs Episode")
            plt.xticks(rotation=90)
            self._save_plot("learning_curve_reward")

        # 4. Clarification Comparison
        if 'clarifications' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=self.df, x="policy", y="clarifications")
            plt.xticks(rotation=45, ha='right')
            plt.title("Clarifications by Policy")
            self._save_plot("clarifications_comparison")

        # 5. VOI Distribution
        # Requires voi_scores extracted from df.
        # 6. Oracle Gap
        if 'oracle_token_gap' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x="policy", y="oracle_token_gap")
            plt.xticks(rotation=45, ha='right')
            plt.title("Oracle Token Gap")
            self._save_plot("oracle_gap")

    def generate_markdown(self):
        agg_df = self.stats.aggregate_metrics()
        
        md = "# CAPS Evaluation Report\n\n"
        md += "## Executive Summary\n"
        md += "This report summarizes the experimental evaluation of the Communication-Adaptive Policy System (CAPS). "
        md += "The primary research goal is to determine if adaptive communication policies improve efficiency while maintaining task success.\n\n"
        
        md += "## Experimental Setup\n"
        md += "All experiments froze non-policy variables including model (Gemini), temperature (0.0), and random seeds. "
        md += "The evaluation spans 100 independent tasks evaluated over 5 random seeds.\n\n"
        
        md += "## Aggregate Metrics\n\n"
        md += agg_df.to_markdown(index=False) + "\n\n"
        
        md += "## Statistical Significance (vs CAPS_MOLinUCB)\n\n"
        md += "| Baseline | Paired t-test p-value | Wilcoxon p-value | Significant (p < 0.05) |\n"
        md += "|---|---|---|---|\n"
        
        caps_policy = "CAPS_MOLinUCB"
        for policy in agg_df['policy'].unique():
            if policy != caps_policy:
                wil = self.stats.perform_wilcoxon(policy, caps_policy)
                tt = self.stats.perform_paired_ttest(policy, caps_policy)
                if wil and tt:
                    sig = "Yes" if wil['significant'] or tt['significant'] else "No"
                    md += f"| {policy} | {tt['p_value']:.4f} | {wil['p_value']:.4f} | {sig} |\n"
                    
        md += "\n## Oracle Comparison\n"
        md += "Oracle metrics demonstrate the minimal sufficient communication context required for task success.\n"
        md += "![Oracle Gap](oracle_gap.png)\n\n"
        
        md += "## Figures\n"
        md += "### Reward Distribution\n![Reward](reward_boxplot.png)\n"
        md += "### Token Usage\n![Tokens](tokens_boxplot.png)\n"
        md += "### Learning Curve\n![Learning](learning_curve_reward.png)\n\n"
        
        md += "## Discussion & Threats to Validity\n"
        md += "Construct validity: Does the reward function accurately proxy efficiency?\n"
        md += "Internal validity: Are there hidden variables? LLM API changes, caching nuances.\n"
        md += "External validity: Does this generalize to other LLMs?\n\n"
        
        md += "## Conclusions\n"
        md += "See the statistical significance tables and aggregate metrics to form a conclusion on the CAPS communication efficiency vs baselines.\n"
                    
        with open(f"{self.output_dir}/report.md", "w") as f:
            f.write(md)
