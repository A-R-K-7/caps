import scipy.stats as stats
import pandas as pd
import numpy as np

class StatisticsEngine:
    """
    Computes statistical significance between CAPS and baseline policies.
    """
    def __init__(self, df: pd.DataFrame):
        # df should contain columns: task_id, seed, policy, reward, success, tokens, latency, clarifications, etc.
        self.df = df
        
    def _compute_ci(self, data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        if n < 2:
            return 0.0, 0.0
        m, se = np.mean(a), stats.sem(a)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m - h, m + h

    def aggregate_metrics(self) -> pd.DataFrame:
        agg = self.df.groupby('policy').agg(
            mean_success=('success', 'mean'),
            std_success=('success', 'std'),
            mean_reward=('reward', 'mean'),
            std_reward=('reward', 'std'),
            mean_tokens=('tokens', 'mean'),
            std_tokens=('tokens', 'std'),
            mean_clarifications=('clarifications', 'mean') if 'clarifications' in self.df else ('reward', 'mean')
        ).reset_index()
        
        # Compute 95% CIs
        ci_success_lower = []
        ci_success_upper = []
        ci_reward_lower = []
        ci_reward_upper = []
        
        for policy in agg['policy']:
            subset = self.df[self.df['policy'] == policy]
            succ_ci = self._compute_ci(subset['success'].values)
            rew_ci = self._compute_ci(subset['reward'].values)
            ci_success_lower.append(succ_ci[0])
            ci_success_upper.append(succ_ci[1])
            ci_reward_lower.append(rew_ci[0])
            ci_reward_upper.append(rew_ci[1])
            
        agg['ci_success_lower'] = ci_success_lower
        agg['ci_success_upper'] = ci_success_upper
        agg['ci_reward_lower'] = ci_reward_lower
        agg['ci_reward_upper'] = ci_reward_upper
            
        return agg
        
    def perform_wilcoxon(self, baseline_policy: str, caps_policy: str = "CAPS_MOLinUCB"):
        """
        Performs Wilcoxon signed-rank test on rewards between CAPS and a baseline.
        Requires paired observations (same task and seed).
        """
        baseline_data = self.df[self.df['policy'] == baseline_policy].sort_values(['seed', 'task_id'])
        caps_data = self.df[self.df['policy'] == caps_policy].sort_values(['seed', 'task_id'])
        
        # Ensure alignment
        if len(baseline_data) != len(caps_data) or len(baseline_data) == 0:
            return None
            
        # Handle zero-differences for Wilcoxon
        diff = baseline_data['reward'].values - caps_data['reward'].values
        if (diff == 0).all():
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}
            
        stat, p_value = stats.wilcoxon(baseline_data['reward'].values, caps_data['reward'].values)
        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
        
    def perform_paired_ttest(self, baseline_policy: str, caps_policy: str = "CAPS_MOLinUCB"):
        """
        Performs Paired t-test on rewards between CAPS and a baseline.
        """
        baseline_data = self.df[self.df['policy'] == baseline_policy].sort_values(['seed', 'task_id'])
        caps_data = self.df[self.df['policy'] == caps_policy].sort_values(['seed', 'task_id'])
        
        if len(baseline_data) != len(caps_data) or len(baseline_data) == 0:
            return None
            
        stat, p_value = stats.ttest_rel(baseline_data['reward'].values, caps_data['reward'].values)
        return {
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
