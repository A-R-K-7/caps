# Baseline Audit

FullContext: sends every CIU and ignores token budget. No learning, one receiver call per episode plus SWE-bench evaluator call where applicable.

Random: shuffles CIUs with `seed + episode`, selects within token budget. Scores logged are generated from `Random(seed).random()` per CIU, which resets for each CIU and does not match the shuffled selection probabilities.

Heuristic: fixed CIU type priorities, greedy budget selection.

LLMPlanner: sends CIU IDs/types/token counts to planner LLM and asks for JSON selected IDs, then enforces budget in original CIU order. On JSON parse failure, falls back to all CIUs then budget-enforces.

CAPS: per-CIU-type LinUCB with epsilon-greedy score randomization and greedy budget selection.

Fairness concern: FullContext ignores the budget while other methods are budget-constrained; LLMPlanner makes an extra LLM selection call; CAPS updates online during evaluation; baselines do not learn.
