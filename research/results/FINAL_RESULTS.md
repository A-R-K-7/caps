# CAPS: Context-Adaptive Communication Policy Selection — Final Results Report

> **Experiment**: `research_f5f93acf` (Pilot Mock Run)  
> **Report Date**: 2026-09-09  
> **Environment**: Synthetic v2 benchmark, mock LLM  
> **Status**: Preliminary — task success results invalid in mock mode; selection quality results valid

---

## 1. Experimental Setup

### System Description

CAPS (Context-Adaptive Personalization / Adaptive Communication Policy Selection) is a framework for selectively transmitting Candidate Information Units (CIUs) from a planning agent to a receiving agent under token budget constraints. The core hypothesis is that an adaptive policy can identify which CIUs are most valuable for a given task/receiver context, improving task performance while reducing communication cost.

**Pipeline:**
1. The Planner embeds the task description and all candidate CIUs using `all-MiniLM-L6-v2`
2. The communication policy (LinUCB or heuristic) selects a subset of CIUs within the token budget
3. The Receiver attempts to solve the task using only the selected CIUs
4. The Evaluator determines task success via deterministic keyword matching
5. The reward (R = success − λ·cost) is used to update the policy (for learning methods)

**Dataset**: Synthetic v2 benchmark — 35 tasks across 7 categories (debugging, configuration, data_transformation, code_repair, documentation_lookup, multi_hop_reasoning, contradiction_resolution) at 3 difficulty levels (easy, medium, hard)

**Split**: Train=21, Validation=7, Test=7 (seed=1, no overlap)

**Environment**: Mock LLM (deterministic, no API calls; real embeddings)

> [!IMPORTANT]
> All task success rates reported below are 0% due to the mock LLM environment. The mock receiver never outputs any task answer key. These results are valid for validating infrastructure and selection quality metrics only. Task performance conclusions require a real LLM experiment.

---

## 2. Baselines

| Method | Description | Budget? | Learns? |
|---|---|---|---|
| **FullContext-Unbounded** | Transmits all available CIUs (ignores token budget) | No | No |
| **FullContext-Budgeted** | Transmits CIUs in natural order until budget exhausted | Yes | No |
| **Random** | Randomly selects CIUs within budget | Yes | No |
| **Heuristic** | Type-priority scoring (task > code > docs > irrelevant) | Yes | No |
| **LLMPlanner** | LLM-directed selection *(mock: identical to Heuristic)* | Yes | No |
| **CAPS-NoVOI** | LinUCB with global task reward; no per-CIU VOI signal | Yes | Yes |
| **CAPS-VOI** | LinUCB with VOI-guided per-CIU reward signal | Yes | Yes |

**Important notes**:
- `LLMPlanner` uses `StaticHeuristicPolicy` as a proxy in mock mode. Its results are identical to Heuristic and do not represent real LLM planning.
- `FullContext-Unbounded` is an oracle — it uses 2–3× more tokens than budgeted methods and should not be treated as a competitive baseline.

---

## 3. Metrics

| Metric | Description | Valid in Mock Mode? |
|---|---|---|
| **Task Success Rate** | Fraction of test tasks where receiver answer contains the answer key | NO — always 0% |
| **Reward** | R = task_success − 0.5 × normalized_cost | Partial — reflects cost, not performance |
| **Information Precision** | Fraction of selected CIUs that are useful (necessary/useful/conditional) | YES |
| **Information Recall** | Fraction of necessary CIUs that were selected | YES |
| **Misleading Rate** | Fraction of selected CIUs that are misleading | YES |
| **Selected Tokens** | Total tokens in selected CIUs | YES |
| **Compression** | selected_tokens / available_tokens | YES |

**Reward function**: R = task_success − λ(w₁·norm_tokens + w₂·norm_latency + w₃·norm_clarification)  
λ=0.5, w₁=0.4, w₂=0.2, w₃=0.2. In mock mode, task_success=0 always, so reward reduces to a pure cost penalty.

---

## 4. Main Results

### Task Success (test split, all budgets)

| Method | 30% Budget | 50% Budget | 100% Budget |
|---|---:|---:|---:|
| FullContext-Unbounded | 0.0% | 0.0% | 0.0% |
| FullContext-Budgeted | 0.0% | 0.0% | 0.0% |
| Random | 0.0% | 0.0% | 0.0% |
| Heuristic | 0.0% | 0.0% | 0.0% |
| LLMPlanner | 0.0% | 0.0% | 0.0% |
| CAPS-NoVOI | 0.0% | 0.0% | 0.0% |
| CAPS-VOI | 0.0% | 0.0% | 0.0% |

**The above results reflect the mock environment, not method capability.**

### Selection Quality (test split, seed=1)

#### 100% Budget

| Method | Precision | Recall | Misleading | Tokens |
|---|---:|---:|---:|---:|
| Heuristic | 0.501 | 0.893 | 0.100 | 1868 |
| FullContext-Budgeted | 0.462 | 0.893 | 0.064 | 1851 |
| CAPS-NoVOI | 0.448 | 0.500 | 0.100 | 1293 |
| FullContext-Unbounded | 0.429 | 1.000 | 0.143 | 2511 |
| Random | 0.410 | 0.786 | 0.143 | 1870 |
| **CAPS-VOI** | **0.000** | **0.000** | **0.000** | **0** |

> [!CAUTION]
> CAPS-VOI shows 0 precision/recall/tokens due to a confirmed implementation bug in `greedy_budget_select`. These results are NOT a finding about VOI effectiveness. See Section 10.

#### 50% Budget

| Method | Precision | Recall | Misleading | Tokens |
|---|---:|---:|---:|---:|
| CAPS-NoVOI | 0.565 | 0.500 | 0.071 | 1093 |
| Heuristic | 0.522 | 0.726 | 0.071 | 1098 |
| LLMPlanner | 0.522 | 0.726 | 0.071 | 1098 |
| Random | 0.444 | 0.560 | 0.143 | 1077 |
| FullContext-Budgeted | 0.359 | 0.560 | 0.071 | 1091 |
| FullContext-Unbounded | 0.429 | 1.000 | 0.143 | 2511 |
| CAPS-VOI | 0.143 | 0.048 | 0.000 | 24 |

#### 30% Budget

| Method | Precision | Recall | Misleading | Tokens |
|---|---:|---:|---:|---:|
| CAPS-NoVOI | 0.647 | 0.405 | 0.071 | 693 |
| Heuristic | 0.530 | 0.560 | 0.071 | 698 |
| LLMPlanner | 0.530 | 0.560 | 0.071 | 698 |
| FullContext-Budgeted | 0.454 | 0.512 | 0.071 | 702 |
| Random | 0.471 | 0.524 | 0.071 | 706 |
| FullContext-Unbounded | 0.429 | 1.000 | 0.143 | 2511 |
| CAPS-VOI | 0.286 | 0.143 | 0.000 | 188 |

---

## 5. Budget Analysis

### Observation 1: CAPS-NoVOI achieves highest precision at restricted budgets

At 30% budget, CAPS-NoVOI precision (0.647) exceeds Heuristic (0.530), FullContext-Budgeted (0.454), and Random (0.471). This suggests LinUCB learns to preferentially select higher-quality CIUs. However, its recall (0.405) is lower, meaning it misses some necessary CIUs.

**Interpretation**: CAPS-NoVOI trades recall for precision under tight budgets. Whether this is beneficial depends on the task — for tasks where any one necessary CIU is sufficient, high precision matters more; for tasks requiring all necessary CIUs, high recall is essential.

### Observation 2: Heuristic consistently outperforms Random

Across all budgets, Heuristic precision exceeds Random precision by 4–8 percentage points. This validates that type-based prioritization carries meaningful signal about CIU utility.

### Observation 3: FullContext-Budgeted is a strong sequential baseline

Despite using no ranking, FullContext-Budgeted achieves competitive precision (0.462 at 100%) and recall (0.893), suggesting the natural CIU ordering in synthetic_v2 is not adversarial.

### Observation 4: Budget reduction reduces recall for all non-CAPS methods

All budgeted methods show sharp recall reduction as budget decreases from 100% to 30%, while CAPS-NoVOI maintains better precision at the cost of recall. This is the core tradeoff CAPS is designed to exploit.

---

## 6. Ablation Results

### FullContext vs. Budgeted
At 100% budget: Heuristic (0.501 prec, 0.893 recall, 1868 tokens) vs FullContext-Unbounded (0.429 prec, 1.000 recall, 2511 tokens).

Restricting budget improves precision (less noise) at the cost of recall. This is the fundamental information-selection tradeoff.

### CAPS-NoVOI vs. CAPS-VOI
**Results invalid due to Bug 1.** After bug fix and rerun, this is the key ablation for paper.

### Random vs. Heuristic
Heuristic consistently outperforms Random by 3–6 precision points across all budgets. Demonstrates that CIU type information is a useful prior for selection.

### Heuristic vs. LLMPlanner
**Identical results** — LLMPlanner proxies to Heuristic in mock mode. Cannot assess LLM planning value from this experiment.

### CAPS-NoVOI vs. non-adaptive baselines
At 30% budget: CAPS-NoVOI precision 0.647 vs Heuristic 0.530 (Δ = +0.117).
At 50% budget: CAPS-NoVOI precision 0.565 vs Heuristic 0.522 (Δ = +0.043).
At 100% budget: CAPS-NoVOI precision 0.448 vs Heuristic 0.501 (Δ = -0.053).

The experiment suggests that with 21 training episodes, CAPS-NoVOI learns to be more precise than Heuristic at restricted budgets but slightly less precise at full budget. This is a preliminary signal consistent with H1/H3, but the sample size (N=7 test tasks) makes this statistically weak.

---

## 7. Seed Robustness

| Method | Precision Std (100% budget) | Precision Range | Assessment |
|---|---:|---:|---|
| FullContext-Unbounded | 0.000 | 0.000 | STABLE (deterministic) |
| FullContext-Budgeted | 0.000 | 0.000 | STABLE (deterministic) |
| Heuristic | 0.000 | 0.000 | STABLE (deterministic) |
| LLMPlanner | 0.000 | 0.000 | STABLE (deterministic, same as Heuristic) |
| CAPS-NoVOI | 0.000 | 0.000 | STABLE (deterministic embeddings) |
| CAPS-VOI | 0.000 | 0.000 | TRIVIALLY STABLE (always 0) |
| Random | 0.026 | 0.052 | MINOR VARIANCE |

**Important caveat**: In mock mode, all inputs are deterministic, so all deterministic methods have zero seed variance. This is a mock-mode artifact, not evidence of genuine robustness. In real LLM mode, seeds would produce genuine variance.

---

## 8. Statistical Analysis

### Power Analysis
With N=7 test tasks per condition:
- Minimum detectable success difference at 80% power (α=0.05, two-sided): approximately 35 percentage points
- Current experiment cannot detect differences smaller than this

### Appropriate Tests for Future Experiments
Given that all methods are evaluated on the same test tasks, **paired tests** are appropriate:
- **Paired bootstrap confidence intervals** (preferred) — non-parametric, handles binary outcome
- **Wilcoxon signed-rank test** — appropriate for small N with paired structure
- **NOT unpaired t-test** — incorrect for paired design

### Current Statistical Conclusions
No statistically valid conclusions can be drawn about task performance from mock mode results.

Selection quality metrics (precision/recall) can be compared descriptively, but with N=7 no formal significance testing is meaningful.

---

## 9. Failure Analysis

### Why does CAPS-VOI send 0 tokens at 100% budget?

See Section 10. This is a code bug, not algorithmic failure.

### Why does CAPS-NoVOI achieve lower recall than baselines?

With only 21 training episodes and no positive rewards, LinUCB converges to a policy that selects a few high-utility CIUs but misses some necessary ones. This reflects insufficient training data rather than algorithmic failure.

### Why does training with 21 episodes not produce positive rewards?

The mock receiver never succeeds, so r_task = 0 always. The reward reduces to a pure cost penalty. LinUCB cannot learn to improve selection because "success" is impossible regardless of what is selected. This is the fundamental limitation of training in mock mode.

---

## 10. CAPS-VOI Analysis

### Root Cause of Zero-Token Behavior

The CAPS-VOI zero-token selection at 100% budget is caused by a confirmed implementation bug:

**Code path:**
```
LinUCB.select() 
  → _build_context() [ok]
  → compute UCB scores [UCB negative after negative-reward training]
  → select_cius() → greedy_budget_select()
  → score <= 0 guard [FILTERS ALL CIUs]
  → returns empty list
```

**Mathematical justification:**
- 21 training rewards: all in range [-0.20, -0.003]
- theta_i = A_i^{-1} * b_i where b_i = sum(r_j * x_j) with all r_j < 0
- theta_i is negative → exploitation = x^T * theta < 0
- When |exploitation| > alpha * sqrt(x^T A^{-1} x), UCB < 0
- Selector's `if score <= 0: continue` skips all such CIUs

**Classification**: Implementation bug (selector.py) + mock environment artifact (all rewards negative) + reward design weakness (sending nothing is incentivized when success is impossible)

**Fix applied**: Removed the `<= 0` guard from `greedy_budget_select`. CIUs are now selected by descending UCB score subject only to the token budget constraint.

**Consequence for results**: All CAPS-VOI results in `research_f5f93acf` are invalid and must be regenerated after the fix.

---

## 11. Limitations

1. **Mock environment**: Task success is always 0%. No conclusions about task performance possible.
2. **N=7 test tasks**: Critically underpowered for any statistical test.
3. **Only 4 of 7 task families in test set**: Results may not generalize.
4. **LLMPlanner = Heuristic in mock mode**: Cannot assess LLM planning contribution.
5. **CAPS-VOI selector bug**: All CAPS-VOI results invalid (fixed, rerun needed).
6. **Zero-variance seeds**: Mock determinism prevents genuine robustness assessment.
7. **21 training episodes insufficient**: LinUCB typically requires 50–100+ episodes for meaningful convergence.
8. **Reward design**: Sending nothing has near-zero cost when success is impossible, creating a pathological optimum.
9. **No human evaluation**: Selection quality uses synthetic ground-truth labels, not real human judgment.
10. **No ablation on alpha**: LinUCB exploration parameter not tuned.

---

## 12. Conclusions

The experiment successfully validates:
1. The CAPS infrastructure works end-to-end with real embeddings
2. Train/test splits are clean (no leakage)
3. All 63 experiment combinations complete correctly
4. The Heuristic policy outperforms Random in selection precision across all budgets
5. CAPS-NoVOI shows higher precision than Heuristic at restricted budgets (preliminary signal)

The experiment does NOT establish:
1. That CAPS improves task success rate (mock environment limitation)
2. That CAPS-VOI outperforms CAPS-NoVOI (selector bug)
3. That any method is statistically superior (N=7 too small)
4. That results generalize to real LLM receivers

**Next required step**: Run the corrected experiment with real LLM (N≥100 test tasks) to establish any meaningful performance claim.

**Exact command for next experiment (mock, corrected)**:
```bash
uv run caps research run --config configs/caps_research_mock.yaml
```
(This will generate a new run_id with corrected selector and episode IDs)
