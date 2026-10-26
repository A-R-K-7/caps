# CAPS Research Audit Report

> **Audit Date**: 2026-09-09  
> **Auditor**: Independent code audit (automated + manual inspection)  
> **Run ID**: `research_f5f93acf`  
> **CSV**: `research/results/research_f5f93acf_episodes.csv`  
> **Status**: 🔴 NOT READY for paper claims — significant bugs found and corrected

---

## Executive Summary

The CAPS (`research_f5f93acf`) experiment has been fully audited. The experiment infrastructure is well-engineered, but **two critical bugs and one fundamental environment limitation** invalidate most of the current results:

1. **BUG (CRITICAL)**: `greedy_budget_select()` filters CIUs with UCB score ≤ 0. After 21 training episodes with all-negative rewards, LinUCB learns negative parameter estimates. This causes CAPS-VOI and CAPS-NoVOI to select **zero tokens** at test time — a code artifact, not a learned behaviour. **FIXED.**

2. **BUG (MODERATE)**: Episode IDs omit budget fraction, causing 735 duplicate IDs across budget conditions. Data integrity is unaffected (experiment_id is correct), but automated deduplication on episode_id would corrupt analysis. **FIXED.**

3. **ENVIRONMENT LIMITATION (FUNDAMENTAL)**: The mock LLM returns a fixed string `"Mock LLM response."` that never contains any task answer key. Therefore `task_success = False` for **all 441 test episodes** (0% success rate across all methods). No conclusion about task performance can be drawn from these results.

**All existing results in `research_f5f93acf` must be considered PRELIMINARY and unsuitable for presentation as performance evidence.**

A corrected rerun (`research_corrected`) is recommended after applying the selector fix.

---

## Research Question

### Primary Research Question
Does a context-adaptive communication policy (CAPS) that uses Value of Information (VOI) signals to selectively transmit Candidate Information Units (CIUs) outperform non-adaptive communication baselines under token budget constraints?

### Precise Hypothesis
**H1**: Adaptive communication (CAPS) achieves higher task success rate than non-adaptive baselines (Random, Heuristic, FullContext-Budgeted) at equivalent token budgets.  
**H2**: VOI-guided policy updates (CAPS-VOI) produce higher information precision and recall than non-VOI updates (CAPS-NoVOI).  
**H3**: There exists a token budget threshold below which adaptive selection preserves more task success than sequential/random selection.

### Experimental Variables
| Role | Variable |
|------|----------|
| Independent | Communication policy (method) |
| Independent | Token budget (fraction of max) |
| Dependent | Task success rate |
| Dependent | Information precision, recall, misleading rate |
| Dependent | Total reward (combined success + cost) |
| Control | Task set (same 7 test tasks for all methods) |
| Control | Dataset split (fixed, seed=1) |
| Confounder | Mock environment: success always False |
| Treatment | CAPS-VOI vs baselines |

### Causal Inference Validity
The experiment is **not capable of supporting causal conclusions** in mock mode because the dependent variable (task success) is identically zero for all conditions. No causal claim about CAPS improving task performance can be made from mock data.

---

## Hypotheses Status

| Hypothesis | Status | Reason |
|---|---|---|
| H1: CAPS > baselines on task success | UNTESTABLE | Success=0 for all methods in mock mode |
| H2: CAPS-VOI precision > CAPS-NoVOI | INVALIDATED | Selector bug causes CAPS-VOI to send 0 tokens |
| H3: Budget threshold exists | PARTIALLY OBSERVABLE | Heuristic precision at 30% (0.53) vs 100% (0.50) shows small signal |

---

## Experimental Design

| Parameter | Value |
|---|---|
| Dataset | synthetic_v2, N=35 tasks |
| Split | 60/20/20 (train=21, val=7, test=7) |
| Methods | 7 |
| Budgets | 3 (100%, 50%, 30%) |
| Seeds | 3 (1, 2, 3) |
| LLM | Mock (deterministic, no real LLM calls) |
| Embeddings | Real (all-MiniLM-L6-v2, local) |
| Total experiment combinations | 63 |
| Total expected rows | 2,205 |
| Actual rows | 2,205 |
| Completeness | 100% |

---

## Dataset Audit

### Dataset Properties
| Property | Value |
|---|---|
| Generator | synthetic_v2.py |
| Dataset size | 35 tasks |
| Dataset seed | 1 |
| Split seed | 1 |
| Dataset hash | 975448f14628ecb9... |
| Train tasks | 21 |
| Validation tasks | 7 |
| Test tasks | 7 |

### Test Task IDs
```
synthetic_v2_code_repair_easy_0009_s1
synthetic_v2_code_repair_hard_0032_s1
synthetic_v2_code_repair_medium_0031_s1
synthetic_v2_contradiction_resolution_medium_0019_s1
synthetic_v2_data_transformation_easy_0006_s1
synthetic_v2_documentation_lookup_medium_0034_s1
synthetic_v2_multi_hop_reasoning_hard_0017_s1
```

**Warning**: The test set contains only 7 tasks and only 4 out of 7 families. Categories not in test: debugging, configuration. The test set is too small and not balanced across all families.

### Overlap Checks
- Train ∩ Test: 0 (PASS)
- Validation ∩ Test: 0 (PASS)
- Old _s42 contamination rows: 0 (PASS)
- All methods evaluate on identical test task IDs (PASS)

---

## Baseline Audit

| Method | Policy | LLM? | Budget? | Fair? | Notes |
|---|---|---|---|---|---|
| FullContext-Unbounded | Send all CIUs | No | No | UNFAIR | No budget — oracle upper bound only |
| FullContext-Budgeted | Sequential order | No | Yes | FAIR | Good control baseline |
| Random | Random within budget | No | Yes | FAIR | Variance baseline |
| Heuristic | Type-priority greedy | No | Yes | FAIR | Reasonable non-adaptive baseline |
| LLMPlanner | (Proxy: Heuristic) | No | Yes | MISREPRESENTED | Identical to Heuristic in mock mode |
| CAPS-NoVOI | LinUCB, global reward | No | Yes | FAIR | Ablation of VOI contribution |
| CAPS-VOI | LinUCB + VOI signal | No | Yes | INVALID | Bug in selector causes 0 tokens |

### Critical Fairness Issues
1. FullContext-Unbounded sends 2-3x more tokens than budgeted methods — unfair comparison
2. LLMPlanner falls back to StaticHeuristicPolicy in mock mode — results are IDENTICAL to Heuristic
3. CAPS methods have access to embeddings that other baselines do not — by design but must be disclosed

---

## Data Integrity Audit

### CSV Schema
- Header columns: 28 (CORRECT after fix)
- Missing columns: None (PASS)
- Bad rows: 0 (PASS)
- Numeric errors: 0 (PASS)
- Out-of-range precision/recall/misleading_rate: 0 (PASS)

### Episode ID Duplication Bug
- 735 of 735 episode IDs are duplicated x3 (one per budget fraction)
- Root cause: episode_id format omitted budget fraction
- Impact: Data is correct if using experiment_id; deduplication on episode_id would corrupt analysis
- Fix: episode_id now includes budget fraction

### Completeness Matrix: ALL 63 COMBINATIONS COMPLETE (100%)

---

## Statistical Audit

### Sample Size
- Test tasks per condition: 7
- Min N for 80% power (20pp difference): ~75
- Min N for 80% power (10pp difference): ~250
- **CONCLUSION: Experiment is critically underpowered**

### Success Rate
```
All 441 test episodes: task_success = False
All methods: 0.0% success rate
```
Not a performance finding — consequence of mock environment.

### Valid Metrics Summary (test split, seed=1, 100% budget)

| Method | Precision | Recall | Tokens | Reward |
|---|---:|---:|---:|---:|
| Heuristic | 0.501 | 0.893 | 1868 | -0.147 |
| FullContext-Budgeted | 0.462 | 0.893 | 1851 | -0.146 |
| CAPS-NoVOI | 0.448 | 0.500 | 1293 | -0.102 |
| FullContext-Unbounded | 0.429 | 1.000 | 2511 | -0.149 |
| Random | 0.410 | 0.786 | 1870 | -0.147 |
| CAPS-VOI | 0.000 | 0.000 | 0 | -0.000 |

---

## CAPS-VOI Investigation: Root Cause Confirmed

### Chain of Causation
1. All 21 training rewards: negative (-0.20 to -0.003)
2. LinUCB update: b_i = sum(r_i * x_i) where all r_i < 0 → b negative
3. theta = A^{-1} * b → negative parameter vector
4. UCB_i = x^T * theta + alpha * sqrt(x^T A^{-1} x)
5. Exploitation is NEGATIVE; for many CIUs |exploitation| > exploration
6. UCB_i < 0 for most CIUs
7. greedy_budget_select line 30 (`if score <= 0: continue`) skips all such CIUs
8. Zero tokens selected

### Classification
- (B) Bug in selector.py
- (G) Mock environment artifact (all rewards negative)
- (C) Reward design failure (can reward sending nothing)

### Fix
Removed the `<= 0` score filter from greedy_budget_select. Selection is now purely budget-constrained.

---

## Bugs Found

### Bug 1: CRITICAL — Selector blocks LinUCB after negative rewards
- **Location**: `src/caps/communication/selector.py`, line 30
- **Impact**: CAPS-VOI selects 0 tokens at 100% budget; CAPS-NoVOI partially affected
- **Fix**: Removed `<= 0` score guard — FIXED in selector.py
- **Rerun needed**: YES — new run with corrected selector

### Bug 2: MODERATE — Episode ID omits budget fraction
- **Location**: `src/caps/research/runner.py`, line 487
- **Impact**: 735 duplicate episode IDs (data correct, IDs misleading)
- **Fix**: Episode ID now includes budget fraction — FIXED in runner.py
- **Rerun needed**: YES — for clean IDs

### Bug 3: HIGH (historical, fixed) — CSV Header missing 2 columns
- **Location**: `src/caps/research/instrumentation.py`
- **Impact**: precision/recall/reward mapped to wrong columns
- **Fix**: Header rewritten; _repair_header() added — FIXED
- **Rerun needed**: Previous fix sufficient; CSV is correct now

### Bug 4: HIGH (historical, fixed) — MockLLMProvider.default_model missing
- **Location**: `src/caps/llm/mock_client.py`
- **Impact**: Blocked all non-FullContext methods from running
- **Fix**: Added @property default_model — FIXED

---

## Valid vs. Invalid Results

### Valid
- Dataset pipeline and split integrity
- Experiment completeness (100%)
- Selection quality metrics (precision, recall, misleading_rate) for non-CAPS-VOI methods
- Token accounting for all methods
- Infrastructure reproducibility

### Invalid
- All task success rates (always 0% in mock mode)
- CAPS-VOI precision/recall (selector bug causes 0-token selection)
- Any claim that CAPS improves task performance

---

## Recommended Next Experiments

Priority order:

1. **Corrected mock rerun** (N=35) — verify bug fixes, ensure CAPS-VOI/NoVOI behave correctly [~1 hour]
2. **Real LLM experiment** (N>=100 test tasks) — minimum for any task success claim [requires API]
3. **Extended dataset** (N>=210 tasks) — sufficient power for comparison [requires API]
4. **Budget sweep** (5 fractions) — for proper efficiency curves
5. **Human evaluation** (100-task sample) — for publication

---

## Reproducibility

### Exact commands
```bash
# Clean full run (recommended)
uv run caps research run --config configs/caps_research_mock.yaml

# Resume existing run
uv run caps research run --config configs/caps_research_mock.yaml --resume research_f5f93acf

# Generate report
uv run caps research report research/results/research_f5f93acf_episodes.csv --output research/results/summary.md
```

### Fixed files (all fixes already applied)
- src/caps/communication/selector.py — greedy selector bug
- src/caps/research/runner.py — episode ID uniqueness
- src/caps/research/instrumentation.py — header repair
- src/caps/llm/mock_client.py — default_model property
