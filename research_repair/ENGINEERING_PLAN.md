# CAPS Research Repair Engineering Plan

## 1. Existing Architecture

The current implementation is a Python package under `src/caps` with these verified components:

- CIU abstraction: `src/caps/communication/ciu.py::CIU`.
- Policy interface: `src/caps/communication/policy.py::CommunicationPolicy`.
- Selection utilities: `src/caps/communication/selector.py`.
- Baselines: `src/caps/communication/baselines.py`.
- CAPS LinUCB: `src/caps/bandits/linucb.py::LinUCBPolicy`.
- Planner/receiver/evaluator agents: `src/caps/agents`.
- Synthetic v1 and lightweight SWE-bench-style datasets: `src/caps/datasets`.
- Environments: `src/caps/environments`.
- Reward and metrics: `src/caps/evaluation`.
- SQLite ORM: `src/caps/storage/database.py`.
- CLI: `src/caps/cli.py`.

The previous audit found 32 experiment records and 2,874 episode rows. Those results are preserved as pilot/history data and are not final evidence for the main research claim.

## 2. Existing Scientific Problems

- Synthetic v1 has small contexts and non-binding budgets.
- Many methods select all CIUs, so communication compression is not meaningfully tested.
- Existing episode rows do not store policy decision features, UCB scores, policy updates, or policy matrices.
- `llm_calls` exists but previous experiments have zero rows.
- Episode `total_tokens` means selected CIU communication tokens, not total API tokens.
- No frozen train/validation/test evaluation is implemented.
- Existing SWE-bench-style evaluation is not official SWE-bench.
- VOI contribution is not isolated against a matched no-VOI condition.
- Failure reasons, retry accounting, and cache status are insufficiently persisted.

## 3. Files / Classes / Functions Requiring Modification or Addition

Modify:

- `src/caps/communication/ciu.py`: add optional research metadata fields while preserving old CIU compatibility.
- `src/caps/communication/baselines.py`: add `BudgetedFullContextPolicy` and `CAPSNoVOIPolicy` compatibility is handled in research runner by controlling update signals.
- `src/caps/bandits/linucb.py`: expose decision diagnostics and update diagnostics without changing the core algorithm.
- `src/caps/storage/database.py`: add backward-compatible research logging tables for policy decisions, policy updates, extended episodes, experiment manifests, and retry/call details.
- `src/caps/cli.py`: add research commands for synthetic_v2 generation, dry-run validation, pilot/final mock experiments, analysis, and reproduce.

Add:

- `src/caps/datasets/synthetic_v2.py`: larger communication-selection benchmark with hidden ground-truth labels.
- `src/caps/environments/synthetic_v2.py`: deterministic hidden-label evaluator.
- `src/caps/research/runner.py`: paired train/validation/frozen-test experiment runner with budgets and methods.
- `src/caps/research/instrumentation.py`: hashes, token accounting, CSV export, and policy logging helpers.
- `src/caps/research/analysis.py`: summary statistics, confidence intervals, paired tests where valid, figures, and tables.
- `src/caps/research/validation.py`: data quality and leakage validation.
- `scripts/reconstruct_policy.py`: reconstruct decisions, updates, and checkpoint comparisons.
- `scripts/validate_research_results.py`: validate completed research results.
- Tests under `tests/test_research_pipeline.py`.

## 4. New Benchmark Design

Create `synthetic_v2` with task families:

- debugging
- configuration
- data_transformation
- code_repair
- documentation_lookup
- multi_hop_reasoning
- contradiction_resolution

Each task contains CIUs with hidden benchmark-only fields:

- `source_agent`
- `information_role`
- `dependency_group`
- `relevance`
- `ground_truth_necessity`
- `ground_truth_sufficiency`
- metadata

The hidden fields must never enter prompts, policy features, receiver inputs, or LLM planner prompts. They are used only by the deterministic evaluator and offline selection-quality metrics.

Contexts must include necessary, useful, redundant, irrelevant, misleading, and conditional CIUs. Candidate token totals should be large enough for budgets of 75%, 50%, 30%, and 15% to bind.

## 5. New Instrumentation

Persist per-episode:

- split, method, seed, budget fraction, budget tokens, policy frozen flag
- available/selected CIU counts and tokens
- LLM input/output/total tokens where available
- communication compression, budget utilization, API calls
- success, reward, precision, recall, irrelevant/redundant/misleading rates
- termination reason and step

Persist per-decision:

- CIU ID/type, selected flag, budget, remaining budget
- feature vector
- estimated mean, uncertainty, exploration bonus, UCB score/predicted reward
- policy version

Persist per-update:

- action, reward, update type, VOI value
- A, b, theta for the affected arm
- context/feature vector

Persist LLM calls:

- provider/model, hashes, token usage, latency, status, retry/cached fields

For mock-mode research experiments, log deterministic mock calls so the logging path is tested even without API usage.

## 6. New Experimental Protocol

Use a paired design:

- Methods: `FullContext-Unbounded`, `FullContext-Budgeted`, `Random`, `Heuristic`, `LLMPlanner`, `CAPS-NoVOI`, `CAPS-VOI`.
- Budgets: 100%, 75%, 50%, 30%, 15%.
- Seeds: 1 through 5 for final mock campaign; smaller set for pilot.
- Split: persisted 60/20/20 train/validation/test.
- Train: CAPS variants update.
- Validation: hyperparameter assessment allowed, not final claim evidence.
- Test: frozen policy; no policy update.

The first implementation target is a deterministic/mock-mode pipeline so CI and reproducibility do not depend on API quota. API execution can then use the same logging and validation path.

## 7. New Baselines

- `FullContext-Unbounded`: sends all CIUs, reports compression relative to full context as 1.0.
- `FullContext-Budgeted`: sends CIUs in original order until the same budget as CAPS is exhausted.
- `Random`: deterministic seeded budgeted selection.
- `Heuristic`: static type/lexical relevance; no reward updates.
- `LLMPlanner`: in mock mode uses a deterministic planner proxy; in API mode uses the existing LLM planner prompt with identical visible CIUs and budget.
- `CAPS-NoVOI`: same LinUCB policy, but updates only with global reward.
- `CAPS-VOI`: same LinUCB policy, updates with benchmark-derived VOI in synthetic_v2 mock mode or configured evaluator VOI in API mode.

## 8. Statistical Analysis

Compute:

- mean, median, standard deviation
- bootstrap confidence intervals
- Wilson confidence intervals for success
- paired bootstrap differences for matched tasks
- Wilcoxon signed-rank where paired numeric data and sample size are appropriate
- McNemar tests where paired binary outcomes are complete
- effect sizes for token reduction, reward, precision, and recall

Do not claim significance when assumptions are not met.

## 9. Reproducibility Strategy

- Persist dataset and splits with hashes.
- Persist final config in `research_repair/FINAL_CONFIG.json`.
- Persist experiment manifests as JSON and CSV.
- Add dry-run validation.
- Add `python -m caps.cli research reproduce <experiment_id>` or equivalent CLI command.
- Add scripts for reconstruction and validation.
- Preserve raw CSV exports under `research/results`.

## 10. Expected Outputs

Create:

- `research_repair/ENGINEERING_PLAN.md`
- `research_repair/EXPERIMENT_PROTOCOL.md`
- `research_repair/PILOT_REPORT.md`
- `research_repair/RECEIVER_INFORMATION_AUDIT.md`
- `research_repair/POLICY_FEATURE_AUDIT.md`
- `research_repair/FINAL_EXPERIMENT_REPORT.md`
- `research_repair/CLAIM_EVIDENCE_MATRIX.md`
- `research_repair/FINAL_CONFIG.json`
- `research/results/*.csv`
- `research/figures/*`
- `research/tables/*`
- `research/checkpoints/*`
- `research/manifests/*`
- `paper_data/*`

## 11. Risks

- A deterministic synthetic environment may overstate real LLM behavior; label mock/deterministic findings accordingly.
- LinUCB per-CIU-type arms may learn type priors rather than task-specific semantic selection.
- If LLM planner is mocked, it is not evidence about real LLM planning quality.
- API quota may prevent live experiments; live results should remain `NOT_RUN` unless actually executed.
- Storing full matrices for many episodes can grow artifacts; use JSONL files when needed.

## 12. Validation Criteria

The pilot is valid only if:

- no ground-truth labels leak into visible CIUs/prompts,
- budgets bind at restricted budget fractions,
- selected tokens never exceed budget for budgeted methods,
- train/validation/test splits are disjoint,
- test episodes have `policy_frozen=true`,
- policy decisions and updates are logged,
- checkpoints contain independently inspectable A/b/theta state,
- raw CSV exports regenerate summary statistics,
- tests pass.

The full research campaign is complete only when the validation script exits successfully and the claim-evidence matrix remains conservative.
