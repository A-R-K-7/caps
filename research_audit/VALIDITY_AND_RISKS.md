# Validity and Risks

## Strongly supported claims

CAPS is implemented as a Python research prototype with CIU objects, budgeted CIU selection, baselines, and a per-CIU-type LinUCB policy.

The system can reduce selected communication tokens relative to FullContext in budgeted settings, but exact claims must use recalculated DB rows.

## Plausible but insufficiently demonstrated claims

CAPS learns useful communication strategies. The code updates LinUCB online, but persisted experiments do not include policy matrices or UCB score traces.

CAPS maintains task performance while reducing context. This depends on small, incomplete, LLM-judged or synthetic runs and needs controlled paired evaluation.

## Unsupported claims

Statistically significant superiority over baselines: NOT SUPPORTED by current artifacts.

Official SWE-bench performance: NOT SUPPORTED because the project uses LLM judge comparison to gold patch, not test execution.

Strict multi-key round-robin infrastructure: CLAIM NOT SUPPORTED BY IMPLEMENTATION; key switching is conditional, not round-robin.

## Claims requiring new experiments

Frozen-policy generalization, train/test separation, robust counterfactual VOI, and official benchmark evaluation all require new experiments.

## Major risks

Small or incomplete sample sizes; no explicit train/test split in the runner; online test-set adaptation; LLM judge leakage on SWE-bench because the evaluator sees the gold patch; token accounting excludes actual LLM prompt/completion tokens; clarification is single-status only, not an interaction loop; retry metrics conflate none of the API retry logic into episode records; caching state is not fully recorded.
