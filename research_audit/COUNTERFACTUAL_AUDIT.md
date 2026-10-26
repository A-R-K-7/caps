# Counterfactual Audit

Counterfactual replay is implemented but not comprehensively persisted. Episode rows store only the resulting `voi_scores` JSON dictionary, not individual counterfactual executions, receiver outputs, or counterfactual success values.

Selection strategy implementation: if selected CIU count is less than or equal to `max_counterfactuals`, test all selected CIUs; otherwise use random sampling. Config may say `uncertainty`, but `_select_for_testing` falls back to random for non-random strategies.

Actual number of counterfactual executions cannot be reconstructed exactly from DB rows without LLM call logs. Counterfactual raw table: NOT AVAILABLE.
