# Metrics Audit

Implemented primary metric functions: success rate, mean/median/std communication tokens, token reduction, success preservation, clarification rate, retry rate.

Implemented secondary metric functions: information precision, information recall, minimal sufficient context distance, policy entropy. These are functions only; no evidence was found that they are computed/stored during the main experiment runner.

Latency is measured as wall-clock elapsed time around planner state building, policy selection, receiver execution, and environment evaluation. For SWE-bench, this includes LLM judge evaluation. It is not separated into planner/receiver/evaluator/API latency in episode rows.

Token accounting in episode rows: selected CIU communication tokens only, based on CIU `token_count` / `estimate_tokens`. It is not total LLM prompt+completion usage. `input_tokens` and `output_tokens` columns exist but the runner does not populate them in `add_episode`.

Clarification is implemented as a receiver status string `needs_clarification`; there is no multi-turn clarification loop. Retry count is always accumulated as 0 in the runner, although API retry logic exists inside the provider.
