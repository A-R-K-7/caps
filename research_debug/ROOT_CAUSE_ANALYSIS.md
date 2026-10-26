# Root Cause Analysis
The 0% success rate across all methods (including FullContext-Unbounded) was caused by the `MockLLMProvider` combined with the strict evaluator.

## Mechanism
1. The `ReceiverAgent` expects a structured JSON output (schema).
2. `MockLLMProvider` generated this mock JSON using default Pydantic values, meaning `answer` was set to `""`.
3. The synthetic_v2 deterministic evaluator expects a specific keyword (e.g. `parser_guard`).
4. `""` != `parser_guard`. Therefore, all tasks failed.

## Additional Issues Found
- `LLMPlannerPolicy` was silently catching JSON parse errors (from the mock provider) and falling back to a `FullContext-Budgeted` style heuristic without logging. Fixed to log error and select 0 CIUs on failure.
- `ResearchLogger.build_decision_rows` lacked the `state` parameter needed to compute LinUCB feature vectors, resulting in `feature_vector_json` being recorded as `[]`. Fixed by passing `state` and using `_build_context`.
