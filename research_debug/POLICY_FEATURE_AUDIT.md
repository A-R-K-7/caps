# Policy Feature Audit
- **Dimension**: 9 features (3 budget features, 5 CIU scalar features, 1 semantic similarity feature).
- **Calculation**: Feature calculations use `_build_context` and work perfectly.
- **Storage Bug**: In `research_cda97b59`, `feature_vector_json` was always `[]` because `build_decision_rows` did not receive the state object.
- **Fix**: Passed `state` to `build_decision_rows` in `runner.py`, and added the necessary `_build_context` reconstruction logic to `instrumentation.py`. Features will now correctly record in CSVs.
