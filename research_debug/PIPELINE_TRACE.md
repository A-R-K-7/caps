# Pipeline Trace
Trace captured in `raw/example_episode_trace.json`.
1. Task: `synthetic_v2_debugging_easy_0000_s1` generated.
2. `get_cius()` strips all hidden labels correctly.
3. State built.
4. Policy selects all 7 CIUs.
5. Receiver gets prompt.
6. Mock LLM returns `""` answer.
7. Evaluator evaluates `""` vs `parser_guard` -> `False`.
8. 0% success logged.
