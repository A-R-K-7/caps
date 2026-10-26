# Debugging Plan
1. Isolate the evaluator: Test if the synthetic_v2 evaluator accepts correct answers.
2. Trace an episode: Watch exactly what goes into the receiver and what comes out.
3. Audit the mock LLM: Check if the mock is forcing failures.
4. Audit LLMPlanner: Check for silent fallback to heuristic.
5. Audit Feature Vectors: Check if LinUCB features are correctly calculated and stored.
6. Audit Seeds: Verify random vs deterministic seeding behavior.
7. Run a tiny diagnostic benchmark to prove that success > 0% is possible.
