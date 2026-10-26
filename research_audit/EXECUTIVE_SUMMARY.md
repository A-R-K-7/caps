# Executive Summary

Project: CAPS - Communication-Adaptive Policy System.

Research objective: learn/select adaptive communication context for multi-agent LLM collaboration instead of always transmitting all available context. The explicit hypothesis is RECONSTRUCTED FROM IMPLEMENTATION and plan.md: an agent can learn which Candidate Information Units (CIUs) to send to preserve task success while reducing communication cost.

Implementation status: runnable Python package with planner, receiver, evaluator, CIU schema, baselines, LinUCB policy, synthetic and lightweight SWE-bench dataset support, SQLite storage, and CLI. It is a research prototype, not a complete official benchmark harness.

Frameworks/libraries: Python, Pydantic, SQLAlchemy, NumPy, pandas/scipy/scikit-learn/matplotlib, sentence-transformers, Typer/Rich, Tenacity, Groq/OpenAI-compatible clients.

LLM providers: Groq provider and OpenRouter-compatible mode via `base_url`. Mock provider for tests. Gemini is proposed in `plan.md` but not implemented in current source.

Embedding model: configured default `all-MiniLM-L6-v2`; local sentence-transformers embeddings cached in SQLite.

Databases: data\experiments.db, data\experiments_archive.db, data\mock_experiments.db, caps_experiments.db, data\llm_cache.db, data\embedding_cache.db, llm_cache.db, mlflow.db, results.db.

Experiments found: 32 experiment records across available databases. Episode rows found: 2874.

## Current Experimental Status

| Database | Method | Experiment ID | Successful | Completed | API failures | Other failures | Correct success | Mean tokens | Mean latency ms | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data\experiments.db | CAPS | caps_ark_CAPS_42 | 182 | 200 | NOT RECORDED IN EPISODES | 18 | 0.91 | 118.46 | 7950.78 | 0.8519 |
| data\experiments.db | FullContext | caps_ark_FullContext_42 | 184 | 200 | NOT RECORDED IN EPISODES | 16 | 0.92 | 118.46 | 7559.54 | 0.8636 |
| data\experiments.db | Heuristic | caps_ark_Heuristic_42 | 188 | 200 | NOT RECORDED IN EPISODES | 12 | 0.94 | 118.46 | 7602.36 | 0.884 |
| data\experiments.db | LLMPlanner | caps_ark_LLMPlanner_42 | 182 | 200 | NOT RECORDED IN EPISODES | 18 | 0.91 | 83.42 | 16331.19 | 0.8371 |
| data\experiments.db | Random | caps_ark_Random_42 | 185 | 200 | NOT RECORDED IN EPISODES | 15 | 0.925 | 118.46 | 7647.52 | 0.8681 |
| data\experiments.db | CAPS | caps_experiment_CAPS_42 | 162 | 200 | NOT RECORDED IN EPISODES | 38 | 0.81 | 118.46 | 7211.45 | 0.7563 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_42 | 23 | 50 | NOT RECORDED IN EPISODES | 27 | 0.46 | 1644.96 | 68107.67 | 0.3006 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_43 | 13 | 50 | NOT RECORDED IN EPISODES | 37 | 0.26 | 1644.96 | 53004.35 | 0.1026 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_44 | 2 | 50 | NOT RECORDED IN EPISODES | 48 | 0.04 | 1644.96 | 23646.18 | -0.1188 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_45 | 0 | 9 | NOT RECORDED IN EPISODES | 9 | 0.0 | 1699.56 | 21622.82 | -0.1599 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_46 | 1 | 25 | NOT RECORDED IN EPISODES | 24 | 0.04 | 1647.28 | 22661.19 | -0.1186 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_47 | 2 | 50 | NOT RECORDED IN EPISODES | 48 | 0.04 | 596.64 | 15155.51 | -0.0628 |
| data\experiments.db | FullContext | caps_swe_bench_v1_FullContext_42 | 2 | 50 | NOT RECORDED IN EPISODES | 48 | 0.04 | 1644.96 | 12177.72 | -0.0775 |
| data\experiments.db | Heuristic | caps_swe_bench_v1_Heuristic_42 | 3 | 50 | NOT RECORDED IN EPISODES | 47 | 0.06 | 1644.96 | 23394.86 | -0.1002 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v1_LLMPlanner_42 | 25 | 50 | NOT RECORDED IN EPISODES | 25 | 0.5 | 1633.36 | 29427.54 | 0.3335 |
| data\experiments.db | Random | caps_swe_bench_v1_Random_42 | 39 | 50 | NOT RECORDED IN EPISODES | 11 | 0.78 | 1644.96 | 26945.15 | 0.6205 |
| data\experiments.db | CAPS | caps_swe_bench_v2_CAPS_42 | 6 | 50 | NOT RECORDED IN EPISODES | 44 | 0.12 | 1162.04 | 17440.11 | -0.011 |
| data\experiments.db | FullContext | caps_swe_bench_v2_FullContext_42 | 47 | 50 | NOT RECORDED IN EPISODES | 3 | 0.94 | 1644.96 | 28565.59 | 0.7764 |
| data\experiments.db | Heuristic | caps_swe_bench_v2_Heuristic_42 | 7 | 50 | NOT RECORDED IN EPISODES | 43 | 0.14 | 1644.96 | 23091.57 | -0.0211 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v2_LLMPlanner_42 | 8 | 50 | NOT RECORDED IN EPISODES | 42 | 0.16 | 1624.96 | 80701.02 | -0.006 |
| data\experiments.db | Random | caps_swe_bench_v2_Random_42 | 25 | 50 | NOT RECORDED IN EPISODES | 25 | 0.5 | 1644.96 | 22721.3 | 0.3451 |
| data\experiments.db | CAPS | caps_swe_bench_v3_CAPS_42 | 2 | 50 | NOT RECORDED IN EPISODES | 48 | 0.04 | 828.6 | 50901.75 | -0.0866 |
| data\experiments.db | FullContext | caps_swe_bench_v3_FullContext_42 | 1 | 50 | NOT RECORDED IN EPISODES | 49 | 0.02 | 1644.96 | 56701.21 | -0.1458 |
| data\experiments.db | Heuristic | caps_swe_bench_v3_Heuristic_42 | 1 | 50 | NOT RECORDED IN EPISODES | 49 | 0.02 | 1644.96 | 67964.01 | -0.1458 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v3_LLMPlanner_42 | 3 | 50 | NOT RECORDED IN EPISODES | 47 | 0.06 | 1642.96 | 68168.72 | -0.1057 |
| data\experiments.db | Random | caps_swe_bench_v3_Random_42 | 2 | 50 | NOT RECORDED IN EPISODES | 48 | 0.04 | 1644.96 | 66742.92 | -0.1258 |
| data\experiments.db | CAPS | caps_swe_bench_v3_full_dataset_CAPS_42 | 1 | 300 | NOT RECORDED IN EPISODES | 299 | 0.0033 | 225.84 | 64213.13 | -0.1057 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v3_full_dataset_LLMPlanner_42 | 37 | 240 | NOT RECORDED IN EPISODES | 203 | 0.1542 | 1590.02 | 42875.3 | -0.0099 |
| data\mock_experiments.db | CAPS | caps_synthetic_mock_CAPS_42 | 0 | 50 | NOT RECORDED IN EPISODES | 50 | 0.0 | 118.52 | 4.34 | -0.006 |
| data\mock_experiments.db | FullContext | caps_synthetic_mock_FullContext_42 | 0 | 50 | NOT RECORDED IN EPISODES | 50 | 0.0 | 118.52 | 270.24 | -0.0077 |
| data\mock_experiments.db | Heuristic | caps_synthetic_mock_Heuristic_42 | 0 | 50 | NOT RECORDED IN EPISODES | 50 | 0.0 | 118.52 | 1.31 | -0.0059 |
| data\mock_experiments.db | Random | caps_synthetic_mock_Random_42 | 0 | 50 | NOT RECORDED IN EPISODES | 50 | 0.0 | 118.52 | 1.63 | -0.0059 |

