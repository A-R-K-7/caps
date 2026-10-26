# Results Audit

All results below are recalculated from raw `episodes` rows, not from precomputed summary Markdown/CSV.

| Database | Method | Experiment ID | Successful | Completed | Other failures | Correct success | 95% CI | Mean tokens | Median tokens | Mean latency ms | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data\experiments.db | CAPS | caps_ark_CAPS_42 | 182 | 200 | 18 | 0.91 | [0.862, 0.942] | 118.46 | 121.0 | 7950.78 | 0.8519 |
| data\experiments.db | FullContext | caps_ark_FullContext_42 | 184 | 200 | 16 | 0.92 | [0.874, 0.950] | 118.46 | 121.0 | 7559.54 | 0.8636 |
| data\experiments.db | Heuristic | caps_ark_Heuristic_42 | 188 | 200 | 12 | 0.94 | [0.898, 0.965] | 118.46 | 121.0 | 7602.36 | 0.884 |
| data\experiments.db | LLMPlanner | caps_ark_LLMPlanner_42 | 182 | 200 | 18 | 0.91 | [0.862, 0.942] | 83.42 | 77.0 | 16331.19 | 0.8371 |
| data\experiments.db | Random | caps_ark_Random_42 | 185 | 200 | 15 | 0.925 | [0.880, 0.954] | 118.46 | 121.0 | 7647.52 | 0.8681 |
| data\experiments.db | CAPS | caps_experiment_CAPS_42 | 162 | 200 | 38 | 0.81 | [0.750, 0.858] | 118.46 | 121.0 | 7211.45 | 0.7563 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_42 | 23 | 50 | 27 | 0.46 | [0.330, 0.596] | 1644.96 | 1589.5 | 68107.67 | 0.3006 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_43 | 13 | 50 | 37 | 0.26 | [0.159, 0.396] | 1644.96 | 1589.5 | 53004.35 | 0.1026 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_44 | 2 | 50 | 48 | 0.04 | [0.011, 0.135] | 1644.96 | 1589.5 | 23646.18 | -0.1188 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_45 | 0 | 9 | 9 | 0.0 | [0.000, 0.299] | 1699.56 | 1673.0 | 21622.82 | -0.1599 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_46 | 1 | 25 | 24 | 0.04 | [0.007, 0.195] | 1647.28 | 1616.0 | 22661.19 | -0.1186 |
| data\experiments.db | CAPS | caps_swe_bench_v1_CAPS_47 | 2 | 50 | 48 | 0.04 | [0.011, 0.135] | 596.64 | 0.0 | 15155.51 | -0.0628 |
| data\experiments.db | FullContext | caps_swe_bench_v1_FullContext_42 | 2 | 50 | 48 | 0.04 | [0.011, 0.135] | 1644.96 | 1589.5 | 12177.72 | -0.0775 |
| data\experiments.db | Heuristic | caps_swe_bench_v1_Heuristic_42 | 3 | 50 | 47 | 0.06 | [0.021, 0.162] | 1644.96 | 1589.5 | 23394.86 | -0.1002 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v1_LLMPlanner_42 | 25 | 50 | 25 | 0.5 | [0.366, 0.634] | 1633.36 | 1589.5 | 29427.54 | 0.3335 |
| data\experiments.db | Random | caps_swe_bench_v1_Random_42 | 39 | 50 | 11 | 0.78 | [0.648, 0.872] | 1644.96 | 1589.5 | 26945.15 | 0.6205 |
| data\experiments.db | CAPS | caps_swe_bench_v2_CAPS_42 | 6 | 50 | 44 | 0.12 | [0.056, 0.238] | 1162.04 | 1254.0 | 17440.11 | -0.011 |
| data\experiments.db | FullContext | caps_swe_bench_v2_FullContext_42 | 47 | 50 | 3 | 0.94 | [0.838, 0.979] | 1644.96 | 1589.5 | 28565.59 | 0.7764 |
| data\experiments.db | Heuristic | caps_swe_bench_v2_Heuristic_42 | 7 | 50 | 43 | 0.14 | [0.070, 0.262] | 1644.96 | 1589.5 | 23091.57 | -0.0211 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v2_LLMPlanner_42 | 8 | 50 | 42 | 0.16 | [0.083, 0.285] | 1624.96 | 1569.5 | 80701.02 | -0.006 |
| data\experiments.db | Random | caps_swe_bench_v2_Random_42 | 25 | 50 | 25 | 0.5 | [0.366, 0.634] | 1644.96 | 1589.5 | 22721.3 | 0.3451 |
| data\experiments.db | CAPS | caps_swe_bench_v3_CAPS_42 | 2 | 50 | 48 | 0.04 | [0.011, 0.135] | 828.6 | 951.0 | 50901.75 | -0.0866 |
| data\experiments.db | FullContext | caps_swe_bench_v3_FullContext_42 | 1 | 50 | 49 | 0.02 | [0.004, 0.105] | 1644.96 | 1589.5 | 56701.21 | -0.1458 |
| data\experiments.db | Heuristic | caps_swe_bench_v3_Heuristic_42 | 1 | 50 | 49 | 0.02 | [0.004, 0.105] | 1644.96 | 1589.5 | 67964.01 | -0.1458 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v3_LLMPlanner_42 | 3 | 50 | 47 | 0.06 | [0.021, 0.162] | 1642.96 | 1589.5 | 68168.72 | -0.1057 |
| data\experiments.db | Random | caps_swe_bench_v3_Random_42 | 2 | 50 | 48 | 0.04 | [0.011, 0.135] | 1644.96 | 1589.5 | 66742.92 | -0.1258 |
| data\experiments.db | CAPS | caps_swe_bench_v3_full_dataset_CAPS_42 | 1 | 300 | 299 | 0.0033 | [0.001, 0.019] | 225.84 | 0.0 | 64213.13 | -0.1057 |
| data\experiments.db | LLMPlanner | caps_swe_bench_v3_full_dataset_LLMPlanner_42 | 37 | 240 | 203 | 0.1542 | [0.114, 0.205] | 1590.02 | 1500.0 | 42875.3 | -0.0099 |
| data\mock_experiments.db | CAPS | caps_synthetic_mock_CAPS_42 | 0 | 50 | 50 | 0.0 | [0.000, 0.071] | 118.52 | 121.0 | 4.34 | -0.006 |
| data\mock_experiments.db | FullContext | caps_synthetic_mock_FullContext_42 | 0 | 50 | 50 | 0.0 | [0.000, 0.071] | 118.52 | 121.0 | 270.24 | -0.0077 |
| data\mock_experiments.db | Heuristic | caps_synthetic_mock_Heuristic_42 | 0 | 50 | 50 | 0.0 | [0.000, 0.071] | 118.52 | 121.0 | 1.31 | -0.0059 |
| data\mock_experiments.db | Random | caps_synthetic_mock_Random_42 | 0 | 50 | 50 | 0.0 | [0.000, 0.071] | 118.52 | 121.0 | 1.63 | -0.0059 |

API failure counts are NOT AVAILABLE from episode rows. No reliable denominator beyond completed episode rows was recoverable unless experiment metadata total_episodes differs from row count.

## 45 vs 50 / Incomplete Episode Problems

No completed experiment with exactly 45 CAPS episode rows was found. Two incomplete CAPS records exist: `caps_swe_bench_v1_CAPS_45` is marked running with 9 episode rows despite total_episodes=50, and `caps_swe_bench_v1_CAPS_46` is marked running with 25 episode rows despite total_episodes=50. The full-dataset `caps_swe_bench_v3_full_dataset_LLMPlanner_42` record is marked running with 240 episode rows despite total_episodes=300. Causes such as quota, timeout, or manual interruption are NOT VERIFIED because episode rows do not preserve stop reasons or API failure records.

## Full-Dataset Experiment

`caps_swe_bench_v3_full_dataset_CAPS_42`: intended 300 tasks, completed 300 rows, successes 1, failures 299, correct success = 1/300 = 0.003333... = 0.33%. Model recorded: `stealth/ox-alpha`; dataset recorded: `swe_bench_lite`; token metric mean = selected CIU tokens only, not total LLM usage.
