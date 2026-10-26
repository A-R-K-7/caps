# CAPS Policy Audit

# Mathematical Formulation

Implemented CAPS policy uses a per-CIU-type contextual LinUCB model.

Context vector actually used by `LinUCBPolicy._build_context` is 9-dimensional:

`x = [token_budget/10000 clipped to 1, planner_confidence, previous_turns/5 clipped to 1, ciu_tokens/2000 clipped to 1, ciu.priority, ciu.historical_utility, ciu.historical_selection_rate, ciu.uncertainty, cosine(task_embedding, ciu.embedding)]`

The raw task embedding, receiver embedding/capability, receiver historical success, latency budget, task category, and difficulty exist in schemas but are not part of the LinUCB feature vector.

For each CIU type arm `t`, initialized with `A_t = I`, `b_t = 0`:

`theta_t = A_t^{-1} b_t`

`UCB_t(x) = x^T theta_t + alpha * sqrt(x^T A_t^{-1} x)`

Selected CIUs are greedily chosen by descending score subject to token budget and positive score. The selected list is then reversed, preserving neither original dataset order nor pure descending order in the returned action.

Update for each selected CIU:

`A_t <- A_t + x x^T`

`b_t <- b_t + r_i x`

where `r_i = reward` unless VOI is available; if VOI exists then `r_i = (voi_i + reward) / 2`.

Reward:

`R = 1[success] - lambda * (w_tokens * min(tokens_used/token_budget,1) + w_latency * min(latency_ms/max_latency_ms,1) + w_clarification * min(clarification_turns/max_clarification,1) + w_retries * min(retry_count/max_retries,1))`

## Does CAPS Actually Learn?

| experiment_id | episodes | mean_reward_first_half | mean_reward_second_half | mean_tokens_first_half | mean_tokens_second_half | evidence |
| --- | --- | --- | --- | --- | --- | --- |
| caps_swe_bench_v1_CAPS_42 | 50 | 0.3997622755071004 | 0.20147291212819418 | 1647.28 | 1642.64 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v1_CAPS_43 | 50 | 0.24180683422292074 | -0.036648764126078294 | 1647.28 | 1642.64 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v1_CAPS_44 | 50 | -0.15670082727864584 | -0.08083119191385907 | 1647.28 | 1642.64 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v1_CAPS_45 | 9 | -0.14480269956588745 | -0.171968 | 1575.0 | 1799.2 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v1_CAPS_46 | 25 | -0.15356376579708525 | -0.08633162279862625 | 1660.5833333333333 | 1635.0 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v1_CAPS_47 | 50 | -0.0619733350845337 | -0.06369715006663006 | 632.24 | 561.04 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_experiment_CAPS_42 | 200 | 0.6355918623936971 | 0.8769561409657796 | 116.39 | 120.53 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_ark_CAPS_42 | 200 | 0.8545848842474619 | 0.8492225452130636 | 116.39 | 120.53 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v2_CAPS_42 | 50 | -0.040130275223795586 | 0.018114283437601703 | 1018.16 | 1305.92 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v3_CAPS_42 | 50 | -0.1256776398203532 | -0.0474224 | 971.64 | 685.56 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_swe_bench_v3_full_dataset_CAPS_42 | 300 | -0.10469733333333334 | -0.10670346666666668 | 284.1 | 167.58666666666667 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |
| caps_synthetic_mock_CAPS_42 | 50 | -0.005876967994689942 | -0.006032939046223959 | 116.84 | 120.2 | Episode rows do not store A/b/theta/UCB scores, so matrix-level policy change is NOT AVAILABLE. Selection/reward trends are observational only. |

Policy checkpoints/A matrices/b vectors/theta values were not found in persisted experiment artifacts. Unit tests show parameters can change, but recorded experiments do not store enough state to verify matrix-level learning after the fact.
