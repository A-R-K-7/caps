# Paper Claims

| Current Claim | Evidence | Supported? | Qualification Needed |
| --- | --- | --- | --- |
| CAPS reduces token usage by 55%. | Must be recomputed per chosen experiment; no universal 55% result found by audit script. | NOT VERIFIED | Specify database, methods, denominator, CIU-token metric. |
| CAPS maintains task performance. | Episode-level success rows exist for some runs. | PARTIAL | Needs paired tasks and confidence intervals; account for incomplete baselines. |
| CAPS improves task success. | Current rows may show differences but sample size/fairness issues remain. | NOT VERIFIED | Controlled paired tests and sufficient power. |
| CAPS learns communication strategies. | LinUCB update exists; persisted policy state absent. | PARTIAL | Log A/b/theta/UCB and frozen-policy evaluation. |
| VOI improves policy learning. | VOI can update CAPS, but no ablation proving improvement found. | NO | Run VOI vs no-VOI ablation. |
| CAPS significantly outperforms baselines. | No valid significance evidence found. | NO | Appropriate paired tests with adequate sample. |
