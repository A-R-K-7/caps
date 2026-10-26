# VOI Audit
- Synthetic VOI correctly utilizes the hidden `ground_truth_necessity` / `information_role` fields from the synthetic task data.
- The `CAPS-VOI` method correctly combines this VOI score with the global reward before running the LinUCB update step.
- The `CAPS-NoVOI` baseline correctly suppresses the VOI signal, operating exactly as `CAPS-VOI` but strictly on the global task reward.
