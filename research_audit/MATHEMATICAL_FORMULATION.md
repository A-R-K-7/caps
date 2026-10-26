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
