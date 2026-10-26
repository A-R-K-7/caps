# LLMPlanner Audit
- **Initial finding**: `LLMPlannerPolicy` silently fell back to sending all CIUs (budget limited) upon encountering a JSON parsing error, causing it to mimic the Heuristic baseline.
- **Fix**: The `except (json.JSONDecodeError, KeyError)` block was updated. It now logs the fallback event (`fallback_triggered: True`), logs the reason, and returns an empty selection (`[]`) to correctly penalize the planner for failing, rather than silently substituting a heuristic.
