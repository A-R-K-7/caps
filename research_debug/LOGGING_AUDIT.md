# Logging Audit
- Episode, Decision, and Update rows all properly log to their respective CSV files.
- The absence of LLM calls in the logging package is because the LLM provider uses an internal `call_log`. Real LLM calls in production (Groq) should add specific API metric tracking to the Episode logger (e.g. `api_tokens`).
- Token accounting is strictly split: `budget_tokens` and `selected_tokens` reflect *communication tokens* (CIU text length).
