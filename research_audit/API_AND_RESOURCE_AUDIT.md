# API and Resource Audit

Providers implemented: Groq SDK and OpenAI-compatible base URL mode used for OpenRouter. Mock provider exists for tests.

Configured secret-bearing entries found, without exposing values:

```json
{
  ".env": {
    "GROQ_API_KEYS": 1,
    "OPEN_ROUTER_API_KEY": 1
  },
  "creds.env": {
    "GEMINI_API_KEY": 1,
    "GEMINI_API_KEY2": 1
  },
  ".env.example": {
    "GROQ_API_KEY": 1
  }
}
```

Multi-key round-robin: partially supported. `get_groq_api_keys` can parse multiple Groq keys; `GroqProvider` creates one client per key and switches keys when an extended/daily rate limit occurs before a problem has started. It does not rotate every request in strict round-robin fashion.

Rate-limit handling: sleeps 2.5 seconds before calls; retries up to 3 attempts; parses retry-after headers and some error-message wait times; can switch keys or fall back to configured fallback models.

LLM cache: `LLMCache` class exists and cache config exists, but `GroqProvider.generate` and the main runner do not call `LLMCache.get/put`. Therefore LLM response caching is implemented as a standalone class but NOT VERIFIED as active in experiments. Embedding cache is active when cache enabled.
