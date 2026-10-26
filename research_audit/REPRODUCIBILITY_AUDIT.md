# Reproducibility Audit

Score: 58/100.

Rationale: dependencies and configs are present; synthetic data generation is deterministic by seed; experiments are stored in SQLite; CLI commands are discoverable. However, experiment rows do not store full LLM call traces, prompts, API token usage, provider choice, exact provider model versions, policy matrices/checkpoints, or failed attempts. SWE-bench evaluation is LLM-judge based rather than official test execution, and there is no explicit train/validation/test split in the runner despite config fields.

Exact available commands:

```powershell
python -m pytest
caps dataset generate --size 50 --seed 42 --output data/synthetic/benchmark.jsonl
caps experiment run --config configs/caps_synthetic_mock.yaml
caps experiment run --config configs/caps_synthetic.yaml
caps experiment run --config configs/caps_swe_bench.yaml
caps experiment run --config configs/caps_swe_bench_full.yaml
caps results export --format csv --config configs/default.yaml --output experiments/results/summary
```
