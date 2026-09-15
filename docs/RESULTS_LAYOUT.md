# Result paths

Generated candidates, captured tensors, calibrated policies, logs and raw
measurements stay local. Use the ignored `results/` directory:

```text
results/
├── benchmark/operator_suite/<run>/
├── benchmark/topdown/<workload>/<run>/
├── evaluation/tier1/<op>/<run>/
├── evaluation/tier2/<op>/<run>/
├── evaluation/tier3/<workload>/block/<run>/
├── evaluation/tier3/<workload>/model/<run>/
└── experiments/<experiment>/<run>/
```

Keep existing result paths and artifacts unchanged. Their embedded identities
and hashes determine which workload they describe.

Publish selected summaries in `docs/experiments/benchmark_run_YYYYMMDD.md`.
Include settings, correctness outcomes, performance, limitations and artifact
identifiers. Candidate source and raw evidence are not included by default.
Small regression-test fixtures remain under `tests/fixtures/`.
