# EvoGrad benchmark specification

This document defines the benchmark task space. See
[`src/evograd/evaluation/README.md`](../src/evograd/evaluation/README.md) for the
execution and validation protocol. Paper terminology takes precedence over
historical code names.

## Benchmark Levels

The top-down benchmark has four integration scopes:

1. **L1 primitive** — one mathematical primitive and its forward/backward
   contract.
2. **L2 composite** — a boundary composed from adjacent primitives in a real
   training workload.
3. **L3 architectural-block integration** — cross-boundary integration inside
   a decoder or protein-model block, including complete returns, saved state,
   and model-derived activation coverage.
4. **L4 whole-model workload** — a complete model training workload used to
   determine whether local gains survive integration.

Level answers “what scope is optimized.” Evaluation Tier answers “in which
execution context is the candidate checked and measured.” They are independent.

## Task sources

EvoGrad retains two benchmark lines:

- `benchmark/operator_suite/` is the implementation-neutral operator suite. It
  selects tasks/configurations from `evograd.ops.OPS` and aggregates by family
  and Level. Trusted Liger adapters remain in `ops/*/liger.py` for now.
- `benchmark/topdown/` starts from whole-model execution, harvest, and frozen
  snapshots. It currently contains `qwen3_0_6b`, `llama3_8b`, and the
  AlphaFold3 declaration.

The existing `ops/level3` direct-block declarations are legacy tasks. They
remain runnable for reproducibility, but do not demonstrate that the paper's
new top-down L3 integration contract is complete.

## Registries

- Operators: `evograd.ops.OPS`
- Whole-model benchmark declarations: `evograd.benchmark.WORKLOADS`
- Top-down snapshot workloads: `evograd.benchmark.topdown.TOPDOWN_WORKLOADS`
- Tier-3 adapters: `evograd.evaluation.tier3.workloads.TIER3_ADAPTERS`

Task counts are derived from these registries rather than duplicated in prose.

## Correctness, provenance, and coverage

An operator contract defines outputs, requested gradients, dtype/shape/layout,
tolerances, permitted backward input overwrites, and saved state. Correctness
is a hard gate before timing. A failed configuration still contributes to
coverage and cannot improve performance by silently disappearing from an
average.

Top-down shapes retain their source: model configuration, harvest manifest,
frozen snapshot, and concrete boundary. A handpicked or reduced configuration
must say so in its provenance; it cannot masquerade as a shape observed in the
model.

## Performance and aggregation

Candidate and reference compare like-for-like full-step latency:

```text
S_i = T_reference(forward + backward) / T_candidate(forward + backward)
```

Geometric aggregation occurs within a task's configurations, then within a
family, then across families. Coverage and retained-state memory are reported
separately and never folded into speedup.

## Result paths

Historical output remains in place. New runs use:

```text
results/benchmark/operator_suite/...
results/benchmark/topdown/<workload>/...
results/evaluation/tier<N>/<workload>/...
```

See [`RESULTS_LAYOUT.md`](RESULTS_LAYOUT.md).
