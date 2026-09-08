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
  owns the fused tasks that belong to no single model, under
  `operator_suite/tasks/level2/`, and aggregates by family and Level. Trusted
  Liger adapters remain co-located with the task that uses them for now.
- `benchmark/topdown/` starts from whole-model execution, harvest, and frozen
  snapshots. It currently contains `qwen3_0_6b`, `llama3_8b`, and the
  AlphaFold3 declaration, and owns the Level-2 tasks whose identity comes from
  a captured model.

Reusable Level-1 primitives stay in `evograd.ops`, which owns the mathematics,
the reference, and the generic correctness cases — and nothing about any
model. Every performance case is bound onto those contracts on the benchmark
side by `evograd.benchmark.cases`: the suite's grids from
`benchmark/operator_suite/cases/`, a model's observed cases from its top-down
manifest. A primitive declares no timed grid, no benchmark coverage, no named
suite, no regime split and no case weighting, and imports neither a snapshot
nor a model configuration table.

`evograd suite` is parsed and coordinated by `evograd.suite_cli` at the root,
not by the benchmark package: selecting what to run is the benchmark's, running
it is evaluation's, and putting the two together is neither.

The legacy `ops/level3` direct-block declarations have been **deleted**. They
were whole-block pair benchmarks and never demonstrated the top-down L3
integration contract, which also requires model-derived activation coverage, a
complete return contract, and a layer-level saved-state contract. No task
declares level 3 today.

## Registries

- Reusable Level-1 primitives: `evograd.ops.PRIMITIVES`
- Executable benchmark tasks (L1 and L2): `evograd.benchmark.TASKS`
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
