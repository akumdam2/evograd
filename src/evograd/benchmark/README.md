# Benchmark

`evograd.benchmark` defines **what is benchmarked**: task scope, workloads,
shapes, provenance, and aggregation. It does not define how a candidate runs;
execution and correctness checks belong to
[`evograd.evaluation`](../evaluation/README.md).

## Four-level task hierarchy

| Level | Paper term | Scope |
| ---: | --- | --- |
| L1 | primitive | One mathematical primitive and its backward contract |
| L2 | composite | A boundary composed from multiple adjacent primitives |
| L3 | architectural-block integration | Cross-operator integration inside a real architectural block |
| L4 | whole-model workload | A complete model-training workload |

A Level is the integration scope of an optimization, not the strength of a
measurement. An L2 task may be evaluated in T1, T2, or T3; Level and Tier are
orthogonal.

## Package layout

```text
benchmark/
├── core/                 task registry and cross-task aggregation
├── operator_suite/       implementation-neutral selection and suite CLI
└── topdown/
    ├── common/           harvest, snapshot, smoke, and workload facilities
    ├── qwen3_0_6b/       the current Qwen3-0.6B L4-to-L1 path
    ├── llama3_8b/        the Llama-3-8B workload and harvest
    └── alphafold3/       the AlphaFold3 whole-model declaration
```

`evograd.ops` contains operator contracts, references, tolerances, and operator
workload shapes. `operator_suite` selects those declarations; it does not copy
`OpDecl`. Whole-model declarations are registered in
`evograd.benchmark.WORKLOADS`.

There is one explicit data-provenance bridge: model-derived operator
declarations read frozen JSON only through the workload-neutral
`evograd.benchmark.topdown` snapshot registry. They may not import a concrete
workload package or any evaluation module. Keeping this narrow bridge avoids a
second copy of observed shapes while the AST layering test enforces its scope.

The declarations under `ops/level3` are legacy direct-block tasks retained for
reproducibility. They do not constitute a completed implementation of the
paper's new top-down L3 architectural-block integration.

## Two benchmark families

- The operator suite selects tasks from the existing Liger-derived and other
  declared operators, then aggregates coverage, full-step speedup, and retained
  state by family and Level.
- The top-down benchmark starts from a real L4 workload and traces through a
  harvest and frozen snapshot to L3/L2/L1. The current Qwen instance is named
  `qwen3_0_6b`; it must not be confused with the planned Qwen3-Next workload.

Registry counts must be queried from `evograd.ops.OPS`,
`evograd.benchmark.WORKLOADS`, and
`evograd.benchmark.topdown.TOPDOWN_WORKLOADS`; documentation does not duplicate
counts that can drift.

## Results

Existing `results/` and external caches remain in place and read-only. Only new
runs use:

```text
results/benchmark/operator_suite/...
results/benchmark/topdown/<workload>/...
```

See [`docs/RESULTS_LAYOUT.md`](../../../docs/RESULTS_LAYOUT.md) for the full
new/legacy path policy.
