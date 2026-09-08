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

Performance cases are bound onto primitive contracts here rather than written
into them. `evograd.benchmark.cases` is the binder and it applies two layers,
in this order:

1. `operator_suite/cases/<primitive>.py` — the suite's timed grid, its untimed
   benchmark coverage, its named suites, the shape-regime split and the case
   weighting. Grids whose dimensions were computed from a published Llama-3 or
   AlphaFold3 configuration live here: the dimensions came from a model, but
   the case belongs to the suite, and calling it top-down coverage would claim
   evidence that does not exist.
2. `topdown/<model>/` — cases *observed* in that model's captured run, read
   from its frozen snapshot. Which primitives a model binds, under which suite
   name, whether the cases precede or follow the primitive's own coverage and
   which suite mirrors that coverage are all that model's decisions, and live
   in its own manifest — for Qwen3-0.6B, `OBSERVED_BINDINGS` in
   `topdown/qwen3_0_6b/levels/level1/manifest.py`.

`benchmark/cases.py` is the mechanism and only the mechanism: it names no
model, holds no task list, and takes the configuration as an argument.
`benchmark/core/registry.py` is the assembly point — the one place that says
which models exist, importing each one's table and handing it to the binder.
Adding a second harvested architecture means writing a table in its manifest
and naming it there; no primitive package and no shared module changes.

Both return a new `OpDecl`; the primitive is never mutated, and a primitive
that declared its own grid would be refused rather than silently merged, so a
case collection has exactly one owner. That is why `evograd.ops` reaches
neither a snapshot nor a model configuration table, and why importing it in a
fresh interpreter loads no benchmark module at all.

The suite *command* is not here either. `evograd suite` asks the benchmark what
to run and evaluation to run it, which is composition rather than definition,
so it lives at the root in `evograd.suite_cli`. Nothing under `benchmark`
imports `evaluation`, and the dependency-direction test carries no exemption.

Reviewed pair baselines (`liger.py`, `cublas.py`, `triton_tutorial.py`) stay
beside the task or primitive whose contract they implement. That is deliberate:
an adapter is an implementation of one declaration, discovered through the
declaration that names it, not a case collection.

### The one remaining exception, stated

`ops/level1/rope` still imports `evograd.opdecl.models`. It is not a case list:
RoPE's input generator reads `rope_theta` from *each workload's own
provenance*, because Llama-3's 500000 and Qwen3's 1000000 produce different
rotations and a hard-coded constant would let a kernel be self-consistent and
wrong. Resolving a workload's model key to its published configuration needs
that table. The rule the generator implements is generic; only the fallback
used when a case carries no provenance names a model, and changing it would
change the inputs generated for the provenance-free correctness cases. It is
declaration infrastructure rather than a benchmark manifest, so importing the
primitives still loads no benchmark package and no frozen snapshot.

The legacy direct-block declarations that used to live under `ops/level3` have
been deleted. They did not constitute a completed implementation of the
top-down L3 architectural-block integration, and no task declares level 3
today.

## Two benchmark families

- The operator suite selects tasks from the existing Liger-derived and other
  declared operators, then aggregates coverage, full-step speedup, and retained
  state by family and Level.
- The top-down benchmark starts from a real L4 workload and traces through a
  harvest and frozen snapshot to L3/L2/L1. The current Qwen instance is named
  `qwen3_0_6b`; it must not be confused with the planned Qwen3-Next workload.

Registry counts must be queried from `evograd.ops.PRIMITIVES`,
`evograd.benchmark.TASKS`, `evograd.benchmark.WORKLOADS`, and
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
