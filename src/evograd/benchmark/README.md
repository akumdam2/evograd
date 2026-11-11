# Benchmark

`evograd.benchmark` defines **what is benchmarked**: task scope, workloads,
shapes, provenance, and aggregation. It does not define how a candidate runs;
execution and correctness checks belong to
[`evograd.evaluation`](../evaluation/README.md).

## Three-level task hierarchy

| Level | Paper term | Scope |
| ---: | --- | --- |
| L1 | primitive | One mathematical primitive and its backward contract |
| L2 | composite | A boundary composed from multiple adjacent primitives |
| L3 | whole-model workload | A complete model-training workload |

An L2 site's real invocations come from the verification capture of a whole
training step; that capture is a data source, not a level.

A Level is the integration scope of an optimization, not the strength of a
measurement. An L2 task may be evaluated in T1, T2, or T3; Level and Tier are
orthogonal.

## Package layout

```text
benchmark/
├── core/                 task registry and cross-task aggregation
└── topdown/
    ├── common/           harvest, snapshot, smoke, and workload facilities
    ├── qwen3_0_6b/       the Qwen3-0.6B L3-to-L1 path
    ├── llama3_2_1b/      the Llama-3.2-1B workload and harvest
    ├── qwen3_5_4b/       the Qwen3.5-4B (text) L3-to-L1 path
    ├── olmoe_1b_7b/      the OLMoE-1B-7B (routed MoE) L3-to-L1 path
    └── megafold/         the MegaFold whole-model declaration
```

`evograd.ops` contains operator contracts, references and input generators.
Whole-model declarations are registered in `evograd.benchmark.WORKLOADS`.
Correctness is not declared on a task: Gate 1/2/3 judge a candidate on recorded
payloads under the model reference (`evograd.verification`).

Cases are bound onto primitive contracts here rather than written into them.
`evograd.benchmark.cases` is the binder: `topdown/<model>/` holds the cases
*observed* in that model's captured run, read from its frozen snapshot. Two
models may bind the same primitive: each contributes its own named suite
(`qwen3_0_6b_observed`, `llama_3_2_1b_observed`) and the binder keeps both, in
the order the registry supplies them. Only a real collision — the same primitive
under the same suite name — is refused. A workload whose harvest has not been
run contributes nothing and blocks no import; a snapshot that exists and fails
its hash check still raises. Which primitives a model binds, under which suite
name and whether the cases precede or follow the coverage bound before them are
that model's decisions, and live in its own manifest — for Qwen3-0.6B,
`OBSERVED_BINDINGS` in `topdown/qwen3_0_6b/levels/level1/manifest.py`.

`benchmark/cases.py` is the mechanism and only the mechanism: it names no
model, holds no task list, and takes the configuration as an argument.
`benchmark/core/registry.py` is the assembly point — the one place that says
which models exist, importing each one's table and handing it to the binder.
Adding a second harvested architecture means writing a table in its manifest
and naming it there; no primitive package and no shared module changes.

Binding returns a new `OpDecl`; the primitive is never mutated. That is why `evograd.ops` reaches
neither a snapshot nor a model configuration table, and why importing it in a
fresh interpreter loads no benchmark module at all.

Nothing under `benchmark` imports `evaluation`, and the dependency-direction
test carries no exemption.

Reviewed pair baselines (`megafold_pair.py`) stay beside the primitive whose
contract they implement. That is deliberate:
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
change the inputs generated for provenance-free cases. It is
declaration infrastructure rather than a benchmark manifest, so importing the
primitives still loads no benchmark package and no frozen snapshot.

L3 workloads are evaluated by `evograd evaluate <model>` at Tier 3: the whole
training step, with candidate kernels patched into their sites. They are `WorkloadDecl`s, never `OpDecl`s.

## The benchmark

The top-down benchmark starts from a real L3 workload and traces through a
harvest and frozen snapshot to L2/L1.

Registry counts must be queried from `evograd.ops.PRIMITIVES`,
`evograd.benchmark.TASKS`, `evograd.benchmark.WORKLOADS`, and
`evograd.benchmark.topdown.TOPDOWN_WORKLOADS`; documentation does not duplicate
counts that can drift.

## Results

Existing `results/` and external caches remain in place and read-only. Only new
runs use:

```text
results/benchmark/topdown/<workload>/...
```

See [`docs/RESULTS_LAYOUT.md`](../../../docs/RESULTS_LAYOUT.md) for the full
new/legacy path policy.
