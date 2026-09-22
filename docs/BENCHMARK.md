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
  snapshots. It currently contains `qwen3_0_6b`, `llama3_8b`, `megafold`, and
  the retired AlphaFold3 declaration, and owns the Level-2 tasks whose identity
  comes from a captured model.

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

## The Level-4 workload: MegaFold

`megafold` is the registered Level-4 task. It runs one full MegaFold training
step -- forward to the model's own combined loss, backward, optimizer step --
with evolved kernels patched into the module tree by surgery.

MegaFold ([arXiv:2506.20686](https://arxiv.org/abs/2506.20686)) is the
implementation every AlphaFold3 shape in `src/evograd/opdecl/models.py` was
transcribed from. Running it here closes the provenance chain at both ends: the
Level-1/2 operator grids came from this model's configuration, and this task
runs that configuration, so `evoattention`'s benchmark grid and the attention
the model executes are the same widths by construction.

**The shapes are declared, not harvested.** Every benchmark case re-derives
from the registered AlphaFold3 configuration, and the workload re-derives them
a second time at build and refuses a case that does not match. Nothing here
comes from a captured run, and the provenance says so rather than implying a
harvest.

**Three providers, one architecture.** MegaFold builds its own fused Triton
kernels directly into the module tree rather than behind a flag, so the choice
between implementations is made at construction:

| Provider | Tree |
|---|---|
| `eager` | every fused module replaced by the plain PyTorch mathematics it stands for, `use_optimized_evo=None`. The reference. |
| `megafold` | the tree exactly as MegaFold builds it, Triton EvoAttention on. No surgery. |
| `compile` | `torch.compile` of the eager model, on eager RNG. No module surgery at all. |
| `evolved` | the eager tree with the candidate kernels patched into the sites. |

`eager` and `evolved` share a base by construction: the eager surgery runs
first and the candidate patches go on top, so the only difference between them
is the kernels. `compile` is the whole-model compiler baseline, which answers
what a general-purpose compiler gets on this architecture before anyone writes
a kernel. It is compiled without `fullgraph`, because MegaFold draws its
recycling count with `random.randint`, loops in Python over recycling steps,
calls activation checkpointing and carries beartype decorators on every
forward, so a single graph is not reachable and demanding one would fail the
provider rather than measure it.

It also sets `torch._inductor.config.fallback_random`. Inductor generates its
own Philox random numbers, and this model is stochastic in three places at once
(diffusion noise, structured row/column dropout, stochastic frame averaging),
so without the fallback a seeded compiled forward trains on different
randomness than the reference. Measured at 64 residues, that moved the loss by
2.0e-1 against a reference the other providers match to 5e-5; with the fallback
the gap is 2.5e-5. The fallback costs some fusion around the random ops, so the
compile provider's step time is conservative.

These four are different module *trees*, not four kernel sets, so the runner
passes the provider's name to the workload. `build_with_provenance` forwards it
only to a workload that declares a `provider` parameter, which is why every
other workload in the tree is unaffected. Five sites are patchable -- `layer_norm`, `layernorm_linear`,
`transition`, `pair_bias_attention`, `triangle_attention` -- and they are
exactly the boundaries MegaFold fuses, which is what makes its own kernels the
line an evolved one has to clear.

**What the workload excludes**, stated in the declaration rather than left to
be discovered:

- Triangle multiplication, outer product mean, MSA pair-weighted averaging, the
  diffusion module internals and the confidence head internals run stock inside
  the timed step and dilute the measured speedup. That dilution is the Level-4
  result, not a flaw.
- Attention-weight dropout is disabled for every provider alike. No
  fused-attention contract models it, MegaFold's own EvoAttention included, and
  a dropout firing in one provider but not another would desynchronize their
  RNG streams.
- The MSA and template data pipeline is replaced by seeded synthetic features
  at a fixed crop. Nothing here measures data loading, and no result speaks to
  prediction quality.
- EMA, gradient clipping, DeepSpeed and sequence parallelism are excluded. The
  harness trains one model in one process with AdamW. Sequence parallelism is
  pinned to a group size of one, which makes MegaFold's collectives no-ops
  rather than removing them from the code path.
- The recycling count and whether the confidence head is called are pinned
  rather than sampled. MegaFold draws the first from
  `randint(1, num_recycling_steps)` every step and calls the second every tenth.
- Weights are randomly initialized from a fixed seed.
- Step times are single-GPU compute per step. They are **not** comparable to
  the MegaFold paper's reported throughput, which is a multi-GPU end-to-end
  number over the real data pipeline.

**Putting MegaFold on PYTHONPATH.** MegaFold is a source checkout, not a
distribution: it ships no `setup.py` and no `pyproject.toml`, so there is no
extra to install and nothing for pip to pin. Clone it and point `PYTHONPATH` at
the root:

```bash
git clone https://github.com/Supercomputing-System-AI-Lab/MegaFold
PYTHONPATH=/path/to/MegaFold:src python -m evograd.evaluation.tier3.cli \
    --model megafold_2l --residues 128 --baseline none
```

Before a first tier-3 run, `scripts/smoke_megafold.py` builds each provider,
runs one forward and backward, and prints the loss, peak memory and step time.
It runs no part of the tier-3 protocol, so nothing it prints is a measurement
to quote; it answers whether the thing runs at all.

```bash
PYTHONPATH=/path/to/MegaFold:src python scripts/smoke_megafold.py
```

Without MegaFold the task reports as uncovered, which is what every unmet
`requires` does. Note that `megafold.model.megafold` constructs its fused Triton modules
at import time and resolves a Triton driver while doing so, so it needs a
visible GPU: on a login or CPU-only node the import fails even with every
dependency installed, and the workload reports that as its own message rather
than an opaque Triton traceback.

`megafold_2l` is the same architecture with the stacks cut to two pairformer
blocks. Every width, head count and input-feature dimension is the real
configuration, so the kernels see the true shapes while a step stays cheap
enough to iterate on. Iterate on it; report from `megafold`.

The `alphafold3` declaration it replaced is still in the tree and still valid,
with its Tier-3 adapters registered. It is no longer part of the suite.

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
