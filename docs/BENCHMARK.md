# Benchmark

Benchmark levels describe the computation being evaluated. Evaluation tiers
describe how an implementation is executed.

| Level | Scope | Example |
| --- | --- | --- |
| L1 | Primitive operator | RMSNorm, linear projection, RoPE |
| L2 | Composite operator | QKV + normalization + RoPE, SwiGLU MLP |
| L3 | Whole model | Model forward, loss, backward and optimizer step |

L3 is measured by Tier 3 (`evograd tier3-bench --model <name>`): loss,
backward, AdamW and gradient reset. The architectural-block level (one decoder
layer, forward plus VJP) was removed on 2026-10-13; the decoder layer a model's
L2 sites take their real invocations from is a data source
(`topdown/<model>/layer`), not a level.

## Cases and ownership

- `evograd.ops` owns reusable primitive contracts and correctness cases.
- `benchmark/operator_suite` owns operator performance grids and aggregation.
- `benchmark/topdown/<model>` owns model configurations, harvested cases and
  captured layers. Current families are Qwen3-0.6B, Llama-3.2-1B and AlphaFold3.
- Qwen3.5-4B (`benchmark/topdown/qwen3_5_4b`, text model): L3 at B1 x T2048 on one
  GH200 (24 Gated DeltaNet + 8 gated attention layers, dense MLP), seven
  calibrated L2 sites and ten bound L1 primitives; L2 invocations come from
  layers 14 and 15, captured or config-derived. No L3 gate yet; see
  `docs/QWEN3_5_4B.md`.
- `evograd.evaluation` owns execution, numerical checks, calibration and timing.

Qwen and Llama each declare their own four L2 tasks. Attention with output
projection, SwiGLU MLP and residual RMSNorm share reference computations but
have different model dimensions. Qwen's QKV task additionally includes Q/K
normalization.

The L3 adapters build the whole model and install providers at their declared
sites on every layer. They are registered through `TIER3_ADAPTERS`; they are not
monolithic `OpDecl` kernels. L1/L2 tasks remain in `benchmark.TASKS`.

- `benchmark/operator_suite/` is the implementation-neutral operator suite. It
  owns the fused tasks that belong to no single model, under
  `operator_suite/tasks/level2/`, and aggregates by family and Level. Trusted
  Liger adapters remain co-located with the task that uses them for now.
- `benchmark/topdown/` starts from whole-model execution, harvest, and frozen
  snapshots. It currently contains `qwen3_0_6b`, `llama3_8b`, `megafold`, and
  the retired AlphaFold3 declaration, and owns the Level-2 tasks whose identity
  comes from a captured model.

## Provenance

Captured cases identify the model execution, layer, weights, inputs and incoming
gradients through verified artifacts. Config-derived cases specify their
architecture and seeds and can run without a full-model capture. Reports retain
this distinction. Changing dimensions, dtype, data or backend changes the case.

The `llama_3_8b` configuration remains registered for existing operator-suite
grids. It is distinct from the Llama-3.2-1B model workload. Select named suites
explicitly when an operator has both generic and model-observed grids.

## Measurement

## The Level-3 workload: MegaFold

`megafold` is the registered Level-3 task. It runs one full MegaFold training
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
  the timed step and dilute the measured speedup. That dilution is the Level-3
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

Correctness comparisons run before timing. Tier 2 and Tier-3 `strict` reject
failed checks before timing. Tier-3 `report-first` records finite numerical
mismatches and continues to end-to-end measurement; unavailable comparisons
remain explicitly incomplete. Structural and execution failures still stop
providers. A timing result is not a numerical correctness verdict.

Speedup is reference latency divided by candidate latency for the same case and
execution boundary. Memory and short training trajectories are reported
separately. The [evaluation guide](../src/evograd/evaluation/README.md) specifies
the current method and input sources; the
[September 17 report](experiments/benchmark_run_20260917.md) identifies the
workloads, timing boundaries and limitations behind the latest Qwen/Llama table.

See [Qwen usage](QWEN3_LEVEL3.md), the [evaluation guide](../src/evograd/evaluation/README.md)
and [result paths](RESULTS_LAYOUT.md).

## Vision and multimodal workloads

ConvNeXt V2 Tiny and Qwen2-VL-2B now have whole-model training adapters,
observed L1 bindings and selected L2 replacement sites. See
[VISION_WORKLOADS.md](../VISION_WORKLOADS.md) for fixed configurations, source
training recipes, harvest/calibration commands and validation status.
