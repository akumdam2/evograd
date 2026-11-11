# Benchmark

Benchmark levels describe the computation being evaluated. Evaluation tiers
describe how an implementation is executed.

| Level | Scope | Example |
| --- | --- | --- |
| L1 | Primitive operator | RMSNorm, linear projection, RoPE |
| L2 | Composite operator | Attention block, SwiGLU MLP |
| L3 | Whole model | Model forward, loss, backward and optimizer step |

L3 is measured by Tier 3 (`evograd evaluate <model> ...`): loss, backward,
AdamW and gradient reset. An L2 site's real invocations come from the
verification capture of a whole training step; that capture is a data source,
not a level.

## Cases and ownership

- `evograd.ops` owns reusable primitive contracts, references and input
  generators.
- `benchmark/topdown/<model>` owns model configurations and harvested cases.
  Current families are Qwen3-0.6B, Llama-3.2-1B and MegaFold.
- Qwen3.5-4B (`benchmark/topdown/qwen3_5_4b`, text model): L3 at B1 x T2048 on one
  GH200 (24 Gated DeltaNet + 8 gated attention layers, dense MLP), six
  L2 sites and ten bound L1 primitives; every case is judged on the training
  calls the verification capture records. No L3 gate yet; see
  `docs/QWEN3_5_4B.md`.
- OLMoE-1B-7B (`benchmark/topdown/olmoe_1b_7b`): L3 at B2 x T2048 with
  grouped-GEMM experts (16 layers of multi-head attention and 64 routed SwiGLU
  experts, 8 active), three L2 sites and six bound L1 primitives; every case is judged
  on the training calls the verification capture records.
  No GPU session yet and no L3 gate; see `docs/OLMOE_1B_7B.md`.
- `evograd.evaluation` owns execution, numerical checks, calibration and timing.

Qwen and Llama each declare their own L2 tasks: the attention block (the
whole attention module), SwiGLU MLP and residual RMSNorm, plus Qwen's LM head
with the loss. They share reference computations but have different model
dimensions; Qwen's attention block additionally includes Q/K normalization.

The L3 workloads (`evaluation/tier3/workloads`) build the whole model and
install providers at their declared sites on every layer; they are not
monolithic `OpDecl` kernels. L1/L2 tasks remain in `benchmark.TASKS`.

- Every Level-2 task belongs to a model. Liger is not a baseline.
- `benchmark/topdown/` starts from whole-model execution, harvest, and frozen
  snapshots. It contains one package per model (`qwen3_0_6b`, `llama3_2_1b`,
  `qwen3_5_4b`, `olmoe_1b_7b`, `convnextv2_tiny`, `qwen2_vl_2b`, `dit_s_2`,
  `torchrec_dlrm`, `megafold`) and owns the Level-2 tasks whose identity comes
  from a captured model.

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
Level-1 `layernorm` and `swiglu` grids came from this model's configuration,
and this task runs that configuration, so those grids and the sites the model
executes are the same widths by construction.

**Real data, harvested cases.** The step runs on one real PDB structure
(7a4d, 384 tokens after MegaFold's crop, a 12,097-row MSA), featurized once by
MegaFold's own training featurization and frozen
(`benchmark/topdown/megafold/data.py`); one structure, because every site's
shapes follow its token, atom and MSA counts, so steps differ by their seed
alone. The Level-1/2 cases are harvested from one such step
(`benchmark/topdown/megafold/harvest/`). The Level-3 case re-derives from the
registered AlphaFold3 configuration, and the workload refuses a case that does
not match.

**Tasks come from the training step** (`docs/AF3.md`), not from which kernels
MegaFold ships. MegaFold trains with float32 parameters under a bfloat16 autocast
over the trunk; the diffusion module, heads and loss run with autocast off. Each
Level-2 contract is declared exactly as MegaFold's eager PyTorch computes it in that
precision. Each site adapter makes autocast's casts itself, runs its kernel with
autocast off and returns the dtype the replaced module returns.

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
other workload in the tree is unaffected. The patchable sites are listed in
`docs/AF3.md` section 9. Sites that nest are declared, and are never installed
together.

**What the workload excludes**, stated in the declaration rather than left to
be discovered:

- Windowed atom attention, the embedding and pooling layers, the output heads,
  the losses and the alignment hold no site. Whatever is outside the installed
  sites runs stock inside the timed step and dilutes the measured speedup. That
  dilution is the Level-3 result, not a flaw.
- Attention-weight dropout is disabled for every provider alike. The warm start
  was trained without it, and a dropout firing in one provider but not another
  would desynchronize their RNG streams.
- The data is one real structure featurized once and frozen. Nothing here
  measures data loading, and no result speaks to prediction quality. Its
  templates are masked (MegaFold's training never passes a kalign binary) and
  it drew no constraints.
- EMA, gradient clipping, DeepSpeed and sequence parallelism are excluded. The
  harness trains one model in one process with AdamW. Sequence parallelism is
  pinned to a group size of one, which makes MegaFold's collectives no-ops
  rather than removing them from the code path.
- The recycling count (one pass) and whether the confidence head is called
  are pinned rather than sampled. MegaFold draws the first from
  `randint(1, 4)` every step and calls the second every tenth. Verification
  and the ADBench evaluation call the confidence head on every step, so that
  its sites get a gradient; the tier-3 CLI leaves it off unless
  `--confidence-head`.
- The template embedder (its output is masked without templates) and the
  recycling projections (multiplied by zero in the one pass) get no gradient
  in this step and hold no sites (`workload.NOT_SITES`); they run their eager
  spelling in every provider.
- Weights are randomly initialized from a fixed seed. Verification and the
  ADBench evaluation load a frozen warm start instead
  (`integrations.warm_start`: 200 updates of MegaFold's own trainer recipe on
  the structure), because at the initialization MegaFold's zero-initialized
  output layers keep every gradient from the MSA module, the diffusion module
  and the input embedder.
- Step times are single-GPU compute per step. They are **not** comparable to
  the MegaFold paper's reported throughput, which is a multi-GPU end-to-end
  number over the real data pipeline.

**Putting MegaFold on PYTHONPATH.** MegaFold is a source checkout, not a
distribution: it ships no `setup.py` and no `pyproject.toml`, so there is no
extra to install and nothing for pip to pin. Clone it and point `PYTHONPATH` at
the root:

```bash
git clone https://github.com/Supercomputing-System-AI-Lab/MegaFold
export PYTHONPATH=/path/to/MegaFold:src
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

## Registries

Correctness is judged by the verification package's Gates 1/2/3 on real
payloads, separately from timing. A timing result is not a numerical
correctness verdict.

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
training recipes, harvest commands and validation status.

## Diffusion and recommendation workloads

DiT-S/2 and the single-device TorchRec DLRM architecture have whole-model
training adapters and execution-derived operator cases. DiT supports generated
and pinned image-derived inputs. DLRM supports generated inputs and fixed
Criteo Kaggle batches prepared from TorchRec-format NumPy files. See the [DiT benchmark](../src/evograd/benchmark/topdown/common/generated/README.md)
and [DLRM benchmark](../src/evograd/benchmark/topdown/torchrec_dlrm/README.md)
for their contracts and harvest commands.

For real-input source setup, run
`python scripts/prepare_benchmark_inputs.py --benchmark NAME` from the
repository root. Supported names are
`convnextv2_tiny`, `qwen2_vl_2b`, `dit_s_2`, `torchrec_dlrm`, and
`qwen3_0_6b`. It downloads the pinned checkpoint/dataset files or the Criteo
Kaggle archive and runs compatible NumPy preprocessing for DLRM. Files stay under
`data/<benchmark>/`: pinned downloads in `hf_cache/`, DLRM's raw and processed
rows in `raw/` and `processed/`, and fixed input batches in `batches/` after
calibration. The benchmark CLI reads these default paths without extra flags.
