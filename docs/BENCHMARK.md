# The evograd benchmark

A benchmark for generated **training** kernels: correctness, speed, and the
memory a forward pass retains for its backward. Every task is a forward/backward
pair, not a forward alone, and every timed shape is traceable to a layer of a
real model.

25 operators across three levels, 191 timed configurations.

## Task hierarchy

| Level | What it is | Operators |
| ---: | --- | ---: |
| 1 | **Primitive operators** — one mathematical operation, plus the saved state its backward needs | 18 |
| 2 | **Fused operators** — compositions that occur in real training, evaluated on whether an implementation can optimize across the operator boundary while preserving autograd semantics | 5 |
| 3 | **Architectural blocks** — a whole decoder layer or protein-model block, where the saved-state decision becomes a property of the block rather than of one kernel | 2 |

Levels are declared, not inferred: `OpDecl.level` and `OpDecl.family`. Family
exists for aggregation — see [Metrics](#metrics).

### Level 1 — primitives

`layernorm` `rmsnorm` `poly_norm` `dyt` (norm) · `softmax` `sparsemax`
(reduction) · `swiglu` `geglu` `relu_squared` (activation) · `cross_entropy`
`kl_div` `jsd` `tvd` (loss) · `matmul` `linear` (gemm) · `conv2d` (conv) ·
`evoattention` (attention) · `rope` (positional)

### Level 2 — fused

`fused_add_rms_norm` (residual add + RMSNorm) · `fused_linear_cross_entropy`
(lm_head projection + loss) · `layernorm_linear` (LayerNorm + projection) ·
`gemm_leaky_relu` (GEMM + activation epilogue) · `fused_moe_swiglu` (routing +
grouped GEMM + SwiGLU + down projection)

### Level 3 — blocks

- **`llama3_decoder_layer`** — RMSNorm → Q/K/V → RoPE → causal GQA → output
  projection → residual → RMSNorm → SwiGLU MLP → residual. Ten differentiable
  inputs, ten gradients.
- **`af3_single_repr_block`** — LayerNorm → Q/K/V → attention with a trainable
  pair bias and residue mask → output projection → residual → LayerNorm →
  SwiGLU transition → residual. Thirteen differentiable inputs.

Both blocks state their exclusions in their declarations rather than leaving a
reader to discover them. The Llama block models the **training** forward pass:
no KV cache (per-step mutable state the declaration model does not express) and
no attention-weight output. The AlphaFold3 block is the **single-representation
update only** — a full pairformer block maps `(single, pair) → (single', pair')`
and cannot be declared with one output. The pair path is still exercised in the
backward, because `pair_bias` is a differentiable input and `d_pair_bias`
reduces over the MSA axis; the triangle-multiplicative pair update is not.

## Where the shapes come from

Benchmark shapes are **derived from frozen model configurations**, not written
by hand. `evograd/opdecl/models.py` holds the configurations; each timed
workload carries a `Provenance` naming the model, the component, and the
dimensions the configuration does not fix (batch, token count, crop length).

```python
Provenance(model="llama_3_8b", component="mlp_down", free={"tokens": 8192})
# -> LLAMA_3_8B.mlp_down_dims(tokens=8192) == {"M": 8192, "K": 14336, "N": 4096}
```

`tests/test_provenance.py` re-derives every `hf_config` workload and fails if a
declared shape and the model it cites ever disagree. Provenance is an assertion,
not a comment — 172 of the 191 timed configurations are checked this way.

Four operators carry a weaker claim, and say so rather than inventing a
configuration to justify their numbers:

| Operator | Source | Why |
| --- | --- | --- |
| `conv2d` | `handpicked` | ResNet-style stage resolutions and channel widths, but the declared contract is stride 1 / padding 0, whereas ResNet's 3×3 convolutions pad by 1. These are not literal ResNet layers. |
| `gemm_leaky_relu` | `handpicked` | From Triton's tutorial. No shipped architecture fuses Leaky-ReLU into a GEMM epilogue. |
| `fused_moe_swiglu` | `paper` | Liger's own MoE benchmark grid; routing matches Mixtral-style top-2 over 8–16 experts, with the widths scaled down to fit one GPU. |
| `sparsemax` | `handpicked` | No shipped architecture contains sparsemax at all. The v1 grid claimed Llama-3's 128256-wide logits — sparsemax substituted for softmax — which asserted more than the evidence supports: Triton implementations cap a row at 65536 columns, which is what you would expect if nobody runs it at vocabulary width. The timed grid uses 32768 (mid-sized-vocabulary scale) and the vocabulary widths stay in untimed `coverage`, so the point where implementations stop is still recorded. |

The test suite requires those to explain themselves in a `note`; it does not
require them to re-derive.

## Fixed- and variable-shape evaluation

**Variable shape** is the default: one deployable implementation is measured
across a whole grid. Declarations that define shape regimes evolve a generalist
plus a small-shape and a large-shape specialist, and `evograd dispatch` searches
for the routing threshold that maximizes the geometric-mean speedup, then emits
a dispatcher that routes at runtime.

**Fixed shape** lets a candidate specialize hard for one configuration. Every
operator exposes one single-case suite per timed workload:

```bash
evograd bench --op layernorm --candidate best.py \
    --suite fixed/rows4096-hidden4096-bfloat16
```

## Metrics

### Correctness

A candidate must match the reference forward output and every requested
gradient, in value, shape, and dtype, within the declared per-dtype tolerances.
Correctness is a hard gate: only candidates that pass every case are timed.

**Level-3 tasks compute their reference in float32** (`reference_dtype`) while
the candidate runs in bfloat16. Composing ten operators makes a same-dtype
reference carry as much rounding error as the candidate, at which point the
tolerance stops meaning "how wrong is the candidate" and becomes a fudge factor.
The declared block tolerance is set from a measured noise floor, not chosen to
make tests pass: the tightest margin over the measured bfloat16-vs-float32
discrepancy is 2.1×.

Blocks also gate on float32 cases at small dimensions. A bfloat16-only gate
cannot separate a real algebra error — an RMSNorm that skips its float32 upcast,
say — from ordinary rounding, because both land at the same magnitude.

The promotion applies to the correctness path only. The eager-PyTorch
*performance* baseline runs through the same reference function, and timing it
at float32 against a bfloat16 candidate would not be a baseline at all.

### Inputs a backward may overwrite

Benchmark inputs are immutable by default, and the final-report protocol checks
it: a candidate that rewrites its inputs both skews repeated timings and skips
work the others do.

Writing a gradient over the activation that produced it is the exception worth
allowing. The gradient has the activation's exact shape, and under autograd the
activation is dead once the backward has read it, so a SwiGLU backward can skip
allocating two full tensors and write in place. Liger does this. Forbidding it
outright would exclude a real optimization from the benchmark.

A declaration therefore names the inputs its backward may overwrite:

```python
backward_may_overwrite=("a", "b")
```

Three properties make this an allowance rather than a loophole. It belongs to
the **operator**, so every candidate for that operator has it — a baseline
cannot enjoy it privately. It covers **contents only**: shape, strides, dtype
and storage offset are still enforced, because a reused buffer is still the same
buffer. And the suite report **lists every relaxation it ran under**, so a
reader knows which numbers were measured with it.

### Coverage

The fraction of tasks and of configurations that build, run, and pass
correctness. Reported separately at every level and never folded into the
speedup: an operator that fails on one shape out of five has 80% coverage, not
four shapes' worth of speedup.

### Performance

Full-step speedup — forward **and** backward:

```
S_i = T_reference_full_step,i / T_candidate_full_step,i
```

Aggregated geometrically: within an operator across its shapes, then within a
family, then across families.

Two choices worth stating.

**Full step, never backward-only.** The eager baseline's backward timing runs
through the oracle, which computes the forward *and* the backward, while the
candidate's backward is timed from pre-saved state with its forward outside the
timed region. For a single elementwise kernel the distortion is small; for a
level-3 block the forward is roughly a third of the step, so a backward-only
ratio is inflated by about half. The suite reads exactly one key,
`speedup_vs_baseline_raw_full_step`.

**Pooled by family.** Fourteen of the twenty-five operators are norms,
activations, and losses. A flat mean over operators would let whichever family
has the most declarations decide the headline number, so each family gets one
vote.

### Saved memory

Bytes of intermediate state the forward retains for the backward, reported
against the operator's declared memory inputs. This is the compute–memory
trade-off the benchmark exists to expose: a candidate may recompute in the
backward instead of saving, and at level 3 that choice spans a whole block.

Inputs that are not model state — integer class labels, routing indices,
attention masks, rotary tables — are excluded from the ratio via
`memory_inputs`, though a candidate remains free to save them.

## What the eager baseline runs

Every operator declares a `forward` that spells its mathematics out in
primitives. That is what the oracle differentiates and what AtenIR lowers into
an unfused seed, and both need it to stay primitive. It is the wrong thing to
**time** against: LayerNorm written as mean/sub/square/mean/rsqrt/mul/add
launches a dozen kernels and re-reads the row from HBM each time, where
`F.layer_norm` is one fused kernel. Timing a candidate against the primitive
spelling reports how much faster it is than a strawman.

Declarations therefore carry a second, optional reference:

```python
runtime_forward="...forward_ref:layernorm_runtime_ref"   # F.layer_norm
```

**The rule is: the eager baseline is the best implementation available in the
PyTorch version being used.** Where a fused `F.xxx` exists, it is used. Where
PyTorch has none — `dyt`, `poly_norm`, `sparsemax`, `jsd`, `tvd`, all recent
enough that no fused equivalent exists — the primitive spelling *is* the best
available, and timing against it is honest.

`verify_runtime_forward` checks the two agree numerically before any timing is
trusted. Without it the suite could time one function while checking the
correctness of another, and a faster-but-different baseline is indistinguishable
from a faster one.

### Why not match what HuggingFace runs

A tempting alternative is to baseline against the implementation a real training
stack executes — for `rmsnorm` that would be HuggingFace's `LlamaRMSNorm`, which
is still written in primitives today. Liger's own published numbers use that
comparison, and for a library it is the right one: it answers "what do you gain
by switching to us".

A benchmark answers a different question and must not adopt that baseline.
Measured on a GH200, the choice moves `rmsnorm` from **0.90x to 5.97x** — the
same kernel, a six-fold difference in headline, decided entirely by which
PyTorch spelling sits on the other side. Benchmarking against the weaker
spelling would flatter every submission, including future generated kernels,
and would not survive the first person who asks why `F.rms_norm` was not used.

The consequence is worth stating plainly: under this rule Liger's RMSNorm and
LayerNorm are **slower** than PyTorch's fused implementations on this hardware,
while its SwiGLU, RoPE and TVD kernels — which have no fused PyTorch
counterpart — remain genuinely faster. A benchmark exists to report that
distinction, not to protect any implementation from it.

## Baselines

| Baseline | Availability | Notes |
| --- | --- | --- |
| `pytorch_autograd` | every operator | Eager PyTorch through the declared forward reference. Always available, so it is the fallback for `--baseline auto`. |
| `liger` | 17 operators | Reviewed adapters around Liger-Kernel's shipped entry points. Selected by `auto` when available. |
| `torch_compile`, `torch_compile_max_autotune` | every operator | Built in, needs no declaration support. Never selected by `auto`: each case compiles a shape specialist, which costs real wall time. |
| `cublas_pair` | `matmul` | |
| `triton_tutorial` | `gemm_leaky_relu` | |

Every non-eager baseline is verified against the autograd oracle before its
timings are trusted. A miscompile or a mis-wired adapter would otherwise show up
only as a suspiciously good baseline.

Known gap: `evoattention` and `af3_single_repr_block` should be compared against
MegaFold's kernels and DeepSpeed's `DS4Sci_EvoformerAttention`, which is what
MegaFold itself compares against. Neither package is installed here, so neither
baseline is declared — shipping a baseline that has never been executed would be
worse than declaring none.

## Measurement protocol

Two harnesses, deliberately separate.

The **evolution harness** is low overhead: it is called thousands of times
inside the search, caches baseline timings, and reports medians.

The **final-report protocol** (`evograd fair-bench`, `evograd suite`,
`evograd-final-runtime-v1`) re-measures both providers under conditions designed
to make the comparison defensible:

- L2 cache cleared before every timed region
- batched CUDA events with a single synchronize, not one per sample
- provider order randomized
- inputs checked for mutation — content, shape, stride, dtype, and storage
  offset — outside the timed regions
- median of the retained samples
- `--identity-control` runs the baseline against itself, which must report ~1.0×

The first, second and last of those are exactly what `triton.testing.do_bench`
does, and therefore what KernelBench, TritonBench and FastKernels measure with.
The order randomization and the mutation check are additions; both are cheap.

**Every published number comes from this protocol.** `evograd suite` uses it by
default; `--protocol fast` selects the evolution harness for iteration only, and
any report produced that way says so in its header. The difference is not
cosmetic: measured on a GH200, the same operators re-run under the evolution
harness moved by up to 17% between runs — the drift concentrates in small
kernels, whose execution time approaches the timing overhead itself, while large
ones stayed within 0.1%.

`evograd fair-bench` additionally accepts `--blocks N` (default 3) and reports
block-bootstrap 95% confidence intervals across those repeats. The suite runs a
single block: the bootstrap resamples *blocks*, so with one block every interval
collapses to zero width, and paying three times the measurement cost for a
statistic the suite does not report would be waste. Use `fair-bench` when the
question is how much a specific difference is supported by the samples.

## Running it

```bash
# one operator
evograd bench --op layernorm --candidate best.py --baseline liger
evograd fair-bench --op layernorm --candidate best.py

# the whole suite, one candidate per operator under programs/
evograd suite --candidates programs/ --out results/
evograd suite --level 1 --level 2 --out results/     # restrict to a level

# the reference line: a reviewed pair baseline as the candidate, so the suite
# reports a number before anything has been generated for all 25 operators
evograd suite --candidate-baseline liger --out results/liger/
```

`evograd suite` writes `suite_report.json` and `SUITE_RESULTS.md`, with
per-operator, per-level, and overall speedup, coverage, and saved memory.

Correctness runs without a GPU:

```bash
evograd verify --op rope --device cpu candidate.py
```

Timing does not: the harness measures with CUDA events, and the declared shapes
are sized for a real device.
