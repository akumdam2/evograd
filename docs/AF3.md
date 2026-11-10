# AlphaFold3 in ADBench: tasks from the training step

AlphaFold3 is ADBench's scientific-ML workload. MegaFold (pinned commit
`845905e`) is the spine: code built on lucidrains/alphafold3-pytorch, with fused Triton
kernels, sequence-parallel code, and a config and trainer whose step fits on one GH200.
The aim is not a strict AF3 run. It is more sites than the earlier four, on MegaFold's
config (activation checkpointing, diffusion in chunks, a bf16 trunk), through its plain
PyTorch path.

That path is not upstream's. Against the closest upstream commit (`2ad09e98`, October
2024, v0.6.4, where about 68% of MegaFold's lines appear), MegaFold's triangle
multiplication has one biased `LayernormLinear` where upstream has two biasless
projections and no LayerNorm on the input, and its incoming equation swaps left and
right. Attention's output gate has a bias, with zero weights at init, and attention
dropout applies to the output, not the weights. Outer product mean divides after its
projection, not before. `Transition`'s default expansion is 4, not 2, and diffusion
augments inside its own step, in chunks. This comes from reading the source of the core
layers, not every line of `MegaFold.forward`, and upstream has not been run.

The earlier MegaFold tasks were chosen because MegaFold ships a hand-written
kernel for them. That criterion is retired. Tasks now come from what AF3's
training step computes, and MegaFold's fused kernels are only an expert baseline
at Tier 2/3, next to eager and `torch.compile`.

This page states every Level-1 and Level-2 task, its exact boundary, how dropout
and masks are handled, its precision, what overlaps what, and what is excluded.
Section 11 lists where this differs from the brief and what needs a decision.
Section 12 is the GPU checklist, with what has run. The CPU tests hold each contract
to a copy of the module it stands for (`tests/verification/megafold_transcription.py`),
and the GPU steps hold it to the real step.

## 1. Levels

| Level | Definition | Here |
| --- | --- | --- |
| L1 | a single op the step runs: normalization, activation, reduction, loss, matmul, attention as one call | bound to the shared generic tasks wherever the mathematics matches exactly |
| L2 | adjacent computations that form a meaningful optimization target in a model's training path, exposing fusion, removal of intermediate memory traffic and joint forward-backward choices. They need not tile a layer and may overlap | the contracts in section 4 |
| L3 | the whole MegaFold training step | unchanged |

## 2. Precision

`MegaFold.forward` opens `torch.autocast("cuda", dtype=bfloat16,
cache_enabled=False)`, and its parameters stay float32. The trunk (input
embedding, MSA module, pairformer) runs inside it. The diffusion module, the
distogram and confidence heads and the loss run with autocast off and their
inputs cast to float32 (`disable_*_casting`).

These are the op lists of the installed torch 2.12 (`ATen/autocast_mode.h`)
that matter here:

| op | under bf16 autocast |
| --- | --- |
| `layer_norm` | inputs cast to float32, result float32 |
| `softmax`, `sum` | input cast to float32, result float32 |
| `linear`, `einsum`, `matmul`, `bmm` | inputs cast to bfloat16, result bfloat16 |
| `sigmoid`, `glu`, `silu`, `relu`, `masked_fill`, `dropout`, `+`, `*`, `/` | not listed: run in their inputs' dtype, and a mixed binary op promotes to float32 |

Two consequences no precision-uniform contract states:

- `OuterProductMean` returns **float32** in the bf16 region, because it divides
  a bf16 result by a float32 MSA count. The pair representation is float32 from
  there on.
- `MSAPairWeightedAveraging`'s softmax is float32 and is cast back to bf16 only
  by the einsum's autocast rule. There is no explicit `.to(dtype)` as in `Attend`.

**Case dtype.** A case's dtype is the dtype of its GEMMs: `bfloat16` under
autocast, `float32` outside it. Inputs that enter a LayerNorm first are float32
at the boundary, so the adapter makes autocast's cast, as
`megafold_layernorm_linear` already does. Weights arrive in the GEMM's dtype.

**The dtype each result is computed through** is recorded per contract in
`benchmark/topdown/megafold/precision.py`. For example `dx` of the bf16
LayerNorm+Linear case is stored float32 but passes through a bf16 GEMM. Shared
tolerance code is not changed.

**A contract declares no tolerances.** Gate 1 uses the dtype defaults and the compile
envelope, calibrated when a task pack is built, and Gate 2 and Gate 3 calibrate on their
own controls. No check runs outside a pack or on generated inputs.

**Still open: results computed through bf16 but stored in float32.** Gate 1 picks its
default by the stored dtype, so those results get float32's tight rule plus the compile
envelope. `precision.py` records which results these are, and no gate reads it yet. The
envelope has been enough so far. If a correct kernel fails on such a result, the shared
tolerance code should read that table.

## 3. Rules every task follows

**Reference.** The reference is what MegaFold's eager tree computes at that
boundary, in MegaFold's precision, spelled as in
`benchmark/topdown/megafold/contracts.py`. Autocast casts an input once per op,
and a reference keeps that: where one float32 tensor feeds several GEMMs, its
gradients then meet in float32, and casting it once would round them differently.

**The spelling matters bit for bit.** `F.linear` dispatches on rank and
contiguity. With a bias, a rank-2 or contiguous rank-3 input takes the fused
`addmm`, and anything else takes `matmul` then `add`. Under grad a non-contiguous
input folds to one `mm`, under `no_grad` it does not. So:

- a contract keeps the rank the module has wherever a biased GEMM is involved;
- where MegaFold's input is a transposed view, the reference spells the path
  training takes (`F.linear(t.contiguous(), w)`, then the bias if there is one), so
  the result is the same under grad and in a no-grad replay;
- MegaFold's `pack_one` is a copy, not a view: einops concatenates, so `ending`'s
  rows are contiguous and its biased `to_q` and `to_gates` take `addmm`, where a
  reshape at a batch of one would stay a view and take `matmul` then `add`.

CPU rounding agrees on those two paths and the GPU's does not, so a wrong path
passes every value comparison on a CPU. The first site check on the GPU found one
at `ending`, as a loss 2e-6 away. The contract tests compare the paths themselves.

**Cases.** One case per observed configuration (dims, dtype, layouts, static
parameters), harvested from one real step, with calls per step recorded.

**Equal shapes, different roles.** A role that no shaped argument separates gets
its **own task key and its own site**, sharing one contract module. This covers
outgoing/incoming and starting/ending. See
section 11 for why this is not a scalar `mix` argument. The same module in the
MSA module's and the pairformer's pairwise blocks is one role, as in the brief's
region list, and its calls per step count both.

**Sequence parallelism.** `gather`, `scatter`, `gather_async`,
`reduce_scatter_sum`, `row_to_col` and `col_to_row` return their input at
`sp_size == 1`. References omit them and every adapter asserts `sp_size == 1`.

**Dropout.** `Dropout` inside a boundary becomes an Inactive input `keep`: the
scaled mask MegaFold drew, `0` or `1 / (1 - p)`, in the case dtype, shaped
`[B, 1, N, D]` for row dropout and `[B, N, 1, D]` for column dropout. A deployed
adapter draws it with the module's own `nn.Dropout` on `new_ones` of the same
shape and dtype, exactly as `Dropout.forward` does, so the step consumes RNG as
the plain step does and every provider applies the same mask. No other RNG is
consumed inside any boundary, so drawing before the kernel changes nothing.

`Attend`'s post-aggregation dropout stays disabled for every provider, as it
already was (`build_patched`). It is the workload's own pin, the warm start was
trained with it off, and lifting it would change L3. See section 11.

**Calls.** Only training calls are recorded: grad enabled and outside a backward
pass, so never an activation-checkpoint recompute or the confidence head's
`no_grad` rollout.

## 4. Level-2 tasks

Five contract modules under `benchmark/topdown/megafold/levels/level2/`, plus the
two kept ones: seven operations and nine task keys, of which eight keys are searched
(`megafold_layernorm_linear` is not). Section 11 says why it is these. All shapes are as the kernel receives them.

Regions below: *trunk* is bf16 (MSA module and pairformer); *confidence* is the
confidence head's pairformer stack, float32; *diffusion* is float32. Calls per
step are from the tracked snapshot's module counts and the harvest replaces them.

### 4.1 `triangle_multiplication`

Task keys `megafold_triangle_multiplication_outgoing` and `_incoming`, one site
each. Regions: trunk (52 calls per mix), confidence (4 per mix).

Boundary: `TriangleMultiplication.forward`. The residual add is outside.

```
y         = layer_norm(x, ln_weight, ln_bias)            float32
combined  = linear(y.to(D), linear_weight.T, linear_bias)   D  [B, N, N, 5C]
lr, gate  = combined[..., :4C], sigmoid(combined[..., 4C:])
left, right = glu(lr, dim=-1).chunk(2, dim=-1)
left, right = left * pair_mask, right * pair_mask        pair_mask = mask[:, :, None] & mask[:, None, :]
mixed     = einsum(left, right)                           D
              outgoing: "... i k d, ... j k d -> ... i j d"
              incoming: "... k i d, ... k j d -> ... i j d"
hidden    = layer_norm(mixed.float(), out_ln_weight, out_ln_bias)  float32
out       = linear(hidden.to(D), out_linear_weight.T) * keep * gate   D
```

Args: `x [B, N, N, C]` float32; `ln_weight, ln_bias [C]` float32;
`linear_weight [C, G]`, `linear_bias [G]` in D (`G = 5C`); `out_ln_weight,
out_ln_bias [C]` float32; `out_linear_weight [C, C]` in D; Inactive `mask [B, N]`
bool, `keep [B, 1, N, C]` in D (row dropout, both mixes); scalar `eps`.
Output `out [B, N, N, C]` in D. MegaFold's `dim_hidden` must equal `dim`, because
`to_out(...) * out_gate` does not broadcast otherwise, so the contract has no
separate hidden width.

The biased `combined` projection was the reason this boundary could not use
`megafold_layernorm_linear`.

### 4.2 `triangle_attention`

Task keys `megafold_triangle_attention_starting` and `_ending`, one site each.
Regions as 4.1. The site is the `PreLayerNorm` wrapping `TriangleAttention`. The
residual add is outside.

```
y      = layer_norm(x, ln_weight, ln_bias)          float32
ending: y = y.transpose(1, 2)
bias   = linear(y.to(D), bias_weight)  [B, n, n, H] -> [B, H, n, n]
seq    = y.reshape(B * n, n, C)
q, k, v, gate = projections of seq.to(D)            D
o      = Attend's default path on q, k, v with bias repeated over n and key mask mask
o      = linear(o * sigmoid(gate), out_weight)      D
ending: o = o.transpose(1, 2)
out    = o * keep
```

Args: `x [B, N, N, C]` float32; `ln_weight, ln_bias [C]` float32; `bias_weight
[A, C]`, `q_weight [I, C]`, `q_bias [I]`, `kv_weight [W, C]`, `gate_weight [I, C]`,
`gate_bias [I]`, `out_weight [C, I]` in D (`A` heads, `I = A * d_head`,
`W = 2I`); Inactive `mask [B, N]` bool, `keep` in D: `[B, 1, N, C]` starting
(row), `[B, N, 1, C]` ending (column); scalar `eps`. Output `out [B, N, N, C]` in D.

`ending`'s transposed `y` is the non-contiguous input of section 3 for `to_attn_bias`
alone. `pack_one` then copies it to rows, so `to_q` and `to_gates` take the fused path
at both nodes.

### 4.3 `transition`

Task key `megafold_transition`. Boundary: `FusedTransition.forward` with its
LayerNorm, residual add outside. Headroom kept it: at 1.055x it has the third
largest ceiling of the seven sites (section 11).

```
y          = layer_norm(x, weight, bias)       float32
h          = linear(y.to(D), linear_weight.T)  D  [M, W]
first, g   = h.chunk(2, dim=-1)
out        = linear(silu(g) * first, out_weight)  D  [M, K]
```

Args: `x [M, K]` float32, leading axes flattened, `weight, bias
[K]` float32, `linear_weight [K, W]` in D (MegaFold's `[K, N]` layout),
`out_weight [K, I]` in D, scalar `eps`. Regions: pair (52 trunk, 4 confidence),
single (48 trunk, 4 confidence), MSA (4 trunk), pairwise and single conditioning
(24 each, diffusion).

### 4.4 `outer_product_mean`

Task key `megafold_outer_product_mean`. Region: MSA module, 4 calls, bf16 only.
Boundary: `OuterProductMean.forward`, residual add outside. The MSA mask branch
is the only one the step takes, and the adapter refuses `msa_mask=None`.

```
h       = layer_norm(x, weight, bias)            float32
a, b    = linear(h.to(D), linear_weight.T).chunk(2, dim=-1)   D
a, b    = a * msa_mask[:, :, None, None], b * msa_mask[:, :, None, None]
o       = einsum("b s i d, b s j e -> b i j d e", a, b)   D
o       = o.flatten(-2) * pair_mask[..., None].to(D)
p       = linear(o, pair_weight, pair_bias)      D
out     = p / msa_count[:, None, None, None].clamp(min=count_eps)   float32
```

Args: `x [B, S, N, M]` float32; `weight, bias [M]` float32; `linear_weight
[M, W]` in D (`W = 2H`); `pair_weight [P, Q]`, `pair_bias [P]` in D (`Q = H * H`);
Inactive `mask [B, N]` bool, `msa_mask [B, S]` in D, `msa_count [B]` float32;
scalars `eps`, `count_eps`. Output `out [B, N, N, P]` **float32** (section 2).

`msa_count` is computed once in `MSAModule.forward`, outside the boundary, by a
`sum` autocast runs in float32, so it is an input. It is a statistic of the real
structure, and a task pack replays it as recorded.

### 4.5 `msa_pair_weighted_averaging`

Task key `megafold_msa_pair_weighted_averaging`. Region: MSA module, 4 calls,
bf16 only. Boundary: `MSAPairWeightedAveraging.forward`, residual add outside.

```
b       = linear(layer_norm(pair, pair_ln_weight, pair_ln_bias).to(D), pair_linear_weight.T)
          -> [B, A, N, N], masked_fill(~mask, -finfo(D).max)
weights = softmax(b.float(), dim=-1)             float32
values, gates = linear(layer_norm(x, ln_weight, ln_bias).to(D), linear_weight.T) -> two [B, A, S, N, d]
out     = einsum("b h i j, b h s j d -> b h s i d", weights.to(D), values) * sigmoid(gates)
out     = linear(merge_heads(out), out_weight) * keep    D
```

Args: `x [B, S, N, M]` float32; `ln_weight, ln_bias [M]` float32; `linear_weight
[M, W]` in D; `pair [B, N, N, C]` float32; `pair_ln_weight, pair_ln_bias [C]`
float32; `pair_linear_weight [C, A]` in D; `out_weight [M, I]` in D; Inactive
`mask [B, N]` bool, `keep [B, 1, N, M]` in D (row dropout); scalar `eps`.
Output `out [B, S, N, M]` in D.

### 4.6 Kept: `megafold_layernorm_linear`, `megafold_evoattention`

Both stay valid composites. Rechecked against pinned MegaFold source:

- `megafold_layernorm_linear` is `LayernormLinear`'s torch path. Sites hold the
  no-linear-bias instances. The biased `combined` projection is now inside 4.1.
- `megafold_evoattention` is `Attend.forward`'s default path, including `sim +
  attn_bias`, `max_neg_value`, the float32 softmax cast back, and `q * scale`.
  It only applies with `use_optimized_evo=None`, `scale=None`, no softclamp and
  no memory key/values, and `_PatchedAttend` refuses anything else.

Item 8 of the brief kept both, though the criterion that chose them is retired.
Evoattention earns it: `Attend` is upstream's, it is the only Level-2 task reaching
pair-bias and diffusion attention, 340 calls, and it is the one kernel that has paid off.
`megafold_layernorm_linear` left search on 2026-11-06 (section 11). It stays registered
and recorded, so both corpora still replay.

## 5. Level-1 tasks

The harvest records every primitive training call through a
`TorchFunctionMode` (section 7) and files it under a task by this table.
`linear` and `linear_no_bias` are library-only: their cases are counted in the
inventory and recorded as the other models do, and the evaluation plan leaves
them out of search. A biased GEMM on a rank-4 or non-contiguous input (the
`combined` and pair projections) takes `matmul` then `add`, where the task's rows take the fused `addmm`. Those calls are not `linear`'s
rounding, so they stay in the snapshot's `unbound_configurations`, counted, with no
case. The same goes for any call a task does not describe, such as a softmax over
another axis.

| primitive | task | status |
| --- | --- | --- |
| `layer_norm` with weight and bias | `layernorm` | shared |
| `layer_norm` with neither | `layernorm_no_affine` | new generic |
| `layer_norm` with weight only | `layernorm_no_bias` | new generic |
| `silu(a) * b` at `SwiGLU` | `swiglu` | shared |
| `glu` | `glu` | new generic |
| `sigmoid` | `sigmoid` | new generic |
| `softmax` | `softmax` | shared |
| `relu` | `relu` | new generic |
| `linear` with a bias | `linear` | shared, library-only |
| `linear` without a bias | `linear_no_bias` | shared, library-only |

Everything else the step calls is counted per step in the snapshot's `census`
and classified there as either scaffolding or an unbound operator with its
reason, and each unbound operator's configurations are listed. A name the
classification does not know fails the snapshot rather than being dropped, so
anything new the harvest finds forces a decision. The harvest file is written
first, so that decision needs no second GPU run (`--from-harvest`).

Unbound operators, from reading the source:

| op | why not bound |
| --- | --- |
| `einsum` | batched contractions, not `matmul`'s `[M, K] @ [K, N]`. Every trunk einsum is inside an L2 contract. `matmul` is a granted vendor primitive, so a bound case would be inventory-only anyway. Counted per equation |
| `matmul` (`@`) | rigid-frame and coordinate arithmetic on `[.., 3]` tensors, batched |
| `cross_entropy` | MegaFold's calls are `reduction="none"` or `ignore_index=-1`; the generic task is mean-reduced with `-100`, so the mathematics differs |
| `mse_loss`, `cdist`, `pairwise_distance`, `normalize` | loss and geometry on coordinates, on small shapes, with no generic task. Candidates for later L1 tasks, not needed by any L2 task here |
| `embedding` | an index lookup with no arithmetic to optimize |
| `svd`, `det`, `cross` | rigid alignment and frames on coordinates, a handful of calls per step, with no generic task |

## 6. Overlaps

`patch_modules` did not refuse nested sites, and neither did the combination:
`verify_combination` deploys every accepted Level-2 candidate. Installing the
outer site would silently drop an inner candidate, which would never be called.

`SiteRegistry` now carries a declared `overlaps` relation and `patch_modules`
raises `OverlappingSites` on any requested pair in it, before touching the tree.
Every route into a model goes through `patch_modules`, so a nested pair is never
installed. The pairs are `MEGAFOLD_SITE_OVERLAPS`, declared with the site table:

| outer site | sites inside it |
| --- | --- |
| `triangle_multiplication_*` | `layernorm_linear` |
| `triangle_attention_*` | `layer_norm`, `triangle_attention` |
| `transition_block` | `layernorm_linear`, `transition` |
| `outer_product_mean` | `layernorm_linear` |
| `msa_pair_weighted_averaging` | `layernorm_linear` |

Among the searched Level-2 cases no pair overlaps. The five that did were all
`layernorm_linear` against a composite, and at site level only: its searched case,
`diffusion_module.pairwise_repr_to_atompair_feat_cond`, is inside none of them.
Evoattention's searched case routes to `pair_bias_attention`. Its two triangle cases are
inside triangle attention, so searching those needs section 11's rule.

## 7. Capture

The old recorder installed every site's adapter with a reference kernel. That
cannot work with nested sites: an outer kernel hides every call inside it, so a
transition site would record no LayerNorm, GEMM or SwiGLU.

Capture and harvest now use **pass-through recorders**, as the other models do:
forward hooks at L2 boundaries and a `TorchFunctionMode` for primitives. Nothing
is replaced, so each level sees every call and the arithmetic does not change.
An L2 recorder hands `site.arguments` the live module and inputs, so it records
exactly what a deployed adapter would pass. `keep` is read from a hook on the
module's own `nn.Dropout`, so a recorder draws nothing.

That splits the old single check into three, and together they say more:

1. **The capture loop's bit-for-bit check** (shared): the instrumented step
   equals the plain step. With pass-through recorders this proves recording is
   harmless and no longer proves the adapters.
2. **`fidelity`** (shared): every recorded call's reference reproduces the live
   output's digest bit for bit. This is the check on each contract, now per call
   and per task rather than one verdict for a whole step.
3. **`site_check`** (new, `integrations.megafold.site_check`): the deployment
   adapters with reference kernels installed give a step bit-identical to the
   plain one: loss, primary output and every gradient. Sites are grouped into
   non-overlapping sets (`SiteRegistry.groups`, two groups today), one model per
   group. This is the old capture check, and it also proves the adapters consume
   RNG as the plain step does.

`expected_site_calls` used to patch every site at once, which `OverlappingSites`
now refuses, so it counts forward calls of the matched modules with hooks instead.

## 8. Gates

MegaFold's gates are as they were. Gate 3 does not apply (`gate3_applies=False`), so a
Level-2 candidate and the combination are accepted on model Gate 2, whose margin is the
protocol's default. Nothing here calibrates a gate: Gate 3's settings for this model and
its margins are not decided, and no shared gate or tolerance code is changed.

## 9. Sites

One `ModulePatch` per site, and each matcher accepts MegaFold's spelling and the
eager one, as before. `NOT_SITES` still applies: the template embedder and the
recycling projections get no gradient and hold no site. Adapters keep the
original's submodules under their original names, so a gradient compared by
parameter name finds the same parameter in every tree.

| site | task | level |
| --- | --- | --- |
| `layer_norm` | `layernorm` | 1 |
| `transition` | `swiglu` | 1 |
| `layernorm_linear` | `megafold_layernorm_linear` | 2 |
| `pair_bias_attention`, `triangle_attention` | `megafold_evoattention` | 2 |
| `triangle_multiplication_outgoing`, `_incoming` | 4.1 | 2 |
| `triangle_attention_starting`, `_ending` | 4.2 | 2 |
| `transition_block` | 4.3 | 2 |
| `outer_product_mean` | 4.4 | 2 |
| `msa_pair_weighted_averaging` | 4.5 | 2 |

`transition` keeps its name so that the tracked snapshot stays valid until it is
regenerated. Its site note files the old cases there.

## 10. Excluded

| what | why |
| --- | --- |
| attention with pair bias as a whole, the transition without its LayerNorm, AdaLN, the adaLN-zero gate | cut on 2026-11-05, section 11. They stay in the model and run stock: not sites, not tasks. Their primitives are still recorded at L1 |
| windowed atom attention | `Attend.local_attn` is a different computation: it windows the keys, the bias and the mask, follows the atom count, and takes a windowed pair representation built outside the module |
| template embedder, recycling projections | no gradient in this step (`NOT_SITES`) |
| `Attend`'s post-aggregation dropout | off for every provider, section 3 |
| relative position encoding, Fourier embedding, atom-to-token pooling, atompair MLPs, distogram and confidence output heads | single linears, ReLUs and table lookups; their ops are in L1, and no adjacent pair is a fusion target worth a contract |
| losses and alignment | geometry on coordinates; section 5 |
| calls under `no_grad` or in a backward pass | not training calls |

## 11. Differences from the brief, and open decisions

**Scope was cut to seven operations on 2026-11-05**, nine task keys. Cutting means
a computation is no longer a modifiable kernel in the workload. It stays in the model
and runs stock. The evaluation plan searches MegaFold on one case per task, so each
key adds a search, and each added case is 6 agent searches of up to 100 minutes. The
earlier campaign (`results/megafold-level4/notes.md`) says which tasks can pay for that:

- The step is about 11 s and bound by host-side submission of roughly 10,000 kernel
  launches (`cpu_bound_fraction` about 1.0), not by GPU time. MegaFold's fused kernels
  win mainly by launching fewer kernels.
- Patching costs per call: the identity control cost about 1% at a 100-call site and
  12 to 23% at the ~1000-call LayerNorm+Linear site.
- Attention is the only kernel that paid off in the model: the evolved program at the
  triangle site alone reached 1.21x eager, level with MegaFold's whole tree. LayerNorm
  bought 2%, SwiGLU lost 2%, and LayerNorm+Linear lost 34%.

So a task has to remove launches without being called too often. In priority order
the seven are `megafold_evoattention`, `triangle_attention`,
`triangle_multiplication`, `outer_product_mean`, `msa_pair_weighted_averaging`,
`transition` and `megafold_layernorm_linear`. `megafold_layernorm_linear` stays
registered with no search proposed.

**Headroom, measured 2026-11-05** (`scripts/megafold_site_headroom.py`, 384 residues,
native step 8.23 s). One site at a time ran a stub that returns cached outputs and
zero gradients, so each figure is a ceiling on what any kernel there could buy, not a
prediction of what one will.

| site | ceiling | share of the step |
| --- | --- | --- |
| `triangle_attention_ending` | 1.104x | 9.4% |
| `triangle_attention_starting` | 1.071x | 6.6% |
| `transition_block` | 1.055x | 5.2% |
| `msa_pair_weighted_averaging` | 1.047x | 4.5% |
| `triangle_multiplication_incoming` | 1.040x | 3.8% |
| `triangle_multiplication_outgoing` | 1.039x | 3.8% |
| `outer_product_mean` | 1.036x | 3.5% |

Together they are 36.8% of the step, about 1.58x if all seven cost nothing. `transition`
ranks above the three operations listed ahead of it after attention, so it stays. Its
160 calls per step pay more for patching than `outer_product_mean`'s 4, which the
identity control measures. Its stub also dropped the peak to 34.8 GiB from 53.7, where
no other site moved it by more than 1.4 GiB. The cause has not been looked at.

**Tier 3 reference line, measured 2026-11-06** at `0d1a761` (384 residues, 10 steps in 3
blocks): eager 8.34 s, MegaFold's tree 7.30 s (1.142x, 95% interval 1.129 to 1.150),
compile 9.32 s (0.895x). The campaign's figures were 11.3 s and 1.213x, and the difference
has not been looked at. Identity at the first site group, the five inner sites, is
9.52 s (0.876x). That provider recomputes each forward in its backward, so 12.4% is an
upper bound on patching cost there. Identity at the seven composite sites is **not
measured**: the job was cancelled 15 minutes into that run, so the composite sites'
patching cost is still an estimate from the per-call figures above. Peak memory was 53.7
GiB for eager and 44.0 GiB for MegaFold's tree. Its loss differs from eager's by 0.023,
where compile's differs by 3.5e-5, so that tree is a speed comparison and not a correct
alternative. These figures and the headroom above use random weights with the confidence
head off, where the corpus uses the warm start with it on.

Cut, against items 4 and 7 of the brief and the token-transformer key added here:
`attention_pair_bias` (its core is `megafold_evoattention`, and pair-bias attention
added about 1% on top of the triangle site), the transition without its LayerNorm
(two GEMMs around a SwiGLU, 363 calls), AdaLN (726 calls) and the adaLN-zero gate (726
calls, one GEMM and a gate). None of the four was ever timed as a composite, so this
is inference. `attention_pair_bias` is the least certain cut, since the token
transformer is the busiest attention in the step. Their contracts are in git history
at `04532c3` if a measurement argues otherwise.

**Search cases, 2026-11-06: sixteen,** one per task, each its most-called configuration:
eight L1 (`linear` and `linear_no_bias` are granted vendor primitives) and eight L2 keys
over six operations. `megafold_layernorm_linear` is out of search. It is MegaFold's own
fusion and not in upstream, it reached 0.66x in the campaign, and every clash among the
searched cases was its. The evaluation plan and the shared `cases.json` do not list these
cases yet. One case per
operation would be six L2 cases, which means changing how the plan picks them.

**Static settings are task keys, not a scalar argument.** The brief says "static
mix" and "static node". A site tells its calls apart by shaped arguments only
(`verification/common/dispatch.py`), and two sites of one task by dims and dtype
(`gates.site_for_case`). Outgoing and incoming agree on every one of those, so
`by_configuration` raises and a candidate could not be deployed. Fixing that
means changing shared gate code, so each setting has its own key over one
contract and reference. Provenance still records the setting.

**`gates.py` does not choose among nested candidates.** `verify_combination` and
`evaluation/protocol.py` pass every accepted (or every faster) Level-2 candidate.
With nested candidates, `patch_modules` now raises `OverlappingSites`: nothing
wrong is installed and per-candidate reports are kept, but the run stops before
`verdict.json` and the combination has no result. Choosing which nested candidate
to keep is a one-line selection in shared code, left out because of the
no-shared-gate-changes rule. No searched pair overlaps now (section 6), so this waits
until evoattention's triangle cases are searched. **Decision needed** before then.

**Attention dropout.** Keep masks would now make it possible to turn `Attend`'s
dropout back on. Doing so changes L3 and invalidates the warm start, so it stays
off, and `eager` still differs from MegaFold's own training in this one respect.

**Seeds have not run on a GPU.** The unified frontend had no lowering for `aten.unbind`
(unpacking a tensor in `msa_pair_weighted_averaging`) or `aten.glu` (both triangle
multiplication keys). A CPU probe at fixture size found both, in both dtypes, and the
other six Level-2 keys and all ten Level-1 tasks lowered. `085e245` respelled those
contracts. Lowering only shows that code was produced: real shapes are where earlier
seeds failed (`results/megafold-level4/notes.md`), and whether the 12,097-row MSA cases
fit unified's 180 s verification limit is unknown. Frontend A, which the plan tries
first, is untested on these tasks.

**LayerNorm+Linear and evoattention cases merge roles at equal shapes** (for
example `to_attn_bias` and `pairwise_repr_to_atompair_feat_cond`, float32). They
stay merged because they are one computation at one site. Splitting them would
need a task key per role over 23 configurations.

## 12. GPU checklist

Steps 0 to 4 ran on 2026-11-05 and 6, at `0d1a761`. Step 2 passed: every site group
gives MegaFold's plain step bit for bit, loss, distogram logits and all 3,800 gradients.
Step 3's four orders each stayed bit-identical while recording. Step 4 passed: all 2,532
payloads reproduce the live output bit for bit, on every task, and a read-only replay
of the old corpus matched on all 440 of its payloads.
CPU rounding is not CUDA's, so steps 2 and 4 are what prove the references, and step 2
is what found the `pack_one` copy of section 3. `085e245` respelled three tasks'
contracts after these runs. Step 2 passed again at that commit, and step 4 on those tasks.

Run jobs from a worktree frozen at a commit, not from the checkout. Every import reads
the tracked snapshot, so one rewritten during a job reaches the steps after it, and a
bad one stops every import, the harvest's own included.

`ghx4-interactive` allows one submitted job per user and 2 h. A limit below 2 h is what
gets a job started: a job estimated for the next morning started within minutes at 1 h,
and another estimated 1.7 h out at 1.5 h. `ghx4` estimated 8 days. Capture cannot
resume, so give it its own job with room to spare.

```bash
#SBATCH -A bcjw-dtai-gh --partition=ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=192g --time=02:00:00
REPO=/u/akumdam/evograd                 # on af3-revamp
OUT=/work/hdd/bcjw/akumdam/af3_revamp   # new; nothing is written under wzhan
PY=/u/akumdam/envs/af3-conda/bin/python
export PYTHONPATH=/u/akumdam/MegaFold:$REPO/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TYPECHECK=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CXX=g++ CC=gcc
export TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID
cd $REPO && mkdir -p $OUT
```

The pinned pool and warm start are read, by hash, from their default paths under
`/work/hdd/bcjw/wzhan/ADBench_payload/megafold`. Nothing writes there.

```bash
# 0. The GPU-gated tests, new sites included.
$PY -m unittest tests.test_tier3_megafold 2>&1 | grep -E "^(Ran |OK|FAILED)|^(ERROR|FAIL):"

# 1. Regenerate the snapshot: one step with the recorder in harvest mode.
$PY -m evograd.benchmark.topdown.megafold.harvest.harvest --out $OUT/harvest.json --write-snapshot
#    If it stops on an unclassified name, the harvest is already saved: bind the name
#    or add it to UNBOUND or SCAFFOLDING in harvest.py, then, on a login node:
#    PYTHONPATH=src $PY -m evograd.benchmark.topdown.megafold.harvest.harvest \
#        --from-harvest $OUT/harvest.json --write-snapshot
#    Then run the CPU tests and commit harvest/snapshot.json.

# 2. The bit-for-bit check of the adapters: two groups of sites against the plain step.
$PY -m evograd.verification.integrations.megafold.site_check --out $OUT/site_check.json

# 3. Record a corpus to a new path. The capture loop checks the recorder is harmless.
$PY -m evograd.verification.integrations megafold capture --out $OUT/corpus

# 4. The bit-for-bit check of every reference, per task, on every recorded call.
$PY -m evograd.verification.integrations megafold fidelity \
    --corpus $OUT/corpus/manifest.json --out $OUT/fidelity.json
```

Measured on a GH200 under the deterministic settings. The recorder turned out cheap: the
harvest took about 2.5 min in all, most of it startup.

| step | runtime | GPU peak | disk |
| --- | --- | --- | --- |
| 0 tests | 251 s | | |
| 1 harvest | about 2.5 min | | 13 MB |
| 2 site_check | 2.7 min, 3.4 with a group rerun by site | | |
| 3 capture | 64 min (4 orders): about 30 recording, 34 hashing | | 355.5 GiB |
| 4 fidelity | 59 min, plus 14 for the old corpus | | |

The corpus holds 493 GiB of tensors over the 239 cases, 355.5 GiB of it on disk in 99
shards, because 137.6 GiB of identical storage is written once. That is about 89 GiB per
order, where the old recording kept about 28. The large tensors are the MSA module's and
softmax's, and `recorded_indices` already keeps one call instead of three above 2 GiB.
`call_bytes` predicted 480 GiB, which is close to the total and does not see the sharing.
On 2026-11-05 `/work` had 1.4 TB free and the `bcjw` allocation stood at 13.74 of its
14.65 TiB soft quota, so check both first. `/u/akumdam` cannot hold it (103 GB quota).

Capture's process peaked at 173 GiB resident. Slurm's MaxRSS says 192 GiB because it
counts page cache, so `--mem=192g` was reached and held by reclaiming that cache.
Hashing reads every shard back at about 0.18 GiB/s, so it costs as much as recording.
Fidelity peaked at 79 GiB of host memory.

**What to look at, and what each failure means:**

- *Step 2 differs.* An adapter or reference is wrong. A group that differs reruns one
  site at a time, so the report names the site. A loss a few parts in a million away
  is a GEMM path (section 3), as it was at `ending`; an RNG slip moves it far more.
- *Step 4 mismatches.* None so far: 2,532 of 2,532, and 440 of 440 on the old corpus.
  `by_task` names the contract, and the likeliest cause is a GEMM path, `addmm` against
  `matmul` then `add` (section 3), which only CUDA's rounding shows. Step 2 settled
  `triangle_attention_ending`'s. The step compares outputs only, through
  `runtime_forward`, so gradients stay step 2's to prove.
- *Step 3 raises `training calls per case differ`.* The snapshot and the step
  disagree, so the step is not reproducible between harvest and capture.
- *An adapter raises in a real step.* `outer_product_mean` refuses `msa_mask=None`.
  That is an assumption read from the source, and the run proves or breaks it.

**Step 5, task packs and seeds.** Packs took about 15 minutes for 17 cases, each calibrated:

```bash
$PY candidates/bench.py tasks --model megafold --root $AD --case-id $(cat cases.txt) \
    --corpus $OUT/corpus/manifest.json --state $AD/state
```

Three settings matter. `TORCHINDUCTOR_COMPREHENSIVE_PADDING=0` is needed for a seed's
capture: with padding, full-size `msa_pair` fails a stride assertion, 387104 rounded to
387136 by 128-byte alignment at bf16. A seed has to be judged with
`EVOGRAD_SEARCH_CORRECTNESS` naming its pack's `task.json`, as `candidates/_common.py` sets
it: outside a pack the declared tolerances rejected six seeds the pack's rule accepts.
And `--root` takes the parent, so packs land in `$AD/megafold/tasks`.

Fifteen of the sixteen searched cases have a unified seed at real size. `msa_pair` has
none. Two memory faults came first, and both are fixed: the parent held 24 GiB of the
capture while the child verified, and a generated graph kept every intermediate until it
returned, 125 live storages and 93.6 GiB in the backward. What remains is a size limit.
`hidden` has 4645248 x 512 = 2.38 billion elements, and the primitive library indexes in
32 bits, `pid * BLOCK + tl.arange(0, BLOCK)`, in 110 kernels. Past 2^31 the index wraps.
The seed gives non-finite gradients and wrong outputs, then illegal memory accesses. No
other searched case is over the limit. That cause is inferred from the size and the
symptoms and has not been isolated. Widening the kernels changes every model's seeds, and
a contract cannot avoid it, because the size comes from the projection itself. An agent
that writes its own kernel is not affected, and neither is frontend A's seed if it uses
PyTorch or vendor calls at that size.

The shared `cases.json`, the evaluation plan and the task packs are wzhan's, and are not
changed here.
