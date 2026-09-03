# Level 4: the Qwen3-0.6B training workload

Levels 1-3 benchmark declared operators. Level 4 is one whole model executed the
way training executes it, and it exists so that a later stage can *observe* a
real training step rather than infer what one contains.

This first milestone establishes only that the execution works and records
exactly what it was. It does no harvesting, no operator extraction, no provider
comparison, and no benchmarking.

## The canonical workload

| field | value |
| --- | --- |
| model | Qwen3-0.6B (596M parameters, 440M non-embedding), randomly initialised |
| batch size | 2 |
| sequence length | 2048 |
| tokens per step | 4096 |
| dtype | bfloat16 |
| device | cuda |
| attention backend | sdpa |
| mode | `model.train()` |
| `use_cache` | False |
| gradient checkpointing | disabled |
| seed | 0 |
| workload id | `qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad` |
| config hash | `6e254de2e1abf1a0` |

The executed step is exactly:

```python
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

No optimizer step, and no gradient zeroing between runs -- one process, one step.

`labels = input_ids.clone()`, and `input_ids` are synthetic, drawn from an
explicit CPU generator seeded with the workload seed so the token stream is
identical on any device.

No pretrained weights and no tokenizer are downloaded. The architecture is
written out in `spec.py` rather than fetched, because a reference execution must
not depend on what the Hub served that day, or on a node having network access.

## Running it

```bash
conda activate evograd
PYTHONPATH=src python -m evograd.bench.workloads.qwen3 \
    --out results/qwen3-level4/canonical.json
```

Exit code 0 means the step ran and every check passed; 1 means it did not, and
the report says why. With no `--out` the JSON goes to stdout.

`--print-spec` resolves the workload id and hashes without running anything.

### Debug variants

`--batch-size`, `--seq-len`, `--dtype`, `--device`, `--attn`, `--seed` and
`--layers` shrink or shift the workload. Any of them changes the workload id,
sets `"canonical": false` in the report, and prints a warning to stderr. None of
them can enable the KV cache or gradient checkpointing -- those are refused by
the spec, not by the parser, so no command-line flag can produce a run that
claims to be this workload while executing a different graph.

## What the report contains

`results/qwen3-level4/canonical.json`, schema `evograd-qwen3-smoke/1`:

- `workload` -- id, workload hash, `canonical` flag, model name, the full
  architecture config and its hash, batch/sequence/token counts, dtype, device,
  requested attention backend, cache and checkpointing state, seed
- `environment` -- Python, platform, torch, transformers, CUDA, cuDNN, GPU name,
  compute capability, total memory, CUDA driver version
- `effective` -- what the *built model* reports, as opposed to what was asked
  for: attention backend at the root and across every submodule, `use_cache`,
  gradient checkpointing, `model.training`, parameter dtypes and devices, buffer
  dtypes, input dtype/shape/checksum, loss and logits dtypes, and whether a KV
  cache came back
- `result` -- loss, whether it is finite, trainable parameter count, how many
  received gradients, the names of any that did not, the names of any whose
  gradient is not finite
- `diagnostics` -- wall time and peak allocated/reserved CUDA memory

**Everything under `diagnostics` is diagnostic only.** It is a single unwarmed
step with no repetition, no L2 flush and no median: a sanity check on the shape
of the allocation, not a measurement. Level-4 timing, when it exists, will go
through the fair protocol like every other number in this repository. See
`docs/BENCHMARK.md`.

## Result on record

The canonical run, on a GH200 120GB (torch 2.11.0+cu128, transformers 5.16.1,
CUDA 12.8), report at `results/qwen3-level4/canonical.json`:

- loss `12.14388656616211`, finite -- a randomly initialised model over a
  151936-token vocabulary predicts near-uniformly, so the expected loss is about
  `ln(151936) = 11.93`
- gradients on 310 of 310 trainable parameters, none missing, all finite
- 596,049,920 parameter elements, matching the config arithmetic exactly
- parameters BF16 on CUDA; rotary `inv_freq` buffers correctly left in float32
- effective attention backend `sdpa` at the root and at every submodule
- `use_cache` off, no KV cache returned, gradient checkpointing off,
  `model.training` true
- logits BF16, loss float32 (Transformers upcasts inside the causal-LM loss)
- peak allocated 16.96 GiB, wall time 19.4 s -- *diagnostic only*, and the wall
  time includes model construction

Two separate processes produced a bit-identical loss, so the workload is
reproducible end to end from the seed.

## Harvesting the workload

The second milestone attaches an observer to the same canonical execution and
exports what it invokes.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.harvest \
    --out results/qwen3-level4/harvest.json \
    --summary-out results/qwen3-level4/harvest-summary.txt
```

It reuses `build_model`, `make_inputs`, `training_step`, the effective-setting
checks and the gradient validation from the smoke path unchanged, so the harvest
describes *the* canonical execution rather than a second, subtly different one.
The same debug overrides apply and mark the manifest non-canonical.

Unlike the smoke run, a failure writes no file. A smoke report saying "it failed"
is useful; a manifest missing a boundary is worse than no manifest, because
everything derived from it inherits the gap silently.

### What is observed

| boundary | mechanism | mandatory |
| --- | --- | --- |
| `Qwen3DecoderLayer` | module hook | yes |
| `Qwen3Attention` | module hook | yes |
| `Qwen3MLP` | module hook | yes |
| `nn.Linear` | module hook | yes |
| `Qwen3RMSNorm` (residual and Q/K head norms) | module hook | yes |
| the MLP activation instance (SiLU) | module hook | yes |
| `Qwen3RotaryEmbedding` | module hook | no |
| `apply_rotary_pos_emb` | function wrapper | yes |
| `torch.nn.functional.scaled_dot_product_attention` | function wrapper | yes |
| `LOSS_MAPPING["ForCausalLM"]` (causal cross entropy) | function wrapper | yes |
| `fixed_cross_entropy` (the flattened CE inside it) | function wrapper | no |

Module hooks are used wherever the boundary is an `nn.Module`, because a hook
sees exactly the tensors the module received and cannot diverge from the real
call. Function wrappers are used only where there is no module, and each names
one attribute on one module.

A mandatory boundary that produces no events raises `MandatoryBoundaryError` and
no manifest is written -- if a future Transformers release moves
`apply_rotary_pos_emb`, the harvest fails loudly rather than quietly omitting
RoPE.

### What is not observed

The softmax and matmuls *inside* SDPA, residual additions, the embedding lookup,
view/reshape/transpose, every ATen operation below the listed boundaries, and
**every backward kernel**. The backward pass runs and gradient coverage is
validated, but nothing traces it. `capture_scope` in the manifest states this
explicitly so the scope is never inferred from what happens to be present.

SDPA's internal softmax is deliberately not a boundary: it will be derived from
the observed attention configuration in a later milestone, and treating it as
directly observed would misstate where the number came from.

### The two views

`events` is the transcript -- one record per invocation, in execution order,
with module path, semantic role, decoder-layer index, input/output/parameter
tensor metadata, operator attributes and workload provenance. Order is
*invocation* order: a decoder layer's ordinal precedes every event inside it.

`configurations` is the working set -- events collapsed by structural
equivalence. The key is the task type, the input/output tensor metadata (shape,
dtype, device, requires-grad, stride, contiguity), the parameter metadata and
the operator attributes. It excludes module path, semantic role, decoder-layer
index and ordinal, so a `gate_proj` and an `up_proj` at the same shape are one
Linear configuration carrying both roles. Every record keeps its frequency, all
source module paths, all roles, all layer indices, `provenance.kind =
"observed"`, and the workload id and config hash.

### Determinism

`manifest_hash` is a SHA-256 over the semantic content only: schema version,
workload id, config hash, capture scope, task counts, events and configurations.
Environment, diagnostics and validation results stay in the file and out of the
hash, so the same workload observed on another machine on another day hashes the
same. Tensor metadata records the device *type*, not the index, for the same
reason, and opaque objects are reduced to their class name rather than `repr`'d,
since a repr carries an address.

### No tensor outlives a hook

Every hook converts its tensors to plain metadata before returning. An observer
that stashed a tensor would keep the 1.2 GiB logits alive to the end of the run
and change the very memory behaviour the workload is meant to describe. A test
walks the whole manifest and fails on any surviving tensor or module.

## Harvest on record

The canonical harvest on a GH200 120GB, at
`results/qwen3-level4/harvest.json` (960 KB) with the summary at
`results/qwen3-level4/harvest-summary.txt`:

- workload `qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad`, config hash
  `6e254de2e1abf1a0` -- the same identity as the smoke run
- manifest hash
  `3ab24571b6d5860859eb5c947daef94f30dfee4d949ec3cf0dea518ad9c7fabc`
- **481 raw events, 18 deduplicated configurations**
- loss `12.14388656616211`, gradients on 310 of 310 parameters, all finite --
  identical to the unobserved smoke run

| task | events | configurations |
| --- | --- | --- |
| `linear` | 197 | 6 |
| `rms_norm` | 113 | 3 |
| `decoder_layer` | 28 | 1 |
| `attention` | 28 | 1 |
| `mlp` | 28 | 1 |
| `sdpa` | 28 | 1 |
| `rope_apply` | 28 | 1 |
| `silu` | 28 | 1 |
| `rotary_embedding` | 1 | 1 |
| `causal_cross_entropy` | 1 | 1 |
| `cross_entropy` | 1 | 1 |

The two derived counts follow from the 28-layer architecture, and are read off
the trace rather than written into it:

**Linear = 197 = 28 x 7 + 1.** Seven per decoder layer -- `q_proj`, `k_proj`,
`v_proj`, `o_proj` in attention and `gate_proj`, `up_proj`, `down_proj` in the
MLP -- plus `lm_head`. The token embedding is an `nn.Embedding`, not a Linear,
and although `lm_head` shares its weight it is a separate module invoked once.

**RMSNorm = 113 = 28 x 4 + 1.** Four per decoder layer -- `input_layernorm` and
`post_attention_layernorm` on the residual stream at width 1024, and Qwen3's
distinguishing `q_norm` and `k_norm` over the 128-wide head dimension -- plus the
final `model.norm`.

The 18 configurations are 6 Linear, 3 RMSNorm and one each of the remaining nine
task types: every decoder layer is structurally identical, so 28 layers collapse
to one of each.

### Representative configurations

```
x56   k_proj,v_proj      [2,2048,1024] bf16 -> [2,2048,1024] bf16
x56   gate_proj,up_proj  [2,2048,1024] bf16 -> [2,2048,3072] bf16
x28   q_proj             [2,2048,1024] bf16 -> [2,2048,2048] bf16
x28   o_proj             [2,2048,2048] bf16 -> [2,2048,1024] bf16
x28   down_proj          [2,2048,3072] bf16 -> [2,2048,1024] bf16
x1    lm_head            [2,2048,1024] bf16 -> [2,2048,151936] bf16
```

`gate_proj` and `up_proj` are one configuration with two roles and 56 module
paths, exactly as intended. `k_proj` and `v_proj` merge for the same reason: both
project 1024 to 8 x 128.

RMSNorm splits three ways, by what is being normalized rather than by name:

```
x57  width=1024  input_layernorm, post_attention_layernorm, norm
                 [2,2048,1024] bf16 -> [2,2048,1024] bf16
x28  width=128   q_norm   [2,2048,16,128] bf16 -> [2,2048,16,128] bf16
x28  width=128   k_norm   [2,2048, 8,128] bf16 -> [2,2048, 8,128] bf16
```

The final `model.norm` joins the two residual norms -- same shape, same width --
and its record carries `null` among the layer indices to say it belongs to no
decoder layer. `q_norm` and `k_norm` normalize the same 128 elements but over 16
and 8 heads, so they stay distinct: different total work, different kernel.

Attention and the loss:

```
sdpa x28   q[2,16,2048,128] k[2,8,2048,128] v[2,8,2048,128] bf16 -> [2,16,2048,128]
           is_causal=True  enable_gqa=True  attn_mask=None  scale=0.08838834764831845
causal_cross_entropy x1   [2,2048,151936] bf16 , [2,2048] int64 -> [] fp32
cross_entropy        x1   [4096,151936] fp32 , [4096] int64 -> [] fp32
```

SDPA receives 8 KV heads with `enable_gqa=True` rather than a materialized
16-head key and value: Transformers takes that path when no attention mask is
needed, which is the case here because the causal pattern is expressed through
`is_causal`. The causal loss enters at BF16 `[2,2048,151936]` and reaches the
flattened cross entropy at FP32 `[4096,151936]` -- the upcast happens inside
`ForCausalLMLoss`, and both boundaries are recorded so the later task can be
declared at whichever one it needs.

### The observer costs nothing measurable

Peak allocated memory is byte-identical with and without observation --
18,214,154,240 bytes in both -- and two independent harvest processes produced
the same manifest hash, the same events, the same configurations and the same
loss. That is the evidence that no hook retained a tensor.

## Level 3: replaying one decoder layer

The harvest says the canonical run invoked 28 identical decoder layers. Whether
one of them can be lifted out and executed on its own is a different claim, and
it needs the actual numbers rather than the manifest.

Two separate commands, deliberately two processes -- a replay that ran inside the
capturing process would prove nothing about standing alone:

```bash
# 1. capture, during the canonical full-model loss and backward
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.capture \
    --layer 14 \
    --harvest results/qwen3-level4/harvest.json \
    --expect-workload-id qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad \
    --expect-manifest-hash 3ab24571b6d5860859eb5c947daef94f30dfee4d949ec3cf0dea518ad9c7fabc \
    --out results/qwen3-level4/layer14.pt

# 2. replay, in a fresh process, from the artifact alone
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.replay \
    --artifact results/qwen3-level4/layer14.pt \
    --report results/qwen3-level4/layer14-replay.json
```

### Why layer 14

Deep enough that its input is a fully mixed residual stream rather than the first
block's near-embedding activations, and far enough from the top that its upstream
gradient has travelled through a realistic length of the backward chain. The
index is not taken on faith: the capture looks it up in the manifest, refuses if
no `decoder_layer` event exists there, and records the event's ordinal and module
path as provenance.

### What is captured

Everything a standalone execution needs, and nothing else:

- the positional and keyword arguments Transformers really passed --
  `hidden_states`, `attention_mask` (**None**, which is what sends SDPA down its
  causal path), `position_embeddings` as a 2-tuple of rotary tensors,
  `position_ids`, `past_key_values` (None), `use_cache` (False)
- the layer's output
- the upstream gradient the **full-model** backward delivered to that output
- the gradient the layer produced for its input
- the layer's weights, and the gradient full-model backward left on each of them

Every tensor is detached, cloned and moved to CPU inside the hook that sees it. A
capture holding a view into the running graph would keep those activations alive
and change the memory behaviour of the run it is describing. The hooks -- module
hooks and the tensor gradient hooks registered during forward -- are all torn
down in a `finally`, so a run that raises mid-backward leaves none behind.

Argument structure is preserved rather than flattened. A format that stored "the
tensors" would lose `attention_mask=None`, and the replay would then take a
different SDPA path than the model did.

### The artifact

`results/qwen3-level4/layer14.pt` (a `torch.save` payload) with a JSON sidecar
`layer14.json` carrying schema version, workload/config/manifest identity,
selected layer index and module path, the full captured signature as tensor
metadata, parameter counts and bytes, the content hash, and the capture
environment and effective runtime settings.

`torch.save` output is not byte-reproducible, but the numbers in it are, so the
**content hash** is logical: it walks the payload in a fixed order and hashes
tensor bytes and scalar values. It covers the captured numbers only -- identity
lives in the metadata and is verified separately, so a corrupted artifact and a
mislabelled one fail differently.

Artifacts are local results. They are not committed.

### Tolerances, and the noise floor

BF16 has a 7-bit explicit mantissa, so the spacing between representable values
near 1.0 -- machine epsilon -- is `2^-7 = 7.81e-3`, and the unit roundoff, half
that spacing, is `2^-8 = 3.91e-3`. The forward path is deterministic -- same
weights, same inputs, same kernels -- and is held to **one unit roundoff**
(`2^-8`): a correct replay should differ by at most a single rounding, if at all.
Backward is not deterministic: SDPA's backward accumulates with atomics, so
replaying the *same* layer twice does not give the same gradients. It is held to
**one epsilon** (`2^-7`) -- one representable step rather than one rounding. That
noise is **measured**, by replaying several times and comparing the replays to
each other, and reported next to the replay-versus-capture error. The tolerances
are fixed constants either way; if an error exceeds one, the report says FAIL.

Errors are relative to the reference tensor's scale, `max|a-b| / max|b|`, not
elementwise. A gradient tensor is mostly near-zero entries, where an elementwise
relative error is meaningless; the scale-normalized form asks how large the
disagreement is compared to the signal. An elementwise figure is also reported,
restricted to entries above 1% of the tensor's scale.

Note what the scale-normalized choice does *not* claim. A small-magnitude tensor
is not near BF16 underflow -- the smallest normal BF16 is about `1.2e-38` -- and
BF16's relative precision is identical at every exponent. The reason to prefer
the scale-normalized metric is that entries far below a tensor's own maximum
contribute nothing to what the tensor is used for, not that they are badly
represented.

### What a comparison must reject

The verdict requires *all* of shape, dtype, stride, replay finiteness, capture
finiteness, and numerical agreement. Each was a way to be wrong that an earlier
version of this file scored as a pass:

- A tensor with the right values and the wrong strides is not the same tensor to
  a kernel that reads it.
- A capture holding a NaN is unusable; a replay that faithfully reproduced the
  NaN would otherwise have been counted as agreement.
- **A zero reference with a non-zero replay.** Normalizing by a zero scale used
  to fall through to a relative error of `0.0` -- a perfect score for the worst
  possible result. The ratio is now reported as `null` with an explicit
  `zero_reference_mismatch` flag, and the verdict fails. Two zero tensors still
  agree exactly.

A negative control in the test suite perturbs one weight by 5%, re-hashes so the
integrity checks still pass, and requires the report to come back FAIL --
otherwise "zero error everywhere" would be indistinguishable from a comparison
that never ran.

`--noise-repeats` is validated rather than trusted: `0` skips the measurement and
anything `>= 2` makes one, but `1` is refused, because a single replay has
nothing to be compared with and would silently report a floor of zero.

### Two hashes, because there are two ways to be wrong

`content_hash` covers the captured numbers alone and catches a corrupted or
truncated file. `artifact_hash` binds those numbers to the schema version and to
every provenance identity field, and catches a file whose *label* was edited: a
correct layer-14 capture relabelled as layer 9, or as belonging to a different
manifest, has intact content and a valid content hash and is exactly as wrong.

Loading verifies both, together, always. Artifacts are read with
`weights_only=True` -- the payload is tensors, plain scalars and the containers
holding them, and an artifact is a file a consumer may not have produced itself.

Anything downstream that claims to be derived from the canonical execution goes
through `artifact.load_canonical`, which checks both hashes and all four identity
fields against the tracked snapshot and **takes no argument that turns any of
that off**. A consumer cannot accidentally build a Level-2 task on a debug
capture, or on the right numbers with the wrong label.

### Proof that the replay stands alone

The report does not assert it in prose. It scans the live heap with `gc` and
counts instances of `Qwen3ForCausalLM`, `Qwen3Model`, `Qwen3DecoderLayer` and
`Qwen3Attention`, and fails if any full-model object exists. "No such object is
alive" is stronger evidence than "we did not call the constructor".

## Layer-14 replay on record

Captured on a GH200 120GB during the canonical run, replayed in a **separate
process** from the artifact alone.

| | |
| --- | --- |
| artifact | `results/qwen3-level4/layer14.pt` (93.0 MiB) |
| metadata | `results/qwen3-level4/layer14.json` |
| report | `results/qwen3-level4/layer14-replay.json` |
| content hash | `9bf5758bf4915d49b26456835e620689bf74ab578014b0f003449ae28ce21c79` |
| source manifest | `3ab24571…` event ordinal 239, `model.layers.14` |
| one layer | 11 parameter tensors, 15,730,944 elements, 30.0 MiB |

The capture did not disturb the run it observed: loss `12.14388656616211` and
peak allocated 18,214,154,240 bytes, both identical to the unobserved smoke run.

### Captured call

```
args[0]              [2, 2048, 1024] bfloat16
attention_mask       None
position_embeddings  ([1, 2048, 128] bfloat16, [1, 2048, 128] bfloat16)
position_ids         [1, 2048] int64
past_key_values      None
use_cache            False
output               [2, 2048, 1024] bfloat16
grad_output          [2, 2048, 1024] bfloat16      (from full-model backward)
grad_input           [2, 2048, 1024] bfloat16
```

### Result: pass, bitwise

Output, input gradient and **all 11** parameter gradients came back
**bitwise identical** to what the full model produced -- max absolute error 0.0
everywhere, dtype, shape and stride all matching.

The replay process held `Qwen3ForCausalLM: 0`, `Qwen3Model: 0`,
`Qwen3DecoderLayer: 1`, `Qwen3Attention: 1` on the heap, and 15.7M parameters
against the full model's 596M. Forward+backward took 969 ms unwarmed at a peak of
461.7 MiB -- diagnostic only, against 17 GiB for the full step.

### The noise floor is real, and it is not zero

Replaying the same layer five times and comparing the replays *to each other*:

| quantity | replay vs capture | replay vs replay (8 runs) |
| --- | --- | --- |
| output | 0.0, bitwise | 0.0 |
| grad_input | 0.0, bitwise | 0.0 |
| parameter gradients | **4.941e-04** (`self_attn.q_proj.weight`) | **4.941e-04** (same tensor) |

SDPA's backward is nondeterministic here, exactly as expected, and the two
columns are the same number: the disagreement between the replay and the full
model is *entirely* the layer's own run-to-run noise, with nothing left over. It
also shows how the floor depends on how hard you look -- three repeats found
0.0, five found 1.2e-4, eight found 4.9e-4 -- so a single measurement is a lower
bound, not the floor. What the numbers support is the claim that matters: the
replay reproduces the full model to within the layer's own reproducibility.

The gradient tolerance stayed at its stated one BF16 epsilon, `2^-7 = 7.81e-3`,
which sits about 16x above the measured floor. It was not adjusted to fit the
result.

A negative control in the test suite perturbs one weight by 5%, re-hashes the
artifact so the integrity check still passes, and requires the report to come
back FAIL with a non-zero forward error -- otherwise "zero error everywhere"
would be indistinguishable from a comparison that never ran.

## Level 2: `qwen3_swiglu_mlp`, the first observed task

The first operator in this repository whose shape was **observed** rather than
chosen. Every dimension comes from the canonical Qwen3-0.6B step: the harvest
found one `Qwen3MLP` configuration running 28 times, once per decoder layer, and
the verified Layer-14 replay is where the numbers were taken from.

### The contract

```python
gate = F.linear(x, gate_weight)
up = F.linear(x, up_weight)
hidden = F.silu(gate.float()) * up.float()
hidden = hidden.to(x.dtype)
output = F.linear(hidden, down_weight)
```

| tensor | shape | dtype |
| --- | --- | --- |
| `x` | `[2, 2048, 1024]` | BF16 |
| `gate_weight` | `[3072, 1024]` | BF16 |
| `up_weight` | `[3072, 1024]` | BF16 |
| `down_weight` | `[1024, 3072]` | BF16 |
| `output` | `[2, 2048, 1024]` | BF16 |

Backward returns `(dx, dgate_weight, dup_weight, ddown_weight)`.

`B` and `T` stay separate axes because that is the shape the module actually
received; a kernel may flatten them internally. `gate_weight` and `up_weight` are
separate tensors, as Qwen3 stores them -- not one fused `[2I, H]` matrix. No
biases.

The reference accumulates the gate/up product in float32 and casts once, which is
deliberately *more* accurate than `Qwen3MLP`, which stays in BF16 throughout. A
reference exists to be the correct answer, not to reproduce a particular
rounding, and every other declaration here follows the same convention. What that
choice costs is measured rather than assumed: the verification compares the
declared reference *and* the BF16 spelling against the capture, so the difference
between the two contracts is a number in a report.

### The frozen workload snapshot

A declaration cannot depend on `results/harvest.json` -- that file is a megabyte
of local transcript on a machine that has run the workload, and declarations must
import on a laptop. But the whole point of this task is that its shape came from
a real training step.

`src/evograd/bench/workloads/qwen3/harvest/snapshot.json` is the reconciliation: small,
tracked, hashed, and carrying only what a task needs to state its provenance --
workload id and config hash, source manifest hash, the harvested configuration
ids, the `Qwen3MLP` frequency of 28, all 28 source module paths and layer
indices, the hidden/intermediate widths and the dtype. It imports with nothing
but `json`, `hashlib` and `pathlib`.

It is a frozen extract, not a second source of truth:

```bash
# regenerate from a full harvest
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.snapshot --write

# or check that the tracked file still agrees with one
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.harvest.snapshot --validate
```

`--validate` re-derives it and prints a field-by-field diff on disagreement.
Neither command is needed to *use* the snapshot.

The declaration then derives its benchmark dims *from* that record rather than
typing them alongside it, so a snapshot that stopped agreeing with the
declaration would fail at import. Provenance is checkable twice over, in two
independent ways: `Provenance(model="qwen3_0_6b", component="swiglu_mlp", ...)`
re-derives the dims from the published Qwen3-0.6B configuration exactly as every
other `hf_config` workload does, and `HARVEST`/`FREQUENCY`/`PROVENANCE_CHAIN` in
the declaration carry the observed record itself as structured data.

### Extraction and verification

```bash
# capture one Qwen3MLP invocation from inside the standalone Layer-14 replay
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp extract \
    --source results/qwen3-level4/layer14.pt \
    --out results/qwen3-level4/layer14-mlp.pt

# check the declaration's reference against what the model computed
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp verify \
    --artifact results/qwen3-level4/layer14-mlp.pt \
    --report results/qwen3-level4/layer14-mlp-verify.json
```

The source is the *replay*, not the full model. The Layer-14 artifact has already
been shown to reproduce the full model bitwise, so a capture taken from replaying
it inherits that guarantee while costing one layer instead of 596M parameters --
and the extraction is then reproducible on any machine that has the artifact,
with no 17 GiB training step in between. Only layer 14 is captured; the other 27
are not, and the harvest is what establishes that they are the same
configuration.

The chain is carried in the artifact and validated link by link against the
snapshot:

```
canonical workload qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad
  -> harvest manifest 3ab24571...
    -> model.layers.14 (event ordinal 239)
      -> Layer-14 artifact (content + identity hashes)
        -> Qwen3MLP invocation at model.layers.14.mlp
          -> qwen3_swiglu_mlp
```

### Level-2 result on record

Extracted and verified on a GH200, `results/qwen3-level4/layer14-mlp.pt`
(68.0 MiB) with report `layer14-mlp-verify.json`:

| | |
| --- | --- |
| status | **PASS**, provenance validated |
| content hash | `8d95f51bc12554a0efe9442fca26db009f0395fc9c5ba3e7af5655b4ad15853f` |
| identity hash | `f8c690bb5872935e24d51fb73eb6a9f5dc12213ce3db11baf8f60e45ff9c8b7c` |
| snapshot hash | `069a64888c451972afc0171d7e3c171ed7ba406ae55fb3ec6176b37575f9c8b3` |
| source Layer-14 artifact | `25bd8ed68fd49eb14361c596886d55e6530e826211f09a10c85e6f379eed6efe` |

**The BF16 spelling -- the same computation Transformers ran -- is bitwise
identical on every quantity**: output, input gradient, and all three weight
gradients. That is the wiring proof. A transposed weight, a swapped gate and up,
or a misrouted upstream gradient would all break it.

**The declared float32-accumulated reference**, against the operator's own
declared BF16 tolerance (now the calibrated `1e-2` base, see below):

| quantity | relative error |
| --- | --- |
| output | 5.848e-03 |
| grad x | 7.692e-03 |
| grad `down_weight` | 6.211e-03 |
| grad `gate_weight` | 5.714e-03 |
| grad `up_weight` | 4.310e-03 |

All within tolerance with an order of magnitude of headroom, and all with a
measured reference noise floor of exactly 0.0 across 8 runs -- unlike the layer
replay, this path is pure GEMMs and elementwise work with no atomics, so it is
deterministic.

Worth stating plainly: `5.8e-3` is about **1.5 BF16 epsilons**, so the float32
upcast is not free. That is the price of a more accurate reference, it is
measured rather than assumed, and it is why the two comparisons carry two
different tolerances instead of one.

## Level 2: `qwen3_attention`, the observed attention boundary

The second observed task. The harvest found one SDPA configuration running 28
times, once per decoder layer, and one `o_proj` configuration alongside it;
together they are the half of `Qwen3Attention` that follows the projections.

### The boundary, stated once

```
q, k, v
  -> F.scaled_dot_product_attention(attn_mask=None, dropout_p=0.0,
                                    is_causal=True, scale=1/sqrt(128),
                                    enable_gqa=True)
  -> transpose heads and tokens, restore [B, T, 2048]
  -> F.linear(..., o_weight)
  -> out [B, T, 1024]
```

It **excludes** `q_proj`, `k_proj`, `v_proj`, the Q/K head-dimension RMSNorms and
the rotary embedding. Those are a separate boundary with a separate cost and
become `qwen3_qkv_norm_rope` later. q, k and v arrive already projected, already
normalized and already rotated -- the state the observed SDPA call received.

| tensor | shape | stride | dtype |
| --- | --- | --- | --- |
| `q` | `[2, 16, 2048, 128]` | `[4194304, 128, 2048, 1]` | BF16 |
| `k` | `[2, 8, 2048, 128]` | `[2097152, 128, 1024, 1]` | BF16 |
| `v` | `[2, 8, 2048, 128]` | same as `k` | BF16 |
| `o_weight` | `[1024, 2048]`, no bias | contiguous | BF16 |
| `out` | `[2, 2048, 1024]` | contiguous | BF16 |

SDPA configuration `9674b971ae24b325` (frequency 28), `o_proj` configuration
`ea5311a8e1cbba90` (frequency 28), representative source
`model.layers.14.self_attn`. Backward returns `(dq, dk, dv, do_weight)`.

**Layout is part of the contract.** q, k and v are non-contiguous head-major
views, because the model projects into `[B, T, heads, D]` and transposes.
`make_qwen3_attention_inputs` reproduces that by building token-major tensors and
transposing them; allocating contiguous substitutes would benchmark a different
access pattern than the one that was observed.

### Two spellings, and which one is timed

`forward` writes the attention out in primitives -- expand the KV heads, score,
mask, float32 softmax, weight the values -- so the oracle differentiates the
definition. It materializes a `[2, 16, 2048, 2048]` float32 score matrix, 512
MiB the real execution never allocates, and is therefore **never** the eager
timing baseline.

`runtime_forward` is the SDPA branch Transformers takes, and is what the eager
baseline is timed through. `verify_runtime_forward` proves the two agree on
every correctness workload before any timing is trusted.

## Tolerances are measured, and one artifact holds the tensors

### Calibration, not round numbers

Both Level-2 tasks declare a `runtime_forward` -- the spelling the model runs --
and the eager baseline is timed through it. That also makes it the right
instrument for calibrating a tolerance: the disagreement between the declared
oracle and the production spelling is the *smallest* error any correct
implementation can have with the oracle, so a gate that rejects it rejects
correct code.

Both `calibrate` commands measure, per result, the smallest base `t` for which
the declaration's own gate `allclose(atol=ma*t, rtol=mr*t)` accepts the
production spelling, over every correctness workload *and* the canonical
invocation:

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp calibrate \
    --report results/qwen3-level4/qwen3_swiglu_mlp-tolerance.json
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.attention calibrate \
    --report results/qwen3-level4/qwen3_attention-tolerance.json
```

**`qwen3_swiglu_mlp`** -- the disagreement is one rounding of the SwiGLU
intermediate, propagated. Relative to each tensor's scale it is strikingly
uniform, 2.8e-3 to 7.6e-3 (0.7 to 2 BF16 epsilons), across both the synthetic
correctness cases and the real Layer-14 tensors. What varies is how much
cancellation each reduction has, which is what the atol multipliers absorb.

| result | measured min atol multiplier at base 1e-2 | declared |
| --- | --- | --- |
| `out` | 1.00 | none |
| `dx` | 1.48 | 2.3 |
| `dgate_weight` | 2.42 | 3.7 |
| `dup_weight` | 3.22 | 4.9 |
| `ddown_weight` | 4.33 | 6.5 |

Base BF16 `(1e-2, 1e-2)`, replacing the arbitrary `8e-2` it was declared with;
each multiplier is the measured minimum times a 1.5 safety margin, rounded up to
one decimal. float32 measured **exactly 0.0** everywhere -- the upcast is a no-op
when the inputs are already float32 -- so that dtype keeps the repository's
ordinary `(2e-5, 2e-5)`.

**`qwen3_attention`** -- worst over all cases: `out` 5.8e-3, `dq` 4.6e-3, `dk`
5.8e-3, `dv` 1.5e-8 (SDPA and the dense spelling compute `dv` identically),
`do_weight` 2.9e-2. Base BF16 `(1e-2, 1e-2)` leaves ~1.7x on the binding
non-multiplied result; only `do_weight` needs a multiplier (measured 3.54,
declared 5.4), because it is the one result that reduces over all B*T tokens and
cancels. float32 measured 4.0e-7 worst, which `(2e-5, 2e-5)` clears by ~50x.

A negative control in each test suite perturbs the production spelling by 2% and
requires `verify_runtime_forward` to reject it -- and, for the MLP, asserts that
the old `8e-2` tolerance *would have accepted* the same error, so the tightening
is demonstrated rather than claimed.

### One tensor artifact, not three

`layer14.pt` is the authoritative tensor store. Both Level-2 tasks are derived
from it by replaying the layer and hooking the boundary, on demand, in about a
second; neither writes tensors.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp derive --metadata-out ...
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level2.attention verify --report ...
```

A derived `.pt` would be tens of MiB of numbers that already exist under a
second name, with nothing forcing the two to stay equal once either is
regenerated. What the derivation does emit is a content hash over the numbers and
a derivation hash binding them to the provenance identity, so two derivations on
two machines can be compared without either writing a file.

## Structured outputs

``OpDecl.output`` accepts either one ``Active`` or an ordered tuple of them. The
single-output form is stored as a bare ``Active``, not a one-element tuple, so
every declaration written before this and every caller reading ``op.output``
keeps working unchanged; internal code reads ``op.outputs``, which is always a
tuple.

The candidate ABI follows:

```python
outputs, saved = candidate_forward(...)      # a Tensor, or an ordered tuple
input_grads    = candidate_backward(output_grads, saved)
```

A single-output candidate keeps its own gradient's name for the first backward
parameter (``dy``, ``dout``, ...) and returns a Tensor; a multi-output candidate
binds the whole tuple to ``output_grads``. ``op.forward_returns()`` renders the
matching line for prompts and config templates, so contract text cannot drift
from the ABI.

``op.upstream_grad_name`` deliberately **raises** for a multi-output
declaration rather than returning the first of several. Every path that reads it
treats it as *the* gradient, and quietly handing back one of three would produce
a backward that is wrong in a way no shape check catches. Paths that need all of
them use ``op.upstream_grad_names`` or ``upstream_grad_values(op, values)``.

Framework paths updated: declaration validation and shape binding, input and
upstream-gradient generation, the autograd oracle, ``runtime_forward``
verification, per-output correctness/dtype/shape/stride/finiteness checks,
per-output tolerance lookup and reporting, ``bind`` and the integrated training
step's ``autograd.Function``, the fair protocol's provider verification, warmup
and timing, the compiled baselines, the NCU profile template, the evolution
evaluator, and the candidate prompt and config templates.

Pipelines B and D refuse a multi-output declaration with a named
``NotImplementedError`` rather than emitting a wrapper that passes one gradient
where the contract requires a tuple. A seed that looked right and failed at its
first backward would be worse than a seed that was never generated.

Limited on purpose to a non-empty tuple of named Tensor outputs: no dicts, no
pytrees, no optional or non-Tensor outputs.

## Level 2: `qwen3_qkv_norm_rope`

The prefix of `Qwen3Attention`, and the first operator here with more than one
output.

```
normalized hidden_states
  -> separate q_proj, k_proj, v_proj
  -> reshape by heads
  -> per-head Q/K RMSNorm over the head dimension
  -> transpose to head-major
  -> apply RoPE to q and k
  -> return (q, k, v)
```

SDPA and `o_proj` are **not** part of it -- they are `qwen3_attention`, and the
two meet exactly where this task's outputs become that one's inputs. The residual
RMSNorm that produces `x` is also outside; it is a later task.

| tensor | shape | stride | notes |
| --- | --- | --- | --- |
| `x` | `[2, 2048, 1024]` BF16 | contiguous | |
| `q_weight` | `[2048, 1024]` | | no bias |
| `k_weight`, `v_weight` | `[1024, 1024]` | | no bias |
| `q_norm_weight`, `k_norm_weight` | `[128]` | | eps `1e-6` |
| `cos`, `sin` | `[1, 2048, 128]` BF16 | | Inactive: shared tables, no gradient |
| `q` | `[2, 16, 2048, 128]` BF16 | `[4194304, 128, 2048, 1]` | |
| `k`, `v` | `[2, 8, 2048, 128]` BF16 | `[2097152, 128, 1024, 1]` | |

Outputs in the strict order `(q, k, v)`; backward returns `dx, dq_weight,
dk_weight, dv_weight, dq_norm_weight, dk_norm_weight`.

Every dimension is read off the snapshot, which is derived from five harvested
records rather than written twice: `apply_rotary_pos_emb`
(`34378ecf454fc895`, frequency 28) whose two outputs *are* q and k, `q_proj`
(`b9cc24095ee7dc62`, 28), the deduplicated `k_proj`/`v_proj`
(`be060ca58cf90863`, 56 -- twice per layer), `q_norm` (`a872639f0512398e`, 28)
and `k_norm` (`494b2d469ae06000`, 28). `v` never passes through RoPE, so its
observed layout is sourced from the SDPA call that consumes it rather than
invented.

### Canonical result

`results/qwen3-level4/layer14-qkv-verify.json`, **PASS**, provenance validated,
snapshot `47b4190f3c18ae4c…`.

The production spelling -- the same computation the model ran -- is **bitwise
identical on all nine quantities** (`q`, `k`, `v` and all six input gradients),
with matching strides, and a measured noise floor of exactly 0.0 across four
runs: unlike attention, this path has no atomics and is deterministic.

The declared float32 reference, against the operator's own gate:

| result | rel vs scale | required_t |
| --- | --- | --- |
| `q` | 6.58e-03 | 7.63e-03 |
| `k` | 6.54e-03 | 9.26e-03 |
| `v` | 0 | 0 |
| `dx` | 2.19e-03 | 1.86e-08 |
| `dq_weight` | 3.95e-03 | 6.72e-08 |
| `dk_weight` | 2.92e-03 | 8.83e-08 |
| `dv_weight` | 0 | 0 |
| `dq_norm_weight` | 4.05e-03 | 3.22e-08 |
| `dk_norm_weight` | 4.08e-03 | 5.68e-08 |

`v` and `dv_weight` measure exactly zero at both dtypes and in every case: the
value path has no RMSNorm and no rotation, so the two spellings are the same
computation there.

### Calibrated tolerances

Base BF16 `(2e-2, 2e-2)`, set by the binding *forward* requirement -- `q` at
1.18e-02 across all cases -- so the outputs are gated by the base alone and only
the reductions carry multipliers; 2e-02 leaves 1.69x. Going lower would mean
putting a multiplier on a forward output, which hides the number a candidate is
primarily judged on. float32 is `(2e-5, 2e-5)` against a measured worst of
2.2e-06, ~9x.

| result | measured min atol multiplier at base 2e-2 | declared |
| --- | --- | --- |
| `q`, `k`, `v`, `dv_weight` | 1.00 | none |
| `dx` | 1.02 | 1.6 |
| `dq_weight` | 4.67 | 7.1 |
| `dk_weight` | 3.56 | 5.4 |
| `dq_norm_weight` | 4.91 | 7.4 |
| `dk_norm_weight` | 2.75 | 4.2 |

Each multiplier is the measured minimum times a 1.5 safety margin, rounded up to
one decimal. The weight gradients need the largest because they reduce over all
B*T tokens with cancelling signs.

### Still one tensor artifact

`layer14.pt` remains the only tensor store. The boundary is re-derived by
replaying the layer and hooking two points -- `self_attn`'s input for `x` and
`dx`, and the SDPA call for `q`/`k`/`v` and their upstream gradients -- and only
JSON is written.

## Level 2: `fused_add_rms_norm` gets its second output

The plan's logical `residual_rmsnorm` boundary already had a physical task here:
`fused_add_rms_norm`. Rather than adding a Qwen-specific duplicate, the existing
generic declaration was corrected -- it was describing the wrong operator.

```python
summed     = x + residual
normalized = RMSNorm(summed, weight, eps)
return normalized, summed
```

A decoder's residual stream keeps the *un-normalized* sum and hands it to the
next block. A kernel returning only `normalized` would force either a second
pass to recompute the sum or the caller keeping it alive anyway -- which is
exactly the memory traffic the fusion exists to avoid. Every real
implementation, Liger's included, returns both.

`normalized` stays first, so the former single output remains the primary one.

```python
(normalized, summed), saved = forward(x, residual, weight, eps)
dx, dresidual, dweight      = backward((dnormalized, dsummed), saved)
```

The backward has to combine the paths:

```
dtotal    = dsummed + RMSNormBackward(dnormalized)
dx        = dtotal
dresidual = dtotal
dweight   = sum over rows of dnormalized * summed * rstd     # normalized path only
```

Ignoring `dsummed` is the characteristic error, and it is a quiet one: it leaves
`dweight` exactly correct, so a check that looked only at the norm's parameter
would pass it. There is a negative test for precisely that.

### Liger

Liger's kernel already implements the corrected contract and the adapter was
throwing half of it away. `fused_add_rms_norm_forward` returns `(Y, S, RSTD, …)`,
so `summed` was already there; `fused_add_rms_norm_backward(dY, dS_out, …)` adds
`dS_out` into `dX` after the normalization backward (`dX_row += dS_out_row`) and
returns `dX, dX, dW` -- the contract exactly. The adapter passed
`torch.zeros_like(summed)` for `dS_out` and discarded `S`. Both are fixed; the
provider now expresses the whole operator, and a test asserts the upstream
signature really has the slot rather than assuming it.

### Qwen3 provenance: 56 sites, one verified

The canonical configuration is rows=4096 (batch 2 x sequence 2048 tokens),
cols=1024, BF16, eps=1e-6, added as the `qwen3_0_6b_observed` suite. The generic
Llama-derived grid stays the default timed set.

It occurs **56 times per step**, derived from the layer count rather than
counted by hand:

| | |
| --- | --- |
| attention residual add -> that layer's `post_attention_layernorm` | 28 |
| MLP residual add -> the **next** layer's `input_layernorm` | 27 |
| final MLP residual add -> `model.norm` | 1 |
| **total** | **56** |
| excluded: layer 0's `input_layernorm`, which has no preceding decoder residual add | 1 |

The harvest observed **57** residual-width RMSNorm invocations, and the snapshot
extractor asserts `56 + 1 == 57` rather than stating it in prose -- if the two
ever stop agreeing, extraction fails.

Frequency and verification are different claims and are recorded separately:
56 is architecture-derived, while `directly_verified_invocations` is **1**. One
invocation, inside layer 14, is compared against the model; the other 55 are the
same configuration by deduplication, not by direct comparison.

### Canonical result

`results/qwen3-level4/layer14-residual-verify.json`, **PASS**. The representative
boundary is the attention residual add and the norm after it:

```
residual   = the decoder layer's input
x          = the self_attn branch's output
summed     = residual + x
normalized = post_attention_layernorm(summed)
```

The exact Transformers spelling is **bitwise identical on all five directly
observable quantities**: `out`, `summed`, `dx`, `dweight`, and `dtotal`. That
last one is the load-bearing check -- `summed`'s own gradient in the layer is
`dsummed + RMSNormBackward(dnormalized)`, so reproducing it bitwise is direct
evidence that the two output paths were combined correctly, and it comes from a
different place in the layer graph than the `dx` comparison does.

| spelling | `out` | `summed` | `dx` | `dweight` | `dtotal` |
| --- | --- | --- | --- | --- | --- |
| Transformers (`Qwen3RMSNorm`) | 0, bitwise | 0, bitwise | 0, bitwise | 0, bitwise | 0, bitwise |
| `runtime_forward` (`F.rms_norm`) | 1.70e-03 | 0, bitwise | 6.07e-04 | 5.59e-03 | 6.07e-04 |
| declared primitive reference | 6.80e-03 | 0, bitwise | 4.85e-03 | 2.79e-03 | 4.85e-03 |

Measured noise floor: **0.0** across four runs; this path has no atomics.

**`dresidual` is not directly comparable, and the report says so.** In the real
layer the decoder input feeds the residual add *and* `input_layernorm`, so its
gradient there is this boundary's `dresidual` plus a path outside the boundary.
Claiming a direct comparison would be a claim the graph cannot support. What the
contract says about it -- `dresidual == dx` -- is proved against an isolated
autograd reference instead, bitwise, in all three spellings, and the report
carries the reason alongside the result.

### Calibrated tolerances

Measured by `bench.workloads.qwen3.levels.level2.residual_rmsnorm calibrate`, comparing the declared
primitive forward against `runtime_forward` on every correctness workload and on
the canonical invocation, under the declaration's own gate:

| dtype | declared base | worst measured requirement | margin |
| --- | --- | --- | --- |
| float32 | 2e-05 | 2.9e-07 (`dx`) | 69x |
| float16 | 5e-02 | 2.2e-03 (`dx`) | 23x |
| bfloat16 | 8e-02 | 1.8e-02 (`dweight`) | 4.4x |

The shared `STANDARD_TOLERANCES` are kept: they are the same constants twelve
other operators use, and the measurements say they are sound here with margin,
so tightening them would be a change to eleven unrelated declarations. The
row-count-aware `dweight` hook is kept for the same reason and re-measured under
the new contract.

One tolerance *is* tightened, because the measurement asked for it: `summed`
requires **exactly 0** in every case at every dtype -- both spellings compute the
same elementwise add -- so it carries a 0.01 multiplier. Holding a plain sum to
the same 8e-2 as a normalized output would let a candidate be badly wrong about
it and still call itself fused. `tolerance_multipliers` now accepts an output
name as well as a gradient name, which is what structured outputs made
meaningful.

### Still one tensor artifact

`layer14.pt` remains the only tensor store; the boundary is re-derived by
replaying the layer and hooking the attention branch's output and the norm
around it. Only JSON is written.

## Level 1: mapping the observed configurations onto generic primitives

Every configuration the canonical Qwen3-0.6B step runs maps onto a generic
Level-1 task. Five already existed and were generalized; one -- fused causal
grouped-query attention -- did not, and was added generically rather than as a
Qwen copy.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping mapping
```

| generic task | harvested config | roles | freq | dims |
| --- | --- | --- | --- | --- |
| `linear_no_bias` | `b9cc24095ee7dc62` | `q_proj` | 28 | M=4096 K=1024 N=2048 |
| | `be060ca58cf90863` | `k_proj`, `v_proj` | 56 | M=4096 K=1024 N=1024 |
| | `ea5311a8e1cbba90` | `o_proj` | 28 | M=4096 K=2048 N=1024 |
| | `512569f88d48860a` | `gate_proj`, `up_proj` | 56 | M=4096 K=1024 N=3072 |
| | `6163dcf09417a929` | `down_proj` | 28 | M=4096 K=3072 N=1024 |
| | `cbada5ef6e5d6e6c` | `lm_head` | 1 | M=4096 K=1024 N=151936 |
| `rmsnorm` | `1bf435a6d3df750a` | `input_layernorm`, `post_attention_layernorm`, `norm` | 57 | rows=4096 hidden=1024 |
| | `a872639f0512398e` | `q_norm` | 28 | rows=65536 hidden=128 |
| | `494b2d469ae06000` | `k_norm` | 28 | rows=32768 hidden=128 |
| `rope` | `34378ecf454fc895` | `apply_rotary_pos_emb` (q) | 28 | B=2 heads=16 T=2048 D=128 |
| | `34378ecf454fc895` | `apply_rotary_pos_emb` (k) | 28 | B=2 heads=8 T=2048 D=128 |
| `swiglu` | `be8d90aa36b9bf50` | `act_fn` | 28 | rows=4096 cols=3072 |
| `causal_gqa_attention` | `9674b971ae24b325` | `scaled_dot_product_attention` | 28 | B=2 HQ=16 HK=8 T=2048 D=128 |
| `cross_entropy` | `b33722d17d3a0916` | `fixed_cross_entropy` | 1 | rows=4096 cols=151936, FP32 |

197 Linear invocations is `28 x 7 + 1`; 113 RMSNorm is `28 x 4 + 1`. One
harvested RoPE record becomes two workloads, because `apply_rotary_pos_emb`
rotates the queries and the keys in a single call at different head counts.

### Biasless, not zero-biased

Qwen3 sets `attention_bias=False` and gives its MLP projections and `lm_head` no
bias -- and so does Llama-3. The first version of this mapping put those
configurations on the bias-carrying `linear` task and generated a zero-valued
bias for them. That is not the same operator: a zero bias still costs a
broadcast add in the forward, a `dbias` row reduction in the backward, and a
third entry in the gradient contract. A candidate would be optimized against
work the model never does, and the baseline would pay for it too.

They now map onto a new generic `linear_no_bias`:

```python
y = F.linear(x, weight, bias=None)          # runtime_forward
dx, dweight = backward(dy)                  # two gradients, no dbias
```

`weight` is `[N, K]` -- `nn.Linear`'s stored layout and the one the harvest
recorded. That is deliberately not `matmul`'s `[K, N]`: the two are different
memory layouts for the same mathematics, and mapping the observed projections
onto `matmul` would benchmark a transpose the model does not perform.

The biasless Llama-3 projections moved with them. What stays on `linear` is the
same widths through a bias-carrying contract, now marked `scaled` with a note
saying so -- an explicit ablation that exercises `dbias` at realistic widths,
not a claim about any model.

### Two deliberate non-mappings

**No standalone `softmax` case.** The model runs fused SDPA and never
materializes a softmax operator, so a Qwen softmax workload would be a shape no
step executes. `softmax` has no `qwen3_0_6b_observed` suite at all rather than an
empty one -- an empty suite would look like a mapping that found nothing.

**No bare SiLU task.** The harvest records a SiLU module, but the production
pointwise boundary is `silu(gate) * up`; the activation never appears without the
multiply. It maps onto the existing `swiglu`, and the SiLU record together with
the gate/up projection it sits between are kept as supporting provenance.

### What was reused, generalized, and added

`linear`, `rmsnorm` and `swiglu` needed only the observed suite. `rope` was
generalized twice: its rotary base now comes from the workload's own provenance
(Llama-3's 500000, Qwen3's 1000000 -- a wrong base gives a kernel that is
self-consistent and completely wrong), and its Qwen inputs reproduce the observed
non-contiguous head-major strides `[4194304, 128, 2048, 1]` instead of a
contiguous substitute. `cross_entropy` needed the observed suite and the
provenance chain from the BF16 `[2, 2048, 151936]` logits through Transformers'
causal-loss wrapper to the flattened FP32 `[4096, 151936]` call.

`rope` gained a `runtime_forward`. Its oracle evaluates the rotation in float32
and casts back, which is the more accurate answer; `apply_rotary_pos_emb` stays
in the model dtype throughout. The two differ by about 5e-3 relative at
bfloat16, so timing the eager baseline through the oracle would charge it for
casts Transformers never executes. The new `rope_runtime_ref` reproduces the
Transformers spelling **bitwise**, stride included.

`causal_gqa_attention` is new. Its declared forward is a dense
score-mask-softmax reference used **only** as the oracle; `runtime_forward` is
one `F.scaled_dot_product_attention` call, which is what the eager baseline is
timed through. Backward returns `dq, dk, dv`. The output projection is a separate
GEMM and stays in the Level-2 `qwen3_attention` task.

### Defaults preserved

Every task keeps its Llama-derived default `benchmark` grid, its regime split and
its `legacy` ablation suite; the Qwen cases are an additional named suite and are
added to `coverage`. The new task gets a Llama-3-8B-derived default of its own so
it is a generic primitive rather than a Qwen-only one.

### Cross entropy, proved against the model's own call

The first version of this check ran the observed cross entropy at `[4096,
151936]` and confirmed the loss sat near `ln(vocab)`. That is a sanity check,
not a proof: it would pass for any implementation with the right scale and the
wrong gradient, because it never looks at a gradient and never touches the
model's tensors.

```bash
PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level1.mapping cross-entropy
```

The canonical Level-4 step is now run on demand with `fixed_cross_entropy`
intercepted. The declared Level-1 reference is evaluated **on the same live
tensors** -- the exact flattened FP32 logits and int64 labels Transformers
passed -- and differentiated against the **same upstream scalar**, captured from
a hook on the loss rather than assumed to be 1.0.

| | model | Level-1 reference |
| --- | --- | --- |
| loss | `12.14388656616211` | `12.14388656616211` (abs error 0.0) |
| `dlogits` | `[4096, 151936]` fp32, stride `[151936, 1]` | identical shape, dtype and stride |
| agreement | | **bitwise**, max abs error 0.0, all finite |

`ignore_index=-100`, `reduction=mean`, upstream scalar `1.0`, all read off the
model's call. Peak 23.9 GiB; the 2.5 GiB logits and their gradient are compared
inside the hooks that see them and dropped with the step. Nothing is persisted.

### Composition back into Level 2

The Level-1 shapes are the Level-2 inputs that were already verified against the
model, and the tests assert it dimension by dimension:

```
linear(q/k/v_proj) + rmsnorm(q/k_norm) + rope   -> qwen3_qkv_norm_rope
causal_gqa_attention + linear(o_proj)           -> qwen3_attention
linear(gate/up, down) + swiglu                  -> qwen3_swiglu_mlp
residual add + rmsnorm(4096x1024)               -> fused_add_rms_norm
```

## Verification, not assertion

Requesting a setting and getting it are different events, and this milestone
treats them as such. `attn_implementation` can be downgraded; the `torch_dtype`
keyword was renamed to `dtype` in Transformers 5 and the old spelling is
swallowed by `**kwargs`, leaving a model silently in float32; `use_cache` on the
config can be overridden at the call site. So the report carries both the
request and what the constructed object says about itself, and the run fails if
they disagree.

## BF16 tolerances at the observed shape, calibrated

The declared BF16 tolerances for the three Qwen Level-2 tasks were originally
calibrated on the correctness grid -- whose longest token reduction is 32 terms
and whose largest tensor has 4096 elements -- and on the captured layer-14
invocation, whose real gradients have magnitude ~1e-5 and therefore pass any
absolute tolerance vacuously. They did **not** cover the observed benchmark
shape, where a weight gradient sums 4096 tokens into 3 million elements. Both
tiers refused it, on the same result names, for the same reason.

### What was measured

`calibrate inventory` runs the declaration's own float32-accumulated oracle
against `runtime_forward` -- the exact Transformers spelling -- from identical
inputs and identical upstream gradients, both differentiated by autograd. That
pair is the floor: no implementation of these semantics can agree with the
oracle more closely than PyTorch's own does, so a tolerance below it would
reject PyTorch. Neither half is a candidate.

Measured on a GH200 over four repeats per case. **Run-to-run noise was exactly
0.0 for every result of all three operators**, so nothing here is gating on
jitter. Two error families, cleanly separated:

| family | results | short by | failing elements |
| --- | --- | --- | --- |
| no token reduction | `q`, `k`, `dx`, `dk`, `out` | 1.1x - 1.45x | 1 in 10^5 to 10^7 |
| token reduction | the eight weight gradients | 7.5x - 13.7x | 4% - 8% |

A separate sweep held every width fixed and moved only the token axis, which
isolates reduction length from element count. The measured exponent of required
atol against reduction length: **0.49-0.60 for the QKV projections and per-head
norms, 0.35-0.39 for the MLP, and 0.032 for attention's `do_weight`** -- flat,
because attention's output is a softmax-weighted average whose scale does not
grow with the number of terms.

The synthetic input distribution is **not** the cause, and the harvested
layer-14 tensors prove it rather than refute it: they pass, but only because
their gradients are ~1e-5 and the absolute tolerance dwarfs them. Their
structure is the same -- a sum over tokens of near-zero-mean products, where
some elements land near zero while their error does not. A model-scale input
generator would not change that, so none was written.

### The formula

`opdecl/tolerance.py::ReductionScaledAtol`, declared per operator:

    atol *= 1 + gain * max(0, sqrt(N / N_a) * sqrt(log M / log M_a) - 1)
    rtol  unchanged

`N` is the length of the result's declared reduction and `M` its element count;
`N_a`/`M_a` are the same quantities at an **anchor** workload, which is one of
the declaration's own bfloat16 correctness cases. The first term is the random
walk -- summing `N` independently rounded terms moves value and error alike by
`sqrt(N)`. The second is extreme-value growth: `allclose` is a maximum over
elements, and the maximum of `M` bounded-variance samples grows like
`sqrt(log M)`.

`gain` multiplies the **excess over the anchor**, not the tolerance. So the
factor is exactly 1.0 at the anchor and clamped to 1.0 below it: adding this to
an already-calibrated declaration cannot loosen anything that passed before.
Measured across all three declared correctness grids, the largest widening is
**1.044x**, on the one attention case whose `do_weight` has more elements than
the anchor's.

| operator | anchor | reduction dims | gain |
| --- | --- | --- | --- |
| `qwen3_swiglu_mlp` | B2 T16 H32 I96 | `dgate/dup/ddown_weight` over (B, T) | 2.0 |
| `qwen3_qkv_norm_rope` | B2 T32 H64 HQ4 HK2 D16 | `dq/dk/dv_weight` over (B, T); `dq/dk_norm_weight` over (B, T, HQ/HK) | 2.0 |
| `qwen3_attention` | B2 T32 HQ4 HK2 D16 H48 | none -- exponent measured at 0.032 | 2.5 |

Attention's `do_weight` multiplier also went 5.4 -> 6.5, the one constant that
moved. That costs its correctness grid a 1.2x looser gate for that gradient
(measured margin 1.47x -> 1.77x) and is the whole of the change to the declared
constants; no base tolerance was inflated and no other operator was touched.

### Remaining margin

Post-calibration, every population passes on both tiers. Worst required scale
per operator (1.0 means the gate is exactly tight):

| operator | correctness | observed | harvested layer-14 |
| --- | --- | --- | --- |
| `qwen3_qkv_norm_rope` | 0.741 | 0.848 | 0.273 |
| `qwen3_attention` | 0.581 | 0.735 | 0.396 |
| `qwen3_swiglu_mlp` | 0.786 | 0.481 | 0.263 |

So the thinnest margin anywhere is **1.18x**, on the QKV observed shape, and the
correctness numbers are byte-identical to what they were before the hook existed.
The reduction gradients are over-provisioned by 3.0x-3.8x, which is the price of
one law covering a 128-fold range of reduction lengths rather than a lookup
table; it is reported rather than tuned away.

### Negative controls

A widened tolerance is only defensible if the widened gate still rejects real
defects, so every changed family gets an injected fault
(`negative_control.py`). Measured at the observed shape, all three operators:

* **scaled fault** -- a uniform relative error on one result. Caught at
  **1% to 5%** for every result of every operator. `rtol` was deliberately left
  untouched by the calibration, and this is why: detection is `rtol`'s job,
  and `atol` only covers the reduction's noise floor.
* **dropped-token fault** -- one token zeroed so the reduction sums one fewer
  contributing term, recomputed through the full forward and backward. This is
  the error a reduction-scaled `atol` is most at risk of hiding. **Caught on
  every result of all three operators.**

The artifacts are `results/qwen3-level4/l2-calibration.json` and
`l2-negative-controls.json`; neither stores a tensor.

## Tier 3: the four drop-in sites

The canonical step is now a tier-3 workload, `Qwen3Workload`, with four places a
kernel can be swapped:

| site | operator | modules | invocations per step |
| --- | --- | --- | --- |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | `Qwen3Attention` | 28 |
| `attention` | `qwen3_attention` | the same `Qwen3Attention` | 28 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | `Qwen3MLP` | 28 |
| `residual_rmsnorm` | `fused_add_rms_norm` | decoder layers + `model.norm` | 56 |

### Nothing is replaced

Every adapter rebinds `forward` on the *existing* module and leaves
`_parameters`, `_buffers` and `_modules` untouched. The parameters are not
copied, they are the same objects, so `state_dict` keys and ordering, parameter
count, dtype, device and `requires_grad` are identical by construction rather
than by assertion -- and patching is reverted by rebuilding the workload.

The two attention sites share **one** adapter on one module. Two sequential
replacements would have meant two wrappers and an ordering question; one adapter
with two independent switches has neither, and an unselected site runs the exact
production spelling.

### The residual fusion is a cross-module boundary

`fused_add_rms_norm` is the only site that is not one module. Its add is at the
end of decoder layer *i* and its norm at the start of layer *i+1*, so the layer
stops performing the add and hands the pair to the next fusion site through a
carrier. Both returned outputs are used: `normalized` continues into the next
sublayer, `summed` **is** the residual stream -- it is never recomputed, and the
fusion is never faked as `fused_add_rms_norm(summed, zero, ...)`.

    28  attention residual -> post_attention_layernorm
    27  MLP residual       -> the next layer's input_layernorm
     1  final MLP residual -> model.norm
    --
    56

Layer 0's `input_layernorm` has no preceding decoder residual add and stays
unfused. That is the whole of the difference between 56 and 57.

What flows between layers therefore becomes the un-added branch rather than the
residual stream. `output_hidden_states` is the one caller that would read it, and
the adapter refuses that mode rather than reporting a tensor that no longer means
what its name says. So is cache-enabled execution.

### Two identity controls, deliberately distinct

**Structural identity** installs every adapter with the production spelling: the
same submodules in the same order, through native autograd. The module structure
changes and the arithmetic does not, so the demand is bitwise. It is *not* the
recomputing tier-3 identity control.

**Bound-pair identity** installs the same adapters with the declared operator
behind `opdecl.bind` -- the deployment path an evolved kernel takes -- and is
gated by the declared tolerances plus the observed-shape preflight.

### What the canonical validation found

Measured on a GH200 at the canonical batch 2 x sequence 2048:

| | structural | bound-pair |
| --- | --- | --- |
| logits | **bitwise identical** | within tolerance |
| loss | **bitwise identical** | 3.24e-05 absolute on 12.14 |
| gradient coverage | 310/310 | 310/310, **0 mismatched** |
| optimizer step | see below | 0 mismatched |
| invocation counts | 28/28/28/56 | 28/28/28/56 |
| state_dict, parameter count | identical | identical |
| inputs mutated | no | no |
| observed-shape preflight | n/a (raw callables) | passes on all four sites |

Live boundary tensors at layer 14, checked against each declaration's own
`runtime_forward` on the model's own tensors:

| site | outputs | result |
| --- | --- | --- |
| `qkv_norm_rope` | q, k, v | **bitwise** |
| `attention` | out | **bitwise** |
| `swiglu_mlp` | out | **bitwise** |
| `residual_rmsnorm` post-attention | out, summed | summed **bitwise**; out within tolerance |
| `residual_rmsnorm` cross-layer | out, summed | summed **bitwise**; out within tolerance |
| `residual_rmsnorm` final norm | out, summed | summed **bitwise**; out within tolerance |

`summed` is bitwise everywhere, which is the load-bearing claim: it *is* the
residual stream. `out` is not, and the reason is a difference between two
spellings of the same contract rather than a defect -- `Qwen3RMSNorm` casts to
bfloat16 before multiplying by the weight and `F.rms_norm`, the declared
`runtime_forward`, does not. It is gated by the declared tolerance and passes.

**The forward is provably exact.** Logits and loss are bitwise identical for
every site individually and for all four together.

The **backward is not bitwise**, and the reason is not the restructure. The
three-way experiment that settles it runs four full steps per setting and
compares each pair:

| comparison | `deterministic_algorithms=False` | `=True` (warn_only) |
| --- | --- | --- |
| logits and loss, every pairing | **bitwise** | **bitwise** |
| eager vs eager | 255 / 310 gradients differ | 223 / 310 |
| structural vs structural | 234 / 310 | 234 / 310 |
| eager vs structural (draw a) | 255 / 310 | 255 / 310 |
| eager vs structural (draw b) | 222 / 310 | 256 / 310 |
| is the mismatch set stable across draws? | no | no |

The unmodified model compared against **itself** differs on most of its
gradients, by the same magnitudes as any patched configuration (embedding
gradient 3.66e-04 absolute, 1.3e-02 relative to the tensor's peak), and the
*set* of differing tensors changes from draw to draw. A systematic difference
cannot do that. `torch.use_deterministic_algorithms` does not remove it either,
because the responsible kernels -- SDPA's backward and cuBLAS's split reductions
-- have no deterministic implementation and `warn_only` lets them through.

So the root cause is the device, and eager-vs-structural is statistically
indistinguishable from eager-vs-eager. No tolerance was widened to accommodate
it: the control was run, and the finding is reported as what it is. On CPU,
where those reductions are deterministic, structural identity is bitwise end to
end -- every gradient and the optimizer step included -- which is what
`tests/test_tier3_adapters.py` asserts on a two-layer model.

The consequence for tier 3 is worth stating plainly: **a gradient-level bitwise
claim is not available on this GPU for any provider, patched or not**, and a
loss-trajectory comparison across providers inherits that floor.

## Tier 3: the whole-model correctness gate

Site preflight proves a kernel correct at its declared shapes. It cannot prove
that 140 of them assembled into a 28-layer model still train, and the failure it
misses is the expensive one: a provider right on a 4096-row grid, wrong in
composition, and fast. So a non-eager provider must clear a **whole-model gate**
before a timer starts -- and that gate needs a threshold.

The one it replaces was `atol=0.08`, applied to everything. It was not derived
from anything, it was the same number for a 151936x1024 embedding gradient and a
128-element per-head norm, and it could not tell a wrong kernel from the
device's own drift.

### Three comparisons, never conflated

| | what it measures |
| --- | --- |
| **E/E** | unmodified eager vs an independently rebuilt unmodified eager -- the hardware's run-to-run drift and nothing else |
| **E/S** | eager vs the structural adapters -- whether restructuring added anything on top of E/E |
| **S/B** | structural vs the bound pair -- what `opdecl.bind` and the declared runtime spellings cost |

Thresholds come from **E/E on the calibration seeds only**. E/S must fit inside
them, or the adapters are not proven equivalent. S/B is bounded separately: it is
a known integration drift, not noise.

### Statistics that survive near-zero elements

Maximum *relative* error is useless on a gradient: an element that should be
1e-9 and comes out 2e-9 is a 100% error and means nothing. The gated statistics
are

    rel_l2            = ||a-b||_2 / max(||b||_2, floor)
    max_abs_over_rms  = max|a-b| / max(rms(b), floor)

with cosine agreement, above-floor sign disagreement, finiteness and
element-exceedance counts reported alongside. Sign flips are counted only where
`|b|` exceeds 1e-3 of the tensor's RMS -- below that a sign is not information.
Every tensor is compared in one streaming pass and dropped; the artifact carries
no tensor payload.

### Grouping and the formula

Parameters are pooled by **role** -- `embedding`, `lm_head`, `final_norm`,
`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and each of
the six projections -- so one role's 28 layers are 28 samples of one
distribution. Roles whose scales differ are never pooled.

    threshold(role, metric) = max(observed E/E maximum, THRESHOLD_FLOOR) * 2.0

The maximum rather than a quantile, because the gate must accept every correct
run. The margin of 2, because the maximum of a finite sample underestimates the
maximum of the process and the holdout seeds have to fit underneath without
having set it. The floor (`rel_l2` 1e-4, `max_abs_over_rms` 1e-2) exists so a
configuration small enough to be deterministic cannot derive a zero threshold
and demand bitwise equality; at the canonical size it binds nothing.

A provider is held to **hardware noise plus known integration drift** -- the
E/E and S/B thresholds summed per role and metric -- against both the original
eager model and the bound-pair path an evolved kernel actually replaces. Both
comparisons are reported.

### The gate

Before timing any non-eager Qwen provider, tier 3 requires:

1. the site-level tier-1 preflight, including the workload-supplied observed
   shapes;
2. every loss, logit, gradient, stepped parameter and checked optimizer state
   finite;
3. one **untimed** canonical forward/backward/AdamW step inside the calibrated
   envelope, against both references;
4. the fresh-batch loss trajectory inside its bound, over exactly the horizon it
   was calibrated for -- five steps, stored in the policy, because a bound
   measured over five is not a bound over fifty;
5. invocation counts and `PatchProvenance` matching the requested sites.

A failure sets `ok=false` and `failed_at="model_correctness"`, and nothing is
timed. The whole gate is outside every timed region.

### What it measured, and what it catches

On a GH200 (driver 595.71.05, CUDA 12.8, cuDNN 91900, torch 2.11.0+cu128,
transformers 5.16.1, TF32 off, `float32_matmul_precision=highest`, SDPA
flash/mem-efficient/math all enabled), 4 calibration seeds x 4 repeats + 2
holdout seeds x 4 repeats, one child process per cell, 4992 samples per
comparison:

| | bitwise | rel_l2 max | rel_l2 mean | max_abs_over_rms max |
| --- | --- | --- | --- | --- |
| E/E | 37.6% | 1.124e-02 | 2.342e-03 | 4.38 |
| E/S | 49.3% | 1.104e-02 | 1.880e-03 | 4.07 |
| S/B | 0% | 2.989e-02 | 1.447e-02 | 6.78 |

**E/S is indistinguishable from E/E** -- fewer differences, not more, and zero
exceedances of the E/E envelope on the calibration seeds *and* on the holdout.
The structural adapters add nothing measurable. S/B is three to eight times
larger and never bitwise, which is the integration drift the residual RMSNorm
spelling difference lives in.

Loss trajectories over the calibrated five-step horizon: E/E, E/S and S/B all
inside their bounds on both holdout seeds, with per-step deltas around 1e-4 on a
loss of ~12.

Negative controls on the holdout seeds, twelve faults, nothing tuned to them:

| fault | smallest magnitude always rejected |
| --- | --- |
| `non_finite` (an infinity in one site) | always |
| `dropped_row` (one token of 2048 zeroed at the MLP) | always, 13x over threshold |
| `grad_scale` (exact forward, backward scaled) | **2%** |
| `output_scale` (MLP output scaled) | **2%** |
| `one_role` (query projection only) | **2%** |
| `stateful` (correct for 8 calls, then 2% high) | **not rejected** |

Two honest gaps. Sub-1% systematic errors are inside the hardware's own noise at
this scale and this gate cannot see them. And the `stateful` fault -- 2% high on
the last 20 of 28 layers, correct on the first 8 -- passed: enough of the model
is right that the whole-model statistics stay inside the envelope. Neither
threshold was loosened to admit a control or tightened to catch one; doing
either would make the envelope a fit to the faults rather than a measurement of
the hardware.

### Bound to one machine

The calibration carries a fingerprint -- GPU name and capability, driver, CUDA,
cuDNN, torch, transformers, SDPA backends, TF32 and matmul precision,
determinism flags, optimizer configuration -- and refuses to apply itself
anywhere they differ. A missing calibration is also a refusal: an ungated timing
is not cheaper than no timing, it is worse.

### The residual RMSNorm spelling, still reported separately

`fused_add_rms_norm`'s declared `runtime_forward` normalizes with `F.rms_norm`
while `Qwen3RMSNorm` casts to bfloat16 before multiplying by the weight. That
difference is systematic, it appears in **S/B only** -- never in E/E or E/S --
and it is bounded in the S/B envelope rather than folded into the hardware
floor.

## Dependency

Transformers is an optional extra:

```bash
pip install 'evograd[qwen3]'      # or: pip install 'transformers>=4.51'
```

Qwen3 arrived in Transformers 4.51.0. This milestone was developed and measured
against **5.16.1** with torch 2.11.0+cu128 on a GH200.

Importing `evograd.bench.workloads.qwen3` never imports Transformers, so the core
package still works on a machine without it; the failure surfaces at
`require_transformers()` and names the extra to install.

## Reuse

The harvester this milestone is built for should call the pieces, not the script:

```python
from evograd.bench.workloads.qwen3.levels.level4.spec import CANONICAL
from evograd.bench.workloads.qwen3.levels.level4.model import build_model, make_inputs, training_step

spec = CANONICAL
model = build_model(spec)
input_ids, labels = make_inputs(spec)
outputs = training_step(model, input_ids, labels)   # forward + backward
```

`run_smoke` adds only verification and reporting on top of those three calls, so
an observer can wrap them under a hook, a tracer or a profiler without
inheriting any of the smoke harness.

## Tests

```bash
PYTHONPATH=src python -m unittest tests.qwen3.test_level4_workload
```

All of it runs on CPU. The integration case builds a two-layer Qwen3 with a
256-token vocabulary; the 0.6B model is never instantiated in a test.
