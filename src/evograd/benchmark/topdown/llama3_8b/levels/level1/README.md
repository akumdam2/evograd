# Level 1 — the six primitives every observed configuration collapses onto

Level 1 is one mathematical operation plus the saved state its backward needs.
Nothing here is a Llama-specific operator: the point of this level is that the
whole canonical step, once deduplicated, is carried by **six generic tasks**
that any decoder-only model would present. What is Llama's is the mapping —
which harvested configuration becomes which generic workload, and at what dims.

Five of the six already existed and were generalized. `causal_gqa_attention` is
new, because a decoder-only training step runs *fused* causal grouped-query
attention and this benchmark had no generic task for it.

## The mapping

| generic task | family | harvested from | roles it keys on | composes into |
| --- | --- | --- | --- | --- |
| `linear_no_bias` | gemm | `linear` | `q_proj`, `k_proj`, `o_proj`, `gate_proj`, `down_proj`, `lm_head` | `llama3_qkv_rope`, `qwen3_attention`, `qwen3_swiglu_mlp` |
| `rmsnorm` | norm | `rms_norm` | `input_layernorm` | `fused_add_rms_norm` |
| `rope` | positional | `rope_apply` | `apply_rotary_pos_emb` | `llama3_qkv_rope` |
| `swiglu` | activation | `silu` | `act_fn` | `qwen3_swiglu_mlp` |
| `causal_gqa_attention` | attention | `sdpa` | `scaled_dot_product_attention` | `qwen3_attention` |
| `cross_entropy` | loss | `cross_entropy` | `fixed_cross_entropy` | — |

`COMPOSES_INTO` in `mapping.py` states that last column once, and the focused
tests check it structurally. `cross_entropy` composes into nothing: it is
reached through `lm_head` rather than through a decoder layer, so it is checked
on its own.

Only *names* live in the mapping. Every dimension comes from the harvest, and
`tests/test_provenance` proves the two agree by re-deriving each shape from the
published configuration.

## The six contracts

### `linear_no_bias` — `x [M,K]`, `weight [N,K]` → `y [M,N]`

**Biasless, not zero-biased.** Every projection and the `lm_head` in a modern
decoder sets `bias=False`, which the harvest records per configuration. They
map onto `linear_no_bias`, not the bias-carrying `linear` task — a zero bias
would be a different kernel doing avoidable work.

Deduplication is shape-driven, so Llama merges configurations that Qwen3 keeps
apart:

- `k_proj` and `v_proj` are one configuration (both 4096 → 1024)
- `gate_proj` and `up_proj` are one (both 4096 → 14336)
- **`q_proj` and `o_proj` are one**, because `n_heads * head_dim == hidden`
  (4096 == 4096). Qwen3 fans out 1024 → 2048 and keeps them separate.

That last merge is the one structural difference the extraction has to survive.
The record keeps its true role list either way, and the extraction picks one
provenance component for the merged entry; both `attn_qkv_dims` and
`attn_out_proj_dims` re-derive the same shape at these widths, so the choice
cannot matter — and a test pins that.

### `rmsnorm` — `x [rows,hidden]`, `weight [hidden]`, `eps` → `y [rows,hidden]`

Llama has **one** RMSNorm configuration. Every norm in the model is at the
residual width, so the harvest deduplicates `input_layernorm`,
`post_attention_layernorm` and the final `model.norm` into a single record —
the last carrying `null` among its layer indices, to say it belongs to no
decoder layer.

Qwen3 produces three, because of its per-head `q_norm` and `k_norm` over the
128-wide head dimension. The mapping table carries entries for those roles and
they simply never match here; an architecture without them produces a smaller
mapping rather than an error.

This is also why `COMPOSES_INTO["rmsnorm"]` has one entry and not Qwen3's two:
Llama's only RMSNorm is the decoder's, which the residual fusion owns, and
there is no per-head norm inside the projection boundary for a second edge to
point at.

### `rope` — `x [B, n_heads, T, head_dim]`, Inactive `cos`/`sin` → `y`

One harvested `apply_rotary_pos_emb` record becomes **two** generic workloads:
the call rotates the queries and the keys in a single invocation, at 32 and 8
heads respectively, and those are two different shapes doing two different
amounts of work.

`cos` and `sin` are shared tables computed once per step, so they are inputs
and receive no gradient.

### `swiglu` — `a [rows,cols]`, `b [rows,cols]` → `c [rows,cols]`

`silu(a) * b`. Harvested from the `silu` record, with the `mlp` and
`gate_proj`/`up_proj` configurations as supporting provenance for the dims.

### `causal_gqa_attention` — `q [B,HQ,T,D]`, `k [B,HK,T,D]`, `v [B,HK,T,D]` → `o [B,HQ,T,D]`

The new one. Its reference calls `F.scaled_dot_product_attention` with
`attn_mask=None`, `is_causal=True` and `enable_gqa=True` — which is what a
training step runs, and what the harvest must observe. `enable_gqa` broadcasts
the 8 key/value heads across 32 query heads rather than materializing them, so
K and V stay the size the model actually holds.

That premise is checked against the observation, not assumed: a harvest whose
SDPA call presents 32 key heads (Transformers having used `repeat_kv` instead)
is refused, because a kernel evolved against it would target tensors four times
larger than the model holds.

### `cross_entropy` — `logits [rows,cols]`, Inactive `target [rows]` int64 → `loss []`

Recorded at two boundaries, because the model crosses two. The causal loss
enters at BF16 `[2, 2048, 128256]` and reaches the flattened cross entropy at
FP32 `[4096, 128256]` — the upcast happens inside `ForCausalLMLoss`. Both are
kept so the task can be declared at whichever one it needs.

## Two deliberate non-mappings

Stated here rather than left as an absence someone has to notice:

- **There is no standalone `softmax` case.** The model runs fused SDPA and
  never materializes one, so a Llama softmax workload would be a shape no step
  executes.
- **The observed `silu` record maps onto `swiglu`, not a bare activation
  task.** The production pointwise boundary is `silu(gate) * up` — the
  activation never appears without the multiply. The SiLU record survives as
  supporting provenance.

An *unmapped* role, by contrast, is an error rather than a smaller snapshot: a
configuration the model really ran, silently dropped, would leave a task with
shapes that do not add up to the step, so a role with no component raises and
names itself.

## Running it

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level1.mapping mapping \
    --report results/llama3-level4/l1-mapping.json

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level1.mapping calibrate \
    --op causal_gqa_attention \
    --report results/llama3-level4/causal_gqa_attention-tolerance.json

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level1.mapping verify \
    --source results/llama3-level4/layer16.pt \
    --report results/llama3-level4/layer16-sdpa-verify.json

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level1.mapping loss \
    --report results/llama3-level4/l1-loss.json

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level1.mapping cross-entropy \
    --report results/llama3-level4/l1-cross-entropy.json
```

- **`mapping`** prints the six-task table with each configuration's id,
  frequency, roles and dims, read out of the tracked snapshot.
- **`calibrate`** measures what a correct implementation of one task needs, by
  comparing the declared oracle against `runtime_forward`.
- **`verify`** checks `causal_gqa_attention` against the model's own SDPA call.
  **Pass `--source` explicitly** — the canonical artifact is `layer16.pt`.
- **`loss`** is an `ln(vocab)` sanity check at the observed shape: a randomly
  initialised model over 128256 tokens predicts near-uniformly, so the expected
  loss is about `ln(128256) = 11.76`. It is a smell test, not the proof.
- **`cross-entropy`** is the proof: it compares the Level-1 contract against
  the model's own loss call.

`layer16.pt` stays the only tensor artifact. The canonical SDPA check
re-derives its tensors by replaying that artifact; the vocabulary-width check
builds its logits on demand and drops them.
