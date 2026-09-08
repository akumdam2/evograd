# Level 3 — one decoder layer, lifted out of the full step

Level 3 is a whole architectural block. Levels 1 and 2 ask whether a kernel is
fast in isolation; this asks whether those choices survive composition, because
what the block saves for backward becomes a whole-layer decision and a
candidate is free to fuse across the boundaries the smaller tasks draw.

## The whole-layer declaration has been removed

`llama3_decoder_layer` — a single declaration spanning the entire block — is
**no longer registered**, and neither is `af3_single_repr_block`. No task
declares level 3. The description below is kept as the record of what that
declaration asserted, because the capture and replay machinery in this package
still brackets exactly that span and the two have to agree about what a layer
is; it is not a description of anything a CLI can run today.

What replaced it is the four Level-2 boundaries the layer decomposes into, each
owned by this model — see `../level2/README.md`. Composition is still checked,
but through the capture-and-replay path in this package rather than through one
ten-operator contract.

Its shape was: level 3, family `llm_block`, dims
`(B, T, hidden, head_dim, q_out, kv_out, intermediate)`.

**Ten differentiable inputs, one output, ten gradients:**

| arg | shape | |
| --- | --- | --- |
| `x` | `[B, T, hidden]` | the residual stream |
| `input_norm_weight` | `[hidden]` | RMSNorm gain before attention |
| `q_weight` | `[q_out, hidden]` | |
| `k_weight` | `[kv_out, hidden]` | fewer heads than q, under GQA |
| `v_weight` | `[kv_out, hidden]` | |
| `o_weight` | `[hidden, q_out]` | |
| `post_norm_weight` | `[hidden]` | RMSNorm gain before the MLP |
| `gate_weight` | `[intermediate, hidden]` | |
| `up_weight` | `[intermediate, hidden]` | |
| `down_weight` | `[hidden, intermediate]` | |
| `cos`, `sin` | `[T, head_dim]` | rotary tables — **Inactive**, no gradient |
| `eps` | scalar | **Inactive** |

Output: `out` `[B, T, hidden]`.

The forward is the whole layer:

```
x → RMSNorm → Q/K/V → RoPE on Q and K → causal GQA attention → o_proj
  → + residual → RMSNorm → SwiGLU MLP (silu(gate) * up, then down)
  → + residual → out
```

RMSNorm statistics, the SiLU and the attention softmax are computed in float32.
Under GQA the key/value gradients must be summed over the query heads that
share each kv head. Both residual connections contribute to `dx`, so it
accumulates three paths.

**What is deliberately excluded, and stated in the declaration rather than left
to be discovered:** this models the *training* forward pass — no KV cache (per-
step mutable state the declaration model does not express) and no attention-
weight output.

**Correctness at this depth is handled three ways**, because composing ten
operators makes a naive gate meaningless:

- `reference_dtype="float32"` — the reference runs in float32 while the
  candidate runs in bfloat16, so the tolerance bounds the candidate's own error
  rather than the difference between two equally-rounded computations;
- float32 correctness cases at small dimensions, because a bfloat16-only gate
  cannot separate a real bug (an RMSNorm that skips its float32 upcast) from
  ordinary rounding — both land at the same magnitude;
- the upstream gradient is scaled by `1/sqrt(batch*tokens)`, which is what a
  real mean-over-tokens loss produces, and which keeps one tolerance valid
  across the token grid.

The timed grid is 1024/2048/4096 tokens; 8192 lives in untimed coverage. The
ceiling is set by the *oracle*, not the candidate — the reference builds an
autograd graph for the whole layer in float32.

## What this package adds

The declaration above is generic over its dims. What is here is the machinery
that proves one real Llama-3 layer can be lifted out of the canonical step and
replayed alone.

| file | what it does |
| --- | --- |
| `artifact.py` | the replay format, and its two hashes |
| `capture.py` | hooks one `LlamaDecoderLayer` during the canonical run and writes it |
| `prepare.py` | rebuilds the layer from the artifact, ready to be run |
| `replay.py` | reruns that layer alone, in a fresh process, and checks it against the full model — **now `evograd.evaluation.workloads.llama3_8b.level3.replay`**, because deciding whether the replay agrees is a verdict |

### `capture.py`

The layer is selected **from the harvest manifest**, not by index alone: the
manifest is checked for self-consistency, checked against the workload the
capture is about to run, and searched for an observed `decoder_layer` event at
that index. The event's module path is what the hook is installed on. So the
artifact cannot describe a layer the canonical run did not execute, and its
provenance points at a specific line of a specific manifest.

What is captured is everything a standalone replay needs and nothing else:

- the positional and keyword arguments Transformers really passed — including
  `attention_mask=None`, which is what sends SDPA down its causal path
- the layer's output
- the upstream gradient the **full-model** backward delivered to that output
- the gradient the layer produced for its input
- the layer's weights, and the gradient full-model backward left on each

Every one of those is detached, cloned and moved to CPU inside the hook that
sees it. Nothing keeps a view into the running graph — a capture that held one
would keep activations alive and change the memory behaviour of the run it is
describing.

**Layer 16**, mid-stack of 32, for the same reason Qwen3 uses 14 of 28: the
first and last layers of a decoder see distributions the rest do not.

### `artifact.py` — two hashes, because there are two ways to be wrong

- **`content_hash`** covers the captured numbers alone, and catches a corrupted
  or truncated file.
- **`artifact_hash`** binds those numbers to the schema version and to every
  provenance identity field, and catches a file whose *label* was edited. A
  correct layer-16 capture relabelled as layer 9, or as belonging to a different
  manifest, has intact content and a valid content hash and is exactly as wrong.

Loading verifies both, together, always, with `weights_only=True`. Anything
downstream that claims to be derived from the canonical execution goes through
`load_canonical`, which additionally checks all four identity fields
(`workload_id`, `config_hash`, `manifest_hash`, `layer_index`) against the
tracked snapshot — and **takes no argument that turns any of that off**. A
consumer cannot accidentally build a Level-2 task on a debug capture, or on the
right numbers with the wrong label.

### `replay.py`

Constructs exactly one `LlamaDecoderLayer`. It never builds `LlamaModel` or
`LlamaForCausalLM`, and the report proves it by scanning the live heap with
`gc` for instances of those classes rather than asserting it in prose — "no
such object is alive" is stronger evidence than "we did not call the
constructor".

The layer is given the captured weights and arguments, run forward, then
backward with the *real* upstream gradient the full-model backward delivered.
Output, input gradient and every parameter gradient are compared against what
the full model produced.

**Tolerances are stated, not tuned.** BF16 carries a 7-bit explicit mantissa,
so machine epsilon near 1.0 is `2^-7` and the unit roundoff is `2^-8`. The
forward path is deterministic and is held to one unit roundoff. Backward is
not: SDPA's backward accumulates with atomics, so a rerun of the same layer
differs from itself. That noise is **measured** — `--noise-repeats` replays
several times and compares the replays to each other — and reported next to the
replay-versus-capture error, so the reader can see which is larger.

Errors are relative to the scale of the reference tensor (`max|a-b| / max|b|`)
rather than elementwise.

## Running it

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.levels.level3.capture \
    --harvest results/llama3-level4/harvest.json \
    --out results/llama3-level4/layer16.pt

PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_8b.level3.replay \
    --artifact results/llama3-level4/layer16.pt \
    --report results/llama3-level4/layer16-replay.json
```

`--layer` defaults to 16; the JSON sidecar defaults to `--out` with a `.json`
suffix. `--expect-workload-id` / `--expect-manifest-hash` / `--expect-layer`
turn the identity checks into assertions with values you supply.

Capture needs the full canonical step. Everything below this level costs one
layer instead, which is the point: `layer16.pt` has already been shown to
reproduce the full model, so a derivation taken from replaying it inherits that
guarantee.
