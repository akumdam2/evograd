# Meta-Llama-3-8B benchmark workload

## Purpose

The second workload, and the one that made the first one's machinery generic.
Almost everything here is shared with Qwen3 — the spec, the builder, the
observer, the manifest, the snapshot reader and its extraction all live in
[`../common/`](../common/). What is in this package is the list of facts only
Llama can supply.

## Canonical workload

Meta-Llama-3-8B, batch 2, sequence 2048, BF16, CUDA, SDPA, `model.train()`,
`use_cache=False`, no gradient checkpointing, fixed seed, no optimizer step:

```python
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

The spec hashes to `meta-llama-3-8b.train.bs2.seq2048.bf16.cuda.sdpa.5f08b9e7`.

Sequence 2048 rather than the 8192 the architecture permits, so the observed
shapes line up with Qwen3's and the two can be read side by side.

**No Hub token is needed.** Llama-3 is a gated repository, but the architecture
is written out in `levels/level4/spec.py` and the model is built with random
weights. Nothing fetches a checkpoint, a config or a tokenizer.

## What this package contains

```
llama3/
├── declaration.py                 the Level4Workload every shared stage reads
├── cli.py                         the smoke entry point
├── levels/level4/
│   ├── spec.py                    the architecture, and the canonical settings
│   ├── model.py                   LlamaConfig / LlamaForCausalLM
│   ├── smoke.py, report.py        thin bindings of the shared versions
└── harvest/
    ├── observe.py                 which classes implement which boundary
    ├── manifest.py                schema, patched call sites, label
    └── snapshot.py                the Level-1/2 mapping, and extraction
```

Every one of those is a declaration. The code that acts on them is shared, which
is the point: a bug in the machinery is fixed once, and a bug here is a wrong
number for this model alone.

## How it differs from Qwen3

| | Qwen3-0.6B | Llama-3-8B |
|---|---|---|
| Per-head q/k RMSNorm | yes | **no** |
| `n_heads × head_dim` vs `hidden` | 2048 vs 1024 (fans out) | 4096 == 4096 |
| `rope_theta` | 1000000 | 500000 |
| `tie_word_embeddings` | true | **false** |
| Layers / vocab | 28 / 151936 | 32 / 128256 |

Two of those have consequences the code has to handle:

**No q/k norm.** Every RMSNorm in Llama is at the residual width, so the harvest
deduplicates all of them — `input_layernorm`, `post_attention_layernorm` and the
final `model.norm` — into *one* configuration. Qwen3 produces three. It also
means Qwen3's `qwen3_qkv_norm_rope` operator has no Llama counterpart.

**`n_heads × head_dim == hidden`.** `q_proj` and `o_proj` are both 4096→4096, so
the harvest merges them into a single configuration. The extraction picks one
provenance component for it; both `attn_qkv_dims` and `attn_out_proj_dims`
re-derive the same shape at these widths, so the choice cannot matter — and a
test pins that.

## State

**Implemented and exercised:** level 4 (spec, model, smoke) and the harvest
(observer, manifest, snapshot extraction). `tests/llama3/` builds a two-layer
model on CPU, runs a real harvest, and extracts a snapshot from it.

**Not present, because it must be derived from a run:** `harvest/snapshot.json`.
A snapshot comes out of a harvest and no one has executed one for Llama-3. Until
then `load_snapshot("llama_3_8b")` refuses with the commands to produce it, and
`evograd.ops` carries no `llama_3_8b_observed` suites.

```bash
python -m evograd.bench.workloads.llama3.harvest.harvest \
    --out results/llama3-level4/harvest.json
python -m evograd.bench.workloads.llama3.harvest.snapshot \
    --harvest results/llama3-level4/harvest.json --write
```

Llama-3-8B in BF16 is ~16 GiB of weights and ~16 GiB of gradients before
activations. The Level-4 step takes no optimizer step, so it fits an 80–120 GiB
card; `--layers 4` shrinks it for a smoke.

**Not written:** levels 3–2–1 and the tier-3 sites. All of them consume the
snapshot, so they follow the harvest rather than preceding it:

- **Level 3** — capture and replay one decoder layer. The machinery is Qwen3's
  and mostly generic; it needs a layer index chosen from the manifest.
- **Level 2** — a `llama3_qkv_rope` operator declaration and reference
  implementation, since Llama's projection prefix has no per-head norm. The
  other three fused boundaries are structurally shared with Qwen3 and want
  those operators generalized rather than copied.
- **Tier 3** — a site registry, its adapters and a `Tier3Adapter`, after the
  operators exist. Until then `llama_3_8b` is absent from `TIER3_ADAPTERS` and
  `evograd tier3-bench` will not offer it, which is the honest state: there is
  nothing yet that can swap a kernel into this model.

## Artifact policy

Same as Qwen3's. `harvest/snapshot.json` will be tracked provenance — versioned
with the code, carrying a hash the tests assert, and the only file declarations
may read. Everything under `results/` is a local run artifact.
