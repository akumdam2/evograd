# Llama-3-8B top-down benchmark

The second top-down workload, and the one that made the first one's machinery
generic. Its stable name is `llama3_8b`; the public model key — the
`Provenance.model` value and the `--model` argument — is `llama_3_8b`.

Almost everything here is shared with Qwen3: the spec, the builder, the
observer, the manifest, the snapshot reader and its extraction all live in
[`../common/`](../common/). What is in this package is the list of facts only
Llama can supply.

## Ownership

```text
llama3_8b/
├── declaration.py       L4 facts shared by top-down tools
├── levels/
│   ├── level4/          model spec, build, smoke, and report
│   ├── level3/          captured decoder-layer artifact and replay
│   ├── level2/          QKV, attention, SwiGLU, residual/RMSNorm boundaries
│   └── level1/          primitive-to-composite mapping
└── harvest/             observation, manifest, and snapshot extraction
```

Model patching and the local/numerics/gradient checks are evaluation concerns
and live under:

```text
evograd.evaluation.tier3.workloads.llama3_8b
```

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
means Qwen3's `qwen3_qkv_norm_rope` operator has no Llama counterpart, which is
why `llama3_qkv_rope` is declared in `evograd/ops/level2/llama3_qkv_rope/`.

**`n_heads × head_dim == hidden`.** `q_proj` and `o_proj` are both 4096→4096, so
the harvest merges them into a single configuration. The extraction picks one
provenance component for it; both `attn_qkv_dims` and `attn_out_proj_dims`
re-derive the same shape at these widths, so the choice cannot matter — and a
test pins that.

## Current scope

- L4: the Llama-3-8B training workload, smoke, harvest, and snapshot extraction.
- L2: `llama3_qkv_rope`, `qwen3_attention`, `qwen3_swiglu_mlp`, and
  `fused_add_rms_norm` model boundaries. 32 layers → **160 invocations**.
- L1: the primitive mapping for those boundaries.
- L3: representative decoder-layer capture/replay infrastructure only, on the
  same footing as Qwen3's — not a completed top-down L3 evolution target.

Benchmark Level describes integration scope. T1/T2/T3 are evaluation execution
contexts; `fast`/`fair` are measurement modes.

## Commands

```bash
# resolve and print the workload id and hashes; runs nothing
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b --print-spec

PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b \
  --out results/llama3-level4/canonical.json
PYTHONPATH=src python -m evograd.benchmark.topdown.llama3_8b.harvest.harvest \
  --out results/benchmark/topdown/llama3_8b/harvest.json
evograd tier3-bench --model llama_3_8b --help
```

The smoke entry point takes no subcommand: `--out` and the override flags go
directly to `python -m evograd.benchmark.topdown.llama3_8b`.

Historical `results/llama3-level4/` and external cache paths are not moved, on
the same rule Qwen3 follows. Only new runs use the new layout.

## Not present, because each must be derived from a run

In the order they unblock each other:

1. `harvest/snapshot.json`. Until it is tracked, `load_snapshot("llama_3_8b")`
   refuses with the commands to produce one, and no declaration carries a
   `llama_3_8b_observed` suite — the hooks are wired and inert, so the suites
   appear when the file lands, with no edit to any operator.

   Llama-3-8B in BF16 is ~16 GiB of weights and ~16 GiB of gradients before
   activations. The Level-4 step takes no optimizer step, so it fits an
   80–120 GiB card; `--layers 4` shrinks it for a smoke.

2. The level-3 capture (`layer16.pt`). Every level-2 derivation consumes it and
   refuses by name rather than substituting synthetic tensors.

3. The tier-3 numerics calibration, measured on the machine it will be enforced
   on and bound to it. Without one the gate refuses to time any patched
   provider, which is the correct behaviour: an ungated timing is not cheaper
   than no timing but worse.

   ```bash
   python -m evograd.evaluation.tier3.workloads.llama3_8b.calibrate run
   ```

## Not inherited from Qwen3

The simplified numerics policy (`simple.py` / `calibrate_simple.py`) and the
four-part real-text protocol (`protocol4.py`, `prediction.py`, `training.py`,
`textdata.py`). Two different reasons:

- The simplified policy was declined while its trusted reference was
  `bound_pair`, which recomputes through the same `runtime_forward` and drives
  every threshold to its floor. That anchor is no longer the default — Qwen3's
  `calibrate_simple` now takes `--trusted-reference {torch_compile,bound_pair}`
  and defaults to `torch_compile`, which is genuinely different arithmetic. The
  original objection therefore no longer applies, and this is an open decision
  rather than a settled one. It needs no harvest and no pretrained weights.
- The real-text protocol needs a pinned pretrained checkpoint and tokenizer.
  Meta-Llama-3-8B is a gated repository, so adopting it would forfeit the
  "no Hub token" property above. That is a design decision, not a port.
