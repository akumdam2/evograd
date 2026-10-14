# Qwen3.5-4B (text model)

[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), text part only
(`Qwen3_5ForCausalLM`; no ViT, no MTP head, no decode/KV cache, no padding, no
FP8). 32 layers: 24 Gated DeltaNet (linear attention) and 8 gated full
attention (every fourth, layers 3, 7, ..., 31), each with a dense SwiGLU MLP
(2560 -> 9216). Hidden 2560, vocabulary 248,320 (tied), GDN 16 key / 32 value
heads of 128, attention 16 query / 4 KV heads of 256 with an output gate and
interleaved MRoPE over 64 of the 256 channels (theta 1e7). 4.206B parameters,
0.636B of them the embedding.

`architecture.py` is the published `text_config`, field by field
(`layer_types` included); `mamba_ssm_dtype`, `mtp_*`, `dtype` and `use_cache`
are kept apart as non-architectural.

## Levels

### L3 -- the whole training step

`Qwen3_5ForCausalLM`, random weights from a seed (through the model's own
`post_init`), synthetic tokens, bf16, SDPA, `use_cache=False`, no gradient
checkpointing, train mode, at **B1 x T2048**. Batch 1, unlike every other
workload here, because the eager step at B2 does not fit one 96 GB GH200
(the reasoning is in `levels/level3/spec.py`). Single card, no offload, no
multi-GPU. The Tier-3 workload adds AdamW
(`tier3.runner.OPTIMIZER_DEFAULTS`). No whole-model correctness gate yet:
every L3 provider row carries `evaluation_complete: false` and the reason.

### Decoder layers 14 and 15 -- where L2 takes its real invocations

Not a level (the block level was removed on 2026-10-13): the two layers the L2
sites' real invocations and the calibration's real population come from.

| layer | kind | sites |
| --- | --- | --- |
| 14 | Gated DeltaNet | `gdn_in_proj`, `gated_delta_rule`, `gdn_norm_out`, `residual_rmsnorm`, `swiglu_mlp` |
| 15 | gated full attention | `qkv_norm_partial_rope`, `gated_attention`, `residual_rmsnorm`, `swiglu_mlp` |

Boundary `hidden_out = layer(hidden_states, position_embeddings)`, cotangent
`g`; no aux loss, no routing. Two sources:

- **captured** (`layer/capture.py`, `artifact.py`): one canonical L3 step with
  layers 14 and 15 hooked -> `layer14.pt`, `layer15.pt` (weights, call
  arguments, output, upstream gradient, reference gradients; hashed content
  and identity);
- **config-derived** (`layer/prepare.py`): the bare layer with the model's
  initialization rules applied explicitly -- a standalone `Qwen3_5DecoderLayer`
  skips `post_init` (**T-init**) -- and seeded inputs.

`layer/prepare.py` pins `sdpa`, no cache, train mode, no checkpointing, and reads
back the import-time kernel dispatch of `modeling_qwen3_5` (**T-dispatch**: an
importable `fla`/`causal_conv1d` silently wins); the production spellings call
the torch functions through `__wrapped__`.

### L2 -- seven sites

| layer kind (calls/step) | site | task | contract |
| --- | --- | --- | --- |
| GDN (x24) | `gdn_in_proj` | `qwen3_5_gdn_in_proj` | four GEMMs (q\|k\|v, z, b, a; no deinterleave), conv1d(K4)+SiLU, split, gates, q/k repeat x2 -- **new contract** |
| GDN (x24) | `gated_delta_rule` | `qwen3_5_gated_delta_rule` | chunked delta rule, in-kernel l2norm, no state; `v` a strided view |
| GDN (x24) | `gdn_norm_out` | `qwen3_5_gdn_norm_out` | gated RMSNorm(128) + out_proj 4096 -> 2560 |
| full (x8) | `qkv_norm_partial_rope` | `qwen3_5_qkv_norm_partial_rope` | q+gate 2560 -> 8192, k/v 2560 -> 1024, (1+w) norm(256), RoPE 64/256 |
| full (x8) | `gated_attention` | `qwen3_5_gated_attention` | causal GQA SDPA 16/4 D256 x sigmoid(gate) + o_proj 4096 -> 2560 |
| all (x32) | `residual_rmsnorm` | `qwen3_5_residual_rmsnorm` | residual add + (1+w) RMSNorm 2560, two outputs |
| all (x32) | `swiglu_mlp` | `qwen3_5_swiglu_mlp` | 2560 -> 9216 -> 2560 |

Each site package has `task.py`, `reference.py` (float32 oracle and the
production spelling, bitwise the layer's own) and `capture.py` (one real
invocation, from a layer artifact when present, else a config-derived layer).
The shared mathematics is `benchmark/topdown/common/hybrid/`.
Canonical cases are config-derived
(`hf_config`, B1 T2048). Whole-model call counts are **observed** (the harvest
snapshot's `observed` entries). Every task is calibrated (L2 run
`20261011_calib`, L1 run `20261013_l1calib`; see `docs/QWEN3_5_4B.md`). Library lines: FLA (sidecar)
for `gated_delta_rule` and `gdn_norm_out`, Liger for `residual_rmsnorm` and
`swiglu_mlp`. `lm_head_ce` is deferred.

### L1 -- primitives

No new primitive. `levels/level1/manifest.py` binds ten existing ones under
`qwen3_5_4b_observed`: `linear_no_bias`, `gemma_rmsnorm`, `gated_rmsnorm`,
`causal_conv1d`, `l2norm`, `gdn_gating`, `partial_rope` (theta read from the
case's provenance), `causal_gqa_attention`, `swiglu`, `cross_entropy`
(248,320-wide rows). The cases appear when `harvest/snapshot.json` is tracked.

## Harvest

`harvest/` observes one whole canonical step (loss + backward) and freezes it
into `harvest/snapshot.json` (provenance `observed`). Every layer is held to its
layer kind's mandatory boundaries; `snapshot.check_declared_dims` holds every
site's declared dims to the harvested shapes.

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest --out harvest.json
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot --harvest harvest.json --write
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b.layer.capture \
    --harvest harvest.json --out-dir <run_dir>
```

The Tier-3 side is `evograd/evaluation/tier3/workloads/qwen3_5_4b/`; the
evaluation bindings are `evograd/evaluation/workloads/qwen3_5_4b/`. How to run
everything: `docs/QWEN3_5_4B.md`.

## Not yet done

The L3 whole-model gate; pretrained weights and real text; the GDN bar; the
`lm_head_ce` site; seeds and evolution; the cross-layer residual fusion.
