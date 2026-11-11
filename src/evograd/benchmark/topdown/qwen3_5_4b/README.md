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
(`tier3.runner.OPTIMIZER_DEFAULTS`). Correctness is judged by the
verification package's Gates 1/2/3, not by Tier 3.

### L2 -- six sites

| layer kind (calls/step) | site | task | contract |
| --- | --- | --- | --- |
| GDN (x24) | `gdn_in_proj` | `qwen3_5_gdn_in_proj` | four GEMMs (q\|k\|v, z, b, a; no deinterleave), conv1d(K4)+SiLU, split, gates, q/k repeat x2 -- **new contract** |
| GDN (x24) | `gated_delta_rule` | `qwen3_5_gated_delta_rule` | chunked delta rule, in-kernel l2norm, no state; `v` a strided view |
| GDN (x24) | `gdn_norm_out` | `qwen3_5_gdn_norm_out` | gated RMSNorm(128) + out_proj 4096 -> 2560 |
| full (x8) | `attention_block` | `qwen3_5_attention_block` | the whole attention module: q+gate 2560 -> 8192, k/v 2560 -> 1024, (1+w) norm(256), RoPE 64/256, causal GQA SDPA 16/4 D256, x sigmoid(gate), o_proj 4096 -> 2560 |
| all (x64) | `residual_rmsnorm` | `qwen3_5_residual_rmsnorm` | residual add + (1+w) RMSNorm 2560, two outputs; both adds of every layer (post-attention, and the closing add fused into the next input norm or `model.norm`) |
| all (x32) | `swiglu_mlp` | `qwen3_5_swiglu_mlp` | 2560 -> 9216 -> 2560 |

Each site package has `task.py` and `reference.py` (float32 oracle and the
production spelling, bitwise the model's own). The real invocations are the
verification capture's recorded training calls
(`evograd.verification.integrations.qwen3_5_4b`); a residual call's `dsummed`
is what the next residual add delivers (zeros for the last sum, which only
`model.norm` reads). The shared mathematics is `benchmark/topdown/common/hybrid/`.
Canonical cases are config-derived
(`hf_config`, B1 T2048). Whole-model call counts are **observed** (the harvest
snapshot's `observed` entries). Every task is calibrated (L2 run
`20261011_calib`, L1 run `20261013_l1calib`; see `docs/QWEN3_5_4B.md`). Library lines: FLA (sidecar)
for `gated_delta_rule` and `gdn_norm_out`. `lm_head_ce` is deferred.

### L1 -- primitives

No new primitive. `levels/level1/manifest.py` binds ten existing ones under
`qwen3_5_4b_observed`: `linear_no_bias`, `gemma_rmsnorm`, `gated_rmsnorm`,
`causal_conv1d`, `l2norm`, `gdn_gating`, `partial_rope` (theta read from the
case's provenance), `causal_gqa_attention`, `swiglu`, `cross_entropy`
(248,320-wide rows). The cases appear when `harvest/snapshot.json` is tracked.

## Harvest

`harvest/` observes one whole training step (loss + backward), the one the
real-payload capture runs first, and freezes it into `harvest/snapshot.json`
(provenance `observed`). Every layer is held to its
layer kind's mandatory boundaries; `snapshot.check_declared_dims` holds every
site's declared dims to the harvested shapes.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src \
    python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest --out harvest.json
PYTHONPATH=src python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot --harvest harvest.json --write
```

The harvest observes the real-payload capture's first step: the pinned checkpoint
on the first WikiText-2 TRAIN batch of data order 101, built by
`evograd.verification.integrations.qwen3_5_4b` under the verification settings
(`CUBLAS_WORKSPACE_CONFIG=:4096:8`). The snapshot records that run (`run`), and
a capture refuses a snapshot that does not record its own run. `--synthetic`
harvests the random-weight, synthetic-token canonical spec instead (debug
overrides apply to it); it cannot replace the tracked snapshot.

The Tier-3 side is `evograd/evaluation/tier3/workloads/qwen3_5_4b/`. How to run
everything: `docs/QWEN3_5_4B.md`.

## Not yet done

The L3 whole-model gate; pretrained weights and real text; the GDN bar; the
`lm_head_ce` site; seeds and evolution.
