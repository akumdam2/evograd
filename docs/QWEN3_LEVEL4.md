# Qwen3-0.6B workloads

The canonical model has 28 layers, hidden size 1024, MLP width 3072,
16 query heads, 8 KV heads and head dimension 128. Defaults are batch 2,
sequence length 2048, BF16 and SDPA, with KV caching and gradient checkpointing
disabled. Seeded configuration initialization and pretrained/real-text runs
have separate workload identities.

## Whole-model reference and capture

Run the reference forward and loss backward:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b \
  --out results/qwen3-level4/canonical.json
```

This smoke command does not run an optimizer or measure steady-state speedup.
To capture the representative decoder layer, first harvest the model:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/qwen3-level4/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.levels.level3.capture \
  --layer 14 --harvest results/qwen3-level4/harvest.json \
  --out results/qwen3-level4/layer14.pt
```

Artifacts contain weights, call arguments, outputs, incoming gradients and
reference gradients. They are local files, not distributed with the repo.

## Block evaluation

Config-derived inputs do not require a model capture:

```bash
evograd tier3-bench --scope block --model qwen3_0_6b \
  --block-source config --layer-index 14 --batch 2 --tokens 2048 \
  --dtype bfloat16 --device cuda --baseline none --structural-identity \
  --out results/evaluation/tier3/qwen3_0_6b/block/config.json
```

Evaluate a candidate against a captured layer:

```bash
evograd tier3-bench --scope block --model qwen3_0_6b \
  --block-source captured --artifact results/qwen3-level4/layer14.pt \
  --layer-index 14 --device cuda --baseline none \
  --candidate swiglu_mlp=candidate.py \
  --calibrate-with structural_identity,bound_pair_identity,trusted_torch_compile \
  --out results/evaluation/tier3/qwen3_0_6b/block/candidate.json
```

Candidate files must implement the corresponding declared operator. Use
`--patch-set NAME:SITE=PATH,SITE=PATH` for a combination, or `SITE=compile` for
a compiled site. Compiling all sites this way is not whole-block compilation.

| Site | Task | Calls per block |
| --- | --- | ---: |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | 1 |
| `attention` | `qwen3_attention` | 1 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | 1 |
| `residual_rmsnorm` | `fused_add_rms_norm` | 1 |

The block retains its initial norm and final residual add. The residual fusion
site is the post-attention boundary; cross-layer and final-model norm fusions
belong to model scope.

Block timing covers forward plus backward with supplied cotangents and resets
gradients/state before every repetition. It has no optimizer or language-model
loss. Model scope retains the training-step timing and model gates.

## Policies and results

The block CLI calibrates before judging providers unless a frozen file is
supplied with `--block-policy`. Policies bind to the case and environment.
Current schema is `evograd-t3-block-policy/2`; v1 policies require recalibration.
A failed calibration cannot authorize candidate timing.

See [correctness checks](L3_CORRECTNESS_CHECKS.md) and
[the 2026-09-15 benchmark](experiments/benchmark_run_20260915.md).
