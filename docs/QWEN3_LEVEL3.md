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
  --out results/qwen3-level3/canonical.json
```

This smoke command does not run an optimizer or measure steady-state speedup.
To capture the representative decoder layer, first harvest the model:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/qwen3-level3/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.layer.capture \
  --layer 14 --harvest results/qwen3-level3/harvest.json \
  --out results/qwen3-level3/layer14.pt
```

Artifacts contain weights, call arguments, outputs, incoming gradients and
reference gradients. They are local files, not distributed with the repo.

## Whole-model evaluation (Level 3)

The Level-3 task is the whole training step, measured by Tier 3 after the
verification package's Gates 1/2/3 judge the candidates on real payloads:

```bash
evograd evaluate qwen3_0_6b --corpus C/manifest.json --out RUN --candidate CASE=prog.py
```

Candidate files must implement the corresponding declared operator.

| Site | Task | Calls per step |
| --- | --- | ---: |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | 28 |
| `attention` | `qwen3_attention` | 28 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | 28 |
| `residual_rmsnorm` | `fused_add_rms_norm` | 56 (both adds per layer; the cross-layer and final-norm fusions included) |

The captured layer above is the source of the Level-2
sites' real invocations.

See the [evaluation method and input sources](../src/evograd/evaluation/README.md)
and [the 2026-09-17 end-to-end report](experiments/benchmark_run_20260917.md).
