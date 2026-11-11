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

The harvest behind the tracked snapshot (`harvest/snapshot.json`) observes the
real-payload capture's first step: the pinned checkpoint on the first WikiText-2
TRAIN batch of data order 101, built by `evograd.verification.integrations.qwen3_0_6b`
under the verification settings. The snapshot records that run (`run`), and a
capture refuses a snapshot that does not record its own run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/qwen3-level3/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot \
  --harvest results/qwen3-level3/harvest.json --write
```

Regenerate the snapshot this way whenever `harvest/snapshot.py`'s `TASK_SOURCES`
change; until then a capture refuses it, and the case list is the old one
without its removed tasks.

## Whole-model evaluation (Level 3)

The Level-3 task is the whole training step, measured by Tier 3 after the
verification package's Gates 1/2/3 judge the candidates on real payloads:

```bash
evograd evaluate qwen3_0_6b --corpus C/manifest.json --out RUN --candidate CASE=prog.py
```

Candidate files must implement the corresponding declared operator.

| Site | Task | Calls per step |
| --- | --- | ---: |
| `attention_block` | `qwen3_attention_block` | 28 (the whole `Qwen3Attention`: projections, q/k norms, RoPE, SDPA, `o_proj`) |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | 28 |
| `residual_rmsnorm` | `fused_add_rms_norm` | 56 (both adds per layer; the cross-layer and final-norm fusions included) |
| `lm_head_ce` | `qwen3_lm_head_ce` | 1 (`lm_head` plus `ForCausalLMLoss`; a candidate returns no logits) |

The Level-2 sites' real invocations are the real-payload capture's
(`python -m evograd.verification.integrations.qwen3_0_6b capture`).

See the [evaluation method and input sources](../src/evograd/evaluation/README.md)
and [the 2026-09-17 end-to-end report](experiments/benchmark_run_20260917.md).
