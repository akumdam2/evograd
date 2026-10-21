# OLMoE-1B-7B

[allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924):
16 decoder layers, each a multi-head attention block (16 query heads over 16
KV heads of 128, full-width RMSNorm on the query and key projections, RoPE at
theta 10,000) and a routed mixture of 64 SwiGLU experts (2048 -> 1024 -> 2048)
of which 8 serve every token. Hidden 2048, vocabulary 50,304 (untied).
6.92B parameters, 1.28B of them active per token.

`architecture.py` is the published `config.json` field by field, with two
stated departures: `rms_norm_eps` (which the file omits) at the library default,
and `output_router_logits` pinned on so the load-balancing auxiliary loss is in
the step. The published file is at
`/projects/bcjw/akumdam/models/OLMoE-1B-7B-0924/config.json`.

## Levels

### L3 -- the whole training step

`OlmoeForCausalLM`, random weights from a seed (through the model's own
`post_init`), synthetic tokens, bf16, SDPA, **grouped-GEMM experts**,
`use_cache=False`, no gradient checkpointing, train mode, at **B2 x T2048**.
The experts kernel is a hashed spec field (`experts_implementation`,
`grouped_mm` or `eager`), pinned the way the attention backend is and read
back off the built model; an unmodified `OlmoeForCausalLM` on this torch
dispatches to the grouped GEMM, so that is the reference execution and the
per-expert loop is the alternative. Single card, no offload, no multi-GPU. The
Tier-3 workload adds AdamW. No whole-model correctness gate yet: every L3
provider row carries `evaluation_complete: false` and the reason.

Batch 2 is the repository's usual and is an estimate until the first GPU
session: about 55 GiB of weights, gradients and bf16 moments plus 15-20 GiB of
activations on a 96 GB GH200 (`levels/level3/spec.py`). If it does not fit, the
batch falls to 1 as Qwen3.5-4B's did.

### Layer 8 -- where L2 takes its real invocations

Not a level: the layer the L2 sites' real invocations and the calibration's
real population come from. Boundary
`hidden_out = layer(hidden_states, position_embeddings)`, cotangent `g`. Two
sources: **captured** (`layer/capture.py`, `artifact.py`: one canonical L3 step
with layer 8 hooked -> `layer8.pt`, which also records the experts kernel the
step ran) and **config-derived** (`layer/prepare.py`: the bare
`OlmoeDecoderLayer` with the model's initialization rules applied explicitly
-- a standalone layer skips `post_init`, which leaves the router at zero and
the expert tensors uninitialized -- and seeded inputs).

### L2 -- four sites

| site | task | contract |
| --- | --- | --- |
| `qkv_norm_rope` | `olmoe_qkv_norm_rope` | q/k/v projections, RMSNorm over the **whole** q (2048) and k (2048) projections, RoPE over the full head -- **new contract** (Qwen3 normalizes per head, Llama not at all) |
| `attention` | `olmoe_attention` | causal SDPA 16/16 D128 + o_proj |
| `residual_rmsnorm` | `olmoe_residual_rmsnorm` | residual add + RMSNorm 2048, two outputs |
| `moe_experts` | `olmoe_moe_experts` | the routed experts: `x [4096, 2048]`, `gate_up_proj [64, 2048, 2048]`, `down_proj [64, 2048, 1024]`, the router's `top_k_index`/`top_k_weights [4096, 8]` -> `out`; the operator suite's `fused_moe_swiglu` boundary at this model's widths, under this model's key |

Each site package has `task.py`, `reference.py` (float32 oracle and the
production spelling, bitwise the layer's own -- for the experts, Transformers'
own `grouped_mm_experts_forward` on the declared tensors) and `capture.py` (one
real invocation, from a layer artifact when present, else a config-derived
layer). Canonical cases are config-derived (`hf_config`, B2 T2048). Whole-model
call counts are **observed** (the harvest snapshot's `observed` entries). The
router (one `[T, E]` GEMM, softmax, top-k) stays native and supplies the
experts site's routing tensors.

**Tolerances are provisional**: every site carries the repository's standard
float32/bfloat16 pair and says so (`PROVISIONAL_TOLERANCES = True`). The
calibration inventory is the first GPU session's job; the experts' weight
gradients, which sum over every token an expert receives, are the likeliest to
need a reduction-scaled gate.

### L1 -- primitives

No new primitive. `levels/level1/manifest.py` binds six existing ones under
`olmoe_1b_7b_observed`: `linear_no_bias` (the attention projections and the
head; the expert GEMMs are 3-D parameters, not `nn.Linear`), `rmsnorm` (the
residual norms and the full-width q/k norms), `rope`, `causal_gqa_attention`,
`swiglu` (the experts' activation on the packed `[32768, 1024]` tensor) and
`cross_entropy`. The cases appear when `harvest/snapshot.json` is tracked.

## Harvest

`harvest/` observes one whole canonical step (loss + backward) and freezes it
into `harvest/snapshot.json` (provenance `observed`). Every layer is held to
the decoder block's mandatory boundaries -- attention, RoPE, SDPA, the MoE
block, its router, its experts and their SiLU -- and the per-task counts to
the depth; `snapshot.check_declared_dims` holds every site's declared dims to
the harvested shapes.

```bash
PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b --out smoke.json
PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest --out harvest.json
PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.snapshot --harvest harvest.json --write
PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b.layer.capture \
    --harvest harvest.json --out-dir <run_dir>
```

The Tier-3 side is `evograd/evaluation/tier3/workloads/olmoe_1b_7b/`; the
evaluation bindings are `evograd/evaluation/workloads/olmoe_1b_7b/`. How to run
everything: `docs/OLMOE_1B_7B.md`.

## Not yet done

The tolerance calibration (the first GPU session, `20261016_gpu1`, ran the
smoke, harvest, snapshot, layer capture, replay and tier-3 baselines, and
showed the provisional gates too tight for the weight-gradient reductions at
the observed size); the L3 whole-model gate; pretrained weights and real text;
seeds and evolution; the cross-layer residual fusion; an `lm_head_ce` site.
