# OLMoE-1B-7B: how to use the cases

The routed mixture-of-experts decoder at every level: the whole training step
(L3) at B2 x T2048 with grouped-GEMM experts, four composite sites (L2) and six
bound primitives (L1). Layer 8, captured or config-derived, is where the L2
sites take their real invocations from. The package README
(`src/evograd/benchmark/topdown/olmoe_1b_7b/README.md`) describes the cases;
this page says how to run them and what is deferred.

Every shell, from the repository root: `export PYTHONPATH=$PWD/src`; check with
`python -c "import evograd; print(evograd.__file__)"`. GPU work needs one
96 GB card and about 16 GiB of host memory for the CPU-side initialization.

## Scope

| | |
| --- | --- |
| model | `OlmoeForCausalLM` from the published config, random weights, synthetic tokens; `output_router_logits` pinned on so the 0.01 load-balancing loss is in the step |
| experts kernel | `grouped_mm` (torch's grouped GEMM, what an unmodified model runs) is canonical; `eager` (the per-expert loop) is the alternative. A hashed spec field, read back off the built model |
| hardware | one GH200: no offload, no multi-GPU |
| shape | canonical **B2 x T2048**, confirmed on a GH200 (run `20261016_gpu1`): 28.2 GiB for the L3 step, 64.5 GiB with AdamW at Tier 3 |
| L3 gate | none in Tier 3: correctness comes from the verification package's Gates 1/2/3 |
| library lines | none |

## Sites and tasks

| site | task | what |
| --- | --- | --- |
| `qkv_norm_rope` | `olmoe_qkv_norm_rope` | projections, full-width q/k RMSNorm, RoPE |
| `attention` | `olmoe_attention` | causal SDPA 16/16 D128 + o_proj |
| `residual_rmsnorm` | `olmoe_residual_rmsnorm` | post-attention add + RMSNorm, two outputs |
| `moe_experts` | `olmoe_moe_experts` | the 64 routed SwiGLU experts, 8 per token; the router stays native |

## Running

**L3 smoke and harvest** (GPU):

```bash
python -m evograd.benchmark.topdown.olmoe_1b_7b --out smoke.json
python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest --out harvest.json
python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.snapshot --harvest harvest.json --write
python -m evograd.benchmark.topdown.olmoe_1b_7b.layer.capture --harvest harvest.json --out-dir <dir>
```

If the B2 step does not fit, `--batch-size 1` on the smoke and harvest commands
runs the B1 variant (non-canonical: the report says so).

**Evaluation** (Gates 1/2/3 on real payloads, then Tier 2/3 timing):

```bash
evograd evaluate olmoe_1b_7b --corpus C/manifest.json --out RUN --candidate CASE=prog.py
```

## Verification

Every gate applies: Gate 1 and operator Gate 2 on the recorded calls, then model
Gate 2 and Gate 3 for a Level-2 candidate. The two model gates treat routing
differently, because routers make top-k choices that tiny numerical differences flip.

- **Model Gate 2** compares the loss and every parameter gradient, so one flipped
  token would look like a large error. It runs on the native run's routing
  (`verification/integrations/olmoe_1b_7b/routing.py`): the router's probabilities
  and gradient stay live, and only its top-k choice is fixed. The load-balancing
  term counts experts from its own top-k of the live router logits, which is not
  fixed.
- **Gate 3** keeps free routing. It judges the loss curve against native and compile
  runs, and compile flips routing too, so its thresholds already hold that spread.

Both gates judge the loss Tier 3 trains, `workload.loss`: the next-token loss plus
the router's 0.01 load-balancing term.

Gate 3's optimizer is copied from Qwen3.5-4B and model Gate 2's margin is the
protocol default. Neither has been measured on this model. On a tiny random model a
10% fault at the experts fails model Gate 2 and a 1% fault passes it; how small a
fault it catches on the real model is not known yet.

## Caveats, stated up front

- **Memory.** 6.92B parameters: 13.8 GiB of bf16 weights, as much again of
  gradients, 27.7 GiB of bf16 AdamW moments, then activations. The B2 x T2048
  estimate is ~75 GiB on a 96 GB card and is unverified.
- **The eager experts loop is a correctness reference, not a timing
  baseline.** 64 small GEMM pairs per layer in Python; under it the harvest
  records one SiLU call per expert that received a token. The canonical run is
  the grouped GEMM.
- **The experts' dense oracle is expensive at the observed size**: it
  materializes `[4096, 64, 2048]` float32 intermediates (about 6 GiB with its
  backward). Fine on the card; not something to run on the login node.
- **No Level-1 primitive for the routed GEMM or the router.** `linear_no_bias`
  sees only the attention projections and the head; the expert computation is
  represented at Level 2.

## Done so far

Run `20261016_gpu1` (one GH200, 10 minutes): canonical smoke, harvest, the
tracked snapshot (`harvest/snapshot.json`), the layer-8 capture (1.6 GiB) and
its replay (output and input gradient bitwise, parameter gradients within
4e-5), tier-3 eager and structural-identity rows (241 ms/step, 0.998x), and
the four tier-2 observed cases.
The write-up is `docs/experiments/olmoe_session_20261016.md`.

## Deferred

1. The tier-3 per-site and whole-model compile rows and the tier-2 timings.
2. The L3 whole-model gate; negative controls and faults.
3. Pretrained `allenai/OLMoE-1B-7B-0924` weights and real text.
4. Seeds and evolution; the cross-layer residual fusion; an `lm_head_ce` site.
