# IU A100 evaluation: evograd and KernelAgent on six models

**evograd's programs pass verification far more often than KernelAgent's at Level 2 (10 of 17
against 3 of 14) and give a whole-model training speedup on 5 of the 6 models (geomean 1.052×
eager); KernelAgent gives one on 2 of 6 (1.038×).** At Level 1 both agents have most programs
accepted (evograd 13/18, KernelAgent 14/18), but evograd's accepted kernels are faster than eager
(geomean 1.32×, 1.41× against `torch.compile`) and KernelAgent's are slower (0.84× eager, 1.02×
compile). The best combinations are convnextv2_tiny (1.119×) and qwen3_0_6b (1.108× eager,
1.013× whole-model compile), both evograd. Four Level-2 cases have no verdict because model-level
Gate 2 does not fit in 40 GB (llama_3_2_1b `residual_rmsnorm` and `swiglu_mlp`, both agents).

Protocol `evograd-evaluation-v3` with the settings' defaults, one NVIDIA A100-SXM4-40GB (Jetstream2),
2026-10-30. qwen3_5_4b, olmoe_1b_7b and MegaFold are evaluated on DeltaAI
(`evolved_programs/EVAL_PLAN_DELTAAI.md`); their results go to `evaluation_results/deltaai_gh200/`.
Run log, data identities, reruns and every deviation: [NOTES.md](NOTES.md).

## What was evaluated

| model | L1 configurations | L2 configurations | evograd programs | KernelAgent programs | corpus |
|---|---|---|---|---|---|
| qwen3_0_6b | 14 | 4 | 8 | 7 | `c9ac8d22345c2cc3` |
| llama_3_2_1b | 11 | 4 | 8 | 7 | `df92518ed95a7f49` |
| qwen2_vl_2b | 16 | 2 | 6 | 6 | `45f88aba9ee8008c` |
| dit_s_2 | 10 | 2 | 3 | 3 | `f009a78c1885f701` |
| convnextv2_tiny | 31 | 4 | 8 | 7 | `7096a8fa75b2f9a2` |
| torchrec_dlrm | 7 | 1 | 2 | 2 | `51574adf12944d8c` |
| **total** | **89** | **17** | **35** | **32** | |

A configuration is one observed (operator, shape, dtype) site of the model. The agents searched
35 of them; the rest count against coverage. Programs are `evolved_programs/<agent>/` as pushed
(sha256 per program in `manifest.csv` and in `summary/cases.csv`).

## How it was measured

- **Acceptance.** Level 1: Gate 1 (each acceptance payload within tolerance of the reference) and
  operator-scope Gate 2 (errors within the envelope calibrated from `torch.compile` on the same
  payloads). Level 2 adds model-scope Gate 2 (the whole model with the site replaced, data orders
  101/202/303/404) and Gate 3 (training trajectory against two native runs and a compiled-site run).
  A program that breaks a static rule is a *violation*: neither verified nor timed.
- **Tier 2 (operator speed).** Forward + autograd backward over every recorded call of the case in
  the acceptance step (data order 303), against eager autograd and `torch.compile`
  (`dynamic=False`); 10 warmups, 3 blocks × 50 repetitions, block order randomized; speedup = sum
  of the baseline's per-call medians / the candidate's; 95% block-bootstrap interval (2000
  iterations).
- **Tier 3 (training-step speed).** One whole training step (forward, loss, backward, optimizer)
  of the model with the site(s) replaced, against the eager model; 3 warmups, 10 blocks × 10 steps,
  each provider in its own process; 95% bootstrap interval; peak-memory ratio against eager.
- **Combination (L3).** Every accepted Level-2 candidate whose single-site Tier 3 interval has its
  lower bound above 1; the combination is verified again (model Gate 2 and Gate 3) and timed
  against eager and, where the workload defines it, whole-model `torch.compile`.
- **Pooling.** Geometric mean over accepted configurations, then tasks, then families
  (`protocol.summarize`). Coverage = accepted / configurations.
- **Numerics.** TF32 off, `float32_matmul_precision="highest"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  8 CPU threads; PyTorch 2.11.0+cu128.

## Results across the six models

L1/L2 speedups are Tier 2 geomeans over accepted configurations; L3 is the training-step geomean of
the accepted combinations.

| agent | level | configurations | submitted | violation | executable | Gate 1 | accepted | coverage | × eager | × torch.compile |
|---|---|---|---|---|---|---|---|---|---|---|
| evograd | L1 | 89 | 18 | 0 | 18 | 18 | **13** | 0.146 | 1.324 | 1.406 |
| evograd | L2 | 17 | 17 | 0 | 17 | 17 | **10** | 0.588 | 1.182 | 1.280 |
| evograd | L3 | 6 models | 5 with a combination | | | | **5** | 0.833 | 1.052 (training step) | — |
| kernelagent | L1 | 89 | 18 | 0 | 18 | 18 | **14** | 0.157 | 0.842 | 1.017 |
| kernelagent | L2 | 17 | 14 | 2 | 12 | 11 | **3** | 0.176 | 1.159 | 1.734 |
| kernelagent | L3 | 6 models | 2 with a combination | | | | **2** | 0.333 | 1.038 (training step) | — |

## Per model and agent

| model | agent | L1 configs | L1 submitted | L1 Gate 1 | L1 Gate 2 | L1 accepted | L1 × eager / × compile | L2 configs | L2 submitted | L2 violation | L2 Gate 1 | L2 Gate 2 operator | L2 Gate 2 model | L2 no verdict (OOM) | L2 Gate 3 | L2 accepted | L2 × eager / × compile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3_0_6b | evograd | 14 | 4 | 4 | 4 | **4** | 2.236 / 1.271 | 4 | 4 | 0 | 4 | 3 | 4 | 0 | 3 | **3** | 1.192 / 1.123 |
| qwen3_0_6b | kernelagent | 14 | 4 | 4 | 4 | **4** | 2.605 / 1.363 | 4 | 3 | 0 | 2 | 1 | 1 | 0 | 0 | **0** | — / — |
| llama_3_2_1b | evograd | 11 | 4 | 4 | 3 | **3** | 1.912 / 1.038 | 4 | 4 | 0 | 4 | 4 | 2 | 2 | 2 | **2** | 1.058 / 0.903 |
| llama_3_2_1b | kernelagent | 11 | 4 | 4 | 3 | **3** | 0.987 / 0.470 | 4 | 3 | 0 | 3 | 2 | 0 | 2 | 0 | **0** | — / — |
| qwen2_vl_2b | evograd | 16 | 4 | 4 | 2 | **2** | 1.131 / 2.315 | 2 | 2 | 0 | 2 | 1 | 2 | 0 | 1 | **1** | 0.860 / 1.000 |
| qwen2_vl_2b | kernelagent | 16 | 4 | 4 | 4 | **4** | 0.311 / 1.791 | 2 | 2 | 0 | 2 | 1 | 2 | 0 | 1 | **1** | 0.490 / 0.579 |
| dit_s_2 | evograd | 10 | 1 | 1 | 1 | **1** | 1.047 / 1.376 | 2 | 2 | 0 | 2 | 2 | 2 | 0 | 2 | **2** | 0.806 / 1.084 |
| dit_s_2 | kernelagent | 10 | 1 | 1 | 1 | **1** | 0.377 / 0.495 | 2 | 2 | 1 | 1 | 1 | 1 | 0 | 1 | **1** | 1.186 / 2.807 |
| convnextv2_tiny | evograd | 31 | 4 | 4 | 2 | **2** | 1.042 / 1.251 | 4 | 4 | 0 | 4 | 3 | 2 | 0 | 1 | **1** | 1.029 / 0.977 |
| convnextv2_tiny | kernelagent | 31 | 4 | 4 | 1 | **1** | 3.494 / 0.798 | 4 | 3 | 1 | 2 | 0 | 0 | 0 | 0 | **0** | — / — |
| torchrec_dlrm | evograd | 7 | 1 | 1 | 1 | **1** | 0.566 / 2.378 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | **1** | 2.025 / 1.991 |
| torchrec_dlrm | kernelagent | 7 | 1 | 1 | 1 | **1** | 0.444 / 2.254 | 1 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | **1** | 2.674 / 3.208 |

## Level 3: combinations

| model | agent | cases | Gate 2 model / Gate 3 | T3 × eager [95% CI] | peak memory ratio | whole-model compile × eager | combination × whole-model compile |
|---|---|---|---|---|---|---|---|
| qwen3_0_6b | evograd | `fused_add_rms_norm/residual_rmsnorm`, `qwen3_swiglu_mlp/swiglu_mlp` | passed / passed | 1.108 [1.108, 1.109] | 0.88 | 1.094 [1.094, 1.095] | 1.013 |
| qwen3_0_6b | kernelagent | — (none qualified) | — / — | — | — | — | — |
| llama_3_2_1b | evograd | `llama3_qkv_rope/qkv_norm_rope` | passed / passed | 1.033 [1.030, 1.035] | 1.00 | 1.015 [1.013, 1.017] | 1.018 |
| llama_3_2_1b | kernelagent | — (none qualified) | — / — | — | — | — | — |
| qwen2_vl_2b | evograd | — (none qualified) | — / — | — | — | — | — |
| qwen2_vl_2b | kernelagent | `qwen2_vl_2b_text_mlp/model.language_model.layers.mlp` | passed / passed | 1.070 [1.039, 1.071] | 1.00 | — | — |
| dit_s_2 | evograd | `dit_s_2_dit_modulation/blocks.modulation` | passed / passed | 1.000 [1.000, 1.000] | 1.00 | — | — |
| dit_s_2 | kernelagent | — (none qualified) | — / — | — | — | — | — |
| convnextv2_tiny | evograd | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.2.layers` | passed / passed | 1.119 [1.117, 1.120] | 0.83 | — | — |
| convnextv2_tiny | kernelagent | — (none qualified) | — / — | — | — | — | — |
| torchrec_dlrm | evograd | `torchrec_dlrm_dlrm_interaction/interaction` | passed / passed | 1.005 [1.004, 1.009] | 1.00 | — | — |
| torchrec_dlrm | kernelagent | `torchrec_dlrm_dlrm_interaction/interaction` | passed / passed | 1.007 [1.004, 1.008] | 1.00 | — | — |

Whole-model `torch.compile` is defined only for the qwen3_0_6b and llama_3_2_1b workloads.

## Why programs were not accepted

Each submitted program counted once, at the first gate it failed (Gate 1, then operator Gate 2,
then model Gate 2, then Gate 3). "No verdict" is model Gate 2 running out of memory, not a failure.

| agent | level | submitted | accepted | violation | Gate 1 | Gate 2 operator | Gate 2 model | Gate 3 | no verdict (OOM) |
|---|---|---|---|---|---|---|---|---|---|
| evograd | L1 | 18 | 13 | 0 | 0 | 5 | 0 | 0 | 0 |
| evograd | L2 | 17 | 10 | 0 | 0 | 3 | 2 | 0 | 2 |
| kernelagent | L1 | 18 | 14 | 0 | 0 | 4 | 0 | 0 | 0 |
| kernelagent | L2 | 14 | 3 | 2 | 1 | 5 | 1 | 0 | 2 |

Every Gate 2 failure (18 operator, 7 model) is `threshold_exceeded`; the one Gate 1 failure is
`tolerance_exceeded`. The two violations are KernelAgent programs
whose public pair functions "only forward to a private `_impl` twin". Search failures (no program,
not evaluated):

| agent | model | case | search status |
|---|---|---|---|
| kernelagent | qwen3_0_6b | `qwen3_qkv_norm_rope/qkv_norm_rope` | generation_failed |
| kernelagent | llama_3_2_1b | `llama3_qkv_rope/qkv_norm_rope` | generation_failed |
| kernelagent | convnextv2_tiny | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.3.layers` | generation_failed |

## Per case

"—": not timed. Rejected candidates are not timed (see "Deviations"), except evograd's
qwen3_0_6b `qwen3_qkv_norm_rope`, which ran before that change (its numbers are informational and
in no summary).

| model | agent | case | L | verdict | T2 × eager [95% CI] | T2 × compile [95% CI] | T3 single site × eager | T3 memory ratio |
|---|---|---|---|---|---|---|---|---|
| qwen3_0_6b | evograd | `cross_entropy/logits` | 1 | accepted | 2.905 [2.904, 2.906] | 1.001 [1.001, 1.002] | — | — |
| qwen3_0_6b | evograd | `fused_add_rms_norm/residual_rmsnorm` | 2 | accepted | 1.733 [1.727, 1.740] | 1.618 [1.612, 1.627] | 1.099 [1.095, 1.104] | 0.95 |
| qwen3_0_6b | evograd | `qwen3_attention/causal_gqa_attention` | 2 | accepted | 0.903 [0.903, 0.903] | 0.885 [0.885, 0.886] | 0.987 [0.984, 0.992] | 1.00 |
| qwen3_0_6b | evograd | `qwen3_qkv_norm_rope/qkv_norm_rope` | 2 | rejected at gate2_operator | 1.840 [1.836, 1.841] | 0.826 [0.825, 0.827] | 1.183 [1.179, 1.188] | 0.95 |
| qwen3_0_6b | evograd | `qwen3_swiglu_mlp/swiglu_mlp` | 2 | accepted | 1.082 [1.082, 1.082] | 0.989 [0.988, 0.989] | 1.014 [1.011, 1.019] | 0.93 |
| qwen3_0_6b | evograd | `rmsnorm/rmsnorm` | 1 | accepted | 1.838 [1.834, 1.842] | 1.582 [1.578, 1.585] | — | — |
| qwen3_0_6b | evograd | `rope/rope` | 1 | accepted | 1.166 [1.164, 1.168] | 0.842 [0.840, 0.845] | — | — |
| qwen3_0_6b | evograd | `swiglu/mlp_activation` | 1 | accepted | 4.018 [4.012, 4.022] | 1.959 [1.949, 1.966] | — | — |
| qwen3_0_6b | kernelagent | `cross_entropy/logits` | 1 | accepted | 2.228 [2.224, 2.234] | 0.769 [0.769, 0.769] | — | — |
| qwen3_0_6b | kernelagent | `fused_add_rms_norm/residual_rmsnorm` | 2 | rejected at gate2_model | — | — | — | — |
| qwen3_0_6b | kernelagent | `qwen3_attention/causal_gqa_attention` | 2 | rejected at gate1_payloads | — | — | — | — |
| qwen3_0_6b | kernelagent | `qwen3_swiglu_mlp/swiglu_mlp` | 2 | rejected at gate2_operator | — | — | — | — |
| qwen3_0_6b | kernelagent | `rmsnorm/rmsnorm` | 1 | accepted | 2.131 [2.130, 2.132] | 1.728 [1.721, 1.739] | — | — |
| qwen3_0_6b | kernelagent | `rope/rope` | 1 | accepted | 2.421 [2.343, 2.452] | 1.540 [1.484, 1.559] | — | — |
| qwen3_0_6b | kernelagent | `swiglu/mlp_activation` | 1 | accepted | 4.006 [4.004, 4.007] | 1.686 [1.667, 1.696] | — | — |
| llama_3_2_1b | evograd | `cross_entropy/logits` | 1 | accepted | 2.295 [2.294, 2.295] | 0.814 [0.813, 0.814] | — | — |
| llama_3_2_1b | evograd | `llama3_attention/causal_gqa_attention` | 2 | accepted | 0.874 [0.874, 0.874] | 0.861 [0.860, 0.861] | 0.986 [0.984, 0.986] | 1.00 |
| llama_3_2_1b | evograd | `llama3_qkv_rope/qkv_norm_rope` | 2 | accepted | 1.280 [1.267, 1.289] | 0.948 [0.947, 0.951] | 1.031 [1.030, 1.034] | 1.00 |
| llama_3_2_1b | evograd | `llama3_residual_rmsnorm/residual_rmsnorm` | 2 | incomplete (OOM) | — | — | — | — |
| llama_3_2_1b | evograd | `llama3_swiglu_mlp/swiglu_mlp` | 2 | incomplete (OOM) | — | — | — | — |
| llama_3_2_1b | evograd | `rmsnorm/rmsnorm` | 1 | accepted | 2.345 [2.337, 2.351] | 1.496 [1.488, 1.500] | — | — |
| llama_3_2_1b | evograd | `rope/rope` | 1 | accepted | 1.299 [1.295, 1.317] | 0.919 [0.915, 0.921] | — | — |
| llama_3_2_1b | evograd | `swiglu/mlp_activation` | 1 | rejected at gate2_operator | — | — | — | — |
| llama_3_2_1b | kernelagent | `cross_entropy/logits` | 1 | accepted | 0.388 [0.388, 0.388] | 0.137 [0.137, 0.137] | — | — |
| llama_3_2_1b | kernelagent | `llama3_attention/causal_gqa_attention` | 2 | rejected at gate2_operator | — | — | — | — |
| llama_3_2_1b | kernelagent | `llama3_residual_rmsnorm/residual_rmsnorm` | 2 | incomplete (OOM) | — | — | — | — |
| llama_3_2_1b | kernelagent | `llama3_swiglu_mlp/swiglu_mlp` | 2 | incomplete (OOM) | — | — | — | — |
| llama_3_2_1b | kernelagent | `rmsnorm/rmsnorm` | 1 | accepted | 1.057 [1.057, 1.057] | 0.464 [0.463, 0.465] | — | — |
| llama_3_2_1b | kernelagent | `rope/rope` | 1 | accepted | 2.348 [2.341, 2.359] | 1.639 [1.625, 1.650] | — | — |
| llama_3_2_1b | kernelagent | `swiglu/mlp_activation` | 1 | rejected at gate2_operator | — | — | — | — |
| qwen2_vl_2b | evograd | `gelu/model.visual.merger.mlp` | 1 | rejected at gate2_operator | — | — | — | — |
| qwen2_vl_2b | evograd | `layernorm/model.visual.blocks.norm1` | 1 | rejected at gate2_operator | — | — | — | — |
| qwen2_vl_2b | evograd | `quick_gelu/model.visual.blocks.mlp.act` | 1 | accepted | 0.572 [0.567, 0.582] | 2.465 [2.438, 2.512] | — | — |
| qwen2_vl_2b | evograd | `qwen2_vl_2b_text_mlp/model.language_model.layers.mlp` | 2 | accepted | 0.860 [0.857, 0.863] | 1.000 [0.999, 1.001] | 0.971 [0.933, 0.999] | 1.00 |
| qwen2_vl_2b | evograd | `qwen2_vl_2b_vision_mlp/model.visual.blocks.mlp` | 2 | rejected at gate2_operator | — | — | — | — |
| qwen2_vl_2b | evograd | `rmsnorm/model.language_model.layers.input_layernorm` | 1 | accepted | 2.235 [2.225, 2.247] | 2.173 [2.165, 2.186] | — | — |
| qwen2_vl_2b | kernelagent | `gelu/model.visual.merger.mlp` | 1 | accepted | 0.114 [0.109, 0.116] | 2.073 [1.989, 2.181] | — | — |
| qwen2_vl_2b | kernelagent | `layernorm/model.visual.blocks.norm1` | 1 | accepted | 0.081 [0.081, 0.084] | 1.267 [1.258, 1.268] | — | — |
| qwen2_vl_2b | kernelagent | `quick_gelu/model.visual.blocks.mlp.act` | 1 | accepted | 0.626 [0.617, 0.636] | 2.734 [2.715, 2.776] | — | — |
| qwen2_vl_2b | kernelagent | `qwen2_vl_2b_text_mlp/model.language_model.layers.mlp` | 2 | accepted | 0.490 [0.489, 0.492] | 0.579 [0.578, 0.581] | 1.046 [1.042, 1.050] | 1.00 |
| qwen2_vl_2b | kernelagent | `qwen2_vl_2b_vision_mlp/model.visual.blocks.mlp` | 2 | rejected at gate2_operator | — | — | — | — |
| qwen2_vl_2b | kernelagent | `rmsnorm/model.language_model.layers.input_layernorm` | 1 | accepted | 1.608 [1.604, 1.614] | 1.434 [1.408, 1.453] | — | — |
| dit_s_2 | evograd | `dit_s_2_dit_mlp/blocks.mlp` | 2 | accepted | 0.976 [0.975, 0.976] | 0.972 [0.972, 0.972] | 0.988 [0.987, 0.988] | 0.84 |
| dit_s_2 | evograd | `dit_s_2_dit_modulation/blocks.modulation` | 2 | accepted | 0.666 [0.662, 0.671] | 1.209 [1.205, 1.213] | 1.000 [1.000, 1.000] | 1.00 |
| dit_s_2 | evograd | `spatial_conv2d/patch` | 1 | accepted | 1.047 [1.047, 1.049] | 1.376 [1.375, 1.378] | — | — |
| dit_s_2 | kernelagent | `dit_s_2_dit_mlp/blocks.mlp` | 2 | violation | — | — | — | — |
| dit_s_2 | kernelagent | `dit_s_2_dit_modulation/blocks.modulation` | 2 | accepted | 1.186 [1.170, 1.225] | 2.807 [2.802, 2.834] | 0.998 [0.998, 0.998] | 1.00 |
| dit_s_2 | kernelagent | `spatial_conv2d/patch` | 1 | accepted | 0.377 [0.377, 0.377] | 0.495 [0.495, 0.496] | — | — |
| convnextv2_tiny | evograd | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.0.layers` | 2 | rejected at gate2_model | — | — | — | — |
| convnextv2_tiny | evograd | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.1.layers` | 2 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | evograd | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.2.layers` | 2 | accepted | 1.029 [1.028, 1.031] | 0.977 [0.975, 0.979] | 1.114 [1.113, 1.116] | 0.83 |
| convnextv2_tiny | evograd | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.3.layers` | 2 | rejected at gate2_model | — | — | — | — |
| convnextv2_tiny | evograd | `gelu/convnextv2.encoder.stages.0.layers.act` | 1 | accepted | 1.010 [1.009, 1.012] | 1.300 [1.194, 1.391] | — | — |
| convnextv2_tiny | evograd | `grn/convnextv2.encoder.stages.0.layers.grn` | 1 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | evograd | `layernorm/convnextv2.embeddings.layernorm` | 1 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | evograd | `spatial_conv2d/convnextv2.embeddings.patch_embeddings` | 1 | accepted | 1.074 [1.074, 1.075] | 1.204 [1.204, 1.206] | — | — |
| convnextv2_tiny | kernelagent | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.0.layers` | 2 | violation | — | — | — | — |
| convnextv2_tiny | kernelagent | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.1.layers` | 2 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | kernelagent | `convnextv2_tiny_convnext_mlp/convnextv2.encoder.stages.2.layers` | 2 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | kernelagent | `gelu/convnextv2.encoder.stages.0.layers.act` | 1 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | kernelagent | `grn/convnextv2.encoder.stages.0.layers.grn` | 1 | accepted | 3.494 [3.339, 3.497] | 0.798 [0.769, 0.798] | — | — |
| convnextv2_tiny | kernelagent | `layernorm/convnextv2.embeddings.layernorm` | 1 | rejected at gate2_operator | — | — | — | — |
| convnextv2_tiny | kernelagent | `spatial_conv2d/convnextv2.embeddings.patch_embeddings` | 1 | rejected at gate2_operator | — | — | — | — |
| torchrec_dlrm | evograd | `embedding_bag/tables` | 1 | accepted | 0.566 [0.559, 0.576] | 2.378 [2.296, 2.438] | — | — |
| torchrec_dlrm | evograd | `torchrec_dlrm_dlrm_interaction/interaction` | 2 | accepted | 2.025 [1.992, 2.041] | 1.991 [1.705, 3.377] | 1.012 [1.011, 1.014] | 1.00 |
| torchrec_dlrm | kernelagent | `embedding_bag/tables` | 1 | accepted | 0.444 [0.442, 0.446] | 2.254 [2.242, 2.281] | — | — |
| torchrec_dlrm | kernelagent | `torchrec_dlrm_dlrm_interaction/interaction` | 2 | accepted | 2.674 [2.643, 2.683] | 3.208 [3.136, 3.244] | 1.004 [1.003, 1.006] | 1.00 |

## Reading the numbers

- **Hardware.** Speedups compare with eager and `torch.compile` on the same A100, and Gate 2/3
  were calibrated on it. The programs were also *searched* with A100 timing. Absolute times are not
  comparable with GH200 results.
- **`torch.compile` slower than eager.** On some small operators compile is far slower than eager,
  so a candidate slower than eager still beats compile: torchrec_dlrm `embedding_bag` (both
  agents), qwen2_vl_2b `quick_gelu` (both), KernelAgent's qwen2_vl_2b `gelu` and `layernorm`
  (0.11× and 0.08× eager), evograd's dit_s_2 `dit_modulation`. This lifts both agents' "× compile"
  geomeans; read these cases against eager.
- **Operator speed does not carry to the step.** Tier 3's `cpu_bound_fraction` (the share of the
  step the CPU spends submitting or blocked) is 0.81–0.98 across the six models (0.92–0.98 for the
  LLMs): the step rarely waits on the GPU, so even 2–4× kernel speedups move it by a few percent. The reverse also occurs:
  KernelAgent's qwen2_vl_2b `text_mlp` is 0.49× at operator level yet 1.046× single site.
- **Marginal selections.** evograd's dit_s_2 combination was selected on a lower bound of 1.0001
  and measures 1.0001×; torchrec_dlrm's combinations measure 1.005× and 1.007×. They pass the rule
  but carry no practical gain. evograd's torchrec_dlrm `dlrm_interaction` × compile has a wide
  interval (1.705–3.377).
- **Provisional setting.** `EVALUATION_SETTINGS.md` marks Qwen2-VL's Gate 3 (10 steps, inside
  warmup) as provisional.

## Deviations from the evaluation plan

The plan evaluates `main` unchanged. IU ran `main`'s code (`2ee0b1d` plus the two deployment
patches, byte-identical to `09783d1`) plus three patches added during the evaluation, each approved
when it was needed. None changes a threshold, factor, sample, horizon, timing parameter or cache
key. They are in [patches/](patches/); apply them in this order, one at a time.

| patch | sha256 | change | runs it affects |
|---|---|---|---|
| `skip_rejected_timing.patch` | `7568c3339677` | A candidate rejected by verification is still checked for executability (the funnel is unchanged) but is not timed in Tier 2, and Tier 3 single site takes only accepted Level-2 candidates. | Every run after qwen3_0_6b evograd. Without it, KernelAgent's rejected qwen3 attention kernel (one Triton program per query row, fp32 atomics) needed an estimated 6 h of Tier 2. |
| `release_before_tier3.patch` | `5ab371d63de5` | Free cached GPU memory in the parent before Tier 3 children start. | llama_3_2_1b kernelagent; llama_3_2_1b evograd's combination timing. |
| `release_before_tier2.patch` | `1e1ff454ed14` | Free cached GPU memory between verification and Tier 2. | llama_3_2_1b kernelagent's Tier 2; llama_3_2_1b evograd's combination rerun. |

Memory limits on 40 GB:

- llama_3_2_1b `llama3_residual_rmsnorm` and `llama3_swiglu_mlp`: model-scope Gate 2 calibration
  climbs from about 23 to 38.7 GiB and its seed-404 native-repeat and compile backward passes run
  out of memory (reproduced in a fresh process). Recorded as no verdict for both agents. Their
  operator-level gates passed. A verdict needs a larger GPU.
- Three stages ran out of memory on GPU memory cached by an earlier stage of the same process
  (qwen3_0_6b kernelagent Tier 2, llama_3_2_1b kernelagent Tier 2, llama_3_2_1b evograd combination
  Tier 3). Each was rerun after deleting only that stage's cached result; the release patches
  prevent it.

## Files

- `<model>/evaluation/<agent>/`: the protocol's output (`evaluation.json`, `cases/`, `gate2_model/`,
  `gate3_single_site/`, `tier2/`, `tier3/`, `combination/`).
- `<model>/state/`: calibrated policies and control runs, shared by the two agents of a model.
- `summary/evograd.json`, `summary/kernelagent.json`: `protocol.summarize` over the six models.
- `summary/cases.csv`: one row per model × agent × program (gates, verdict, T2/T3 with intervals
  and milliseconds, program sha256). `summary/combinations.csv`: one row per model × agent.
- `patches/`, [NOTES.md](NOTES.md).

To recompute an agent's summary, or to pool with DeltaAI's results later:

    PYTHONPATH=src python -c "from evograd.evaluation.protocol import summarize_main; summarize_main()" \
        evaluation_results/iu_a100/*/evaluation/evograd/evaluation.json
