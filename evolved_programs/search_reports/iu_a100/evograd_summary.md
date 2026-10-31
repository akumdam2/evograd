# evograd search on the IU A100: summary

2026-10-29, 08:24-13:49 (5.4 h wall for 16.0 case-hours), `session_par.sh AGENT=evograd N=3`, exit 0.
Code: 3930220 + `patches/gpu_lock.patch` + `patches/seed_unified_first.patch` (see `ADBench/evograd/NOTES.md`).
Rerun 16:51-17:12 (user's exception): the 3 seed_failed cases, on `patches/seed_fixes.patch` added; their first runs are in `<model>/superseded/`.
Per case: `bench.py run --budget 10 --time-limit 6000 -- --frontend unified --fallback-a`, gpt-5.5.

## Per model

| model | cases | submitted | seed unified | fell back to A (accepted) | seed_failed | evolve_failed | budget_exhausted | requests | wall (h) |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_0_6b | 8 | 8 | 7 | 1 (1) | 0 | 0 | 1 | 80 | 2.27 |
| llama_3_2_1b | 8 | 8 | 8 | 0 | 0 | 0 | 1 | 80 | 2.46 |
| qwen3_5_4b | 15 | 15 | 11 | 4 (4) | 0 | 0 | 1 | 150 | 4.49 |
| qwen2_vl_2b | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 60 | 1.56 |
| convnextv2_tiny | 8 | 8 | 8 | 0 | 0 | 0 | 0 | 80 | 2.38 |
| dit_s_2 | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 30 | 0.90 |
| torchrec_dlrm | 2 | 2 | 0 | 2 (2) | 0 | 0 | 0 | 20 | 0.50 |
| olmoe_1b_7b | 8 | 8 | 7 | 1 (1) | 0 | 0 | 0 | 80 | 2.23 |
| **total** | **58** | **58** | **50** | **8 (8)** | **0** | **0** | **3** | **580** | **16.81** |

## Fallbacks to A (unified gave no accepted seed)

| case | why unified failed | A |
|---|---|---|
| qwen3_0_6b cross_entropy/logits | CUDA OOM in Gate 1 (vocab 151936 fp32; the driver kept 9.8 GB from capture) | accepted, 3 requests |
| qwen3_5_4b causal_conv1d/gdn_conv1d | capture: no mapping for `convolution_backward` | accepted, 2 |
| qwen3_5_4b qwen3_5_gdn_in_proj/gdn_in_proj | capture: `convolution_backward` | accepted, 3 |
| qwen3_5_4b qwen3_5_gated_delta_rule/gated_delta_rule | capture: Inductor `assert_size_stride` | accepted, 3 |
| qwen3_5_4b qwen3_5_gated_attention/gated_attention | capture: `_scaled_dot_product_flash_attention_backward` | first run **failed** (3 attempts SIGSEGV in seed acceptance, no diagnostic, 4 requests); rerun with seed_fixes: attempt 1 failed with the declaration check's diagnostic attached, attempt 2 accepted, 3 requests |
| torchrec_dlrm embedding_bag/tables | capture: external kernel `aten._embedding_bag` | accepted, 3 |
| torchrec_dlrm torchrec_dlrm_dlrm_interaction/interaction | lowering: `aten.unbind.int` | accepted, 2 |
| olmoe_1b_7b olmoe_moe_experts/moe_experts | capture: external kernel `aten.sort.stable` | accepted, 3 |

## Other notes

- `budget_exhausted` (qwen3_0_6b fused_add_rms_norm, llama_3_2_1b rmsnorm; qwen3_5_4b gated_attention's rerun used its 10 too): one gpt-5.5 request passed OpenEvolve's 180 s LLM timeout and its retry counted, so the last iterations were refused; both submitted the best evaluated candidate.
- seed_fixes.patch: two spatial_conv2d cases (convnextv2_tiny, dit_s_2) had failed in both frontends with 0 requests: unified's Inductor capture made the convolution input channels-last (layout optimization) and the backward's stride assert failed; A traced the reference without its stride/padding/groups. With the fix both took a unified seed and ran 10 iterations. gated_attention's A seed crashed (SIGSEGV) under Gate 1 with no output; the fix adds the declaration check's report (cuDNN: head dim 256 > 128) to the repair prompt, and the rerun's second attempt was accepted.
- Checks: every case has a finished `run.json` and a `submission.json`; no `api_calls.jsonl` over 10; no aborted runs; NOTES.md lists every fallback and failure.
- Time: about 97% of wall time waits on the API (`tools/timing_report.py`). GPU lock: 648 holds, no overlap, longest wait 153 s (OpenEvolve's evaluation timeout is 900 s).
