# KernelAgent search on the IU A100: summary

2026-10-29/30. The kept runs come from three sessions: 19:40-22:10 (21 kept), 22:17-01:42 (35) and 02:49-03:50 (2 reruns), `session_par.sh AGENT=kernelagent N=3`, exit 0.
Code: 3930220 + 8 local patches in `patches/` (see `ADBench/kernelagent/NOTES.md`); KernelAgent e064717, its own code unchanged.
Per case: `bench.py run --budget 10 --time-limit 6000 -- --gpu-name "NVIDIA A100 SXM4 40GB" --test-timeout 3600` (1200 for the 22 runs of the earlier session), gpt-5.5 at the API's default reasoning effort, output caps of at least 32000 tokens.

## Per model

| model | cases | submitted | budget_exhausted | completed | generation_failed | requests | wall (h) |
|---|---|---|---|---|---|---|---|
| qwen3_0_6b | 8 | 7 | 5 | 2 | 1 | 72 | 1.71 |
| llama_3_2_1b | 8 | 7 | 7 | 0 | 1 | 79 | 2.21 |
| qwen3_5_4b | 15 | 11 | 10 | 1 | 4 | 138 | 3.18 |
| qwen2_vl_2b | 6 | 6 | 6 | 0 | 0 | 60 | 0.69 |
| convnextv2_tiny | 8 | 7 | 7 | 0 | 1 | 78 | 2.88 |
| dit_s_2 | 3 | 3 | 3 | 0 | 0 | 30 | 0.36 |
| torchrec_dlrm | 2 | 2 | 2 | 0 | 0 | 20 | 0.24 |
| olmoe_1b_7b | 8 | 7 | 7 | 0 | 1 | 79 | 3.54 |
| **total** | **58** | **50** | **47** | **3** | **8** | **556** | **14.81** |

`generation_failed`: no kernel passed KernelAgent's generation stage (2 rounds per worker under budget 10), so there is no submission. "Submitted" is not "accepted": the final gates (`bench.py evaluate`) have not been run on these runs, as for evograd's.

## The 8 cases without a submission (each worker round read)

| case | why | whose |
|---|---|---|
| qwen3_0_6b qwen3_qkv_norm_rope | 8 gradients for a 6-gradient contract, illegal memory access, numerics | kernel |
| llama_3_2_1b llama3_qkv_rope | compile errors, 6 gradients for 4, transposed output shapes | kernel |
| qwen3_5_4b gdn_in_proj | compile errors, flat output shapes, 4 outputs for 6, dtypes | kernel |
| qwen3_5_4b gated_delta_rule | compile errors, numerics (max abs err up to 7.8e4), non-finite values | kernel |
| qwen3_5_4b qkv_norm_partial_rope | 3 outputs for 4, 8 gradients for 6, numerics | kernel |
| qwen3_5_4b gated_attention | numerics, assertions, one candidate over the evaluator's 900 s | kernel |
| convnextv2_tiny convnext_mlp (stages.3) | compile errors, numerics | kernel |
| olmoe_1b_7b olmoe_qkv_norm_rope | 8 gradients for 6, illegal memory access, numerics, dtypes | kernel |

All 8 are the kernels' own errors. Two more cases, llama_3_2_1b llama3_attention and olmoe_1b_7b moe_experts, first ended generation_failed because Gate-1-correct rounds failed the driver's `kernel_function` check: the driver asks for its `kernel_function` verbatim (`torch.cat`), while KernelAgent's own prompts allow the wrapper no PyTorch compute, so every round rewrote it. With that check made a warning (`kernelagent_glue_check`) both were rerun and submitted.

## Harness problems found and fixed during this agent's runs

| problem | effect | fix (patch) | runs |
|---|---|---|---|
| Search feedback (Gate 1) passed pairs the deployment layer cannot call | smoke rmsnorm submitted, then failed Gate 2 | the search check also runs the deployment entry (`search_deployment_check`) | all |
| KernelAgent's own `reasoning_effort="high"` with 20000-token caps | responses empty (`finish_reason: length`), e.g. rope 8/8 | default effort as evograd's requests; caps raised to 32000 (`request_settings`) | first session superseded |
| Deployment layer unpacked a one-gradient backward as `dx, = ...` | `return dx`, as the statement asks, failed | either form accepted (`deployment_single_grad`) | first session superseded |
| Deployment layer passed declared scalars positionally | eps-less backwards (Gate 1 accepts them) failed: 5 cases | Gate 1's rule: named scalars by keyword (`deployment_scalars`) | 5 rerun |
| Tests waited for the GPU lock past KernelAgent's timeouts (1200 s generation, 30 s optimization, 300 s ncu) | 3 cases with rounds never evaluated | `--test-timeout 3600`, used for optimization tests and ncu too (`kernelagent_timeouts`) | 3 rerun |
| The driver's test failed a Gate-1-correct pair whose `kernel_function` KernelAgent had rewritten (its prompts bar PyTorch compute in the wrapper) | 2 cases | the mismatch is a warning (`kernelagent_glue_check`) | 2 rerun |

Not fixed, recorded: KernelAgent's kernel benchmark subprocess has its own 300 s timeout, so olmoe_attention lost 2 optimization benchmarks to lock waits (it has a submission); before the timeout fix, qwen3_0_6b qwen3_attention and qwen3_5_4b cross_entropy each lost optimization tests to the 30 s limit (both submitted). Kept, as runs with a submission are not rerun (user, 2026-10-29).

## Cost and time

- Kept runs: 556 requests, none over 10. Superseded runs: 19 runs, 146 requests. Smoke and tests: 4 runs, 31 requests.
- Time: about 7 h of wall clock for the kept runs (19:40-22:10, 22:17-01:42, 02:49-03:50), 14.8 case-hours, about 15 min per case (max 61 min); most of it waits on the API.
- GPU lock in the last session: 457 holds (235 correctness checks, 168 KernelAgent timings, 54 ncu), waits p50 9 s, p95 1075 s, max 2410 s; the long waits come from large candidates that run up to the evaluator's 900 s while holding the lock.
