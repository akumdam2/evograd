# Qwen3.5-4B: how to use the cases

The text model of Qwen3.5-4B at every level: the whole training step (L3) at
B1 x T2048 on one GH200, seven composite sites (L2) and ten bound primitives
(L1). Layer 14 (Gated DeltaNet) and layer 15 (gated full attention), captured
or config-derived, are where the L2 sites take their real invocations from;
they are not a level.
The package README (`src/evograd/benchmark/topdown/qwen3_5_4b/README.md`)
describes the cases; this page says how to run them and what is deferred.

Every shell, from the repository root: `conda activate evograd`, then
`export PYTHONPATH=$PWD/src`; check with
`python -c "import evograd; print(evograd.__file__)"`.
GPU work needs one 96 GB card (the numbers below are from a GH200) and 32 GB of
host memory.

## Scope

| | |
| --- | --- |
| model | text only (`Qwen3_5ForCausalLM`); no ViT, MTP, decode/KV cache, padding, FP8 |
| hardware | the timed workload is one GH200: no offload, no multi-GPU |
| shape | canonical **B1 x T2048** (B2 does not fit eagerly; B1 is the per-GPU workload of 2-GPU DDP at global batch 2) |
| baselines | recorded, no bar is chosen (eager, site and whole-model compile, FLA) |
| L3 gate | none in Tier 3: correctness comes from the verification package's Gates 1/2/3 |

## Sites, tasks and reference lines

| layer | site | task | library line |
| --- | --- | --- | --- |
| 14 | `gdn_in_proj` | `qwen3_5_gdn_in_proj` | none |
| 14 | `gated_delta_rule` | `qwen3_5_gated_delta_rule` | `fla` (`chunk_gated_delta_rule`, sidecar) |
| 14 | `gdn_norm_out` | `qwen3_5_gdn_norm_out` | `fla` (`FusedRMSNormGated` + out_proj, sidecar) |
| 15 | `qkv_norm_partial_rope` | `qwen3_5_qkv_norm_partial_rope` | none |
| 15 | `gated_attention` | `qwen3_5_gated_attention` | none |
| both | `residual_rmsnorm` | `qwen3_5_residual_rmsnorm` | none |
| both | `swiglu_mlp` | `qwen3_5_swiglu_mlp` | none |

FLA lives outside the environment (`~/.cache/evograd-fla/0.5.2`, never on
`PYTHONPATH`); only a baseline's own child imports it, after `modeling_qwen3_5`
has chosen its torch kernels (`evograd/opdecl/sidecar.py`).

## Running

**L3 smoke and harvest** (GPU):

```bash
python -m evograd.benchmark.topdown.qwen3_5_4b --out smoke.json
python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest --out harvest.json
python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot --harvest harvest.json --write
python -m evograd.benchmark.topdown.qwen3_5_4b.layer.capture --harvest harvest.json --out-dir <dir>
```

**Evaluation** (Gates 1/2/3 on real payloads, then Tier 2/3 timing):

```bash
evograd evaluate qwen3_5_4b --corpus C/manifest.json --out RUN --candidate CASE=prog.py
```

**Baseline sessions**: `bash scripts/qwen3_5_4b/session.sh <run_id>` inside a GPU
allocation; results go to
`results/experiments/qwen3_5_4b_cases/<run_id>/` and every step resumes from its
DONE marker on a new node (`--session 1|2|3` stops after that session's steps).

## cuDNN SDPA: the hazard and the policy that closes it

**Hazard** (torch 2.11.0+cu128, cuDNN 9.19, GH200). The default SDPA backend
here is cuDNN, and its backward caches an execution plan without keying on the
upstream gradient's (dO's) strides. Once a shape has run with one dO layout in a
process, a later backward at that shape with the other layout (contiguous vs the
transposed view Transformers' attention hands back) silently returns wrong
dq/dk/dv: rel_l2 ~1.4 against fp32, **in both orders**, at every head
configuration in this repository (Qwen3.5, Qwen3-0.6B, Llama-3.2-1B;
B1 and B2). q/k/v layouts do not trigger it; flash and math are unaffected. A
model run alone is fine; a process that mixes layouts at one shape is not (an L1
attention case followed by the L2 gated attention does exactly that).
cuDNN's default backward is also nondeterministic, and under
`torch.use_deterministic_algorithms(True)` cuDNN is not selectable at all.

**Policy** (`evograd.sdpa_policy`, applied in every process that imports
`evograd`; `EVOGRAD_SDPA_POLICY`, recorded in the tier-3 environment fingerprint
and the L3 smoke report):

| policy | what | correctness | attention fwd+bwd (Qwen3.5 shape) |
| --- | --- | --- | --- |
| `cudnn_guarded` (default) | cuDNN, dO made contiguous before its backward | correct, both orders, all shapes | 0.50 ms (cuDNN alone 0.46) |
| `no_cudnn` | `enable_cudnn_sdp(False)`: flash | correct | 0.75 ms |
| `torch_default` | untouched, hazard included | wrong when layouts mix | 0.46 ms |

Disabling cuDNN would have made every attention baseline 1.5-1.8x slower and
inflated attention candidates' speedups by as much; the guard costs ~9% of the
attention op and nothing measurable at L3 (1.83 s/step either way: GDN
dominates). Bitwise checks run under deterministic algorithms, which select
flash's deterministic backward: the whole Qwen3.5 L3 step is then bitwise
eager-vs-eager and identity-vs-eager (0/426 parameters differ), at ~1.8x the
step time. Policies calibrated before the change do not bind (the fingerprint now
carries cuDNN and the policy) and are recalibrated.

Evidence: `results/experiments/qwen3_5_4b_cases/20261011_sdpa/`. Probes:
`scripts/qwen3_5_4b/probes/{cudnn_sdpa_plan_cache,cudnn_sdpa_scope,sdpa_policy,
sdpa_guard,sdpa_policy_validate}.py`.

## Correctness

Tasks declare no tolerances. A candidate is judged by Gates 1/2/3 on the
model's recorded training payloads under the model reference
(`evograd.verification`). The declared tolerances calibrated here earlier (runs
`20261011_calib`, `20261013_l1calib`) and the scripts that fitted them were
removed.

Input magnitudes: the captured call's upstream gradients are ~1e-6..1e-5 (the
LM loss over a 248k vocabulary); gradients are linear in them and bfloat16 has
float32's exponent range, so generators keep unit cotangents. One magnitude
does change the numerics: `gdn_norm_out` receives `o` at rms 8.5e-5, where
mean(o^2) is far below eps = 1e-6 and the gated RMSNorm is eps-dominated. Its
generator draws `o` per token over 10**[-5, 0] and scales that token's `dy` by
sqrt(scale^2 + eps) so that `do` stays O(1) and one absolute atol is tight in
every regime.

Gradient noise to expect at L3: with nondeterministic attention backward the
whole-model `A_log` gradient differs eager-vs-eager by rel_l2 up to 4e-2 (a
sum over 2048 tokens with heavy cancellation); bitwise comparisons need
deterministic algorithms.

## Deferred

1. The L3 whole-model gate; negative controls and faults; the cross-layer
   residual fusion site.
2. Pretrained `Qwen/Qwen3.5-4B@<rev>` weights (text part) and WikiText-2 with
   the Qwen3.5 tokenizer (`weights`/`data` are reserved in the spec).
3. The GDN bar (eager, compile or FLA).
4. The optional `lm_head_ce` site.
5. Seeds and evolution.
6. The open model-scope holes of the 2026-10-06 protocol audit.
