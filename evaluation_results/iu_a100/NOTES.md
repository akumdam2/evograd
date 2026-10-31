# IU A100 evaluation: run notes

Companion to [README.md](README.md): what was run, on what, and everything that did not go as the
plan says. Times are UTC, 2026-10-30.

## Machine, code, programs

- **GPU:** one NVIDIA A100-SXM4-40GB, Jetstream2 instance `wzhan-evograd-a100-1028`; one evaluation
  process at a time, nothing else on the GPU.
- **Software:** PyTorch 2.11.0+cu128 (CUDA 12.8), Triton 3.6.0, Transformers 5.16.1.
- **Code:** a clone of `2ee0b1d` with `deployment_single_grad` and `deployment_scalars` applied
  (`src/evograd/pipelines/shared/artifact.py` is byte-identical to `main` at `09783d1`), plus the
  three patches in [patches/](patches/), applied during the evaluation in this order:
  `skip_rejected_timing` (06:57), `release_before_tier3` (14:06), `release_before_tier2` (14:26).
  Applied one at a time on `09783d1` they reproduce the evaluated `protocol.py` byte for byte.
- **Programs:** `evolved_programs/<agent>/<model>/*.py` at `09783d1` (unchanged since `2ee0b1d`),
  via `evolved_programs/eval_command.py`; `--tier3-blocks 10 --deadline-min 45`.
- **Settings:** code defaults (`EVALUATION_SETTINGS.md`); nothing overridden.

## Data

Every corpus is the frozen DeltaAI capture, copied verbatim (rsync) and checked shard by shard,
size and sha256, against its `manifest.json` before evaluation; qwen3_0_6b's was already on IU.
The shards were deleted after both agents finished (manifests kept); llama_3_2_1b's are still on IU.

| model | corpus | corpus sha256 | shards | size | manifest (`ADBench_payload/`) |
|---|---|---|---|---|---|
| qwen3_0_6b | `c9ac8d22345c2cc3` | `c9ac8d22345c2cc37089dcad010453039b4f190614947d262ed48e6625c54556` | 16 | 60.6 GiB | `qwen3_0_6b/20261021_main/manifest.json` |
| llama_3_2_1b | `df92518ed95a7f49` | `df92518ed95a7f49f4f23bb654188509fe5666a29e4891a11c000e68169ce668` | 16 | 55.7 GiB | `llama_3_2_1b/llama_3_2_1b_20261021/manifest.json` |
| qwen2_vl_2b | `45f88aba9ee8008c` | `45f88aba9ee8008c24d8e7df522c2d696241a2ce0d11439243544d850cdd1756` | 3 | 11.5 GiB | `qwen2_vl_2b/qwen2_vl_2b_20261021/manifest.json` |
| dit_s_2 | `f009a78c1885f701` | `f009a78c1885f70166b3be018d75f585f54578e3d512ccbdb34f7af54c6ff71a` | 19 | 73.0 GiB | `dit_s_2/dit_s_2_20261021/manifest.json` |
| convnextv2_tiny | `7096a8fa75b2f9a2` | `7096a8fa75b2f9a27896fbf3981b49a1a3d85ea149cbb9ed8b5a6864d160c471` | 5 | 16.1 GiB | `convnextv2_tiny/convnextv2_tiny_20261021/manifest.json` |
| torchrec_dlrm | `51574adf12944d8c` | `51574adf12944d8ca0870d96dd0bf819d3451db078214edbc4c8fedab8cdd7ed` | 1 | 0.6 GiB | `torchrec_dlrm/torchrec_dlrm_20261021/manifest.json` |

Other inputs, copied verbatim from DeltaAI: the data pools under
`evograd_verification/data/` (convnextv2_tiny, qwen2_vl_2b, dit_s_2 `latent_pool.pt`, torchrec_dlrm
`processed/`) and dit_s_2's frozen warm start
`evograd_verification/data/dit_s_2/batches/warm_start.pt` (131904351 bytes, sha256
`c96aa4e16af396785f3f4cfd57d2c6b12e33cf9f11145cdcd7b180f7f87cb7af`). Pretrained weights are the
local Hugging Face caches the searches used.

## Runs

| model | agent | first start | final exit | starts | what happened |
|---|---|---|---|---|---|
| qwen3_0_6b | evograd | 04:31:41 | 05:24:16 | 1 (+1 resume) | Exit 3 at the 45-min deadline during the combination's Gate 3; resumed by `eval_model.sh`. |
| qwen3_0_6b | kernelagent | 05:28:26 | 06:59:28 | 3 | (a) Tier 2 `cross_entropy` eager baseline OOM with 11.5 GiB of verification's cached memory in the process; stopped, deleted that case's `tier2/` file, reran in a fresh process (peak 20.1 GB). (b) Rejected `qwen3_attention`'s Tier 2 estimated at ~6 h; stopped after 55 min; `skip_rejected_timing`; reran. |
| convnextv2_tiny | evograd | 07:07:49 | 07:13:59 | 1 | |
| convnextv2_tiny | kernelagent | 07:14:15 | 07:15:49 | 1 | Static-rule violation: `convnext_mlp stages.0`. |
| torchrec_dlrm | evograd | 07:16:06 | 07:20:19 | 1 | |
| torchrec_dlrm | kernelagent | 07:20:34 | 07:22:52 | 1 | |
| qwen2_vl_2b | evograd | 07:32:18 | 08:04:41 | 2 | First a one-case memory probe (`text_mlp`, peak 27.0 GiB), then the full run (peak 21.1 GiB). |
| qwen2_vl_2b | kernelagent | 08:05:22 | 08:24:30 | 1 | Peak 23.9 GiB. |
| dit_s_2 | evograd | 08:24:51 | 09:21:51 | 2 | Exit 1: `warm_start.pt` missing (not in the hand-over list); nothing model-level had been cached. DeltaAI copied it; rerun. Peak 25.2 GiB. |
| dit_s_2 | kernelagent | 09:22:10 | 09:26:35 | 1 | Static-rule violation: `dit_mlp`. |
| llama_3_2_1b | evograd | 12:52:51 | 14:33:11 | 4 | (a) `llama3_residual_rmsnorm` model Gate 2 incomplete (OOM); stopped, deleted that case's `gate2_model/` file, reran in a fresh process: same OOM, so kept as no verdict (user's decision); `llama3_swiglu_mlp` the same. (b) Combination Tier 3: the combination child OOM'd (parent held 19.8 GiB); `release_before_tier3`, deleted `tier3/combination.json`, reran after the kernelagent run. |
| llama_3_2_1b | kernelagent | 14:07:02 | 14:28:09 | 2 | Same two no-verdict cases. Tier 2 `cross_entropy` OOM in the executability check (19.2 GiB cached), which marked it not executable; `release_before_tier2`, deleted that `tier2/` file, reran. |

Only the cached result of the failed stage was deleted before each rerun; everything else was
reused from the run's own outputs. No completed result was changed.

## Model-scope Gate 2 out of memory on llama_3_2_1b

The calibration of sites `residual_rmsnorm` and `swiglu_mlp` keeps every execution's outputs and
gradients on the GPU for comparison; usage climbs from about 23 to 38.7 GiB and the seed-404
native-repeat and compile backward passes fail allocating 1.96 GiB (the float32 logits). It
reproduced in a fresh process. The same calibration also reports "the torch.compile control did
not reproduce its own results bit for bit" at seed 303, which may be a consequence of the memory
pressure; this was not separated. The policy is site-level, so both agents get no verdict for
those two cases (their Gate 1 and operator Gate 2 passed). The `attention` and `qkv_rope`
calibrations peaked at 26.4 and 29.4 GiB and completed.

## Other differences from the DeltaAI plan

- **Rejected candidates are not timed** (`skip_rejected_timing`), except in qwen3_0_6b evograd,
  which ran first: its rejected `qwen3_qkv_norm_rope` has Tier 2 and Tier 3 numbers. They appear
  in no summary. A stale Tier 2 file of KernelAgent's rejected qwen3_0_6b `fused_add_rms_norm`
  (timed before the patch) was left out of this folder; it is not referenced by `evaluation.json`.
- **`megafold` Tier 2 baseline:** the `swiglu` operator declares a third Tier 2 baseline, MegaFold's
  kernel, which fails here with `ModuleNotFoundError` (no MegaFold package). Eager and compile are
  unaffected; the error is recorded in those cases' `tier2`.
- **Deadline:** `--deadline-min 45` caused one resume (qwen3_0_6b evograd). IU has no job time limit.

## Runtime notes

- A model-scope Gate 2 execution on qwen3_0_6b takes about 7.5 s with the GPU busy under 1 s: the
  model is rebuilt with `from_pretrained` for every execution and its weights and gradients are
  hashed on the CPU (single-threaded sha256) before and after.
- Native results do not depend on the site under test: on qwen3_0_6b the 8 native Gate 3 runs have
  bit-identical loss trajectories and the native model-Gate 2 digests match across all 5 patch sets.
  Sharing them across sites would have saved about 15% of qwen3_0_6b's time.
- On a resume, single-site Gate 3 is not cached and rebuilds the model for each accepted Level-2
  case (about 30 s each).
