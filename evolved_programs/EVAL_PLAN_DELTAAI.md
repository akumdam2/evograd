# DeltaAI plan: evograd and KernelAgent on the models the IU A100 cannot hold

The IU A100 (40 GB) evaluates evograd and KernelAgent on six models. This plan covers the rest of
their work, on DeltaAI GH200s. It is about these two agents only. How to schedule it on Slurm
(allocations, GPUs, parallelism) is left to you, within the constraints below.

## Scope

| Model | Work | evograd programs | KernelAgent programs |
|---|---|---|---|
| `qwen3_5_4b` | evaluate | 15 | 11 |
| `olmoe_1b_7b` | evaluate | 8 | 7 |
| `qwen2_vl_2b` | evaluate **only if IU reports that it does not fit in 40 GB** | 6 | 6 |
| `megafold` (2 L1 + 2 L2, `cases.json`) | search with both agents, then evaluate | — | — |

The programs are in this folder: `<agent>/manifest.csv` (one row per case, sha256) and
`<agent>/<model>/<case>.py`. A case without a program is a search failure; list it in the
results, do not evaluate it.

## Code

- **Evaluation**: `main` at or after the commit that adds this file. It includes
  `deployment_single_grad` and `deployment_scalars`, which the evaluation's deployment calls need.
  Change nothing else. IU evaluates the same commit.
- **MegaFold search only**: a separate tree, the same commit plus, in this order, these patches
  from `patches/iu_a100/`: `gpu_lock`, `seed_unified_first`, `seed_fixes`,
  `search_deployment_check`, `request_settings`, `kernelagent_timeouts`, `kernelagent_glue_check`
  (they apply cleanly on that commit). Do not commit them.

## Evaluation (protocol `evograd-evaluation-v3`)

- All settings are the code's defaults (they match the team's `EVALUATION_SETTINGS.md`). Do not
  change thresholds, factors, horizons or timing parameters.
- Command for one agent on one model:
  `python evolved_programs/eval_command.py AGENT MODEL --adbench /work/hdd/bcjw/wzhan/ADBench --out OUT_ROOT`
  prints the `evograd.evaluation.protocol` command. Exit 3 means it stopped at its deadline:
  rerun the same command; saved stages are reused. Set `--deadline-min` to fit your allocation.
- The two agents of one model share `OUT_ROOT/<model>/state/` (calibrated policies and control
  runs): run them one after the other. Different models can run on different GPUs at once.
- One evaluation process per GPU; nothing else on that GPU while it runs.
- Corpora and data pools: the frozen ones already on DeltaAI (`cases.json[model].corpus`), read-only.
- OLMoE has no model-scope Gate 2 or Gate 3: its L2 acceptance is Gate 1 + operator Gate 2, and its
  combination cannot be an accepted L3. MegaFold has no Gate 3.

## MegaFold search (before its evaluation)

Same settings as the IU searches of these agents:

- `bench.py run --budget 10 --time-limit 6000`, model gpt-5.5, one search per GPU.
- evograd: `-- --frontend unified --fallback-a`.
- KernelAgent: `-- --test-timeout 3600` (keep the driver's default `--gpu-name`, its H100
  stand-in for GH200), with `ADBENCH_DEFAULT_REASONING=1 ADBENCH_MIN_OUTPUT_TOKENS=32000` in the
  environment (API default reasoning effort, output caps of at least 32000 tokens).
- Add the submitted programs as `<agent>/megafold/<case>.py` with rows in `<agent>/manifest.csv`
  (same columns; `site` = `deltaai_gh200`, `gpu` = the GPU), then evaluate them like the others.

## Results back to IU (through GitHub)

Push `evaluation_results/deltaai_gh200/<model>/evaluation/<agent>/` (the protocol's output) and
`evaluation_results/deltaai_gh200/<model>/state/` (the policies; skip large control checkpoints if
any), plus `evaluation_results/deltaai_gh200/NOTES.md` with: the commit evaluated, GPU, corpus
manifest per model, every rerun after exit 3, and any failure, OOM or deviation from this plan.
IU merges them into the report.

## Known limitations (state them, do not fix them)

- These programs were searched with timing feedback from an A100; their GH200 speedups say how they
  carry over, not what a GH200 search would find. Speedups compare only with eager/compile on the
  same GPU.
- `EVALUATION_SETTINGS.md` marks the Qwen3.5 Gate 3 setting as not yet measured with seeds/faults.
