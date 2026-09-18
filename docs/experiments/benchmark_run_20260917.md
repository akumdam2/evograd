# Qwen/Llama report-first evaluation — 2026-09-17

**The QKV + MLP + residual combination was the fastest candidate configuration
on both models. Adding the attention candidate made both slower than eager.**
All 16 configurations completed 200 training steps. Numerical mismatches were
recorded and did not stop execution.

This is a re-evaluation of existing kernels, with no generation or evolution.
See the [evaluation guide](../../src/evograd/evaluation/README.md) for the method.
Raw data are local under `results/experiments/qwen_llama_report_first/20260917/`
(`RUN` below).

## Workloads and data sources

| Setting | Qwen3-0.6B | Llama-3.2-1B |
| --- | --- | --- |
| Layers; hidden / MLP width | 28; 1024 / 3072 | 16; 2048 / 8192 |
| Query / KV heads; head dimension | 16 / 8; 128 | 32 / 8; 64 |
| Batch / sequence / dtype | 2 / 2048 / BF16 | 2 / 2048 / BF16 |
| Weights and data | Pretrained checkpoint, WikiText-2 | Seeded random weights, synthetic tokens |
| Captured block, zero-based | Layer 14 | Layer 8 |
| Full-model calls: QKV / attention / MLP / residual | 28 / 28 / 28 / 56 | 16 / 16 / 16 / 32 |

Shapes come from the pinned model configurations and task/harvest metadata.
The [L2 shape table](../../src/evograd/evaluation/README.md#qwenllama-l2-boundaries-and-canonical-shapes)
lists each site's inputs and outputs.

- **Operator prechecks:** seeded task inputs and external gradients; outputs and
  gradients compared with the declared PyTorch oracle.
- **Block:** saved weights, model arguments and incoming gradients from historical
  config-initialized model runs; native and candidate replay the same capture.
- **Whole model:** token IDs and labels enter the model; activations and backward
  gradients arise from the actual forward and LM loss. Every configuration starts
  from matching weights, optimizer state and data order.

Qwen uses `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and
WikiText-2 raw revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The tokenizer
packs text into 2048-token blocks. Llama uses separate seeded training and
validation token streams. **Llama's losses are execution diagnostics, not
language-model quality results, and cannot be compared with Qwen's losses.**

## Measurement

| Item | Protocol |
| --- | --- |
| Hardware/software | GH200; CUDA 12.8; PyTorch 2.11.0+cu128; Triton 3.6.0; Transformers 5.16.1 |
| Model settings | Full depth, SDPA, no KV cache or gradient checkpointing |
| Block timing | Forward + external-gradient backward; 5 warmups, 5 blocks × 10 samples |
| Model timing | Forward/loss + backward + AdamW + gradient reset; 5 warmups, 5 blocks × 20 steps |
| Reported time | Median of the five per-step block means; separate provider processes and recorded random order |
| Training study | Separately reset 200-step runs; AdamW lr 1e-4, betas 0.9/0.999, weight decay 0.01 |
| Validation | Steps 0/50/100/200, trained weights in the same unpatched evaluator; 64 real-text batches for Qwen, 8 synthetic batches for Llama |
| Enforcement | `report-first`: record numerical failures and continue; structural/execution failures still stop |

Compilation, loading and numerical checks are outside timing. Speedup is
current-run baseline time / candidate time; above 1 is faster. Each model has
one measurement session, so small advantages need repeat measurements.

## Whole-model results

**Execution completed for every row; this is not a correctness PASS table.**
Only Qwen attention has a matching complete whole-model policy verdict, and it
fails. The other 15 rows lack a bound verdict or bypass the standard gate.
Local checks and a separate uniform numerical comparison still ran.

`Train peak` comes from the separate 200-step training process, avoiding memory
left resident by the preceding numerical checks in the timing process.

### Qwen3-0.6B

| Configuration | Step ms | x eager | x compile | Train peak GiB | Validation NLL @200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 120.68 | 1.000 | 0.607 | 19.16 | 2.73784 |
| Whole-model compile | 73.23 | 1.648 | 1.000 | 16.76 | 2.76150 |
| QKV | 92.36 | 1.307 | 0.793 | 17.85 | 2.73746 |
| Attention B | 183.61 | 0.657 | 0.399 | 19.16 | 2.73657 |
| MLP | 124.52 | 0.969 | 0.588 | 18.51 | 2.73638 |
| Residual | 111.12 | 1.086 | 0.659 | 18.29 | 2.73693 |
| QKV + MLP + residual | 79.38 | 1.520 | 0.923 | 16.32 | 2.73994 |
| All four | 140.00 | 0.862 | 0.523 | 16.32 | 2.73914 |

### Llama-3.2-1B

| Configuration | Step ms | x eager | x compile | Train peak GiB | Synthetic validation NLL @200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 111.46 | 1.000 | 0.874 | 21.84 | 11.91815 |
| Whole-model compile | 97.40 | 1.144 | 1.000 | 20.20 | 11.91791 |
| QKV | 105.97 | 1.052 | 0.919 | 22.03 | 11.91779 |
| Attention | 161.52 | 0.690 | 0.603 | 29.83 | 11.91728 |
| MLP | 109.81 | 1.015 | 0.887 | 20.84 | 11.91755 |
| Residual | 100.06 | 1.114 | 0.973 | 20.85 | 11.91802 |
| QKV + MLP + residual | 92.84 | 1.201 | 1.049 | 20.03 | 11.91760 |
| All four | 143.35 | 0.778 | 0.679 | 28.02 | 11.91835 |

The three-site combination is 1.52× eager on Qwen and 1.20× on Llama.
Whole-model compile remains fastest on Qwen. On Llama, the combination measured
4.9% higher throughput than compile in this session.

## Numerical agreement and training observations

- **Operator prechecks:** 8/8 pass `evograd verify` on their declared grids.
  Despite the local report's "Tier-2" label, this was not a fresh target-shape
  `tier2-bench` module matrix; QKV/attention/MLP grid records omit `T=2048`.
- **Block:** all 12 patched rows exceed the forward-output envelope, while
  structural checks pass. Policies used structural/bound-pair controls, unlike
  September 15's additional site-compile control. Different verdicts do not
  imply that unchanged kernels regressed.
- **Live local checks:** all Llama rows pass. Qwen failures are QKV 30/56,
  attention 1/56, MLP 12/28, residual 0/56, three-site 42/140 and all-four 47/140.
  These local failures are outputs, not emitted gradients. Attention B's one
  violating element exceeds its allowance by about 1.2%.
- **Model comparisons:** Qwen compile has KL `1.364e-3` and gradient relative
  L2 `3.849e-2`; candidates have similar magnitudes. This is diagnostic context,
  not permission to apply the attention-only policy to other configurations.
  Llama aggregates are measured without a bound model policy.
- **Training:** all runs finish without nonfinite steps. Candidate validation
  deltas at step 200 are −0.00145 to +0.00211 on Qwen and −0.00087 to +0.00020
  on Llama. These observations do not establish long-term equivalence.
- **Open anomaly:** Qwen compile's validation loss is +0.02367 above eager, with
  a 0.45199 per-step loss difference at step 193. The cause remains unverified.

## Reproduction and remaining limits

The working tree was based on `d370611` plus uncommitted evaluation changes;
that commit alone does not reproduce this run. Candidate sources were unchanged.
Qwen attention uses Arm B instead of the historical `e27c6128c346`.

| Site | Qwen candidate SHA256 prefix | Llama candidate SHA256 prefix |
| --- | --- | --- |
| QKV | `40abfacaccd3` | `c32041fbb8e1` |
| Attention | `c4e6e7c5f673` | `dda5b0065854` |
| MLP | `896f84ea7a18` | `b0e5414d6e4a` |
| Residual | `a2fcdc137e51` | `537071c5e144` |

Under `RUN`, `inventory/INVENTORY.json` records full hashes and paths;
`block/`, `raw/`, `numerics/` and `training/` hold the measurements;
`analysis/discrepancies.json` keeps all failures; `logs/` and `run_gpu.sh`
record execution. Model keys are `qwen3_0_6b` and `llama_3_2_1b`.
Candidates, captures, policies and raw data remain local.

Remaining work is the target-shape Tier-2 module matrix, missing model-policy
verdicts, the compile loss event and memory attribution. For example, Llama's
three-site timing process peaks at 32.87 GiB, but its separate training process
uses 20.03 GiB; the difference should not be attributed to kernel storage without
isolation. Llama's full checking stage also required 192 GiB of host memory after
patched workers were killed at 96 GiB; model depth was not reduced.
