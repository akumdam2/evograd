# evograd / ADBench update — 2026-11-13

## Summary

1. **Tier 3 timing replays a CUDA graph (T3-G).** On `main` since 2026-11-10.
2. **L1/L2 cases rebuilt and payloads recaptured.** On branch
   `rebuild/adbench-cases-20261110`, not yet in `main`.
3. **Calibration audit in progress.** The branch is merged into `main` once the audit ends.

**`main` does not work with the new corpora.** Use the rebuild branch for new searches and
evaluations (see [Compatibility](#compatibility)).

## 1. Tier 3 timing: T3-G (on `main`)

- **Why:** the old Tier 3 timer was CPU-bound (`cpu_bound_fraction` 0.94–1.0), so Level-3
  speedups mostly measured Python and kernel-launch overhead rather than GPU kernels.
- **What T3-G does:** times the training step replayed from a CUDA graph, for eager and for every
  candidate.
- **How the combination is chosen:** a model's Level-3 combination holds the accepted L2
  candidates whose single-site T3-G speedup over eager has a 95% CI with lower bound above 1.
- **Models that cannot be captured:** MegaFold and OLMoE fall back to the old T3 interval.
- **Older results:** Tier 3 numbers measured before 2026-11-10 used the old timer. Re-time them
  before comparing.

## 2. Rebuilt cases and payloads (branch `rebuild/adbench-cases-20261110`)

### What changed

- **Case extraction:** cases for all eight models are re-extracted. The harvest and the payload
  capture come from the same real training run.
- **Attention:** a single `attention_block` case for Qwen3, Llama, OLMoE and Qwen3.5 replaces the
  old qkv/norm/rope + attention pair.
- **New L2 cases:**
  - Qwen3 LM head + loss (FLCE)
  - DiT attention and adaLN residual
  - Qwen2-VL attention and residual
  - ConvNeXt dwconv_layernorm
- **Other case changes:**
  - Residual sites cover more calls.
  - GEMMs are no longer separate search cases.
  - In total: 69 searched cases (34 L1, 35 L2) across nine models.
- **Gates:**
  - Calibration rules and safety factors (γ) are unchanged; calibration and judgement use the
    same payloads.
  - Model Gate 2 compares logits again for the language models.
- **Submission rules:**
  - Static-rule holes closed: torch aliases, torch modules used as values, auto-grants, `sys`.
  - New grants for batched, grouped and bias GEMMs (`vendor_bmm`, `vendor_grouped_mm`,
    `vendor_addmm`).
- **Evaluation:**
  - Shared native controls run once per model, on top of the existing calibration reuse.
  - `verify` exits 1 on a refused combination.
  - Results record the evograd commit and host.
- **Seeds:** regenerated. On GPU: ConvNeXt 5/5, Qwen3 8/8 and Llama 7/7 accepted.

### Where things are

| What | Where |
|---|---|
| Searched cases (with the reason each other case is left out) | `src/evograd/benchmark/search_set.json` |
| Case definitions | `src/evograd/benchmark/topdown/<model>/` |
| Payload capture and gates per model | `src/evograd/verification/integrations/<model>/` |
| Docs | `src/evograd/benchmark/README.md`, `candidates/README.md` |
| New corpora | `/work/hdd/bcjw/wzhan/ADBench_payload/<model>/<model>_20261111/manifest.json` |
| MegaFold corpus (Ajay's) | `/work/hdd/bcjw/akumdam/af3_revamp/corpus/manifest.json` |
| Our task packs, seeds and calibrated state (reference only; write to your own root) | `/work/hdd/bcjw/wzhan/ADBench_v2/<model>/` |

### How to run

```bash
git fetch origin
git switch -c rebuild origin/rebuild/adbench-cases-20261110
export PYTHONPATH=$PWD/src ADBENCH_ROOT=<your results root> ADBENCH_MODEL=gpt-5.5
M=qwen3_0_6b
CORPUS=/work/hdd/bcjw/wzhan/ADBench_payload/$M/${M}_20261111/manifest.json
python candidates/bench.py tasks     --model $M --searched --corpus $CORPUS   # task packs (GPU)
python candidates/bench.py run       --model $M --agent gepa --budget 10
python candidates/bench.py evaluate  --model $M --agent gepa --corpus $CORPUS
python candidates/bench.py summarize --model $M --agent seed gepa
```

See `candidates/README.md` for agents, budgets and the run directory layout.

## 3. Calibration audit (in progress)

- **Question:** are the gate thresholds and safety factors (γ) set well?
- **Method:** every searched case gets fault controls, where one gradient is scaled by ×1.01,
  ×1.05, ×1.1 and ×1.5. Every seed runs through the same gates.
- **Criterion:** seeds pass; ×1.05 is rejected; ×1.01 is reported as it comes out.
- **Status:** faults are done for 7 of 9 models. Qwen3.5, MegaFold and the seed column are
  running. Results will be shared when the audit ends.

## Compatibility

- **Why `main` fails:** `main` and the new corpora split the cases differently. For example,
  `main` expects `qkv_norm_partial_rope` + `gated_attention` for Qwen3.5, but the corpus has
  `attention_block`.
- **The old cases are gone too:** the tensor shards of the old `_20261021` corpora were deleted on
  2026-11-10 for disk quota (their manifests are kept), so `main` cannot evaluate the old cases
  either.
- **What to use now:** the rebuild branch with the `_20261111` corpora. It is merged into `main`
  after the calibration audit.
