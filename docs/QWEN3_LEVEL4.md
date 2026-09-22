# Qwen3-0.6B workloads

The canonical model has 28 layers, hidden size 1024, MLP width 3072,
16 query heads, 8 KV heads and head dimension 128. Defaults are batch 2,
sequence length 2048, BF16 and SDPA, with KV caching and gradient checkpointing
disabled. Seeded configuration initialization and pretrained/real-text runs
have separate workload identities.

## Whole-model reference and capture

Run the reference forward and loss backward:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b \
  --out results/qwen3-level4/canonical.json
```

This smoke command does not run an optimizer or measure steady-state speedup.
To capture the representative decoder layer, first harvest the model:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/qwen3-level4/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.levels.level3.capture \
  --layer 14 --harvest results/qwen3-level4/harvest.json \
  --out results/qwen3-level4/layer14.pt
```

Artifacts contain weights, call arguments, outputs, incoming gradients and
reference gradients. They are local files, not distributed with the repo.

## Block evaluation

Config-derived inputs do not require a model capture:

```bash
evograd tier3-bench --scope block --model qwen3_0_6b \
  --block-source config --layer-index 14 --batch 2 --tokens 2048 \
  --dtype bfloat16 --device cuda --baseline none --structural-identity \
  --out results/evaluation/tier3/qwen3_0_6b/block/config.json
```

Evaluate a candidate against a captured layer:

```bash
evograd tier3-bench --scope block --model qwen3_0_6b \
  --block-source captured --artifact results/qwen3-level4/layer14.pt \
  --layer-index 14 --device cuda --baseline none \
  --candidate swiglu_mlp=candidate.py \
  --calibrate-with structural_identity,bound_pair_identity,trusted_torch_compile \
  --out results/evaluation/tier3/qwen3_0_6b/block/candidate.json
```

Candidate files must implement the corresponding declared operator. Use
`--patch-set NAME:SITE=PATH,SITE=PATH` for a combination, or `SITE=compile` for
a compiled site. Compiling all sites this way is not whole-block compilation.

| Site | Task | Calls per block |
| --- | --- | ---: |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | 1 |
| `attention` | `qwen3_attention` | 1 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | 1 |
| `residual_rmsnorm` | `fused_add_rms_norm` | 1 |

The block retains its initial norm and final residual add. The residual fusion
site is the post-attention boundary; cross-layer and final-model norm fusions
belong to model scope.

Block timing covers forward plus backward with supplied cotangents and resets
gradients/state before every repetition. It has no optimizer or language-model
loss. Model scope retains the training-step timing and model gates.

## Policies and results

The block CLI calibrates before judging providers unless a frozen file is
supplied with `--block-policy`. Policies bind to the case and environment.
Current schema is `evograd-t3-block-policy/2`; v1 policies require recalibration.
A failed calibration cannot authorize candidate timing.

See [correctness checks](L3_CORRECTNESS_CHECKS.md), the
[evaluation method and input sources](../src/evograd/evaluation/README.md),
[the 2026-09-15 block benchmark](experiments/benchmark_run_20260915.md), and
[the 2026-09-17 end-to-end report](experiments/benchmark_run_20260917.md).

## Report-first numerical evaluation (model scope)

A Tier-3 model-scope verdict answers two questions, and `--numerical-enforcement`
decides what the first one does to the second:

| question | field | mode `strict` (default) | mode `report-first` |
| --- | --- | --- | --- |
| did the comparisons meet their limits? | `numerical_ok`, `numerical_status` | recorded | recorded, identically |
| may the provider be trained and timed? | `ok`, `execution_ok` | no, if anything failed | yes, unless a structural or execution failure occurred |

`numerical_status` is one of `within_limits`, `mismatches_recorded` or
`unavailable`; a comparison that could not be made (a policy that does not bind,
a missing holdout verdict) is reported `unavailable` and never as a pass. Nothing
in report-first mode turns a numerical failure into a numerical pass.

The current implementation also allows execution after an unavailable comparison,
but sets `evaluation_complete: false`. Read that field and
`evaluation_incomplete_because` even when the status is `mismatches_recorded`:
a provider can have both a measured mismatch and a missing comparison.

These stop a provider in **both** modes, because none of them is a tolerance
question: a missing or extra gradient, a shape or dtype mismatch, parameter
misalignment, patch coverage or invocation counts, purity and input mutation,
candidate permissions, a non-finite output, gradient or loss, a kernel that
raises, and any runtime failure. Numerical findings do not short-circuit later
stages; structural and execution failures do. `--no-verify` remains a different
thing entirely (it skips the checks, and its reports say so).

```bash
evograd tier3-bench --model qwen3_0_6b --real-text --whole-model-compile \
    --numerical-enforcement report-first \
    --patch-set B:attention=candidate.py \
    --protocol4-calibration policy.json --protocol4-verdict holdout_B.json \
    --out results/report_first_B.json
```

Each failed tensor comparison is kept in full under
`providers.<name>.model_correctness`: the provider and its content hash, the
scope, data seed, layer and invocation, the tensor and whether it is a forward
output or a backward gradient, the reference implementation and the exact rule
and thresholds, shape and dtype, maximum absolute error, relative L2, violation
count and fraction, and a few representative violating coordinates with the
actual value, the reference value, the allowance there and the error-to-allowance
ratio. Whole-model metrics carry their own localization: the worst KL position,
and the parameters that dominate the gradient vector's squared error. The
scale-normalized `e_rms`/`e_max` of the budget experiment travel beside each
record as reported diagnostics, with no threshold attached.
