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
  --out results/qwen3-level3/canonical.json
```

This smoke command does not run an optimizer or measure steady-state speedup.
To capture the representative decoder layer, first harvest the model:

```bash
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest \
  --out results/qwen3-level3/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.layer.capture \
  --layer 14 --harvest results/qwen3-level3/harvest.json \
  --out results/qwen3-level3/layer14.pt
```

Artifacts contain weights, call arguments, outputs, incoming gradients and
reference gradients. They are local files, not distributed with the repo.

## Whole-model evaluation (Level 3)

The Level-3 task is the whole training step, measured by Tier 3:

```bash
evograd tier3-bench --model qwen3_0_6b --structural-identity --whole-model-compile \
  --patch-set B:swiglu_mlp=candidate.py \
  --out results/evaluation/tier3/qwen3_0_6b/model/candidate.json
```

Candidate files must implement the corresponding declared operator. Use
`--patch-set NAME:SITE=PATH,SITE=PATH` for a combination, `--compile-site SITE`
for one compiled site, and `--whole-model-compile` for the compiler baseline.

| Site | Task | Calls per step |
| --- | --- | ---: |
| `qkv_norm_rope` | `qwen3_qkv_norm_rope` | 28 |
| `attention` | `qwen3_attention` | 28 |
| `swiglu_mlp` | `qwen3_swiglu_mlp` | 28 |
| `residual_rmsnorm` | `fused_add_rms_norm` | 56 (both adds per layer; the cross-layer and final-norm fusions included) |

The captured layer above is the source of the Level-2
sites' real invocations and of the layer replay
(`python -m evograd.evaluation.workloads.qwen3_0_6b.layer.replay`).

See the [evaluation method and input sources](../src/evograd/evaluation/README.md)
and [the 2026-09-17 end-to-end report](experiments/benchmark_run_20260917.md).

## Report-first numerical evaluation

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
