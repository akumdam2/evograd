# Evaluation

**Tier 2 checks an operator before timing it. Tier 3 measures the operator
inside a block or model, recording numerical differences alongside performance.**

For end-to-end research runs, use `--numerical-enforcement report-first`:
finite numerical mismatches do not stop training. Structural errors, missing
gradients, nonfinite values and runtime failures still do.

See the [Qwen/Llama results](../../../docs/experiments/benchmark_run_20260917.md)
for the latest measurements and their limits.

## What each tier measures

| Tier | Measured work |
| --- | --- |
| T1 | Call the forward/backward pair directly |
| T2 | Run one operator through PyTorch autograd, without an optimizer |
| T3 block | Run one decoder layer forward and backward with supplied output gradients |
| T3 model | Run the full model: forward/loss, backward, AdamW update and gradient reset |

Levels describe tasks: L1 primitive, L2 composite operator, L3 block, L4 model.
An L2 candidate can be evaluated at T2 and installed in either T3 scope.
Block and whole-model times must be reported separately.

## Tier 2: keep the declared correctness rules

Each provider receives identical generated inputs and copied weights. The
reference, or **oracle**, is the declared PyTorch forward differentiated with
`torch.autograd.grad`. The candidate runs through its deployment module;
PyTorch autograd produces the activation and parameter gradients.

Every declared output and requested gradient is compared elementwise:

```text
abs(candidate - oracle) <= atol(workload, result) + rtol(workload, result) * abs(oracle)
```

The declaration supplies the tolerances, result multipliers and optional
shape-dependent hook. They are not selected from the candidate's error.
Some tasks use [ReductionScaledAtol](../opdecl/tolerance.py):

```text
raw = sqrt(N / N_anchor) * sqrt(log(M) / log(M_anchor))
atol = base_atol * result_multiplier * (1 + gain * max(0, raw - 1))
```

`N` is the declared reduction extent and `M` the result element count; `rtol`
is unchanged. This is an empirical per-task rule, not a universal error bound.
For Qwen attention output, the element-count scaling raises `atol` from `0.01`
to about `0.019451` at the target shape, while `rtol` stays `0.01`.

Failed checks prevent T2 timing. `verify` and `tier2-bench` default explicitly
to `declared`, independently of Tier-3 settings. The offline alternative-budget experiments are not part of this workflow.

**`evograd verify` is a direct-pair grid check, not a target-shape Tier-2 run.**
Use `tier2-bench` to check and time the deployment module:

```bash
evograd tier2-bench --op qwen3_attention --candidate candidate.py \
  --suite qwen3_0_6b_observed --baseline none --dtype bfloat16 \
  --out results/qwen_attention_t2.json
```

Llama's suite is `llama_3_2_1b_observed`. Select the suite explicitly when a task
also has a generic grid. Eager uses the task's runtime implementation, compile
compiles that provider, and both are timed baselines—not extra acceptance gates.
The oracle can differ from the efficient runtime implementation, notably SDPA.

Current T2 timing uses `measure_module` and `triton.testing.do_bench`: forward
and forward+backward, 25 ms warmup and 500 ms measurement budgets by default,
with median and 20/80 percentiles. Providers normally run in separate processes.
Activation gradients are reset between training samples; parameter gradients
accumulate. Older report headers may contain obsolete fixed-repetition fields;
the current report identifies the actual `do_bench` driver.

## Tier 3: record differences and continue evaluation

Select `report-first` explicitly; the CLI default remains `strict`.

| Finding | Strict | Report-first |
| --- | --- | --- |
| Finite numerical mismatch | Stop | Record and continue |
| Missing comparison or unbound policy | Stop | Continue, marked incomplete |
| Contract, permission, coverage or purity failure | Stop | Stop |
| Nonfinite value or runtime failure | Stop | Stop |

All executable checks still run. This is not `--no-verify`, and a mismatch is
never converted to a numerical pass. Read these results separately:

- **Execution:** did checking, training and timing finish?
- **Numerical agreement:** which outputs or gradients exceeded their limits?
- **Completeness:** were all required comparisons available?
- **Training and performance:** loss behavior, step time, throughput and memory.

In JSON, `ok` means admission under the selected mode. `numerical_status`
describes agreement; `evaluation_complete` identifies missing comparisons.
A row can have both recorded mismatches and incomplete evaluation.

T3 checks the replacement sites, actual call counts, purity, local outputs and
gradients, then block/model results. Native execution is the reference.
Local comparisons use the same live inputs and incoming gradients. Frozen
policies supply any additional numerical limits and must match the case,
environment and patch set. Compile-relative Tier-2 gates and extra diagnostic
backends are not added to normal evaluation.

Keep every failed tensor/invocation in the detailed report: layer, tensor,
reference, rule, limits, max error, relative L2, violation count and representative
elements with actual/reference values. A maximum absolute error is not always
the element that violates its tolerance most strongly.

## Where shapes, inputs and gradients come from

Shapes follow the selected task's model configuration and harvest metadata.
**Model-derived dimensions do not mean the tensor values were captured.**
Read the case's provenance, not just its suite name.

| Scope | Inputs | Outputs and reference gradients |
| --- | --- | --- |
| T1/T2 operator | Seeded task recipe: activations, weights, layouts and an external gradient for each output | Oracle and candidate on identical values |
| Captured block | Saved layer weights, actual model arguments and incoming output gradients | Native layer replay; the capture also stores original outputs and gradients |
| Config-derived block | Seeded layer, inputs and external gradients | Native layer on that case |
| Whole model | Checkpoint or seeded weights; real-text or synthetic token IDs and labels | Model logits and LM loss; backward supplies the intermediate and parameter gradients |

Capture hooks save detached CPU copies from a full-model forward/backward.
The replay verifies artifact identity. A historical capture need not use the
same weights or data as a later whole-model training experiment.

### Qwen/Llama L2 boundaries and canonical shapes

The current matrix uses BF16, `B=2`, `T=2048` and `rows=B*T=4096`.

| Dimension | Qwen3-0.6B | Llama-3.2-1B |
| --- | ---: | ---: |
| Layers | 28 | 16 |
| Hidden width `H` | 1024 | 2048 |
| MLP width `I` | 3072 | 8192 |
| Query / KV heads `HQ / HK` | 16 / 8 | 32 / 8 |
| Head dimension `D` | 128 | 64 |
| `QO=HQ*D` / `KVO=HK*D` | 2048 / 1024 | 2048 / 512 |

| L2 site | Inputs and weights | Outputs |
| --- | --- | --- |
| QKV + RoPE | `x[B,T,H]`, `Wq[QO,H]`, `Wk,Wv[KVO,H]`, rotary tables `[1,T,D]`; Qwen adds Q/K norm weights `[D]` | `q[B,HQ,T,D]`, `k,v[B,HK,T,D]` |
| Attention + output projection | Q/K/V above, `Wo[H,QO]` | `[B,T,H]` |
| SwiGLU MLP | `x[B,T,H]`, gate/up weights `[I,H]`, down weight `[H,I]` | `[B,T,H]` |
| Residual + RMSNorm | `x,r[rows,H]`, weight `[H]`, epsilon | Normalized output and residual sum, both `[rows,H]` |

Q/K/V are head-major views of token-major tensors. Attention begins after QKV
projection and RoPE; those operations belong to the QKV site. Each site runs
once per block, but full-model residual fusion also covers cross-layer/final
norm boundaries: 56 calls for Qwen and 32 for Llama.

Sources: [model dimensions](../opdecl/models.py),
[Qwen sites](../benchmark/topdown/qwen3_0_6b/levels/level2/manifest.py),
[Llama sites](../benchmark/topdown/llama3_2_1b/levels/level2/manifest.py).
The generic Llama-3-8B operator grids are a different workload.

## Running Tier 3 and reading its timings

Captured block example, with identical external gradients for native and candidate:

```bash
evograd tier3-bench --scope block --model llama_3_2_1b \
  --block-source captured --artifact layer8.pt --layer-index 8 \
  --candidate attention=candidate.py --baseline none \
  --numerical-enforcement report-first \
  --calibrate-with structural_identity,bound_pair_identity,trusted_torch_compile \
  --out results/llama_attention_block.json
```

Record the calibration controls and policy hash. Different control sets can
produce different limits. Failed controls cannot establish a valid calibration.
Block timing includes forward and backward only; gradient/state reset is outside
samples. Site-wise compile combinations are not whole-block compilation.

Whole-model example:

```bash
evograd tier3-bench --scope model --model qwen3_0_6b --real-text \
  --whole-model-compile --baseline none --patch-set ours:attention=candidate.py \
  --numerical-enforcement report-first \
  --protocol4-calibration policy.json --protocol4-verdict holdout.json \
  --warmup 5 --blocks 5 --steps 20 --out results/qwen_model.json
```

Policy and holdout files must match the workload/candidate. Whole-model timing
includes forward/loss, backward, AdamW and gradient reset. It reports the median
per-step wall time across synchronized timing blocks. Loading, compilation and
checks are excluded; L2 is not flushed within a training step.

The command above is **not** a 200-step training study. The latest experiment
ran that separately through the shared [training loop](tier3/gate/training.py),
resetting state and evaluating trained weights with the native model at steps
0/50/100/200. Short loss agreement does not establish long-term equivalence.

Report speedup as baseline time / candidate time. Separate execution memory
from allocations retained by numerical checking. Unpatched eager/whole-model
compile currently bypass the standard model gate; separate numerical measurements
do not automatically make their policy-based evaluation complete.

Further details: [block checks](../../../docs/L3_CORRECTNESS_CHECKS.md),
[Qwen usage](../../../docs/QWEN3_LEVEL4.md),
[result storage](../../../docs/RESULTS_LAYOUT.md).
