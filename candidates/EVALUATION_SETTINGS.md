# Verification, timing, and combination settings

This document describes the current local implementation of formal
`evograd-evaluation-v3`, called by `candidates/evaluate_model.py`. It is a
reference for agreeing on an experiment, not evidence that GPU evaluation has
completed or that all provisional settings have been validated.

One evaluation contains one agent's frozen submissions for one model. Gates
measure numerical correctness; tiers measure runtime. Gate 3 training and
Tier 3 timing are separate experiments with different data and optimizer
execution settings.

## 0. Evaluation values by stage

### Shared verification

| Variable | Value |
| --- | --- |
| Protocol | `evograd-evaluation-v3` |
| Reference | Native eager model |
| Compile backend | `inductor` |
| cuBLAS workspace | `:4096:8` |
| Deterministic algorithms | `True` |
| cuDNN deterministic | `True` |
| cuDNN benchmark | `False` |
| Matmul TF32 | `False` |
| Float32 matmul precision | `highest` |
| CPU threads | `8` |
| Import timeout | `600 s` |

### Optional calibration reuse

| Variable | Value |
| --- | --- |
| `--share-gate3-native` | `False` |
| `--share-gate2-calibration-native` | `False` |
| `--share-gate2-verification-reference` | `False` |
| Reuse version | `native-control-reuse/1` |
| Shared-native Gate 3 policy, when enabled | `evograd-verification-gate3-global-loss-policy/2` |
| Raw Gate 2 execution cache lifetime | One job |
| Gate 2 calibration reference/repeat versus verification reference | Separate execution IDs |

### Gate 1

| Variable | Value |
| --- | --- |
| Calibration order seeds | `101, 202` |
| Acceptance order seeds | `303, 404` |
| FP16 `(atol, rtol)` | `(1e-5, 1e-3)` |
| BF16 `(atol, rtol)` | `(1e-5, 1.6e-2)` |
| FP32 `(atol, rtol)` | `(1e-5, 1.3e-6)` |
| FP64 `(atol, rtol)` | `(1e-7, 1e-7)` |
| Comparison precision | `float64` |
| Compile envelope factor | `4` |
| Rounding floor | `4 × ULP(max(abs(reference)))` |
| Allowed elementwise error | `max(atol + rtol × abs(reference), 4 × calibration_compile_max_abs_error, rounding_floor)` |
| Declaration tolerance scaling | `False` |

### Operator Gate 2

| Variable | Value |
| --- | --- |
| Calibration order seeds | `101, 202` |
| Acceptance order seeds | `303, 404` |
| Sample protocol | `disjoint_holdout` |
| Safety factor | `2` |
| Comparison precision | `float64` |
| Denominator guard | `1e-30` |
| E2 | `norm(candidate-reference, 2) / max(norm(reference, 2), 1e-30)` |
| Emax | `max(abs(candidate-reference)) / max(RMS(reference), 1e-30)` |
| Output E2 floor | `1e-5` |
| Output Emax floor | `1e-3` |
| Gradient E2 floor | `1e-4` |
| Gradient Emax floor | `1e-2` |
| Per-tensor metric threshold | `2 × max(native_repeat_error, compile_error, floor)` |
| Control pooling | Maximum over calibration calls |
| Compile repeatability | Bitwise equal |
| Pass condition | `E2 <= T_E2 AND Emax <= T_Emax` |

### Model Gate 2

| Variable | Value |
| --- | --- |
| Calibration/judgement order seeds | `101, 202, 303, 404` |
| Batch index per order | `0` |
| Sample protocol | `matched_inputs` |
| Optimizer updates | `0` |
| Safety factor | `4`; ConvNeXt: `3` |
| Comparison precision | `float64` |
| Denominator guard | `1e-30` |
| Output `(E2, Emax)` floors | `(1e-5, 1e-3)` |
| Gradient `(E2, Emax)` floors | `(1e-4, 1e-2)` |
| Per-tensor metric threshold | `gamma × max(native_repeat_error, compile_error, floor)` |
| Control pooling | Maximum over four samples |
| Local boundary replay | `False` |
| Pass condition | `E2 <= T_E2 AND Emax <= T_Emax` |
| OLMoE model gates | Disabled |

### Gate 3 — single-site and combined

| Variable | Value |
| --- | --- |
| Training order seed | `707` |
| RNG seed | `0` |
| Optimizer | `AdamW` |
| Betas | `(0.9, 0.999)` |
| Optimizer epsilon | `1e-8` |
| Weight decay | `0.01` |
| `foreach` | `False` |
| Controls | Native A, native B, site compile |
| Loss recording | Before each update |
| LM objective | FP32 next-token CE / valid-token count |
| Ignored LM label | `-100` |
| RMS floor | `1e-4` |
| Maximum-error floor | `1e-4` |
| Safety factor | `2`; ConvNeXt: `3` |
| Loss difference | `d[t] = candidate_loss[t] - native_A_loss[t]` |
| RMS metric | `R = sqrt(mean(d[t]**2))` |
| Maximum metric | `M = max(abs(d[t]))` |
| RMS threshold | `gamma × max(R_native_repeat, R_compile, 1e-4)` |
| Maximum threshold | `gamma × max(M_native_repeat, M_compile, 1e-4)` |
| Pass condition | `R <= T_R AND M <= T_M` |
| Excluded updates | `0` |
| Validation | `False` |
| Checkpoint interval | `100 updates`; deadline checkpoints |
| Warmup lr | `target_lr × min(1, update / warmup_updates)` |

| Model | Updates | Target lr | Warmup updates | Clip norm | Accumulation | G3 gamma |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3-0.6B | 100 | 1e-4 | 0 | None | 1 | 2 |
| Qwen3.5-4B | 100 | 3e-5 | 100 | 1.0 | 1 | 2 |
| Llama-3.2-1B | 100 | 3e-5 | 100 | 1.0 | 1 | 2 |
| Qwen2-VL-2B | 10 | 3e-5 | 100 | 1.0 | 8 | 2 |
| ConvNeXt Tiny | 50 | 1e-4 | 0 | None | 1 | 3 |
| DiT-S/2 | 100 | 1e-4 | 0 | None | 1 | 2 |
| DLRM | 50 | 1e-4 | 0 | None | 1 | 2 |
| OLMoE | Disabled | — | — | — | — | — |

### Tier 2

| Variable | Value |
| --- | --- |
| Data-order seed | `303` |
| Calls | All recorded case calls in the step |
| Measured work | Deployment forward + autograd backward |
| Optimizer updates | `0` |
| Gradient accumulation | `False` |
| Baselines | Eager, operator compile, available declared libraries |
| Warmup iterations | `10` |
| Requested repetitions | `50` |
| Paired blocks | `3` |
| Base seed | `0` |
| Call-k ordering seed | `k` |
| Estimation iterations | `3` |
| Repetition budget | `100 ms` |
| Minimum repetitions | `5` |
| Actual repetitions | `max(5, min(50, int(100 / estimated_ms)))` |
| Provider ordering | Single agent: paired blocks; all agents: shared randomized provider blocks |
| Timer | CUDA events |
| Cache flush | Before each timed region |
| Flush between forward/backward, full metric | `False` |
| Input copies | One per provider |
| Baseline timing cache | Single agent: disabled; all agents: one shared case measurement |
| Call latency | Median across pooled timed samples |
| Step latency | Sum of call medians |
| Speedup | `baseline_ms / candidate_ms` |
| Bootstrap draws | `2000` |
| Bootstrap resampling | Paired blocks, with replacement |
| Bootstrap seeds `(backward, full, forward)` | `(0, 1, 2)` |
| Confidence interval | `95%` |
| Percentiles | `2.5%, 97.5%` |
| Deterministic algorithms | `False` |

### Tier 3 — single-site and combined

| Variable | Value |
| --- | --- |
| Timed batch seed | `0` |
| Timed batch | `workload.batch_for(seed=0)` |
| Measured work | Forward/loss + backward + AdamW + gradient reset |
| Optimizer | `AdamW` |
| Learning rate | `1e-4` |
| Betas | `(0.9, 0.999)` |
| Optimizer epsilon | `1e-8` |
| Weight decay | `0.01` |
| `amsgrad` | `False` |
| `foreach`, fused | PyTorch defaults (`None`) |
| LR warmup | `0` |
| Gradient clipping | None |
| Accumulation | `1` |
| Gradient reset | `zero_grad(set_to_none=True)` |
| Preliminary loss steps | `5` |
| Preliminary batch seeds | `1, 2, 3, 4, 5` |
| Settling steps | `1` |
| Memory-probe steps | `1` |
| Timing warmup steps | `3` |
| Steps per block | `10` |
| Blocks per provider | `10` |
| Timed updates per provider | `100` |
| Post-timing loss-check steps | `1` |
| Model/optimizer reset between blocks | `False` |
| Provider ordering seed | `0` |
| Provider isolation | One spawned process per provider |
| Timer | Synchronized wall clock |
| Cache flushing | `False` |
| Standalone numerical preflight | `verify=False` |
| Step latency | Median of block mean step times |
| Speedup | `eager_step_ms / candidate_step_ms` |
| Bootstrap draws | `2000` |
| Bootstrap resampling | Independent provider blocks, with replacement |
| Bootstrap seed | `0 + sorted_provider_index` |
| Confidence interval | `95%` |
| Percentiles | `2.5%, 97.5%` |
| Memory metric | Peak CUDA allocated bytes during memory probe |
| Memory ratio | `candidate_peak_bytes / eager_peak_bytes` |
| Deterministic algorithms | `False` |

### Combination selection and acceptance

| Variable | Value |
| --- | --- |
| Agents per combination | `1` |
| Models per combination | `1` |
| Eligible level | `L2` |
| Individual acceptance | `True` |
| Single-site T3 execution | `ok=True` |
| Selection reference | Eager |
| Lower confidence bound | Finite and strictly `> 1.0` |
| Selected candidates | All qualifying L2 cases |
| Top-k limit | None |
| Subset search | None |
| T1/T2 speedup cutoff | None |
| Memory-ratio cutoff | None |
| Candidates per replacement site | `1` |
| Independently installed L1 candidates | `0` |
| Combined verification | Model Gate 2 + Gate 3 |
| Combined acceptance | `gate2_model == passed AND gate3 == passed` |
| Combined timing baselines | Eager, whole-model compile where available |
| Accepted OLMoE L3 | Unavailable |

### Run management

| Variable | Value |
| --- | --- |
| Default agent | `openevolve` |
| All-agent mode | `--agent all`; `openevolve, gepa, codex`; case-major, one model per job |
| All-agent checkpoint reuse | One pristine CPU snapshot per checkpoint/model spec/job |
| All-agent payload reads | One read per unique storage per case |
| All-agent T2 baseline scope | Once per case, shared across agents |
| All-agent T3 baseline scope | Once per model, shared across single-site and combined candidates |
| Default deadline | `45 min` |
| Default T3 blocks | `10` |
| Search root | `/work/hdd/bcjw/wzhan/ADBench` |
| Submission root | `/work/hdd/bcjw/wzhan/ADBench-redeploy` |
| Corpus path | `cases.json[model][corpus]` |
| Output root | `~/adbench-evaluation-trials` |
| Result path | `OUT_ROOT/MODEL/evaluation/AGENT/evaluation.json` |
| Shared state | `OUT_ROOT/MODEL/state` |
| Resume exit code | `3` |

## 1. Data and references

| Purpose | Input selection |
| --- | --- |
| Operator Gate 2 calibration | Recorded calibration payloads, orders 101 and 202 |
| Formal Gate 1 | Recorded acceptance payloads, orders 303 and 404 |
| Operator Gate 2 acceptance | Recorded acceptance payloads, orders 303 and 404 |
| Model Gate 2 calibration and judgement | First TRAIN batch of each order 101, 202, 303, 404; judgement uses fresh executions of those same sample contents |
| Gate 3 training | Consecutive batches of TRAIN order 707; RNG seed 0 |
| Formal Tier 2 | Every recorded call of the case in the training step of acceptance order 303, sorted by call index |
| Tier 3 timed batch | The model workload's `batch_for(seed=0)`, repeated during timing |

The reference is the model's native computation. Model-specific declarations,
workload configuration, checkpoint, data processing, shapes, dtypes, layouts,
objective, and reference source identity must agree between evaluators.
Operator payload replay preserves recorded arguments, parameters, incoming
output gradients, strides, storage offsets, and aliasing. Missing or incomplete
Tier 2 call coverage is an error; generated inputs are not substituted.

Search-time Gate 1 feedback on calibration payloads does not itself constitute
formal Gate 1 acceptance on the acceptance split. Tier 1 search timing is not
rerun by the formal protocol and is not a final benchmark metric.

## 2. Verification execution settings

| Setting | Value |
| --- | --- |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8`, set before process startup |
| PyTorch deterministic algorithms | Enabled |
| cuDNN deterministic | Enabled |
| cuDNN benchmark | Disabled |
| CUDA matmul TF32 | Disabled |
| Float32 matmul precision | `highest` |
| PyTorch CPU threads | 8 |
| Trusted compile backend | `inductor` |

Record actual attention backend settings and the runtime environment; a
deterministic run can select different attention kernels from a timing run.

Static artifact checks enforce the submission's recorded primitive grants and
deployment contract. A static violation prevents verification and timing.
Unimportable candidates are recorded as failures; their import precheck has a
600-second timeout. No numerical tolerance overrides a structural failure,
missing required comparison, missing gradient, invalid policy, nonfinite
result, or prohibited mutation.

## 3. Gate 1: elementwise agreement

For each required output and gradient, compare in float64:

```text
abs(candidate - reference)
    <= max(atol + rtol * abs(reference), envelope)
```

The reference result's dtype selects the tolerances, independently of the
workload's nominal activation dtype:

| Reference result dtype | atol | rtol |
| --- | ---: | ---: |
| float16 | 1e-5 | 1e-3 |
| bfloat16 | 1e-5 | 1.6e-2 |
| float32 | 1e-5 | 1.3e-6 |
| float64 | 1e-7 | 1e-7 |

For model-reference tasks, declaration tolerance overrides, result multipliers,
and shape scaling do not change this table.

The compile envelope is per result tensor:

```text
compile_deviation = maximum elementwise abs(compile - native)
                    over operator calibration payloads

effective envelope = max(4 * compile_deviation,
                         4 * ULP(max(abs(current reference result))))
```

The rounding floor is zero for an empty or all-zero reference result. For a
finite nonzero largest magnitude `a`, its implementation is
`4 * finfo(dtype).eps * 2**(frexp(a).exponent - 1)`.

The compile envelope factor is **4**, separate from Gate 2's factor. If an
operator calibration policy is unavailable, Gate 1 can still use the dtype
rule and rounding floor; missing Gate 2 calibration still prevents Gate 2
acceptance. Gate 1 also checks structure, shapes, dtypes, finiteness, required
gradients, and input side effects. Backward uses the candidate's own forward
state. Model Gate 1 policy identifier: `evograd-gate1-model-dtype-tolerances/5`.

## 4. Gate 2: tensor-level error budgets

Both metrics are computed in float64 for each required tensor:

```text
E2   = norm(candidate - reference, 2) / max(norm(reference, 2), 1e-30)
Emax = max(abs(candidate - reference)) / max(RMS(reference), 1e-30)
```

`1e-30` is a denominator guard, not an acceptance floor. Both metrics must be
within their thresholds. In particular, Emax is normalized by reference RMS,
not by the largest reference element or elementwise reference magnitude.

The fixed floors are:

| Result family | E2 floor | Emax floor |
| --- | ---: | ---: |
| Outputs, including scalar loss where applicable | 1e-5 | 1e-3 |
| Gradients | 1e-4 | 1e-2 |

For each result tensor `z` and metric `m`:

```text
threshold[z,m] = gamma * max(
    worst native-repeat error[z,m],
    worst trusted-compile error[z,m],
    floor[result family,m]
)
```

Candidate errors never calibrate their own thresholds. Therefore, the numeric
thresholds are calibrated values, not a universal fixed table; save the
generated policies as part of the experiment.

### Operator scope

- Safety factor `gamma=2` for every model.
- Calibrate on orders 101/202; judge on disjoint orders 303/404.
- Pool control maxima per result over calibration invocations.
- Exercise the actual autograd deployment and compare required outputs and
  activation/parameter gradients.
- Policy sample protocol: `disjoint_holdout`.

### Model scope

- One L2 candidate is installed at its model site, or the selected L3
  candidates are installed together.
- One full forward/backward per sample; no optimizer update.
- Compare whole-model outputs/loss and required parameter gradients, plus
  replacement coverage and side effects.
- Pool each tensor's control maxima over the four samples from orders
  101/202/303/404; judge fresh executions of those same contents.
- This is a matched-input comparison, not an independent holdout.
- Per-call local boundary replay is disabled at model scope. Local operator
  comparisons remain in operator Gate 2.
- Safety factor `gamma=4`, except ConvNeXt Tiny uses `gamma=3`.
- Policy sample protocol: `matched_inputs`.

A trusted compile control must qualify, including actual compiler invocation
at replacement sites and bitwise reproducibility of its repeated execution.
Missing/disqualified controls or mismatched policy bindings leave verification
incomplete. Policies bind scope, reference, environment, numerics, patch set,
sample identities and content, and formula. Policy schema:
`evograd-verification-gate2-policy/4`.

## 5. Gate 3: continuous-training agreement

Run single-site Gate 3 only after Gate 1, operator Gate 2, and model Gate 2
pass. Run it again for the selected combination. OLMoE has no model-scope
Gate 2 or Gate 3.

### Optimizer and data

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Betas | (0.9, 0.999) |
| Epsilon | 1e-8 |
| Weight decay | 0.01 |
| `foreach` | False |
| RNG seed | 0 |
| Training data order | 707 |
| Data progression | Consecutive batches; workload's stream/cycling behavior |
| Learning-rate schedule | Constant unless model-specific linear warmup is configured |
| Periodic checkpoint interval | 100 updates, plus deadline checkpoints |
| Held-out validation | Disabled for these training-only plans |

Current model overrides:

| Model | AdamW updates | Target lr | Warmup updates | Global gradient norm clip | Batches/update | Model G2 gamma | G3 gamma |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| qwen3_0_6b | 100 | 1e-4 | 0 | None | 1 | 4 | 2 |
| qwen3_5_4b | 100 | 3e-5 | 100 | 1.0 | 1 | 4 | 2 |
| llama_3_2_1b | 100 | 3e-5 | 100 | 1.0 | 1 | 4 | 2 |
| qwen2_vl_2b | 10 | 3e-5 | 100 | 1.0 | 8 | 4 | 2 |
| convnextv2_tiny | 50 | 1e-4 | 0 | None | 1 | 3 | 3 |
| dit_s_2 | 100 | 1e-4 | 0 | None | 1 | 4 | 2 |
| torchrec_dlrm | 50 | 1e-4 | 0 | None | 1 | 4 | 2 |
| olmoe_1b_7b | Not applicable | — | — | — | — | — | — |

Linear warmup at update `t` (1-based) uses
`target_lr * min(1, t / warmup_updates)`. Consequently, Qwen2-VL's 10-update
run remains within warmup and reaches lr 3e-6, not 3e-5. With accumulation 8,
each microbatch objective is divided by 8 before backward and the recorded
per-update objective is the arithmetic mean of the eight microbatch objectives.

For language-model adapters, Gate 3 explicitly computes FP32 next-token cross
entropy, shifts logits/labels once, ignores labels -100, and divides the loss
sum by valid-token count. Other workload adapters use the workload's primary
objective. Record the objective actually differentiated before each update.

### Controls and acceptance

Each patch set has native A, native B, and site-compile control runs with
identical initial weights, optimizer, data sequence, seed and training horizon.
Site compile is a correctness control; it is different from T3's whole-model
compile performance baseline.

For losses recorded at the same update:

```text
d[t] = candidate_objective[t] - native_A_objective[t]
R = sqrt(sum(d[t]**2) / N)
M = max(abs(d[t]))

T_R = gamma * max(R(native_B, native_A), R(compile, native_A), 1e-4)
T_M = gamma * max(M(native_B, native_A), M(compile, native_A), 1e-4)

pass = R <= T_R AND M <= T_M
```

Equality passes. Use the complete horizon: no excluded updates, permitted
violation count, selected windows, or averaging away a failed trajectory.
Initial state, optimizer, plan, data, seed, environment and patch set must
match the policy. Candidate trajectories never increase the budget. Policy
identifier: `evograd-verification-gate3-global-loss-policy/1`.

## 6. Acceptance and executable status

- L1 acceptance requires Gate 1 and operator Gate 2 to pass.
- L2 acceptance additionally requires model Gate 2 and single-site Gate 3 to
  pass where model-scope gates are supported.
- OLMoE L2 acceptance currently requires only Gate 1 and operator Gate 2.
- Failed, skipped or incomplete required gates do not count as passed.
- Executability is separate: the deployed forward/backward must run, return
  the required eager shapes/dtypes, and respect input mutation rules. The
  formal T2 precheck uses the first selected recorded call; later failures are
  still reported by the timer.
- Executable rejected candidates can receive diagnostic T2/T3 timings, but
  do not enter accepted speedup aggregates or positive-candidate selection.

## 7. Formal Tier 2 performance settings

| Setting | Value |
| --- | --- |
| Work | Deployed forward and `torch.autograd.grad` backward |
| Optimizer / `.grad` accumulation | Neither |
| Inputs | All recorded calls for the case in acceptance order 303 |
| Warmup | 10 forward/backward iterations per provider and recorded call |
| Requested timed repetitions | 50 per measured region per block |
| Blocks | 3 paired blocks |
| Base seed | 0; recorded call k uses scheduling seed k |
| Providers | Candidate vs eager, operator `torch.compile`, each declared available library baseline |
| Provider ordering | Randomized within each paired block |
| Timer | CUDA events; synchronization outside individual timed regions |
| L2 cache | Clear before each timed region; no clear between forward/backward in the full metric |
| Input copies | Separate copies per provider, prepared outside timing |
| Baseline timing reuse | Disabled inside each fair measurement; saved whole-stage results can resume |

Forward-only, backward-only, and full forward+backward regions are measured
separately. The backward-only region prepares forward state outside its timed
region. The primary metric is the directly measured full region, not the sum
of independently measured forward and backward latencies.

The timer estimates each forward/full region using 3 iterations, then chooses
the repetition count independently:

```text
actual_reps = max(5, min(50, int(100 milliseconds / estimated_region_ms)))
```

If the estimate is nonpositive, use the requested 50. This is the current
adaptive rule; 50 repetitions are not guaranteed for expensive regions.

Per-call reported latency is the median of that provider's pooled timed
samples across blocks. Aggregate over the calls in the step:

```text
candidate_ms = sum(per-call candidate medians)
baseline_ms  = sum(per-call baseline medians)
speedup      = baseline_ms / candidate_ms
```

### Confidence intervals and stored fields

- 2000 bootstrap draws; 2.5th and 97.5th percentiles.
- Resample paired block indices with replacement separately for each call,
  preserving candidate/baseline pairing within each call.
- Within a draw, take the median of selected block medians per provider and
  call, then form the ratio of sums across calls.
- Bootstrap seeds: backward 0, full 1, forward 2.
- In single-agent mode each baseline comparison remeasures its candidate, so candidate latencies
  may differ between baseline rows.
- Formal rows retain full candidate/baseline latency, speedup and interval,
  plus forward/backward speedups. The formal wrapper does not retain every
  raw fair-timer field: per-call timings, actual adaptive counts, raw T2
  blocks, separate forward/backward absolute latencies and their intervals
  are not all preserved in the formal result.
- Missing optional-library results do not discard eager/compile results.
- Corpus and selected payload IDs are recorded in `tier2_inputs`.

### All-agent T2 scheduling

`--agent all` uses `case-major-shared-baselines/1`: for each recorded call, all
agent candidates and available baselines participate in the same three
randomized blocks. Each provider is measured once per block, and every
candidate/baseline comparison uses matching block indices in its bootstrap.
Candidates are not remeasured for each baseline. A private input copy is
prepared outside timing for each provider block, keeping simultaneous GPU
copies bounded. Shared reports retain raw blocks under `MODEL/shared/tier2/`;
all agents reference those same baseline measurements. Per-case payload
storages are read once into a shared CPU cache and privately copied for each
execution. Scientific timing defaults and recorded-call coverage are unchanged.

## 8. Formal Tier 3 performance settings

Each executable L2 case is timed alone in its model, with its replacement
installed at every applicable invocation of the designated site. All other
sites remain native. Single-site does not mean one decoder layer. ConvNeXt
stage cases target their own stage/site, not every stage configuration.

| Setting | Value |
| --- | --- |
| Work | Model forward/loss, backward, AdamW update, `zero_grad(set_to_none=True)` |
| Learning rate | 1e-4, independent of Gate 3's model-specific lr |
| AdamW betas / eps / weight decay / amsgrad | (0.9, 0.999) / 1e-8 / 0.01 / False |
| AdamW `foreach` / fused | Not explicitly set by this runner; PyTorch runtime defaults apply |
| Gate 3 warmup/clipping/accumulation overrides | Not applied by the timing runner |
| Seed | 0 |
| Preliminary loss trajectory | 5 updating steps on `batch_for(seed=1)` through `batch_for(seed=5)` |
| Untimed preparation on timed batch | One settling step, then one memory-probe step |
| Warmup on timed batch | 3 updating steps |
| Timed steps per block | 10 |
| Timed blocks per provider | 10 |
| Timed batch | Same `batch_for(seed=0)` throughout |
| State progression | Model/optimizer continue updating; no reset between timed steps or blocks |
| Timer | Synchronized wall-clock time per block |
| L2 cache flushing | None inside model-step measurement |
| Provider isolation | One newly spawned process per provider |
| Provider order | Seeded shuffle, seed 0, recorded in the report |

Timing disables deterministic algorithms; the parent timing configuration
also disables cuDNN deterministic mode. Loading, compilation, gate checks,
loss checks and the preparation steps are outside timed blocks. In the formal
path, T3 providers run with `verify=False`: correctness acceptance comes from
the formal gates, rather than repeating standalone T3 numerical preflight.
Loss finiteness checks around the timed blocks remain part of execution.

For each provider:

```text
block_step_ms[b] = synchronized_wall_time_of_10_steps[b] / 10
step_ms = median(block_step_ms)
speedup_vs_eager = eager_step_ms / candidate_step_ms
```

The 100 timed updates per provider are not the Gate 3 horizon. They are
preceded by loss/preparation/warmup updates and followed by an untimed loss
check. T3 uses the workload's training loss entry point, which is separate
from Gate 3's explicit language-model CE adapter.

### Confidence intervals and memory

- 2000 bootstrap draws; 2.5th/97.5th percentiles.
- Resample reference blocks and candidate blocks independently with
  replacement; take their median ratio for each draw.
- Bootstrap seed is `0 + provider_index` when enumerating sorted provider
  names (including the eager entry in the enumeration).
- T3 blocks are not paired across providers; do not use T2's paired bootstrap.
- Peak memory is CUDA peak allocated memory over the untimed memory-probe
  step, after optimizer state has been allocated. It is not an assertion
  about the maximum over every later timed step or about reserved memory.
- Memory ratio is candidate peak allocated bytes / eager peak allocated bytes.
- Retain per-block timings, confidence intervals, throughput, peak memory,
  provider order, and CPU submission-and-blocking fraction. The last metric
  is not a pure CPU dispatch-overhead measurement.

### All-agent T3 baseline sharing

After every case's verification and T2 finish for all agents, the grouped
scheduler measures all individual L2 providers in one model-wide cohort. Eager
and available whole-model compile are each measured once and their raw block
samples are reused in both individual and combined reports. Every candidate
provider still starts with fresh weights and optimizer state in its own
process. Candidate/bootstrap confidence intervals use the shared reference
samples and independent candidate samples. A second cohort measures each
agent's separately selected combination without remeasuring eager/compile.
Shared provider caches and reports are stored under `MODEL/shared/tier3/`.
Changing from the earlier agent-major scheduler requires a new output root.

## 9. Automatic positive-candidate and combination rule

For each Level-2 case:

```text
positive = accepted
           AND single_site_T3.ok
           AND finite(single_site_T3.ci95.low)
           AND single_site_T3.ci95.low > 1
```

All eligible cases are selected; no manual top-k or subset search is done.
An interval whose lower endpoint equals 1 is insufficient. Neither T1 nor T2
speedup selects the combination, and the rule does not require beating
operator compile or whole-model compile individually. There is no memory
ratio cutoff in positive selection.

The combination:

1. Uses candidates from one agent and one model only.
2. Installs Level-2 artifacts at their distinct model sites; L1 artifacts are
   not independently added by this path. Two candidates for the same site
   cause an error rather than silent replacement.
3. Reruns model Gate 2 and Gate 3 on the entire patch set, with matching native
   A/B and site-compile controls.
4. Is accepted only if both combined gates pass.
5. Is timed against eager and, where supplied by the workload, whole-model
   `torch.compile`, with the same T3 settings above.

The implementation can time a nonempty combination even when combined gates
reject it; such measurements are diagnostic, not accepted L3 results. If no
candidate qualifies, the combination is empty. One qualifying candidate makes
a one-candidate combination. Passing all individual candidates does not imply
their combination passes.

OLMoE is a special limitation: individual L2 acceptance omits model gates;
`verify_combination` explicitly marks model-scope acceptance unavailable.
Nonempty diagnostic combination timing can still occur, but it must not be
reported as an accepted L3 result.

## 10. Reporting, resume and agreement between evaluators

Use the same source tree, including local modifications, dependencies,
hardware class, workload/corpus identity, submissions and all settings above.
Keep failure/incomplete/skipped statuses beside timings. Formal L1/L2 speedup
aggregates include accepted configurations with valid baseline timings;
rejected diagnostic timings do not contribute. Report full observed-case
coverage separately from pass rate over the trial's selected cases. There is
no single combined score over T2 and T3.

`--deadline-min` controls suspension, not the scientific training horizon,
number of T3 blocks, or Slurm allocation. Default in the wrapper is 45 minutes.
Exit 3 means rerun the same command. Deadline checks occur at stage boundaries
and Gate 3 checkpoints; a long stage can exceed the deadline. Compatible
saved stages and controls are reused. Changed scientific settings require
checking bindings and using a new output root for a distinct experiment;
do not assume every higher-level cache detects arbitrary source edits.

With the wrapper, save:

```text
OUT_ROOT/MODEL/
  state/                              reusable control runs and calibrated policies
  evaluation/AGENT/
    selection.json                    selected cases, submission hashes, search failures
    evaluation.json                   settings, verdicts, timings, combination
    cases/                            detailed Gate 1/operator Gate 2 reports
    gate2_model/                      individual model Gate 2 reports
    gate3_single_site/                individual training runs and Gate 3 policies/verdicts
    tier2/                            formal T2 result caches
    tier3/                            individual/combined T3 reports and block timings
    combination/                     combined model Gate 2 and Gate 3 evidence
```

The wrapper only exposes model, agent, corpus/root overrides, deadline and T3
block count. Tolerances, factors, floors, T2 defaults, Gate 3 settings and
positive selection are defined in the evaluation source. Existing environment
variables can override the wrapper's default data/cache paths.

Search ledgers and task packs are read from `--shared-root` (ADBench).
Evaluation artifacts are read from `--submission-root` (ADBench-redeploy),
using `AGENT/MODEL/runs/AGENT/CASE/submission.py` and its adjacent
`submission.json`. Both original and repackaged hashes are checked; the
EVOLVE-BLOCK contents must match byte for byte. The wrapper does not fall back
to original artifacts if a repackaged artifact is absent. Use a fresh output
root after switching artifacts; retain earlier reports as historical evidence.

The older six-agent plan's provisional Gate 3 table is inconsistent with the
current implementation. Its 1000/200/125-update horizons are not what these
commands execute. In addition, code comments identify Qwen3.5's Gate 3 setting
as not yet measured with seeds/faults and Qwen2-VL's shortened horizon as
interim. Agree on these limitations before naming a run final. Changing
thresholds because a particular candidate fails defines a different experiment.

## 11. Implementation sources

- [Formal protocol, timing defaults, selection and summary](../src/evograd/evaluation/protocol.py)
- [Shared verification configuration and default horizon](../src/evograd/verification/integrations/base.py)
- [Gate orchestration, model factors, training controls](../src/evograd/verification/integrations/gates.py)
- [Gate 1 tolerances and rounding floor](../src/evograd/verification/gate1/tolerances.py)
- [Gate 2 metrics](../src/evograd/verification/common/tensors.py)
- [Gate 2 floors, calibration formula and sample protocols](../src/evograd/verification/gate2/calibration.py)
- [Gate 3 optimizer and objective adapters](../src/evograd/verification/gate3/adapters.py)
- [Gate 3 accumulation and training execution](../src/evograd/verification/gate3/runner.py)
- [Gate 3 global-loss acceptance formula](../src/evograd/verification/gate3/global_loss.py)
- [T2 fair timing, adaptive repetitions and bootstrap](../src/evograd/evaluation/tier1/fair.py)
- [T3 step measurement, optimizer and bootstrap](../src/evograd/evaluation/tier3/runner.py)
- [All-agent case-major scheduler](../src/evograd/evaluation/case_group.py)
- [Shared T2 provider blocks](../src/evograd/evaluation/group_timing.py)
- [Per-case payload storage cache](../src/evograd/evaluation/payload_cache.py)
- Model overrides: `src/evograd/verification/integrations/<model package>/__init__.py`.


Optional native-control reuse is documented in
[EVALUATE_ALL_AGENTS.md](EVALUATE_ALL_AGENTS.md#optional-calibration-reuse).
All three flags default to false; they change reuse scope, not numerical or
training settings. Use a new output root when changing flags.
