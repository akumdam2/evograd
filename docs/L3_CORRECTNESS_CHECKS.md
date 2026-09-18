# Block correctness

Tier 3 block evaluation supports Qwen3-0.6B and Llama-3.2-1B. It compares a
native decoder layer with selected candidate operators installed. Every run
declares its case, input source, patch set and numerical policy.

The [evaluation guide](../src/evograd/evaluation/README.md) defines the current
Tier-2/Tier-3 workflow, shape and input provenance, and how to read report-first
results. The requirements below define the checks; whether a finite numerical
failure prevents timing is controlled separately by enforcement. `strict` is
the CLI default; end-to-end research runs explicitly select `report-first`.

## Checks before timing

| Check | Requirement |
| --- | --- |
| Case identity | Architecture, layer, dtype, shapes/layouts and source hashes match. Captured artifacts pass identity verification. |
| Native reference | Finite outputs/gradients, expected gradients present, inputs unmodified, bounded repeatability; captured cases also agree with stored reference results. |
| Policy | Valid schema/hash, matching case/environment, successful required controls and finite nonnegative thresholds. |
| Operator preflight | Declared pairs pass oracle checks on correctness grids and workload-supplied cases. Raw trusted callables without a pair are marked unverifiable at this stage. |
| Purity | Repeated calls do not depend on undeclared state or call history. |
| Local boundaries | Each live site agrees with its operator reference on outputs and emitted input/weight gradients, using the same local inputs and incoming gradients. |
| Patch coverage | Requested and actual sites agree, including supporting sites and invocation counts. |
| Block outputs | Return structure, metadata, shapes, dtypes, strides and tensor values agree with native under the policy. |
| Block gradients | Identical external cotangents produce acceptable input and parameter gradients, with matching presence, names/order and finite values. |

Qwen/Llama block local checks use declared operator tolerances. They do not
automatically use the model protocol's optional local envelope.

## Policy v2

`evograd-t3-block-policy/2` calibrates thresholds from native repeatability and
trusted controls before evaluating candidates. Candidate errors never set their
own thresholds.

Structural forward must be bitwise identical to native. Structural backward
must fit an independent envelope derived only from native repetitions and
fixed numerical floors. Each requested control must also pass its structural,
invocation-count and applicable local checks. A failed control aborts calibration.

Policy loading and candidate evaluation reject failed controls, missing required
controls and invalid thresholds. Version 1 policies require recalibration.

Regression tests include correct providers, missing gradients, mutation, invalid
policies and exact-forward/wrong-backward controls scaled by 1.5, 2, 10 and 100.
GPU validation covers the Qwen layer14 and Llama layer8 captured cases.

## Scope

These checks establish agreement for the declared cases. They do not evaluate
model prediction KL, optimizer updates or training convergence. T2 benchmarking
is a separate stage, not automatically repeated for every block evaluation.

The current parameter identity check compares names/order; it is not a general
shared-storage proof. Additional tied-parameter or MoE architectures require
validated adapters. Disabled checks and unavailable measurements remain explicit
in reports. See `evaluation/tier3/block.py` and `gate/block.py` for implementation.

## Numerical rules

Tier-1/Tier-2 use the operator's declared tolerances and shape-dependent hooks.
Tier-3 local checks use those tolerances and any applicable reference envelope;
block/model comparisons use their bound numerical policy. Enforcement decides
whether a numerical failure stops execution; it does not change these rules.
The abandoned dtype-default profile is not a supported evaluation mode.

## Enforcement: what a numerical mismatch does

`evograd.evaluation.tier3.gate.enforcement` separates the numerical verdict from
admission to training and timing. `--numerical-enforcement strict` (the default)
is the historical behaviour: any failure stops the provider. `report-first`
records finite numerical mismatches and unavailable comparisons and lets
execution continue, so an end-to-end result exists beside them.

At block scope the stage that failed decides the kind: `envelope` and `profile`
are numerical; `finiteness` is an execution failure; `structure`, `aliases`,
`gradient_presence`, `input_mutation` and `metadata_outputs` are structural;
`no_policy` and `invalid_policy` are unavailable. Any stage this table does not
know is treated as structural, so a new check is never waved through as a
rounding difference. The block report records the active mode.

### An unavailable comparison is not a pass

Report-first lets an unavailable comparison continue, which is right for
execution and wrong for the claim: a provider whose required comparison could
not be made has not been checked, whatever it managed to measure. Two fields
keep those apart, and a report must read the second before calling a row
  evaluated:

* **`ok`** — may this provider be timed, under the active mode. Nothing else.
* **`evaluation_complete`** — `execution_ok and not unavailable`. False whenever
  a required comparison did not run, with `evaluation_incomplete_because`
  naming the stages. It is never true merely because execution finished, and
  a completed comparison may still have failed numerically. If only unavailable
  numerical comparisons exist, `numerical_ok` is `None` with status
  `unavailable`. If recorded mismatches also exist, `numerical_ok` is `False`
  with status `mismatches_recorded`, and completeness remains false.

The same fields are reported for a provider the whole-model gate never judged.
`runner.model_correctness_check` returns early in three cases — the workload
declares no whole-model gate, `--no-verify` was given, or the provider patches
no site — and each used to return a bare `ok: True`. Whole-model
`torch.compile` is why that mattered: it patches no site, so the gate skips it,
and it does change the arithmetic. Those early returns now carry
`numerical_status: "unavailable"` and `evaluation_complete: False` under the
same names every other row uses, so one reader serves every row and "not
checked" cannot be read as "checked and fine". Timing such a provider is still
allowed; describing it as verified is not.

A workload whose frozen policy does not describe a given provider follows the
same rule rather than borrowing a neighbour's thresholds. Qwen3's protocol-4
policy is bound to a patch set. In the
[September 17 run](experiments/benchmark_run_20260917.md), only the Qwen
attention patch set had a bound whole-model policy and Llama had none.
Without one, the threshold-free checks still run — site preflight,
provider purity, every live invocation against its *declared* tolerances,
invocation counts and provenance, gradient presence and finiteness — the
whole-model aggregates are measured and recorded without a verdict, and the row
has unavailable comparisons and is incomplete.
