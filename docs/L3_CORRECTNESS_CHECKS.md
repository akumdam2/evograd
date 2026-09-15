# Block correctness

Tier 3 block evaluation supports Qwen3-0.6B and Llama-3.2-1B. It compares a
native decoder layer with selected candidate operators installed. Every run
declares its case, input source, patch set and numerical policy.

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
