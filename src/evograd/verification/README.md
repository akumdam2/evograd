# Canonical candidate verification

`evograd.verification` provides reusable execution and acceptance APIs for
three gates. It is separate from performance measurement and evolution.
Import gate modules explicitly:

```python
from evograd.verification import gate1, gate2, gate3
```

This change adds the verifier and its tests. It does not migrate existing model
declarations, switch the evolution evaluator, install an experiment driver, or
add the end-to-end workflow. Applications choose their adapters, inputs,
reference implementations, and policies explicitly.

## Package layout

| Module | Responsibility |
| --- | --- |
| `common/` | Result contracts, tensor isolation, mutation and finiteness checks, fingerprints, reports, deployment hooks |
| `gate1/` | Local forward/backward pair against the declared PyTorch reference |
| `gate2/` | Actual autograd integration, calibrated tensor errors, operator/model adapters |
| `gate3/` | Continuous training measurement and calibrated loss-trajectory acceptance |
| `payloads/` | Generic storage and loading of recorded training calls, with exact layouts and frozen splits |
| `integrations/` | Shared capture, replay and gate drivers; per-model recorders and references (Qwen3-0.6B, Qwen3.5-4B) |

Model dependencies are imported inside the adapters that need them. The CPU
core tests do not require a checkpoint, real dataset, GPU, or LLM service.
Optional model-adapter tests use small, randomly initialized models.

## Reports and reference contracts

Every verdict distinguishes `passed`, `failed`, and `incomplete`. A candidate
error can be `failed` while other missing evidence makes `complete` false.
An invalid reference, absent policy, incompatible identity, or smoke-only check
never yields a pass. Reports serialize with `allow_nan=False`; removed
non-finite numbers are explicitly identified.

`OpDecl.reference_contract` is an optional extension for model-derived tasks:

- A generic declaration keeps its existing mathematical forward, optional
  `reference_dtype`, and separate performance `runtime_forward`.
- A model-reference declaration names one numerical computation. It cannot
  request implicit `reference_dtype` promotion or a different runtime forward.
  Explicit casts inside the selected reference remain part of its semantics.
- Reports record the contract and, when the reference module exposes it,
  `source_identity()`. The caller remains responsible for selecting and pinning
  an actual model source; this field alone does not prove extraction fidelity.

Reference fingerprints use canonical, type-preserving encodings for code and
constants, including nested frozensets. They are stable across hash seeds but
change with semantic dependencies. Gate 2 binds the fingerprint version and
rejects incompatible policies. Unidentifiable references fail closed.

## Gate 1: local numerical verification

```python
report = gate1.verify_pair(op, candidate_module, device="cpu")
assert report.status == "passed", report.to_dict()
```

The candidate backward receives its own forward's saved state. Every declared
output and required gradient is checked by name, structure, shape, dtype,
finiteness, and the elementwise rule `abs(c-r) <= atol + rtol*abs(r)`.
Comparisons use float64. Candidate input mutation fails unless the declaration
explicitly allows it; allowed content overwrites must preserve layout.

Generic tasks use the declaration's effective tolerance. Model-reference tasks
use a fixed, versioned table based on the reference result's dtype:

| Result dtype | atol | rtol |
| --- | ---: | ---: |
| float16 | 1e-5 | 1e-3 |
| bfloat16 | 1e-5 | 1.6e-2 |
| float32 | 1e-5 | 1.3e-6 |
| float64 | 1e-7 | 1e-7 |

The table in `gate1/tolerances.py` is frozen from PyTorch 2.11's
`torch.testing.assert_close` defaults. Model tasks bypass declaration overrides,
result multipliers, and shape scaling. The policy identity is recorded in the
report. This is the selected numerical policy, not a proof that all meaningful
errors are detected; tiny gradients can still fall below the absolute floor.

Callers may supply `Gate1Case` values, lazy recorded payloads, or declaration-
generated cases. `mode="smoke"` checks execution and contracts but never passes
numerical verification. This core API does not enforce an application-wide
choice of real versus generated inputs.

## Gate 2: framework integration

`OperatorScope` runs a declared operator as a parameter-owning module through
its actual autograd deployment entry. `PayloadOperatorScope` uses that same
path with recorded inputs, parameters and output cotangents. `ModuleScope`
and `model_scope` support full-model evaluation; `WorkloadModelScope` bridges
existing model workloads. There is no block scope. The deprecated `block.py`
import shim only rejects the old scope and does not restore a block benchmark.

Each sample performs one forward/backward with matched inputs, weights, RNG
conditions, and cotangents, without an optimizer update. Check every required
output, parameter gradient, and differentiable-input gradient. Also check
parameter identity/tying, initial state, mutation, and replacement call counts.

For reference tensor `r` and candidate tensor `c`:

```text
E2   = ||c-r||2   / max(||r||2, epsilon)
Emax = ||c-r||inf / max(RMS(r), epsilon)
epsilon = 1e-30
```

Each required result must satisfy both frozen limits. Calibration runs at least
two native executions and one identified `torch.compile` control, run twice.
Controls must execute valid, finite results with the required structure, state
and replacement coverage; the compiler backend must actually be invoked, and the
compile control must reproduce itself bit for bit. The native-repeat term stays
in the budget; whether the native repeats were bitwise is recorded per sample
(`repeatability`).

```text
T = gamma * max(native-repeat error, compile-native error, floor)
```

The sample protocol depends on scope:

- **Operator:** pool per-result maxima over calibration samples, then evaluate
  on disjoint samples. This works for generated samples and recorded calls.
- **Model:** calibrate separately per sample/result; fresh verification must
  match that sample's input, labels/context, cotangent, initial state, mode and
  seed. A budget cannot be transferred to another model sample.

Policies bind scope, reference, environment, numerical configuration, patch set,
samples and result contracts. Candidate errors never set the budget. Missing or
incompatible calibration withholds numerical judgement.

With `boundaries=True`, replay each replaced call using native and candidate
implementations on that call's local inputs and external cotangents. Check
replay outputs/gradients, candidate repeat, and consistency with live execution.
These are tolerance checks, not a general bitwise determinism guarantee.
Compile-calibration and candidate-verification internal activations may differ
even when their external model inputs match. Arbitrary internal temporaries
and candidate saved states need not match the reference.

## Recorded training payloads

The generic `payloads` package stores recorded call arguments and output
cotangents with content hashes, strides, storage offsets, aliases and frozen
splits. It loads samples lazily and rejects corruption, invalid split
membership, and any tensor whose loaded content differs from the digest
recorded at capture. Gapped (non-dense, non-overlapping) views are stored as
they are. Local reference outputs and gradients are recomputed for checking;
they are not supplied as candidate answers.

`payloads.payload_cases` provides Gate 1 cases. `PayloadOperatorScope` provides
Gate 2 operator samples, with corpus identity bound into its policy. A generic
Gate 1 CLI is available:

```bash
python -m evograd.verification.payloads.cli gate1 \
  --declaration task.py:op --program candidate.py \
  --corpus corpus/manifest.json --case CASE_ID --split acceptance --out report.json
```

Model integrations live in `integrations/`. A model (`integrations/<model>/`)
supplies a `base.Model`: its pinned pretrained workload on real WikiText-2, a
recorder of its training calls (`capture.py`), and its model reference
(`reference.py`: each observed case's task re-declared with the model's own
computation). The capture loop, replay and fidelity (`base.py`), the gates
(`gates.py`) and the command line (`cli.py`) are shared. Every L1 call and L2
boundary of one training step per data order becomes a payload of the case
its shapes bind. Gate 1 runs on generated inputs and acceptance payloads,
operator Gate 2 calibrates on orders 101/202 and judges 303/404; Level-2
candidates then get model Gate 2 and Gate 3 (`Model.gate3_steps`). Level-1
candidates get Gate 1 and operator Gate 2 only. Integrated: Qwen3-0.6B and
Qwen3.5-4B. Evaluator routing is not changed.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.qwen3_0_6b capture --out CORPUS
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.qwen3_0_6b verify \
  --corpus CORPUS/manifest.json --out DIR --candidate qwen3_swiglu_mlp/swiglu_mlp=candidate.py
```

## Gate 3: continuous training and global loss

`measure_training` creates a fresh model and optimizer, fixes the run's RNG,
and performs the prescribed updates independently. It records the scalar
objective actually differentiated before each optimizer update. Measurement
alone and `compare_curves` are not acceptance verdicts.

For one fixed batch sequence and training configuration:

```text
d[t] = objective_candidate[t] - objective_native[t]
R = sqrt(mean(d[t] ** 2))
M = max(abs(d[t]))
T_R = gamma * max(R_native_repeat, R_compile_native, floor_R)
T_M = gamma * max(M_native_repeat, M_compile_native, floor_M)
PASS loss iff R <= T_R and M <= T_M
```

```python
controls = [gate3.TrainingControl(native_a, native_b, compiled)]
calibration = gate3.calibrate_global_loss(
    controls, floor_rms=1e-4, floor_max=1e-4, gamma=2.0)
verdict = gate3.verify_training(candidate_run, fresh_native_run,
                               policy=calibration.policy)
```

The supplied runs must share the complete binding: plan, initial model state,
optimizer configuration, adapter, environment, reference/numerical context,
exact data sequence and seed. With multiple control groups, compute each
trajectory's metrics before taking maxima. Each candidate trajectory is judged
separately; there is no cancellation by averaging repetitions or removing steps.
Missing, reordered, truncated or non-finite records cannot pass. A calibration
execution is not independent candidate-verification evidence.

This core checks finite losses, complete step records, and that every replaced
site runs in every step (a site that did not run leaves the loss native). It
does not scan parameters, gradients and optimizer-state tensors for finiteness;
a non-finite gradient reaches the next step's loss through the optimizer update.
Applications can add such checks through `on_step`. Calibration records whether
the native repeat was bitwise (`native_repeat_bitwise`). Deterministic settings
remain application responsibilities.

Historical window and stepwise policies remain available through separate
schemas for compatibility. `verify_training` dispatches by schema; it never
reinterprets them as global-loss policies. Their tests are retained separately.

## Validation

From the repository root, with the project dependencies installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests/verification -t .
```

Tests cover numerical verdicts, malformed evidence, fingerprints, sample
binding, external cotangents, live gradients, aliases, mutations, native/compile
controls, corpus round trips, and uninterrupted training on small CPU fixtures.
Experiment-driver tests and their historical scripts are deliberately separate.
Passing this suite does not establish a GPU candidate's whole-model correctness.
