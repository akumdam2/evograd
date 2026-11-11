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
| `integrations/` | Shared capture, replay and gate drivers; per-model recorders and references (every model in `integrations.PACKAGES`) |

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
cases = payload_cases(op, workload, corpus, case_id=case_id, split="acceptance")
report = gate1.verify_pair(op, candidate_module, cases=cases, envelope=envelope)
assert report.status == "passed", report.to_dict()
```

The candidate backward receives its own forward's saved state. Every declared
output and required gradient is checked by name, structure, shape, dtype,
finiteness, and the elementwise rule `abs(c-r) <= max(atol + rtol*abs(r), envelope)`,
where `envelope` is gamma times the largest elementwise deviation the trusted
`torch.compile` control showed on that result over the operator Gate 2
calibration payloads, never less than gamma ulps of the reference's largest
magnitude. Comparisons use float64. Candidate input mutation fails unless the declaration
explicitly allows it; allowed content overwrites must preserve layout.

Gate 1 judges only model-reference tasks; any other declaration is refused.
Declarations carry no tolerances. The tolerance is a fixed, versioned table
based on the reference result's dtype:

| Result dtype | atol | rtol |
| --- | ---: | ---: |
| float16 | 1e-5 | 1e-3 |
| bfloat16 | 1e-5 | 1.6e-2 |
| float32 | 1e-5 | 1.3e-6 |
| float64 | 1e-7 | 1e-7 |

The table in `gate1/tolerances.py` is frozen from PyTorch 2.11's
`torch.testing.assert_close` defaults, or the compile envelope where that is
larger. The policy identity is recorded in the report. This is the selected numerical policy, not a proof that all meaningful
errors are detected; tiny gradients can still fall below the absolute floor.

Callers supply the cases: lazy recorded payloads (`payload_cases`) or values
given the same way (`Gate1Case`). There is no default case set and no
generated input; an empty case set never passes. `mode="smoke"` checks
execution and contracts but never passes numerical verification.

## Gate 2: framework integration

`OperatorScope` runs a declared operator as a parameter-owning module through
its actual autograd deployment entry. `PayloadOperatorScope` uses that same
path with recorded inputs, parameters and output cotangents. `ModuleScope`
and `model_scope` support full-model evaluation; `WorkloadModelScope` bridges
existing model workloads.

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
  The floor term is the larger of the fixed floor and one rounding of every
  element in the result's dtype, from the native reference: `E2` at the unit
  roundoff, `Emax` at 2 ulps of the largest magnitude over the RMS. Where the
  compile control reproduces native bit for bit (a vendor GEMM, SDPA or
  convolution), the fixed floor alone was about 100x below one bf16 rounding.
- **Model:** calibrate separately per sample/result; fresh verification must
  match that sample's input, labels/context, cotangent, initial state, mode and
  seed. A budget cannot be transferred to another model sample. A third control,
  the rounding perturbation, adds its own term: the native site with each
  floating output and input gradient moved one ulp with probability 1/3, each
  tensor kept within one rounding (`E2` at most the unit roundoff), run under
  four fixed seeds (11, 22, 33, 44) and taken at its largest. Where compile
  reproduces the native site bit for bit, it is what a correct reimplementation
  that rounds differently does to the whole model (Qwen3.5-4B gdn_in_proj:
  logits `E2` 4-5%, against 0 from compile).

Policies bind scope, reference, environment, numerical configuration, patch set,
samples and result contracts. Candidate errors never set the budget. Missing or
incompatible calibration withholds numerical judgement.

With `boundaries=True` (not used by the model-scope gate since 2026-10-25), replay each replaced call using native and candidate
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
its shapes bind. The cases and their calls per step come from the model's
harvest snapshot, and that harvest observes the capture's own first step
(`base.real_run`: this package's settings, workload and model builder on the
first TRAIN batch of order 101). The snapshot records that run (`run`: pinned
workload, data identity, the order-101 batch and its digest, the initial
parameter state), and `capture` refuses, before writing anything, a snapshot
whose `run` is missing or differs from its own (`base.require_harvested_run`;
MegaFold, whose harvest records its run its own way, sets
`requires_harvested_run=False`). Gate 1 runs on the acceptance payloads (real
inputs only).
Operator Gate 2 calibrates on orders 101/202 and judges 303/404, with gamma 2.
Model Gate 2 runs the candidate alone at its site, pooling budgets over the
four batches (native repeats, compile and the rounding perturbation) with
gamma 4, on the whole model's outputs and parameter gradients
plus coverage and side effects. Per-call boundary replay is off at model scope
(`GATE2_MODEL_BOUNDARIES`): the local checks of each call belong to operator Gate 2.

| Object | Required correctness gates |
| --- | --- |
| Level-1 candidate | Gate 1 + operator Gate 2 |
| Level-2 candidate | Gate 1 + operator Gate 2 + single-site model Gate 2 + single-site Gate 3 |
| Level-3 combination | Model Gate 2 on the combined patch set + Gate 3 |

Single-site Gate 3 starts only after the preceding three gates pass; otherwise
its stage is marked `skipped`. A candidate is accepted only when every
applicable gate passes. Models without model-scope gates retain operator-only
acceptance and report the combination as not applicable. Gate 3 uses the
model's `Model.gate3_steps`, optimizer settings and accumulation for both the
single-site and combined runs.

`verify` combines every accepted Level-2 candidate. The benchmark protocol
(`evograd evaluate`, `evaluation/protocol.py`) selects each agent's positive
Level-2 candidates: accepted, with single-site Tier 3 speedup over eager whose
95% bootstrap interval has a lower bound strictly above one. The resulting
combination still needs its own model Gate 2 and Gate 3; individual passes do
not establish combined correctness. This deployment path installs Level-2
model sites, not independent Level-1 artifacts.

Single-site training records are stored under `gate3_single_site/<case>/`,
separately from `combination/`. Controls are shared through `--state` when their
training identities and patch sets match. Candidate runs also bind to their
own deployment, and interrupted Gate 3 runs resume from their checkpoints.
The [evaluation protocol](../evaluation/README.md) reports accepted L1/L2
results at Tier 2 through deployment and autograd, and whole-model results at
Tier 3; Tier 1 remains search feedback and standalone pair diagnostics.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest --out HARVEST.json
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot --harvest HARVEST.json --write
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
