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
  that rounds differently does to the whole model (Qwen3.5-4B gdn_in_proj,
  measured while model Gate 2 still compared the logits: logits `E2` 4-5%,
  against 0 from compile).

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
(`base.real_run`; see [Regenerating the snapshots](#regenerating-the-snapshots)).
`capture` refuses, before writing anything, a snapshot whose `run` is missing
or differs from its own (`base.require_harvested_run`). Gate 1 and operator
Gate 2 (gamma 2) both use every payload of the case, orders 101/202/303/404
(`PAYLOAD_SPLITS`): operator Gate 2 calibrates on those payloads and judges
fresh executions of the same ones. The corpus split into calibration (101/202)
and acceptance (303/404) only keeps 303/404 out of the search's feedback.
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

`verify` combines every accepted Level-2 candidate, without timing; that
combination is a verification diagnostic, not a Level-3 result. The benchmark's
Level-3 combination is the protocol's (`evograd evaluate`,
`evaluation/protocol.py`, `COMBINATION_RULE`): each agent's accepted Level-2
candidates whose single-site speedup over eager, both with the training step
replayed from one CUDA graph (T3-G), has a 95% bootstrap interval with its
lower bound strictly above one. A candidate without a T3-G result is not
selected. A model whose eager step cannot be captured (on GH200, MegaFold and
OLMoE) is judged by the eagerly launched step's interval (T3) instead. The
resulting combination still needs its own model Gate 2 and Gate 3; individual
passes do not establish combined correctness. This deployment path installs
Level-2 model sites, not independent Level-1 artifacts.

Single-site training records are stored under `gate3_single_site/<case>/`,
separately from `combination/`. Controls are shared through `--state` when their
training identities and patch sets match. Candidate runs also bind to their
own deployment, and interrupted Gate 3 runs resume from their checkpoints.
The [evaluation protocol](../evaluation/README.md) reports accepted L1/L2
results at Tier 2 through deployment and autograd, and whole-model results at
Tier 3; Tier 1 remains search feedback and standalone pair diagnostics.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.qwen3_0_6b capture --out CORPUS
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.qwen3_0_6b verify \
  --corpus CORPUS/manifest.json --out DIR --candidate qwen3_swiglu_mlp/swiglu_mlp=candidate.py
```

### Regenerating the snapshots

Every model but MegaFold harvests the step its capture records first:
`base.real_run` runs `base.prepare_run` (the setup `capture` uses too: a CUDA
check before anything is read, `configure()`, the pinned workload and the
first TRAIN batch of each order) for order 101, builds a fresh native model
and returns the run's identity, which the harvest records as `run` in its
manifest and snapshot. One format for all eight models (`base.run_identity`):

| Key | Content |
| --- | --- |
| `model` | the integration's name |
| `workload` | `Model.workload_config`, the weights and data pins (`base.workload_pins`); a file it names, DiT's `init_checkpoint`, appears as `init_checkpoint_sha256`, the digest of its content, never as its environment-dependent path |
| `data` | the data identity `Model.train_batches` records, without its per-order batches |
| `order` | `101` (`base.HARVEST_ORDER`) |
| `batch` | that order's first batch, with its content digest |
| `parameter_state` | the fresh model's parameters (`state_sha256`) |
| `execution` | the settings it ran under; recorded, not compared |

Every snapshot writer but MegaFold's (`harvest.snapshot --write` to the tracked
file, `harvest.harvest --write-snapshot`) applies `base.require_real_run`: no `run`
(a `--synthetic` or reduced harvest), a missing key, another order or model,
workload pins that changed since the harvest (a regenerated warm start has
another digest) or another workload id refuse the write. `capture` compares
every key but `execution` with its own run. MegaFold is deliberately exempt
(`requires_harvested_run=False`): its harvest is left as it is and records its
run as `config` and `data`.

On DeltaAI: one GH200 on `ghx4-interactive` (1 h, 8 CPUs; `--mem` per model
below), the evograd Python environment active, from the repository root, with
the environment `candidates/evaluate_model.py` (`environment()`) gives a model:

```bash
srun -A bcjw-dtai-gh --partition=ghx4-interactive --gpus=1 --cpus-per-task=8 --mem=32g \
  --time=01:00:00 --pty bash
export PYTHONPATH=$PWD/src CUBLAS_WORKSPACE_CONFIG=:4096:8 CXX=/usr/bin/g++
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export V=/work/hdd/bcjw/wzhan/evograd_verification
export HF_HOME=$V/hf_home EVOGRAD_BENCHMARK_DATA_ROOT=$V/data
export EVOGRAD_LLAMA_HF_CACHE=/work/hdd/bcjw/kramesh/hub   # the gated Llama checkpoint
export EVOGRAD_HF_CACHE=$V/hf_cache   # qwen3_0_6b, qwen3_5_4b: /work/hdd/bcjw/wzhan/hf_cache
export TMPDIR=/tmp/$USER TRITON_CACHE_DIR=/tmp/$USER/triton TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor
mkdir -p $TMPDIR
```

The four language models write a manifest, then derive the snapshot without
a GPU: `--validate` may differ from the tracked snapshot in identity fields
only (the schema, the workload id and the hashes that follow it, `run`), never
in a case's dims, dtype or frequency; then `--write`.

```bash
# qwen3_0_6b, --mem=32g, EVOGRAD_HF_CACHE=/work/hdd/bcjw/wzhan/hf_cache
H=results/benchmark/topdown/qwen3_0_6b/harvest.json
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest --out $H
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot --harvest $H --validate
python -m evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot --harvest $H --write
# llama_3_2_1b, --mem=32g, EVOGRAD_HF_CACHE=$V/hf_cache (WikiText) + EVOGRAD_LLAMA_HF_CACHE
H=results/benchmark/topdown/llama3_2_1b/harvest.json
python -m evograd.benchmark.topdown.llama3_2_1b.harvest.harvest --out $H
python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot --harvest $H --validate
python -m evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot --harvest $H --write
# olmoe_1b_7b, --mem=64g (14 GB of BF16 weights are read on the host), EVOGRAD_HF_CACHE=$V/hf_cache
H=results/benchmark/topdown/olmoe_1b_7b/harvest.json
python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest --out $H
python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.snapshot --harvest $H --validate
python -m evograd.benchmark.topdown.olmoe_1b_7b.harvest.snapshot --harvest $H --write
# qwen3_5_4b, --mem=64g (needed to load the checkpoint), EVOGRAD_HF_CACHE=/work/hdd/bcjw/wzhan/hf_cache
H=results/benchmark/topdown/qwen3_5_4b/harvest.json
python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest --out $H
python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot --harvest $H --validate
python -m evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot --harvest $H --write
```

Each language model's `TASK_SOURCES` changed after its snapshot was written, so
its `--validate` also differs in these Level-2 `tasks` entries:

- Qwen3-0.6B: `qwen3_attention_block` (the whole attention module, 28 calls)
  and `qwen3_lm_head_ce` (the LM head with the loss, 1 call) replace
  `qwen3_qkv_norm_rope` and `qwen3_attention`.
- Llama-3.2-1B: `llama3_attention_block` (16 calls) replaces `llama3_qkv_rope`
  and `llama3_attention`.
- OLMoE-1B-7B: `olmoe_attention_block` (16 calls) replaces `olmoe_qkv_norm_rope`
  and `olmoe_attention`; `olmoe_residual_rmsnorm` counts 32 calls, not 16.
- Qwen3.5-4B: `qwen3_5_attention_block` (8 calls) replaces
  `qwen3_5_qkv_norm_partial_rope` and `qwen3_5_gated_attention`;
  `qwen3_5_residual_rmsnorm` counts 64 calls, not 32.

The other Level-2 entries and every `level1` case must not move. Until a
snapshot is written, its removed entries are no cases and its new tasks have
none.

The vision and generated models write the snapshot in the harvest, all at
`--mem=32g` (Qwen2-VL: 64g if it runs out) with `EVOGRAD_HF_CACHE=$V/hf_cache`. They read their real-data
pools under `$EVOGRAD_BENCHMARK_DATA_ROOT/<model>/batches` (DiT also its warm
start `warm_start.pt` there; overrides `EVOGRAD_VISION_POOL_<MODEL>`,
`EVOGRAD_DIT_LATENT_POOL`, `EVOGRAD_DIT_S_2_WARM_START`) and DLRM its Criteo
arrays under `$EVOGRAD_BENCHMARK_DATA_ROOT/torchrec_dlrm/processed`
(`EVOGRAD_DLRM_CRITEO_DIR`):

```bash
for m in convnextv2_tiny qwen2_vl_2b dit_s_2 torchrec_dlrm; do
  python -m evograd.benchmark.topdown.$m.harvest.harvest \
    --out results/benchmark/topdown/$m/harvest.json --write-snapshot
done
```

Their cases must not move either; per snapshot, against the committed one. The
check compares the Level-2 tasks both snapshots hold: Qwen2-VL adds
`vision_attention`, `vision_residual_layernorm`, `text_attention_block` and
`text_residual_rmsnorm`, ConvNeXt V2 `dwconv_layernorm`, and DiT
`dit_attention` and `dit_adaln_residual` (its `blocks.residual` at 23 calls and
`final_residual` at 1).

```bash
python -c 'import json, subprocess, sys
cases = lambda s: {k: {t: [(c["config_id"], c["frequency"], c["dims"]) for c in e["configurations"]]
                       for t, e in s[k].items()} for k in ("level1", "tasks")}
p = sys.argv[1]
old, new = (cases(json.loads(subprocess.check_output(["git", "show", "HEAD:" + p]))),
            cases(json.load(open(p))))
new["tasks"] = {t: e for t, e in new["tasks"].items() if t in old["tasks"]}
print(old == new)' src/evograd/benchmark/topdown/convnextv2_tiny/harvest/snapshot.json
```

Then commit the eight `snapshot.json` files and update what pins the old
identity: `tests/test_task_registry.py` (the Qwen3 manifest's workload id),
`tests/qwen3/test_level2_swiglu_mlp.py` (`PROVENANCE_CHAIN`: workload id and
manifest hash), `tests/llama3/test_real_text.py` (the canonical workload id
equals the snapshot's), and the old revision in each language model's
`READABLE_SCHEMAS`. After that, the random canonical Tier-3 vision and
generated workloads fall back to specification preflight cases (their workload
id is no longer the snapshot's). Corpora
captured before keep the old `snapshot_hash` and still verify; new captures
refuse until the snapshots are regenerated.

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
