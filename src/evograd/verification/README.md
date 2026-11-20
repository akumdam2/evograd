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
report = gate1.verify_pair(op, candidate_module, cases=cases)
assert report.status == "passed", report.to_dict()
```

The candidate backward receives its own forward's saved state. Every declared
output and required gradient is checked by name, structure, shape, dtype,
finiteness, and the rule `max(abs(c-t)) <= bound*max(abs(t))` per result and call, where
`t` is the float32 truth (the model reference with its floating inputs below
float32 widened, TF32 off) and `max(abs(t))` that result's largest magnitude on
that call. Structure (shape, dtype, exact values of non-floating results) is
checked against the reference at the real dtypes ("native"). Comparisons use
float64. Candidate input mutation fails unless the declaration explicitly allows
it; allowed content overwrites must preserve layout.

Gate 1 is the search-time check that a kernel is written correctly: it needs no
calibration and never consults `torch.compile`; small faults are operator Gate 2's
job. Why a float32 truth: native carries its own rounding (a bfloat16 RMSNorm
weight gradient is 2-3% of its largest value off), so judging against it rejects
implementations that are more accurate than the model; float32 is within 4e-4 of
a float64 evaluation on the benchmark's cases (4e-3 on one ill-conditioned
attention block), at least 30 times below native's own error, and runs every
kernel the model uses (float64 does not: grouped matmul). Every element is checked,
with one allowance per result: a small element is usually the difference of
larger terms, whose rounding sets its error. There is no absolute term: recorded
gradients reach down to 1e-13 as a whole, where any fixed absolute allowance would
accept all zeros; a result whose truth is exactly zero passes only when the
candidate's is exactly zero.

`bound` is the larger of the budget of the call's least precise floating dtype (its
tensor inputs, upstream gradients and the result: a float32 gradient computed from
bfloat16 activations carries bfloat16 rounding) and twice native's largest
deviation from the truth on the judged calls, `max(abs(n-t))/max(abs(t))` over every
call (`gate1.runner.native_errors`, recorded in the report). It is operator Gate 2's
form, `err(candidate) <= max(floor, 2 * err(trusted))`, with Gate 1's lenient budget
as the floor and native as the only trusted implementation. Where native is less
accurate than the budget the result is ill-conditioned at its dtype, and a
candidate is held to native's accuracy. On the benchmark that is the attention
tasks; everywhere else twice native's error stays within the budget (rejudged on
the 69 benchmark cases, 2026-11-19: no verdict of a seed, `torch.compile` or a
rounding perturbation changes, and the only fault verdict that changes is OLMoE's
attention `dx` scaled by 1.05, which operator Gate 2 rejects). The attention
tasks' query and key gradients are ill-conditioned at low
precision. Each row of the softmax gradient sums to zero, so `dq_i = s * sum_j
dS_ij k_j` is the residue left after any component common to all keys cancels,
and the q/k-norm weight gradients sum such residues over every token and head;
ordinary bfloat16 rounding (0.2-0.4% per element) then moves them by several
percent. Measured on the search payloads (float32 truth, one rounding point
changed at a time): the terms of `dq` are a median 138, 9.6 and 5.5 times `|dq|`
on Qwen2-VL's, Qwen3's and Qwen3.5's worst layer (2.6, 2.9 and 4.4 once the keys
are centred); Qwen2-VL's layer-0 keys are its k_proj bias (norm 382 and 1044
against 12-14 for the input-dependent part) in the slowest RoPE dimensions, and
Qwen3's carry a k-norm weight of 96.5 (median 2.3) on one slowly rotating
dimension. Native is then up to 8% off the truth on Qwen3's and Qwen3.5's blocks
and up to 89% on Qwen2-VL's text attention, and a candidate is held to native's
accuracy instead. Native error never depends on the candidate: a search run
(`EVOGRAD_GATE1_NATIVE_ERROR_CACHE`) and `evograd evaluate --state` compute it
once per set of calls and reuse it. One case is beyond this rule: on Qwen2-VL-2B's
text attention block native is 47% off on `dx`, so Gate 1 there cannot reject a
gradient scaled by 1.05, and `torch.compile` fails it (`dq_bias`, 3.4 times native's
error). It is a recorded exception, as is MegaFold's layernorm bias gradient, a
cancellation residue of 1e-23 on which the frontend seed is 3-7 times native's
error (D12).

Per call, native runs first, then the candidate (whose working state is then
released), then the truth: the truth is twice native's size, and the largest
calls (MegaFold's SwiGLU, 1.2e9 elements per tensor) fit in device memory only
when it never coexists with the candidate's working state.

Gate 1 judges only model-reference tasks; any other declaration is refused.
Declarations carry no tolerances. The budgets are a fixed, versioned table
(`gate1/tolerances.py`, recorded in the report):

| Dtype | budget |
| --- | ---: |
| float16 | 5e-3 |
| bfloat16 | 4e-2 |
| float32 | 1e-5 |
| float64 | 1e-7 |

A gradient scaled by 1.05 moves its largest element by 5% of the largest
magnitude, so the bfloat16 budget rejects it wherever native is accurate. Correct
implementations (`torch.compile`, the frontend seeds) were measured against a
float64 evaluation on the benchmark's cases (2026-11-19); float16's budget is not
measured. This is the selected numerical policy, not a proof that all meaningful
errors are detected.

Callers supply the cases: lazy recorded payloads (`payload_cases`) or values
given the same way (`Gate1Case`). There is no default case set and no
generated input; an empty case set never passes. `mode="smoke"` checks
execution and contracts but never passes numerical verification.

## Gate 2: framework integration

`OperatorScope` runs a declared operator as a parameter-owning module through
its actual autograd deployment entry. `PayloadOperatorScope` uses that same
path with recorded inputs, parameters and output cotangents. `ModuleScope`
and `model_scope` support full-model evaluation; `WorkloadModelScope` bridges
existing model workloads. Each sample performs one forward/backward without an
optimizer update, and every required output, parameter gradient and
differentiable-input gradient is checked.

### Operator scope (`gate2/operator.py`)

Every recorded payload of the case (orders 101/202/303/404, every layer's call)
runs four times on its exact values: the float32 truth `t` (Gate 1's), native,
the `torch.compile` control (native under inductor) and the candidate, the last
three through the scope's autograd module. Each result `z` is measured against
the truth, in float64:

```text
E2(a, t) = ||a - t||_2 / ||t||_2        overall error
M(a, t)  = max|a - t| / max|t|          worst element, in units of the largest value

pass  <=>  on every payload x, for m in (E2, M):
           m(candidate_x[z], t_x[z]) <= T[z, m]
T[z, m] = max( 2 * max over payloads x and trusted r in (native, compile) of m(r_x[z], t_x[z]),
               floor(dtype of z) )
floor: bfloat16 3.9e-3 (2^-8), float16 4.9e-4 (2^-11), float32 1e-5
```

It is Gate 1's form, `err(candidate) <= max(floor, 2 * err(trusted))`, with
`torch.compile` added to the trusted implementations, `E2` added to the worst
element, and a floor of one rounding instead of Gate 1's lenient budget. One
threshold per result per case, pooled over every payload as Gate 1 pools native's
error. The factor 2 follows FlashAttention's tests (a kernel's error at most twice
the PyTorch implementation's against an fp32 reference). The floor matters only
where both trusted implementations are (nearly) exact: for bfloat16 and float16
it is the unit roundoff (one correct rounding gives `E2 <= u`;
flash-linear-attention and DeepGEMM hold bfloat16 kernels to an `E2` of about
4e-3 to 5e-3), for float32 1e-5, as in torch.testing and TransformerEngine
(different float32 algorithms differ by up to 1.4e-6 on the benchmark's calls).
A result whose truth is exactly zero has error 0 when the candidate is zero too.

It catches precision lost by a kernel that computes the right thing:
accumulation in bfloat16 (Liger's fused linear cross-entropy weight gradient,
1.3% against native's 0.5%), dropped float32 upcasts, a small systematic scale
(`E2`), or a few elements less accurate than the trusted implementations (`M`).
Gross errors and input mutation are Gate 1's; propagation through the model
and through training are model Gate 2's and Gate 3's. Nothing is calibrated or
cached, and no structural check beyond what the comparison needs (the
candidate runs, and each result has native's shape and dtype and is finite).
If the compile control cannot be built or fails on a payload, native alone is
trusted there (recorded in the report).

Checked offline (2026-11-19) on the fp64 check's 60 cases and the Liger study's
7, with fp64 as the truth: every frontend seed passes (largest ratio 0.51), Liger
passes except its default fused linear cross-entropy (its weight gradient in
bfloat16 chunks), every gradient scaled by 1.05 is rejected (67/67), and 6 of 7
scaled by 1.01.

### Model scope (`gate2/calibration.py`)

For reference tensor `r` and candidate tensor `c`:

```text
E2   = ||c-r||2   / max(||r||2, epsilon)
Emax = ||c-r||inf / max(RMS(r), epsilon)
epsilon = 1e-30
```

Each required result must satisfy both frozen limits. Calibration runs at least
two native executions and one identified `torch.compile` control, run twice.
Controls must execute valid, finite results with the required structure, state
and replacement coverage; the compiler backend must actually be invoked, and
both the native reference and the compile control must reproduce themselves bit
for bit (recorded per sample as `repeatability`). A repeat therefore adds
nothing to the budget: under the deterministic settings no native ever differed
from its repeat (P8 audit, 2026-11-13), and one that does leaves no policy.
Parameter identity/tying, initial state, mutation and replacement call counts
are checked too.

```text
T = gamma * max(compile-native error, floor, rounding perturbation)
```

Thresholds are pooled per result over the calibration samples (the four
batches), and judge fresh executions of exactly those samples: each must match
its calibrated sample's input, labels/context, cotangent, initial state, mode and
seed. A third control, the rounding perturbation, adds its own term: the native
site with each floating output and input gradient moved one ulp with probability
1/3, each tensor kept within one rounding (`E2` at most the unit roundoff), run
under four fixed seeds (11, 22, 33, 44) and taken at its largest. Where compile
reproduces the native site bit for bit, it is what a correct reimplementation
that rounds differently does to the whole model (Qwen3.5-4B gdn_in_proj: logits
`E2` 4-5%, against 0 from compile).

At model scope a language model is called as `workload.loss` calls it, and
that loss, seeded with 1, is the backward: the parameter gradients compared are
the ones training applies. The forward is judged on the logits, which determine
the next-token loss; the loss only seeds the backward (`backward_only`) and is
not a result. Under `lm_head_ce`, the LM head fused with the loss, no deployment
has logits, so the LM head's input (the decoder's last hidden state) and the
loss are compared instead. Other models compare their primary output and their
loss. Every parameter gradient is compared in each case.

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
or differs from its own (`base.require_harvested_run`). It records payloads for
the cases its `--cases` selection picks (`base.recorded_cases`): by default
(`benchmark`, user decision 2026-11-11) the model's benchmark rows in
`benchmark/search_set.json` (its searched cases plus its `identical_config_of`
cases, the only cases an evaluation verifies or times), refused when the search
set was generated from another snapshot; `--cases all` records every observed
case. Every recorded call still binds exactly one observed case, and each
recorded case must receive its calls per step in every order. The corpus
identity lists the recorded cases (`cases`) and states the rule
(`case_selection`: rule, the search set's SHA-256 and snapshot hash, the
observed cases left out); a request for any other case fails
(`Corpus.require_case`), in `open_run` before any gate. Gate 1 and operator
Gate 2 both judge every payload of the case, orders 101/202/303/404
(`PAYLOAD_SPLITS`). The corpus split into calibration (101/202) and acceptance
(303/404) only keeps 303/404 out of the search's feedback.
Model Gate 2 runs the candidate alone at its site, pooling budgets over the
four batches (compile and the rounding perturbation) with
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
# The model's benchmark rows; --cases all records every observed case.
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

Each language model's `TASK_SOURCES` had changed since its previous snapshot, so
the regeneration of 2026-11-11 also moved these Level-2 `tasks` entries:

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

The other Level-2 entries and every `level1` case must not move.

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
manifest hash) and `tests/llama3/test_real_text.py` (the snapshot's workload id
is the real run's). After that, the random canonical Tier-3 vision and
generated workloads fall back to specification preflight cases (their workload
id is no longer the snapshot's). Corpora captured before keep the old
`snapshot_hash`, which `verify` refuses; recapture them.

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
T_R = gamma * max(R_compile_native, floor_R)
T_M = gamma * max(M_compile_native, floor_M)
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
Applications can add such checks through `on_step`. Calibration refuses a
native repeat (`native_b`) that does not reproduce `native_a` bit for bit.
Deterministic settings remain application responsibilities.

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
