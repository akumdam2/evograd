# Evaluate one model across all agents

Inside the evaluation environment on a GPU allocation:

```bash
python candidates/evaluate_model.py qwen3_0_6b --agent all \
  --out-root "$HOME/adbench-evaluation-all-v2" --dry-run

python candidates/evaluate_model.py qwen3_0_6b --agent all \
  --out-root "$HOME/adbench-evaluation-all-v2" --deadline-min 100
```

`all` selects exactly `openevolve`, `gepa`, and `codex`. Seed, evograd and the
other agents are excluded from this group. It preflights every selected
agent before launching: missing/unfinished searches, missing repaired artifacts,
changed kernel blocks and invalid hashes stop the job. The printed agent list
is the group being evaluated; all three agents are required. There is no fallback from ADBench-redeploy to
original submissions.

The scheduler is case-major on one GPU:

1. For case A, verify every agent and run shared T2 measurements.
2. Repeat for case B and the remaining cases.
3. Measure model eager and whole-model compile T3 once; measure every agent's
   executable single-site L2 candidates against those shared samples.
4. Select and verify a separate combination for each agent.
5. Measure each combination against the same shared model baseline samples.

Each agent retains separate acceptance decisions and combination results.
Compatible native/compile correctness controls share the model state
directory. Numerical compile controls remain specific to the replacement site
or patch set; they are separate from performance baselines.

## Optional calibration reuse

The existing command keeps its current behavior. These independent flags opt
into broader native-control reuse; they work with `--agent all` and with a
single agent, and pass through `adbench_evaluate.sbatch`:

| Flag | Default | Enabled behavior |
| --- | --- | --- |
| `--share-gate3-native` | off | Two native training trajectories shared across compatible cases and combinations; compile controls stay patch-specific |
| `--share-gate2-calibration-native` | off | Model Gate 2 native reference and independent repeat shared across compatible patch sets; compile and compile-repeat still run for each patch set |
| `--share-gate2-verification-reference` | off | Model Gate 2 verification reference shared across compatible candidates, separately from calibration executions |

For example, inside an existing GPU allocation:

```bash
python candidates/evaluate_model.py qwen3_0_6b \
  --agent all \
  --submission-root /work/hdd/bcjw/wzhan/ADBench-redeploy \
  --out-root "$HOME/adbench-evaluation-three-agents-native-reuse-v1" \
  --deadline-min 100 \
  --tier3-blocks 10 \
  --share-gate3-native \
  --share-gate2-calibration-native \
  --share-gate2-verification-reference
```

Use a new output root when changing these options. The launcher freezes them
in `all_agents.json`, and reports record `settings.calibration_reuse`.
Controls use a separate `state/native_reuse_v1/<three flag values>/` namespace.
The same options and output root resume the experiment.

Gate 3 shared native records contain no installed patch set; a version-2
whole-horizon policy compares their common execution settings against the
patch-specific compiled/candidate runs. Initial state, exact data sequence,
seed, optimizer, plan, reference, environment and numerical checks remain.
Compile and candidate patch sets must still match the experiment. Both
independent native trajectories are retained. Shared native records are saved
under `gate3/shared_native/<settings hash>/` within the selected state namespace.

Gate 2 sharing is model-scope only. Raw outputs and gradients are saved to
private job-local files under `SLURM_TMPDIR` (or the system temporary directory)
and cleaned up on exit. They are read one execution at a time, preserving
original execution IDs, and hits appear in the log and reference evidence.
With four samples and both Gate 2 flags enabled, storage can approach **12 full
native result snapshots**, mostly parameter gradients, plus model snapshot
storage. Ensure scratch has room; serialization and reads mean wall-clock
savings depend on model compute and scratch speed. Completed Gate 2 policies
still persist across allocations, but these raw execution files do not.

Verification references are measured independently of calibration runs;
calibration execution IDs remain forbidden as candidate verification evidence.
The verification-reference flag deliberately changes the rule from a fresh
native execution per candidate to one independently measured reference shared
within the job. Candidate executions and all content/weight checks remain
separate. Calibration samples, repeat counts, training horizons, accumulation,
tolerances, floors, safety factors and timing settings are unchanged.
Models without the relevant model gates receive no additional checks.

## Shared baseline timing

T2 measures each provider once per block per recorded call: all agent
candidates, eager, operator compile, and available declared library baselines.
All providers participate in the same three randomized blocks. Candidate and
baseline bootstrap samples retain matching block indices. Baseline latencies
and raw measurements are identical across agent comparisons for that case;
each candidate is also measured once, rather than remeasured per baseline.

T3 measures each provider in a separate process. One eager and one available
whole-model compile measurement are shared by the model's individual and
combined candidate reports. Confidence intervals independently resample the
same reference blocks and each candidate's blocks.

Shared T2 raw blocks live under `OUT_ROOT/MODEL/shared/tier2/`. Shared T3
provider measurements and assembled reports live under
`OUT_ROOT/MODEL/shared/tier3/`. Reports record
`group_execution=case-major-shared-baselines/1`. Use a fresh output root when
switching from the earlier agent-major scheduler.

## Weight loading

Recorded payloads are enough for operator checks and T2, but do not contain
a complete runnable model and optimizer for model Gate 2, Gate 3 or T3.
The existing model adapters normally build/load fresh model instances for
their controls, samples and candidate runs.

For each case, payload storages are read and hash-checked once into a CPU cache
shared by all agents' verification and T2 calls. Each execution gets private
device copies preserving recorded aliasing/layout. This cache is released
before the next case. It uses host memory proportional to that case's unique
recorded storages.

All-agent mode also caches a pristine unpatched CPU model for checkpoint-backed
Qwen3, Qwen3.5, Llama, OLMoE, ConvNeXt and Qwen2-VL workloads. The shared
checkpoint is loaded once per distinct CPU model spec per job. The CPU model
is serialized once into a private job-local temporary directory. Isolated T3
workers mmap that snapshot rather than reopening the shared Hugging Face
checkpoint. Each build receives an independent model copy before site patches,
gradients or optimizer state are created. Weights still have to be copied to
the GPU for a fresh training run; trained weights are never carried between
agents, cases or controls.

Generated DLRM/DiT models keep their deterministic initialization path; DiT's
warm-start loading is unchanged. The temporary snapshot is deleted when the
worker exits. A new allocation loads the checkpoint again, while compatible
completed evaluation stages and control results can resume from persistent
state. This is checkpoint I/O reuse, not eliminating all model construction or
GPU transfers.

Qwen3.5-4B has a model-specific RNG replay path. Its checkpoint loader draws
small float32 CPU random vectors for Gated DeltaNet `A_log` initialization,
even when the checkpoint supplies the final weights. On the first load, the
cache verifies that repeating precisely those per-layer draws reproduces the
loader's CPU RNG transition, and that CUDA RNG is unchanged. Every cache hit
(including T3 workers) repeats only those small draws from its current RNG
state, then copies the pristine model. Priming restores the caller's RNG.
This permits a single checkpoint load per job without resetting evaluation
seeds or sharing mutable models. Partial checkpoints are rejected because
missing weights could retain seed-dependent initialization. Unknown RNG
changes still fail with an explanation; the generic cache for other models
continues to require RNG-neutral loading.

After syncing the files below, Qwen3.5 no longer needs `--no-weight-cache` for
this known RNG behavior. Use a fresh root when switching from a run that used
that flag (the frozen group plan includes the weight-cache setting):

```bash
python candidates/evaluate_model.py qwen3_5_4b --agent all \
  --submission-root /work/hdd/bcjw/wzhan/ADBench-redeploy \
  --out-root "$HOME/adbench-evaluation-qwen35-rng-cache-v1" \
  --deadline-min 100 --tier3-blocks 10 \
  --share-gate3-native --share-gate2-calibration-native \
  --share-gate2-verification-reference
```

Snapshotting still fails before candidate evaluation for unsupported RNG
changes or models that cannot be serialized. To use the original model
loading path while retaining the all-agent driver:

```bash
python candidates/evaluate_model.py qwen3_0_6b --agent all \
  --out-root "$HOME/adbench-evaluation-all-uncached-v1" --no-weight-cache
```

## Resources and sbatch

CPU snapshots increase host RAM and node-local disk use. The sbatch launcher
requests 96G host memory, one GPU, 8 CPUs, and two hours; adjust for a model
whose host/device memory exceeds that allocation. Loading, copying, patching
and preparation remain outside reported T2/T3 timed regions.

From the cluster checkout on the login node:

```bash
sbatch candidates/adbench_evaluate.sbatch qwen3_0_6b all \
  --out-root "$HOME/adbench-evaluation-all-v2"
```

The default evaluation deadline is 100 minutes in this sbatch script and is
shared across the whole group, not restarted for each agent. Exit 3 means
resubmit the identical command. Compatible completed case/provider stages
are reused before continuing. Reuse the exact agent list and output root;
changing the frozen group requires a new root. Avoid overlapping jobs for the
same model/state directory.

## Files to transfer

From the local repository checkout:

```bash
rsync -avR \
  candidates/evaluate_model.py \
  candidates/adbench_evaluate.sbatch \
  src/evograd/evaluation/model_cache.py \
  src/evograd/benchmark/topdown/qwen3_5_4b/levels/level3/model.py \
  src/evograd/evaluation/model_group.py \
  src/evograd/evaluation/case_group.py \
  src/evograd/evaluation/group_timing.py \
  src/evograd/evaluation/payload_cache.py \
  src/evograd/evaluation/protocol.py \
  src/evograd/pipelines/shared/artifact.py \
  src/evograd/verification/reuse.py \
  src/evograd/verification/integrations/gates.py \
  src/evograd/verification/gate2/calibration.py \
  src/evograd/verification/gate2/runner.py \
  src/evograd/verification/gate3/global_loss.py \
  src/evograd/verification/gate3/policy.py \
  hding4@gh-login03:~/evograd/
```

Transfer only while no job is running from the destination checkout. The
generator change addresses the earlier single-gradient wrapper bug; the
launcher still evaluates Wangjia's already-repaired artifacts.
