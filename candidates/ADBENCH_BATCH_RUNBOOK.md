# OpenEvolve / GEPA / GEAK: DeltaAI batch runbook

This follows `EVALUATION_PLAN_SIX_AGENTS.md`. The launcher runs on the login
node, reads shared records, and submits a two-hour `ghx4` job only when pending
searches exist. Each job uses one GPU, 8 CPUs and 32G. OpenEvolve and GEPA have
10 admitted requests per case; GEAK has two optimization directions with
uncapped, recorded requests. Formal Gate 2/3 evaluation belongs to Wangjia.

## 1. Transfer the new scripts once

These files are local additions, not already on GitHub or DeltaAI. From your
local checkout, set `DELTA_HOST` to the SSH destination you normally use, then:

```bash
rsync -av \
  candidates/adbench_search.py \
  candidates/adbench_search.sbatch \
  candidates/adbench_smoke.sbatch \
  candidates/adbench_cluster_env.sh \
  candidates/submit_adbench_search.sh \
  candidates/ADBENCH_BATCH_RUNBOOK.md \
  "$DELTA_HOST:~/evograd/candidates/"
```

Use your already-updated checkout on DeltaAI. Do not change source files while
jobs using that checkout are running. The scripts do not install dependencies,
rebuild shared task packs, copy payloads, or run formal evaluation.

## 2. Login-node preparation

Your Python 3.12 environment is already installed. Do not recreate it.

```bash
cd "$HOME"
cd "$HOME/evograd"
export ADBENCH_ENV="$HOME/venvs/adbench-20261028"
export PATH="$ADBENCH_ENV/bin:$HOME/.local/bin:$PATH"
export SHARED=/work/hdd/bcjw/wzhan/ADBench
python --version
python -c 'import torch, triton, transformers, gepa, openevolve; print(torch.__version__); print("core imports OK")'
```

The earlier `nvidia-cusparselt-cu12` pip-check warning is not resolved by this
launcher. Your OpenEvolve GPU smoke passed; retain the warning in setup notes.

Create the key file only if you have not already done so. These Bash commands
read keys without echoing them and save shell-quoted values with private access:

```bash
if [[ ! -s "$HOME/.adbench_keys.txt" ]]; then
  set +x
  umask 077
  read -rsp 'OpenAI API key: ' OPENAI_API_KEY; printf '\n'
  read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY; printf '\n'
  printf 'export OPENAI_API_KEY=%q\nexport ANTHROPIC_API_KEY=%q\n' \
    "$OPENAI_API_KEY" "$ANTHROPIC_API_KEY" > "$HOME/.adbench_keys.txt"
fi
chmod 600 "$HOME/.adbench_keys.txt"
```

If GEAK setup has not completed, finish it before GEAK smoke:

```bash
mkdir -p references
if [[ ! -d references/GEAK ]]; then
  git clone https://github.com/AMD-AGI/GEAK references/GEAK
fi
git -C references/GEAK checkout 0881e29473e5554f535ece584bfd15e2c09db906
bash candidates/geak/setup.sh
```

Check Node.js (v20+) and the Codex CLI before GEAK smoke:

```bash
node --version
codex --version
```

GEAK runs its unchanged workflow through Codex with `--model gpt-5.5` and
`OPENAI_API_KEY`. Its endpoint must support the Responses API. Nsight Compute
(`ncu`) is used if available. Refresh the working copy even if a previous
Claude-only GEAK setup exists. Historical Claude smoke runs do not qualify for
this plan: preserve them separately before starting fresh Codex smoke runs.

The plan's broad access check is:

```bash
find /work/hdd/bcjw/wzhan/{ADBench,ADBench_payload,hf_cache,evograd_verification/{hf_home,hf_cache,data}} /work/hdd/bcjw/kramesh/hub/models--meta-llama--Llama-3.2-1B ! -readable 2>&1 | head
test -w "$SHARED" && echo 'ADBench writable'
```

Your earlier errors were in shared `tasks/*/state/gate2/operator`; your
OpenEvolve smoke subsequently read the shared payloads and completed evaluation.
Report those access errors to Wangjia. Do not chmod his files. A missing or
unreadable task pack/payload actually needed by a search must be resolved.

## 3. Finish smoke (GPU jobs, submitted from login)

OpenEvolve's smoke searches and evaluation have already completed. Check the
saved records without rerunning the searches:

```bash
python candidates/adbench_search.py check-smoke --agent openevolve
```

For each of GEPA and GEAK whose smoke is not yet complete, submit this chain
once. Each search and the evaluation get separate allocations, so one slow smoke
case does not consume the next case's allocation. Do not submit the chain twice
while it is queued/running.

```bash
AGENT=gepa                  # repeat later with AGENT=geak
J1=$(sbatch --parsable --job-name="adb-smoke-$AGENT-rmsnorm" \
  --export="ALL,AGENT=$AGENT" candidates/adbench_smoke.sbatch search rmsnorm/rmsnorm)
J2=$(sbatch --parsable --dependency="afterok:$J1" \
  --job-name="adb-smoke-$AGENT-mlp" --export="ALL,AGENT=$AGENT" \
  candidates/adbench_smoke.sbatch search qwen3_swiglu_mlp/swiglu_mlp)
JE=$(sbatch --parsable --dependency="afterok:$J2" \
  --job-name="adb-smoke-$AGENT-eval" --export="ALL,AGENT=$AGENT" \
  candidates/adbench_smoke.sbatch evaluate)
printf 'Smoke job IDs: %s %s %s\n' "$J1" "$J2" "$JE"
```

Monitor from login:

```bash
squeue -u "$USER"
sacct -j "$J1,$J2,$JE" --format=JobID,JobName,State,ExitCode,Elapsed
python candidates/adbench_search.py check-smoke --agent "$AGENT"
```

Wait for successful smoke before formal search for that agent. The checker
validates configured model, budgets, request counts, real packaged files,
positive wall times, and a completed evaluation containing both cases. Also
confirm actual model routing in the agent logs, as the plan asks; the request
ledger itself does not record model names. Logs are `adbench-smoke-JOBID.out`
and `$SHARED/smoke/$AGENT/qwen3_0_6b/runs/$AGENT/*.log`.

If evaluation exits 3, it reached its checkpoint deadline: submit ONLY the same
evaluation again, after the previous job has ended:

```bash
sbatch --export="ALL,AGENT=$AGENT" candidates/adbench_smoke.sbatch evaluate
```

A failed smoke search needs inspection. Do not delete its directory or reset
its ledger merely to make the chain continue. Downstream `afterok` jobs will
not run if an earlier job fails; cancel those waiting jobs if abandoning that
chain. Smoke outputs stay separate from formal runs.

## 4. Formal search: one launcher command

The Python launcher selects the queue with `--partition`. The default is
`ghx4` (two hours); `ghx4-interactive` uses a separate sbatch file requesting
one hour. Both use the same preflight, duplicate-job checks, and serial worker:

```bash
python candidates/adbench_search.py submit --agent openevolve --model qwen3_0_6b
python candidates/adbench_search.py submit --agent openevolve --model qwen3_0_6b --partition ghx4-interactive
# --skip-smoke is also supported with either queue.
```

Codex is also supported with `--agent codex`. To run all pending Codex cases
inside an existing GPU allocation (with credentials and the environment already
loaded), use:

```bash
python candidates/adbench_search.py run --agent codex --skip-smoke
```

This uses the plan's `gpt-5.5`, high reasoning effort, 10 model requests per
case, an 80-minute agent limit, and a 900-second evaluator timeout. The normal
allocation deadline, existing-run checks, and per-agent lock still apply.

Use `submit` on the login node. The `run` action executes inside an allocation
and cannot switch its partition. The worker reads the actual Slurm end time
to respect either allocation length. Queue selection does not change the
per-case search budget; a long case can still reach the allocation cutoff.
The shell launcher below also accepts `--partition ghx4-interactive`.

Run on login, after that agent's smoke passes:

```bash
bash candidates/submit_adbench_search.sh openevolve qwen3_0_6b
```

This prints finished/pending/active-or-interrupted counts and the next pending
case. If everything selected has finished, no GPU job is submitted. Otherwise
one batch searches pending cases serially until time runs short. Finished
failures count as trials and are not retried. An active/interrupted or
unpackaged run is flagged for inspection, never treated as done or restarted.

To select all eight models, omit the model:

```bash
bash candidates/submit_adbench_search.sh openevolve
```

Use either the model-specific or all-model command; do not submit both for one
agent at once. Both are protected by the same per-agent execution lock. After
GEPA and GEAK smoke passes, use the same launcher for each:

```bash
bash candidates/submit_adbench_search.sh gepa qwen3_0_6b
bash candidates/submit_adbench_search.sh geak qwen3_0_6b
```

To bypass smoke preflight explicitly, add `--skip-smoke` after the agent or
model. For example:

```bash
bash candidates/submit_adbench_search.sh gepa --skip-smoke
# Or restrict to one model:
bash candidates/submit_adbench_search.sh gepa qwen3_5_4b --skip-smoke
```

This bypasses only smoke verification. The case roster, existing-run checks,
API key, GPU, and formal search budget remain in force. The GPU runner records
the deviation in the agent's `NOTES.md`. Without measured smoke duration it
starts a new case only when at least 23 minutes remain: the plan's 20-minute
minimum plus three minutes for cleanup. A long case can still be interrupted
at the allocation deadline; its directory and ledger are preserved. The plan
calls for smoke first; use this flag only when intentionally deviating from
that procedure.

### Overlap GEPA or OpenEvolve searches on one GPU

The scheduler and evaluator changes must both be on DeltaAI. From the local
checkout, using your usual SSH destination in `DELTA_HOST`, transfer them while
no job using this checkout is running:

```bash
rsync -avR \
  candidates/adbench_search.py \
  candidates/submit_adbench_search.sh \
  candidates/openevolve/run.py \
  src/evograd/adbench_tools/gpu_lock.py \
  src/evograd/adbench_tools/evaluate.py \
  src/evograd/evolve/evaluator.py \
  "$DELTA_HOST:~/evograd/"
```

`--workers` defaults to 1. For these two agents only, use two workers to
overlap independent case searches while serializing their GPU evaluations:

```bash
bash candidates/submit_adbench_search.sh gepa --skip-smoke --workers 2
bash candidates/submit_adbench_search.sh openevolve qwen3_5_4b --workers 2
# Equivalent Python submission entry point:
python candidates/adbench_search.py submit --agent gepa --skip-smoke --workers 2
# Inside an existing one-GPU srun allocation, with the environment already set:
python candidates/adbench_search.py run --agent gepa --skip-smoke --workers 2
```

Choose one submission command per agent. A single scheduler assigns each case
once and retains the per-agent batch lock. Every case has its own run directory
and 10-request budget. GEAK and Codex currently support only one worker.
The Slurm script still requests one GPU. LLM requests and CPU preparation can
overlap; all search correctness and timing subprocesses for these two agents
use one shared `ADBENCH_GPU_LOCK`. The default path is under
`/work/hdd/bcjw/wzhan/ADBench/gpu-locks/`, with acquire/release events, wait
durations, and hold durations saved in `<lock>.log`. The node, allocation and
device identify the lock. To share with other cooperating searches on that
same GPU, export the same absolute `ADBENCH_GPU_LOCK` path to all of them.

Evaluation timeouts begin after acquiring the GPU. OpenEvolve's outer evaluator
watchdog is extended to the case limit in shared mode, while its isolated GPU
evaluation keeps its original timeout. The shared 6000-second case limit and
allocation deadline still include waiting time. An evaluator timeout or a
batch termination kills the evaluation process group before releasing its
lock. A failed case stops new launches; other active cases may finish. On an
allocation cutoff all active searches are stopped and their records preserved.

This changes the plan's GPU-per-search arrangement and is recorded in the
agent's `NOTES.md`. It affects search throughput only; formal unified evaluation
remains separate. Start with two workers and compare case wall time and lock
waiting before increasing concurrency.

Model names and planned case counts:

| Model | Cases |
|---|---:|
| qwen3_0_6b | 8 |
| qwen3_5_4b | 15 |
| llama_3_2_1b | 8 |
| qwen2_vl_2b | 6 |
| convnextv2_tiny | 8 |
| dit_s_2 | 3 |
| torchrec_dlrm | 2 |
| olmoe_1b_7b | 8 |

The case roster comes from the shared task packs and is checked against the
plan's counts (and `cases`/`case_ids` in the shared index when present). The
shared index and Slurm cannot be inspected from the local development machine;
preflight fails before submission if their runtime layout is inconsistent.

## 5. Inspect progress, then repeat the launcher

```bash
squeue -u "$USER"
python candidates/adbench_search.py status --agent openevolve --model qwen3_0_6b
# All agents, all models:
python candidates/adbench_search.py status
```

For a particular job, replace `JOBID` with its number:

```bash
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
# tail -n 60 adbench-search-JOBID.out
```

When the prior job is finished, repeat the same launcher command. It continues
with the next pending cases; it never resubmits itself automatically. It checks
for overlapping queued/running jobs, and a runtime shared lock also protects
against direct or concurrent sbatch submissions. The batch follows the plan's
start rule: at least twice the longest smoke search and 20 minutes remaining,
plus a 3-minute cleanup margin. A case that exceeds the remaining allocation
can still be interrupted. Its files are preserved and the job reports failure.

The outer search limit remains 6000 seconds. The wrapper reads Slurm's actual
EndTime, including startup time, and terminates the search process group if the
allocation margin is reached. GEAK also retains its 4800-second inner limit.

Saved kernels:

```text
/work/hdd/bcjw/wzhan/ADBench/<agent>/<model>/runs/<agent>/<case-with-__>/submission.py
```

`status` counts actual packaged kernels and search-feedback evaluator calls.
It does not imply formal acceptance; costs from `aborted/` are retained
separately and must also be reported. Wangjia runs unified formal evaluation.

## Interrupted formal searches

First confirm with `squeue`/`sacct` that no job still owns the run, and inspect
its log. Only an allocation/node interruption qualifies for the plan's one
rerun. Preserve the original directory and its adjacent `.log` under the
model's `aborted/`; record the reason and both attempts' costs in the agent's
`NOTES.md`. Then the normal launcher can see that case as pending again. Never
retry a finished failure, and never select the better of
multiple trials. A second interruption needs explicit coordination rather
than another automatic retry.
