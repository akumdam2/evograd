#!/usr/bin/env bash
# Login-node launcher. AGENT [MODEL] [--skip-smoke] [--workers N] [--partition PARTITION]
set -euo pipefail
if (( $# < 1 )); then
  echo "Usage: bash candidates/submit_adbench_search.sh {openevolve|gepa|geak|codex} [MODEL] [--skip-smoke] [--workers N] [--partition {ghx4|ghx4-interactive}]" >&2
  exit 2
fi
agent=$1
shift
model=
skip_smoke=0
partition=ghx4
partition_set=0
workers=1
workers_set=0
case "$agent" in openevolve|gepa|geak|codex) ;; *) echo "Unknown agent: $agent" >&2; exit 2 ;; esac
while (( $# )); do
  arg=$1
  case "$arg" in
    --skip-smoke)
      if (( skip_smoke )); then echo "Duplicate --skip-smoke" >&2; exit 2; fi
      skip_smoke=1 ;;
    --partition)
      if (( partition_set )); then echo "Duplicate --partition" >&2; exit 2; fi
      if (( $# < 2 )); then echo "--partition requires a value" >&2; exit 2; fi
      partition=$2
      partition_set=1
      shift ;;
    --workers)
      if (( workers_set )); then echo "Duplicate --workers" >&2; exit 2; fi
      if (( $# < 2 )); then echo "--workers requires a positive integer" >&2; exit 2; fi
      workers=$2
      workers_set=1
      shift ;;
    --*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *)
      if [[ -n "$model" ]]; then echo "Specify at most one model" >&2; exit 2; fi
      model=$arg ;;
  esac
  shift
done
if [[ ! "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "--workers must be a positive integer" >&2; exit 2
fi
if [[ "$workers" != 1 && "$agent" != gepa && "$agent" != openevolve ]]; then
  echo "Shared-GPU workers are supported only for GEPA and OpenEvolve" >&2; exit 2
fi
case "$partition" in
  ghx4) batch_script=candidates/adbench_search.sbatch; walltime=2-hour ;;
  ghx4-interactive) batch_script=candidates/adbench_search_interactive.sbatch; walltime=1-hour ;;
  *) echo "Unknown partition: $partition" >&2; exit 2 ;;
esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
python="${ADBENCH_ENV:-$HOME/venvs/adbench-20261028}/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Python environment missing: $python" >&2
  exit 2
fi
args=(--agent "$agent")
batch_args=()
args+=(--workers "$workers")
batch_args+=(--workers "$workers")
if [[ -n "$model" ]]; then
  args+=(--model "$model")
  batch_args+=(--model "$model")
fi
if (( skip_smoke )); then
  args+=(--skip-smoke)
  batch_args+=(--skip-smoke)
fi

# Serialize checks/submission across login shells. The GPU runner has a second
# shared lock for the lifetime of its search batch (including direct sbatch).
mkdir -p "$HOME/.cache/adbench"
exec 9>"$HOME/.cache/adbench/submit-${agent}.lock"
if ! flock -n 9; then
  echo "Another launcher is checking/submitting $agent. No duplicate submitted."
  exit 0
fi
jobs=$(squeue -h -u "$USER" -o '%i|%j|%T')
existing=$(printf '%s\n' "$jobs" | awk -F '|' -v prefix="adb-${agent}-" \
  'index($2,prefix)==1 || $2=="adbench-search" {print}')
if [[ -n "$existing" ]]; then
  echo "An overlapping batch is already queued/running (job|name|state):"
  echo "$existing"
  echo "No duplicate submitted. Check progress with:"
  echo "$python candidates/adbench_search.py status --agent $agent"
  exit 0
fi

echo "Checking shared searches before requesting a GPU..."
if "$python" candidates/adbench_search.py plan "${args[@]}"; then
  :
else
  rc=$?
  case "$rc" in
    3) echo "No batch job needed."; exit 0 ;;
    4) echo "Inspect the unfinished runs above; no batch job submitted." >&2; exit 4 ;;
    *) echo "Preflight failed (exit $rc); no batch job submitted." >&2; exit "$rc" ;;
  esac
fi
if [[ ! -r "$HOME/.adbench_keys.txt" ]]; then
  echo "Missing readable $HOME/.adbench_keys.txt; no job submitted." >&2
  exit 2
fi
if [[ "$agent" == geak || "$agent" == codex ]]; then
  test -x "$HOME/.local/bin/codex" || command -v codex >/dev/null
fi
if [[ "$agent" == geak ]]; then
  command -v node >/dev/null
  test -f candidates/geak/.work/GEAK/interface/runtime/engine/run_workflow.mjs
  test -f candidates/geak/.work/GEAK/kernel_workflow/kernel_workflow.js
fi
job_name="adb-${agent}-${model:-all}"
echo "Submitting one $walltime GPU job on $partition for $agent / ${model:-all models}, workers=$workers..."
sbatch --chdir="$repo" --job-name="$job_name" --export="ALL,AGENT=$agent" \
  "$batch_script" "${batch_args[@]}"
echo "Progress: $python candidates/adbench_search.py status --agent $agent"
echo "After this job finishes, repeat this launcher command to process remaining cases."
