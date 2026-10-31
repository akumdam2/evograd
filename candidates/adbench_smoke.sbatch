#!/usr/bin/env bash
# One smoke search or evaluation per allocation; never touches formal outputs.
#SBATCH --job-name=adbench-smoke
#SBATCH --account=betw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=adbench-smoke-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the evograd checkout}"
source candidates/adbench_cluster_env.sh
SMOKE="/work/hdd/bcjw/wzhan/ADBench/smoke/$AGENT"
CORPUS=$(python -c 'import json; print(json.load(open("/work/hdd/bcjw/wzhan/ADBench/cases.json"))["qwen3_0_6b"]["corpus"])')
case "$AGENT" in
  openevolve) flags=(--model gpt-5.5) ;;
  codex) flags=(--model gpt-5.5 --effort high --time-limit 4800 --eval-timeout 900) ;;
  gepa) flags=(--model gpt-5.5 --eval-timeout 900) ;;
  geak) flags=(--backend codex --model gpt-5.5 --effort high --geak-budget 2 --time-limit 4800) ;;
esac
case "${1:-}" in
  search)
    case "${2:-}" in rmsnorm/rmsnorm|qwen3_swiglu_mlp/swiglu_mlp) ;; *) echo 'Specify one of the two smoke cases.' >&2; exit 2 ;; esac
    python candidates/bench.py run --root "$SMOKE" --model qwen3_0_6b \
      --agent "$AGENT" --case-id "$2" --budget 10 --time-limit 6000 -- "${flags[@]}"
    ;;
  evaluate)
    # An exit 3 is a resumable evaluation deadline; rerun this job unchanged.
    python candidates/bench.py evaluate --root "$SMOKE" --model qwen3_0_6b \
      --agent "$AGENT" --case-id rmsnorm/rmsnorm qwen3_swiglu_mlp/swiglu_mlp \
      --corpus "$CORPUS" --deadline-min 100
    python candidates/adbench_search.py check-smoke --agent "$AGENT"
    ;;
  *) echo 'Usage: sbatch --export=ALL,AGENT=... candidates/adbench_smoke.sbatch search CASE_ID | evaluate' >&2; exit 2 ;;
esac
