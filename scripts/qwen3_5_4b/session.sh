#!/usr/bin/env bash
# One resumable GPU session for the Qwen3.5-4B baseline driver.
#
#   bash scripts/qwen3_5_4b/session.sh <run_id> [driver args...]
#
# Run inside a GPU allocation (1 GPU, 8 CPUs, 32 GB host memory). Steps that already have a DONE marker in
# results/experiments/qwen3_5_4b_cases/<run_id>/ are skipped, so the same command
# resumes on a new node. Extra arguments go to driver.py (e.g. --session 1).
set -euo pipefail

RUN_ID=${1:?usage: bash scripts/qwen3_5_4b/session.sh <run_id> [driver args...]}
shift
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="$REPO/results/experiments/qwen3_5_4b_cases/$RUN_ID"

cd "$REPO"
# Import this checkout, whatever PYTHONPATH the environment set.
export PYTHONPATH="$REPO/src"
# FLA is a sidecar: no process may see it unless the harness grants it.
unset EVOGRAD_SIDECARS
python -c "
import sys, evograd
path = evograd.__file__
print('evograd:', path)
sys.exit(0 if path.startswith('$REPO/src/') else 1)
"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$OUT/session.log"
python scripts/qwen3_5_4b/driver.py --run-dir "$OUT" "$@" 2>&1 | tee -a "$OUT/session.log"
