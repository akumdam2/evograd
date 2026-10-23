#!/bin/bash
# Run every agent on one operator configuration and compare them.
#
#   bash candidates/run_all.sh OP [DTYPE] [BUDGET] [OUT]
#
#   OP      an evograd task, e.g. llama3_swiglu_mlp (`evograd ops` lists them)
#   DTYPE   default bfloat16; the task is the op's first timed case of DTYPE
#   BUDGET  candidate programs per agent, default 30
#   OUT     default results/adbench_tools/<OP>
#
# Environment: OPENAI_API_KEY, ADBENCH_MODEL (default gpt-5.6-sol),
# ADBENCH_API_BASE (default https://api.openai.com/v1), AGENTS (default
# "evograd openevolve gepa kernelagent geak"), CLAUDE_MODEL (GEAK only),
# TIME_LIMIT (seconds per agent, default 21600).
#
# Agents run one after another on the same GPU (CUDA_VISIBLE_DEVICES), so no
# two share a device while timing. A failed agent does not stop the others.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OP="${1:?usage: run_all.sh OP [DTYPE] [BUDGET] [OUT]}"
DTYPE="${2:-bfloat16}"
BUDGET="${3:-30}"
OUT="${4:-$REPO/results/adbench_tools/$OP}"
AGENTS="${AGENTS:-evograd openevolve gepa kernelagent geak}"
TIME_LIMIT="${TIME_LIMIT:-21600}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

TASK="$OUT/task"
if [ ! -f "$TASK/task.json" ]; then
    python -m evograd.adbench_tools task --op "$OP" --dtype "$DTYPE" --output-dir "$TASK" || exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/runs"
for agent in $AGENTS; do
    run="$OUT/runs/${agent}_${STAMP}"
    echo "=== $agent -> $run"
    extra=()
    [ "$agent" = geak ] && [ -n "${CLAUDE_MODEL:-}" ] && extra+=(--claude-model "$CLAUDE_MODEL")
    timeout --signal=INT "$TIME_LIMIT" \
        python "$HERE/$agent/run.py" --task "$TASK" --run-dir "$run" --budget "$BUDGET" "${extra[@]}" \
        2>&1 | tee "$OUT/runs/${agent}_${STAMP}.log"
    if [ ! -f "$run/final.json" ] && [ -f "$run/run.json" ]; then
        best="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('best_program') or '')" "$run/run.json")"
        [ -n "$best" ] && python -m evograd.adbench_tools score --task "$TASK" --run-dir "$run" "$best"
    fi
done
python -m evograd.adbench_tools compare "$OUT"/runs/*_"$STAMP"/final.json | tee "$OUT/comparison_$STAMP.md"
