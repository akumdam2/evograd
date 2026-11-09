#!/bin/bash
# Run every agent on one model case and compare them.
#
#   bash candidates/run_all.sh MODEL CASE_ID CORPUS [BUDGET] [OUT]
#
#   MODEL    a verified model, e.g. llama_3_2_1b
#   CASE_ID  one of its observed cases, e.g. llama3_swiglu_mlp/swiglu_mlp
#   CORPUS   the model's payload corpus (its manifest.json)
#   BUDGET   candidate programs per agent, default 30
#   OUT      default results/adbench_tools/<MODEL>/<CASE_ID>
#
# Environment: OPENAI_API_KEY, ADBENCH_MODEL (default gpt-5.5),
# ADBENCH_API_BASE (default https://api.openai.com/v1), AGENTS (default
# "evograd openevolve gepa kernelagent codex claude_code geak"), CLAUDE_MODEL
# (Claude Code only), TIME_LIMIT (seconds per agent, default 21600).
#
# Agents run one after another on the same GPU (CUDA_VISIBLE_DEVICES), so no
# two share a device while timing. A failed agent does not stop the others.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
USAGE="usage: run_all.sh MODEL CASE_ID CORPUS [BUDGET] [OUT]"
MODEL="${1:?$USAGE}"
CASE_ID="${2:?$USAGE}"
CORPUS="${3:?$USAGE}"
BUDGET="${4:-${ADBENCH_BUDGET:-10}}"
OUT="${5:-$REPO/results/adbench_tools/$MODEL/${CASE_ID//\//__}}"
AGENTS="${AGENTS:-evograd openevolve gepa kernelagent codex claude_code geak}"
TIME_LIMIT="${TIME_LIMIT:-21600}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

TASK="$OUT/task"
if [ ! -f "$TASK/task.json" ]; then
    python -m evograd.adbench_tools task --model "$MODEL" --case-id "$CASE_ID" --corpus "$CORPUS" \
        --output-dir "$TASK" || exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/runs"
for agent in $AGENTS; do
    run="$OUT/runs/${agent}_${STAMP}"
    echo "=== $agent -> $run"
    extra=()
    case "$agent" in
        claude_code) [ -n "${CLAUDE_MODEL:-}" ] && extra+=(--claude-model "$CLAUDE_MODEL") ;;
    esac
    timeout --signal=INT "$TIME_LIMIT" \
        python "$HERE/$agent/run.py" --task "$TASK" --run-dir "$run" --budget "$BUDGET" "${extra[@]}" \
        2>&1 | tee "$OUT/runs/${agent}_${STAMP}.log"
    if [ ! -f "$run/final.json" ] && [ -f "$run/run.json" ]; then
        best="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('best_program') or '')" "$run/run.json")"
        [ -n "$best" ] && python -m evograd.adbench_tools score --task "$TASK" --run-dir "$run" "$best"
    fi
done
python -m evograd.adbench_tools compare "$OUT"/runs/*_"$STAMP"/final.json | tee "$OUT/comparison_$STAMP.md"
