#!/bin/bash
# Test and run the OpenAI-model agents (evograd, raw OpenEvolve, GEPA,
# KernelAgent) on one operator, end to end. GEAK is left out.
#
#   bash candidates/run_openai.sh [OP] [DTYPE] [BUDGET]
#
#   OP      default llama3_swiglu_mlp
#   DTYPE   default bfloat16
#   BUDGET  candidate programs per agent in the full run, default 30
#
# Stages, in order (STAGES picks a subset, e.g. STAGES="sanity smoke"):
#   setup   install GEPA and KernelAgent at the pinned commits (only when SETUP=1)
#   unit    CPU unit tests of the harness
#   task    build the task pack (kept if it already exists)
#   sanity  no LLM: the skeleton must fail, evograd's seed must pass and score
#   smoke   every agent with a tiny budget; checks the plumbing
#   full    every agent with BUDGET, one after another, then the comparison table
#
# Environment: OPENAI_API_KEY (required for smoke/full), ADBENCH_MODEL
# (default gpt-5.6-sol), ADBENCH_API_BASE (default https://api.openai.com/v1),
# CUDA_VISIBLE_DEVICES (default 0), TIME_LIMIT (seconds per agent in the full
# run, default 21600), OUT (default results/adbench_tools/<OP>).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

OP="${1:-llama3_swiglu_mlp}"
DTYPE="${2:-bfloat16}"
BUDGET="${3:-30}"
STAGES="${STAGES:-setup unit task sanity smoke full}"
AGENTS="evograd openevolve gepa kernelagent"
OUT="${OUT:-$REPO/results/adbench_tools/$OP}"
TASK="$OUT/task"
STAMP="$(date +%Y%m%d_%H%M%S)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ADBENCH_MODEL="${ADBENCH_MODEL:-gpt-5.6-sol}"
export ADBENCH_API_BASE="${ADBENCH_API_BASE:-https://api.openai.com/v1}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

KERNELAGENT_COMMIT=e0647170da36ef9b059ac0bd3d60103aa4ed378b
GEPA_COMMIT=8f287de942f41a6d1e0e7e5b2d576247c4bedee9

want() { [[ " $STAGES " == *" $1 "* ]]; }
banner() { echo; echo "=============== $* ==============="; }
die() { echo "FAILED: $*" >&2; exit 1; }

echo "op=$OP dtype=$DTYPE budget=$BUDGET model=$ADBENCH_MODEL gpu=$CUDA_VISIBLE_DEVICES"
echo "stages: $STAGES"
echo "output: $OUT"

if want setup && [ "${SETUP:-0}" = 1 ]; then
    banner "setup"
    pip install -e ".[gpu,llm]" || die "evograd install"
    if [ ! -d references/KernelAgent/.git ]; then
        git clone https://github.com/pytorch-labs/KernelAgent references/KernelAgent || die "clone KernelAgent"
    fi
    git -C references/KernelAgent checkout -q "$KERNELAGENT_COMMIT" || die "checkout KernelAgent"
    pip install -e references/KernelAgent || die "KernelAgent install"
    pip install "gepa @ git+https://github.com/gepa-ai/gepa@$GEPA_COMMIT" litellm || die "GEPA install"
fi

if want smoke || want full; then
    [ -n "${OPENAI_API_KEY:-}" ] || die "OPENAI_API_KEY is not set"
    python -c "import gepa" 2>/dev/null || die "gepa is not installed (rerun with SETUP=1)"
    python -c "import triton_kernel_agent" 2>/dev/null || die "KernelAgent is not installed (rerun with SETUP=1)"
    [ -d references/KernelAgent/examples/configs ] || die "references/KernelAgent checkout missing (rerun with SETUP=1)"
fi

if want unit; then
    banner "unit tests"
    python -m pytest tests/adbench_tools -q || die "unit tests"
fi

if want task; then
    banner "task pack"
    if [ -f "$TASK/task.json" ]; then
        echo "kept existing $TASK"
    else
        python -m evograd.adbench_tools task --op "$OP" --dtype "$DTYPE" --output-dir "$TASK" || die "task pack"
    fi
    python -c "import json,sys; s=json.load(open(sys.argv[1])); print('task', s['task_id']); print('dims', s['workload']['dims']); print('primitives', s['allowed_primitives'])" "$TASK/task.json"
fi
[ -f "$TASK/task.json" ] || die "no task pack at $TASK (include the task stage)"

if want sanity; then
    banner "sanity: skeleton must fail"
    S="$OUT/sanity_$STAMP"
    mkdir -p "$S"
    python -m evograd.adbench_tools evaluate --task "$TASK" --json "$S/skeleton.json" "$TASK/skeleton.py" >/dev/null
    rc=$?
    [ $rc -eq 1 ] || die "the skeleton should be rejected (exit $rc, see $S/skeleton.json)"
    echo "ok: skeleton rejected"

    banner "sanity: evograd seed must pass"
    DIMS=$(python -c "import json,sys; d=json.load(open(sys.argv[1]))['workload']['dims']; print(' '.join(f'{k}={v}' for k,v in d.items()))" "$TASK/task.json")
    evograd seed unified --op "$OP" --dtype "$DTYPE" --dims $DIMS --output-dir "$S/seed" || die "evograd seed unified"
    SEED="$S/seed/initial_program_autograd_pair.py"
    python -m evograd.adbench_tools evaluate --task "$TASK" --json "$S/seed_eval.json" "$SEED" >/dev/null \
        || die "the seed should pass the in-loop evaluator (see $S/seed_eval.json)"
    echo "ok: seed passes the in-loop evaluator"

    banner "sanity: final measurement of the seed"
    contract=()
    [ -f "$S/seed/unified_contract.json" ] && contract=(--contract "$S/seed/unified_contract.json")
    python -m evograd.adbench_tools score --task "$TASK" --run-dir "$S/score" "${contract[@]}" "$SEED" >/dev/null \
        || die "the seed should pass the final measurement (see $S/score/final.json)"
    python -c "import json,sys; r=json.load(open(sys.argv[1])); m=r['metrics']; print('ok: accepted', r['accepted'], 'combined_score', r['combined_score'], 'speedup', m.get('speedup'), 'full_step_speedup', m.get('full_step_speedup'))" "$S/score/final.json"
fi

summarize() {  # summarize RUN_ROOT AGENT...
    local root="$1"; shift
    printf '%-12s %-6s %-9s %-8s %-14s %s\n' agent evals accepted correct combined_score failure
    for a in "$@"; do
        local d="$root/$a"
        if [ ! -f "$d/final.json" ]; then
            printf '%-12s %s\n' "$a" "NO final.json (see $root/$a.log)"
            continue
        fi
        python - "$d" "$a" <<'EOF'
import json, sys
from pathlib import Path
d, a = Path(sys.argv[1]), sys.argv[2]
r = json.loads((d / "final.json").read_text())
ev = d / "evaluations.jsonl"
n = sum(1 for _ in ev.open()) if ev.is_file() else 0
f = r.get("failure") or {}
reason = f.get("stage", "") + (": " + str(f.get("error") or (f.get("artifacts") or {}).get("error_type", ""))
                               if f else "")
print(f"{a:<12} {n:<6} {str(r.get('accepted')):<9} {str(r.get('correct')):<8} "
      f"{str(r.get('combined_score')):<14} {reason[:80]}")
EOF
    done
}

if want smoke; then
    banner "smoke: tiny budget per agent"
    R="$OUT/smoke_$STAMP"
    mkdir -p "$R"
    for a in $AGENTS; do
        echo "--- $a -> $R/$a"
        case "$a" in
            kernelagent) extra=(--budget 8 --skip-optimization) ;;  # one round of 4 workers
            *) extra=(--budget 2) ;;
        esac
        timeout --signal=INT 3600 python "candidates/$a/run.py" --task "$TASK" --run-dir "$R/$a" \
            "${extra[@]}" 2>&1 | tee "$R/$a.log" | tail -n 5
    done
    banner "smoke summary"
    summarize "$R" $AGENTS
    echo "(at this budget only evograd is expected to be correct; every agent must have a final.json)"
fi

if want full; then
    banner "full run: budget $BUDGET"
    AGENTS="$AGENTS" TIME_LIMIT="${TIME_LIMIT:-21600}" OUT="$OUT" \
        bash "$HERE/run_all.sh" "$OP" "$DTYPE" "$BUDGET" "$OUT"
fi
