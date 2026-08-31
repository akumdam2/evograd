#!/usr/bin/env bash
# Evolve one benchmark's A/B/C/D seeds several times over and record every
# iteration, so the four pipelines can be compared as starting points.
#
#   OP=layernorm bash scripts/seed_ablation.sh
#   OP=rmsnorm ITERATIONS=25 TRIALS=5 bash scripts/seed_ablation.sh
#   OP=softmax PIPELINES="A D" TRIALS=1 GPU=1 bash scripts/seed_ablation.sh
#
# Evolution is stochastic — the LLM samples at temperature 0.5 and no seed is
# pinned — so a single run of one seed says nothing about that seed. TRIALS
# independent runs per pipeline is what makes the comparison an ablation rather
# than an anecdote.
#
# What it does differently from a loop around `evograd evolve`:
#
#   * one baseline timing cache for the whole sweep. Every speedup is a ratio
#     against a re-measured baseline, and evolve defaults that cache per
#     output-dir, so N runs would each re-time the denominator and baseline
#     jitter would show up as a difference between seeds;
#   * dtype is derived from the declaration here rather than read from
#     $EG/manifest.tsv, which holds only whatever the last seed_matrix run was
#     scoped to and would silently disagree with this op;
#   * finished trials are skipped, so an interrupted sweep resumes;
#   * trials run strictly serially. These are timing measurements: a concurrent
#     run on the same GPU contaminates them.

set -uo pipefail

OP=${OP:-}
[ -n "$OP" ] || { echo "usage: OP=<benchmark> bash scripts/seed_ablation.sh" >&2; exit 2; }

EG=${EG:-$HOME/evograd_runs}          # where seed_matrix.sh put the seeds
ABL=${ABL:-$EG/ablation}              # where this experiment writes
PYTHON=${PYTHON:-python}
PIPELINES=${PIPELINES:-"A B C D"}
ITERATIONS=${ITERATIONS:-10}
TRIALS=${TRIALS:-3}
SCORING=${SCORING:-speed_memory}
PRIMARY_MODEL=${PRIMARY_MODEL:-gpt-5.5}
SECONDARY_MODEL=${SECONDARY_MODEL:-$PRIMARY_MODEL}
API_BASE=${API_BASE:-https://api.openai.com/v1}
TIMEOUT=${TIMEOUT:-7200}
FORCE=${FORCE:-0}                     # 1 re-runs trials that already finished
GPU=${GPU:-}                          # pin one device; unset means inherit

[ -n "${OPENAI_API_KEY:-}" ] || echo "warning: OPENAI_API_KEY is unset" >&2
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"

ROOT="$ABL/$OP"
mkdir -p "$ROOT"

# One denominator for every pipeline and every trial of this op.
export EVOGRAD_BASELINE_TIMING_CACHE_PATH="$ROOT/baseline_timings.json"

STATUS="$ROOT/status.tsv"
: > "$STATUS"

TIMEOUT_BIN=$(command -v timeout || true)

# ── dtype ─────────────────────────────────────────────────────────────────────
# Same rule as scripts/seed_matrix.sh: gate and measure on a dtype that both the
# correctness and benchmark suites declare. "-" means pass no --dtype at all, so
# evolve gates on every declared correctness dtype and measures the full suite.
read -r dtype bench_dtype <<< "$("$PYTHON" - "$OP" <<'PY'
import sys

from evograd.ops import get_op

PREFERENCE = ("float16", "bfloat16", "float32")


def pick(available):
    for dtype in PREFERENCE:
        if dtype in available:
            return dtype
    return sorted(available)[0] if available else None


op = get_op(sys.argv[1])
correctness = {w.dtype for w in op.correctness}
benchmark = {w.dtype for w in op.benchmark}
shared = correctness & benchmark
chosen = pick(shared) if shared else pick(correctness)
print(chosen, chosen if shared else "-")
PY
)" || { echo "could not resolve a dtype for $OP" >&2; exit 1; }

echo "=== $OP  dtype=$dtype  bench=$bench_dtype  pipelines='$PIPELINES'"
echo "    ${ITERATIONS} iteration(s) × ${TRIALS} trial(s) per pipeline"

# ── helpers ───────────────────────────────────────────────────────────────────

detail() {
    grep -aoE '[A-Z][A-Za-z_.]*(Error|Exception): [^"\\]*' "$1" 2>/dev/null \
        | tail -1 | cut -c1-200 || true
}

run_step() {  # run_step <pipeline> <trial> <log> <cmd...>
    local pipeline=$1 trial=$2 log=$3
    shift 3
    local start=$SECONDS rc=0
    if [ -n "$TIMEOUT_BIN" ]; then
        "$TIMEOUT_BIN" "$TIMEOUT" "$@" > "$log" 2>&1 < /dev/null
    else
        "$@" > "$log" 2>&1 < /dev/null
    fi
    rc=$?
    local elapsed=$((SECONDS - start)) why=""
    if [ "$rc" -ne 0 ]; then
        [ "$rc" -eq 124 ] && why="timed out after ${TIMEOUT}s" || why=$(detail "$log")
        [ -z "$why" ] && why="exit $rc (see $log)"
        printf '  %-3s trial %-3s FAILED  %5ds  %s\n' "$pipeline" "$trial" "$elapsed" "$why"
    else
        printf '  %-3s trial %-3s ok      %5ds\n' "$pipeline" "$trial" "$elapsed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$OP" "$pipeline" "$trial" "$rc" "$elapsed" "$why" >> "$STATUS"
    return $rc
}

# ── evolve ────────────────────────────────────────────────────────────────────

for pipeline in $PIPELINES; do
    dir="$EG/$OP/$pipeline"
    seed="$dir/initial_program_autograd_pair.py"
    # Seeds produced before seed_matrix.sh learned to promote A/C winners still
    # live under best/; accept either rather than reporting a missing seed.
    [ -f "$seed" ] || seed="$dir/best/initial_program_autograd_pair.py"
    if [ ! -f "$seed" ]; then
        printf '  %-3s skipped — no seed under %s\n' "$pipeline" "$dir"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$OP" "$pipeline" "-" 1 0 "no seed" >> "$STATUS"
        continue
    fi

    for trial in $(seq 1 "$TRIALS"); do
        out="$ROOT/$pipeline/trial_$trial"
        if [ -f "$out/programs/index.jsonl" ] && [ "$FORCE" != "1" ]; then
            printf '  %-3s trial %-3s cached\n' "$pipeline" "$trial"
            continue
        fi
        [ "$FORCE" = "1" ] && rm -rf "$out"
        mkdir -p "$out"

        evolve_args=(--baseline auto --scoring "$SCORING" --iterations "$ITERATIONS" --save-programs)
        [ "$bench_dtype" != "-" ] && evolve_args+=(--dtype "$dtype")

        run_step "$pipeline" "$trial" "$ROOT/evolve_${pipeline}_${trial}.log" \
            evograd evolve --op "$OP" --seed "$seed" \
            "${evolve_args[@]}" \
            --primary-model "$PRIMARY_MODEL" --secondary-model "$SECONDARY_MODEL" \
            --api-base "$API_BASE" --output-dir "$out"
    done
done

# ── collect ───────────────────────────────────────────────────────────────────

"$PYTHON" "$(dirname "$0")/collect_evolution.py" \
    --root "$ABL" --op "$OP" \
    --out "$ROOT/curves.csv" --summary "$ROOT/summary.json"

echo
echo "curves:  $ROOT/curves.csv"
echo "summary: $ROOT/summary.json"
echo "plot:    $PYTHON scripts/plot_evolution.py --csv $ROOT/curves.csv --out $ROOT"
