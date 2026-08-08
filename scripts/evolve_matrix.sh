#!/usr/bin/env bash
# Evolve every seed that survived scripts/seed_matrix.sh, then re-verify and
# re-bench the winner against the same baseline.
#
#   OPENAI_API_KEY=... bash scripts/evolve_matrix.sh
#   OPS="layernorm rmsnorm" ITERATIONS=200 bash scripts/evolve_matrix.sh
#   PIPELINES=D bash scripts/evolve_matrix.sh
#
# Only ops whose seed benched clean are evolved — evolving a seed that never
# produced a number wastes LLM calls. Dtype comes from $EG/manifest.tsv, so the
# evolved program is gated and measured on exactly what the seed was.

set -uo pipefail

EG=${EG:-$HOME/evograd_runs}
PYTHON=${PYTHON:-python}
PIPELINES=${PIPELINES:-"D B"}
ITERATIONS=${ITERATIONS:-100}
SCORING=${SCORING:-speed_memory}
PRIMARY_MODEL=${PRIMARY_MODEL:-gpt-5.5}
SECONDARY_MODEL=${SECONDARY_MODEL:-$PRIMARY_MODEL}
API_BASE=${API_BASE:-https://api.openai.com/v1}
TIMEOUT=${TIMEOUT:-7200}
SAVE_PROGRAMS=${SAVE_PROGRAMS:-1}   # keep every evaluated candidate under evolve_<P>/programs
OPS=${OPS:-}

export EVOGRAD_BASELINE_TIMING_CACHE_PATH="${EVOGRAD_BASELINE_TIMING_CACHE_PATH:-$EG/baseline_timings.json}"

MANIFEST="$EG/manifest.tsv"
STATUS="$EG/evolve_status.tsv"
[ -f "$MANIFEST" ] || { echo "no $MANIFEST — run scripts/seed_matrix.sh first" >&2; exit 1; }
[ -n "${OPENAI_API_KEY:-}" ] || echo "warning: OPENAI_API_KEY is unset" >&2
: > "$STATUS"

TIMEOUT_BIN=$(command -v timeout || true)

detail() {
    grep -aoE '[A-Z][A-Za-z_.]*(Error|Exception): [^"\\]*' "$1" 2>/dev/null \
        | tail -1 | cut -c1-200 || true
}

run_step() {  # run_step <op> <pipeline> <stage> <log> <cmd...>
    local op=$1 pipeline=$2 stage=$3 log=$4
    shift 4
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
        printf '  %-6s %-7s FAILED  %5ds  %s\n' "$pipeline" "$stage" "$elapsed" "$why"
    else
        printf '  %-6s %-7s ok      %5ds\n' "$pipeline" "$stage" "$elapsed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$op" "$pipeline" "$stage" "$rc" "$elapsed" "$why" >> "$STATUS"
    return $rc
}

benched_ok() {  # a seed is worth evolving only if its bench produced numbers
    "$PYTHON" -c "
import json, sys
try:
    print('yes' if json.load(open(sys.argv[1])).get('ok') else 'no')
except Exception:
    print('no')" "$1" 2>/dev/null
}

while IFS=$'\t' read -r op dtype bench_dtype note; do
    [ -n "$op" ] || continue
    if [ -n "$OPS" ]; then
        case " $OPS " in *" $op "*) ;; *) continue ;; esac
    fi
    echo "=== $op ($dtype)"

    for pipeline in $PIPELINES; do
        seed="$EG/$op/$pipeline/initial_program_autograd_pair.py"
        report="$EG/$op/bench_$pipeline.json"
        best="$EG/$op/evolved_$pipeline.py"

        if [ ! -f "$seed" ] || [ "$(benched_ok "$report")" != "yes" ]; then
            printf '  %-6s %-7s skipped (seed never benched clean)\n' "$pipeline" "evolve"
            continue
        fi

        # --dtype gates correctness on this dtype and measures on it too. When
        # the benchmark suite has no case for it (bench_dtype "-"), evolve a
        # generalist instead: every declared correctness dtype, whole suite.
        evolve_args=(--baseline auto --scoring "$SCORING" --iterations "$ITERATIONS")
        [ "$bench_dtype" != "-" ] && evolve_args+=(--dtype "$dtype")
        [ "$SAVE_PROGRAMS" = "1" ] && evolve_args+=(--save-programs)

        run_step "$op" "$pipeline" "evolve" "$EG/$op/evolve_$pipeline.log" \
            evograd evolve --op "$op" --seed "$seed" \
            "${evolve_args[@]}" \
            --primary-model "$PRIMARY_MODEL" --secondary-model "$SECONDARY_MODEL" \
            --api-base "$API_BASE" \
            --output-dir "$EG/$op/evolve_$pipeline" --save-best-to "$best" \
            || continue

        run_step "$op" "$pipeline" "verify" "$EG/$op/verify_evolved_$pipeline.json" \
            evograd verify --op "$op" --dtype "$dtype" "$best" || continue

        bench_args=(--baseline auto)
        [ "$bench_dtype" != "-" ] && bench_args+=(--dtype "$bench_dtype")
        run_step "$op" "$pipeline" "bench" "$EG/$op/bench_evolved_$pipeline.log" \
            evograd bench --op "$op" "${bench_args[@]}" \
            --candidate "$best" --out "$EG/$op/bench_evolved_$pipeline.json"
    done
done < "$MANIFEST"

# ── seed vs evolved ───────────────────────────────────────────────────────────

"$PYTHON" - "$EG" "$STATUS" "$PIPELINES" <<'PY'
import json
import pathlib
import sys

root, status_path, pipelines = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split()

status = {}
for line in status_path.read_text(encoding="utf-8").splitlines():
    op, pipeline, stage, rc, seconds, why = (line.split("\t") + [""] * 6)[:6]
    status[(op, pipeline, stage)] = (int(rc), why)


def backward_ms(path):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not report.get("ok", True):
        return None
    return report["aggregate"]["backward_from_saved_ms"]


header = f"{'op':<20}{'dtype':<10}" + "".join(
    f"{p + ' seed':>12}{p + ' evolved':>14}{'gain':>8}" for p in pipelines
)
print()
print(header)
print("-" * len(header))

failures = []
for line in (root / "manifest.tsv").read_text(encoding="utf-8").splitlines():
    op, seed_dtype, bench_dtype, _note = (line.split("\t") + [""] * 4)[:4]
    if not op:
        continue
    row = f"{op:<20}{('suite' if bench_dtype in ('', '-') else bench_dtype):<10}"
    printable = False
    for p in pipelines:
        seed_ms = backward_ms(root / op / f"bench_{p}.json")
        evolved_ms = backward_ms(root / op / f"bench_evolved_{p}.json")
        printable = printable or seed_ms is not None or evolved_ms is not None
        row += f"{seed_ms:>12.4f}" if seed_ms else f"{'—':>12}"
        row += f"{evolved_ms:>14.4f}" if evolved_ms else f"{'—':>14}"
        row += f"{seed_ms / evolved_ms:>8.2f}" if seed_ms and evolved_ms else f"{'—':>8}"
        for stage in ("evolve", "verify", "bench"):
            rc, why = status.get((op, p, stage), (0, ""))
            if rc:
                failures.append(f"  {op:<20} {p:<3} {stage:<7} {why}")
                break
    if printable:
        print(row)

if failures:
    print(f"\n{len(failures)} failure(s):")
    print("\n".join(failures))
print(f"\nevolved programs: {root}/<op>/evolved_<pipeline>.py")
PY
