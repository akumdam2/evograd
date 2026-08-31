#!/usr/bin/env bash
# Seed → verify → bench every declared operator through one or more pipelines.
#
#   bash scripts/seed_matrix.sh                      # all ops, pipelines D and B
#   bash scripts/seed_matrix.sh layernorm rmsnorm    # just these two
#   PIPELINES=D RESEED=1 bash scripts/seed_matrix.sh
#   PIPELINES="A B C D" OPENAI_API_KEY=... bash scripts/seed_matrix.sh
#
# Operators come from the argument list, or from $OPS when no arguments are
# given. Prefer the arguments: `OPS=...` on its own line is a shell variable,
# not an environment variable, so it never reaches this script -- and the run
# silently widens to every declared operator instead of failing.
#
# What it does differently from the obvious loop:
#
#   * dtype is chosen from the intersection of the declaration's correctness and
#     benchmark suites, so ops whose benchmark suite is bfloat16-only (dyt, jsd,
#     tvd, poly_norm, relu_squared, fused_add_rms_norm) are not benched with a
#     --dtype their suite has no cases for;
#   * every step keeps its log and the failure reason is printed inline instead
#     of vanishing into /dev/null;
#   * `verify` runs between seed and bench, because bench only times — it never
#     compares against the oracle, so "bench ok" says nothing about correctness;
#   * baseline timings are cached across pipelines, so the D and B runs of one op
#     do not re-time the same baseline.

set -uo pipefail

EG=${EG:-$HOME/evograd_runs}
PYTHON=${PYTHON:-python}
PIPELINES=${PIPELINES:-"D B"}
TIMEOUT=${TIMEOUT:-1800}
RESEED=${RESEED:-0}
AUTOTUNE=${AUTOTUNE:-0}   # AUTOTUNE=1 lets Pipeline D sweep launch configs
OPS=${OPS:-}
[ "$#" -gt 0 ] && OPS="$*"
# Pipelines A and C are LLM-driven; B and D are not.
MODEL=${MODEL:-gpt-5.5}
API_BASE=${API_BASE:-https://api.openai.com/v1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}

export EVOGRAD_BASELINE_TIMING_CACHE_PATH="${EVOGRAD_BASELINE_TIMING_CACHE_PATH:-$EG/baseline_timings.json}"

MANIFEST="$EG/manifest.tsv"
STATUS="$EG/status.tsv"
mkdir -p "$EG"
: > "$STATUS"

TIMEOUT_BIN=$(command -v timeout || true)

# ── dtype selection ───────────────────────────────────────────────────────────
# One interpreter start for the whole matrix instead of one per op (importing
# torch costs seconds). Emits: op <TAB> seed dtype <TAB> bench dtype <TAB> note
# A bench dtype of "-" means "pass no --dtype at all", so bench uses whatever
# the declaration's benchmark suite holds. It is a sentinel rather than an empty
# field because tab is IFS whitespace: `read` would collapse the empty column
# and shift every later field left.
"$PYTHON" - "$MANIFEST" ${OPS:+$OPS} <<'PY' || { echo "manifest step failed" >&2; exit 1; }
import sys

from evograd.ops import OPS, get_op

PREFERENCE = ("float16", "bfloat16", "float32")

manifest, names = sys.argv[1], sys.argv[2:] or sorted(OPS)


def pick(available):
    for dtype in PREFERENCE:
        if dtype in available:
            return dtype
    return sorted(available)[0] if available else None


rows = []
for name in names:
    op = get_op(name)
    correctness = {w.dtype for w in op.correctness}
    benchmark = {w.dtype for w in op.benchmark}
    shared = correctness & benchmark
    if shared:
        dtype = pick(shared)
        rows.append((name, dtype, dtype, ""))
    else:
        # No dtype is both verifiable and benchable. Seed/verify on the
        # correctness pick and bench the whole declared suite: passing a
        # --dtype the suite has no cases for is a setup error, not a run.
        rows.append(
            (
                name,
                pick(correctness),
                "-",
                f"correctness {sorted(correctness)} vs benchmark {sorted(benchmark)}"
                "; benching the full suite",
            )
        )

with open(manifest, "w", encoding="utf-8") as handle:
    for name, seed_dtype, bench_dtype, note in rows:
        handle.write(f"{name}\t{seed_dtype}\t{bench_dtype}\t{note}\n")
        print(f"{name:<20} {seed_dtype:<10} {note}")
PY

# ── helpers ───────────────────────────────────────────────────────────────────

detail() {  # last "SomeError: message" in a log, for one-line failure reporting
    # Unanchored so it also finds the exception inside a JSON-encoded traceback
    # (verify writes its report to stdout); [^"\] stops at the JSON escape.
    local log=$1
    grep -aoE '[A-Z][A-Za-z_.]*(Error|Exception): [^"\\]*' "$log" 2>/dev/null \
        | tail -1 | cut -c1-200 || true
}

record() {  # op pipeline stage rc seconds detail
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$STATUS"
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
    local elapsed=$((SECONDS - start))
    local why=""
    if [ "$rc" -ne 0 ]; then
        [ "$rc" -eq 124 ] && why="timed out after ${TIMEOUT}s" || why=$(detail "$log")
        [ -z "$why" ] && why="exit $rc (see $log)"
        printf '  %-6s %-6s FAILED  %3ds  %s\n' "$pipeline" "$stage" "$elapsed" "$why"
    else
        printf '  %-6s %-6s ok      %3ds\n' "$pipeline" "$stage" "$elapsed"
    fi
    record "$op" "$pipeline" "$stage" "$rc" "$elapsed" "$why"
    return $rc
}

# ── seed → verify → bench ─────────────────────────────────────────────────────

while IFS=$'\t' read -r op dtype bench_dtype note; do
    [ -n "$op" ] || continue
    echo "=== $op ($dtype)${note:+  [$note]}"
    mkdir -p "$EG/$op"

    for pipeline in $PIPELINES; do
        lower=$(echo "$pipeline" | tr '[:upper:]' '[:lower:]')
        dir="$EG/$op/$pipeline"
        seed="$dir/initial_program_autograd_pair.py"

        # A and C write their winner to <output-dir>/best/; B and D write it at
        # the top level. Normalize so the cache check and every later stage see
        # one path per pipeline.
        promote() {
            [ -f "$seed" ] || [ ! -f "$dir/best/initial_program_autograd_pair.py" ] \
                || cp "$dir/best/initial_program_autograd_pair.py" "$seed"
        }
        promote

        if [ -f "$seed" ] && [ "$RESEED" != "1" ]; then
            printf '  %-6s %-6s cached\n' "$pipeline" "seed"
        else
            case "$lower" in
            a|c)
                # A and C take no --dtype: they verify on every declared
                # correctness dtype and need an LLM endpoint instead.
                seed_args=(--model "$MODEL" --api-base "$API_BASE"
                           --max-attempts "$MAX_ATTEMPTS")
                ;;
            *)
                seed_args=(--dtype "$dtype")
                # --autotune is Pipeline D's default; --no-autotune pins each kernel
                # to the config capture chose, which makes the seed deterministic.
                [ "$lower" = "d" ] && [ "$AUTOTUNE" != "1" ] && seed_args+=(--no-autotune)
                ;;
            esac
            run_step "$op" "$pipeline" "seed" "$EG/$op/seed_$pipeline.log" \
                evograd seed "$lower" --op "$op" "${seed_args[@]}" --output-dir "$dir" \
                || { promote; [ -f "$seed" ] || continue; }
            promote
        fi

        [ -f "$seed" ] || { record "$op" "$pipeline" "seed" 1 0 "no seed emitted"; continue; }

        # Correctness gate — bench never checks it.
        run_step "$op" "$pipeline" "verify" "$EG/$op/verify_$pipeline.json" \
            evograd verify --op "$op" --dtype "$dtype" "$seed" || continue

        bench_args=(--baseline auto)
        [ "$bench_dtype" != "-" ] && bench_args+=(--dtype "$bench_dtype")
        run_step "$op" "$pipeline" "bench" "$EG/$op/bench_$pipeline.log" \
            evograd bench --op "$op" "${bench_args[@]}" \
            --candidate "$seed" --out "$EG/$op/bench_$pipeline.json"
    done
done < "$MANIFEST"

# ── summary ───────────────────────────────────────────────────────────────────

"$PYTHON" - "$EG" "$STATUS" "$PIPELINES" <<'PY'
import json
import pathlib
import sys

root, status_path, pipelines = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3].split()

status = {}
for line in status_path.read_text(encoding="utf-8").splitlines():
    op, pipeline, stage, rc, seconds, why = (line.split("\t") + [""] * 6)[:6]
    status[(op, pipeline, stage)] = (int(rc), why)


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def bench_ms(report):
    # A report can be present but failed; ["ok"] is the verdict (older reports
    # predate the field and only exist when the whole run succeeded).
    if report is None or not report.get("ok", True):
        return None
    return report["aggregate"]["backward_from_saved_ms"]


primary, secondary = (pipelines + ["", ""])[:2]
header = (
    f"{'op':<20}{'dtype':<9}"
    + "".join(f"{p + ' bwd ms':>12}{p + ' saved':>12}" for p in pipelines)
    + f"{primary + '/' + secondary:>10}{'baseline':>18}"
)
print()
print(header)
print("-" * len(header))

failures = []
for line in (root / "manifest.tsv").read_text(encoding="utf-8").splitlines():
    op, seed_dtype, bench_dtype, _note = (line.split("\t") + [""] * 4)[:4]
    if not op:
        continue
    # "suite" = no --dtype was passed, so every declared benchmark dtype ran
    dtype = "suite" if bench_dtype in ("", "-") else bench_dtype
    reports = {p: load(root / op / f"bench_{p}.json") for p in pipelines}
    times = {p: bench_ms(r) for p, r in reports.items()}

    row = f"{op:<20}{dtype:<9}"
    for p in pipelines:
        report, ms = reports[p], times[p]
        if ms is None:
            row += f"{'—':>12}{'—':>12}"
        else:
            row += f"{ms:>12.4f}{report['aggregate']['saved_bytes']:>12,.0f}"
    a, b = times.get(primary), times.get(secondary)
    row += f"{(a / b):>10.2f}" if a and b else f"{'—':>10}"
    baseline = next(
        (r["performance_baseline"] for r in reports.values() if r and r.get("performance_baseline")),
        "—",
    )
    print(row + f"{baseline:>18}")

    for p in pipelines:
        for stage in ("seed", "verify", "bench"):
            rc, why = status.get((op, p, stage), (0, ""))
            if rc:
                failures.append(f"  {op:<20} {p:<3} {stage:<7} {why}")
                break  # later stages were skipped; the first failure is the cause

if failures:
    print(f"\n{len(failures)} failure(s):")
    print("\n".join(failures))
    print(f"\nlogs: {root}/<op>/{{seed,verify,bench}}_<pipeline>.{{log,json}}")
PY
