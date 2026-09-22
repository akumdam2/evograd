"""Run every declared operator's eager and torch.compile providers through the
tier-2 correctness gate, and say -- per operator, live -- whether they cleared it.

    PYTHONPATH=src python scripts/gate_sweep.py --out results/baseline-gate

**What this measures.** The gate's negative controls prove it *rejects* wrong
kernels. Nothing has ever proved it *accepts* a right one that is not
bit-identical to the oracle. Eager is the identity control -- the reference
compared against itself, which must pass or the harness disagrees with itself.
``torch.compile`` is the real probe: Inductor lowers the same graph, so it is
semantically identical by construction, but it fuses reductions and reorders
accumulation exactly the way an evolved Triton kernel does. A tolerance that
rejects it would reject every legitimate candidate.

**Timings here are deliberately worthless.** ``--rep-ms``/``--warmup-ms`` are
driven to their floor and per-provider process isolation is off, because both
exist to make *timing* trustworthy and neither affects an ``allclose`` verdict.
Re-run the interesting operators without ``--fast`` if you want numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def declared_operators():
    """Every task in the registry, ordered the way ``evograd ops`` orders them."""
    from evograd.benchmark import TASKS

    return sorted(TASKS.items(), key=lambda kv: (kv[1].level or 99, kv[0]))


def tolerance_for_case(op, case, name):
    """The declared (atol, rtol) for one result at the shape a case ran.

    Looked up by matching the case's dims and dtype back to the declared
    workload, because the report records the measured error but not the
    threshold it was judged against.
    """
    dims, dtype = case.get("dims"), case.get("dtype")
    for workload in op.benchmark:
        if dict(workload.dims) == dims and workload.dtype == dtype:
            return op.tolerance_for(workload, name)
    return (None, None)


def headroom_of(op, case, check, name):
    """Fraction of the allclose allowance a result actually used.

    Prefers the value the runner recorded. Older reports carry only
    ``max_abs_error``, so the tolerance is re-derived from the declaration and
    the ``rtol * |expected|`` term is unavailable -- those fall back to
    ``max_abs_error / atol``, which *overstates* how close a passing provider
    came. A number above 1 from an old report usually means ``rtol`` was
    carrying the gate, not that anything was marginal.
    """
    if check.get("headroom") is not None:
        return check["headroom"]
    error = check.get("max_abs_error")
    if error is None:
        return None
    atol = check.get("atol")
    if atol is None:
        atol, _rtol = tolerance_for_case(op, case, name)
    return (error / atol) if atol else None


def verdict_for(op, report):
    """Per provider: did it clear the gate, and by how much.

    ``worst`` is the largest fraction of the allowance any passing result used.
    Below 1 means it passed; near 1 means the gate is one unlucky shape from
    rejecting a correct implementation.
    """
    providers: dict[str, dict] = {}
    for case in report.get("cases", []):
        for name, entry in (case.get("providers") or {}).items():
            slot = providers.setdefault(name, {"ok": True, "worst": 0.0, "where": None,
                                               "cases": 0, "failed": []})
            slot["cases"] += 1
            if entry.get("ok") is False and "correctness" not in entry:
                # Died before the gate -- a crash, not a tolerance verdict.
                slot["ok"] = False
                slot["failed"].append((case.get("dims"), case.get("dtype"),
                                       entry.get("error", "error"), None, None))
                continue
            for result, check in (entry.get("correctness", {}).get("checks") or {}).items():
                used = headroom_of(op, case, check, result)
                error = check.get("max_abs_error")
                if not check.get("ok", True):
                    slot["ok"] = False
                    # `used` is the elementwise fraction of the allowance the
                    # worst element consumed -- the only figure that agrees
                    # with the gate. Rebuilding an allowance from `ref_absmax`
                    # here would repeat the error it was added to fix.
                    slot["failed"].append((case.get("dims"), case.get("dtype"),
                                           result, error, used))
                if used is not None and used > slot["worst"]:
                    slot["worst"] = used
                    slot["where"] = (
                        f"{result} {case.get('dtype')} {case.get('dims')}"
                    )
    return providers


#: What the sweep expects to find in every report. A provider that never
#: appears is not a provider that passed -- without this the report reads green
#: when the probe simply did not run.
EXPECTED_PROVIDERS = ("eager", "torch_compile")


def redundant_providers(providers) -> list[str]:
    """Providers whose numbers are identical to eager's, to full precision.

    Two different implementations do not land on the same float. When
    ``torch_compile`` matches ``eager`` exactly, Inductor produced the same
    arithmetic -- it fused nothing, or lowered to the same library call, which
    is what happens to a single SDPA or a cuBLAS GEMM. That row is one
    measurement wearing two labels, and it is evidence about the tolerance
    from eager alone. Worth marking, because it is invisible otherwise and it
    changed how a fifth of the last sweep should have been read.
    """
    eager = providers.get("eager")
    if eager is None:
        return []
    return [
        name for name, slot in providers.items()
        if name != "eager" and slot["ok"] == eager["ok"]
        and abs(slot["worst"] - eager["worst"]) < 1e-12
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/baseline-gate"))
    parser.add_argument("--ops", default=None,
                        help="comma-separated subset; default is every declared task")
    parser.add_argument("--baseline", default="none",
                        help="declared pair baseline to gate as well (liger, ...)")
    parser.add_argument("--fast", action="store_true", default=True,
                        help="floor the timing budget and skip process isolation")
    parser.add_argument("--timed", dest="fast", action="store_false",
                        help="real timing protocol; much slower")
    parser.add_argument("--reuse", action="store_true",
                        help="re-read the reports already in --out instead of "
                             "running anything; the verdicts live in the JSON, "
                             "so only the summary changes")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "sweep.log"
    log = log_path.open("a", encoding="utf-8")

    def say(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    operators = declared_operators()
    if args.ops:
        wanted = {o.strip() for o in args.ops.split(",")}
        operators = [(n, op) for n, op in operators if n in wanted]

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")

    say("=" * 78)
    try:
        import torch

        environment = (f"torch {torch.__version__}  "
                       f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    except Exception as exc:  # a machine without torch cannot run the sweep anyway
        environment = f"torch unavailable: {exc}"
    say(environment)
    say(f"gate sweep  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"{len(operators)} operators  "
        + ("re-reading existing reports" if args.reuse
           else "fast (timings meaningless)" if args.fast
           else "full timing protocol"))
    say("PASS means the gate did not reject these providers -- it measures only "
        "that the tolerance is not too tight.")
    say("It cannot see whether the tolerance is tight enough; that needs the "
        "negative controls.")
    say("=" * 78)

    summary, started = [], time.time()
    for index, (name, op) in enumerate(operators, 1):
        out = args.out / f"tier2-{name}.json"
        argv_child = [sys.executable, "-m", "evograd.cli", "tier2-bench",
                      "--op", name, "--baseline", args.baseline, "--out", str(out)]
        if args.fast:
            argv_child += ["--no-isolate", "--rep-ms", "1", "--warmup-ms", "1"]

        head = f"[{index:2d}/{len(operators)}] {name:26} L{op.level or '-'}"
        clock = time.time()
        if args.reuse:
            elapsed = 0.0
            if not out.is_file():
                say(f"{head}  ABSENT  no report at {out}")
                summary.append((name, "ABSENT", str(out)))
                continue
        else:
            print(f"{head}  running...", end="\r", flush=True)
            child = subprocess.run(argv_child, capture_output=True, text=True, env=env)
            elapsed = time.time() - clock
            (args.out / f"tier2-{name}.log").write_text(
                child.stdout + child.stderr, encoding="utf-8"
            )
            if child.returncode != 0 or not out.is_file():
                reason = (child.stderr.strip().splitlines() or ["no output"])[-1]
                say(f"{head}  CRASH  ({elapsed:5.1f}s)  {reason[:90]}")
                summary.append((name, "CRASH", reason[:60]))
                continue

        providers = verdict_for(op, json.loads(out.read_text(encoding="utf-8")))
        if not providers:
            say(f"{head}  EMPTY  ({elapsed:5.1f}s)  no providers in report")
            summary.append((name, "EMPTY", ""))
            continue

        redundant = set(redundant_providers(providers))
        missing = [p for p in EXPECTED_PROVIDERS if p not in providers]
        parts, worst_overall, failed_any = [], 0.0, False
        for provider in sorted(providers):
            slot = providers[provider]
            worst_overall = max(worst_overall, slot["worst"])
            failed_any = failed_any or not slot["ok"]
            mark = "=eager" if provider in redundant else ""
            parts.append(
                f"{provider}={'PASS' if slot['ok'] else 'FAIL'}"
                f"({slot['worst']:.2f}){mark}"
            )
        state = "FAIL " if failed_any or missing else "PASS "
        if missing:
            failed_any = True
        cases = max(slot["cases"] for slot in providers.values())
        say(f"{head}  {state} ({elapsed:5.1f}s)  cases={cases:3d}  "
            + "  ".join(parts)
            + (f"  MISSING={','.join(missing)}" if missing else ""))
        worst_slot = max(providers.values(), key=lambda s: s["worst"])
        if worst_slot["where"] and worst_slot["worst"] >= 0.25:
            say(f"{'':14}  tightest: {worst_slot['where']} "
                f"at {worst_slot['worst']:.2f} of allowance")

        for provider in sorted(providers):
            for dims, dtype, result, error, used in providers[provider]["failed"][:4]:
                detail = (f"max_abs={error:.3e} used {used:.2f}x its allowance"
                          if error is not None and used else str(result))
                say(f"{'':14}  -> {provider}: {result if error is not None else ''} "
                    f"{dtype} {dims} {detail}")
        summary.append((name, "FAIL" if failed_any else "PASS",
                        f"worst {worst_overall:.2f} of allowance"))

    say("-" * 78)
    passed = [s for s in summary if s[1] == "PASS"]
    say(f"{len(passed)}/{len(summary)} operators clear the gate "
        f"in {(time.time() - started) / 60:.1f} min")
    for name, state, note in summary:
        if state != "PASS":
            say(f"  {state:6} {name:26} {note}")
    say(f"log: {log_path}")
    log.close()
    return 0 if all(s[1] == "PASS" for s in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
