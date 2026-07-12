"""Parity gate: evograd's evaluator vs the old repo's, on the same candidate.

Run on a GPU node with both repos present:

    PYTHONPATH=src python scripts/gpu_parity.py \\
        --op layernorm \\
        --old-repo /u/akumdam/openevolve \\
        --candidate /u/akumdam/openevolve/benchmark/triton_layernorm_backward_bench/initial_program_autograd_pair.py

Correctness verdicts must MATCH (hard gate). Speedups are timing-based and
compared with a tolerance band — large drift is flagged for investigation, not
failed outright.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_OLD_BENCH = "benchmark/triton_{op}_backward_bench/evaluator_autograd_pair.py"

REPORT_KEYS = (
    "correct",
    "partial_correctness",
    "speedup",
    "full_step_speedup",
    "saved_memory_ratio",
)


def _run_json(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> dict:
    completed = subprocess.run(
        cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "metrics": {},
            "_error": {
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True)
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--speedup-band",
        type=float,
        default=0.15,
        help="relative drift in speedup metrics that triggers a warning (default 15%%)",
    )
    args = parser.parse_args()

    from evograd.pipelines.shared.runner import evograd_env

    old_evaluator = args.old_repo / _OLD_BENCH.format(op=args.op)
    if not old_evaluator.exists():
        print(f"old evaluator not found: {old_evaluator}")
        return 2

    print(f"[parity] old evaluator: {old_evaluator}")
    old = _run_json(
        [sys.executable, str(old_evaluator), str(args.candidate)], cwd=args.old_repo
    )
    print(f"[parity] new evaluator: evograd.evolve.evaluator --op {args.op}")
    new = _run_json(
        [
            sys.executable,
            "-m",
            "evograd.evolve.evaluator",
            "--op",
            args.op,
            "--scoring",
            "speed",
            str(args.candidate),
        ],
        env=evograd_env(),
    )

    old_metrics = old.get("metrics", {})
    new_metrics = new.get("metrics", {})
    failures, warnings = [], []

    for key in REPORT_KEYS:
        old_value = old_metrics.get(key)
        new_value = new_metrics.get(key)
        marker = " "
        if key in ("correct", "partial_correctness"):
            if old_value != new_value:
                failures.append(key)
                marker = "✗"
        elif isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)) and old_value:
            drift = abs(new_value - old_value) / abs(old_value)
            if drift > args.speedup_band:
                warnings.append(f"{key} drift {drift:.1%}")
                marker = "~"
        print(f"  {marker} {key:24s} old={old_value!r:>12} new={new_value!r:>12}")

    if "_error" in old:
        print(f"[parity] old evaluator errored: {old['_error']}")
    if "_error" in new:
        print(f"[parity] new evaluator errored: {new['_error']}")

    if failures or "_error" in old or "_error" in new:
        print(f"\nPARITY FAIL: {failures or 'evaluator error'}")
        return 1
    if warnings:
        print(f"\nPARITY OK (with timing drift warnings: {warnings})")
        return 0
    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
