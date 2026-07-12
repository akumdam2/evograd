"""The evaluator file handed to ``openevolve-run``.

OpenEvolve loads an evaluator by file path and calls its module-level
``evaluate``. This single entry file serves every operator: the target op and
scoring policy come from the environment, set by ``evograd.evolve.run``:

    EVOGRAD_OP       — operator name (required, see evograd.ops.OPS)
    EVOGRAD_SCORING  — scoring policy name (default: speed_memory)

Optional knobs (mirroring the old AUTOGRAD_PAIR_* env vars):

    EVOGRAD_FULL_STEP_WEIGHT, EVOGRAD_MEMORY_PENALTY_WEIGHT,
    EVOGRAD_WORST_CASE_GUARD, EVOGRAD_GEOMEAN_WEIGHTS (comma-separated),
    EVOGRAD_BENCHMARK_WARMUP, EVOGRAD_BENCHMARK_REPS
"""

from __future__ import annotations

import json
import os
import sys

from evograd.evolve.evaluator import build_evaluate
from evograd.evolve.scoring import get_policy
from evograd.ops import get_op


def _policy_from_env():
    overrides = {}
    if os.environ.get("EVOGRAD_FULL_STEP_WEIGHT"):
        overrides["full_step_weight"] = float(os.environ["EVOGRAD_FULL_STEP_WEIGHT"])
    if os.environ.get("EVOGRAD_MEMORY_PENALTY_WEIGHT"):
        overrides["memory_penalty_weight"] = float(os.environ["EVOGRAD_MEMORY_PENALTY_WEIGHT"])
    if os.environ.get("EVOGRAD_WORST_CASE_GUARD"):
        overrides["worst_case_guard"] = os.environ["EVOGRAD_WORST_CASE_GUARD"].lower() not in (
            "0",
            "false",
            "no",
        )
    if os.environ.get("EVOGRAD_GEOMEAN_WEIGHTS"):
        overrides["geomean_weights"] = tuple(
            float(part) for part in os.environ["EVOGRAD_GEOMEAN_WEIGHTS"].split(",") if part.strip()
        )
    return get_policy(os.environ.get("EVOGRAD_SCORING", "speed_memory"), **overrides)


def _int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


_OP = get_op(os.environ["EVOGRAD_OP"])

evaluate = build_evaluate(
    _OP,
    _policy_from_env(),
    warmup=_int_env("EVOGRAD_BENCHMARK_WARMUP"),
    reps=_int_env("EVOGRAD_BENCHMARK_REPS"),
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: EVOGRAD_OP=<op> {argv[0]} PROGRAM_PATH")
        return 2
    result = evaluate(argv[1])
    print(json.dumps({"metrics": result.metrics, "artifacts": result.artifacts}, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
