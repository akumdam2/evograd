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


def _env(name: str, legacy: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value is not None else os.environ.get(legacy) if legacy else None


def _policy_from_env():
    overrides = {}
    full_step_weight = _env("EVOGRAD_FULL_STEP_WEIGHT", "AUTOGRAD_PAIR_FULL_STEP_WEIGHT")
    memory_weight = _env(
        "EVOGRAD_MEMORY_PENALTY_WEIGHT", "AUTOGRAD_PAIR_MEMORY_PENALTY_WEIGHT"
    )
    worst_guard = _env("EVOGRAD_WORST_CASE_GUARD", "AUTOGRAD_PAIR_WORST_CASE_GUARD")
    weights = _env("EVOGRAD_GEOMEAN_WEIGHTS", "AUTOGRAD_PAIR_GEOMEAN_WEIGHTS")
    if full_step_weight:
        overrides["full_step_weight"] = float(full_step_weight)
    if memory_weight:
        overrides["memory_penalty_weight"] = float(memory_weight)
    if worst_guard:
        overrides["worst_case_guard"] = worst_guard.lower() not in (
            "0",
            "false",
            "no",
        )
    if weights:
        overrides["geomean_weights"] = tuple(
            float(part) for part in weights.split(",") if part.strip()
        )
    scoring = _env("EVOGRAD_SCORING", "AUTOGRAD_PAIR_SCORE_MODE") or "speed_memory"
    if scoring == "speed_only":
        scoring = "speed"
    return get_policy(scoring, **overrides)


def _int_env(name: str, legacy: str | None = None, minimum: int = 0) -> int | None:
    value = _env(name, legacy)
    if not value:
        return None
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _benchmark_selection(op):
    suite = os.environ.get("EVOGRAD_BENCHMARK_SUITE")
    if suite is None and op.name == "layernorm":
        suite = os.environ.get("LAYERNORM_BENCHMARK_SUITE") or os.environ.get(
            "AUTOGRAD_PAIR_SHAPE_PROFILE"
        )
    raw_dtypes = os.environ.get("EVOGRAD_BENCHMARK_DTYPES")
    if raw_dtypes is None and op.name == "layernorm":
        raw_dtypes = os.environ.get("LAYERNORM_BENCHMARK_DTYPES")
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtypes = None
    if raw_dtypes:
        dtypes = tuple(
            aliases.get(part.strip().lower(), part.strip().lower())
            for part in raw_dtypes.split(",")
            if part.strip()
        )
    return op.benchmark_workloads(suite=suite, dtypes=dtypes)


_OP = get_op(os.environ["EVOGRAD_OP"])

evaluate = build_evaluate(
    _OP,
    _policy_from_env(),
    warmup=_int_env(
        "EVOGRAD_BENCHMARK_WARMUP", "LAYERNORM_BENCHMARK_WARMUP", minimum=0
    ),
    reps=_int_env("EVOGRAD_BENCHMARK_REPS", "LAYERNORM_BENCHMARK_REPS", minimum=1),
    benchmark_workloads=_benchmark_selection(_OP),
    performance_baseline=os.environ.get(
        "EVOGRAD_PERFORMANCE_BASELINE",
        os.environ.get("AUTOGRAD_PAIR_PERF_BASELINE", "pytorch_autograd"),
    ),
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
