"""The evaluator file handed to ``openevolve-run``.

OpenEvolve loads an evaluator by file path and calls its module-level
``evaluate``. This single entry file serves every model case: the task pack and
scoring policy come from the environment, set by ``evograd.evolve.run`` (or
``evograd.adbench_tools.evaluate.evaluator_env``):

    EVOGRAD_DECLARATION — the pack's scoped declaration, ``path.py:op`` (required)
    EVOGRAD_OP       — its operator name (required; checked against the declaration)
    EVOGRAD_SEARCH_CORRECTNESS — the pack's ``task.json``, whose search rule
                       judges every candidate (required)
    EVOGRAD_SCORING  — scoring policy name (default: speed_memory)
    EVOGRAD_PROGRAM_ARCHIVE_DIR — when set, keep every evaluated candidate here
                                  (evograd evolve --save-programs)

Optional knobs (mirroring the old AUTOGRAD_PAIR_* env vars):

    EVOGRAD_FULL_STEP_WEIGHT, EVOGRAD_MEMORY_PENALTY_WEIGHT,
    EVOGRAD_WORST_CASE_GUARD, EVOGRAD_GEOMEAN_WEIGHTS (comma-separated),
    EVOGRAD_BENCHMARK_WARMUP, EVOGRAD_BENCHMARK_REPS
"""

from __future__ import annotations

import json
import os
import sys

from evograd.evolve.evaluator import archive_program, build_evaluate, evaluate_isolated
from evograd.evolve.scoring import get_policy
from evograd.benchmark import load_task
from evograd.verification.payloads.gate1 import SEARCH_CORRECTNESS_ENV


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


if not os.environ.get(SEARCH_CORRECTNESS_ENV):
    raise RuntimeError(f"{SEARCH_CORRECTNESS_ENV} is not set: this evaluator judges candidates "
                       "only by a task pack's search rule (evograd evolve --pack)")
_OP = load_task(os.environ["EVOGRAD_DECLARATION"])
if _OP.name != os.environ["EVOGRAD_OP"]:
    raise ValueError("evaluator declaration does not match EVOGRAD_OP")

_evaluate_direct = build_evaluate(
    _OP,
    _policy_from_env(),
    warmup=_int_env(
        "EVOGRAD_BENCHMARK_WARMUP", "LAYERNORM_BENCHMARK_WARMUP", minimum=0
    ),
    reps=_int_env("EVOGRAD_BENCHMARK_REPS", "LAYERNORM_BENCHMARK_REPS", minimum=1),
    performance_baseline=os.environ.get(
        "EVOGRAD_PERFORMANCE_BASELINE",
        os.environ.get("AUTOGRAD_PAIR_PERF_BASELINE", "pytorch_autograd"),
    ),
)

_ARCHIVE_DIR = os.environ.get("EVOGRAD_PROGRAM_ARCHIVE_DIR")

if os.environ.get("EVOGRAD_EVAL_CHILD") == "1":
    # The isolation child evaluates the same file the parent already archives.
    evaluate = _evaluate_direct
elif _ARCHIVE_DIR:

    def evaluate(program_path):
        result = evaluate_isolated(__file__, program_path)
        archive_program(program_path, result, _ARCHIVE_DIR)
        return result

else:
    evaluate = lambda program_path: evaluate_isolated(__file__, program_path)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: EVOGRAD_DECLARATION=... EVOGRAD_OP=... {SEARCH_CORRECTNESS_ENV}=... "
              f"{argv[0]} PROGRAM_PATH")
        return 2
    result = evaluate(argv[1])
    payload = {"metrics": result.metrics, "artifacts": result.artifacts}
    result_path = os.environ.get("EVOGRAD_RESULT_JSON")
    if result_path:
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    print(json.dumps(payload, indent=2))
    return 0 if result.metrics.get("correct", 0.0) == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
