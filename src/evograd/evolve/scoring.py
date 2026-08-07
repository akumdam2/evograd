"""Scoring policies for the generic OpenEvolve evaluator.

The old repo encoded these as per-bench evaluator *files* selected by env vars
(``evaluator_autograd_pair_speed_memory.py``,
``..._speed_memory_min_geomean.py``, ...). Here a policy is data: one
:class:`ScoringPolicy` per mode, applied by one evaluator. The arithmetic is
ported unchanged from ``_score_from_aggregate`` so scores are comparable
across repos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from evograd.opdecl.activity import Workload


@dataclass(frozen=True)
class ScoringPolicy:
    name: str
    mode: str  # speed_only | speed_memory | speed_memory_min | *_geomean | *_weighted_geomean
    full_step_weight: float = 0.5
    memory_penalty_weight: float = 0.05
    worst_case_guard: bool = True
    # Optional per-benchmark-case weights for the weighted-geomean modes; None
    # means uniform.
    geomean_weights: tuple[float, ...] | None = None


# ── MAP-Elites feature dimensions ────────────────────────────────────────────
# OpenEvolve's program database is a MAP-Elites grid; these name its axes.
# Built-ins are computed by OpenEvolve itself. Custom names must be metrics the
# evaluator returns on *every* path (OpenEvolve raises if a configured
# dimension is missing from a program's metrics), so each custom dimension
# carries the worst-case default `_result` injects into failure results.
BUILTIN_FEATURE_DIMENSIONS = ("complexity", "diversity", "score")
CUSTOM_FEATURE_DIMENSION_DEFAULTS: dict[str, float] = {
    # Failed candidates bin as if they cached every input (heaviest behavior).
    "saved_memory_ratio": 1.0,
    "speedup": 0.0,
    "full_step_speedup": 0.0,
    # Regime axes from config_map_shape_template.yaml; the evaluator's emit()
    # computes them via regime_speedup_metrics and already defaults both to
    # 0.0 on failure paths — registering them here keeps validate/setdefault
    # consistent with that contract.
    "small_regime_speedup": 0.0,
    "large_regime_speedup": 0.0,
    # >1 = large-shape specialist, <1 = small-shape specialist, 1 = generalist.
    "shape_specialization": 1.0,
}
# saved_memory_ratio is the behavioral axis that matters for autograd pairs:
# it keeps recompute-heavy and cache-heavy backward strategies in separate
# cells instead of letting the memory penalty collapse the tradeoff.
DEFAULT_FEATURE_DIMENSIONS = ("complexity", "saved_memory_ratio")


def shape_specialization_from_cases(cases: list[dict]) -> float:
    """Large-shape vs small-shape full-step speedup ratio for MAP-Elites.

    Splits the benchmark cases at the median problem size (product of the
    case's dims) and returns geomean(large speedups) / geomean(small
    speedups). Unlike the declared-split regime axes above, this is a single
    scale-free ratio usable on ops with no declared regime_feature. Keeping it
    as a feature dimension retains small-shape and large-shape specialists as
    separate elites instead of the aggregate score collapsing them into one
    generalist.
    """
    sized = []
    for case in cases:
        dims = case.get("dims") or {}
        speedup = case.get("speedup_vs_baseline_full_step")
        if not dims or not speedup or speedup <= 0.0:
            continue
        size = 1
        for value in dims.values():
            size *= int(value)
        sized.append((size, float(speedup)))
    if len(sized) < 2:
        return 1.0
    cutoff = sorted(size for size, _ in sized)[len(sized) // 2]
    small = [s for size, s in sized if size < cutoff]
    large = [s for size, s in sized if size >= cutoff]
    if not small or not large:
        return 1.0
    return geomean(large) / geomean(small)


def validate_feature_dimensions(dimensions: tuple[str, ...]) -> tuple[str, ...]:
    allowed = set(BUILTIN_FEATURE_DIMENSIONS) | set(CUSTOM_FEATURE_DIMENSION_DEFAULTS)
    unknown = [d for d in dimensions if d not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown MAP-Elites feature dimensions {unknown}; "
            f"built-ins: {sorted(BUILTIN_FEATURE_DIMENSIONS)}, "
            f"custom: {sorted(CUSTOM_FEATURE_DIMENSION_DEFAULTS)}"
        )
    if not dimensions:
        raise ValueError("feature_dimensions must not be empty")
    return dimensions


POLICIES: dict[str, ScoringPolicy] = {
    "speed": ScoringPolicy(name="speed", mode="speed_only"),
    "speed_memory": ScoringPolicy(name="speed_memory", mode="speed_memory"),
    "speed_memory_min": ScoringPolicy(name="speed_memory_min", mode="speed_memory_min"),
    "speed_memory_min_geomean": ScoringPolicy(
        name="speed_memory_min_geomean", mode="speed_memory_min_geomean"
    ),
    "speed_memory_min_weighted_geomean": ScoringPolicy(
        name="speed_memory_min_weighted_geomean", mode="speed_memory_min_weighted_geomean"
    ),
}


def get_policy(name: str, **overrides) -> ScoringPolicy:
    try:
        policy = POLICIES[name]
    except KeyError:
        raise KeyError(f"Unknown scoring policy {name!r}; available: {sorted(POLICIES)}") from None
    return replace(policy, **overrides) if overrides else policy


def geomean(values: list[float]) -> float:
    positive = [v for v in values if v > 0.0]
    return math.exp(sum(math.log(v) for v in positive) / max(len(positive), 1))


def weighted_geomean(values: list[float], weights: list[float] | None) -> float:
    positive = [(v, w) for v, w in zip(values, weights or [1.0] * len(values)) if v > 0.0]
    total_weight = sum(w for _, w in positive)
    if total_weight <= 0.0:
        raise ValueError("geomean weights must sum to a positive value")
    return math.exp(sum(w * math.log(v) for v, w in positive) / total_weight)


def score_from_aggregate(
    aggregate: dict[str, float], policy: ScoringPolicy
) -> tuple[float, dict[str, float]]:
    """Ported unchanged from the old evaluator's ``_score_from_aggregate``."""
    backward_speedup = float(aggregate["speedup_vs_baseline_backward"])
    full_step_speedup = float(aggregate["speedup_vs_baseline_full_step"])
    saved_memory_ratio = float(aggregate["saved_memory_ratio"])
    weighted_speedup = (
        (1.0 - policy.full_step_weight) * backward_speedup
        + policy.full_step_weight * full_step_speedup
    )
    min_speedup = min(backward_speedup, full_step_speedup)
    geomean_backward_speedup = float(
        aggregate.get("geomean_speedup_vs_baseline_backward", backward_speedup)
    )
    geomean_full_step_speedup = float(
        aggregate.get("geomean_speedup_vs_baseline_full_step", full_step_speedup)
    )
    geomean_min_speedup = min(geomean_backward_speedup, geomean_full_step_speedup)
    weighted_geomean_backward_speedup = float(
        aggregate.get("weighted_geomean_speedup_vs_baseline_backward", geomean_backward_speedup)
    )
    weighted_geomean_full_step_speedup = float(
        aggregate.get("weighted_geomean_speedup_vs_baseline_full_step", geomean_full_step_speedup)
    )
    weighted_geomean_min_speedup = float(
        aggregate.get(
            "weighted_geomean_min_speedup_per_case",
            min(weighted_geomean_backward_speedup, weighted_geomean_full_step_speedup),
        )
    )
    worst_case_min_speedup = float(aggregate.get("worst_case_min_speedup", min_speedup))
    worst_case_guard_factor = (
        min(1.0, worst_case_min_speedup) if policy.worst_case_guard else 1.0
    )
    memory_penalty_factor = 1.0 + policy.memory_penalty_weight * saved_memory_ratio

    if policy.mode == "speed_memory":
        score = weighted_speedup / memory_penalty_factor
    elif policy.mode == "speed_memory_min":
        score = min_speedup / memory_penalty_factor
    elif policy.mode == "speed_memory_min_geomean":
        score = geomean_min_speedup / memory_penalty_factor
    elif policy.mode == "speed_memory_min_weighted_geomean":
        score = (weighted_geomean_min_speedup * worst_case_guard_factor) / memory_penalty_factor
    else:  # speed_only
        score = backward_speedup

    return score, {
        "backward_speedup": backward_speedup,
        "full_step_speedup": full_step_speedup,
        "weighted_speedup": weighted_speedup,
        "min_speedup": min_speedup,
        "geomean_backward_speedup": geomean_backward_speedup,
        "geomean_full_step_speedup": geomean_full_step_speedup,
        "geomean_min_speedup": geomean_min_speedup,
        "weighted_geomean_backward_speedup": weighted_geomean_backward_speedup,
        "weighted_geomean_full_step_speedup": weighted_geomean_full_step_speedup,
        "weighted_geomean_min_speedup": weighted_geomean_min_speedup,
        "worst_case_min_speedup": worst_case_min_speedup,
        "worst_case_guard_factor": worst_case_guard_factor,
        "saved_memory_ratio": saved_memory_ratio,
        "memory_penalty_factor": memory_penalty_factor,
        "score_mode_speed_memory": 1.0 if policy.mode == "speed_memory" else 0.0,
        "score_mode_speed_memory_min": 1.0 if policy.mode == "speed_memory_min" else 0.0,
        "score_mode_speed_memory_min_geomean": (
            1.0 if policy.mode == "speed_memory_min_geomean" else 0.0
        ),
        "score_mode_speed_memory_min_weighted_geomean": (
            1.0 if policy.mode == "speed_memory_min_weighted_geomean" else 0.0
        ),
    }


def regime_speedup_metrics(
    cases: list[dict] | None,
    *,
    regime_feature,
    regime_split: float,
) -> dict[str, float]:
    """Geomean full-step speedups on the declared small/large shape regimes.

    Always returns both keys so OpenEvolve MAP-Elites feature axes stay defined
    even when a candidate fails before benchmarking.
    """
    empty = {"small_regime_speedup": 0.0, "large_regime_speedup": 0.0}
    if not cases or regime_feature is None or regime_split is None:
        return empty
    small: list[float] = []
    large: list[float] = []
    for case in cases:
        speedup = float(case.get("speedup_vs_baseline_raw_full_step", 0.0))
        if speedup <= 0.0:
            continue
        feature = float(
            regime_feature(
                Workload(dims=dict(case["dims"]), dtype=str(case["dtype"]))
            )
        )
        if feature < float(regime_split):
            small.append(speedup)
        else:
            large.append(speedup)
    return {
        "small_regime_speedup": geomean(small) if small else 0.0,
        "large_regime_speedup": geomean(large) if large else 0.0,
    }
