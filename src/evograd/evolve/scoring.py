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
    }
