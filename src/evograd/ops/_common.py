"""Shared declaration helpers for the 2-D Liger-derived operator suite."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable

from evograd.opdecl import Workload


STANDARD_TOLERANCES = {
    "float32": (2e-5, 2e-5),
    "float16": (5e-2, 5e-2),
    "bfloat16": (8e-2, 8e-2),
}
STANDARD_CORRECTNESS_SHAPES = {
    "float32": ((8, 64), (17, 128)),
    "float16": ((32, 256), (64, 512)),
    "bfloat16": ((32, 256), (64, 512)),
}


def workloads_2d(
    shapes: Iterable[tuple[int, int]],
    dtypes: Iterable[str],
    *,
    row_dim: str = "rows",
    col_dim: str = "cols",
    tolerances: dict[str, tuple[float, float]] | None = None,
) -> tuple[Workload, ...]:
    tolerances = tolerances or {}
    return tuple(
        Workload(
            dims={row_dim: rows, col_dim: cols},
            dtype=dtype,
            atol=tolerances.get(dtype, (None, None))[0],
            rtol=tolerances.get(dtype, (None, None))[1],
        )
        for rows, cols in shapes
        for dtype in dtypes
    )


def standard_correctness(
    *,
    dtypes: Iterable[str] = ("float32", "float16", "bfloat16"),
    row_dim: str = "rows",
    col_dim: str = "cols",
    tolerances: dict[str, tuple[float, float]] | None = None,
) -> tuple[Workload, ...]:
    tolerances = tolerances or STANDARD_TOLERANCES
    return tuple(
        Workload(
            dims={row_dim: rows, col_dim: cols},
            dtype=dtype,
            atol=tolerances[dtype][0],
            rtol=tolerances[dtype][1],
        )
        for dtype in dtypes
        for rows, cols in STANDARD_CORRECTNESS_SHAPES[dtype]
    )


def regime_suites(
    workloads: Iterable[Workload],
    feature: Callable[[Workload], float],
    split: float,
) -> dict[str, tuple[Workload, ...]]:
    cases = tuple(workloads)
    return {
        "full": cases,
        "small": tuple(case for case in cases if feature(case) < split),
        "large": tuple(case for case in cases if feature(case) >= split),
    }


def log_distance_weight(feature: Callable[[Workload], float], split: float):
    def weight(workload: Workload) -> float:
        distance = abs(math.log2(max(feature(workload), 1.0)) - math.log2(split))
        return float(max(1.0, round(distance)))

    return weight


def dtype_for(torch, name: str):
    aliases = {"bf16": "bfloat16"}
    return getattr(torch, aliases.get(name, name))


def seed_2d(workload: Workload) -> int:
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    return rows * 100003 + cols


def make_pair_baseline(
    factory: Callable[[], tuple[Callable, Callable]],
    forward_args: tuple[str, ...],
    backward_extras: tuple[str, ...] = (),
):
    """Adapt a reviewed Liger autograd pair to evograd's timing-hook API."""

    def measure(torch, inputs, dout, warmup, reps, median_ms, median_ms_timed_region):
        forward, backward = factory()
        positional = [inputs[name] for name in forward_args]
        extras = [inputs[name] for name in backward_extras]

        def normalize(saved):
            return tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)

        def setup():
            _y, saved = forward(*positional)
            return normalize(saved)

        def run_backward(saved):
            return backward(dout, saved, *extras)

        def full_step():
            return run_backward(setup())

        backward_ms = median_ms_timed_region(setup, run_backward, warmup, reps)
        full_ms = median_ms(full_step, warmup, reps)
        return backward_ms, full_ms

    measure.__name__ = f"measure_{factory.__name__}"
    measure.available = lambda: baseline_available(factory)
    measure.pair_factory = factory
    measure.forward_args = tuple(forward_args)
    measure.backward_extras = tuple(backward_extras)
    return measure


def baseline_available(factory: Callable) -> bool:
    """Cheaply probe a Liger factory without importing Liger at declaration time."""
    try:
        factory()
        return True
    except Exception:
        return False


def accepted_kwargs(fn: Callable, values: dict) -> dict:
    """Return the subset of values accepted by a baseline/candidate callable."""
    parameters = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return values
    return {name: value for name, value in values.items() if name in parameters}
