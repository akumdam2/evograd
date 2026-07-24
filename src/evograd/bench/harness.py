"""Latency/memory benchmark harness, generic over operator declarations.

Ported from the per-bench ``_benchmark_case`` / ``_run_benchmarks`` in the old
``evaluator_autograd_pair.py`` files, with the operator-specific parts
(argument lists, oracle, input construction) supplied by the :class:`OpDecl`.
Used by both the OpenEvolve evaluator and the standalone ``evograd bench``
command.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from typing import Any, Callable

import torch

from evograd.opdecl.activity import Active, OpDecl, Workload
from evograd.opdecl.baselines import (
    resolve_performance_baseline,
    verify_performance_baseline,
)
from evograd.opdecl.bind import backward_inactive_kwargs, bind, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle, resolve_forward
from evograd.evolve.scoring import geomean, weighted_geomean

DEFAULT_WARMUP = 10
DEFAULT_REPS = 50

_BASELINE_CACHE_MEMO: dict[str, dict[str, list[float]]] = {}


def _cache_enabled() -> bool:
    return os.environ.get("EVOGRAD_BASELINE_TIMING_CACHE", "1").lower() not in (
        "0",
        "false",
        "no",
    )


def _baseline_cache_path() -> str | None:
    if not _cache_enabled():
        return None
    return os.environ.get("EVOGRAD_BASELINE_TIMING_CACHE_PATH")


def _baseline_cache_key(
    op: OpDecl,
    workload: Workload,
    baseline: str,
    warmup: int,
    reps: int,
) -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return json.dumps(
        {
            "op": op.name,
            "baseline": baseline,
            "gpu": gpu,
            "warmup": warmup,
            "reps": reps,
            "dims": workload.dims,
            "dtype": workload.dtype,
        },
        sort_keys=True,
    )


def _baseline_cache_get(path: str | None, key: str) -> tuple[float, float] | None:
    if path is None:
        return None
    if path not in _BASELINE_CACHE_MEMO:
        try:
            with open(path, encoding="utf-8") as handle:
                _BASELINE_CACHE_MEMO[path] = json.load(handle)
        except Exception:
            _BASELINE_CACHE_MEMO[path] = {}
    value = _BASELINE_CACHE_MEMO[path].get(key)
    return (float(value[0]), float(value[1])) if value else None


def _baseline_cache_put(
    path: str | None, key: str, backward_ms: float, full_ms: float
) -> None:
    if path is None:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    data[key] = [float(backward_ms), float(full_ms)]
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".evograd_baseline_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
        _BASELINE_CACHE_MEMO[path] = data
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def normalize_saved(saved: Any) -> tuple[Any, ...]:
    """Normalize candidate saved state while preserving plain scalar metadata.

    ``bind`` knows how to route tensors through ``save_for_backward`` and keep
    immutable Python values in a side layout. The benchmark must accept the same
    contract; only tensors contribute to saved-memory accounting.
    """
    if isinstance(saved, (tuple, list)):
        saved_tuple = tuple(saved)
    else:
        saved_tuple = (saved,)
    return saved_tuple


def saved_bytes(saved: tuple[Any, ...]) -> int:
    return int(
        sum(t.numel() * t.element_size() for t in saved if isinstance(t, torch.Tensor))
    )


def describe_saved(saved: tuple[Any, ...]) -> list[dict[str, Any]]:
    descriptions = []
    for value in saved:
        if isinstance(value, torch.Tensor):
            descriptions.append(
                {
                    "kind": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": value.numel() * value.element_size(),
                }
            )
        else:
            descriptions.append(
                {"kind": "value", "type": type(value).__name__, "repr": repr(value)[:200]}
            )
    return descriptions


def median_ms(
    fn: Callable[[], object], warmup: int = DEFAULT_WARMUP, reps: int = DEFAULT_REPS
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def median_ms_timed_region(
    setup: Callable[[], Any],
    timed: Callable[[Any], object],
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
) -> float:
    for _ in range(warmup):
        state = setup()
        timed(state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        state = setup()
        start.record()
        timed(state)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return float(statistics.median(times))


def benchmark_case(
    op: OpDecl,
    module,
    workload: Workload,
    *,
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
    device: str = "cuda",
    performance_baseline: str = "pytorch_autograd",
) -> dict[str, Any]:
    performance_baseline = resolve_performance_baseline(op, performance_baseline)
    fwd, bwd = lookup_pair(op, module)
    inputs = make_case_inputs(op, workload, device=device)
    dout = inputs[op.upstream_grad_name]
    positional = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
    bwd_kwargs = backward_inactive_kwargs(op, bwd, inputs)
    forward_ref = resolve_forward(op)

    def forward_only():
        return fwd(*positional)

    def setup_saved():
        _y, saved = fwd(*positional)
        return normalize_saved(saved)

    def backward_from_saved(saved_tensors):
        return bwd(dout, saved_tensors, **bwd_kwargs)

    def candidate_raw_full_step():
        _y, saved = fwd(*positional)
        return bwd(dout, normalize_saved(saved), **bwd_kwargs)

    bound = bind(op, module)
    active_positions = [i for i, a in enumerate(op.args) if isinstance(a, Active)]

    def candidate_autograd_full_step():
        args = list(positional)
        leaves = []
        for i in active_positions:
            leaf = args[i].detach().clone().requires_grad_(True)
            args[i] = leaf
            leaves.append(leaf)
        y = bound(*args)
        torch.autograd.backward(y, dout.detach().clone())
        return [leaf.grad for leaf in leaves]

    def baseline_backward():
        return oracle(op, inputs)

    def baseline_full_step():
        args = list(positional)
        leaves = []
        for i in active_positions:
            leaf = args[i].detach().clone().requires_grad_(True)
            args[i] = leaf
            leaves.append(leaf)
        y = forward_ref(*args)
        torch.autograd.backward(y, dout.detach().clone())
        return [leaf.grad for leaf in leaves]

    with torch.no_grad():
        _y, saved = fwd(*positional)
        saved_tensors = normalize_saved(saved)

    forward_ms = median_ms(forward_only, warmup, reps)
    backward_ms = median_ms_timed_region(setup_saved, backward_from_saved, warmup, reps)
    raw_full_ms = median_ms(candidate_raw_full_step, warmup, reps)
    autograd_full_ms = median_ms(candidate_autograd_full_step, warmup, reps)
    cache_path = _baseline_cache_path()
    cache_key = _baseline_cache_key(
        op, workload, performance_baseline, warmup, reps
    )
    cached = _baseline_cache_get(cache_path, cache_key)
    if cached is not None:
        baseline_ms, baseline_full_ms = cached
    else:
        if performance_baseline == "pytorch_autograd":
            baseline_ms = median_ms(baseline_backward, warmup, reps)
            baseline_full_ms = median_ms(baseline_full_step, warmup, reps)
        else:
            try:
                baseline_hook = op.performance_baselines[performance_baseline]
            except KeyError:
                available = ["pytorch_autograd", *sorted(op.performance_baselines)]
                raise KeyError(
                    f"{op.name}: unknown performance baseline {performance_baseline!r}; "
                    f"available: {available}"
                ) from None
            baseline_ms, baseline_full_ms = baseline_hook(
                torch,
                inputs,
                dout,
                warmup,
                reps,
                median_ms,
                median_ms_timed_region,
            )
        _baseline_cache_put(
            cache_path, cache_key, baseline_ms, baseline_full_ms
        )

    saved_byte_count = saved_bytes(saved_tensors)
    input_byte_count = int(
        sum(
            inputs[name].numel() * inputs[name].element_size()
            for name in op.memory_input_names()
        )
    )
    return {
        "dims": dict(workload.dims),
        "dtype": workload.dtype,
        "performance_baseline": performance_baseline,
        "forward_ms": forward_ms,
        "backward_from_saved_ms": backward_ms,
        "forward_backward_full_step_ms": raw_full_ms,
        "raw_forward_backward_full_step_ms": raw_full_ms,
        "autograd_forward_backward_full_step_ms": autograd_full_ms,
        "baseline_backward_ms": baseline_ms,
        "baseline_full_step_ms": baseline_full_ms,
        "baseline_raw_full_step_ms": baseline_full_ms,
        "pytorch_autograd_backward_ms": baseline_ms,
        "pytorch_autograd_full_step_ms": baseline_full_ms,
        "speedup_vs_baseline_backward": baseline_ms / max(backward_ms, 1e-9),
        "speedup_vs_baseline_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "speedup_vs_baseline_raw_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "speedup_vs_baseline_autograd_full_step": baseline_full_ms / max(autograd_full_ms, 1e-9),
        "speedup_vs_pytorch_autograd_backward": baseline_ms / max(backward_ms, 1e-9),
        "speedup_vs_pytorch_autograd_full_step": baseline_full_ms / max(raw_full_ms, 1e-9),
        "saved_bytes": saved_byte_count,
        "input_bytes": input_byte_count,
        "saved_memory_ratio": saved_byte_count / max(input_byte_count, 1),
        "saved_tensors": describe_saved(saved_tensors),
    }


def run_benchmarks(
    op: OpDecl,
    module,
    *,
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
    device: str = "cuda",
    geomean_weights: tuple[float, ...] | None = None,
    workloads: tuple[Workload, ...] | None = None,
    performance_baseline: str = "pytorch_autograd",
) -> dict[str, Any]:
    performance_baseline = resolve_performance_baseline(op, performance_baseline)
    verify_performance_baseline(
        op, performance_baseline, device=device
    )
    cases = []
    totals = {
        "forward_ms": 0.0,
        "backward_from_saved_ms": 0.0,
        "forward_backward_full_step_ms": 0.0,
        "raw_forward_backward_full_step_ms": 0.0,
        "autograd_forward_backward_full_step_ms": 0.0,
        "baseline_backward_ms": 0.0,
        "baseline_full_step_ms": 0.0,
        "baseline_raw_full_step_ms": 0.0,
        "saved_bytes": 0.0,
        "input_bytes": 0.0,
    }
    selected = op.benchmark if workloads is None else workloads
    if not selected:
        raise ValueError(f"{op.name}: no benchmark workloads selected")
    for workload in selected:
        report = benchmark_case(
            op,
            module,
            workload,
            warmup=warmup,
            reps=reps,
            device=device,
            performance_baseline=performance_baseline,
        )
        cases.append(report)
        for key in totals:
            totals[key] += float(report[key])

    totals["speedup_vs_baseline_backward"] = totals["baseline_backward_ms"] / max(
        totals["backward_from_saved_ms"], 1e-9
    )
    totals["speedup_vs_baseline_full_step"] = totals["baseline_full_step_ms"] / max(
        totals["forward_backward_full_step_ms"], 1e-9
    )
    totals["speedup_vs_baseline_raw_full_step"] = totals["baseline_raw_full_step_ms"] / max(
        totals["raw_forward_backward_full_step_ms"], 1e-9
    )
    totals["pytorch_autograd_backward_ms"] = totals["baseline_backward_ms"]
    totals["pytorch_autograd_full_step_ms"] = totals["baseline_full_step_ms"]
    totals["speedup_vs_pytorch_autograd_backward"] = totals["speedup_vs_baseline_backward"]
    totals["speedup_vs_pytorch_autograd_full_step"] = totals["speedup_vs_baseline_full_step"]

    backward_speedups = [
        float(c["speedup_vs_baseline_backward"])
        for c in cases
        if float(c["speedup_vs_baseline_backward"]) > 0.0
    ]
    full_step_speedups = [
        float(c["speedup_vs_baseline_raw_full_step"])
        for c in cases
        if float(c["speedup_vs_baseline_raw_full_step"]) > 0.0
    ]
    min_speedups = [
        min(
            float(c["speedup_vs_baseline_backward"]),
            float(c["speedup_vs_baseline_raw_full_step"]),
        )
        for c in cases
        if float(c["speedup_vs_baseline_backward"]) > 0.0
        and float(c["speedup_vs_baseline_raw_full_step"]) > 0.0
    ]
    weights = list(geomean_weights) if geomean_weights else None
    if weights is not None:
        if any(weight <= 0 for weight in weights):
            raise ValueError("geomean weights must contain only positive values")
        if len(weights) != len(cases):
            unique_dims: list[tuple[tuple[str, int], ...]] = []
            for case in cases:
                dims_key = tuple(sorted((name, int(value)) for name, value in case["dims"].items()))
                if dims_key not in unique_dims:
                    unique_dims.append(dims_key)
            if len(weights) == len(unique_dims):
                by_dims = dict(zip(unique_dims, weights))
                weights = [
                    by_dims[
                        tuple(
                            sorted((name, int(value)) for name, value in case["dims"].items())
                        )
                    ]
                    for case in cases
                ]
            else:
                raise ValueError(
                    "geomean_weights length must match benchmark cases "
                    f"({len(cases)}) or unique shapes ({len(unique_dims)}), got {len(weights)}"
                )
    totals["geomean_speedup_vs_baseline_backward"] = geomean(backward_speedups)
    totals["geomean_speedup_vs_baseline_full_step"] = geomean(full_step_speedups)
    totals["geomean_min_speedup_per_case"] = geomean(min_speedups)
    totals["weighted_geomean_speedup_vs_baseline_backward"] = weighted_geomean(
        backward_speedups, weights
    )
    totals["weighted_geomean_speedup_vs_baseline_full_step"] = weighted_geomean(
        full_step_speedups, weights
    )
    totals["weighted_geomean_min_speedup_per_case"] = weighted_geomean(min_speedups, weights)
    totals["worst_case_speedup_vs_baseline_backward"] = (
        min(backward_speedups) if backward_speedups else 0.0
    )
    totals["worst_case_speedup_vs_baseline_full_step"] = (
        min(full_step_speedups) if full_step_speedups else 0.0
    )
    totals["worst_case_min_speedup"] = min(min_speedups) if min_speedups else 0.0
    totals["saved_memory_ratio"] = totals["saved_bytes"] / max(totals["input_bytes"], 1e-9)
    return {
        "aggregate": totals,
        "cases": cases,
        "geomean_weights": weights or [],
        "performance_baseline": performance_baseline,
    }
