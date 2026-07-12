"""Latency/memory benchmark harness, generic over operator declarations.

Ported from the per-bench ``_benchmark_case`` / ``_run_benchmarks`` in the old
``evaluator_autograd_pair.py`` files, with the operator-specific parts
(argument lists, oracle, input construction) supplied by the :class:`OpDecl`.
Used by both the OpenEvolve evaluator and the standalone ``evograd bench``
command.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

import torch

from evograd.opdecl.activity import Duplicated, OpDecl, Workload
from evograd.opdecl.bind import backward_const_kwargs, bind, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle, resolve_forward
from evograd.evolve.scoring import geomean, weighted_geomean

DEFAULT_WARMUP = 10
DEFAULT_REPS = 50


def normalize_saved(saved: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(saved, torch.Tensor):
        saved_tuple = (saved,)
    elif isinstance(saved, (tuple, list)):
        saved_tuple = tuple(saved)
    else:
        raise TypeError("saved_tensors must be a Tensor or a tuple/list of Tensors")
    if not all(isinstance(t, torch.Tensor) for t in saved_tuple):
        raise TypeError("all saved_tensors entries must be torch.Tensor instances")
    return saved_tuple


def saved_bytes(saved: tuple[torch.Tensor, ...]) -> int:
    return int(sum(t.numel() * t.element_size() for t in saved))


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
) -> dict[str, Any]:
    fwd, bwd = lookup_pair(op, module)
    inputs = make_case_inputs(op, workload, device=device)
    dout = inputs[op.upstream_grad_name]
    positional = [inputs.get(a.name, getattr(a, "default", None)) for a in op.args]
    bwd_kwargs = backward_const_kwargs(op, bwd, inputs)
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
    dup_positions = [i for i, a in enumerate(op.args) if isinstance(a, Duplicated)]

    def candidate_autograd_full_step():
        args = list(positional)
        leaves = []
        for i in dup_positions:
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
        for i in dup_positions:
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
    baseline_ms = median_ms(baseline_backward, warmup, reps)
    baseline_full_ms = median_ms(baseline_full_step, warmup, reps)

    saved_byte_count = saved_bytes(saved_tensors)
    input_byte_count = int(
        sum(
            inputs[a.name].numel() * inputs[a.name].element_size()
            for a in op.args
            if getattr(a, "shape", None) is not None
        )
    )
    return {
        "dims": dict(workload.dims),
        "dtype": workload.dtype,
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
        "saved_tensors": [
            {"shape": list(t.shape), "dtype": str(t.dtype), "bytes": t.numel() * t.element_size()}
            for t in saved_tensors
        ],
    }


def run_benchmarks(
    op: OpDecl,
    module,
    *,
    warmup: int = DEFAULT_WARMUP,
    reps: int = DEFAULT_REPS,
    device: str = "cuda",
    geomean_weights: tuple[float, ...] | None = None,
) -> dict[str, Any]:
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
    for workload in op.benchmark:
        report = benchmark_case(op, module, workload, warmup=warmup, reps=reps, device=device)
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
    if weights is not None and len(weights) != len(cases):
        raise ValueError(
            f"geomean_weights length {len(weights)} must match number of benchmark cases {len(cases)}"
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
    return {"aggregate": totals, "cases": cases, "geomean_weights": weights or []}
