"""Tier 1 (direct pair), `fair` protocol — the numbers that get published.

Two axes run through ``bench/``, and it helps to keep them apart:

    *tier*      what is measured — a kernel pair, an operator through the
                autograd engine (``tier2``), a training step (tier 3, unbuilt)
    *protocol*  how carefully — ``fast`` for the evolutionary search,
                ``fair`` for anything anyone reads

This file is tier 1 x fair. ``harness.py`` is tier 1 x fast: the same interface
measured cheaply, because the search calls it thousands of times per run and
caches the baseline across candidates. The two are not interchangeable and the
suite records which produced a report.

What ``fair`` buys, and why the split exists: both providers are remeasured
every time (no baseline cache), L2 is cleared before each timed region, event
recording is batched with a single synchronize, provider order is randomized in
paired blocks, inputs are checked for mutation outside the timed regions, and
speedups carry block-bootstrap confidence intervals.
"""

from __future__ import annotations

import random
import statistics
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torch

from evograd.opdecl.activity import OpDecl, Workload
from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values

# The provider boundary and the input-mutation guards live in bench.provider so
# that protocols other than this one can reuse them. Re-exported here because
# `from evograd.bench.tier1 import PairProvider` is the spelling in existing
# callers, CLIs and tests.
from evograd.bench.provider import (  # noqa: F401  (re-export)
    PairProvider,
    TensorSnapshot,
    assert_tensors_unchanged,
    candidate_provider,
    clone_values,
    declared_provider,
    liger_provider,
    pytorch_autograd_provider,
    renamed_provider,
    saved_state_report,
    snapshot_tensors,
    torch_compile_provider,
    verify_pair_provider,
)
from evograd.bench.provider import saved_state_report as _saved_state_report  # noqa: F401

PROTOCOL_VERSION = "evograd-final-runtime-v1"

#: Wall-clock budget per timed loop, in milliseconds. ``triton.testing.do_bench``
#: sizes its loops the same way (its ``rep`` argument is a duration, not a
#: count) and defaults to 100 ms.
_REP_BUDGET_MS = 100.0

#: Never drop below this many samples, however slow one iteration is — a median
#: over two points is not a median.
_MIN_REPS = 5


class L2Cache:
    """Use the same Triton driver cache flush primitive as ``do_bench``."""

    def __init__(self):
        from triton import runtime

        self._driver = runtime.driver.active
        self._cache = self._driver.get_empty_cache_for_benchmark()

    def clear(self) -> None:
        self._driver.clear_cache(self._cache)


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "q20_ms": _percentile(samples, 0.2),
        "median_ms": statistics.median(samples),
        "q80_ms": _percentile(samples, 0.8),
    }


def _warm_provider(
    provider: PairProvider,
    values: dict[str, Any],
    dout: Any,
    *,
    warmup: int,
) -> None:
    for _ in range(warmup):
        _output, saved = provider.forward(values)
        provider.backward(dout, saved, values)
    torch.cuda.synchronize()


def _adaptive_reps(estimate_ms: float, reps: int, budget_ms: float) -> int:
    """How many timed iterations to run for a region taking ``estimate_ms``.

    ``do_bench`` sizes its loops by a time budget rather than a fixed count, and
    that is not only about spending measurement time evenly. A block runs
    without synchronizing — that is the point, it keeps event overhead out of
    the samples — so every iteration's intermediates stay resident until the
    block ends. ``fused_linear_cross_entropy`` allocates 3 GiB per forward at
    its largest workload; fifty of those is 150 GiB on a 95 GiB card, and the
    allocator does not fail cleanly, it thrashes. Fast kernels still get their
    full ``reps``; slow, memory-hungry ones get as many as fit the budget.
    """
    if estimate_ms <= 0:
        return reps
    return max(_MIN_REPS, min(reps, int(budget_ms / estimate_ms)))


def _estimate_ms(fn: Callable[[], Any], *, iterations: int = 3) -> float:
    """Rough cost of one call, used only to size the timed loops."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _measure_provider_block(
    provider: PairProvider,
    values: dict[str, Any],
    # A Tensor, or an ordered tuple of them for a multi-output declaration; it
    # is passed through to the provider untouched either way, so every declared
    # output is produced and differentiated inside the timed region.
    dout: Any,
    *,
    reps: int,
    clear_l2: Callable[[], None],
) -> dict[str, list[float]]:
    forward_samples = []
    backward_samples = []
    full_samples = []

    def _full_step():
        _output, saved = provider.forward(values)
        return provider.backward(dout, saved, values)

    # Size each loop from its own cost: the forward alone is cheaper than a
    # full step, and giving both the same count would either over-measure the
    # cheap one or exhaust memory on the expensive one.
    forward_reps = _adaptive_reps(
        _estimate_ms(lambda: provider.forward(values)), reps, _REP_BUDGET_MS
    )
    step_reps = _adaptive_reps(_estimate_ms(_full_step), reps, _REP_BUDGET_MS)
    torch.cuda.empty_cache()

    forward_starts = [torch.cuda.Event(enable_timing=True) for _ in range(forward_reps)]
    forward_ends = [torch.cuda.Event(enable_timing=True) for _ in range(forward_reps)]
    for start, end in zip(forward_starts, forward_ends):
        clear_l2()
        start.record()
        provider.forward(values)
        end.record()
    torch.cuda.synchronize()
    forward_samples.extend(
        float(start.elapsed_time(end))
        for start, end in zip(forward_starts, forward_ends)
    )

    backward_starts = [
        torch.cuda.Event(enable_timing=True) for _ in range(step_reps)
    ]
    backward_ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(step_reps)
    ]
    for start, end in zip(backward_starts, backward_ends):
        _output, saved = provider.forward(values)
        clear_l2()
        start.record()
        provider.backward(dout, saved, values)
        end.record()
    torch.cuda.synchronize()
    backward_samples.extend(
        float(start.elapsed_time(end))
        for start, end in zip(backward_starts, backward_ends)
    )

    full_starts = [torch.cuda.Event(enable_timing=True) for _ in range(step_reps)]
    full_ends = [torch.cuda.Event(enable_timing=True) for _ in range(step_reps)]
    for start, end in zip(full_starts, full_ends):
        clear_l2()
        start.record()
        _full_step()
        end.record()
    torch.cuda.synchronize()
    full_samples.extend(
        float(start.elapsed_time(end))
        for start, end in zip(full_starts, full_ends)
    )
    return {
        "forward": forward_samples,
        "backward": backward_samples,
        "full": full_samples,
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_fingerprint() -> dict[str, Any]:
    return {
        "gpu_name": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": _package_version("triton"),
        "liger_kernel": _package_version("liger-kernel"),
    }


def _block_bootstrap_speedup(
    cases: list[dict[str, Any]],
    candidate_name: str,
    baseline_name: str,
    metric: str,
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        candidate_total = 0.0
        baseline_total = 0.0
        for case in cases:
            per_provider = {}
            candidate_blocks = {
                block["pair_index"]: block
                for block in case["raw_blocks"][candidate_name]
            }
            baseline_blocks = {
                block["pair_index"]: block
                for block in case["raw_blocks"][baseline_name]
            }
            pair_indices = sorted(candidate_blocks.keys() & baseline_blocks.keys())
            selected_pairs = [rng.choice(pair_indices) for _ in pair_indices]
            for name, blocks_by_pair in (
                (candidate_name, candidate_blocks),
                (baseline_name, baseline_blocks),
            ):
                selected = [blocks_by_pair[pair] for pair in selected_pairs]
                block_medians = [
                    statistics.median(block["samples_ms"][metric])
                    for block in selected
                ]
                per_provider[name] = statistics.median(block_medians)
            candidate_total += per_provider[candidate_name]
            baseline_total += per_provider[baseline_name]
        estimates.append(baseline_total / candidate_total)
    return {
        "iterations": iterations,
        "low": _percentile(estimates, 0.025),
        "high": _percentile(estimates, 0.975),
    }


def benchmark_case_fair(
    op: OpDecl,
    workload: Workload,
    providers: tuple[PairProvider, PairProvider],
    *,
    warmup: int = 10,
    reps: int = 50,
    blocks: int = 3,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    canonical = make_case_inputs(op, workload, device=device)
    values_by_provider = {
        provider.name: clone_values(canonical) for provider in providers
    }
    # The exemption comes from the operator's declaration, so it applies to
    # every provider alike — a baseline cannot enjoy it while a candidate is
    # held to the stricter rule.
    may_overwrite = tuple(getattr(op, "backward_may_overwrite", ()) or ())
    snapshots = {
        provider.name: snapshot_tensors(
            values_by_provider[provider.name], may_overwrite=may_overwrite
        )
        for provider in providers
    }
    cache = L2Cache()

    saved_reports = {}
    for provider in providers:
        values = values_by_provider[provider.name]
        _output, saved = provider.forward(values)
        provider.backward(upstream_grad_values(op, values), saved, values)
        assert_tensors_unchanged(
            values, snapshots[provider.name], provider=provider.name
        )
        saved_reports[provider.name] = saved_state_report(saved, values)
        _warm_provider(
            provider,
            values,
            upstream_grad_values(op, values),
            warmup=warmup,
        )
        assert_tensors_unchanged(
            values, snapshots[provider.name], provider=provider.name
        )

    rng = random.Random(seed)
    schedule = []
    for pair_index in range(blocks):
        pair_order = [provider.name for provider in providers]
        rng.shuffle(pair_order)
        schedule.extend((pair_index, name) for name in pair_order)
    order = [name for _pair_index, name in schedule]
    by_name = {provider.name: provider for provider in providers}
    block_results = {provider.name: [] for provider in providers}
    for block_index, (pair_index, name) in enumerate(schedule):
        provider = by_name[name]
        values = values_by_provider[name]
        measured = _measure_provider_block(
            provider,
            values,
            upstream_grad_values(op, values),
            reps=reps,
            clear_l2=cache.clear,
        )
        block_results[name].append(
            {
                "block_index": block_index,
                "pair_index": pair_index,
                "samples_ms": measured,
            }
        )
        assert_tensors_unchanged(values, snapshots[name], provider=name)

    summaries = {}
    for provider in providers:
        blocks_for_provider = block_results[provider.name]
        backward = [
            sample
            for block in blocks_for_provider
            for sample in block["samples_ms"]["backward"]
        ]
        full = [
            sample
            for block in blocks_for_provider
            for sample in block["samples_ms"]["full"]
        ]
        full_summary = _summary(full)
        summaries[provider.name] = {
            "forward": _summary(
                [
                    sample
                    for block in blocks_for_provider
                    for sample in block["samples_ms"]["forward"]
                ]
            ),
            "backward": _summary(backward),
            "pair_full": full_summary,
            # Backward-compatible alias retained for existing LayerNorm reports.
            "liger_compatible_raw_fused_full": full_summary,
            "saved_state": saved_reports[provider.name],
            "source_hash": provider.source_hash,
            "adapter_kind": provider.adapter_kind,
        }

    candidate_name, baseline_name = (provider.name for provider in providers)
    return {
        "dims": dict(workload.dims),
        "dtype": workload.dtype,
        "block_order": order,
        "providers": summaries,
        "speedup": {
            "backward": (
                summaries[baseline_name]["backward"]["median_ms"]
                / summaries[candidate_name]["backward"]["median_ms"]
            ),
            "liger_compatible_raw_fused_full": (
                summaries[baseline_name]["liger_compatible_raw_fused_full"][
                    "median_ms"
                ]
                / summaries[candidate_name]["liger_compatible_raw_fused_full"][
                    "median_ms"
                ]
            ),
            "pair_full": (
                summaries[baseline_name]["pair_full"]["median_ms"]
                / summaries[candidate_name]["pair_full"]["median_ms"]
            ),
        },
        "raw_blocks": block_results,
    }


def run_fair_benchmarks(
    op: OpDecl,
    candidate: PairProvider,
    baseline: PairProvider,
    *,
    workloads: tuple[Workload, ...],
    warmup: int = 10,
    reps: int = 50,
    blocks: int = 3,
    seed: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    cases = [
        benchmark_case_fair(
            op,
            workload,
            (candidate, baseline),
            warmup=warmup,
            reps=reps,
            blocks=blocks,
            seed=seed + index,
            device=device,
        )
        for index, workload in enumerate(workloads)
    ]
    candidate_backward = sum(
        case["providers"][candidate.name]["backward"]["median_ms"] for case in cases
    )
    baseline_backward = sum(
        case["providers"][baseline.name]["backward"]["median_ms"] for case in cases
    )
    candidate_forward = sum(
        case["providers"][candidate.name]["forward"]["median_ms"] for case in cases
    )
    baseline_forward = sum(
        case["providers"][baseline.name]["forward"]["median_ms"] for case in cases
    )
    candidate_full = sum(
        case["providers"][candidate.name]["liger_compatible_raw_fused_full"][
            "median_ms"
        ]
        for case in cases
    )
    baseline_full = sum(
        case["providers"][baseline.name]["liger_compatible_raw_fused_full"][
            "median_ms"
        ]
        for case in cases
    )
    backward_ci = _block_bootstrap_speedup(
        cases,
        candidate.name,
        baseline.name,
        "backward",
        seed=seed,
    )
    forward_ci = _block_bootstrap_speedup(
        cases,
        candidate.name,
        baseline.name,
        "forward",
        seed=seed + 2,
    )
    full_ci = _block_bootstrap_speedup(
        cases,
        candidate.name,
        baseline.name,
        "full",
        seed=seed + 1,
    )
    return {
        "protocol": PROTOCOL_VERSION,
        "baseline_timing_cache": {"enabled": False, "reason": "final protocol"},
        "timing_protocol": {
            "provider_contract": (
                "forward(values)->(output,saved); "
                "backward(output_grads,saved,values); output and output_grads "
                "are ordered tuples for a multi-output declaration"
            ),
            "primary_full_metric": "pair_full",
            "l2_policy": "clear_before_each_timed_region",
            "forward_backward_cache": "no flush between forward and backward",
            "event_scheduling": "batched events with one synchronize per block",
            "provider_order": "paired randomized blocks",
            "mutation_check": "content and tensor metadata outside timed regions",
        },
        "environment": environment_fingerprint(),
        "aggregate": {
            "candidate_forward_ms": candidate_forward,
            "baseline_forward_ms": baseline_forward,
            "speedup_forward": baseline_forward / candidate_forward,
            "speedup_forward_ci95": forward_ci,
            "candidate_backward_ms": candidate_backward,
            "baseline_backward_ms": baseline_backward,
            "speedup_backward": baseline_backward / candidate_backward,
            "speedup_backward_ci95": backward_ci,
            "candidate_liger_compatible_raw_fused_full_ms": candidate_full,
            "baseline_liger_compatible_raw_fused_full_ms": baseline_full,
            "speedup_liger_compatible_raw_fused_full": baseline_full
            / candidate_full,
            "speedup_liger_compatible_raw_fused_full_ci95": full_ci,
            "candidate_pair_full_ms": candidate_full,
            "baseline_pair_full_ms": baseline_full,
            "speedup_pair_full": baseline_full / candidate_full,
            "speedup_pair_full_ci95": full_ci,
        },
        "cases": cases,
    }
