"""Versioned final-report protocol with symmetric provider routing.

This is deliberately separate from the low-overhead evolution harness: final
reports remeasure both providers, clear L2 at each timed step entry, retain raw
samples, and reject input mutation.
"""

from __future__ import annotations

import hashlib
import inspect
import random
import statistics
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable

import torch

from evograd.bench.harness import describe_saved, normalize_saved
from evograd.opdecl.activity import OpDecl, Workload
from evograd.opdecl.bind import backward_inactive_kwargs, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import resolve_forward

PROTOCOL_VERSION = "evograd-final-runtime-v1"


@dataclass(frozen=True)
class PairProvider:
    """A thin provider boundary consumed identically by the common runner."""

    name: str
    forward: Callable[[dict[str, Any]], tuple[Any, tuple[Any, ...]]]
    backward: Callable[[torch.Tensor, tuple[Any, ...], dict[str, Any]], Any]
    source_hash: str
    adapter_kind: str


@dataclass(frozen=True)
class TensorSnapshot:
    name: str
    value: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    storage_offset: int


def _callable_hash(*functions: Callable) -> str:
    digest = hashlib.sha256()
    for function in functions:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = repr(function)
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def candidate_provider(op: OpDecl, module) -> PairProvider:
    forward_fn, backward_fn = lookup_pair(op, module)
    arg_names = tuple(arg.name for arg in op.args)

    def forward(values):
        output, saved = forward_fn(*(values[name] for name in arg_names))
        return output, normalize_saved(saved)

    def backward(dout, saved, values):
        kwargs = backward_inactive_kwargs(op, backward_fn, values)
        return backward_fn(dout, saved, **kwargs)

    return PairProvider(
        name="candidate",
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_fn, backward_fn),
        adapter_kind="candidate_pair",
    )


def liger_provider(op: OpDecl) -> PairProvider:
    try:
        hook = op.performance_baselines["liger"]
    except KeyError:
        raise ValueError(f"{op.name}: no Liger pair baseline is declared") from None
    factory = getattr(hook, "pair_factory", None)
    if factory is None:
        raise ValueError(f"{op.name}: Liger baseline does not expose a pair factory")
    forward_fn, backward_fn = factory()
    forward_args = tuple(getattr(hook, "forward_args", ()))
    backward_extras = tuple(getattr(hook, "backward_extras", ()))

    def forward(values):
        output, saved = forward_fn(*(values[name] for name in forward_args))
        return output, normalize_saved(saved)

    def backward(dout, saved, values):
        return backward_fn(
            dout,
            saved,
            *(values[name] for name in backward_extras),
        )

    return PairProvider(
        name="liger",
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_fn, backward_fn),
        adapter_kind="stock_liger_pair" if op.name == "layernorm" else "declared_liger_pair",
    )


def pytorch_autograd_provider(op: OpDecl) -> PairProvider:
    """Expose eager PyTorch autograd through the common provider boundary."""
    forward_ref = resolve_forward(op)
    arg_names = tuple(arg.name for arg in op.args)
    active_args = tuple(op.active_args())
    active_names = tuple(arg.name for arg in active_args)
    by_grad = {arg.grad_name: arg.name for arg in active_args}

    def forward(values):
        for name in active_names:
            value = values[name]
            if not value.requires_grad:
                value.requires_grad_(True)
        output = forward_ref(*(values[name] for name in arg_names))
        return output, (output,)

    def backward(dout, saved, values):
        (output,) = saved
        gradients = torch.autograd.grad(
            output,
            tuple(values[name] for name in active_names),
            dout,
            retain_graph=False,
            create_graph=False,
        )
        by_name = dict(zip(active_names, gradients))
        return tuple(by_name[by_grad[name]] for name in op.grad_names())

    return PairProvider(
        name="pytorch_autograd",
        forward=forward,
        backward=backward,
        source_hash=_callable_hash(forward_ref, forward, backward),
        adapter_kind="pytorch_eager_autograd",
    )


def renamed_provider(provider: PairProvider, name: str) -> PairProvider:
    """Use the exact same callables under another label for identity controls."""
    return PairProvider(
        name=name,
        forward=provider.forward,
        backward=provider.backward,
        source_hash=provider.source_hash,
        adapter_kind=provider.adapter_kind,
    )


def clone_values(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }


def snapshot_tensors(values: dict[str, Any]) -> tuple[TensorSnapshot, ...]:
    return tuple(
        TensorSnapshot(
            name=name,
            value=value.detach().clone(),
            shape=tuple(value.shape),
            stride=tuple(value.stride()),
            dtype=value.dtype,
            storage_offset=value.storage_offset(),
        )
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    )


def assert_tensors_unchanged(
    values: dict[str, Any],
    snapshots: tuple[TensorSnapshot, ...],
    *,
    provider: str,
) -> None:
    for snapshot in snapshots:
        current = values[snapshot.name]
        metadata_ok = (
            tuple(current.shape) == snapshot.shape
            and tuple(current.stride()) == snapshot.stride
            and current.dtype == snapshot.dtype
            and current.storage_offset() == snapshot.storage_offset
        )
        if not metadata_ok or not torch.equal(current, snapshot.value):
            raise RuntimeError(
                f"{provider}: mutated benchmark input {snapshot.name!r}"
            )


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


def _saved_state_report(
    saved: tuple[Any, ...], values: dict[str, Any]
) -> dict[str, Any]:
    input_storages = {
        value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
        for value in values.values()
        if isinstance(value, torch.Tensor)
    }
    unique_storages: dict[int, int] = {}
    retained_input_storages: dict[int, int] = {}
    logical_bytes = 0
    for value in saved:
        if not isinstance(value, torch.Tensor):
            continue
        logical_bytes += value.numel() * value.element_size()
        storage = value.untyped_storage()
        pointer = storage.data_ptr()
        unique_storages[pointer] = storage.nbytes()
        if pointer in input_storages:
            retained_input_storages[pointer] = input_storages[pointer]
    return {
        "logical_saved_bytes": logical_bytes,
        "unique_saved_storage_bytes": sum(unique_storages.values()),
        "retained_input_bytes": sum(retained_input_storages.values()),
        "saved_tensors": describe_saved(saved),
    }


def _warm_provider(
    provider: PairProvider,
    values: dict[str, Any],
    dout: torch.Tensor,
    *,
    warmup: int,
) -> None:
    for _ in range(warmup):
        _output, saved = provider.forward(values)
        provider.backward(dout, saved, values)
    torch.cuda.synchronize()


def _measure_provider_block(
    provider: PairProvider,
    values: dict[str, Any],
    dout: torch.Tensor,
    *,
    reps: int,
    clear_l2: Callable[[], None],
) -> dict[str, list[float]]:
    forward_samples = []
    backward_samples = []
    full_samples = []
    forward_starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    forward_ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
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
        torch.cuda.Event(enable_timing=True) for _ in range(reps)
    ]
    backward_ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(reps)
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

    full_starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    full_ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for start, end in zip(full_starts, full_ends):
        def full_step():
            _output, saved = provider.forward(values)
            return provider.backward(dout, saved, values)

        clear_l2()
        start.record()
        full_step()
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
    snapshots = {
        provider.name: snapshot_tensors(values_by_provider[provider.name])
        for provider in providers
    }
    cache = L2Cache()

    saved_reports = {}
    for provider in providers:
        values = values_by_provider[provider.name]
        _output, saved = provider.forward(values)
        provider.backward(values[op.upstream_grad_name], saved, values)
        assert_tensors_unchanged(
            values, snapshots[provider.name], provider=provider.name
        )
        saved_reports[provider.name] = _saved_state_report(saved, values)
        _warm_provider(
            provider,
            values,
            values[op.upstream_grad_name],
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
            values[op.upstream_grad_name],
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
            "provider_contract": "forward(values)->(output,saved); backward(dout,saved,values)",
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
