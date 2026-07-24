"""Performance-baseline selection independent of the GPU benchmark runtime."""

from __future__ import annotations

import json
import os
import tempfile

from evograd.opdecl.activity import OpDecl

_VERIFIED: set[tuple[str, str, str]] = set()


def _verification_cache_key(op: OpDecl, baseline: str, gpu: str) -> str:
    return "__baseline_verified__:" + json.dumps(
        {
            "op": op.name,
            "forward": op.forward,
            "baseline": baseline,
            "gpu": gpu,
            "correctness": [
                {"dims": case.dims, "dtype": case.dtype}
                for case in op.correctness
            ],
        },
        sort_keys=True,
    )


def _cached(path: str | None, key: str) -> bool:
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            return bool(json.load(handle).get(key))
    except Exception:
        return False


def _mark_cached(path: str | None, key: str) -> None:
    if not path:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    data[key] = True
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".evograd_baseline_verify_", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def resolve_performance_baseline(op: OpDecl, requested: str) -> str:
    """Resolve ``auto`` without silently downgrading an explicit baseline."""
    if requested == "auto":
        hook = op.performance_baselines.get("liger")
        if hook is not None:
            probe = getattr(hook, "available", None)
            if probe is None or probe():
                return "liger"
        return "pytorch_autograd"
    if requested != "pytorch_autograd" and requested not in op.performance_baselines:
        available = ["auto", "pytorch_autograd", *sorted(op.performance_baselines)]
        raise KeyError(
            f"{op.name}: unknown performance baseline {requested!r}; "
            f"available: {available}"
        )
    if requested != "pytorch_autograd":
        probe = getattr(op.performance_baselines[requested], "available", None)
        if probe is not None and not probe():
            raise RuntimeError(
                f"{op.name}: {requested} baseline was explicitly requested but "
                "its implementation is unavailable"
            )
    return requested


def verify_performance_baseline(
    op: OpDecl, baseline: str, *, device: str = "cuda"
) -> None:
    """Verify a declaration-local pair baseline against the autograd oracle.

    Timing hooks produced by ``make_pair_baseline`` carry the underlying pair
    factory and argument routing as metadata. Custom hooks without that
    metadata are assumed to have their own review gate.
    """
    if baseline == "pytorch_autograd":
        return
    hook = op.performance_baselines[baseline]
    factory = getattr(hook, "pair_factory", None)
    if factory is None:
        return

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
    key = (op.name, baseline, gpu)
    cache_path = os.environ.get("EVOGRAD_BASELINE_TIMING_CACHE_PATH")
    persistent_key = _verification_cache_key(op, baseline, gpu)
    if key in _VERIFIED or _cached(cache_path, persistent_key):
        _VERIFIED.add(key)
        return

    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import oracle

    forward, backward = factory()
    forward_args = tuple(getattr(hook, "forward_args", ()))
    backward_extras = tuple(getattr(hook, "backward_extras", ()))
    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        y_ref, expected = oracle(op, values)
        y, saved = forward(*(values[name] for name in forward_args))
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        actual = backward(
            values[op.upstream_grad_name],
            saved,
            *(values[name] for name in backward_extras),
        )
        actual = (actual,) if torch.is_tensor(actual) else tuple(actual)
        if len(actual) != len(op.grad_names()):
            raise RuntimeError(
                f"{op.name}: {baseline} baseline returned {len(actual)} gradients; "
                f"expected {len(op.grad_names())}"
            )
        atol, rtol = op.tolerance_for(workload)
        if (
            y.shape != y_ref.shape
            or y.dtype != y_ref.dtype
            or not torch.allclose(y.float(), y_ref.float(), atol=atol, rtol=rtol)
        ):
            raise RuntimeError(
                f"{op.name}: {baseline} baseline forward failed at "
                f"{workload.dims}/{workload.dtype}"
            )
        for name, got in zip(op.grad_names(), actual):
            ref = expected[name]
            atol, rtol = op.tolerance_for(workload, name)
            if (
                got.shape != ref.shape
                or got.dtype != ref.dtype
                or not torch.allclose(
                    got.float(), ref.float(), atol=atol, rtol=rtol
                )
            ):
                raise RuntimeError(
                    f"{op.name}: {baseline} baseline {name} failed at "
                    f"{workload.dims}/{workload.dtype}"
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _VERIFIED.add(key)
    _mark_cached(cache_path, persistent_key)
