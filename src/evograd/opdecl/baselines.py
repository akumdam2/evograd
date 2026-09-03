"""Performance-baseline selection independent of the GPU benchmark runtime."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace

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


def baseline_hook(op: OpDecl, name: str):
    """The timing hook for a resolved baseline name, built-in or declared.

    ``pytorch_autograd`` has no hook — the harness measures the eager oracle
    directly — so it resolves to None.
    """
    if name == "pytorch_autograd":
        return None
    from evograd.opdecl.compiled import builtin_baseline

    return builtin_baseline(op, name) or op.performance_baselines[name]


_RUNTIME_FORWARD_VERIFIED: set[str] = set()


def verify_runtime_forward(op: OpDecl, *, device: str = "cuda") -> None:
    """A declared ``runtime_forward`` must compute what ``forward`` computes.

    The eager baseline is timed through ``runtime_forward`` and checked for
    correctness through ``forward``. If the two ever disagreed, the benchmark
    would be comparing the runtime of one function against the correctness of
    another, and the speedups would be meaningless in a way no other check
    catches — a faster-but-different baseline looks exactly like a faster
    baseline. Verified once per operator, on the correctness workloads.
    """
    if not op.runtime_forward or op.name in _RUNTIME_FORWARD_VERIFIED:
        return

    import torch

    from evograd.opdecl.inputs import make_case_inputs
    from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

    definition = resolve_forward(op)
    production = resolve_runtime_forward(op)
    arg_names = tuple(arg.name for arg in op.args)

    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        args = [values.get(name, getattr(arg, "default", None))
                for name, arg in zip(arg_names, op.args)]
        from evograd.opdecl.inputs import as_output_tuple

        expected_outputs = as_output_tuple(op, definition(*args))
        actual_outputs = as_output_tuple(op, production(*args))
        for out, actual, expected in zip(op.outputs, actual_outputs, expected_outputs):
            atol, rtol = op.tolerance_for(workload, out.name)
            if (
                actual.shape != expected.shape
                or actual.dtype != expected.dtype
                or actual.stride() != expected.stride()
                or not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
            ):
                raise RuntimeError(
                    f"{op.name}: runtime_forward disagrees with forward on output "
                    f"{out.name!r} at {workload.dims}/{workload.dtype}; the eager "
                    "baseline would be timing a different function than the one it "
                    "is verified against"
                )
    _RUNTIME_FORWARD_VERIFIED.add(op.name)


def baseline_candidate_module(op: OpDecl, baseline: str):
    """Present a reviewed pair baseline as if it were a candidate seed module.

    The suite measures candidates, so without this the benchmark cannot report a
    number until something has been generated for all 25 operators. A reviewed
    baseline is already an autograd pair that has been verified against the
    oracle, which makes it the one honest stand-in: running `liger` as the
    candidate against `pytorch_autograd` as the baseline produces the suite's
    reference line — what a hand-written production kernel achieves over eager
    PyTorch — rather than a placeholder.

    Only pair baselines qualify. Compiled baselines (`torch_compile`) time a
    fused forward+backward and never expose the saved state that the candidate
    contract is defined around, so they remain baselines only.
    """
    try:
        hook = baseline_hook(op, baseline)
    except KeyError:
        raise ValueError(
            f"{op.name} declares no {baseline!r} baseline "
            f"(has: {sorted(op.performance_baselines) or 'none'})"
        ) from None
    if hook is None:
        raise ValueError(
            f"{op.name}: 'pytorch_autograd' is the eager oracle itself, not a "
            "candidate; it would be timed against itself."
        )
    factory = getattr(hook, "pair_factory", None)
    if factory is None:
        raise ValueError(
            f"{op.name}: baseline {baseline!r} is not an autograd pair, so it "
            "cannot stand in as a candidate. Compiled baselines time a fused "
            "forward+backward and never expose saved state; use them via "
            "--baseline instead."
        )

    forward, backward = factory()
    forward_args = tuple(getattr(hook, "forward_args", ()))
    backward_extras = tuple(getattr(hook, "backward_extras", ()))
    arg_names = tuple(arg.name for arg in op.args)

    # `backward_extras` are declaration inputs this baseline's backward needs
    # but the candidate contract does not hand it — a candidate backward
    # receives only dout, the saved state, and scalar Inactive args. Capture
    # them on the forward rather than appending them to `saved`: anything in
    # `saved` is counted as retained state and would inflate this baseline's
    # saved-memory ratio, which is one of the numbers being reported. The
    # harness always calls the forward before the backward for a given case,
    # single-threaded, including when it re-times the backward from pre-saved
    # state.
    captured: dict[str, tuple] = {"extras": ()}

    def forward_with_saved(*args):
        values = dict(zip(arg_names, args))
        captured["extras"] = tuple(values[name] for name in backward_extras)
        return forward(*(values[name] for name in forward_args))

    def backward_from_saved(dout, saved):
        return backward(dout, saved, *captured["extras"])

    module = SimpleNamespace(
        forward_with_saved=forward_with_saved,
        backward_from_saved=backward_from_saved,
    )
    # bind.lookup_pair reports this in its error messages.
    module.__name__ = f"<{baseline} baseline as candidate for {op.name}>"
    return module


def available_baselines(op: OpDecl) -> list[str]:
    from evograd.opdecl.compiled import BUILTIN_MODES

    return [
        "auto",
        "pytorch_autograd",
        *sorted(BUILTIN_MODES),
        *sorted(op.performance_baselines),
    ]


def resolve_performance_baseline(op: OpDecl, requested: str) -> str:
    """Resolve ``auto`` without silently downgrading an explicit baseline."""
    from evograd.opdecl.compiled import BUILTIN_MODES

    if requested == "auto":
        hook = op.performance_baselines.get("liger")
        if hook is not None:
            probe = getattr(hook, "available", None)
            if probe is None or probe():
                return "liger"
        # torch_compile is never chosen by auto: it costs minutes of compilation
        # and is a deliberate comparison, not a default.
        return "pytorch_autograd"
    if (
        requested != "pytorch_autograd"
        and requested not in BUILTIN_MODES
        and requested not in op.performance_baselines
    ):
        raise KeyError(
            f"{op.name}: unknown performance baseline {requested!r}; "
            f"available: {available_baselines(op)}"
        )
    if requested != "pytorch_autograd":
        probe = getattr(baseline_hook(op, requested), "available", None)
        if probe is not None and not probe():
            raise RuntimeError(
                f"{op.name}: {requested} baseline was explicitly requested but "
                "its implementation is unavailable"
            )
    return requested


def verify_performance_baseline(
    op: OpDecl, baseline: str, *, device: str = "cuda"
) -> None:
    """Verify a baseline against the autograd oracle before trusting its timings.

    Timing hooks produced by ``make_pair_baseline`` carry the underlying pair
    factory and argument routing as metadata; built-in compiled baselines carry
    a ``reference_run`` instead. Custom hooks with neither are assumed to have
    their own review gate.
    """
    if baseline == "pytorch_autograd":
        return
    hook = baseline_hook(op, baseline)
    factory = getattr(hook, "pair_factory", None)
    reference = getattr(hook, "reference_run", None)
    if factory is None and reference is None:
        return

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
    key = (op.name, baseline, gpu)
    cache_path = os.environ.get("EVOGRAD_BASELINE_TIMING_CACHE_PATH")
    persistent_key = _verification_cache_key(op, baseline, gpu)
    if key in _VERIFIED or _cached(cache_path, persistent_key):
        _VERIFIED.add(key)
        return

    from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values
    from evograd.opdecl.oracle import oracle

    if factory is not None:
        forward, backward = factory()
        forward_args = tuple(getattr(hook, "forward_args", ()))
        backward_extras = tuple(getattr(hook, "backward_extras", ()))

        def run(values):
            y, saved = forward(*(values[name] for name in forward_args))
            saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
            grads = backward(
                upstream_grad_values(op, values),
                saved,
                *(values[name] for name in backward_extras),
            )
            return y, grads
    else:
        run = reference

    for workload in op.correctness:
        values = make_case_inputs(op, workload, device=device)
        y_ref, expected = oracle(op, values)
        y, actual = run(values)
        actual = (actual,) if torch.is_tensor(actual) else tuple(actual)
        if len(actual) != len(op.grad_names()):
            raise RuntimeError(
                f"{op.name}: {baseline} baseline returned {len(actual)} gradients; "
                f"expected {len(op.grad_names())}"
            )
        from evograd.opdecl.inputs import as_output_tuple

        for out, got, ref in zip(
            op.outputs, as_output_tuple(op, y), as_output_tuple(op, y_ref)
        ):
            atol, rtol = op.tolerance_for(workload, out.name)
            if (
                got.shape != ref.shape
                or got.dtype != ref.dtype
                or not torch.allclose(got.float(), ref.float(), atol=atol, rtol=rtol)
            ):
                raise RuntimeError(
                    f"{op.name}: {baseline} baseline forward failed on output "
                    f"{out.name!r} at {workload.dims}/{workload.dtype}"
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
