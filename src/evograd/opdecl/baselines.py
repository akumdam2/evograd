"""Performance-baseline selection independent of the GPU benchmark runtime."""

from __future__ import annotations

from types import SimpleNamespace

from evograd.opdecl.activity import OpDecl


def baseline_hook(op: OpDecl, name: str):
    """The timing hook for a resolved baseline name, built-in or declared.

    ``pytorch_autograd`` has no hook — the harness measures the eager oracle
    directly — so it resolves to None.
    """
    if name == "pytorch_autograd":
        return None
    from evograd.opdecl.compiled import builtin_baseline

    return builtin_baseline(op, name) or op.performance_baselines[name]


def baseline_candidate_module(op: OpDecl, baseline: str):
    """Present a reviewed pair baseline as if it were a candidate seed module.

    The suite measures candidates, so without this the benchmark cannot report a
    number until something has been generated for every operator. A reviewed
    baseline is already an autograd pair, checked against eager on a recorded
    call before it is timed (``evaluation.common.provider.check_baseline``),
    which makes it the one honest stand-in: running `megafold` or
    `fla` as the candidate against `pytorch_autograd` as the baseline produces
    a reference line — what a hand-written production kernel achieves over eager
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

    sidecar = getattr(hook, "sidecar", None)
    if sidecar is None:
        forward, backward = factory()
    else:
        # A sidecar library (FLA) may be imported only in the child measuring
        # this baseline, so the pair is built on first use rather than here:
        # a parent assembling providers never touches it.
        pair: list = []

        def _pair():
            if not pair:
                pair.extend(factory())
            return pair

        def forward(*args):
            return _pair()[0](*args)

        def backward(*args):
            return _pair()[1](*args)
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
    # Which sidecar libraries the process running this module must be allowed.
    module.sidecars = (sidecar,) if sidecar else ()
    return module


def kernel_sidecars(kernels) -> tuple[str, ...]:
    """The sidecar libraries a Tier-3 kernel set's sources need, sorted."""
    names = set()
    for source in getattr(kernels, "sources", ()):
        names.update(getattr(getattr(source, "module", None), "sidecars", ()) or ())
    return tuple(sorted(names))


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
