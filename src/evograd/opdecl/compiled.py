"""``torch.compile`` as a performance baseline, generic over any declaration.

The reviewed Liger baselines are per-operator: each declaration supplies a
factory returning a hand-written forward/backward pair. This one needs no
declaration support at all — the activity annotations already say which args
carry gradients, and the declared forward reference is the thing to compile —
so it is available for every operator as a built-in baseline name.

Why it matters: ``pytorch_autograd`` measures *eager* PyTorch. A Pipeline D
seed is captured from AOTAutograd + Inductor, i.e. it is roughly what
``torch.compile`` would have produced, so beating eager is expected and says
little. ``torch_compile`` is the baseline that actually asks whether an evolved
kernel is worth anything.

Timing splits the same way the candidate is measured (see
``evograd.bench.harness.benchmark_case``): the forward builds the autograd graph
in the *untimed* setup, and only ``torch.autograd.grad`` sits in the timed
region. The compiled backward is produced lazily on the first backward call, so
it is compiled during warmup, not inside a measured rep.
"""

from __future__ import annotations

from typing import Any, Callable

from evograd.opdecl.activity import Active, OpDecl

# name -> torch.compile(mode=...) argument
BUILTIN_MODES: dict[str, str | None] = {
    "torch_compile": None,
    "torch_compile_max_autotune": "max-autotune",
}


def _leaf_args(op: OpDecl, inputs: dict) -> tuple[list[Any], list[tuple[str, Any]]]:
    """Positional args with fresh differentiable leaves for every Active arg.

    Mirrors ``oracle`` and the harness's ``baseline_full_step``: a leaf must be
    a fresh clone per call, otherwise the second backward accumulates into a
    ``.grad`` that is already populated.
    """
    positional: list[Any] = []
    leaves: list[tuple[str, Any]] = []
    for arg in op.args:
        value = inputs.get(arg.name, getattr(arg, "default", None))
        if isinstance(arg, Active):
            if value is None:
                raise ValueError(f"{op.name}: missing input {arg.name!r}")
            value = value.detach().clone().requires_grad_(True)
            leaves.append((arg.grad_name, value))
        positional.append(value)
    return positional, leaves


def compile_forward(op: OpDecl, mode: str | None, dynamic: bool | None):
    import torch

    from evograd.opdecl.oracle import resolve_forward

    return torch.compile(resolve_forward(op), mode=mode, dynamic=dynamic)


def reference_run(op: OpDecl, inputs: dict, *, mode: str | None, dynamic: bool | None):
    """Run the compiled forward/backward once and return ``(y, {grad_name: g})``.

    Same shape as :func:`evograd.opdecl.oracle.oracle`, so the verification gate
    can compare a compiled baseline against the eager oracle before trusting its
    timings — a miscompile that silently changes numerics would otherwise show
    up only as a suspiciously good baseline.
    """
    import torch

    compiled = compile_forward(op, mode, dynamic)
    positional, leaves = _leaf_args(op, inputs)
    y = compiled(*positional)
    grads = torch.autograd.grad(
        y, [leaf for _, leaf in leaves], grad_outputs=inputs[op.upstream_grad_name]
    )
    by_name = {name: g for (name, _), g in zip(leaves, grads)}
    return y, tuple(by_name[name] for name in op.grad_names())


def make_compiled_baseline(
    op: OpDecl, *, mode: str | None = None, dynamic: bool | None = False
) -> Callable:
    """Build a timing hook matching the ``performance_baselines`` protocol.

    ``dynamic=False`` compiles a shape specialist per workload, which is the
    strongest form of the baseline: each benchmark case gets its own compiled
    callable, so no recompilation churn lands in the timings.
    """

    def measure(torch, inputs, dout, warmup, reps, median_ms, median_ms_timed_region):
        compiled = compile_forward(op, mode, dynamic)

        def setup():
            positional, leaves = _leaf_args(op, inputs)
            y = compiled(*positional)
            return y, [leaf for _, leaf in leaves]

        def backward(state):
            y, leaves = state
            return torch.autograd.grad(y, leaves, grad_outputs=dout)

        def full_step():
            return backward(setup())

        backward_ms = median_ms_timed_region(setup, backward, warmup, reps)
        full_ms = median_ms(full_step, warmup, reps)
        return backward_ms, full_ms

    def available() -> bool:
        try:
            import torch
        except ImportError:
            return False
        # Probing without compiling: a real compile costs minutes and would run
        # at declaration-resolution time, long before any benchmark starts.
        return hasattr(torch, "compile")

    measure.__name__ = f"measure_torch_compile_{mode or 'default'}"
    measure.available = available
    measure.compiled_mode = mode
    measure.compiled_dynamic = dynamic
    measure.reference_run = lambda values: reference_run(
        op, values, mode=mode, dynamic=dynamic
    )
    return measure


def builtin_baseline(op: OpDecl, name: str) -> Callable | None:
    """Return the hook for a built-in baseline name, or None if not built in."""
    if name not in BUILTIN_MODES:
        return None
    return make_compiled_baseline(op, mode=BUILTIN_MODES[name])
