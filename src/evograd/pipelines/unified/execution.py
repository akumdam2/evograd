"""Check that preserved library calls execute on the public training path."""

from __future__ import annotations

from functools import wraps

from evograd.opdecl.activity import OpDecl


def validate_external_execution(
    op: OpDecl, module, contract: dict, *, device: str = "cuda"
) -> None:
    """Exercise each correctness case with counted, restored library wrappers.

    Source validation pins the providers and their call sites. This check also
    rejects a candidate that retains those calls in unused old helpers while
    redirecting its public forward/backward to a replacement implementation.
    It runs outside benchmark timing and does not alter the candidate source.
    """
    regions = contract["external_regions"]
    if not regions:
        return

    import torch

    from evograd.opdecl.bind import (
        backward_inactive_kwargs,
        lookup_pair,
    )
    from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values

    if not op.correctness:
        raise ValueError("cannot verify preserved library execution without correctness cases")
    expected: dict[str, str] = {}
    originals = {}
    for region in regions:
        name, side = region["wrapper"], region["side"]
        if side not in {"forward", "backward"} or name in expected:
            raise ValueError(f"invalid preserved external region: {name!r}, {side!r}")
        original = getattr(module, name, None)
        if not callable(original):
            raise ValueError(f"preserved external wrapper is missing: {name}")
        expected[name] = side
        originals[name] = original

    phase = "forward"
    counts = {name: 0 for name in expected}

    def counted(name, original):
        @wraps(original)
        def invoke(*args, **kwargs):
            counts[name] += 1
            if expected[name] != phase:
                raise ValueError(
                    f"{phase} invoked preserved {expected[name]} external region {name}"
                )
            return original(*args, **kwargs)

        return invoke

    def check_phase() -> None:
        for name, side in expected.items():
            required = int(side == phase)
            if counts[name] != required:
                raise ValueError(
                    f"{phase} must execute preserved external region {name} "
                    f"{required} time(s), observed {counts[name]}"
                )

    forward, backward = lookup_pair(op, module)
    try:
        for name, original in originals.items():
            setattr(module, name, counted(name, original))
        for workload in op.correctness:
            inputs = make_case_inputs(op, workload, device=device)
            positional = [inputs.get(arg.name, getattr(arg, "default", None)) for arg in op.args]
            with torch.no_grad():
                phase = "forward"
                counts = {name: 0 for name in expected}
                _, saved = forward(*positional)
                check_phase()
                phase = "backward"
                counts = {name: 0 for name in expected}
                backward(
                    upstream_grad_values(op, inputs),
                    saved,
                    **backward_inactive_kwargs(op, backward, inputs),
                )
                check_phase()
    finally:
        for name, original in originals.items():
            setattr(module, name, original)
