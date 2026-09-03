"""A synthetic multi-output declaration, used only by the structured-output tests.

Deliberately heterogeneous: ``hi`` is ``[R, C]`` and ``lo`` is ``[R]``, so a
framework path that silently used one output's shape or tolerance for the other
fails rather than passing by coincidence.
"""

from __future__ import annotations

import torch

from evograd.opdecl import Active, Inactive, Workload, declare_op


def split_scale_forward_ref(x, w, alpha=2.0):
    """The declared reference, spelled in primitives."""
    scaled = x * w
    return scaled * alpha, scaled.sum(-1)


def split_scale_forward_production(x, w, alpha=2.0):
    """The same mathematics, spelled the way a user would write it."""
    return torch.mul(x * w, alpha), torch.einsum("rc,c->r", x, w)


def make_split_scale_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(dims["R"] * 1009 + dims["C"])
    return {
        "x": torch.randn((dims["R"], dims["C"]), device=device, dtype=dtype),
        "w": torch.randn((dims["C"],), device=device, dtype=dtype),
        "dhi": torch.randn((dims["R"], dims["C"]), device=device, dtype=dtype),
        "dlo": torch.randn((dims["R"],), device=device, dtype=dtype),
    }


op = declare_op(
    name="split_scale",
    forward="tests._structured_fixture:split_scale_forward_ref",
    runtime_forward="tests._structured_fixture:split_scale_forward_production",
    dims=("R", "C"),
    args=(
        Active("x", "[R, C]"),
        Active("w", "[C]"),
        Inactive("alpha", None, default=2.0),
    ),
    output=(Active("hi", "[R, C]"), Active("lo", "[R]")),
    forward_semantics="hi = x * w * alpha; lo = sum(x * w, dim=-1).",
    backward_semantics="Return dx then dw.",
    correctness=(
        Workload(dims={"R": 4, "C": 8}, dtype="float32"),
        Workload(dims={"R": 7, "C": 5}, dtype="float32"),
    ),
    tolerances={"float32": (1e-5, 1e-5)},
    make_inputs=make_split_scale_inputs,
)


# ── candidate pairs ──────────────────────────────────────────────────────


def split_scale_forward_with_saved(x, w, alpha=2.0):
    scaled = x * w
    return (scaled * alpha, scaled.sum(-1)), (x, w)


def split_scale_backward_from_saved(output_grads, saved_tensors, alpha=2.0):
    dhi, dlo = output_grads
    x, w = saved_tensors
    dx = dhi * w * alpha + dlo.unsqueeze(-1) * w
    dw = (dhi * x * alpha).sum(0) + (dlo.unsqueeze(-1) * x).sum(0)
    return dx, dw


class _Module:
    """A stand-in for a candidate module; ``lookup_pair`` reads attributes."""

    def __init__(self, forward, backward):
        setattr(self, op.forward_fn_name, forward)
        setattr(self, op.backward_fn_name, backward)


def good_module():
    return _Module(split_scale_forward_with_saved, split_scale_backward_from_saved)


def single_output_module():
    """Returns a Tensor where the contract requires a tuple."""

    def forward(x, w, alpha=2.0):
        return x * w * alpha, (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def wrong_arity_module():
    """Returns three outputs where the contract declares two."""

    def forward(x, w, alpha=2.0):
        scaled = x * w
        return (scaled * alpha, scaled.sum(-1), scaled), (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def swapped_outputs_module():
    """Returns the outputs in the wrong order."""

    def forward(x, w, alpha=2.0):
        scaled = x * w
        return (scaled.sum(-1), scaled * alpha), (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def wrong_dtype_module(which: int):
    def forward(x, w, alpha=2.0):
        scaled = x * w
        outs = [scaled * alpha, scaled.sum(-1)]
        outs[which] = outs[which].to(torch.float64)
        return tuple(outs), (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def wrong_shape_module(which: int):
    def forward(x, w, alpha=2.0):
        scaled = x * w
        outs = [scaled * alpha, scaled.sum(-1)]
        outs[which] = outs[which][:-1]
        return tuple(outs), (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def wrong_second_output_module():
    """Correct ``hi``, wrong ``lo`` -- the case a single fused check misses."""

    def forward(x, w, alpha=2.0):
        scaled = x * w
        return (scaled * alpha, scaled.sum(-1) + 1.0), (x, w)

    return _Module(forward, split_scale_backward_from_saved)


def wrong_gradient_module():
    def backward(output_grads, saved_tensors, alpha=2.0):
        dx, dw = split_scale_backward_from_saved(output_grads, saved_tensors, alpha)
        return dx, dw + 1.0

    return _Module(split_scale_forward_with_saved, backward)


def ignores_second_grad_module():
    """A backward that uses only the first upstream gradient.

    The failure structured outputs exist to catch: it is right about ``hi`` and
    silently wrong about everything ``lo`` contributes.
    """

    def backward(output_grads, saved_tensors, alpha=2.0):
        dhi, _dlo = output_grads
        x, w = saved_tensors
        return dhi * w * alpha, (dhi * x * alpha).sum(0)

    return _Module(split_scale_forward_with_saved, backward)


def non_contiguous_output_module():
    def forward(x, w, alpha=2.0):
        scaled = x * w
        hi = (scaled * alpha).t().contiguous().t()
        return (hi, scaled.sum(-1)), (x, w)

    return _Module(forward, split_scale_backward_from_saved)
