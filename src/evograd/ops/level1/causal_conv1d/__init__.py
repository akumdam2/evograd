"""Operator declaration: depthwise causal 1-D convolution, optional SiLU.

The short convolution a Gated DeltaNet runs over its concatenated q|k|v
projections before the delta rule. Generic: any channel count, any kernel
width, any sequence length.
"""

import math

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline


_DIMS = ("B", "C", "T", "K")


def make_causal_conv1d_inputs(torch, op, workload, device="cuda"):
    dims = workload.dims
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(dims["B"] * 100003 + dims["C"] * 1009 + dims["T"] * 13 + dims["K"])
    x = torch.randn((dims["B"], dims["C"], dims["T"]), device=device, dtype=dtype)
    weight = (
        torch.randn((dims["C"], dims["K"]), device=device, dtype=torch.float32)
        * dims["K"] ** -0.5
    ).to(dtype)
    dy = torch.randn((dims["B"], dims["C"], dims["T"]), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "silu": 1, "dy": dy}


def _fla_factory():
    """FLA's Triton ``causal_conv1d``, which takes x token-major ``[B, T, C]``.

    This primitive's x is ``[B, C, T]`` -- in the model a transpose of a
    ``[B, T, C]`` projection -- so the adapter hands FLA ``x.transpose(1, 2)``
    (for the model's layout, exactly the contiguous tensor the projection made)
    and transposes the result and the input gradient back.
    """
    from evograd.ops.library_pairs import fla_causal_conv1d_pair

    forward, backward = fla_causal_conv1d_pair()

    def forward_with_saved(x, weight, silu):
        y, saved = forward(x.transpose(1, 2), weight, "silu" if silu else None)
        return y.transpose(1, 2), saved

    def backward_from_saved(dy, saved):
        dx, dweight = backward(dy.transpose(1, 2), saved)
        return dx.transpose(1, 2), dweight

    return forward_with_saved, backward_from_saved


op = declare_op(
    name="causal_conv1d",
    level=1,
    family="conv",
    forward="evograd.ops.level1.causal_conv1d.forward_ref:causal_conv1d_forward_ref",
    runtime_forward=(
        "evograd.ops.level1.causal_conv1d.forward_ref:causal_conv1d_forward_production"
    ),
    dims=_DIMS,
    args=(
        Active("x", "[B, C, T]"),
        Active("weight", "[C, K]"),
        Inactive("silu", default=1, note="1 applies SiLU to the convolution output, 0 does not"),
    ),
    output=Active("y", "[B, C, T]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Depthwise causal convolution over the last axis of x [B, C, T]: "
        "y[b, c, t] = sum_{j=0}^{K-1} weight[c, j] * x[b, c, t - (K - 1) + j], "
        "with x taken as zero for negative time indices. Accumulate in float32. "
        "If silu is 1, apply SiLU to the sum. Return y in x's dtype. There is no "
        "bias and no state carried in from a previous segment. Do not call "
        "F.conv1d or autograd in the generated math."
    ),
    backward_semantics=(
        "Return dx, dweight IN THIS ORDER. Let s be the pre-activation sum and "
        "ds = dy * silu'(s) when silu is 1 (ds = dy otherwise), with "
        "silu'(s) = sigmoid(s) * (1 + s * (1 - sigmoid(s))). Then "
        "dx[b, c, t] = sum_j weight[c, j] * ds[b, c, t + (K - 1) - j] over the "
        "terms with 0 <= t + (K - 1) - j < T, and dweight[c, j] = sum over b and "
        "t of ds[b, c, t] * x[b, c, t - (K - 1) + j]. Accumulate in float32."
    ),
    extra_constraints=(
        "x may be a non-contiguous view: a Gated DeltaNet reaches it by "
        "transposing a [B, T, C] projection. weight is the depthwise kernel "
        "with its singleton input-channel axis removed."
    ),
    performance_baselines={
        "fla": make_pair_baseline(_fla_factory, ("x", "weight", "silu"), sidecar="fla")},
    make_inputs=make_causal_conv1d_inputs,
)
