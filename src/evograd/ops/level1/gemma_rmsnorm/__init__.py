"""Operator declaration: zero-centred RMSNorm, ``y = norm(x) * (1 + w)``.

A separate primitive from ``rmsnorm`` on purpose. The weight is an offset from
one rather than a gain (a fresh model stores zeros), and the cast back to the
input dtype happens after the scale rather than before it. A kernel written for
one is wrong for the other, by the whole weight in the first case and by a
rounding step in the second.
"""

import math

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for


def make_gemma_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + hidden)
    # Zero-centred, as the stored parameter is: a weight drawn around 1 would be
    # a gain of 2 here.
    weight = (0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(dtype)
    return {
        "x": torch.randn((rows, hidden), device=device, dtype=dtype),
        "weight": weight,
        "eps": 1e-6,
        "dy": torch.randn((rows, hidden), device=device, dtype=dtype),
    }


op = declare_op(
    name="gemma_rmsnorm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.gemma_rmsnorm.forward_ref:gemma_rmsnorm_forward_ref",
    dims=("rows", "hidden"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Zero-centred RMSNorm over the last axis of x [rows, hidden], entirely "
        "in float32: xhat = x * rsqrt(mean(x^2, -1, keepdim=True) + eps); "
        "y = xhat * (1 + weight), then cast y to x's dtype. weight is an offset "
        "from one: a weight of zero is the identity scale. The cast happens "
        "after the scale, not before it. Do not call F.rms_norm or autograd in "
        "the generated math."
    ),
    backward_semantics=(
        "Return dx, dweight IN THIS ORDER. With r = rsqrt(mean(x^2) + eps), "
        "xhat = x * r and g = dy * (1 + weight): dx = r * (g - xhat * "
        "mean(g * xhat, -1, keepdim=True)) in x's dtype; dweight = sum over rows "
        "of dy * xhat in weight's dtype. Accumulate every reduction in float32."
    ),
    make_inputs=make_gemma_rmsnorm_inputs,
)
