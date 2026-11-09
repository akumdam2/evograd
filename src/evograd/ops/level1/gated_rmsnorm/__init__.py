"""Operator declaration: RMSNorm gated by ``silu(z)``.

``y = (norm(x) * w) * silu(z)`` with the casts of the reference implementation
-- see :mod:`.forward_ref`. The weight is a plain gain (initialized to ones),
unlike ``gemma_rmsnorm``'s offset.
"""

import math

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline


def make_gated_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + hidden * 7 + 1)
    weight = (1.0 + 0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(dtype)
    return {
        "x": torch.randn((rows, hidden), device=device, dtype=dtype),
        "z": torch.randn((rows, hidden), device=device, dtype=dtype),
        "weight": weight,
        "eps": 1e-6,
        "dy": torch.randn((rows, hidden), device=device, dtype=dtype),
    }


def _fla_factory():
    """FLA's ``FusedRMSNormGated`` (``LayerNormGatedFunction``, RMS, swish gate)."""
    from evograd.ops.library_pairs import fla_rms_norm_gated_pair

    return fla_rms_norm_gated_pair(activation="swish")


op = declare_op(
    name="gated_rmsnorm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.gated_rmsnorm.forward_ref:gated_rmsnorm_forward_ref",
    dims=("rows", "hidden"),
    args=(
        Active("x", "[rows, hidden]"),
        Active("z", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-6),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics=(
        "Gated RMSNorm over the last axis, with these casts in this order: "
        "h = float32(x); h = h * rsqrt(mean(h^2, -1, keepdim=True) + eps); "
        "h = weight * h.to(x.dtype) (the product is in x's dtype); "
        "y = (h * silu(float32(z))).to(x.dtype). The weight is a gain "
        "(initialized to one). Do not call F.rms_norm or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return dx, dz, dweight IN THIS ORDER. With r = rsqrt(mean(x^2) + eps), "
        "xhat = x * r, s = silu(z) and n = weight * xhat: dz = dy * n * "
        "silu'(z) where silu'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z))); "
        "dweight = sum over rows of dy * s * xhat; with g = dy * s * weight, "
        "dx = r * (g - xhat * mean(g * xhat, -1, keepdim=True)). Each gradient in "
        "its input's dtype, every reduction accumulated in float32."
    ),
    performance_baselines={
        "fla": make_pair_baseline(_fla_factory, ("x", "z", "weight", "eps"), sidecar="fla")},
    make_inputs=make_gated_rmsnorm_inputs,
)
