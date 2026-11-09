"""Operator declaration: row-wise L2 normalization (FLA's definition).

What a gated delta rule applies to its queries and keys before the recurrence.
Generic over rows and width.
"""

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops._common import dtype_for, make_pair_baseline


def make_l2norm_inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "eps": 1e-6,
        "dy": torch.randn((rows, cols), device=device, dtype=dtype),
    }


def _fla_factory():
    """FLA's ``l2norm`` (``L2NormFunction``), the normalization its gated delta
    rule applies in kernel: ``x * rsqrt(sum(x^2) + eps)``, as here."""
    from evograd.ops.library_pairs import fla_l2norm_pair

    return fla_l2norm_pair()


op = declare_op(
    name="l2norm",
    level=1,
    family="norm",
    forward="evograd.ops.level1.l2norm.forward_ref:l2norm_forward_ref",
    runtime_forward="evograd.ops.level1.l2norm.forward_ref:l2norm_forward_production",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"), Inactive("eps", default=1e-6)),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics=(
        "Row-wise L2 normalization over the last axis of x [rows, cols]: "
        "r = rsqrt(sum(x^2, axis=-1, keepdim=True) + eps); y = x * r. The eps is "
        "added to the sum of squares, inside the root, not to the norm. "
        "Accumulate the sum in float32 and return y in x's dtype. There is no "
        "weight. Do not call F.normalize, torch.linalg or autograd in the "
        "generated math."
    ),
    backward_semantics=(
        "Return dx: with r as above and y = x * r, "
        "dx = r * (dy - y * sum(dy * y, axis=-1, keepdim=True)), in x's dtype. "
        "Accumulate the row reduction in float32."
    ),
    performance_baselines={"fla": make_pair_baseline(_fla_factory, ("x", "eps"), sidecar="fla")},
    make_inputs=make_l2norm_inputs,
)
