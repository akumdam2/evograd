"""Operator declaration: row-wise softmax."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import dtype_for


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dy": torch.randn((rows, cols), device=device, dtype=dtype),
    }


op = declare_op(
    name="softmax",
    level=1,
    family="reduction",
    forward="evograd.ops.level1.softmax.forward_ref:softmax_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Numerically stable softmax over the last dimension in fp32.",
    backward_semantics="Return dx = y * (dy - sum(dy*y, dim=-1, keepdim=True)).",
    make_inputs=_inputs,
)
