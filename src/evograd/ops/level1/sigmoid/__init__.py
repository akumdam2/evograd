"""Operator declaration: the elementwise sigmoid."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import dtype_for


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {"x": torch.randn((rows, cols), device=device, dtype=dtype),
            "dy": torch.randn((rows, cols), device=device, dtype=dtype)}


op = declare_op(
    name="sigmoid",
    level=1,
    family="activation",
    forward="evograd.ops.level1.sigmoid.forward_ref:sigmoid_forward_ref",
    runtime_forward="evograd.ops.level1.sigmoid.forward_ref:sigmoid_runtime_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Elementwise y = sigmoid(x).",
    backward_semantics="Return dx = dy * y * (1 - y) in x's dtype.",
    make_inputs=_inputs,
)
