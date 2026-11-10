"""Operator declaration: the elementwise rectifier."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {"x": torch.randn((rows, cols), device=device, dtype=dtype),
            "dy": torch.randn((rows, cols), device=device, dtype=dtype)}


op = declare_op(
    name="relu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.relu.forward_ref:relu_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]"),),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="Elementwise y = max(x, 0).",
    backward_semantics="Return dx = dy where x > 0, else 0, in x's dtype.",
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    make_inputs=_inputs,
)
