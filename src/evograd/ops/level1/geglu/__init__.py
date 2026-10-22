"""Operator declaration: tanh-approximate GeGLU."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for, standard_correctness


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "a": torch.randn((rows, cols), device=device, dtype=dtype),
        "b": torch.randn((rows, cols), device=device, dtype=dtype),
        "dc": torch.randn((rows, cols), device=device, dtype=dtype),
    }


op = declare_op(
    name="geglu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.geglu.forward_ref:geglu_forward_ref",
    dims=("rows", "cols"),
    args=(Active("a", "[rows, cols]"), Active("b", "[rows, cols]")),
    output=Active("c", "[rows, cols]"),
    parameter_args=(),
    forward_semantics=(
        "Element-wise c = gelu(a, approximate='tanh') * b. Use the tanh "
        "approximation, fp32 intermediates, and preserve input dtype."
    ),
    backward_semantics="Return da and db for both differentiable inputs.",
    correctness=standard_correctness(),
    tolerances=STANDARD_TOLERANCES,
    make_inputs=_inputs,
)
