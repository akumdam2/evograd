"""Operator declaration: sparsemax."""

from evograd.opdecl import Active, declare_op
from evograd.ops._common import dtype_for, workloads_2d

_TOLERANCES = {"float32": (2e-5, 2e-5)}


def _inputs(torch, op, workload, device="cuda"):
    rows, cols = workload.dims["rows"], workload.dims["cols"]
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {
        "x": torch.randn((rows, cols), device=device, dtype=dtype),
        "dout": torch.randn((rows, cols), device=device, dtype=dtype),
    }


op = declare_op(
    name="sparsemax",
    level=1,
    family="reduction",
    forward="evograd.ops.level1.sparsemax.forward_ref:sparsemax_forward_ref",
    dims=("rows", "cols"),
    args=(Active("x", "[rows, cols]", dtype="float32"),),
    output=Active("out", "[rows, cols]", dtype="float32"),
    parameter_args=(),
    forward_semantics="Project each row onto the probability simplex with sparsemax.",
    backward_semantics=(
        "On support S={i:out_i>0}, return dx_i=dout_i-mean_S(dout); "
        "return zero outside S."
    ),
    correctness=workloads_2d(
        ((8, 64), (17, 128), (32, 256), (64, 512)),
        ("float32",),
        tolerances=_TOLERANCES,
    ),
    tolerances=_TOLERANCES,
    make_inputs=_inputs,
)
