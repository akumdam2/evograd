"""Operator declaration: the gated linear unit over the last axis.

``silu(a) * b`` is ``swiglu`` and ``sigmoid(b) * a`` on halves of one tensor is this.
One input, because the two halves are one tensor in memory.
"""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops._common import STANDARD_TOLERANCES, dtype_for


def _inputs(torch, op, workload, device="cuda"):
    rows, width, cols = (workload.dims[d] for d in ("rows", "width", "cols"))
    dtype = dtype_for(torch, workload.dtype)
    torch.manual_seed(rows * 100003 + cols)
    return {"x": torch.randn((rows, width), device=device, dtype=dtype),
            "dy": torch.randn((rows, cols), device=device, dtype=dtype)}


op = declare_op(
    name="glu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.glu.forward_ref:glu_forward_ref",
    runtime_forward="evograd.ops.level1.glu.forward_ref:glu_runtime_ref",
    dims=("rows", "width", "cols"),
    args=(Active("x", "[rows, width]", note="width = 2 * cols"),),
    output=Active("y", "[rows, cols]"),
    parameter_args=(),
    forward_semantics="first, gates = chunk(x, 2) over the last axis; y = first * sigmoid(gates).",
    backward_semantics=(
        "Return dx [rows, width] in x's dtype: its first half dy * sigmoid(gates), its second half "
        "dy * first * sigmoid(gates) * (1 - sigmoid(gates))."),
    correctness=(
        Workload(dims=dict(rows=8, width=128, cols=64), dtype="float32"),
        Workload(dims=dict(rows=17, width=256, cols=128), dtype="float32"),
        Workload(dims=dict(rows=32, width=512, cols=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, width=1024, cols=512), dtype="bfloat16"),
    ),
    tolerances=STANDARD_TOLERANCES,
    make_inputs=_inputs,
)
