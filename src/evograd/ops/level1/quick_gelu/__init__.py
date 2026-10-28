from evograd.opdecl import Active, Workload, declare_op

op = declare_op(
    name="quick_gelu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.quick_gelu.reference:forward",
    dims=("rows", "hidden"),
    args=(Active("x", "[rows, hidden]"),),
    output=Active("out", "[rows, hidden]"),
    parameter_args=(),
    forward_semantics="QuickGELU x*sigmoid(1.702*x).",
    backward_semantics="Return dx by the analytic derivative, multiplied by dout.",
    correctness=(Workload(dims=dict(rows=7, hidden=13), dtype="float32"),),
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (0.02, 0.02)},
)
