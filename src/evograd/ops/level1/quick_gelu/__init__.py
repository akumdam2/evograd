from evograd.opdecl import Active, declare_op

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
)
