from evograd.opdecl import Active, declare_op

op = declare_op(
    name="gelu",
    level=1,
    family="activation",
    forward="evograd.ops.level1.gelu.reference:forward",
    dims=("rows", "hidden"),
    args=(Active("x", "[rows, hidden]"),),
    output=Active("out", "[rows, hidden]"),
    parameter_args=(),
    forward_semantics="Exact GELU x*Phi(x), no tanh approximation.",
    backward_semantics="Return dx by the analytic derivative, multiplied by dout.",
)
