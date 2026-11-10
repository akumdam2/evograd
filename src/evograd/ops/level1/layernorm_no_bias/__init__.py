"""Operator declaration: LayerNorm with a gamma and no beta.

``nn.LayerNorm(dim, bias=False)``, what AdaLN normalizes its condition with. A
zero-valued beta would still cost an add and a ``dbias`` reduction, so it is not
``layernorm``.
"""

from evograd.opdecl import Active, Inactive, declare_op


def _inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    return {"x": torch.randn((rows, hidden), device=device, dtype=dtype),
            "weight": torch.randn((hidden,), device=device, dtype=dtype), "eps": 1e-5,
            "dy": torch.randn((rows, hidden), device=device, dtype=dtype)}


op = declare_op(
    name="layernorm_no_bias",
    forward="evograd.ops.level1.layernorm_no_bias.forward_ref:layernorm_no_bias_forward_ref",
    runtime_forward="evograd.ops.level1.layernorm_no_bias.forward_ref:layernorm_no_bias_runtime_ref",
    level=1,
    family="norm",
    dims=("rows", "hidden"),
    args=(Active("x", "[rows, hidden]"), Active("weight", "[hidden]"), Inactive("eps", default=1e-5)),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics="Row-wise LayerNorm over the last dimension, scaled by weight, with no beta.",
    backward_semantics="Return dx in x's dtype and dweight in weight's dtype.",
    make_inputs=_inputs,
)
