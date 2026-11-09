"""Operator declaration: LayerNorm with no parameters.

``nn.LayerNorm(dim, elementwise_affine=False)``, what AdaLN normalizes its input
with: there is no gamma or beta, so the contract has no parameter and no parameter
gradient, which is real work ``layernorm`` would otherwise charge a candidate for.
"""

from evograd.opdecl import Active, Inactive, Workload, declare_op


def _inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    return {"x": torch.randn((rows, hidden), device=device, dtype=dtype), "eps": 1e-5,
            "dy": torch.randn((rows, hidden), device=device, dtype=dtype)}


op = declare_op(
    name="layernorm_no_affine",
    forward="evograd.ops.level1.layernorm_no_affine.forward_ref:layernorm_no_affine_forward_ref",
    runtime_forward="evograd.ops.level1.layernorm_no_affine.forward_ref:layernorm_no_affine_runtime_ref",
    level=1,
    family="norm",
    dims=("rows", "hidden"),
    args=(Active("x", "[rows, hidden]"), Inactive("eps", default=1e-5)),
    output=Active("y", "[rows, hidden]"),
    parameter_args=(),
    forward_semantics="Row-wise LayerNorm over the last dimension with no gamma and no beta.",
    backward_semantics="Return dx in x's dtype.",
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="bfloat16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="bfloat16"),
    ),
    tolerances={"float32": (2e-5, 2e-5), "bfloat16": (8e-2, 8e-2)},
    make_inputs=_inputs,
)
