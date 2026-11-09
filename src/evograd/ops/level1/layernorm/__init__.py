"""Operator declaration: layernorm."""

from evograd.opdecl import Active, Inactive, declare_op


def make_layernorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    bias = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias, "eps": (workload.provenance.free.get("eps", 1e-5) if workload.provenance else 1e-5), "dy": dy}


op = declare_op(
    name="layernorm",
    forward="evograd.ops.level1.layernorm.forward_ref:layernorm_forward_ref",
    runtime_forward="evograd.ops.level1.layernorm.forward_ref:layernorm_runtime_ref",
    level=1,
    family="norm",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Active("bias", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight", "bias"),
    forward_semantics='Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. Forward must produce the same `y` as row-wise LayerNorm over the last dimension.',
    backward_semantics='Backward must consume only `dy`, `saved_tensors`, and `eps`. Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.',
    make_inputs=make_layernorm_inputs,
)
