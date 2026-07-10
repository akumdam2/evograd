"""Operator declaration: layernorm."""

from evograd.opdecl import declare_op, Duplicated, Const

op = declare_op(
    name="layernorm",
    forward="evograd.ops.layernorm_forward_ref:layernorm_forward_ref",
    dims=('rows', 'hidden'),
    args=(
        Duplicated("x", "[rows, hidden]"),
        Duplicated("weight", "[hidden]"),
        Duplicated("bias", "[hidden]"),
        Const("eps", default=1e-5),
    ),
    output=Duplicated("y", "[rows, hidden]"),
    forward_semantics='Do not call PyTorch autograd or PyTorch reference LayerNorm in the generated math. Forward must produce the same `y` as row-wise LayerNorm over the last dimension.',
    backward_semantics='Backward must consume only `dy`, `saved_tensors`, and `eps`. Return `dx` with `x` dtype, `dweight` with `weight` dtype, and `dbias` with `bias` dtype.',
)
