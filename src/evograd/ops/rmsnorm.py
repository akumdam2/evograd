"""Operator declaration: rmsnorm."""

from evograd.opdecl import declare_op, Duplicated, Const

op = declare_op(
    name="rmsnorm",
    forward="evograd.ops.rmsnorm_forward_ref:rmsnorm_forward_ref",
    dims=('rows', 'hidden'),
    args=(
        Duplicated("x", "[rows, hidden]"),
        Duplicated("weight", "[hidden]"),
        Const("eps", default=1e-5),
    ),
    output=Duplicated("y", "[rows, hidden]"),
    forward_semantics="Forward computes row-wise RMSNorm over the last dimension of x [rows, hidden]: rrms = rsqrt(mean(x^2, axis=-1, keepdim=True) + eps); y = (x * rrms) * weight. Compute the mean-of-squares and rrms in float32, then cast y back to x's dtype. weight is a 1D tensor of length hidden. Do not call torch RMSNorm/LayerNorm or autograd in the generated math.",
    backward_semantics="Backward must return (dx, dweight). With xhat = x * rrms and g = dy * weight: dx = (g - xhat * mean(g * xhat, axis=-1, keepdim=True)) * rrms, returned in x's dtype and shape [rows, hidden]. dweight = sum(dy * xhat, dim=0), returned in weight's dtype and shape [hidden]. Use float32 accumulation for all reductions.",
    extra_constraints='Tensor layout notes:\n- x, dy: [rows, hidden], contiguous CUDA, float32 or float16\n- weight: [hidden], contiguous CUDA, same dtype as x\n- y, dx: [rows, hidden], same dtype as x\n- There is no bias term and no mean-subtraction (RMSNorm, not LayerNorm).',
)
