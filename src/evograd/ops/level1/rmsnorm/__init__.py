"""Operator declaration: rmsnorm."""

from evograd.opdecl import Active, Inactive, declare_op


def make_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    # gamma centred on 1.0, not on 0.0. A zero-mean gain is not what any trained
    # RMSNorm holds, and it cancels the signal path when this operator is
    # composed into a block.
    weight = (1.0 + 0.1 * torch.randn((hidden,), device=device, dtype=torch.float32)).to(
        dtype
    )
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "eps": (workload.provenance.free.get("eps", 1e-5) if workload.provenance else 1e-5), "dy": dy}


# Untimed compile/runtime coverage: the non-power-of-two hidden width and the
# small row counts the pre-v1 grid used to time, which are worth running but are
# not part of the performance objective.
#: Three deduplicated RMSNorm configurations per Qwen3-0.6B step: the
#: residual-stream norm (4096 x 1024, 57 invocations) and the two per-head
#: norms, which are many short rows rather than few long ones.


op = declare_op(
    name="rmsnorm",
    forward="evograd.ops.level1.rmsnorm.forward_ref:rmsnorm_forward_ref",
    runtime_forward="evograd.ops.level1.rmsnorm.forward_ref:rmsnorm_runtime_ref",
    level=1,
    family="norm",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    parameter_args=("weight",),
    forward_semantics="Forward computes row-wise RMSNorm over the last dimension of x [rows, hidden]: rrms = rsqrt(mean(x^2, axis=-1, keepdim=True) + eps); y = (x * rrms) * weight. Compute the mean-of-squares and rrms in float32, then cast y back to x's dtype. weight is a 1D tensor of length hidden. Do not call torch RMSNorm/LayerNorm or autograd in the generated math.",
    backward_semantics="Backward must return (dx, dweight). With xhat = x * rrms and g = dy * weight: dx = (g - xhat * mean(g * xhat, axis=-1, keepdim=True)) * rrms, returned in x's dtype and shape [rows, hidden]. dweight = sum(dy * xhat, dim=0), returned in weight's dtype and shape [hidden]. Use float32 accumulation for all reductions.",
    extra_constraints='Tensor layout notes:\n- x, dy: [rows, hidden], contiguous CUDA, float32 or float16\n- weight: [hidden], contiguous CUDA, same dtype as x\n- y, dx: [rows, hidden], same dtype as x\n- There is no bias term and no mean-subtraction (RMSNorm, not LayerNorm).',
    make_inputs=make_rmsnorm_inputs,
)
