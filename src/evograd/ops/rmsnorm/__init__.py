"""Operator declaration: rmsnorm."""

from evograd.opdecl import Active, Inactive, Workload, declare_op


def make_rmsnorm_inputs(torch, op, workload, device="cuda"):
    rows, hidden = workload.dims["rows"], workload.dims["hidden"]
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed(rows * 100000 + hidden)
    x = torch.randn((rows, hidden), device=device, dtype=dtype)
    weight = torch.randn((hidden,), device=device, dtype=dtype)
    dy = torch.randn((rows, hidden), device=device, dtype=dtype)
    return {"x": x, "weight": weight, "eps": 1e-5, "dy": dy}

op = declare_op(
    name="rmsnorm",
    forward="evograd.ops.rmsnorm.forward_ref:rmsnorm_forward_ref",
    dims=('rows', 'hidden'),
    args=(
        Active("x", "[rows, hidden]"),
        Active("weight", "[hidden]"),
        Inactive("eps", default=1e-5),
    ),
    output=Active("y", "[rows, hidden]"),
    forward_semantics="Forward computes row-wise RMSNorm over the last dimension of x [rows, hidden]: rrms = rsqrt(mean(x^2, axis=-1, keepdim=True) + eps); y = (x * rrms) * weight. Compute the mean-of-squares and rrms in float32, then cast y back to x's dtype. weight is a 1D tensor of length hidden. Do not call torch RMSNorm/LayerNorm or autograd in the generated math.",
    backward_semantics="Backward must return (dx, dweight). With xhat = x * rrms and g = dy * weight: dx = (g - xhat * mean(g * xhat, axis=-1, keepdim=True)) * rrms, returned in x's dtype and shape [rows, hidden]. dweight = sum(dy * xhat, dim=0), returned in weight's dtype and shape [hidden]. Use float32 accumulation for all reductions.",
    extra_constraints='Tensor layout notes:\n- x, dy: [rows, hidden], contiguous CUDA, float32 or float16\n- weight: [hidden], contiguous CUDA, same dtype as x\n- y, dx: [rows, hidden], same dtype as x\n- There is no bias term and no mean-subtraction (RMSNorm, not LayerNorm).',
    # Ported from benchmark/triton_rmsnorm_backward_bench/task_spec.py.
    correctness=(
        Workload(dims=dict(rows=8, hidden=64), dtype="float32"),
        Workload(dims=dict(rows=17, hidden=128), dtype="float32"),
        Workload(dims=dict(rows=32, hidden=256), dtype="float16"),
        Workload(dims=dict(rows=64, hidden=512), dtype="float16"),
    ),
    benchmark=tuple(
        Workload(dims=dict(rows=r, hidden=h), dtype=dtype)
        for (r, h) in (
            (1, 768), (8, 1024), (32, 1536), (8, 4096), (1, 8192),
            (17, 127), (17, 513), (17, 1000),
        )
        for dtype in ("float32", "float16")
    ),
    tolerances={"float32": (2e-5, 2e-5), "float16": (5e-2, 5e-2)},
    make_inputs=make_rmsnorm_inputs,
)
