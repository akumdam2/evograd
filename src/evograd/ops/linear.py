"""Operator declaration: linear."""

from evograd.opdecl import declare_op, Duplicated, Const, Workload

op = declare_op(
    name="linear",
    forward="evograd.ops.linear_forward_ref:linear_forward_ref",
    dims=('M', 'K', 'N'),
    args=(
        Duplicated("x", "[M, K]"),
        Duplicated("weight", "[N, K]"),
        Duplicated("bias", "[N]"),
    ),
    output=Duplicated("y", "[M, N]"),
    forward_semantics='Forward computes a Linear layer y = x @ weight.T + bias, where x is [M, K], weight is [N, K], bias is [N], and y is [M, N], all contiguous CUDA tensors. There is no eps. Accumulate the matmul in float32 and cast y back to the input dtype. Do not call F.linear, torch.matmul, the @ operator, or autograd in the generated math; use a Triton tiled matmul (tl.dot).',
    backward_semantics="Backward must return (dx, dweight, dbias). dx = dy @ weight, shape [M, K], x's dtype. dweight = dy.T @ x, shape [N, K], weight's dtype. dbias = sum(dy, dim=0), shape [N], dy's dtype. Accumulate all reductions/matmuls in float32. dbias does not depend on the bias value, so the bias tensor need not be saved.",
    extra_constraints='Tensor layout notes:\n- x: [M, K], weight: [N, K], bias: [N], dy: [M, N], contiguous CUDA, float32 or float16\n- dx: [M, K] (x.dtype), dweight: [N, K] (weight.dtype), dbias: [N] (dy.dtype)\n- These shapes are compute-bound; prefer tensor-core tiled matmul with fp32 accumulation and boundary masking for non-tile-aligned M/N/K.',
    # Ported from benchmark/triton_linear_backward_bench/task_spec.py
    # (task_spec cases are (M, N, K); declaration dims are M/K/N).
    correctness=(
        Workload(dims=dict(M=64, K=64, N=64), dtype="float32"),
        Workload(dims=dict(M=129, K=127, N=257), dtype="float32"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="float16"),
        Workload(dims=dict(M=512, K=256, N=512), dtype="float16"),
    ),
    benchmark=tuple(
        Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
        for (m, n, k) in (
            (512, 512, 512), (1024, 1024, 1024), (2048, 1024, 1024),
            (1024, 2048, 1024), (2048, 2048, 1024), (4096, 1024, 1024),
        )
        for dtype in ("float32", "float16")
    ),
    tolerances={"float32": (8e-2, 2e-2), "float16": (1e-1, 2e-2)},
)
