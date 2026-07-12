"""Operator declaration: matmul."""

from evograd.opdecl import declare_op, Duplicated, Const, Workload

op = declare_op(
    name="matmul",
    forward="evograd.ops.matmul_forward_ref:matmul_forward_ref",
    dims=('M', 'K', 'N'),
    args=(
        Duplicated("a", "[M, K]"),
        Duplicated("b", "[K, N]"),
    ),
    output=Duplicated("c", "[M, N]"),
    forward_semantics='Forward computes a plain GEMM c = a @ b, where a is [M, K], b is [K, N], and c is [M, N], all contiguous 2D CUDA tensors. There is no eps and no bias. Accumulate in float32 and cast c back to the input dtype. Do not call torch.matmul, the @ operator, or autograd in the generated math; use a Triton tiled matmul (tl.dot).',
    backward_semantics="Backward must return (da, db). da = dc @ b.T with shape [M, K] and a's dtype. db = a.T @ dc with shape [K, N] and b's dtype. Accumulate both matmuls in float32. The forward output c is NOT needed by backward and should not be saved.",
    extra_constraints='Tensor layout notes:\n- a: [M, K], b: [K, N], dc: [M, N], contiguous CUDA, float32 or float16\n- da: [M, K] (a.dtype), db: [K, N] (b.dtype)\n- These shapes are compute-bound; prefer tensor-core tiled matmul with fp32 accumulation, reasonable BLOCK_M/BLOCK_N/BLOCK_K, and handle non-tile-aligned M/N/K with boundary masking.',
    # Ported from benchmark/triton_matmul_backward_bench/task_spec.py
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
