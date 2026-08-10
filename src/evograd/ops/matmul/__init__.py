"""Operator declaration: matmul."""

from evograd.opdecl import Active, Workload, declare_op
from evograd.ops.matmul.cublas import measure_cublas_pair_baseline


_LEGACY_SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 1024, 1024),
    (1024, 1024, 2048),
    (2048, 1024, 2048),
    (4096, 1024, 1024),
)
_INDUSTRIAL_BF16_SHAPES = (
    (256, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1024, 1024),
    (1024, 4096, 4096),
    (4096, 4096, 1024),
    (2048, 4096, 4096),
    (8192, 4096, 14336),
)
_LARGE_BF16_SHAPES = tuple(
    shape for shape in _INDUSTRIAL_BF16_SHAPES if shape[1] >= 4096
)
_EXTREME_BF16_SHAPES = ((8192, 4096, 14336),)


def _workloads(shapes, dtypes):
    return tuple(
        Workload(dims=dict(M=m, K=k, N=n), dtype=dtype)
        for m, k, n in shapes
        for dtype in dtypes
    )


def make_matmul_inputs(torch, op, workload, device="cuda"):
    m, k, n = (workload.dims[name] for name in ("M", "K", "N"))
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed((m * 100003 + n) * 100003 + k)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    dc = torch.randn((m, n), device=device, dtype=dtype)
    return {"a": a, "b": b, "dc": dc}

op = declare_op(
    name="matmul",
    forward="evograd.ops.matmul.forward_ref:matmul_forward_ref",
    dims=('M', 'K', 'N'),
    args=(
        Active("a", "[M, K]"),
        Active("b", "[K, N]"),
    ),
    output=Active("c", "[M, N]"),
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
        Workload(dims=dict(M=128, K=128, N=256), dtype="bfloat16"),
        Workload(dims=dict(M=257, K=255, N=129), dtype="bfloat16"),
    ),
    coverage=_workloads(_INDUSTRIAL_BF16_SHAPES, ("bfloat16",)),
    benchmark=_workloads(_INDUSTRIAL_BF16_SHAPES, ("bfloat16",)),
    benchmark_suites={
        "industrial_bf16": _workloads(_INDUSTRIAL_BF16_SHAPES, ("bfloat16",)),
        "large_bf16": _workloads(_LARGE_BF16_SHAPES, ("bfloat16",)),
        "extreme_bf16": _workloads(_EXTREME_BF16_SHAPES, ("bfloat16",)),
        "legacy": _workloads(_LEGACY_SHAPES, ("float32", "float16")),
    },
    performance_baselines={"cublas_pair": measure_cublas_pair_baseline},
    tolerances={
        "float32": (8e-2, 2e-2),
        "float16": (1e-1, 2e-2),
        "bfloat16": (1.5e-1, 3e-2),
    },
    make_inputs=make_matmul_inputs,
)
