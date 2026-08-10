"""Operator declaration: GEMM with a fused Leaky-ReLU epilogue."""

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.ops.gemm_leaky_relu.triton_tutorial import (
    measure_triton_tutorial_baseline,
)


_BENCHMARK_SHAPES = (
    (256, 1024, 1024),
    (1024, 1024, 1024),
    (4096, 1024, 1024),
    (1024, 4096, 4096),
    (4096, 4096, 1024),
    (2048, 4096, 4096),
    (8192, 4096, 14336),
)


def _workloads(shapes):
    return tuple(
        Workload(dims=dict(M=m, K=k, N=n), dtype="bfloat16")
        for m, k, n in shapes
    )


def make_gemm_leaky_relu_inputs(torch, op, workload, device="cuda"):
    m, k, n = (workload.dims[name] for name in ("M", "K", "N"))
    dtype = getattr(torch, workload.dtype)
    torch.manual_seed((m * 100003 + n) * 100003 + k)
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    dc = torch.randn((m, n), device=device, dtype=dtype)
    return {"a": a, "b": b, "negative_slope": 0.01, "dc": dc}


op = declare_op(
    name="gemm_leaky_relu",
    forward=(
        "evograd.ops.gemm_leaky_relu.forward_ref:"
        "gemm_leaky_relu_forward_ref"
    ),
    dims=("M", "K", "N"),
    args=(
        Active("a", "[M, K]"),
        Active("b", "[K, N]"),
        Inactive("negative_slope", default=0.01),
    ),
    output=Active("c", "[M, N]"),
    forward_semantics=(
        "Compute pre = a @ b with float32 accumulation, then apply a fused "
        "Leaky-ReLU epilogue: c = pre when pre >= 0, otherwise "
        "c = negative_slope * pre. Cast c to the input dtype. This contract "
        "matches the fused activation form in Triton's matrix-multiplication "
        "tutorial. Generated math must use a tiled tl.dot GEMM and must not "
        "call torch.matmul, the @ operator, or PyTorch autograd."
    ),
    backward_semantics=(
        "Let dpre = dc where c >= 0 and negative_slope * dc otherwise. "
        "Return da = dpre @ b.T and db = a.T @ dpre, both accumulated in "
        "float32 and cast to their corresponding input dtype. The activation "
        "decision may be recovered from c because positive negative_slope "
        "preserves the sign."
    ),
    extra_constraints=(
        "All tensors are contiguous 2D CUDA tensors. Optimize the GEMM and "
        "activation as one Triton kernel rather than materializing pre. "
        "The default performance baseline is PyTorch autograd backed by "
        "cuBLAS; it is not a Liger kernel."
    ),
    correctness=(
        Workload(dims=dict(M=64, K=64, N=64), dtype="float32"),
        Workload(dims=dict(M=129, K=127, N=257), dtype="float32"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="float16"),
        Workload(dims=dict(M=257, K=255, N=129), dtype="float16"),
        Workload(dims=dict(M=128, K=128, N=256), dtype="bfloat16"),
        Workload(dims=dict(M=257, K=255, N=129), dtype="bfloat16"),
    ),
    coverage=_workloads(_BENCHMARK_SHAPES),
    benchmark=_workloads(_BENCHMARK_SHAPES),
    benchmark_suites={"industrial_bf16": _workloads(_BENCHMARK_SHAPES)},
    performance_baselines={
        "triton_tutorial": measure_triton_tutorial_baseline,
    },
    tolerances={
        "float32": (8e-2, 2e-2),
        "float16": (1e-1, 2e-2),
        "bfloat16": (1.5e-1, 3e-2),
    },
    make_inputs=make_gemm_leaky_relu_inputs,
)
