"""Triton matrix-multiplication tutorial baseline with fused Leaky-ReLU."""

import torch
import triton
import triton.language as tl

from evograd.ops._common import make_pair_baseline


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    negative_slope,
    APPLY_ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Grouped program ordering and fused epilogue from Triton's GEMM tutorial."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        # Preserve the declaration's FP32 correctness contract. For BF16/FP16
        # inputs Triton still uses tensor cores; for FP32 this disables TF32
        # truncation, which otherwise amplifies through the two backward GEMMs.
        accumulator += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if APPLY_ACTIVATION:
        accumulator = tl.where(
            accumulator >= 0,
            accumulator,
            negative_slope * accumulator,
        )
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def _matmul(a, b, *, negative_slope=0.01, activation=False):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected compatible 2D matrices")
    m, k = a.shape
    n = b.shape[1]
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    block_m, block_n, block_k = 128, 128, 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        float(negative_slope),
        APPLY_ACTIVATION=activation,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output


def _triton_tutorial_factory():
    def forward(a, b, negative_slope):
        output = _matmul(
            a,
            b,
            negative_slope=negative_slope,
            activation=True,
        )
        return output, (a, b, output, float(negative_slope))

    def backward(dc, saved):
        a, b, output, negative_slope = saved
        dpre = torch.where(output >= 0, dc, dc * negative_slope)
        da = _matmul(dpre, b.T)
        db = _matmul(a.T, dpre)
        return da, db

    return forward, backward


measure_triton_tutorial_baseline = make_pair_baseline(
    _triton_tutorial_factory,
    ("a", "b", "negative_slope"),
)
