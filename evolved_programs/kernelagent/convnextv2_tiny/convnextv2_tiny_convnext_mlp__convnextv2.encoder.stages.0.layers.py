"""adbench-tools artifact: task convnextv2_tiny.convnextv2_tiny_convnext_mlp.convnextv2.encoder.stages.0.layers, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the ConvNeXtV2 ConvNext MLP forward + backward pair.
#
# Fused/implemented stages:
#   forward:
#     1. up projection:       x @ up_weight.T + up_bias
#     2. GELU activation
#     3. GRN:                 a + gamma * a * (||a||_HW / mean_C(||a||_HW)) + beta
#     4. down projection:     grn_out @ down_weight.T + down_bias
#
#   backward:
#     1. gradients for down projection
#     2. GRN backward, including spatial/channel reductions
#     3. GELU backward
#     4. gradients for up projection
#
# `kernel_function` returns a single flattened fp32 vector matching:
#   torch.cat([forward_output.float().reshape(-1), *grads.float().reshape(-1)])
#
# The Python wrappers only allocate tensors, inspect shapes, and launch Triton kernels.
# All numerical work, including packing the final flat output, is performed in Triton.

import torch
import triton
import triton.language as tl


_EPS = 1.0e-6


@triton.jit
def _linear_forward_kernel(
    a_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)

    acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr + offs_m[:, None] * I + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < I),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_o[None, :] * I + offs_k[:, None],
            mask=(offs_o[None, :] < O) & (offs_k[:, None] < I),
            other=0.0,
        )
        acc += tl.dot(a, w)

    bias = tl.load(b_ptr + offs_o, mask=offs_o < O, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * O + offs_o[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_o[None, :] < O),
    )


@triton.jit
def _linear_grad_input_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    M: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for o0 in range(0, O, BLOCK_K):
        offs_o = o0 + tl.arange(0, BLOCK_K)

        dy = tl.load(
            dy_ptr + offs_m[:, None] * O + offs_o[None, :],
            mask=(offs_m[:, None] < M) & (offs_o[None, :] < O),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_o[:, None] * I + offs_i[None, :],
            mask=(offs_o[:, None] < O) & (offs_i[None, :] < I),
            other=0.0,
        )
        acc += tl.dot(dy, w)

    tl.store(
        dx_ptr + offs_m[:, None] * I + offs_i[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_i[None, :] < I),
    )


@triton.jit
def _linear_grad_weight_bias_kernel(
    dy_ptr,
    a_ptr,
    dw_ptr,
    db_ptr,
    M: tl.constexpr,
    I: tl.constexpr,
    O: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)

    acc_w = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_O,), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)

        dy_t = tl.load(
            dy_ptr + offs_o[:, None] + offs_m[None, :] * O,
            mask=(offs_o[:, None] < O) & (offs_m[None, :] < M),
            other=0.0,
        )
        a = tl.load(
            a_ptr + offs_m[:, None] * I + offs_i[None, :],
            mask=(offs_m[:, None] < M) & (offs_i[None, :] < I),
            other=0.0,
        )
        acc_w += tl.dot(dy_t, a)
        acc_b += tl.sum(dy_t, axis=1)

    tl.store(
        dw_ptr + offs_o[:, None] * I + offs_i[None, :],
        acc_w,
        mask=(offs_o[:, None] < O) & (offs_i[None, :] < I),
    )

    if pid_i == 0:
        tl.store(db_ptr + offs_o, acc_b, mask=offs_o < O)


@triton.jit
def _gelu_forward_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _gelu_backward_kernel(
    pre_ptr,
    dy_ptr,
    dx_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(pre_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    erf_part = tl.erf(x * 0.7071067811865476)
    exp_part = tl.exp(-0.5 * x * x)
    grad = 0.5 * (1.0 + erf_part) + x * exp_part * 0.3989422804014327

    tl.store(dx_ptr + offs, dy * grad, mask=mask)


@triton.jit
def _grn_partial_sums_kernel(
    a_ptr,
    partial_ptr,
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    PB: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)
    pid_pb = tl.program_id(2)

    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_p = pid_pb * BLOCK_P + tl.arange(0, BLOCK_P)

    vals = tl.load(
        a_ptr + (pid_b * P + offs_p[:, None]) * D + offs_d[None, :],
        mask=(offs_p[:, None] < P) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    ss = tl.sum(vals * vals, axis=0)
    tl.store(
        partial_ptr + (pid_b * D + offs_d) * PB + pid_pb,
        ss,
        mask=offs_d < D,
    )


@triton.jit
def _grn_reduce_g_kernel(
    partial_ptr,
    g_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    PB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PB: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_pb = tl.arange(0, BLOCK_PB)

    vals = tl.load(
        partial_ptr + (pid_b * D + offs_d[:, None]) * PB + offs_pb[None, :],
        mask=(offs_d[:, None] < D) & (offs_pb[None, :] < PB),
        other=0.0,
    ).to(tl.float32)

    ss = tl.sum(vals, axis=1)
    g = tl.sqrt(ss)
    tl.store(g_ptr + pid_b * D + offs_d, g, mask=offs_d < D)


@triton.jit
def _grn_mean_kernel(
    g_ptr,
    inv_mean_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    g = tl.load(g_ptr + pid_b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
    s = tl.sum(g, axis=0)
    mean = s / D + 1.0e-6
    tl.store(inv_mean_ptr + pid_b, 1.0 / mean)


@triton.jit
def _grn_forward_kernel(
    a_ptr,
    gamma_ptr,
    beta_ptr,
    g_ptr,
    inv_mean_ptr,
    y_ptr,
    TOTAL: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    d = offs % D
    bp = offs // D
    b = bp // P

    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + d, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + d, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * D + d, mask=mask, other=0.0).to(tl.float32)
    inv_m = tl.load(inv_mean_ptr + b, mask=mask, other=0.0).to(tl.float32)

    nx = g * inv_m
    y = a + gamma * a * nx + beta
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _grn_backward_partial_kernel(
    dy_ptr,
    a_ptr,
    gamma_ptr,
    g_ptr,
    inv_mean_ptr,
    q_part_ptr,
    dgamma_part_ptr,
    dbeta_part_ptr,
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    PB: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)
    pid_pb = tl.program_id(2)

    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_p = pid_pb * BLOCK_P + tl.arange(0, BLOCK_P)

    idx = (pid_b * P + offs_p[:, None]) * D + offs_d[None, :]

    dy = tl.load(
        dy_ptr + idx,
        mask=(offs_p[:, None] < P) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    a = tl.load(
        a_ptr + idx,
        mask=(offs_p[:, None] < P) & (offs_d[None, :] < D),
        other=0.0,
    ).to(tl.float32)

    gamma = tl.load(gamma_ptr + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + pid_b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
    inv_m = tl.load(inv_mean_ptr + pid_b).to(tl.float32)
    nx = g * inv_m

    q = tl.sum(dy * gamma[None, :] * a, axis=0)
    dgamma = tl.sum(dy * a * nx[None, :], axis=0)
    dbeta = tl.sum(dy, axis=0)

    base = (pid_b * D + offs_d) * PB + pid_pb
    tl.store(q_part_ptr + base, q, mask=offs_d < D)
    tl.store(dgamma_part_ptr + base, dgamma, mask=offs_d < D)
    tl.store(dbeta_part_ptr + base, dbeta, mask=offs_d < D)


@triton.jit
def _grn_backward_reduce_bd_kernel(
    q_part_ptr,
    dgamma_part_ptr,
    dbeta_part_ptr,
    q_ptr,
    dgamma_bd_ptr,
    dbeta_bd_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    PB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PB: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_db = tl.program_id(1)

    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_pb = tl.arange(0, BLOCK_PB)

    base = (pid_b * D + offs_d[:, None]) * PB + offs_pb[None, :]
    mask = (offs_d[:, None] < D) & (offs_pb[None, :] < PB)

    qv = tl.load(q_part_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dgv = tl.load(dgamma_part_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dbv = tl.load(dbeta_part_ptr + base, mask=mask, other=0.0).to(tl.float32)

    q = tl.sum(qv, axis=1)
    dg = tl.sum(dgv, axis=1)
    db = tl.sum(dbv, axis=1)

    out = pid_b * D + offs_d
    tl.store(q_ptr + out, q, mask=offs_d < D)
    tl.store(dgamma_bd_ptr + out, dg, mask=offs_d < D)
    tl.store(dbeta_bd_ptr + out, db, mask=offs_d < D)


@triton.jit
def _grn_sum_qg_kernel(
    q_ptr,
    g_ptr,
    sum_qg_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(q_ptr + pid_b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + pid_b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)

    s = tl.sum(q * g, axis=0)
    tl.store(sum_qg_ptr + pid_b, s)


@triton.jit
def _grn_reduce_param_grads_kernel(
    dgamma_bd_ptr,
    dbeta_bd_ptr,
    dgamma_ptr,
    dbeta_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_db = tl.program_id(0)
    offs_d = pid_db * BLOCK_D + tl.arange(0, BLOCK_D)

    acc_g = tl.zeros((BLOCK_D,), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for b in range(0, B):
        acc_g += tl.load(dgamma_bd_ptr + b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        acc_b += tl.load(dbeta_bd_ptr + b * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)

    tl.store(dgamma_ptr + offs_d, acc_g, mask=offs_d < D)
    tl.store(dbeta_ptr + offs_d, acc_b, mask=offs_d < D)


@triton.jit
def _grn_backward_apply_kernel(
    dy_ptr,
    a_ptr,
    gamma_ptr,
    g_ptr,
    inv_mean_ptr,
    q_ptr,
    sum_qg_ptr,
    da_ptr,
    TOTAL: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    d = offs % D
    bp = offs // D
    b = bp // P

    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + d, mask=mask, other=0.0).to(tl.float32)

    g = tl.load(g_ptr + b * D + d, mask=mask, other=0.0).to(tl.float32)
    inv_m = tl.load(inv_mean_ptr + b, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + b * D + d, mask=mask, other=0.0).to(tl.float32)
    sum_qg = tl.load(sum_qg_ptr + b, mask=mask, other=0.0).to(tl.float32)

    nx = g * inv_m
    dg = q * inv_m - sum_qg * inv_m * inv_m / D
    norm_grad = tl.where(g > 0.0, a / g, 0.0)

    da = dy * (1.0 + gamma * nx) + dg * norm_grad
    tl.store(da_ptr + offs, da, mask=mask)


@triton.jit
def _pack_kernel(
    src_ptr,
    dst_ptr,
    OFFSET: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    v = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + OFFSET + offs, v, mask=mask)


def _forward_impl(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    D = up_weight.shape[0]
    P = H * W
    M = B * P
    PB = triton.cdiv(P, 256)

    up_pre = torch.empty((M, D), device=x.device, dtype=x.dtype)
    act = torch.empty((M, D), device=x.device, dtype=x.dtype)
    grn_out = torch.empty((M, D), device=x.device, dtype=x.dtype)
    out = torch.empty((B, H, W, C), device=x.device, dtype=x.dtype)

    partial = torch.empty((B, D, PB), device=x.device, dtype=torch.float32)
    g = torch.empty((B, D), device=x.device, dtype=torch.float32)
    inv_mean = torch.empty((B,), device=x.device, dtype=torch.float32)

    grid_up = (triton.cdiv(M, 16), triton.cdiv(D, 32))
    _linear_forward_kernel[grid_up](
        x,
        up_weight,
        up_bias,
        up_pre,
        M,
        C,
        D,
        BLOCK_M=16,
        BLOCK_O=32,
        BLOCK_K=32,
        num_warps=4,
    )

    total_hidden = M * D
    _gelu_forward_kernel[(triton.cdiv(total_hidden, 256),)](
        up_pre,
        act,
        total_hidden,
        BLOCK=256,
        num_warps=4,
    )

    _grn_partial_sums_kernel[(B, triton.cdiv(D, 32), PB)](
        act,
        partial,
        B,
        P,
        D,
        PB,
        BLOCK_P=256,
        BLOCK_D=32,
        num_warps=4,
    )

    _grn_reduce_g_kernel[(B, triton.cdiv(D, 32))](
        partial,
        g,
        B,
        D,
        PB,
        BLOCK_D=32,
        BLOCK_PB=triton.next_power_of_2(PB),
        num_warps=4,
    )

    _grn_mean_kernel[(B,)](
        g,
        inv_mean,
        B,
        D,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )

    _grn_forward_kernel[(triton.cdiv(total_hidden, 256),)](
        act,
        grn_weight,
        grn_bias,
        g,
        inv_mean,
        grn_out,
        total_hidden,
        P,
        D,
        BLOCK=256,
        num_warps=4,
    )

    grid_down = (triton.cdiv(M, 16), triton.cdiv(C, 32))
    _linear_forward_kernel[grid_down](
        grn_out,
        down_weight,
        down_bias,
        out,
        M,
        D,
        C,
        BLOCK_M=16,
        BLOCK_O=32,
        BLOCK_K=32,
        num_warps=4,
    )

    saved = (
        x,
        up_weight,
        grn_weight,
        down_weight,
        up_pre,
        act,
        grn_out,
        g,
        inv_mean,
    )
    return out, saved


def convnextv2_tiny_convnext_mlp_forward_with_saved(
    x,
    up_weight,
    up_bias,
    grn_weight,
    grn_bias,
    down_weight,
    down_bias,
):
    """Forward wrapper for ConvNeXtV2 MLP. Numerical work is executed in Triton."""
    return _forward_impl(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)


def convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved):
    """Backward wrapper for ConvNeXtV2 MLP. Numerical work is executed in Triton."""
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    x, up_weight, grn_weight, down_weight, up_pre, act, grn_out, g, inv_mean = saved

    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    D = up_weight.shape[0]
    P = H * W
    M = B * P
    PB = triton.cdiv(P, 256)

    dx = torch.empty_like(x)
    dup_weight = torch.empty_like(up_weight)
    dup_bias = torch.empty((D,), device=x.device, dtype=x.dtype)
    dgrn_weight = torch.empty_like(grn_weight)
    dgrn_bias = torch.empty_like(grn_weight)
    ddown_weight = torch.empty_like(down_weight)
    ddown_bias = torch.empty((C,), device=x.device, dtype=x.dtype)

    dgrn_out = torch.empty((M, D), device=x.device, dtype=x.dtype)
    dact = torch.empty((M, D), device=x.device, dtype=x.dtype)
    dup_pre = torch.empty((M, D), device=x.device, dtype=x.dtype)

    q_part = torch.empty((B, D, PB), device=x.device, dtype=torch.float32)
    dgamma_part = torch.empty((B, D, PB), device=x.device, dtype=torch.float32)
    dbeta_part = torch.empty((B, D, PB), device=x.device, dtype=torch.float32)

    q = torch.empty((B, D), device=x.device, dtype=torch.float32)
    dgamma_bd = torch.empty((B, D), device=x.device, dtype=torch.float32)
    dbeta_bd = torch.empty((B, D), device=x.device, dtype=torch.float32)
    sum_qg = torch.empty((B,), device=x.device, dtype=torch.float32)

    # Down projection backward.
    _linear_grad_weight_bias_kernel[(triton.cdiv(C, 16), triton.cdiv(D, 32))](
        dout,
        grn_out,
        ddown_weight,
        ddown_bias,
        M,
        D,
        C,
        BLOCK_O=16,
        BLOCK_I=32,
        BLOCK_M=64,
        num_warps=4,
    )

    _linear_grad_input_kernel[(triton.cdiv(M, 16), triton.cdiv(D, 32))](
        dout,
        down_weight,
        dgrn_out,
        M,
        D,
        C,
        BLOCK_M=16,
        BLOCK_I=32,
        BLOCK_K=32,
        num_warps=4,
    )

    # GRN backward.
    _grn_backward_partial_kernel[(B, triton.cdiv(D, 32), PB)](
        dgrn_out,
        act,
        grn_weight,
        g,
        inv_mean,
        q_part,
        dgamma_part,
        dbeta_part,
        B,
        P,
        D,
        PB,
        BLOCK_P=256,
        BLOCK_D=32,
        num_warps=4,
    )

    _grn_backward_reduce_bd_kernel[(B, triton.cdiv(D, 32))](
        q_part,
        dgamma_part,
        dbeta_part,
        q,
        dgamma_bd,
        dbeta_bd,
        B,
        D,
        PB,
        BLOCK_D=32,
        BLOCK_PB=triton.next_power_of_2(PB),
        num_warps=4,
    )

    _grn_sum_qg_kernel[(B,)](
        q,
        g,
        sum_qg,
        B,
        D,
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=8,
    )

    _grn_reduce_param_grads_kernel[(triton.cdiv(D, 32),)](
        dgamma_bd,
        dbeta_bd,
        dgrn_weight,
        dgrn_bias,
        B,
        D,
        BLOCK_D=32,
        num_warps=4,
    )

    total_hidden = M * D
    _grn_backward_apply_kernel[(triton.cdiv(total_hidden, 256),)](
        dgrn_out,
        act,
        grn_weight,
        g,
        inv_mean,
        q,
        sum_qg,
        dact,
        total_hidden,
        P,
        D,
        BLOCK=256,
        num_warps=4,
    )

    # GELU backward.
    _gelu_backward_kernel[(triton.cdiv(total_hidden, 256),)](
        up_pre,
        dact,
        dup_pre,
        total_hidden,
        BLOCK=256,
        num_warps=4,
    )

    # Up projection backward.
    _linear_grad_weight_bias_kernel[(triton.cdiv(D, 16), triton.cdiv(C, 32))](
        dup_pre,
        x,
        dup_weight,
        dup_bias,
        M,
        C,
        D,
        BLOCK_O=16,
        BLOCK_I=32,
        BLOCK_M=64,
        num_warps=4,
    )

    _linear_grad_input_kernel[(triton.cdiv(M, 16), triton.cdiv(C, 32))](
        dup_pre,
        up_weight,
        dx,
        M,
        C,
        D,
        BLOCK_M=16,
        BLOCK_I=32,
        BLOCK_K=32,
        num_warps=4,
    )

    return dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias


def _pack_into(dst, offset, src):
    n = src.numel()
    _pack_kernel[(triton.cdiv(n, 256),)](
        src,
        dst,
        offset,
        n,
        BLOCK=256,
        num_warps=4,
    )
    return offset + n


def kernel_function(
    x,
    up_weight,
    up_bias,
    grn_weight,
    grn_bias,
    down_weight,
    down_bias,
    upstream,
):
    """
    Full forward+backward step for the ConvNeXtV2 MLP.

    Returns a single flattened fp32 tensor containing:
      forward output, dx, d_up_weight, d_up_bias, d_grn_weight, d_grn_bias,
      d_down_weight, d_down_bias.

    All numerical computation and final flatten/pack are performed by Triton
    kernels. The wrapper only allocates outputs and launches kernels.
    """
    out, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(
        x,
        up_weight,
        up_bias,
        grn_weight,
        grn_bias,
        down_weight,
        down_bias,
    )
    grads = convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved)

    total = (
        out.numel()
        + grads[0].numel()
        + grads[1].numel()
        + grads[2].numel()
        + grads[3].numel()
        + grads[4].numel()
        + grads[5].numel()
        + grads[6].numel()
    )

    flat = torch.empty((total,), device=x.device, dtype=torch.float32)

    off = 0
    off = _pack_into(flat, off, out)
    off = _pack_into(flat, off, grads[0])
    off = _pack_into(flat, off, grads[1])
    off = _pack_into(flat, off, grads[2])
    off = _pack_into(flat, off, grads[3])
    off = _pack_into(flat, off, grads[4])
    off = _pack_into(flat, off, grads[5])
    _pack_into(flat, off, grads[6])

    return flat
# EVOLVE-BLOCK-END


# ── trusted primitive: vendor_gemm ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it.

def vendor_gemm(a, b):
    """Row-major 2-D GEMM through the vendor library (cuBLAS via ``torch.mm``).

    Two dimensions only, so the caller owns every reshape and the primitive
    cannot be handed a batched contraction it would silently loop. A
    ``.t()`` argument stays a metadata-only view -- cuBLAS consumes the
    transposed layout directly -- so ``vendor_gemm(x, w.t())`` costs one
    kernel and no copy.

    ``no_grad`` is not an optimisation: the pair differentiates itself, and a
    graph built here would be a second, hidden derivative.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"vendor_gemm is 2-D; got {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"vendor_gemm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    with torch.no_grad():
        return torch.mm(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'convnextv2_tiny_convnext_mlp', 'arguments': ['x', 'up_weight', 'up_bias', 'grn_weight', 'grn_bias', 'down_weight', 'down_bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'gradient_return_order': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'scalars': [], 'deployment_entry': 'convnextv2_tiny_convnext_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Convnextv2TinyConvnextMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `convnextv2_tiny_convnext_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        out, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return out

    @staticmethod
    def backward(ctx, dout):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dout = dout.contiguous() if (dout is not None and not dout.is_contiguous()) else dout
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias = convnextv2_tiny_convnext_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias


def convnextv2_tiny_convnext_mlp_deployment(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        grn_weight = grn_weight.reshape(-1, grn_weight.shape[-1])
        grn_bias = grn_bias.reshape(-1, grn_bias.shape[-1])
    out = Convnextv2TinyConvnextMlpFunction.apply(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Convnextv2TinyConvnextMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        super().__init__()
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.up_bias = (up_bias if isinstance(up_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(up_bias))
        self.grn_weight = (grn_weight if isinstance(grn_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_weight))
        self.grn_bias = (grn_bias if isinstance(grn_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_bias))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))
        self.down_bias = (down_bias if isinstance(down_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(down_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return convnextv2_tiny_convnext_mlp_deployment(x, self.up_weight, self.up_bias, self.grn_weight, self.grn_bias, self.down_weight, self.down_bias)


DEPLOYMENT_ENTRY = "convnextv2_tiny_convnext_mlp_deployment"
