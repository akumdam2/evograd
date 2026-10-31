"""adbench-tools artifact: task llama_3_2_1b.llama3_attention.causal_gqa_attention, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for a Llama-style causal grouped-query attention block:
#
#   1. causal GQA attention:
#        ctx[b, t, h, d] = softmax((q[b,h,t] @ k[b,kvh,:t]^T) / sqrt(D)) @ v[b,kvh,:t]
#      where kvh = h // (num_query_heads / num_kv_heads)
#
#   2. output projection:
#        out[b, t, n] = sum_i ctx_flat[b,t,i] * o_weight[n, i]
#
# Backward is implemented in Triton:
#   - d_o_weight = upstream^T @ ctx_flat
#   - d_ctx      = upstream @ o_weight
#   - attention backward recomputes softmax probabilities and accumulates
#     d_q, d_k, d_v.
#
# Important correctness note:
#   d_k and d_v receive contributions from many query heads / time positions.
#   Accumulating atomically directly into bf16 tensors loses too much precision
#   and caused rare tolerance failures.  We therefore accumulate d_k/d_v in
#   temporary fp32 buffers with fp32 atomics, then cast/store once to the final
#   gradient tensors.  This preserves the fused attention-backward computation
#   while avoiding bf16 atomic accumulation error.

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_kernel(x, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(x + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@triton.jit
def _attention_fwd_kernel(
    q, k, v, ctx,
    B: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    q_sb: tl.constexpr, q_sh: tl.constexpr, q_st: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_sh: tl.constexpr, k_st: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_sh: tl.constexpr, v_st: tl.constexpr, v_sd: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    tmp = pid // T
    h = tmp % H
    b = tmp // H

    group = H // HKV
    kvh = h // group

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    valid_t = offs_t <= t
    valid_d = offs_d < D

    q_vec = tl.load(
        q + b * q_sb + h * q_sh + t * q_st + offs_d * q_sd,
        mask=valid_d,
        other=0.0,
    ).to(tl.float32)

    k_mat = tl.load(
        k + b * k_sb + kvh * k_sh + offs_t[:, None] * k_st + offs_d[None, :] * k_sd,
        mask=valid_t[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k_mat * q_vec[None, :], axis=1) * SCALE
    scores = tl.where(valid_t, scores, -3.4028234663852886e38)

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom

    v_mat = tl.load(
        v + b * v_sb + kvh * v_sh + offs_t[:, None] * v_st + offs_d[None, :] * v_sd,
        mask=valid_t[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)

    out_vec = tl.sum(p[:, None] * v_mat, axis=0)

    # ctx is stored as contiguous [B, T, H, D]
    ctx_base = ((b * T + t) * H + h) * D
    tl.store(ctx + ctx_base + offs_d, out_vec, mask=valid_d)


@triton.jit
def _linear_fwd_kernel(
    x, w, y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
    w_sn: tl.constexpr, w_sk: tl.constexpr,
    y_sb: tl.constexpr, y_st: tl.constexpr, y_sn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k

        x_vals = tl.load(
            x + offs_m[:, None] * K + kk[None, :],
            mask=(offs_m[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        w_vals = tl.load(
            w + offs_n[None, :] * w_sn + kk[:, None] * w_sk,
            mask=(offs_n[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(x_vals, w_vals, acc)

    b = offs_m // T
    t = offs_m - b * T
    tl.store(
        y + b[:, None] * y_sb + t[:, None] * y_st + offs_n[None, :] * y_sn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _linear_dx_kernel(
    dy, w, dx,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
    dy_sb: tl.constexpr, dy_st: tl.constexpr, dy_sn: tl.constexpr,
    w_sn: tl.constexpr, w_sk: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    b = offs_m // T
    t = offs_m - b * T

    for n0 in range(0, N, BLOCK_N):
        nn = n0 + offs_n

        dy_vals = tl.load(
            dy + b[:, None] * dy_sb + t[:, None] * dy_st + nn[None, :] * dy_sn,
            mask=(offs_m[:, None] < M) & (nn[None, :] < N),
            other=0.0,
        )

        w_vals = tl.load(
            w + nn[:, None] * w_sn + offs_k[None, :] * w_sk,
            mask=(nn[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc = tl.dot(dy_vals, w_vals, acc)

    tl.store(
        dx + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _linear_dw_kernel(
    dy, x, dw,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr,
    dy_sb: tl.constexpr, dy_st: tl.constexpr, dy_sn: tl.constexpr,
    dw_sn: tl.constexpr, dw_sk: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        mm = m0 + offs_m
        b = mm // T
        t = mm - b * T

        dy_vals = tl.load(
            dy + b[None, :] * dy_sb + t[None, :] * dy_st + offs_n[:, None] * dy_sn,
            mask=(offs_n[:, None] < N) & (mm[None, :] < M),
            other=0.0,
        )

        x_vals = tl.load(
            x + mm[:, None] * K + offs_k[None, :],
            mask=(mm[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dy_vals, x_vals)

    tl.store(
        dw + offs_n[:, None] * dw_sn + offs_k[None, :] * dw_sk,
        acc,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
    )


@triton.jit
def _attention_bwd_kernel(
    q, k, v, dctx, dq, dk_acc, dv_acc,
    B: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    q_sb: tl.constexpr, q_sh: tl.constexpr, q_st: tl.constexpr, q_sd: tl.constexpr,
    k_sb: tl.constexpr, k_sh: tl.constexpr, k_st: tl.constexpr, k_sd: tl.constexpr,
    v_sb: tl.constexpr, v_sh: tl.constexpr, v_st: tl.constexpr, v_sd: tl.constexpr,
    dq_sb: tl.constexpr, dq_sh: tl.constexpr, dq_st: tl.constexpr, dq_sd: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    tmp = pid // T
    h = tmp % H
    b = tmp // H

    group = H // HKV
    kvh = h // group

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    valid_t = offs_t <= t
    valid_d = offs_d < D

    q_vec = tl.load(
        q + b * q_sb + h * q_sh + t * q_st + offs_d * q_sd,
        mask=valid_d,
        other=0.0,
    ).to(tl.float32)

    k_mat = tl.load(
        k + b * k_sb + kvh * k_sh + offs_t[:, None] * k_st + offs_d[None, :] * k_sd,
        mask=valid_t[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)

    v_mat = tl.load(
        v + b * v_sb + kvh * v_sh + offs_t[:, None] * v_st + offs_d[None, :] * v_sd,
        mask=valid_t[:, None] & valid_d[None, :],
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k_mat * q_vec[None, :], axis=1) * SCALE
    scores = tl.where(valid_t, scores, -3.4028234663852886e38)

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom

    dctx_base = ((b * T + t) * H + h) * D
    do_vec = tl.load(dctx + dctx_base + offs_d, mask=valid_d, other=0.0).to(tl.float32)

    # dV += p * dO.
    # Accumulate into contiguous fp32 temp [B, HKV, T, D] to avoid bf16 atomic error.
    acc_base = ((b * HKV + kvh) * T + offs_t[:, None]) * D + offs_d[None, :]
    dv_add = p[:, None] * do_vec[None, :]
    tl.atomic_add(
        dv_acc + acc_base,
        dv_add,
        sem="relaxed",
        mask=valid_t[:, None] & valid_d[None, :],
    )

    # Softmax backward.
    dp = tl.sum(v_mat * do_vec[None, :], axis=1)
    delta = tl.sum(p * dp, axis=0)
    ds = p * (dp - delta) * SCALE
    ds = tl.where(valid_t, ds, 0.0)

    # dQ = sum_t ds[t] * K[t]
    dq_vec = tl.sum(ds[:, None] * k_mat, axis=0)
    tl.store(
        dq + b * dq_sb + h * dq_sh + t * dq_st + offs_d * dq_sd,
        dq_vec,
        mask=valid_d,
    )

    # dK += ds * Q.
    # Accumulate into contiguous fp32 temp [B, HKV, T, D] to avoid bf16 atomic error.
    dk_add = ds[:, None] * q_vec[None, :]
    tl.atomic_add(
        dk_acc + acc_base,
        dk_add,
        sem="relaxed",
        mask=valid_t[:, None] & valid_d[None, :],
    )


@triton.jit
def _acc_to_grad_kernel(
    acc, grad,
    n_elements: tl.constexpr,
    HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    g_sb: tl.constexpr, g_sh: tl.constexpr, g_st: tl.constexpr, g_sd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    d = offs % D
    tmp0 = offs // D
    t = tmp0 % T
    tmp1 = tmp0 // T
    h = tmp1 % HKV
    b = tmp1 // HKV

    vals = tl.load(acc + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        grad + b * g_sb + h * g_sh + t * g_st + d * g_sd,
        vals,
        mask=mask,
    )


@triton.jit
def _pack_flat_kernel(src, dst, base: tl.constexpr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + base + offs, vals, mask=mask)


def _logical_shape_and_strides(q, k, v):
    """
    Supports the two common layouts:
      - q: [B, H, T, D], k/v: [B, HKV, T, D]
      - q: [B, T, H, D], k/v: [B, T, HKV, D]
    Returns logical B,H,HKV,T,D and logical strides ordered as (b,h,t,d).
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    # GQA usually has H != HKV, making this layout inference unambiguous.
    if q.shape[1] == k.shape[1] and q.shape[2] != k.shape[2]:
        # [B, T, H, D]
        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        D = q.shape[3]
        HKV = k.shape[2]

        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        layout = 1
    else:
        # [B, H, T, D]
        B = q.shape[0]
        H = q.shape[1]
        T = q.shape[2]
        D = q.shape[3]
        HKV = k.shape[1]

        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        layout = 0

    return B, H, HKV, T, D, q_strides, k_strides, v_strides, layout


def _grad_strides_for_layout(t, layout):
    if layout == 1:
        return (t.stride(0), t.stride(2), t.stride(1), t.stride(3))
    return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))


def llama3_attention_forward_with_saved(q, k, v, o_weight):
    """
    Forward pass for causal GQA attention fused with the output projection.

    Fusion note:
      The attention softmax and value accumulation are fused into one Triton
      row kernel.  The final output projection is a separate Triton matmul
      because it is a dense all-channel projection over the attention result;
      fully fusing it into each attention row would require each attention
      program to compute all output channels and would greatly increase work
      and register pressure.
    """
    B, H, HKV, T, D, qst, kst, vst, _layout = _logical_shape_and_strides(q, k, v)

    Kflat = H * D
    Nout = o_weight.shape[0]
    assert o_weight.shape[1] == Kflat
    assert H % HKV == 0

    ctx = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
    out = torch.empty((B, T, Nout), device=q.device, dtype=q.dtype)

    block_t = triton.next_power_of_2(T)
    block_d = triton.next_power_of_2(D)
    scale = 1.0 / (D ** 0.5)

    grid_attn = (B * H * T,)
    _attention_fwd_kernel[grid_attn](
        q, k, v, ctx,
        B, H, HKV, T, D,
        qst[0], qst[1], qst[2], qst[3],
        kst[0], kst[1], kst[2], kst[3],
        vst[0], vst[1], vst[2], vst[3],
        scale,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=8,
    )

    M = B * T
    grid_linear = (
        triton.cdiv(M, 16),
        triton.cdiv(Nout, 16),
    )
    _linear_fwd_kernel[grid_linear](
        ctx, o_weight, out,
        M, Nout, Kflat, T,
        o_weight.stride(0), o_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
        num_stages=4,
    )

    saved = (q, k, v, o_weight, ctx)
    return out, saved


def llama3_attention_backward_from_saved(upstream, saved):
    """
    Backward pass for the fused causal GQA attention + output projection.

    Returns gradients with respect to:
      (q, k, v, o_weight)
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    q, k, v, o_weight, ctx = saved

    B, H, HKV, T, D, qst, kst, vst, layout = _logical_shape_and_strides(q, k, v)
    Kflat = H * D
    Nout = o_weight.shape[0]
    M = B * T

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dctx = torch.empty_like(ctx)
    do_weight = torch.empty_like(o_weight)

    # dK and dV are accumulated by many programs.  Use fp32 accumulation buffers
    # and cast once to the final bf16/fp32 gradient tensors.
    dk_acc = torch.empty((B, HKV, T, D), device=q.device, dtype=torch.float32)
    dv_acc = torch.empty((B, HKV, T, D), device=q.device, dtype=torch.float32)

    block_zero = 1024
    acc_numel = B * HKV * T * D
    _zero_kernel[(triton.cdiv(acc_numel, block_zero),)](
        dk_acc, acc_numel, BLOCK=block_zero, num_warps=4
    )
    _zero_kernel[(triton.cdiv(acc_numel, block_zero),)](
        dv_acc, acc_numel, BLOCK=block_zero, num_warps=4
    )

    # dctx = upstream @ o_weight
    grid_dx = (
        triton.cdiv(M, 16),
        triton.cdiv(Kflat, 16),
    )
    _linear_dx_kernel[grid_dx](
        upstream, o_weight, dctx,
        M, Nout, Kflat, T,
        upstream.stride(0), upstream.stride(1), upstream.stride(2),
        o_weight.stride(0), o_weight.stride(1),
        BLOCK_M=16,
        BLOCK_K=16,
        BLOCK_N=32,
        num_warps=4,
        num_stages=4,
    )

    # do_weight = upstream^T @ ctx
    grid_dw = (
        triton.cdiv(Nout, 16),
        triton.cdiv(Kflat, 16),
    )
    _linear_dw_kernel[grid_dw](
        upstream, ctx, do_weight,
        M, Nout, Kflat, T,
        upstream.stride(0), upstream.stride(1), upstream.stride(2),
        do_weight.stride(0), do_weight.stride(1),
        BLOCK_N=16,
        BLOCK_K=16,
        BLOCK_M=32,
        num_warps=4,
        num_stages=4,
    )

    dqst = _grad_strides_for_layout(dq, layout)
    dkst = _grad_strides_for_layout(dk, layout)
    dvst = _grad_strides_for_layout(dv, layout)

    block_t = triton.next_power_of_2(T)
    block_d = triton.next_power_of_2(D)
    scale = 1.0 / (D ** 0.5)

    grid_attn = (B * H * T,)
    _attention_bwd_kernel[grid_attn](
        q, k, v, dctx, dq, dk_acc, dv_acc,
        B, H, HKV, T, D,
        qst[0], qst[1], qst[2], qst[3],
        kst[0], kst[1], kst[2], kst[3],
        vst[0], vst[1], vst[2], vst[3],
        dqst[0], dqst[1], dqst[2], dqst[3],
        scale,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=8,
    )

    # Cast/store fp32 accumulation buffers to final gradient tensors using the
    # original logical layout/strides.
    block_copy = 1024
    _acc_to_grad_kernel[(triton.cdiv(acc_numel, block_copy),)](
        dk_acc, dk,
        acc_numel,
        HKV, T, D,
        dkst[0], dkst[1], dkst[2], dkst[3],
        BLOCK=block_copy,
        num_warps=4,
    )
    _acc_to_grad_kernel[(triton.cdiv(acc_numel, block_copy),)](
        dv_acc, dv,
        acc_numel,
        HKV, T, D,
        dvst[0], dvst[1], dvst[2], dvst[3],
        BLOCK=block_copy,
        num_warps=4,
    )

    return dq, dk, dv, do_weight


def kernel_function(q, k, v, o_weight, upstream):
    """
    Full forward+backward step used by the benchmark harness.

    This wrapper launches Triton kernels for:
      1. causal GQA attention forward,
      2. output projection forward,
      3. output projection backward,
      4. causal GQA attention backward,
      5. packing outputs/grads into the evaluator's flattened convention.

    No PyTorch numerical compute is used in this wrapper.
    """
    out, saved = llama3_attention_forward_with_saved(q, k, v, o_weight)
    grads = llama3_attention_backward_from_saved(upstream, saved)
    dq, dk, dv, do_weight = grads

    total = out.numel() + dq.numel() + dk.numel() + dv.numel() + do_weight.numel()
    packed = torch.empty((total,), device=q.device, dtype=torch.float32)

    block = 1024
    base = 0

    _pack_flat_kernel[(triton.cdiv(out.numel(), block),)](
        out, packed, base, out.numel(), BLOCK=block, num_warps=4
    )
    base += out.numel()

    _pack_flat_kernel[(triton.cdiv(dq.numel(), block),)](
        dq, packed, base, dq.numel(), BLOCK=block, num_warps=4
    )
    base += dq.numel()

    _pack_flat_kernel[(triton.cdiv(dk.numel(), block),)](
        dk, packed, base, dk.numel(), BLOCK=block, num_warps=4
    )
    base += dk.numel()

    _pack_flat_kernel[(triton.cdiv(dv.numel(), block),)](
        dv, packed, base, dv.numel(), BLOCK=block, num_warps=4
    )
    base += dv.numel()

    _pack_flat_kernel[(triton.cdiv(do_weight.numel(), block),)](
        do_weight, packed, base, do_weight.numel(), BLOCK=block, num_warps=4
    )

    return packed
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


# ── trusted primitive: vendor_sdpa ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it. The fused-attention kernel eager PyTorch itself runs, chosen the
# way its dispatcher chooses it: cuDNN in a training configuration, flash
# under torch.use_deterministic_algorithms (cuDNN's backward is not
# reproducible, so the deterministic dispatcher excludes it). Verification
# runs deterministic and compares flash with flash; timing runs as training
# does and compares cuDNN with cuDNN.

def _vendor_sdpa_deterministic():
    return torch.are_deterministic_algorithms_enabled()


def vendor_sdpa_forward(q, k, v, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, forward.

    ``q`` is ``[B, HQ, T, D]``; ``k`` and ``v`` are ``[B, HK, T, D]`` with
    ``HK`` dividing ``HQ`` (grouped-query attention is the kernel's own). No
    mask, no dropout. Returns ``(out, logsumexp)``; ``logsumexp`` is what the
    backward needs and is the caller's to save (its layout is the backend's;
    pass it back unchanged). ``no_grad`` because the pair differentiates itself.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("vendor_sdpa_forward takes [B, H, T, D] tensors")
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, bool(is_causal), False, scale=scale)
        else:
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
                q, k, v, None, True, 0.0, bool(is_causal), False, scale=scale)
    return out, logsumexp


def vendor_sdpa_backward(dout, q, k, v, out, logsumexp, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, backward.

    Takes the forward's inputs and both of its results, from the same mode
    (deterministic or not) as the forward; returns ``(dq, dk, dv)`` with the
    shapes of ``q``, ``k`` and ``v``.
    """
    if dout.shape != out.shape:
        raise ValueError(f"vendor_sdpa_backward: dout {tuple(dout.shape)} vs out {tuple(out.shape)}")
    # No dropout, no mask, dense [B, H, T, D] inputs: philox state, attn_bias
    # and the nested-layout cum_seq tensors stay undefined (None). A defined
    # cum_seq tensor would select the ragged layout.
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, q.shape[2], k.shape[2], 0.0,
                bool(is_causal), None, None, scale=scale)
        else:
            dq, dk, dv = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, None, None, None,
                q.shape[2], k.shape[2], 0.0, bool(is_causal), scale=scale)
    return dq, dk, dv


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_attention', 'arguments': ['q', 'k', 'v', 'o_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dq', 'dk', 'dv', 'do_weight'], 'gradient_return_order': ['dq', 'dk', 'dv', 'do_weight'], 'scalars': [], 'deployment_entry': 'llama3_attention_deployment', 'allowed_primitives': ['vendor_gemm', 'vendor_sdpa']}


class Llama3AttentionFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_attention`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, o_weight):
        out, saved = llama3_attention_forward_with_saved(q, k, v, o_weight)
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
        dq, dk, dv, do_weight = llama3_attention_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, do_weight


def llama3_attention_deployment(q, k, v, o_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Llama3AttentionFunction.apply(q, k, v, o_weight)


class Llama3AttentionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, o_weight):
        super().__init__()
        self.o_weight = (o_weight if isinstance(o_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(o_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v):
        return llama3_attention_deployment(q, k, v, self.o_weight)


DEPLOYMENT_ENTRY = "llama3_attention_deployment"
