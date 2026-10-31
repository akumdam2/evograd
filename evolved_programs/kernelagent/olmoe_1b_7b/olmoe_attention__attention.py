"""adbench-tools artifact: task olmoe_1b_7b.olmoe_attention.attention, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the OLMoE attention pair:
#   forward:  causal scaled-dot-product attention(q, k, v) followed by output projection
#   backward: gradients w.r.t. q, k, v, and o_weight from an upstream output gradient
#
# Fix relative to the failing version:
# - dK/dV were atomically accumulated directly into bf16 tensors.  That rounds
#   every atomic contribution and produces rare but test-failing bf16 ULP errors.
# - This version accumulates dK/dV atomics into fp32 temporary buffers, then casts
#   once to the output dtype.  The backward attention computation remains fused
#   per query row; the final cast is a required epilogue because Triton atomics
#   cannot both fp32-accumulate and directly materialize bf16 with one rounding.

import torch
import triton
import triton.language as tl


@triton.jit
def _load_qkv(ptr, b, h, s, d, st0, st1, st2, st3, LAYOUT: tl.constexpr):
    # LAYOUT == 0: [B, H, S, D]
    # LAYOUT == 1: [B, S, H, D]
    if LAYOUT == 0:
        return ptr + b * st0 + h * st1 + s * st2 + d * st3
    else:
        return ptr + b * st0 + s * st1 + h * st2 + d * st3


@triton.jit
def _attention_forward_z_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    q_s0: tl.constexpr,
    q_s1: tl.constexpr,
    q_s2: tl.constexpr,
    q_s3: tl.constexpr,
    k_s0: tl.constexpr,
    k_s1: tl.constexpr,
    k_s2: tl.constexpr,
    k_s3: tl.constexpr,
    v_s0: tl.constexpr,
    v_s1: tl.constexpr,
    v_s2: tl.constexpr,
    v_s3: tl.constexpr,
    scale: tl.constexpr,
    GROUP: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid % S
    tmp = pid // S
    h = tmp % H
    b = tmp // H
    kh = h // GROUP

    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D

    q_ptrs = _load_qkv(q_ptr, b, h, m, offs_d, q_s0, q_s1, q_s2, q_s3, LAYOUT)
    q = tl.load(q_ptrs, mask=dmask, other=0.0).to(tl.float32)

    row_max = tl.full((), -3.4028234663852886e38, dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = (offs_n < S) & (offs_n <= m)

        k_ptrs = _load_qkv(
            k_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            k_s0,
            k_s1,
            k_s2,
            k_s3,
            LAYOUT,
        )
        kval = tl.load(k_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(kval * q[None, :], axis=1) * scale
        scores = tl.where(nmask, scores, -3.4028234663852886e38)
        row_max = tl.maximum(row_max, tl.max(scores, axis=0))

    denom = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = (offs_n < S) & (offs_n <= m)

        k_ptrs = _load_qkv(
            k_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            k_s0,
            k_s1,
            k_s2,
            k_s3,
            LAYOUT,
        )
        v_ptrs = _load_qkv(
            v_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            v_s0,
            v_s1,
            v_s2,
            v_s3,
            LAYOUT,
        )

        kval = tl.load(k_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        vval = tl.load(v_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

        scores = tl.sum(kval * q[None, :], axis=1) * scale
        scores = tl.where(nmask, scores, -3.4028234663852886e38)
        p = tl.exp(scores - row_max)
        p = tl.where(nmask, p, 0.0)

        denom += tl.sum(p, axis=0)
        acc += tl.sum(vval * p[:, None], axis=0)

    out = acc / denom
    z_offsets = (b * S + m) * C + h * D + offs_d
    tl.store(z_ptr + z_offsets, out, mask=dmask)


@triton.jit
def _z_w_forward_kernel(
    z_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        kk = k0 + offs_k
        z = tl.load(
            z_ptr + offs_m[:, None] * K + kk[None, :],
            mask=(offs_m[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + kk[:, None],
            mask=(offs_n[None, :] < N) & (kk[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(z, w, acc)

    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _dz_from_dout_w_kernel(
    dout_ptr,
    w_ptr,
    dz_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    do_s0: tl.constexpr,
    do_s1: tl.constexpr,
    do_s2: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    bidx = offs_m // S
    sidx = offs_m - bidx * S

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n0 in tl.range(0, N, BLOCK_N):
        nn = n0 + offs_n
        dout = tl.load(
            dout_ptr + bidx[:, None] * do_s0 + sidx[:, None] * do_s1 + nn[None, :] * do_s2,
            mask=(offs_m[:, None] < M) & (nn[None, :] < N),
            other=0.0,
        )
        w = tl.load(
            w_ptr + nn[:, None] * K + offs_k[None, :],
            mask=(nn[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc = tl.dot(dout, w, acc)

    tl.store(
        dz_ptr + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _dw_kernel(
    dout_ptr,
    z_ptr,
    dw_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    do_s0: tl.constexpr,
    do_s1: tl.constexpr,
    do_s2: tl.constexpr,
    S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        mm = m0 + offs_m
        bidx = mm // S
        sidx = mm - bidx * S

        dout = tl.load(
            dout_ptr + bidx[None, :] * do_s0 + sidx[None, :] * do_s1 + offs_n[:, None] * do_s2,
            mask=(offs_n[:, None] < N) & (mm[None, :] < M),
            other=0.0,
        )
        z = tl.load(
            z_ptr + mm[:, None] * K + offs_k[None, :],
            mask=(mm[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(dout, z)

    tl.store(
        dw_ptr + offs_n[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
    )


@triton.jit
def _zero_kernel(ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=offs < n_elements)


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + offs, x, mask=mask)


@triton.jit
def _attention_backward_query_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    dz_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    q_s0: tl.constexpr,
    q_s1: tl.constexpr,
    q_s2: tl.constexpr,
    q_s3: tl.constexpr,
    k_s0: tl.constexpr,
    k_s1: tl.constexpr,
    k_s2: tl.constexpr,
    k_s3: tl.constexpr,
    v_s0: tl.constexpr,
    v_s1: tl.constexpr,
    v_s2: tl.constexpr,
    v_s3: tl.constexpr,
    gq_s0: tl.constexpr,
    gq_s1: tl.constexpr,
    gq_s2: tl.constexpr,
    gq_s3: tl.constexpr,
    gk_s0: tl.constexpr,
    gk_s1: tl.constexpr,
    gk_s2: tl.constexpr,
    gk_s3: tl.constexpr,
    gv_s0: tl.constexpr,
    gv_s1: tl.constexpr,
    gv_s2: tl.constexpr,
    gv_s3: tl.constexpr,
    scale: tl.constexpr,
    GROUP: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid % S
    tmp = pid // S
    h = tmp % H
    b = tmp // H
    kh = h // GROUP

    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D

    q_ptrs = _load_qkv(q_ptr, b, h, m, offs_d, q_s0, q_s1, q_s2, q_s3, LAYOUT)
    q = tl.load(q_ptrs, mask=dmask, other=0.0).to(tl.float32)

    dz = tl.load(dz_ptr + (b * S + m) * C + h * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + (b * S + m) * C + h * D + offs_d, mask=dmask, other=0.0).to(tl.float32)
    delta = tl.sum(dz * z, axis=0)

    row_max = tl.full((), -3.4028234663852886e38, dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = (offs_n < S) & (offs_n <= m)

        k_ptrs = _load_qkv(
            k_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            k_s0,
            k_s1,
            k_s2,
            k_s3,
            LAYOUT,
        )
        kval = tl.load(k_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(kval * q[None, :], axis=1) * scale
        scores = tl.where(nmask, scores, -3.4028234663852886e38)
        row_max = tl.maximum(row_max, tl.max(scores, axis=0))

    denom = tl.zeros((), dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = (offs_n < S) & (offs_n <= m)

        k_ptrs = _load_qkv(
            k_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            k_s0,
            k_s1,
            k_s2,
            k_s3,
            LAYOUT,
        )
        kval = tl.load(k_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(kval * q[None, :], axis=1) * scale
        scores = tl.where(nmask, scores, -3.4028234663852886e38)

        p_unnorm = tl.exp(scores - row_max)
        p_unnorm = tl.where(nmask, p_unnorm, 0.0)
        denom += tl.sum(p_unnorm, axis=0)

    dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for start_n in tl.range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = (offs_n < S) & (offs_n <= m)

        k_ptrs = _load_qkv(
            k_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            k_s0,
            k_s1,
            k_s2,
            k_s3,
            LAYOUT,
        )
        v_ptrs = _load_qkv(
            v_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            v_s0,
            v_s1,
            v_s2,
            v_s3,
            LAYOUT,
        )

        kval = tl.load(k_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        vval = tl.load(v_ptrs, mask=nmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)

        scores = tl.sum(kval * q[None, :], axis=1) * scale
        scores = tl.where(nmask, scores, -3.4028234663852886e38)

        p = tl.exp(scores - row_max) / denom
        p = tl.where(nmask, p, 0.0)

        dp = tl.sum(vval * dz[None, :], axis=1)
        ds = p * (dp - delta) * scale
        ds = tl.where(nmask, ds, 0.0)

        dq_acc += tl.sum(kval * ds[:, None], axis=0)

        dk_add = ds[:, None] * q[None, :]
        dv_add = p[:, None] * dz[None, :]

        dk_ptrs = _load_qkv(
            dk_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            gk_s0,
            gk_s1,
            gk_s2,
            gk_s3,
            LAYOUT,
        )
        dv_ptrs = _load_qkv(
            dv_ptr,
            b,
            kh,
            offs_n[:, None],
            offs_d[None, :],
            gv_s0,
            gv_s1,
            gv_s2,
            gv_s3,
            LAYOUT,
        )

        # dk_ptr/dv_ptr are fp32 accumulation buffers.  This avoids bf16
        # per-atomic rounding and matches the one-rounding behavior expected by
        # the reference gradients.
        tl.atomic_add(dk_ptrs, dk_add, sem="relaxed", mask=nmask[:, None] & dmask[None, :])
        tl.atomic_add(dv_ptrs, dv_add, sem="relaxed", mask=nmask[:, None] & dmask[None, :])

    dq_ptrs = _load_qkv(dq_ptr, b, h, m, offs_d, gq_s0, gq_s1, gq_s2, gq_s3, LAYOUT)
    tl.store(dq_ptrs, dq_acc, mask=dmask)


@triton.jit
def _pack_kernel(
    out_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dw_ptr,
    flat_ptr,
    n_out: tl.constexpr,
    n_dq: tl.constexpr,
    n_dk: tl.constexpr,
    n_dv: tl.constexpr,
    n_dw: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    x = tl.zeros((BLOCK,), dtype=tl.float32)

    m0 = offs < n_out
    x = tl.where(m0, tl.load(out_ptr + offs, mask=mask & m0, other=0.0).to(tl.float32), x)

    o1 = offs - n_out
    m1 = (offs >= n_out) & (offs < n_out + n_dq)
    x = tl.where(m1, tl.load(dq_ptr + o1, mask=mask & m1, other=0.0).to(tl.float32), x)

    o2 = offs - n_out - n_dq
    m2 = (offs >= n_out + n_dq) & (offs < n_out + n_dq + n_dk)
    x = tl.where(m2, tl.load(dk_ptr + o2, mask=mask & m2, other=0.0).to(tl.float32), x)

    o3 = offs - n_out - n_dq - n_dk
    m3 = (offs >= n_out + n_dq + n_dk) & (offs < n_out + n_dq + n_dk + n_dv)
    x = tl.where(m3, tl.load(dv_ptr + o3, mask=mask & m3, other=0.0).to(tl.float32), x)

    o4 = offs - n_out - n_dq - n_dk - n_dv
    m4 = offs >= (n_out + n_dq + n_dk + n_dv)
    x = tl.where(m4, tl.load(dw_ptr + o4, mask=mask & m4, other=0.0).to(tl.float32), x)

    tl.store(flat_ptr + offs, x, mask=mask)


def _layout_and_dims(q, k, o_weight):
    # Prefer common HF attention layout [B, H, S, D].  If that cannot match the
    # projection's input dimension, use [B, S, H, D].
    if q.shape[1] * q.shape[3] == o_weight.shape[1]:
        layout = 0
        B, H, S, D = q.shape
        HK = k.shape[1]
    else:
        layout = 1
        B, S, H, D = q.shape
        HK = k.shape[2]
    return layout, B, H, HK, S, D


def _block_d(D: int) -> int:
    return triton.next_power_of_2(D)


def olmoe_attention_forward_with_saved(q, k, v, o_weight):
    """
    Computes causal scaled-dot-product attention followed by output projection.
    All numerical work is done in Triton kernels.
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and o_weight.is_cuda

    layout, B, H, HK, S, D = _layout_and_dims(q, k, o_weight)
    assert H % HK == 0

    C = H * D
    N = o_weight.shape[0]
    M = B * S

    z = torch.empty((B, S, C), device=q.device, dtype=q.dtype)
    out = torch.empty((B, S, N), device=q.device, dtype=q.dtype)

    bd = _block_d(D)
    bn = 32
    scale = D ** -0.5
    group = H // HK

    _attention_forward_z_kernel[(B * H * S,)](
        q,
        k,
        v,
        z,
        B,
        H,
        HK,
        S,
        D,
        C,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        scale,
        group,
        layout,
        BLOCK_N=bn,
        BLOCK_D=bd,
    )

    grid_proj = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _z_w_forward_kernel[grid_proj](
        z,
        o_weight,
        out,
        M,
        C,
        N,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
    )

    saved = (q, k, v, o_weight, z, layout)
    return out, saved


def olmoe_attention_backward_from_saved(upstream, saved):
    """
    Returns gradients for q, k, v, and o_weight.  Attention probabilities are
    recomputed in the Triton backward kernel instead of saved.
    """
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    q, k, v, o_weight, z, layout = saved
    layout2, B, H, HK, S, D = _layout_and_dims(q, k, o_weight)
    assert layout == layout2

    C = H * D
    N = o_weight.shape[0]
    M = B * S

    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    dw = torch.empty(o_weight.shape, device=o_weight.device, dtype=o_weight.dtype)
    dz = torch.empty((B, S, C), device=q.device, dtype=q.dtype)

    # fp32 accumulation buffers for atomic dK/dV updates.
    dk_acc = torch.empty(k.shape, device=k.device, dtype=torch.float32)
    dv_acc = torch.empty(v.shape, device=v.device, dtype=torch.float32)

    grid_dz = (triton.cdiv(M, 16), triton.cdiv(C, 16))
    _dz_from_dout_w_kernel[grid_dz](
        dout,
        o_weight,
        dz,
        M,
        C,
        N,
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        S,
        BLOCK_M=16,
        BLOCK_K=16,
        BLOCK_N=32,
    )

    grid_dw = (triton.cdiv(N, 16), triton.cdiv(C, 16))
    _dw_kernel[grid_dw](
        dout,
        z,
        dw,
        M,
        C,
        N,
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        S,
        BLOCK_N=16,
        BLOCK_K=16,
        BLOCK_M=32,
    )

    zero_block = 1024
    _zero_kernel[(triton.cdiv(dk_acc.numel(), zero_block),)](
        dk_acc, dk_acc.numel(), BLOCK=zero_block
    )
    _zero_kernel[(triton.cdiv(dv_acc.numel(), zero_block),)](
        dv_acc, dv_acc.numel(), BLOCK=zero_block
    )

    bd = _block_d(D)
    bn = 32
    scale = D ** -0.5
    group = H // HK

    _attention_backward_query_kernel[(B * H * S,)](
        q,
        k,
        v,
        z,
        dz,
        dq,
        dk_acc,
        dv_acc,
        B,
        H,
        HK,
        S,
        D,
        C,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dk_acc.stride(0),
        dk_acc.stride(1),
        dk_acc.stride(2),
        dk_acc.stride(3),
        dv_acc.stride(0),
        dv_acc.stride(1),
        dv_acc.stride(2),
        dv_acc.stride(3),
        scale,
        group,
        layout,
        BLOCK_N=bn,
        BLOCK_D=bd,
    )

    # Single-rounding cast epilogue from fp32 accumulation buffers to gradient dtype.
    _cast_kernel[(triton.cdiv(dk.numel(), zero_block),)](
        dk_acc, dk, dk.numel(), BLOCK=zero_block
    )
    _cast_kernel[(triton.cdiv(dv.numel(), zero_block),)](
        dv_acc, dv, dv.numel(), BLOCK=zero_block
    )

    return dq, dk, dv, dw


def kernel_function(q, k, v, o_weight, upstream):
    """
    Forward+backward entrypoint required by the evaluator.

    Computes:
      1. output = linear(causal_sdpa(q, k, v), o_weight)
      2. gradients w.r.t. q, k, v, and o_weight from upstream
      3. packs (output, dq, dk, dv, dw) into one flat fp32 tensor
    """
    out, saved = olmoe_attention_forward_with_saved(q, k, v, o_weight)
    dq, dk, dv, dw = olmoe_attention_backward_from_saved(upstream, saved)

    n_out = out.numel()
    n_dq = dq.numel()
    n_dk = dk.numel()
    n_dv = dv.numel()
    n_dw = dw.numel()
    total = n_out + n_dq + n_dk + n_dv + n_dw

    flat = torch.empty((total,), device=q.device, dtype=torch.float32)
    block = 1024

    _pack_kernel[(triton.cdiv(total, block),)](
        out,
        dq,
        dk,
        dv,
        dw,
        flat,
        n_out,
        n_dq,
        n_dk,
        n_dv,
        n_dw,
        total,
        BLOCK=block,
    )

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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'olmoe_attention', 'arguments': ['q', 'k', 'v', 'o_weight'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dq', 'dk', 'dv', 'do_weight'], 'gradient_return_order': ['dq', 'dk', 'dv', 'do_weight'], 'scalars': [], 'deployment_entry': 'olmoe_attention_deployment', 'allowed_primitives': ['vendor_gemm', 'vendor_sdpa']}


class OlmoeAttentionFunction(torch.autograd.Function):
    """Static autograd wiring for `olmoe_attention`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, o_weight):
        y, saved = olmoe_attention_forward_with_saved(q, k, v, o_weight)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return y

    @staticmethod
    def backward(ctx, dy):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dy = dy.contiguous() if (dy is not None and not dy.is_contiguous()) else dy
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dq, dk, dv, do_weight = olmoe_attention_backward_from_saved(
            dy, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, do_weight


def olmoe_attention_deployment(q, k, v, o_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return OlmoeAttentionFunction.apply(q, k, v, o_weight)


class OlmoeAttentionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, o_weight):
        super().__init__()
        self.o_weight = (o_weight if isinstance(o_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(o_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v):
        return olmoe_attention_deployment(q, k, v, self.o_weight)


DEPLOYMENT_ENTRY = "olmoe_attention_deployment"
