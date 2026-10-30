"""adbench-tools artifact: task qwen3_0_6b.qwen3_attention.causal_gqa_attention, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, attn_ptr,
    B: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    q_s0: tl.constexpr, q_s1: tl.constexpr, q_s2: tl.constexpr, q_s3: tl.constexpr,
    k_s0: tl.constexpr, k_s1: tl.constexpr, k_s2: tl.constexpr, k_s3: tl.constexpr,
    v_s0: tl.constexpr, v_s1: tl.constexpr, v_s2: tl.constexpr, v_s3: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Causal GQA attention forward.

    q:    [B, H,   T, D]
    k/v:  [B, HKV, T, D]
    attn: contiguous [B, T, H, D]
    """
    pid = tl.program_id(0)
    t = pid % T
    h = (pid // T) % H
    b = pid // (T * H)

    GROUP: tl.constexpr = H // HKV
    hk = h // GROUP

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    t_mask = offs_t < T
    d_mask = offs_d < D

    q_vals = tl.load(
        q_ptr + b * q_s0 + h * q_s1 + t * q_s2 + offs_d * q_s3,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    k_vals = tl.load(
        k_ptr
        + b * k_s0
        + hk * k_s1
        + offs_t[:, None] * k_s2
        + offs_d[None, :] * k_s3,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k_vals * q_vals[None, :], axis=1) * SCALE
    causal_mask = (offs_t <= t) & t_mask
    scores = tl.where(causal_mask, scores, -3.4028234663852886e38)

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom

    v_vals = tl.load(
        v_ptr
        + b * v_s0
        + hk * v_s1
        + offs_t[:, None] * v_s2
        + offs_d[None, :] * v_s3,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    out = tl.sum(p[:, None] * v_vals, axis=0)

    out_offsets = ((b * T + t) * H + h) * D + offs_d
    tl.store(attn_ptr + out_offsets, out, mask=d_mask)


@triton.jit
def _proj_forward_kernel(
    x_ptr, w_ptr, y_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    w_s0: tl.constexpr, w_s1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Projection forward: y = x @ w.T.

    x: contiguous [M, K]
    w: [N, K]
    y: contiguous [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )

        w = tl.load(
            w_ptr + offs_n[None, :] * w_s0 + k_idxs[:, None] * w_s1,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        )

        acc += tl.dot(x, w)

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _zero_kernel(ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@triton.jit
def _cast_from_fp32_kernel(src_ptr, dst_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    """
    Copy/cast an fp32 accumulation buffer into the requested output dtype.

    This is used for dK/dV after fp32 atomic accumulation.  The previous
    implementation atomically accumulated directly into bf16 dK/dV buffers,
    which introduced order-dependent bf16 rounding error and caused rare
    tolerance failures.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + offs, vals, mask=mask)


@triton.jit
def _proj_dx_kernel(
    dy_ptr, w_ptr, dx_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, T: tl.constexpr,
    dy_s0: tl.constexpr, dy_s1: tl.constexpr, dy_s2: tl.constexpr,
    w_s0: tl.constexpr, w_s1: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Projection input-gradient: dx = dy @ w.

    dy: [B, T, N]
    w:  [N, K]
    dx: contiguous [M, K]
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    b = offs_m // T
    t = offs_m - b * T

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n

        dy = tl.load(
            dy_ptr + b[:, None] * dy_s0 + t[:, None] * dy_s1 + n_idxs[None, :] * dy_s2,
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        )

        w = tl.load(
            w_ptr + n_idxs[:, None] * w_s0 + offs_k[None, :] * w_s1,
            mask=(n_idxs[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dy, w)

    tl.store(
        dx_ptr + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _proj_dw_kernel(
    dy_ptr, x_ptr, dw_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, T: tl.constexpr,
    dy_s0: tl.constexpr, dy_s1: tl.constexpr, dy_s2: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    """
    Projection weight-gradient: dw = dy.T @ x.

    dy: [B, T, N]
    x: contiguous [M, K]
    dw: contiguous [N, K]
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m
        b = m_idxs // T
        t = m_idxs - b * T

        dy = tl.load(
            dy_ptr + b[:, None] * dy_s0 + t[:, None] * dy_s1 + offs_n[None, :] * dy_s2,
            mask=(m_idxs[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )

        x = tl.load(
            x_ptr + m_idxs[:, None] * K + offs_k[None, :],
            mask=(m_idxs[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(tl.trans(dy), x)

    tl.store(
        dw_ptr + offs_n[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
    )


@triton.jit
def _attention_backward_kernel(
    q_ptr, k_ptr, v_ptr, dattn_ptr,
    dq_ptr, dk_acc_ptr, dv_acc_ptr,
    B: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
    q_s0: tl.constexpr, q_s1: tl.constexpr, q_s2: tl.constexpr, q_s3: tl.constexpr,
    k_s0: tl.constexpr, k_s1: tl.constexpr, k_s2: tl.constexpr, k_s3: tl.constexpr,
    v_s0: tl.constexpr, v_s1: tl.constexpr, v_s2: tl.constexpr, v_s3: tl.constexpr,
    dq_s0: tl.constexpr, dq_s1: tl.constexpr, dq_s2: tl.constexpr, dq_s3: tl.constexpr,
    dk_s0: tl.constexpr, dk_s1: tl.constexpr, dk_s2: tl.constexpr, dk_s3: tl.constexpr,
    dv_s0: tl.constexpr, dv_s1: tl.constexpr, dv_s2: tl.constexpr, dv_s3: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
):
    """
    Backward for one causal GQA attention row.

    Recomputes softmax probabilities and computes:
      dQ directly for this query row.
      dK/dV via atomic adds into fp32 accumulation buffers.

    Important fix:
      dK/dV are accumulated in fp32 buffers, not bf16 output buffers.  Direct
      bf16 atomics caused small but test-visible accumulation-order errors.
    """
    pid = tl.program_id(0)
    t = pid % T
    h = (pid // T) % H
    b = pid // (T * H)

    GROUP: tl.constexpr = H // HKV
    hk = h // GROUP

    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)

    t_mask = offs_t < T
    d_mask = offs_d < D

    q_vals = tl.load(
        q_ptr + b * q_s0 + h * q_s1 + t * q_s2 + offs_d * q_s3,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    k_vals = tl.load(
        k_ptr
        + b * k_s0
        + hk * k_s1
        + offs_t[:, None] * k_s2
        + offs_d[None, :] * k_s3,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    v_vals = tl.load(
        v_ptr
        + b * v_s0
        + hk * v_s1
        + offs_t[:, None] * v_s2
        + offs_d[None, :] * v_s3,
        mask=t_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    do_vals = tl.load(
        dattn_ptr + ((b * T + t) * H + h) * D + offs_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k_vals * q_vals[None, :], axis=1) * SCALE
    causal_mask = (offs_t <= t) & t_mask
    scores = tl.where(causal_mask, scores, -3.4028234663852886e38)

    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    denom = tl.sum(p, axis=0)
    p = p / denom

    dp = tl.sum(v_vals * do_vals[None, :], axis=1)
    delta = tl.sum(p * dp, axis=0)
    ds = p * (dp - delta) * SCALE
    ds = tl.where(causal_mask, ds, 0.0)

    dq = tl.sum(ds[:, None] * k_vals, axis=0)
    tl.store(
        dq_ptr + b * dq_s0 + h * dq_s1 + t * dq_s2 + offs_d * dq_s3,
        dq,
        mask=d_mask,
    )

    dk = ds[:, None] * q_vals[None, :]
    dv = p[:, None] * do_vals[None, :]

    active_mask = t_mask[:, None] & d_mask[None, :] & (offs_t[:, None] <= t)

    tl.atomic_add(
        dk_acc_ptr
        + b * dk_s0
        + hk * dk_s1
        + offs_t[:, None] * dk_s2
        + offs_d[None, :] * dk_s3,
        dk,
        sem="relaxed",
        mask=active_mask,
    )

    tl.atomic_add(
        dv_acc_ptr
        + b * dv_s0
        + hk * dv_s1
        + offs_t[:, None] * dv_s2
        + offs_d[None, :] * dv_s3,
        dv,
        sem="relaxed",
        mask=active_mask,
    )


@triton.jit
def _copy_to_flat_kernel(src, dst, n_elements: tl.constexpr, dst_offset: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    vals = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + dst_offset + offs, vals, mask=mask)


def qwen3_attention_forward_with_saved(q, k, v, o_weight):
    """
    Forward for Qwen3-style causal grouped-query attention.

    Fused stages:
      1. Triton causal GQA attention computes softmax(q k^T / sqrt(D)) v.
      2. Triton projection computes output = attention_output @ o_weight.T.

    The projection remains a separate Triton GEMM because fully fusing it into
    the row-wise attention kernel would require recomputing each attention row
    for every output-channel tile, greatly increasing arithmetic work.
    """
    B, H, T, D = q.shape
    HKV = k.shape[1]
    N = o_weight.shape[0]
    Ktot = H * D
    M = B * T

    attn = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
    out = torch.empty((B, T, N), device=q.device, dtype=q.dtype)

    block_t = _next_power_of_2(T)
    block_d = _next_power_of_2(D)
    scale = 1.0 / (D ** 0.5)

    _attention_forward_kernel[(B * H * T,)](
        q, k, v, attn,
        B, H, HKV, T, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        SCALE=scale,
        num_warps=4,
    )

    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _proj_forward_kernel[grid](
        attn, o_weight, out,
        M, Ktot, N,
        o_weight.stride(0), o_weight.stride(1),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        num_warps=4,
        num_stages=4,
    )

    saved = (q, k, v, o_weight, attn)
    return out, saved


def qwen3_attention_backward_from_saved(upstream, saved):
    """
    Backward for the saved forward pass.

    Computes gradients for q, k, v, and o_weight entirely with Triton kernels.

    Numeric fix:
      dK/dV use temporary fp32 accumulation buffers followed by one final cast
      to the original gradient dtype.  This preserves fusion of the attention
      backward row computation while avoiding bf16 atomic accumulation error.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    q, k, v, o_weight, attn = saved

    B, H, T, D = q.shape
    HKV = k.shape[1]
    N = o_weight.shape[0]
    Ktot = H * D
    M = B * T

    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
    dw = torch.empty(o_weight.shape, device=o_weight.device, dtype=o_weight.dtype)
    dattn = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)

    # fp32 temporary accumulation buffers for dK/dV atomics
    dk_acc = torch.empty(k.shape, device=k.device, dtype=torch.float32)
    dv_acc = torch.empty(v.shape, device=v.device, dtype=torch.float32)

    # d(attn_output) = dy @ o_weight
    grid_dx = (triton.cdiv(M, 16), triton.cdiv(Ktot, 16))
    _proj_dx_kernel[grid_dx](
        dy, o_weight, dattn,
        M, Ktot, N, T,
        dy.stride(0), dy.stride(1), dy.stride(2),
        o_weight.stride(0), o_weight.stride(1),
        BLOCK_M=16,
        BLOCK_K=16,
        BLOCK_N=32,
        num_warps=4,
        num_stages=4,
    )

    # d(o_weight) = dy.T @ attn_output
    grid_dw = (triton.cdiv(N, 16), triton.cdiv(Ktot, 16))
    _proj_dw_kernel[grid_dw](
        dy, attn, dw,
        M, Ktot, N, T,
        dy.stride(0), dy.stride(1), dy.stride(2),
        BLOCK_N=16,
        BLOCK_K=16,
        BLOCK_M=32,
        num_warps=4,
        num_stages=4,
    )

    zero_block = 1024
    _zero_kernel[(triton.cdiv(dk_acc.numel(), zero_block),)](
        dk_acc, dk_acc.numel(), BLOCK=zero_block, num_warps=4
    )
    _zero_kernel[(triton.cdiv(dv_acc.numel(), zero_block),)](
        dv_acc, dv_acc.numel(), BLOCK=zero_block, num_warps=4
    )

    block_t = _next_power_of_2(T)
    block_d = _next_power_of_2(D)
    scale = 1.0 / (D ** 0.5)

    _attention_backward_kernel[(B * H * T,)](
        q, k, v, dattn,
        dq, dk_acc, dv_acc,
        B, H, HKV, T, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        dk_acc.stride(0), dk_acc.stride(1), dk_acc.stride(2), dk_acc.stride(3),
        dv_acc.stride(0), dv_acc.stride(1), dv_acc.stride(2), dv_acc.stride(3),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        SCALE=scale,
        num_warps=4,
    )

    cast_block = 1024
    _cast_from_fp32_kernel[(triton.cdiv(dk.numel(), cast_block),)](
        dk_acc, dk, dk.numel(), BLOCK=cast_block, num_warps=4
    )
    _cast_from_fp32_kernel[(triton.cdiv(dv.numel(), cast_block),)](
        dv_acc, dv, dv.numel(), BLOCK=cast_block, num_warps=4
    )

    return dq, dk, dv, dw


def _append_flat(src, dst, offset: int):
    block = 1024
    n = src.numel()
    _copy_to_flat_kernel[(triton.cdiv(n, block),)](
        src, dst, n, offset, BLOCK=block, num_warps=4
    )
    return offset + n


def kernel_function(q, k, v, o_weight, upstream):
    """
    Forward+backward step required by the evaluator.

    Returns:
      torch.cat([
        output.float().reshape(-1),
        dq.float().reshape(-1),
        dk.float().reshape(-1),
        dv.float().reshape(-1),
        d_o_weight.float().reshape(-1),
      ])

    The wrapper only allocates tensors and launches Triton kernels.  All math is
    performed in Triton kernels.
    """
    out, saved = qwen3_attention_forward_with_saved(q, k, v, o_weight)
    dq, dk, dv, dw = qwen3_attention_backward_from_saved(upstream, saved)

    total = out.numel() + dq.numel() + dk.numel() + dv.numel() + dw.numel()
    flat = torch.empty((total,), device=q.device, dtype=torch.float32)

    off = 0
    off = _append_flat(out, flat, off)
    off = _append_flat(dq, flat, off)
    off = _append_flat(dk, flat, off)
    off = _append_flat(dv, flat, off)
    _append_flat(dw, flat, off)

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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_attention', 'arguments': ['q', 'k', 'v', 'o_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dq', 'dk', 'dv', 'do_weight'], 'gradient_return_order': ['dq', 'dk', 'dv', 'do_weight'], 'scalars': [], 'deployment_entry': 'qwen3_attention_deployment', 'allowed_primitives': ['vendor_gemm', 'vendor_sdpa']}


class Qwen3AttentionFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_attention`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, o_weight):
        out, saved = qwen3_attention_forward_with_saved(q, k, v, o_weight)
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
        dq, dk, dv, do_weight = qwen3_attention_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, do_weight


def qwen3_attention_deployment(q, k, v, o_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen3AttentionFunction.apply(q, k, v, o_weight)


class Qwen3AttentionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, o_weight):
        super().__init__()
        self.o_weight = (o_weight if isinstance(o_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(o_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v):
        return qwen3_attention_deployment(q, k, v, self.o_weight)


DEPLOYMENT_ENTRY = "qwen3_attention_deployment"
