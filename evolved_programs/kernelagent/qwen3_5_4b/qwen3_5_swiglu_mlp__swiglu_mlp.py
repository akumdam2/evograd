"""adbench-tools artifact: task qwen3_5_4b.qwen3_5_swiglu_mlp.swiglu_mlp, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Complete Triton implementation for Qwen3.5 SwiGLU MLP forward + backward.
#
# Operator implemented:
#   gate = x @ gate_weight.T
#   up   = x @ up_weight.T
#   h    = silu(gate) * up
#   y    = h @ down_weight.T
#
# Backward implemented:
#   dh             = dy @ down_weight
#   dup            = dh * silu(gate)
#   dgate          = dh * up * silu'(gate)
#   ddown_weight   = dy.T @ h
#   dup_weight     = dup.T @ x
#   dgate_weight   = dgate.T @ x
#   dx             = dgate @ gate_weight + dup @ up_weight
#
# Fusion notes:
#   * gate/up projections and SwiGLU hidden product are fused in one Triton kernel.
#   * dgate/dup elementwise backward is fused in one Triton kernel.
#   * dx accumulation fuses both branches, avoiding two separate dx buffers.
#   * The final benchmark-required flattened tensor is packed by a Triton copy kernel,
#     not by torch.cat.

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_swiglu_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    hidden_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * H + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < H),
            other=0.0,
        )

        gw = tl.load(
            gate_w_ptr + offs_n[:, None] * H + k_idxs[None, :],
            mask=(offs_n[:, None] < I) & (k_idxs[None, :] < H),
            other=0.0,
        )

        uw = tl.load(
            up_w_ptr + offs_n[:, None] * H + k_idxs[None, :],
            mask=(offs_n[:, None] < I) & (k_idxs[None, :] < H),
            other=0.0,
        )

        acc_gate = tl.dot(x, tl.trans(gw), acc_gate)
        acc_up = tl.dot(x, tl.trans(uw), acc_up)

    gate_f32 = acc_gate
    up_f32 = acc_up

    sig = 1.0 / (1.0 + tl.exp(-gate_f32))
    silu = gate_f32 * sig
    hidden = silu * up_f32

    out_offsets = offs_m[:, None] * I + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < I)

    tl.store(gate_ptr + out_offsets, gate_f32, mask=mask)
    tl.store(up_ptr + out_offsets, up_f32, mask=mask)
    tl.store(hidden_ptr + out_offsets, hidden, mask=mask)


@triton.jit
def _matmul_abt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[N, K]^T
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * K + k_idxs[None, :],
            mask=(offs_n[:, None] < N) & (k_idxs[None, :] < K),
            other=0.0,
        )

        acc = tl.dot(a, tl.trans(b), acc)

    c_offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=mask)


@triton.jit
def _matmul_ab_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[K, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    c_offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offs, acc, mask=mask)


@triton.jit
def _swiglu_backward_elem_kernel(
    d_hidden_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    dh = tl.load(d_hidden_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    silu_prime = sig * (1.0 + gate * (1.0 - sig))

    dup = dh * silu
    dgate = dh * up * silu_prime

    tl.store(dup_ptr + offs, dup, mask=mask)
    tl.store(dgate_ptr + offs, dgate, mask=mask)


@triton.jit
def _grad_weight_kernel(
    a_ptr,
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # out[R, C] = a[M, R]^T @ x[M, C]
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        a = tl.load(
            a_ptr + m_idxs[:, None] * R + offs_r[None, :],
            mask=(m_idxs[:, None] < M) & (offs_r[None, :] < R),
            other=0.0,
        )
        x = tl.load(
            x_ptr + m_idxs[:, None] * C + offs_c[None, :],
            mask=(m_idxs[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        )

        acc = tl.dot(tl.trans(a), x, acc)

    out_offs = offs_r[:, None] * C + offs_c[None, :]
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    tl.store(out_ptr + out_offs, acc, mask=mask)


@triton.jit
def _grad_x_fused_kernel(
    dgate_ptr,
    dup_ptr,
    gate_w_ptr,
    up_w_ptr,
    dx_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # dx[M, H] = dgate[M, I] @ gate_w[I, H] + dup[M, I] @ up_w[I, H]
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        k_idxs = k0 + offs_k

        dg = tl.load(
            dgate_ptr + offs_m[:, None] * I + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < I),
            other=0.0,
        )
        du = tl.load(
            dup_ptr + offs_m[:, None] * I + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < I),
            other=0.0,
        )

        gw = tl.load(
            gate_w_ptr + k_idxs[:, None] * H + offs_h[None, :],
            mask=(k_idxs[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )
        uw = tl.load(
            up_w_ptr + k_idxs[:, None] * H + offs_h[None, :],
            mask=(k_idxs[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc = tl.dot(dg, gw, acc)
        acc = tl.dot(du, uw, acc)

    out_offs = offs_m[:, None] * H + offs_h[None, :]
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(dx_ptr + out_offs, acc, mask=mask)


@triton.jit
def _pack_kernel(
    y_ptr,
    dx_ptr,
    dgw_ptr,
    duw_ptr,
    ddw_ptr,
    out_ptr,
    size_y: tl.constexpr,
    size_dx: tl.constexpr,
    size_dgw: tl.constexpr,
    size_duw: tl.constexpr,
    size_ddw: tl.constexpr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    m0 = offs < size_y
    v0 = tl.load(y_ptr + offs, mask=mask & m0, other=0.0).to(tl.float32)
    val = tl.where(m0, v0, val)

    o1 = offs - size_y
    m1 = (offs >= size_y) & (offs < size_y + size_dx)
    v1 = tl.load(dx_ptr + o1, mask=mask & m1, other=0.0).to(tl.float32)
    val = tl.where(m1, v1, val)

    base2 = size_y + size_dx
    o2 = offs - base2
    m2 = (offs >= base2) & (offs < base2 + size_dgw)
    v2 = tl.load(dgw_ptr + o2, mask=mask & m2, other=0.0).to(tl.float32)
    val = tl.where(m2, v2, val)

    base3 = base2 + size_dgw
    o3 = offs - base3
    m3 = (offs >= base3) & (offs < base3 + size_duw)
    v3 = tl.load(duw_ptr + o3, mask=mask & m3, other=0.0).to(tl.float32)
    val = tl.where(m3, v3, val)

    base4 = base3 + size_duw
    o4 = offs - base4
    m4 = (offs >= base4) & (offs < base4 + size_ddw)
    v4 = tl.load(ddw_ptr + o4, mask=mask & m4, other=0.0).to(tl.float32)
    val = tl.where(m4, v4, val)

    tl.store(out_ptr + offs, val, mask=mask)


def qwen3_5_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    """
    Forward pass for Qwen-style SwiGLU MLP.

    All numerical work is performed by Triton kernels.  The wrapper only
    allocates tensors and launches kernels.
    """
    H = x.shape[-1]
    I = gate_weight.shape[0]
    M = x.numel() // H

    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    gate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    up = torch.empty((M, I), device=x.device, dtype=x.dtype)
    hidden = torch.empty((M, I), device=x.device, dtype=x.dtype)

    BM = 16
    BN = 32
    BK = 64

    grid_gate = (triton.cdiv(M, BM), triton.cdiv(I, BN))
    _fused_gate_up_swiglu_kernel[grid_gate](
        x,
        gate_weight,
        up_weight,
        gate,
        up,
        hidden,
        M,
        H,
        I,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    grid_out = (triton.cdiv(M, BM), triton.cdiv(H, BN))
    _matmul_abt_kernel[grid_out](
        hidden,
        down_weight,
        y,
        M,
        H,
        I,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    saved = (x, gate_weight, up_weight, down_weight, gate, up, hidden)
    return y, saved


def qwen3_5_swiglu_mlp_backward_from_saved(upstream, saved):
    """
    Backward pass for the saved SwiGLU MLP forward state.

    Returns gradients in input order:
      dx, d_gate_weight, d_up_weight, d_down_weight
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, gate_weight, up_weight, down_weight, gate, up, hidden = saved

    H = x.shape[-1]
    I = gate_weight.shape[0]
    M = x.numel() // H

    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    d_gate_weight = torch.empty(gate_weight.shape, device=gate_weight.device, dtype=gate_weight.dtype)
    d_up_weight = torch.empty(up_weight.shape, device=up_weight.device, dtype=up_weight.dtype)
    d_down_weight = torch.empty(down_weight.shape, device=down_weight.device, dtype=down_weight.dtype)

    d_hidden = torch.empty((M, I), device=x.device, dtype=x.dtype)
    dgate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    dup = torch.empty((M, I), device=x.device, dtype=x.dtype)

    BM = 16
    BN = 32
    BK = 64

    # d_hidden = dy @ down_weight
    grid_dh = (triton.cdiv(M, BM), triton.cdiv(I, BN))
    _matmul_ab_kernel[grid_dh](
        dy,
        down_weight,
        d_hidden,
        M,
        I,
        H,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    total_hidden = M * I
    _swiglu_backward_elem_kernel[(triton.cdiv(total_hidden, 256),)](
        d_hidden,
        gate,
        up,
        dgate,
        dup,
        total_hidden,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    # d_gate_weight = dgate.T @ x
    grid_gw = (triton.cdiv(I, BN), triton.cdiv(H, BN))
    _grad_weight_kernel[grid_gw](
        dgate,
        x,
        d_gate_weight,
        M,
        I,
        H,
        BLOCK_R=BN,
        BLOCK_C=BN,
        BLOCK_M=BK,
        num_warps=4,
        num_stages=3,
    )

    # d_up_weight = dup.T @ x
    _grad_weight_kernel[grid_gw](
        dup,
        x,
        d_up_weight,
        M,
        I,
        H,
        BLOCK_R=BN,
        BLOCK_C=BN,
        BLOCK_M=BK,
        num_warps=4,
        num_stages=3,
    )

    # d_down_weight = dy.T @ hidden
    grid_dw = (triton.cdiv(H, BN), triton.cdiv(I, BN))
    _grad_weight_kernel[grid_dw](
        dy,
        hidden,
        d_down_weight,
        M,
        H,
        I,
        BLOCK_R=BN,
        BLOCK_C=BN,
        BLOCK_M=BK,
        num_warps=4,
        num_stages=3,
    )

    # dx = dgate @ gate_weight + dup @ up_weight
    grid_dx = (triton.cdiv(M, BM), triton.cdiv(H, BN))
    _grad_x_fused_kernel[grid_dx](
        dgate,
        dup,
        gate_weight,
        up_weight,
        dx,
        M,
        H,
        I,
        BLOCK_M=BM,
        BLOCK_H=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    return dx, d_gate_weight, d_up_weight, d_down_weight


def kernel_function(x, gate_weight, up_weight, down_weight, upstream):
    """
    Benchmark entry point.

    Performs the full forward + backward step and returns a single flat fp32
    tensor matching:
        torch.cat([output.float().reshape(-1),
                   dx.float().reshape(-1),
                   d_gate_weight.float().reshape(-1),
                   d_up_weight.float().reshape(-1),
                   d_down_weight.float().reshape(-1)])

    No PyTorch math or torch.cat is used; packing is done by a Triton kernel.
    """
    y, saved = qwen3_5_swiglu_mlp_forward_with_saved(
        x, gate_weight, up_weight, down_weight
    )
    dx, dgw, duw, ddw = qwen3_5_swiglu_mlp_backward_from_saved(upstream, saved)

    size_y = y.numel()
    size_dx = dx.numel()
    size_dgw = dgw.numel()
    size_duw = duw.numel()
    size_ddw = ddw.numel()
    total = size_y + size_dx + size_dgw + size_duw + size_ddw

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)

    _pack_kernel[(triton.cdiv(total, 1024),)](
        y,
        dx,
        dgw,
        duw,
        ddw,
        packed,
        size_y,
        size_dx,
        size_dgw,
        size_duw,
        size_ddw,
        total,
        BLOCK_SIZE=1024,
        num_warps=4,
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


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_swiglu_mlp', 'arguments': ['x', 'gate_weight', 'up_weight', 'down_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'gradient_return_order': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'scalars': [], 'deployment_entry': 'qwen3_5_swiglu_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen35SwigluMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_swiglu_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        out, saved = qwen3_5_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
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
        dx, dgate_weight, dup_weight, ddown_weight = qwen3_5_swiglu_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_weight, dup_weight, ddown_weight


def qwen3_5_swiglu_mlp_deployment(x, gate_weight, up_weight, down_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = Qwen35SwigluMlpFunction.apply(x, gate_weight, up_weight, down_weight)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Qwen35SwigluMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, gate_weight, up_weight, down_weight):
        super().__init__()
        self.gate_weight = (gate_weight if isinstance(gate_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(gate_weight))
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return qwen3_5_swiglu_mlp_deployment(x, self.gate_weight, self.up_weight, self.down_weight)


DEPLOYMENT_ENTRY = "qwen3_5_swiglu_mlp_deployment"
