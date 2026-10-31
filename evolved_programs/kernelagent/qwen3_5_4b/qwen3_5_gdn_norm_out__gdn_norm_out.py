"""adbench-tools artifact: task qwen3_5_4b.qwen3_5_gdn_norm_out.gdn_norm_out, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for Qwen3 GDN norm_out forward + backward.
#
# Operation implemented:
#   h = RMSNorm(o, weight=w_norm, eps=eps) * silu(z)
#   y = linear(flatten_feature(h), w_out)
#
# Important shape detail fixed here:
#   In the evaluated Qwen3 cases, RMSNorm is applied over the last per-head
#   dimension D (e.g. 128), while the output projection consumes the flattened
#   value dimension HV * D (e.g. 4096).  The previous implementation incorrectly
#   assumed w_norm.shape[0] == w_out.shape[1], which triggered AssertionError.
#
# Public API:
#   qwen3_5_gdn_norm_out_forward_with_saved
#   qwen3_5_gdn_norm_out_backward_from_saved
#   kernel_function
#
# The Python wrappers only validate/allocate/launch.  All numerical work is in
# Triton kernels.

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_silu_gate_fwd_kernel(
    o_ptr,
    z_ptr,
    w_ptr,
    h_ptr,
    inv_ptr,
    M_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    eps,
    BLOCK_K: tl.constexpr,
):
    """Per RMSNorm row: h = o * rsqrt(mean(o^2)+eps) * w_norm * silu(z)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K_NORM

    o = tl.load(o_ptr + row * K_NORM + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row * K_NORM + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(o * o, axis=0) / K_NORM
    inv = tl.rsqrt(ss + eps)

    sig = 1.0 / (1.0 + tl.exp(-z))
    silu = z * sig
    h = o * inv * w * silu

    tl.store(h_ptr + row * K_NORM + offs, h, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _matmul_h_w_kernel(
    h_ptr,
    w_ptr,
    y_ptr,
    M_OUT: tl.constexpr,
    N_OUT: tl.constexpr,
    K_PROJ: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_w_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Logical projection:
      y[M_OUT, N_OUT] = h_flat[M_OUT, K_PROJ] @ W_logical[N_OUT, K_PROJ]^T

    W_logical[n, k] is addressed by:
      w_ptr + n * stride_w_n + k * stride_w_k

    This supports both physical w_out layouts:
      [N_OUT, K_PROJ] with strides (K_PROJ, 1)
      [K_PROJ, N_OUT] with logical strides (1, N_OUT)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K_PROJ, BLOCK_K):
        k_idxs = k0 + offs_k

        h = tl.load(
            h_ptr + offs_m[:, None] * K_PROJ + k_idxs[None, :],
            mask=(offs_m[:, None] < M_OUT) & (k_idxs[None, :] < K_PROJ),
            other=0.0,
        )

        w = tl.load(
            w_ptr + offs_n[None, :] * stride_w_n + k_idxs[:, None] * stride_w_k,
            mask=(offs_n[None, :] < N_OUT) & (k_idxs[:, None] < K_PROJ),
            other=0.0,
        )

        acc = tl.dot(h, w, acc)

    tl.store(
        y_ptr + offs_m[:, None] * N_OUT + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M_OUT) & (offs_n[None, :] < N_OUT),
    )


@triton.jit
def _matmul_dy_w_kernel(
    dy_ptr,
    w_ptr,
    gh_ptr,
    M_OUT: tl.constexpr,
    N_OUT: tl.constexpr,
    K_PROJ: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_w_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_OUT: tl.constexpr,
    BLOCK_N_RED: tl.constexpr,
):
    """grad_h_flat[M_OUT,K_PROJ] = dy[M_OUT,N_OUT] @ W_logical[N_OUT,K_PROJ]."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K_OUT + tl.arange(0, BLOCK_K_OUT)
    offs_n = tl.arange(0, BLOCK_N_RED)

    acc = tl.zeros((BLOCK_M, BLOCK_K_OUT), dtype=tl.float32)

    for n0 in tl.range(0, N_OUT, BLOCK_N_RED):
        n_idxs = n0 + offs_n

        dy = tl.load(
            dy_ptr + offs_m[:, None] * N_OUT + n_idxs[None, :],
            mask=(offs_m[:, None] < M_OUT) & (n_idxs[None, :] < N_OUT),
            other=0.0,
        )

        w = tl.load(
            w_ptr + n_idxs[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
            mask=(n_idxs[:, None] < N_OUT) & (offs_k[None, :] < K_PROJ),
            other=0.0,
        )

        acc = tl.dot(dy, w, acc)

    tl.store(
        gh_ptr + offs_m[:, None] * K_PROJ + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M_OUT) & (offs_k[None, :] < K_PROJ),
    )


@triton.jit
def _matmul_dy_t_h_kernel(
    dy_ptr,
    h_ptr,
    gwout_ptr,
    M_OUT: tl.constexpr,
    N_OUT: tl.constexpr,
    K_PROJ: tl.constexpr,
    stride_gw_n: tl.constexpr,
    stride_gw_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M_RED: tl.constexpr,
):
    """
    Logical grad W:
      grad_W_logical[N_OUT,K_PROJ] = dy[M_OUT,N_OUT]^T @ h_flat[M_OUT,K_PROJ]

    Stored using logical strides into grad_w_out so both [N,K] and [K,N]
    physical layouts are supported.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M_RED)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m0 in tl.range(0, M_OUT, BLOCK_M_RED):
        m_idxs = m0 + offs_m

        dy = tl.load(
            dy_ptr + m_idxs[None, :] * N_OUT + offs_n[:, None],
            mask=(m_idxs[None, :] < M_OUT) & (offs_n[:, None] < N_OUT),
            other=0.0,
        )

        h = tl.load(
            h_ptr + m_idxs[:, None] * K_PROJ + offs_k[None, :],
            mask=(m_idxs[:, None] < M_OUT) & (offs_k[None, :] < K_PROJ),
            other=0.0,
        )

        acc = tl.dot(dy, h, acc)

    tl.store(
        gwout_ptr + offs_n[:, None] * stride_gw_n + offs_k[None, :] * stride_gw_k,
        acc,
        mask=(offs_n[:, None] < N_OUT) & (offs_k[None, :] < K_PROJ),
    )


@triton.jit
def _rms_silu_gate_bwd_kernel(
    o_ptr,
    z_ptr,
    w_ptr,
    gh_ptr,
    inv_ptr,
    go_ptr,
    gz_ptr,
    M_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Backward for h = RMSNorm(o) * w_norm * silu(z), per norm row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K_NORM

    o = tl.load(o_ptr + row * K_NORM + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + row * K_NORM + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gh = tl.load(gh_ptr + row * K_NORM + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-z))
    silu = z * sig
    dsilu = sig + z * sig * (1.0 - sig)

    # Let a_i = dL/dh_i * w_i * silu_i.
    # RMSNorm backward:
    #   dL/do_i = inv*a_i - o_i*inv^3/K * sum_j(a_j*o_j)
    a = gh * w * silu
    dot = tl.sum(a * o, axis=0)

    go = inv * a - o * (inv * inv * inv) * dot / K_NORM
    gz = gh * (o * inv * w) * dsilu

    tl.store(go_ptr + row * K_NORM + offs, go, mask=mask)
    tl.store(gz_ptr + row * K_NORM + offs, gz, mask=mask)


@triton.jit
def _grad_w_norm_kernel(
    o_ptr,
    z_ptr,
    gh_ptr,
    inv_ptr,
    gw_ptr,
    M_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """grad_w_norm[k] = sum_rows grad_h[row,k] * o[row,k] * inv[row] * silu(z[row,k])."""
    pid_k = tl.program_id(0)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for m0 in tl.range(0, M_NORM, BLOCK_M):
        m_idxs = m0 + offs_m
        mask = (m_idxs[:, None] < M_NORM) & (offs_k[None, :] < K_NORM)

        o = tl.load(o_ptr + m_idxs[:, None] * K_NORM + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + m_idxs[:, None] * K_NORM + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
        gh = tl.load(gh_ptr + m_idxs[:, None] * K_NORM + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + m_idxs, mask=m_idxs < M_NORM, other=0.0).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-z))
        silu = z * sig

        part = gh * o * inv[:, None] * silu
        acc += tl.sum(part, axis=0)

    tl.store(gw_ptr + offs_k, acc, mask=offs_k < K_NORM)


@triton.jit
def _pack_kernel(
    y_ptr,
    go_ptr,
    gz_ptr,
    gw_ptr,
    gwout_ptr,
    out_ptr,
    size_y: tl.constexpr,
    size_go: tl.constexpr,
    size_gz: tl.constexpr,
    size_gw: tl.constexpr,
    size_gwout: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack forward output and all gradients into one flat float32 tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    v = tl.zeros((BLOCK,), dtype=tl.float32)

    m0 = offs < size_y
    v_y = tl.load(y_ptr + offs, mask=m0 & mask, other=0.0).to(tl.float32)
    v = tl.where(m0, v_y, v)

    base1 = size_y
    idx1 = offs - base1
    m1 = (offs >= base1) & (offs < base1 + size_go)
    v_go = tl.load(go_ptr + idx1, mask=m1 & mask, other=0.0).to(tl.float32)
    v = tl.where(m1, v_go, v)

    base2 = base1 + size_go
    idx2 = offs - base2
    m2 = (offs >= base2) & (offs < base2 + size_gz)
    v_gz = tl.load(gz_ptr + idx2, mask=m2 & mask, other=0.0).to(tl.float32)
    v = tl.where(m2, v_gz, v)

    base3 = base2 + size_gz
    idx3 = offs - base3
    m3 = (offs >= base3) & (offs < base3 + size_gw)
    v_gw = tl.load(gw_ptr + idx3, mask=m3 & mask, other=0.0).to(tl.float32)
    v = tl.where(m3, v_gw, v)

    base4 = base3 + size_gw
    idx4 = offs - base4
    m4 = (offs >= base4) & (offs < base4 + size_gwout)
    v_gwout = tl.load(gwout_ptr + idx4, mask=m4 & mask, other=0.0).to(tl.float32)
    v = tl.where(m4, v_gwout, v)

    tl.store(out_ptr + offs, v, mask=mask)


def _projection_layout(o, w_out):
    """
    Determine logical projection layout without doing tensor compute.

    Returns:
      K_PROJ, N_OUT, stride_w_n, stride_w_k, physical_layout
    where logical W has shape [N_OUT, K_PROJ].
    """
    total = o.numel()

    # Common nn.Linear weight layout: [out_features, in_features].
    if total % w_out.shape[1] == 0:
        return w_out.shape[1], w_out.shape[0], w_out.stride(0), w_out.stride(1), 0

    # Alternative already-transposed layout: [in_features, out_features].
    if total % w_out.shape[0] == 0:
        return w_out.shape[0], w_out.shape[1], w_out.stride(1), w_out.stride(0), 1

    # Keep this assertion explicit: projection input feature must tile o.numel().
    assert False, "w_out has no dimension compatible with flattened o"


def _make_output(o, N_OUT, K_PROJ):
    """
    Allocate output with the expected leading dimensions.

    If projection consumes the last two dimensions, e.g. [B,T,HV,D] -> [B,T,H],
    drop the last two dimensions.  If it consumes only the last dimension,
    e.g. [B,T,K] -> [B,T,H], drop only the last dimension.
    """
    if o.dim() >= 2 and o.shape[-1] * o.shape[-2] == K_PROJ:
        return torch.empty(tuple(o.shape[:-2]) + (N_OUT,), device=o.device, dtype=o.dtype)
    if o.dim() >= 1 and o.shape[-1] == K_PROJ:
        return torch.empty(tuple(o.shape[:-1]) + (N_OUT,), device=o.device, dtype=o.dtype)
    M_OUT = o.numel() // K_PROJ
    return torch.empty((M_OUT, N_OUT), device=o.device, dtype=o.dtype)


def _launch_forward(o, z, w_norm, w_out, eps):
    assert o.is_cuda and z.is_cuda and w_norm.is_cuda and w_out.is_cuda
    assert o.is_contiguous() and z.is_contiguous()
    assert w_norm.is_contiguous()
    assert o.shape == z.shape

    K_NORM = w_norm.shape[0]
    assert o.numel() % K_NORM == 0

    M_NORM = o.numel() // K_NORM

    K_PROJ, N_OUT, stride_w_n, stride_w_k, _layout = _projection_layout(o, w_out)
    assert o.numel() % K_PROJ == 0
    M_OUT = o.numel() // K_PROJ

    h = torch.empty_like(o)
    inv = torch.empty((M_NORM,), device=o.device, dtype=torch.float32)
    y = _make_output(o, N_OUT, K_PROJ)

    block_k_rms = triton.next_power_of_2(K_NORM)

    _rms_silu_gate_fwd_kernel[(M_NORM,)](
        o,
        z,
        w_norm,
        h,
        inv,
        M_NORM,
        K_NORM,
        eps,
        BLOCK_K=block_k_rms,
        num_warps=4 if block_k_rms <= 256 else 8,
    )

    # RMS/gate and projection matmul cannot be fully fused efficiently here:
    # projection requires reduction over the entire flattened K_PROJ dimension,
    # while RMSNorm reductions are per K_NORM row.  Materializing h avoids
    # recomputing RMSNorm for every output tile.
    BM = 16
    BN = 64
    BK = 64
    grid_mm = (triton.cdiv(M_OUT, BM), triton.cdiv(N_OUT, BN))
    _matmul_h_w_kernel[grid_mm](
        h,
        w_out,
        y,
        M_OUT,
        N_OUT,
        K_PROJ,
        stride_w_n,
        stride_w_k,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    saved = (o, z, w_norm, w_out, h, inv)
    return y, saved


def qwen3_5_gdn_norm_out_forward_with_saved(o, z, w_norm, w_out, eps):
    return _launch_forward(o, z, w_norm, w_out, eps)


def qwen3_5_gdn_norm_out_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    o, z, w_norm, w_out, h, inv = saved

    K_NORM = w_norm.shape[0]
    M_NORM = o.numel() // K_NORM

    K_PROJ, N_OUT, stride_w_n, stride_w_k, layout = _projection_layout(o, w_out)
    M_OUT = o.numel() // K_PROJ

    grad_h = torch.empty_like(o)
    grad_o = torch.empty_like(o)
    grad_z = torch.empty_like(z)
    grad_w_norm = torch.empty_like(w_norm)
    grad_w_out = torch.empty(w_out.shape, device=w_out.device, dtype=w_out.dtype)

    BM = 16
    BK_OUT = 64
    BN_RED = 64
    grid_gh = (triton.cdiv(M_OUT, BM), triton.cdiv(K_PROJ, BK_OUT))
    _matmul_dy_w_kernel[grid_gh](
        dy,
        w_out,
        grad_h,
        M_OUT,
        N_OUT,
        K_PROJ,
        stride_w_n,
        stride_w_k,
        BLOCK_M=BM,
        BLOCK_K_OUT=BK_OUT,
        BLOCK_N_RED=BN_RED,
        num_warps=4,
        num_stages=4,
    )

    if layout == 0:
        stride_gw_n = grad_w_out.stride(0)
        stride_gw_k = grad_w_out.stride(1)
    else:
        stride_gw_n = grad_w_out.stride(1)
        stride_gw_k = grad_w_out.stride(0)

    BN = 32
    BKG = 64
    BMR = 64
    grid_gwout = (triton.cdiv(N_OUT, BN), triton.cdiv(K_PROJ, BKG))
    _matmul_dy_t_h_kernel[grid_gwout](
        dy,
        h,
        grad_w_out,
        M_OUT,
        N_OUT,
        K_PROJ,
        stride_gw_n,
        stride_gw_k,
        BLOCK_N=BN,
        BLOCK_K=BKG,
        BLOCK_M_RED=BMR,
        num_warps=4,
        num_stages=4,
    )

    block_k_rms = triton.next_power_of_2(K_NORM)
    _rms_silu_gate_bwd_kernel[(M_NORM,)](
        o,
        z,
        w_norm,
        grad_h,
        inv,
        grad_o,
        grad_z,
        M_NORM,
        K_NORM,
        BLOCK_K=block_k_rms,
        num_warps=4 if block_k_rms <= 256 else 8,
    )

    BWK = 64
    BWM = 32
    _grad_w_norm_kernel[(triton.cdiv(K_NORM, BWK),)](
        o,
        z,
        grad_h,
        inv,
        grad_w_norm,
        M_NORM,
        K_NORM,
        BLOCK_M=BWM,
        BLOCK_K=BWK,
        num_warps=4,
        num_stages=4,
    )

    return grad_o, grad_z, grad_w_norm, grad_w_out


def kernel_function(o, z, w_norm, w_out, eps, upstream):
    """
    Forward + backward task entrypoint.

    Returns one flat float32 tensor:
      cat([
        y.flatten(),
        grad_o.flatten(),
        grad_z.flatten(),
        grad_w_norm.flatten(),
        grad_w_out.flatten(),
      ])
    """
    y, saved = qwen3_5_gdn_norm_out_forward_with_saved(o, z, w_norm, w_out, eps)
    grad_o, grad_z, grad_w_norm, grad_w_out = qwen3_5_gdn_norm_out_backward_from_saved(upstream, saved)

    size_y = y.numel()
    size_go = grad_o.numel()
    size_gz = grad_z.numel()
    size_gw = grad_w_norm.numel()
    size_gwout = grad_w_out.numel()
    total = size_y + size_go + size_gz + size_gw + size_gwout

    packed = torch.empty((total,), device=o.device, dtype=torch.float32)

    BLOCK = 1024
    _pack_kernel[(triton.cdiv(total, BLOCK),)](
        y,
        grad_o,
        grad_z,
        grad_w_norm,
        grad_w_out,
        packed,
        size_y,
        size_go,
        size_gz,
        size_gw,
        size_gwout,
        total,
        BLOCK=BLOCK,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_gdn_norm_out', 'arguments': ['o', 'z', 'w_norm', 'w_out', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['do', 'dz', 'dw_norm', 'dw_out'], 'gradient_return_order': ['do', 'dz', 'dw_norm', 'dw_out', 'None'], 'scalars': ['eps'], 'deployment_entry': 'qwen3_5_gdn_norm_out_deployment', 'allowed_primitives': ['vendor_gemm']}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class Qwen35GdnNormOutFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_gdn_norm_out`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, o, z, w_norm, w_out, eps):
        y, saved = qwen3_5_gdn_norm_out_forward_with_saved(o, z, w_norm, w_out, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
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
        do, dz, dw_norm, dw_out = qwen3_5_gdn_norm_out_backward_from_saved(
            dy, saved, **_backward_scalars(qwen3_5_gdn_norm_out_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return do, dz, dw_norm, dw_out, None


def qwen3_5_gdn_norm_out_deployment(o, z, w_norm, w_out, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen35GdnNormOutFunction.apply(o, z, w_norm, w_out, eps)


class Qwen35GdnNormOutModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, w_norm, w_out, eps=1e-06):
        super().__init__()
        self.w_norm = (w_norm if isinstance(w_norm, torch.nn.Parameter)
                  else torch.nn.Parameter(w_norm))
        self.w_out = (w_out if isinstance(w_out, torch.nn.Parameter)
                  else torch.nn.Parameter(w_out))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, o, z):
        return qwen3_5_gdn_norm_out_deployment(o, z, self.w_norm, self.w_out, self.eps)


DEPLOYMENT_ENTRY = "qwen3_5_gdn_norm_out_deployment"
