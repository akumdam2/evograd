"""adbench-tools artifact: task dit_s_2.dit_s_2_dit_mlp.blocks.mlp, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for DiT MLP forward + backward:
#   y = Linear2(GELU_tanh(Linear1(x)))
#
# Fix note:
#   The previous implementation used Triton's default fp32 tl.dot precision, which
#   is TF32 on NVIDIA GPUs. The evaluator's forward reference is fp32/IEEE, so the
#   forward output failed with ~1e-2 absolute error. All tl.dot calls below use
#   input_precision="ieee" to match the reference numerics.
#
# Fusion note:
#   Forward fc1+bias+GELU is fused and fc2+bias is fused. The backward path keeps
#   separate reduction/matmul kernels because dw/db reductions span the full M
#   dimension and feed multiple independent gradient products; fully fusing them
#   would require either global cross-program synchronization or excessive atomic
#   accumulation.

import torch
import triton
import triton.language as tl


# -----------------------------
# Forward kernels
# -----------------------------

@triton.jit
def _linear1_gelu_kernel(
    x_ptr, w1_ptr, b1_ptr,
    z_ptr, a_ptr,
    M: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in tl.range(0, C, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * C + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < C),
            other=0.0,
        )
        w = tl.load(
            w1_ptr + offs_h[None, :] * C + k_idxs[:, None],
            mask=(offs_h[None, :] < H) & (k_idxs[:, None] < C),
            other=0.0,
        )
        acc += tl.dot(x, w, input_precision="ieee")

    bias = tl.load(b1_ptr + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
    z = acc + bias[None, :]

    # GELU approximate="tanh":
    # 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    tanh_u = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
    a = 0.5 * z * (1.0 + tanh_u)

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(z_ptr + offs_m[:, None] * H + offs_h[None, :], z, mask=mask)
    tl.store(a_ptr + offs_m[:, None] * H + offs_h[None, :], a, mask=mask)


@triton.jit
def _linear2_kernel(
    a_ptr, w2_ptr, b2_ptr, out_ptr,
    M: tl.constexpr, H: tl.constexpr, O: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for k0 in tl.range(0, H, BLOCK_K):
        k_idxs = k0 + offs_k

        a = tl.load(
            a_ptr + offs_m[:, None] * H + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < H),
            other=0.0,
        )
        w = tl.load(
            w2_ptr + offs_o[None, :] * H + k_idxs[:, None],
            mask=(offs_o[None, :] < O) & (k_idxs[:, None] < H),
            other=0.0,
        )
        acc += tl.dot(a, w, input_precision="ieee")

    bias = tl.load(b2_ptr + offs_o, mask=offs_o < O, other=0.0).to(tl.float32)
    out = acc + bias[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * O + offs_o[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_o[None, :] < O),
    )


# -----------------------------
# Backward kernels
# -----------------------------

@triton.jit
def _dw2_kernel(
    grad_out_ptr, a_ptr, dw2_ptr,
    M: tl.constexpr, O: tl.constexpr, H: tl.constexpr,
    BLOCK_O: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_O, BLOCK_H), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        go = tl.load(
            grad_out_ptr + m_idxs[None, :] * O + offs_o[:, None],
            mask=(m_idxs[None, :] < M) & (offs_o[:, None] < O),
            other=0.0,
        )
        a = tl.load(
            a_ptr + m_idxs[:, None] * H + offs_h[None, :],
            mask=(m_idxs[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )
        acc += tl.dot(go, a, input_precision="ieee")

    tl.store(
        dw2_ptr + offs_o[:, None] * H + offs_h[None, :],
        acc,
        mask=(offs_o[:, None] < O) & (offs_h[None, :] < H),
    )


@triton.jit
def _db2_kernel(
    grad_out_ptr, db2_ptr,
    M: tl.constexpr, O: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr,
):
    pid_o = tl.program_id(0)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m
        vals = tl.load(
            grad_out_ptr + m_idxs[:, None] * O + offs_o[None, :],
            mask=(m_idxs[:, None] < M) & (offs_o[None, :] < O),
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)

    tl.store(db2_ptr + offs_o, acc, mask=offs_o < O)


@triton.jit
def _da_kernel(
    grad_out_ptr, w2_ptr, da_ptr,
    M: tl.constexpr, O: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in tl.range(0, O, BLOCK_K):
        k_idxs = k0 + offs_k

        go = tl.load(
            grad_out_ptr + offs_m[:, None] * O + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < O),
            other=0.0,
        )
        w = tl.load(
            w2_ptr + k_idxs[:, None] * H + offs_h[None, :],
            mask=(k_idxs[:, None] < O) & (offs_h[None, :] < H),
            other=0.0,
        )
        acc += tl.dot(go, w, input_precision="ieee")

    tl.store(
        da_ptr + offs_m[:, None] * H + offs_h[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
    )


@triton.jit
def _dz_kernel(
    da_ptr, z_ptr, dz_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    da = tl.load(da_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    tanh_u = 2.0 / (1.0 + tl.exp(-2.0 * u)) - 1.0
    sech2 = 1.0 - tanh_u * tanh_u
    du_dz = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z)

    gelu_grad = 0.5 * (1.0 + tanh_u) + 0.5 * z * sech2 * du_dz
    dz = da * gelu_grad

    tl.store(dz_ptr + offs, dz, mask=mask)


@triton.jit
def _dw1_kernel(
    dz_ptr, x_ptr, dw1_ptr,
    M: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_C), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        dz = tl.load(
            dz_ptr + m_idxs[None, :] * H + offs_h[:, None],
            mask=(m_idxs[None, :] < M) & (offs_h[:, None] < H),
            other=0.0,
        )
        x = tl.load(
            x_ptr + m_idxs[:, None] * C + offs_c[None, :],
            mask=(m_idxs[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        )
        acc += tl.dot(dz, x, input_precision="ieee")

    tl.store(
        dw1_ptr + offs_h[:, None] * C + offs_c[None, :],
        acc,
        mask=(offs_h[:, None] < H) & (offs_c[None, :] < C),
    )


@triton.jit
def _db1_kernel(
    dz_ptr, db1_ptr,
    M: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m
        vals = tl.load(
            dz_ptr + m_idxs[:, None] * H + offs_h[None, :],
            mask=(m_idxs[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)

    tl.store(db1_ptr + offs_h, acc, mask=offs_h < H)


@triton.jit
def _dx_kernel(
    dz_ptr, w1_ptr, dx_ptr,
    M: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for k0 in tl.range(0, H, BLOCK_K):
        k_idxs = k0 + offs_k

        dz = tl.load(
            dz_ptr + offs_m[:, None] * H + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < H),
            other=0.0,
        )
        w = tl.load(
            w1_ptr + k_idxs[:, None] * C + offs_c[None, :],
            mask=(k_idxs[:, None] < H) & (offs_c[None, :] < C),
            other=0.0,
        )
        acc += tl.dot(dz, w, input_precision="ieee")

    tl.store(
        dx_ptr + offs_m[:, None] * C + offs_c[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
    )


# -----------------------------
# Final flatten/concatenate kernel
# -----------------------------

@triton.jit
def _concat_kernel(
    out_ptr, dx_ptr, dw1_ptr, db1_ptr, dw2_ptr, db2_ptr, result_ptr,
    N_OUT: tl.constexpr, N_DX: tl.constexpr, N_DW1: tl.constexpr,
    N_DB1: tl.constexpr, N_DW2: tl.constexpr, N_DB2: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < TOTAL

    base_out = 0
    base_dx = N_OUT
    base_dw1 = base_dx + N_DX
    base_db1 = base_dw1 + N_DW1
    base_dw2 = base_db1 + N_DB1
    base_db2 = base_dw2 + N_DW2

    mask_out = valid & (offs >= base_out) & (offs < base_dx)
    mask_dx = valid & (offs >= base_dx) & (offs < base_dw1)
    mask_dw1 = valid & (offs >= base_dw1) & (offs < base_db1)
    mask_db1 = valid & (offs >= base_db1) & (offs < base_dw2)
    mask_dw2 = valid & (offs >= base_dw2) & (offs < base_db2)
    mask_db2 = valid & (offs >= base_db2)

    idx_out = tl.where(mask_out, offs - base_out, 0)
    idx_dx = tl.where(mask_dx, offs - base_dx, 0)
    idx_dw1 = tl.where(mask_dw1, offs - base_dw1, 0)
    idx_db1 = tl.where(mask_db1, offs - base_db1, 0)
    idx_dw2 = tl.where(mask_dw2, offs - base_dw2, 0)
    idx_db2 = tl.where(mask_db2, offs - base_db2, 0)

    v_out = tl.load(out_ptr + idx_out, mask=mask_out, other=0.0).to(tl.float32)
    v_dx = tl.load(dx_ptr + idx_dx, mask=mask_dx, other=0.0).to(tl.float32)
    v_dw1 = tl.load(dw1_ptr + idx_dw1, mask=mask_dw1, other=0.0).to(tl.float32)
    v_db1 = tl.load(db1_ptr + idx_db1, mask=mask_db1, other=0.0).to(tl.float32)
    v_dw2 = tl.load(dw2_ptr + idx_dw2, mask=mask_dw2, other=0.0).to(tl.float32)
    v_db2 = tl.load(db2_ptr + idx_db2, mask=mask_db2, other=0.0).to(tl.float32)

    val = v_out + v_dx + v_dw1 + v_db1 + v_dw2 + v_db2
    tl.store(result_ptr + offs, val, mask=valid)


# -----------------------------
# Python launch helpers
# -----------------------------

def _check_inputs(x, w1, b1, w2, b2):
    assert x.is_cuda and w1.is_cuda and b1.is_cuda and w2.is_cuda and b2.is_cuda
    assert x.is_contiguous()
    assert w1.is_contiguous()
    assert b1.is_contiguous()
    assert w2.is_contiguous()
    assert b2.is_contiguous()
    assert x.shape[-1] == w1.shape[1]
    assert w1.shape[0] == b1.shape[0]
    assert w2.shape[1] == w1.shape[0]
    assert w2.shape[0] == b2.shape[0]


def _forward_impl(x, w1, b1, w2, b2):
    _check_inputs(x, w1, b1, w2, b2)

    C = x.shape[-1]
    H = w1.shape[0]
    O = w2.shape[0]
    M = x.numel() // C

    out_shape = tuple(x.shape[:-1]) + (O,)

    z = torch.empty((M, H), device=x.device, dtype=torch.float32)
    a = torch.empty((M, H), device=x.device, dtype=torch.float32)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    bm = 16
    bh = 32
    bk = 64

    grid1 = (triton.cdiv(M, bm), triton.cdiv(H, bh))
    _linear1_gelu_kernel[grid1](
        x, w1, b1, z, a,
        M, C, H,
        BLOCK_M=bm, BLOCK_H=bh, BLOCK_K=bk,
        num_warps=4,
        num_stages=3,
    )

    bo = 32
    grid2 = (triton.cdiv(M, bm), triton.cdiv(O, bo))
    _linear2_kernel[grid2](
        a, w2, b2, out,
        M, H, O,
        BLOCK_M=bm, BLOCK_O=bo, BLOCK_K=bk,
        num_warps=4,
        num_stages=3,
    )

    return out, (x, w1, b1, w2, b2, z, a)


def _backward_impl(grad_out, saved):
    x, w1, b1, w2, b2, z, a = saved

    assert grad_out.is_cuda
    assert grad_out.is_contiguous()

    C = x.shape[-1]
    H = w1.shape[0]
    O = w2.shape[0]
    M = x.numel() // C

    dx = torch.empty_like(x)
    dw1 = torch.empty_like(w1)
    db1 = torch.empty_like(b1)
    dw2 = torch.empty_like(w2)
    db2 = torch.empty_like(b2)

    da = torch.empty((M, H), device=x.device, dtype=torch.float32)
    dz = torch.empty((M, H), device=x.device, dtype=torch.float32)

    grid_dw2 = (triton.cdiv(O, 16), triton.cdiv(H, 32))
    _dw2_kernel[grid_dw2](
        grad_out, a, dw2,
        M, O, H,
        BLOCK_O=16, BLOCK_H=32, BLOCK_M=64,
        num_warps=4,
        num_stages=3,
    )

    grid_db2 = (triton.cdiv(O, 32),)
    _db2_kernel[grid_db2](
        grad_out, db2,
        M, O,
        BLOCK_M=256, BLOCK_O=32,
        num_warps=4,
    )

    grid_da = (triton.cdiv(M, 16), triton.cdiv(H, 32))
    _da_kernel[grid_da](
        grad_out, w2, da,
        M, O, H,
        BLOCK_M=16, BLOCK_H=32, BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )

    n_hidden = M * H
    block = 256
    _dz_kernel[(triton.cdiv(n_hidden, block),)](
        da, z, dz,
        n_hidden,
        BLOCK_SIZE=block,
        num_warps=4,
    )

    grid_dw1 = (triton.cdiv(H, 16), triton.cdiv(C, 32))
    _dw1_kernel[grid_dw1](
        dz, x, dw1,
        M, H, C,
        BLOCK_H=16, BLOCK_C=32, BLOCK_M=64,
        num_warps=4,
        num_stages=3,
    )

    grid_db1 = (triton.cdiv(H, 32),)
    _db1_kernel[grid_db1](
        dz, db1,
        M, H,
        BLOCK_M=256, BLOCK_H=32,
        num_warps=4,
    )

    grid_dx = (triton.cdiv(M, 16), triton.cdiv(C, 32))
    _dx_kernel[grid_dx](
        dz, w1, dx,
        M, H, C,
        BLOCK_M=16, BLOCK_C=32, BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )

    return dx, dw1, db1, dw2, db2


# -----------------------------
# Public API expected by tests
# -----------------------------

def dit_s_2_dit_mlp_forward_with_saved(x, w1, b1, w2, b2):
    return _forward_impl(x, w1, b1, w2, b2)


def dit_s_2_dit_mlp_backward_from_saved(upstream, saved):
    grad_out = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    return _backward_impl(grad_out, saved)


def kernel_function(x, w1, b1, w2, b2, upstream):
    out, saved = _forward_impl(x, w1, b1, w2, b2)
    dx, dw1, db1, dw2, db2 = _backward_impl(upstream, saved)

    n_out = out.numel()
    n_dx = dx.numel()
    n_dw1 = dw1.numel()
    n_db1 = db1.numel()
    n_dw2 = dw2.numel()
    n_db2 = db2.numel()
    total = n_out + n_dx + n_dw1 + n_db1 + n_dw2 + n_db2

    result = torch.empty((total,), device=x.device, dtype=torch.float32)

    block = 1024
    _concat_kernel[(triton.cdiv(total, block),)](
        out, dx, dw1, db1, dw2, db2, result,
        n_out, n_dx, n_dw1, n_db1, n_dw2, n_db2,
        total,
        BLOCK_SIZE=block,
        num_warps=4,
    )

    return result
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'dit_s_2_dit_mlp', 'arguments': ['x', 'w1', 'b1', 'w2', 'b2'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dw1', 'db1', 'dw2', 'db2'], 'gradient_return_order': ['dx', 'dw1', 'db1', 'dw2', 'db2'], 'scalars': [], 'deployment_entry': 'dit_s_2_dit_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class DitS2DitMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `dit_s_2_dit_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        out, saved = dit_s_2_dit_mlp_forward_with_saved(x, w1, b1, w2, b2)
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
        dx, dw1, db1, dw2, db2 = dit_s_2_dit_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dw1, db1, dw2, db2


def dit_s_2_dit_mlp_deployment(x, w1, b1, w2, b2):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = DitS2DitMlpFunction.apply(x, w1, b1, w2, b2)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class DitS2DitMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = (w1 if isinstance(w1, torch.nn.Parameter)
                  else torch.nn.Parameter(w1))
        self.b1 = (b1 if isinstance(b1, torch.nn.Parameter)
                  else torch.nn.Parameter(b1))
        self.w2 = (w2 if isinstance(w2, torch.nn.Parameter)
                  else torch.nn.Parameter(w2))
        self.b2 = (b2 if isinstance(b2, torch.nn.Parameter)
                  else torch.nn.Parameter(b2))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return dit_s_2_dit_mlp_deployment(x, self.w1, self.b1, self.w2, self.b2)


DEPLOYMENT_ENTRY = "dit_s_2_dit_mlp_deployment"
