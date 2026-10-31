import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _dlrm_dense_copy_kernel(dense, out,
                            dense_s0, dense_s1,
                            out_s0, out_s1,
                            D: tl.constexpr,
                            BLOCK_D: tl.constexpr):
    pid_d = tl.program_id(0)
    b = tl.program_id(1)
    offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    vals = tl.load(dense + b * dense_s0 + offs * dense_s1, mask=mask, other=0.0)
    tl.store(out + b * out_s0 + offs * out_s1, vals, mask=mask)


_dlrm_dense_copy_kernel = triton.jit(_dlrm_dense_copy_kernel)


def _dlrm_pairwise_dot_kernel(dense, sparse, out,
                              dense_s0, dense_s1,
                              sparse_s0, sparse_s1, sparse_s2,
                              out_s0, out_s1,
                              D: tl.constexpr,
                              K: tl.constexpr,
                              BLOCK_K: tl.constexpr,
                              BLOCK_D: tl.constexpr):
    pid_k = tl.program_id(0)
    b = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)

    k_f = offs_k.to(tl.float32)
    row = tl.floor((tl.sqrt(k_f * 8.0 + 1.0) + 1.0) * 0.5).to(tl.int64)
    tri_base = row * (row - 1) // 2
    col = offs_k.to(tl.int64) - tri_base

    mask_k = offs_k < K
    mask_d = offs_d < D
    mask = mask_k[:, None] & mask_d[None, :]

    a = tl.load(
        sparse + b * sparse_s0 + (row[:, None] - 1) * sparse_s1 + offs_d[None, :] * sparse_s2,
        mask=mask,
        other=0.0,
    )

    dense_col = tl.load(
        dense + b * dense_s0 + offs_d[None, :] * dense_s1,
        mask=mask & (col[:, None] == 0),
        other=0.0,
    )
    sparse_col = tl.load(
        sparse + b * sparse_s0 + (col[:, None] - 1) * sparse_s1 + offs_d[None, :] * sparse_s2,
        mask=mask & (col[:, None] > 0),
        other=0.0,
    )
    c = dense_col + sparse_col

    prod = a * c
    acc = tl.sum(prod, axis=1)
    tl.store(out + b * out_s0 + (D + offs_k) * out_s1, acc, mask=mask_k)


_dlrm_pairwise_dot_kernel = triton.jit(_dlrm_pairwise_dot_kernel)


def _dlrm_forward_tiled_kernel(dense, sparse, out,
                               dense_s0, dense_s1,
                               sparse_s0, sparse_s1, sparse_s2,
                               out_s0, out_s1,
                               D: tl.constexpr,
                               N: tl.constexpr,
                               BLOCK_N: tl.constexpr,
                               BLOCK_D: tl.constexpr):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    b = tl.program_id(2)

    rn = rb * BLOCK_N + tl.arange(0, BLOCK_N)
    cn = cb * BLOCK_N + tl.arange(0, BLOCK_N)
    od = tl.arange(0, BLOCK_D)
    md = od < D

    d0 = tl.load(
        dense + b * dense_s0 + od * dense_s1,
        mask=md,
        other=0.0,
    )
    tl.store(
        out + b * out_s0 + od * out_s1,
        d0,
        mask=(rb == 0) & (cb == 0) & md,
    )

    rdense = tl.load(
        dense + b * dense_s0 + od[None, :] * dense_s1,
        mask=(rn[:, None] == 0) & md[None, :],
        other=0.0,
    )
    rsparse = tl.load(
        sparse + b * sparse_s0 + (rn[:, None] - 1) * sparse_s1 + od[None, :] * sparse_s2,
        mask=(rn[:, None] > 0) & (rn[:, None] < N) & md[None, :],
        other=0.0,
    )
    rf = rdense + rsparse

    cdense = tl.load(
        dense + b * dense_s0 + od[None, :] * dense_s1,
        mask=(cn[:, None] == 0) & md[None, :],
        other=0.0,
    )
    csparse = tl.load(
        sparse + b * sparse_s0 + (cn[:, None] - 1) * sparse_s1 + od[None, :] * sparse_s2,
        mask=(cn[:, None] > 0) & (cn[:, None] < N) & md[None, :],
        other=0.0,
    )
    cf = cdense + csparse

    dots = tl.dot(rf, tl.trans(cf), input_precision="ieee")

    rr = rn[:, None]
    cc = cn[None, :]
    k = rr * (rr - 1) // 2 + cc
    tl.store(
        out + b * out_s0 + (D + k) * out_s1,
        dots,
        mask=(rr < N) & (cc < rr),
    )


_dlrm_forward_tiled_kernel = triton.jit(_dlrm_forward_tiled_kernel)


def _dlrm_backward_kernel(dout, dense, sparse, ddense, dsparse,
                          dout_s0, dout_s1,
                          dense_s0, dense_s1,
                          sparse_s0, sparse_s1, sparse_s2,
                          ddense_s0, ddense_s1,
                          dsparse_s0, dsparse_s1, dsparse_s2,
                          D: tl.constexpr,
                          N: tl.constexpr,
                          BLOCK_D: tl.constexpr):
    pid_d = tl.program_id(0)
    i = tl.program_id(1)
    b = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    acc = tl.zeros((BLOCK_D,), tl.float32)

    for j in range(0, N):
        if j == 0:
            feat = tl.load(
                dense + b * dense_s0 + offs_d * dense_s1,
                mask=mask_d,
                other=0.0,
            )
        else:
            feat = tl.load(
                sparse + b * sparse_s0 + (j - 1) * sparse_s1 + offs_d * sparse_s2,
                mask=mask_d,
                other=0.0,
            )

        idx_lower = i * (i - 1) // 2 + j
        idx_upper = j * (j - 1) // 2 + i
        coeff_idx = tl.where(j < i, idx_lower, idx_upper)
        coeff_idx = tl.where(i == j, 0, coeff_idx)
        use_coeff = i != j
        coeff = tl.load(
            dout + b * dout_s0 + (D + coeff_idx) * dout_s1,
            mask=use_coeff,
            other=0.0,
        )
        acc += coeff * feat

    direct = tl.load(
        dout + b * dout_s0 + offs_d * dout_s1,
        mask=(i == 0) & mask_d,
        other=0.0,
    )
    total = acc + direct

    tl.store(
        ddense + b * ddense_s0 + offs_d * ddense_s1,
        total,
        mask=(i == 0) & mask_d,
    )

    sparse_i = tl.where(i > 0, i - 1, 0)
    tl.store(
        dsparse + b * dsparse_s0 + sparse_i * dsparse_s1 + offs_d * dsparse_s2,
        total,
        mask=(i > 0) & mask_d,
    )


_dlrm_backward_kernel = triton.jit(_dlrm_backward_kernel)


def _dlrm_backward2_kernel(dout, dense, sparse, ddense, dsparse,
                           dout_s0, dout_s1,
                           dense_s0, dense_s1,
                           sparse_s0, sparse_s1, sparse_s2,
                           ddense_s0, ddense_s1,
                           dsparse_s0, dsparse_s1, dsparse_s2,
                           D: tl.constexpr,
                           N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    p = tl.program_id(0)
    b = tl.program_id(1)

    i0 = p * 2
    i1 = i0 + 1
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc0 = tl.zeros((BLOCK_D,), tl.float32)
    acc1 = tl.zeros((BLOCK_D,), tl.float32)

    for j in range(0, N):
        if j == 0:
            feat = tl.load(
                dense + b * dense_s0 + offs_d * dense_s1,
                mask=mask_d,
                other=0.0,
            )
        else:
            feat = tl.load(
                sparse + b * sparse_s0 + (j - 1) * sparse_s1 + offs_d * sparse_s2,
                mask=mask_d,
                other=0.0,
            )

        idx0_l = i0 * (i0 - 1) // 2 + j
        idx0_u = j * (j - 1) // 2 + i0
        idx0 = tl.where(j < i0, idx0_l, idx0_u)
        c0 = tl.load(
            dout + b * dout_s0 + (D + idx0) * dout_s1,
            mask=(i0 < N) & (i0 != j),
            other=0.0,
        )
        acc0 += c0 * feat

        idx1_l = i1 * (i1 - 1) // 2 + j
        idx1_u = j * (j - 1) // 2 + i1
        idx1 = tl.where(j < i1, idx1_l, idx1_u)
        c1 = tl.load(
            dout + b * dout_s0 + (D + idx1) * dout_s1,
            mask=(i1 < N) & (i1 != j),
            other=0.0,
        )
        acc1 += c1 * feat

    direct = tl.load(
        dout + b * dout_s0 + offs_d * dout_s1,
        mask=mask_d,
        other=0.0,
    )

    total0 = acc0 + tl.where(i0 == 0, direct, 0.0)
    total1 = acc1 + tl.where(i1 == 0, direct, 0.0)

    tl.store(
        ddense + b * ddense_s0 + offs_d * ddense_s1,
        total0,
        mask=(i0 == 0) & mask_d,
    )
    s0 = tl.where(i0 > 0, i0 - 1, 0)
    tl.store(
        dsparse + b * dsparse_s0 + s0 * dsparse_s1 + offs_d * dsparse_s2,
        total0,
        mask=(i0 > 0) & (i0 < N) & mask_d,
    )

    s1 = tl.where(i1 > 0, i1 - 1, 0)
    tl.store(
        dsparse + b * dsparse_s0 + s1 * dsparse_s1 + offs_d * dsparse_s2,
        total1,
        mask=(i1 > 0) & (i1 < N) & mask_d,
    )


_dlrm_backward2_kernel = triton.jit(_dlrm_backward2_kernel)


def torchrec_dlrm_dlrm_interaction_forward_with_saved(dense, sparse):
    if dense.dim() != 2:
        raise ValueError("dense must be a 2-D tensor")
    if sparse.dim() != 3:
        raise ValueError("sparse must be a 3-D tensor")
    if not dense.is_cuda or not sparse.is_cuda:
        raise ValueError("dense and sparse must be CUDA tensors")
    if dense.device != sparse.device:
        raise ValueError("dense and sparse must be on the same device")
    if dense.dtype != torch.float32 or sparse.dtype != torch.float32:
        raise ValueError("only torch.float32 inputs are supported")

    B = dense.shape[0]
    D = dense.shape[1]
    if sparse.shape[0] != B or sparse.shape[2] != D:
        raise ValueError("sparse shape must be [dense.shape[0], S, dense.shape[1]]")
    if B <= 0:
        raise ValueError("empty batch is not supported")
    if D <= 0:
        raise ValueError("empty embedding dimension is not supported")

    S = sparse.shape[1]
    N = S + 1
    K = N * (N - 1) // 2

    block_d_dot = triton.next_power_of_2(D)
    if block_d_dot > 1024:
        raise ValueError("embedding dimension larger than 1024 is not supported")

    out = torch.empty((B, D + K), device=dense.device, dtype=dense.dtype)

    if D >= 32 and D <= 128 and N <= 64:
        block_n = 16
        grid_fused = (triton.cdiv(N, block_n), triton.cdiv(N, block_n), B)
        _dlrm_forward_tiled_kernel[grid_fused](
            dense, sparse, out,
            dense.stride(0), dense.stride(1),
            sparse.stride(0), sparse.stride(1), sparse.stride(2),
            out.stride(0), out.stride(1),
            D, N,
            BLOCK_N=block_n,
            BLOCK_D=block_d_dot,
            num_warps=4,
        )
    else:
        block_copy = 256
        grid_copy = (triton.cdiv(D, block_copy), B)
        _dlrm_dense_copy_kernel[grid_copy](
            dense, out,
            dense.stride(0), dense.stride(1),
            out.stride(0), out.stride(1),
            D,
            BLOCK_D=block_copy,
        )

        if K > 0:
            block_k = 16
            grid_dot = (triton.cdiv(K, block_k), B)
            _dlrm_pairwise_dot_kernel[grid_dot](
                dense, sparse, out,
                dense.stride(0), dense.stride(1),
                sparse.stride(0), sparse.stride(1), sparse.stride(2),
                out.stride(0), out.stride(1),
                D, K,
                BLOCK_K=block_k,
                BLOCK_D=block_d_dot,
            )

    saved_tensors = (dense, sparse)
    return out, saved_tensors


def torchrec_dlrm_dlrm_interaction_backward_from_saved(dout, saved_tensors):
    if not isinstance(saved_tensors, (tuple, list)) or len(saved_tensors) != 2:
        raise ValueError("saved_tensors must be a pair: (dense, sparse)")

    dense = saved_tensors[0]
    sparse = saved_tensors[1]

    if dense.dim() != 2:
        raise ValueError("saved dense must be a 2-D tensor")
    if sparse.dim() != 3:
        raise ValueError("saved sparse must be a 3-D tensor")
    if dout.dim() != 2:
        raise ValueError("dout must be a 2-D tensor")
    if not dense.is_cuda or not sparse.is_cuda or not dout.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")
    if dense.device != sparse.device or dense.device != dout.device:
        raise ValueError("all tensors must be on the same device")
    if dense.dtype != torch.float32 or sparse.dtype != torch.float32 or dout.dtype != torch.float32:
        raise ValueError("only torch.float32 tensors are supported")

    B = dense.shape[0]
    D = dense.shape[1]
    if sparse.shape[0] != B or sparse.shape[2] != D:
        raise ValueError("saved sparse shape must be [dense.shape[0], S, dense.shape[1]]")
    if B <= 0:
        raise ValueError("empty batch is not supported")
    if D <= 0:
        raise ValueError("empty embedding dimension is not supported")

    S = sparse.shape[1]
    N = S + 1
    K = N * (N - 1) // 2
    if dout.shape[0] != B or dout.shape[1] != D + K:
        raise ValueError("dout has incompatible shape")

    block_d = triton.next_power_of_2(D)
    if block_d > 1024:
        raise ValueError("embedding dimension larger than 1024 is not supported")

    ddense = torch.empty_like(dense)
    dsparse = torch.empty_like(sparse)

    grid = (triton.cdiv(N, 2), B)
    _dlrm_backward2_kernel[grid](
        dout, dense, sparse, ddense, dsparse,
        dout.stride(0), dout.stride(1),
        dense.stride(0), dense.stride(1),
        sparse.stride(0), sparse.stride(1), sparse.stride(2),
        ddense.stride(0), ddense.stride(1),
        dsparse.stride(0), dsparse.stride(1), dsparse.stride(2),
        D, N,
        BLOCK_D=block_d,
        num_warps=2,
    )

    return ddense, dsparse

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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'torchrec_dlrm_dlrm_interaction', 'arguments': ['dense', 'sparse'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['ddense', 'dsparse'], 'gradient_return_order': ['ddense', 'dsparse'], 'scalars': [], 'deployment_entry': 'torchrec_dlrm_dlrm_interaction_deployment', 'allowed_primitives': ['vendor_gemm']}


class TorchrecDlrmDlrmInteractionFunction(torch.autograd.Function):
    """Static autograd wiring for `torchrec_dlrm_dlrm_interaction`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, dense, sparse):
        out, saved = torchrec_dlrm_dlrm_interaction_forward_with_saved(dense, sparse)
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
        ddense, dsparse = torchrec_dlrm_dlrm_interaction_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return ddense, dsparse


def torchrec_dlrm_dlrm_interaction_deployment(dense, sparse):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return TorchrecDlrmDlrmInteractionFunction.apply(dense, sparse)


class TorchrecDlrmDlrmInteractionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, dense, sparse):
        return torchrec_dlrm_dlrm_interaction_deployment(dense, sparse)


DEPLOYMENT_ENTRY = "torchrec_dlrm_dlrm_interaction_deployment"
