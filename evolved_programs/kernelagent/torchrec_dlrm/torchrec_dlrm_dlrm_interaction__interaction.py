"""adbench-tools artifact: task torchrec_dlrm.torchrec_dlrm_dlrm_interaction.interaction, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation of TorchRec DLRM interaction forward + backward.
#
# Fused stages for kernel_function:
#   1. Forward DLRM interaction:
#        output = concat(dense, lower_triangle(dot([dense, sparse], [dense, sparse])))
#   2. Backward w.r.t. dense and sparse inputs from the upstream gradient.
#   3. Flatten/pack [output, grad_dense, grad_sparse] directly into one 1-D tensor.
#
# The Python wrappers only allocate tensors, validate shapes, compute launch grids,
# and launch Triton kernels. All numerical work is done inside Triton.

import torch
import triton
import triton.language as tl


@triton.jit
def _interaction_forward_kernel(
    dense_ptr,
    sparse_ptr,
    out_ptr,
    D: tl.constexpr,
    F: tl.constexpr,
    O: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sf: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward DLRM interaction.

    dense:  [B, D]
    sparse: [B, F, D]
    out:    [B, D + F*(F+1)//2]

    Pair ordering matches torch.tril_indices(F+1, F+1, offset=-1):
      (1,0), (2,0), (2,1), ...
    Feature 0 is dense; features 1..F are sparse[:, feature-1, :].
    """
    b = tl.program_id(0)
    pid_c = tl.program_id(1)

    cols = pid_c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    valid_col = cols < O

    is_dense_col = valid_col & (cols < D)
    is_interaction_col = valid_col & (cols >= D)

    # Dense prefix output.
    dense_prefix = tl.load(
        dense_ptr + b * stride_db + cols * stride_dd,
        mask=is_dense_col,
        other=0.0,
    ).to(tl.float32)

    # Map flattened interaction index p to lower-triangular pair (i, j).
    p = tl.where(is_interaction_col, cols - D, 0)
    p_f = p.to(tl.float32)
    i = tl.floor((tl.sqrt(1.0 + 8.0 * p_f) + 1.0) * 0.5).to(tl.int64)
    tri = (i * (i - 1)) // 2
    j = p - tri

    offs_d = tl.arange(0, BLOCK_D)
    dmask = offs_d < D

    i_mat = i[:, None]
    j_mat = j[:, None]
    d_mat = offs_d[None, :]

    # Load feature i.
    dense_i = tl.load(
        dense_ptr + b * stride_db + d_mat * stride_dd + i_mat * 0,
        mask=is_interaction_col[:, None] & (i_mat == 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    sparse_i = tl.load(
        sparse_ptr + b * stride_sb + (i_mat - 1) * stride_sf + d_mat * stride_sd,
        mask=is_interaction_col[:, None] & (i_mat > 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    vi = dense_i + sparse_i

    # Load feature j.
    dense_j = tl.load(
        dense_ptr + b * stride_db + d_mat * stride_dd + j_mat * 0,
        mask=is_interaction_col[:, None] & (j_mat == 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    sparse_j = tl.load(
        sparse_ptr + b * stride_sb + (j_mat - 1) * stride_sf + d_mat * stride_sd,
        mask=is_interaction_col[:, None] & (j_mat > 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    vj = dense_j + sparse_j

    dot = tl.sum(vi * vj, axis=1)
    result = tl.where(is_dense_col, dense_prefix, dot)

    tl.store(out_ptr + b * stride_ob + cols * stride_od, result, mask=valid_col)


@triton.jit
def _interaction_backward_dense_kernel(
    dense_ptr,
    sparse_ptr,
    upstream_ptr,
    grad_dense_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    O: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sf: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_ub: tl.constexpr,
    stride_ud: tl.constexpr,
    stride_gdb: tl.constexpr,
    stride_gdd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward contribution for dense input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    b = offs // D
    d = offs - b * D

    # Direct gradient from dense prefix of output.
    acc = tl.load(
        upstream_ptr + b * stride_ub + d * stride_ud,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # Interactions containing feature 0: pairs (k, 0), k=1..F.
    for k in tl.static_range(1, F + 1):
        p = (k * (k - 1)) // 2
        u = tl.load(
            upstream_ptr + b * stride_ub + (D + p) * stride_ud,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        s = tl.load(
            sparse_ptr + b * stride_sb + (k - 1) * stride_sf + d * stride_sd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += u * s

    tl.store(grad_dense_ptr + b * stride_gdb + d * stride_gdd, acc, mask=mask)


@triton.jit
def _interaction_backward_sparse_kernel(
    dense_ptr,
    sparse_ptr,
    upstream_ptr,
    grad_sparse_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    O: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sf: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_ub: tl.constexpr,
    stride_ud: tl.constexpr,
    stride_gsb: tl.constexpr,
    stride_gsf: tl.constexpr,
    stride_gsd: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Backward contribution for sparse input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    d = offs % D
    tmp = offs // D
    f = tmp % F
    b = tmp // F

    # Combined feature index for sparse feature f.
    k_feat = f + 1

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # For every other feature m, contribution is upstream(pair(k, m)) * feature_m.
    for m in tl.static_range(0, F + 1):
        use = mask & (k_feat != m)

        hi = tl.maximum(k_feat, m)
        lo = tl.minimum(k_feat, m)
        p = (hi * (hi - 1)) // 2 + lo

        u = tl.load(
            upstream_ptr + b * stride_ub + (D + p) * stride_ud,
            mask=use,
            other=0.0,
        ).to(tl.float32)

        if m == 0:
            feat = tl.load(
                dense_ptr + b * stride_db + d * stride_dd,
                mask=use,
                other=0.0,
            ).to(tl.float32)
        else:
            feat = tl.load(
                sparse_ptr + b * stride_sb + (m - 1) * stride_sf + d * stride_sd,
                mask=use,
                other=0.0,
            ).to(tl.float32)

        acc += u * feat

    tl.store(
        grad_sparse_ptr + b * stride_gsb + f * stride_gsf + d * stride_gsd,
        acc,
        mask=mask,
    )


@triton.jit
def _interaction_pair_flat_kernel(
    dense_ptr,
    sparse_ptr,
    upstream_ptr,
    flat_ptr,
    TOTAL: tl.constexpr,
    OUT_NUMEL: tl.constexpr,
    GD_NUMEL: tl.constexpr,
    GS_NUMEL: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    O: tl.constexpr,
    stride_db: tl.constexpr,
    stride_dd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sf: tl.constexpr,
    stride_sd: tl.constexpr,
    stride_ub: tl.constexpr,
    stride_ud: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused forward + backward + flatten/pack kernel for kernel_function."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TOTAL

    # -------------------------------------------------------------------------
    # Section 1: flattened forward output [B, O]
    # -------------------------------------------------------------------------
    out_mask = valid & (offs < OUT_NUMEL)
    out_local = offs
    out_b = out_local // O
    out_c = out_local - out_b * O

    is_dense_col = out_mask & (out_c < D)
    is_interaction_col = out_mask & (out_c >= D)

    dense_prefix = tl.load(
        dense_ptr + out_b * stride_db + out_c * stride_dd,
        mask=is_dense_col,
        other=0.0,
    ).to(tl.float32)

    p = tl.where(is_interaction_col, out_c - D, 0)
    p_f = p.to(tl.float32)
    i = tl.floor((tl.sqrt(1.0 + 8.0 * p_f) + 1.0) * 0.5).to(tl.int64)
    tri = (i * (i - 1)) // 2
    j = p - tri

    d_red = tl.arange(0, BLOCK_D)
    dmask = d_red < D
    d_mat = d_red[None, :]
    i_mat = i[:, None]
    j_mat = j[:, None]
    b_mat = out_b[:, None]

    dense_i = tl.load(
        dense_ptr + b_mat * stride_db + d_mat * stride_dd + i_mat * 0,
        mask=is_interaction_col[:, None] & (i_mat == 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    sparse_i = tl.load(
        sparse_ptr + b_mat * stride_sb + (i_mat - 1) * stride_sf + d_mat * stride_sd,
        mask=is_interaction_col[:, None] & (i_mat > 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)

    dense_j = tl.load(
        dense_ptr + b_mat * stride_db + d_mat * stride_dd + j_mat * 0,
        mask=is_interaction_col[:, None] & (j_mat == 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)
    sparse_j = tl.load(
        sparse_ptr + b_mat * stride_sb + (j_mat - 1) * stride_sf + d_mat * stride_sd,
        mask=is_interaction_col[:, None] & (j_mat > 0) & dmask[None, :],
        other=0.0,
    ).to(tl.float32)

    dot = tl.sum((dense_i + sparse_i) * (dense_j + sparse_j), axis=1)
    fwd_val = tl.where(is_dense_col, dense_prefix, dot)

    # -------------------------------------------------------------------------
    # Section 2: flattened grad_dense [B, D]
    # -------------------------------------------------------------------------
    gd_mask = valid & (offs >= OUT_NUMEL) & (offs < OUT_NUMEL + GD_NUMEL)
    gd_local = offs - OUT_NUMEL
    gd_b = gd_local // D
    gd_d = gd_local - gd_b * D

    gd_acc = tl.load(
        upstream_ptr + gd_b * stride_ub + gd_d * stride_ud,
        mask=gd_mask,
        other=0.0,
    ).to(tl.float32)

    for k in tl.static_range(1, F + 1):
        p_k = (k * (k - 1)) // 2
        u = tl.load(
            upstream_ptr + gd_b * stride_ub + (D + p_k) * stride_ud,
            mask=gd_mask,
            other=0.0,
        ).to(tl.float32)
        s = tl.load(
            sparse_ptr + gd_b * stride_sb + (k - 1) * stride_sf + gd_d * stride_sd,
            mask=gd_mask,
            other=0.0,
        ).to(tl.float32)
        gd_acc += u * s

    # -------------------------------------------------------------------------
    # Section 3: flattened grad_sparse [B, F, D]
    # -------------------------------------------------------------------------
    gs_mask = valid & (offs >= OUT_NUMEL + GD_NUMEL)
    gs_local = offs - OUT_NUMEL - GD_NUMEL

    gs_d = gs_local % D
    gs_tmp = gs_local // D
    gs_f = gs_tmp % F
    gs_b = gs_tmp // F
    gs_k = gs_f + 1

    gs_acc = tl.zeros((BLOCK,), dtype=tl.float32)

    for m in tl.static_range(0, F + 1):
        use = gs_mask & (gs_k != m)

        hi = tl.maximum(gs_k, m)
        lo = tl.minimum(gs_k, m)
        p_m = (hi * (hi - 1)) // 2 + lo

        u = tl.load(
            upstream_ptr + gs_b * stride_ub + (D + p_m) * stride_ud,
            mask=use,
            other=0.0,
        ).to(tl.float32)

        if m == 0:
            feat = tl.load(
                dense_ptr + gs_b * stride_db + gs_d * stride_dd,
                mask=use,
                other=0.0,
            ).to(tl.float32)
        else:
            feat = tl.load(
                sparse_ptr + gs_b * stride_sb + (m - 1) * stride_sf + gs_d * stride_sd,
                mask=use,
                other=0.0,
            ).to(tl.float32)

        gs_acc += u * feat

    final_val = tl.where(out_mask, fwd_val, tl.where(gd_mask, gd_acc, gs_acc))
    tl.store(flat_ptr + offs, final_val, mask=valid)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def torchrec_dlrm_dlrm_interaction_forward_with_saved(dense, sparse):
    """Forward wrapper for DLRM interaction.

    Returns:
        output: [B, D + F*(F+1)//2]
        saved:  tuple containing tensors needed by backward
    """
    assert dense.is_cuda and sparse.is_cuda
    assert dense.dim() == 2
    assert sparse.dim() == 3
    assert dense.shape[0] == sparse.shape[0]
    assert dense.shape[1] == sparse.shape[2]

    B = dense.shape[0]
    D = dense.shape[1]
    F = sparse.shape[1]
    O = D + (F * (F + 1)) // 2

    block_d = _next_power_of_2(D)
    assert block_d <= 1024

    out = torch.empty((B, O), device=dense.device, dtype=dense.dtype)

    block_cols = 16
    grid = (B, triton.cdiv(O, block_cols))
    _interaction_forward_kernel[grid](
        dense,
        sparse,
        out,
        D,
        F,
        O,
        dense.stride(0),
        dense.stride(1),
        sparse.stride(0),
        sparse.stride(1),
        sparse.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_COLS=block_cols,
        BLOCK_D=block_d,
    )

    return out, (dense, sparse)


def torchrec_dlrm_dlrm_interaction_backward_from_saved(upstream, saved):
    """Backward wrapper for DLRM interaction.

    Args:
        upstream: gradient of forward output, shape [B, O]
        saved: saved tuple from forward

    Returns:
        grad_dense, grad_sparse
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    dense, sparse = saved

    assert dense.is_cuda and sparse.is_cuda and upstream.is_cuda
    assert dense.dim() == 2
    assert sparse.dim() == 3

    B = dense.shape[0]
    D = dense.shape[1]
    F = sparse.shape[1]
    O = D + (F * (F + 1)) // 2

    grad_dense = torch.empty_like(dense)
    grad_sparse = torch.empty_like(sparse)

    block = 256

    n_dense = B * D
    grid_dense = (triton.cdiv(n_dense, block),)
    _interaction_backward_dense_kernel[grid_dense](
        dense,
        sparse,
        upstream,
        grad_dense,
        n_dense,
        D,
        F,
        O,
        dense.stride(0),
        dense.stride(1),
        sparse.stride(0),
        sparse.stride(1),
        sparse.stride(2),
        upstream.stride(0),
        upstream.stride(1),
        grad_dense.stride(0),
        grad_dense.stride(1),
        BLOCK=block,
    )

    n_sparse = B * F * D
    grid_sparse = (triton.cdiv(n_sparse, block),)
    _interaction_backward_sparse_kernel[grid_sparse](
        dense,
        sparse,
        upstream,
        grad_sparse,
        n_sparse,
        D,
        F,
        O,
        dense.stride(0),
        dense.stride(1),
        sparse.stride(0),
        sparse.stride(1),
        sparse.stride(2),
        upstream.stride(0),
        upstream.stride(1),
        grad_sparse.stride(0),
        grad_sparse.stride(1),
        grad_sparse.stride(2),
        BLOCK=block,
    )

    return grad_dense, grad_sparse


def kernel_function(dense, sparse, upstream):
    """Fused forward+backward step required by the evaluator.

    Computes, entirely in Triton, the equivalent of:

        output, saved = forward_with_saved(dense, sparse)
        grad_dense, grad_sparse = backward_from_saved(upstream, saved)
        return torch.cat([
            output.float().reshape(-1),
            grad_dense.float().reshape(-1),
            grad_sparse.float().reshape(-1),
        ])

    The result is written directly into one flat float32 tensor by a single
    fused Triton kernel. The wrapper performs only allocation and launch setup.
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    assert dense.is_cuda and sparse.is_cuda and upstream.is_cuda
    assert dense.dim() == 2
    assert sparse.dim() == 3
    assert dense.shape[0] == sparse.shape[0]
    assert dense.shape[1] == sparse.shape[2]

    B = dense.shape[0]
    D = dense.shape[1]
    F = sparse.shape[1]
    O = D + (F * (F + 1)) // 2

    out_numel = B * O
    gd_numel = B * D
    gs_numel = B * F * D
    total = out_numel + gd_numel + gs_numel

    block_d = _next_power_of_2(D)
    assert block_d <= 1024

    # The evaluator compares against tensors converted to float32, so return
    # a flat float32 tensor directly.
    flat = torch.empty((total,), device=dense.device, dtype=torch.float32)

    # Small block keeps the forward interaction reduction tile compact:
    # shape is [BLOCK, BLOCK_D].
    block = 16
    grid = (triton.cdiv(total, block),)

    _interaction_pair_flat_kernel[grid](
        dense,
        sparse,
        upstream,
        flat,
        TOTAL=total,
        OUT_NUMEL=out_numel,
        GD_NUMEL=gd_numel,
        GS_NUMEL=gs_numel,
        D=D,
        F=F,
        O=O,
        stride_db=dense.stride(0),
        stride_dd=dense.stride(1),
        stride_sb=sparse.stride(0),
        stride_sf=sparse.stride(1),
        stride_sd=sparse.stride(2),
        stride_ub=upstream.stride(0),
        stride_ud=upstream.stride(1),
        BLOCK=block,
        BLOCK_D=block_d,
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
