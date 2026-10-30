"""adbench-tools artifact: task qwen3_5_4b.gdn_gating.gdn_gating, produced by kernelagent.

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


# ---------------------------------------------------------------------------
# Gated-DeltaNet gating:
#
# Forward:
#   g_i    = -exp(A_log[h]) * softplus(a_i + dt_bias[h])
#   beta_i = sigmoid(b_i)
#
# Backward:
#   da_i        = dg_i * (-exp(A_log[h]) * sigmoid(a_i + dt_bias[h]))
#   db_i        = dbeta_i * beta_i * (1 - beta_i)
#   dA_log[h]   = sum_i(dg_i * g_i)
#   ddt_bias[h] = sum_i(dg_i * (-exp(A_log[h]) * sigmoid(a_i + dt_bias[h])))
#
# Fix vs previous implementation:
#   The failing terms were the per-head reductions.  Atomic accumulation over
#   many rows gives non-deterministic / differently-rounded reductions and can
#   also repeatedly round when the destination is low precision.  The reduction
#   is now performed by one Triton program per head using tl.sum in fp32, then
#   stored once to the requested output dtype.  kernel_function uses a fused
#   per-head kernel that computes forward values, elementwise gradients, and
#   both reductions in the same pass, avoiding the previous zero+atomic tail.
# ---------------------------------------------------------------------------


@triton.jit
def _softplus_f32(x):
    ax = tl.abs(x)
    return tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-ax))


@triton.jit
def _sigmoid_f32(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _logical_offset_and_head(
    idx,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    NDIM: tl.constexpr,
    H_AXIS: tl.constexpr,
):
    """Map contiguous logical flat index -> physical storage offset and head."""
    if NDIM == 1:
        i0 = idx
        off = i0 * S0
        h = i0
    elif NDIM == 2:
        i0 = idx // D1
        i1 = idx - i0 * D1
        off = i0 * S0 + i1 * S1
        if H_AXIS == 0:
            h = i0
        else:
            h = i1
    else:
        d12 = D1 * D2
        i0 = idx // d12
        rem = idx - i0 * d12
        i1 = rem // D2
        i2 = rem - i1 * D2
        off = i0 * S0 + i1 * S1 + i2 * S2
        if H_AXIS == 0:
            h = i0
        elif H_AXIS == 1:
            h = i1
        else:
            h = i2
    return off, h


@triton.jit
def _logical_index_and_offset_from_head_row(
    h,
    r,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    NDIM: tl.constexpr,
    H_AXIS: tl.constexpr,
):
    """Map (head, flattened non-head row) -> logical flat index and storage offset."""
    if NDIM == 1:
        i0 = h
        idx = i0
        off = i0 * S0
    elif NDIM == 2:
        if H_AXIS == 0:
            i0 = h
            i1 = r
        else:
            i0 = r
            i1 = h
        idx = i0 * D1 + i1
        off = i0 * S0 + i1 * S1
    else:
        if H_AXIS == 0:
            i0 = h
            i1 = r // D2
            i2 = r - i1 * D2
        elif H_AXIS == 1:
            i0 = r // D2
            i1 = h
            i2 = r - i0 * D2
        else:
            i0 = r // D1
            i1 = r - i0 * D1
            i2 = h
        idx = i0 * (D1 * D2) + i1 * D2 + i2
        off = i0 * S0 + i1 * S1 + i2 * S2
    return idx, off


@triton.jit
def _gdn_forward_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    n_elements: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    NDIM: tl.constexpr,
    H_AXIS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    a_off, h = _logical_offset_and_head(idx, D0, D1, D2, AS0, AS1, AS2, NDIM, H_AXIS)
    b_off, _ = _logical_offset_and_head(idx, D0, D1, D2, BS0, BS1, BS2, NDIM, H_AXIS)

    a = tl.load(a_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    A_log = tl.load(A_log_ptr + h, mask=mask, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h, mask=mask, other=0.0).to(tl.float32)

    x = a + dt_bias
    g = -tl.exp(A_log) * _softplus_f32(x)
    beta = _sigmoid_f32(b)

    tl.store(g_ptr + idx, g, mask=mask)
    tl.store(beta_ptr + idx, beta, mask=mask)


@triton.jit
def _gdn_backward_by_head_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dg_ptr,
    dbeta_ptr,
    grad_a_ptr,
    grad_b_ptr,
    grad_A_log_ptr,
    grad_dt_bias_ptr,
    rows_per_head: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    DGS0: tl.constexpr,
    DGS1: tl.constexpr,
    DGS2: tl.constexpr,
    DBS0: tl.constexpr,
    DBS1: tl.constexpr,
    DBS2: tl.constexpr,
    GAS0: tl.constexpr,
    GAS1: tl.constexpr,
    GAS2: tl.constexpr,
    GBS0: tl.constexpr,
    GBS1: tl.constexpr,
    GBS2: tl.constexpr,
    NDIM: tl.constexpr,
    H_AXIS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    h = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < rows_per_head

    idx_a, a_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, AS0, AS1, AS2, NDIM, H_AXIS
    )
    _, b_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, BS0, BS1, BS2, NDIM, H_AXIS
    )
    _, dg_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, DGS0, DGS1, DGS2, NDIM, H_AXIS
    )
    _, db_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, DBS0, DBS1, DBS2, NDIM, H_AXIS
    )
    _, ga_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, GAS0, GAS1, GAS2, NDIM, H_AXIS
    )
    _, gb_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, GBS0, GBS1, GBS2, NDIM, H_AXIS
    )

    a = tl.load(a_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    dg = tl.load(dg_ptr + dg_off, mask=mask, other=0.0).to(tl.float32)
    dbeta = tl.load(dbeta_ptr + db_off, mask=mask, other=0.0).to(tl.float32)

    A_log = tl.load(A_log_ptr + h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h).to(tl.float32)

    x = a + dt_bias
    eA = tl.exp(A_log)
    sp = _softplus_f32(x)
    sig_x = _sigmoid_f32(x)
    beta = _sigmoid_f32(b)

    g = -eA * sp
    pre = -eA * sig_x

    grad_a = dg * pre
    grad_b = dbeta * beta * (1.0 - beta)

    grad_A_sum = tl.sum(tl.where(mask, dg * g, 0.0), axis=0)
    grad_dt_sum = tl.sum(tl.where(mask, dg * pre, 0.0), axis=0)

    tl.store(grad_a_ptr + ga_off, grad_a, mask=mask)
    tl.store(grad_b_ptr + gb_off, grad_b, mask=mask)
    tl.store(grad_A_log_ptr + h, grad_A_sum)
    tl.store(grad_dt_bias_ptr + h, grad_dt_sum)


@triton.jit
def _gdn_pair_flat_by_head_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    dg_ptr,
    dbeta_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    h_size: tl.constexpr,
    rows_per_head: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
    DGS0: tl.constexpr,
    DGS1: tl.constexpr,
    DGS2: tl.constexpr,
    DBS0: tl.constexpr,
    DBS1: tl.constexpr,
    DBS2: tl.constexpr,
    NDIM: tl.constexpr,
    H_AXIS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    h = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    mask = r < rows_per_head

    idx, a_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, AS0, AS1, AS2, NDIM, H_AXIS
    )
    _, b_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, BS0, BS1, BS2, NDIM, H_AXIS
    )
    _, dg_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, DGS0, DGS1, DGS2, NDIM, H_AXIS
    )
    _, db_off = _logical_index_and_offset_from_head_row(
        h, r, D0, D1, D2, DBS0, DBS1, DBS2, NDIM, H_AXIS
    )

    a = tl.load(a_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
    dg = tl.load(dg_ptr + dg_off, mask=mask, other=0.0).to(tl.float32)
    dbeta = tl.load(dbeta_ptr + db_off, mask=mask, other=0.0).to(tl.float32)

    A_log = tl.load(A_log_ptr + h).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + h).to(tl.float32)

    x = a + dt_bias
    eA = tl.exp(A_log)
    sp = _softplus_f32(x)
    sig_x = _sigmoid_f32(x)

    g = -eA * sp
    beta = _sigmoid_f32(b)

    pre = -eA * sig_x
    grad_a = dg * pre
    grad_b = dbeta * beta * (1.0 - beta)

    grad_A_sum = tl.sum(tl.where(mask, dg * g, 0.0), axis=0)
    grad_dt_sum = tl.sum(tl.where(mask, dg * pre, 0.0), axis=0)

    base_g = 0
    base_beta = n_elements
    base_ga = 2 * n_elements
    base_gb = 3 * n_elements
    base_gA = 4 * n_elements
    base_gdt = 4 * n_elements + h_size

    # Match the wrapper outputs exactly after their dtype casts:
    # g is float32; beta, da, db, dA_log, ddt_bias follow their destination/input dtypes.
    tl.store(out_ptr + base_g + idx, g, mask=mask)
    tl.store(out_ptr + base_beta + idx, beta.to(b_ptr.dtype.element_ty).to(tl.float32), mask=mask)
    tl.store(out_ptr + base_ga + idx, grad_a.to(a_ptr.dtype.element_ty).to(tl.float32), mask=mask)
    tl.store(out_ptr + base_gb + idx, grad_b.to(b_ptr.dtype.element_ty).to(tl.float32), mask=mask)

    tl.store(
        out_ptr + base_gA + h,
        grad_A_sum.to(A_log_ptr.dtype.element_ty).to(tl.float32),
    )
    tl.store(
        out_ptr + base_gdt + h,
        grad_dt_sum.to(dt_bias_ptr.dtype.element_ty).to(tl.float32),
    )


def _meta(x, h_size: int):
    """Return shape/stride metadata for 1D, 2D, or 3D tensors."""
    ndim = x.dim()
    assert 1 <= ndim <= 3, "This Triton implementation supports 1D, 2D, or 3D tensors"

    shape = tuple(x.shape)
    stride = tuple(x.stride())

    d0 = shape[0]
    d1 = shape[1] if ndim >= 2 else 1
    d2 = shape[2] if ndim >= 3 else 1

    s0 = stride[0]
    s1 = stride[1] if ndim >= 2 else 1
    s2 = stride[2] if ndim >= 3 else 1

    if ndim == 1:
        h_axis = 0
    elif shape[-1] == h_size:
        h_axis = ndim - 1
    elif ndim >= 2 and shape[1] == h_size:
        h_axis = 1
    elif shape[0] == h_size:
        h_axis = 0
    else:
        h_axis = ndim - 1

    return ndim, h_axis, d0, d1, d2, s0, s1, s2


def _pow2_ceil(x: int):
    return 1 << (x - 1).bit_length()


def gdn_gating_forward_with_saved(a, b, A_log, dt_bias):
    assert a.is_cuda and b.is_cuda and A_log.is_cuda and dt_bias.is_cuda
    assert a.shape == b.shape
    assert A_log.numel() == dt_bias.numel()

    h_size = A_log.numel()
    n_elements = a.numel()

    ndim, h_axis, d0, d1, d2, as0, as1, as2 = _meta(a, h_size)
    _, _, _, _, _, bs0, bs1, bs2 = _meta(b, h_size)

    g = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    beta = torch.empty_like(b)

    block = 256
    grid = (triton.cdiv(n_elements, block),)
    _gdn_forward_kernel[grid](
        a,
        b,
        A_log,
        dt_bias,
        g,
        beta,
        n_elements,
        d0,
        d1,
        d2,
        as0,
        as1,
        as2,
        bs0,
        bs1,
        bs2,
        ndim,
        h_axis,
        BLOCK_SIZE=block,
    )

    saved = (a, b, A_log, dt_bias)
    return (g, beta), saved


def gdn_gating_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        dg, dbeta = upstream
    else:
        dg = upstream
        dbeta = upstream

    a, b, A_log, dt_bias = saved
    assert a.is_cuda and b.is_cuda and A_log.is_cuda and dt_bias.is_cuda
    assert dg.is_cuda and dbeta.is_cuda

    h_size = A_log.numel()
    n_elements = a.numel()
    assert n_elements % h_size == 0
    rows_per_head = n_elements // h_size
    block_r = _pow2_ceil(rows_per_head)

    ndim, h_axis, d0, d1, d2, as0, as1, as2 = _meta(a, h_size)
    _, _, _, _, _, bs0, bs1, bs2 = _meta(b, h_size)
    _, _, _, _, _, dgs0, dgs1, dgs2 = _meta(dg, h_size)
    _, _, _, _, _, dbs0, dbs1, dbs2 = _meta(dbeta, h_size)

    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    grad_A_log = torch.empty_like(A_log)
    grad_dt_bias = torch.empty_like(dt_bias)

    _, _, _, _, _, gas0, gas1, gas2 = _meta(grad_a, h_size)
    _, _, _, _, _, gbs0, gbs1, gbs2 = _meta(grad_b, h_size)

    _gdn_backward_by_head_kernel[(h_size,)](
        a,
        b,
        A_log,
        dt_bias,
        dg,
        dbeta,
        grad_a,
        grad_b,
        grad_A_log,
        grad_dt_bias,
        rows_per_head,
        d0,
        d1,
        d2,
        as0,
        as1,
        as2,
        bs0,
        bs1,
        bs2,
        dgs0,
        dgs1,
        dgs2,
        dbs0,
        dbs1,
        dbs2,
        gas0,
        gas1,
        gas2,
        gbs0,
        gbs1,
        gbs2,
        ndim,
        h_axis,
        BLOCK_R=block_r,
    )

    return grad_a, grad_b, grad_A_log, grad_dt_bias


def kernel_function(a, b, A_log, dt_bias, *upstream):
    """
    Fused forward+backward evaluator entry point.

    Returns:
      cat([
        g.float().reshape(-1),
        beta.float().reshape(-1),
        grad_a.float().reshape(-1),
        grad_b.float().reshape(-1),
        grad_A_log.float().reshape(-1),
        grad_dt_bias.float().reshape(-1),
      ])

    The forward elementwise operations, backward elementwise operations, and
    per-head reductions are fused into a single per-head Triton kernel.  This
    avoids atomic reductions and performs the reduced stores once per head.
    """
    assert a.is_cuda and b.is_cuda and A_log.is_cuda and dt_bias.is_cuda
    assert a.shape == b.shape
    assert A_log.numel() == dt_bias.numel()
    assert len(upstream) >= 2, "gdn_gating has two outputs, so two upstream tensors are expected"

    dg = upstream[0]
    dbeta = upstream[1]
    assert dg.is_cuda and dbeta.is_cuda

    h_size = A_log.numel()
    n_elements = a.numel()
    assert n_elements % h_size == 0
    rows_per_head = n_elements // h_size
    block_r = _pow2_ceil(rows_per_head)

    total = 4 * n_elements + 2 * h_size
    out = torch.empty((total,), device=a.device, dtype=torch.float32)

    ndim, h_axis, d0, d1, d2, as0, as1, as2 = _meta(a, h_size)
    _, _, _, _, _, bs0, bs1, bs2 = _meta(b, h_size)
    _, _, _, _, _, dgs0, dgs1, dgs2 = _meta(dg, h_size)
    _, _, _, _, _, dbs0, dbs1, dbs2 = _meta(dbeta, h_size)

    _gdn_pair_flat_by_head_kernel[(h_size,)](
        a,
        b,
        A_log,
        dt_bias,
        dg,
        dbeta,
        out,
        n_elements,
        h_size,
        rows_per_head,
        d0,
        d1,
        d2,
        as0,
        as1,
        as2,
        bs0,
        bs1,
        bs2,
        dgs0,
        dgs1,
        dgs2,
        dbs0,
        dbs1,
        dbs2,
        ndim,
        h_axis,
        BLOCK_R=block_r,
    )

    return out
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gdn_gating', 'arguments': ['a', 'b', 'A_log', 'dt_bias'], 'outputs': ['g', 'beta'], 'upstream_grads': ['dg', 'dbeta'], 'input_grads': ['da', 'db', 'dA_log', 'ddt_bias'], 'gradient_return_order': ['da', 'db', 'dA_log', 'ddt_bias'], 'scalars': [], 'deployment_entry': 'gdn_gating_deployment', 'allowed_primitives': []}


class GdnGatingFunction(torch.autograd.Function):
    """Static autograd wiring for `gdn_gating`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, a, b, A_log, dt_bias):
        (g, beta), saved = gdn_gating_forward_with_saved(a, b, A_log, dt_bias)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return g, beta

    @staticmethod
    def backward(ctx, dg, dbeta):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dg = dg.contiguous() if (dg is not None and not dg.is_contiguous()) else dg
        dbeta = dbeta.contiguous() if (dbeta is not None and not dbeta.is_contiguous()) else dbeta
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        da, db, dA_log, ddt_bias = gdn_gating_backward_from_saved(
            (dg, dbeta), saved
        )
        # One value per forward argument, `None` for every inactive one.
        return da, db, dA_log, ddt_bias


def gdn_gating_deployment(a, b, A_log, dt_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if a.dim() > 2:
        _leading = a.shape[:-1]
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    g, beta = GdnGatingFunction.apply(a, b, A_log, dt_bias)
    if _leading is not None:
        g = g.view(*_leading, g.shape[-1])
        beta = beta.view(*_leading, beta.shape[-1])
    return g, beta


class GdnGatingModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, A_log, dt_bias):
        super().__init__()
        self.A_log = (A_log if isinstance(A_log, torch.nn.Parameter)
                  else torch.nn.Parameter(A_log))
        self.dt_bias = (dt_bias if isinstance(dt_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(dt_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, a, b):
        return gdn_gating_deployment(a, b, self.A_log, self.dt_bias)


DEPLOYMENT_ENTRY = "gdn_gating_deployment"
