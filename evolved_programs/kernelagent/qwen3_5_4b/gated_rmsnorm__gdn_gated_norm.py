"""adbench-tools artifact: task qwen3_5_4b.gated_rmsnorm.gdn_gated_norm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the fused forward+backward step of a gated RMSNorm.
#
# Operator:
#   rstd = rsqrt(mean(x^2, dim=-1) + eps)
#   gate = silu(z) = z * sigmoid(z)
#   y    = x * rstd * weight * gate
#
# Backward:
#   Computes dx, dz, and dweight.
#
# kernel_function returns:
#   cat([y.float().reshape(-1),
#        dx.float().reshape(-1),
#        dz.float().reshape(-1),
#        dweight.float().reshape(-1)])
#
# Notes on fusion:
#   The per-row forward, dx, dz, and per-row dweight contribution are fused.
#   The final dweight requires a reduction across all rows, so it is performed
#   in a second Triton kernel. Keeping this separate avoids unsafe global
#   atomics and preserves deterministic reduction behavior.

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_rmsnorm_fwd_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    y_ptr,
    rstd_ptr,
    n_rows: tl.constexpr,
    dim: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row * dim + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / dim
    rstd = tl.rsqrt(var + eps)

    sig = tl.sigmoid(z)
    gate = z * sig

    y = x * rstd * w * gate

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _gated_rmsnorm_bwd_rows_kernel(
    dy_ptr,
    x_ptr,
    z_ptr,
    w_ptr,
    rstd_ptr,
    dx_ptr,
    dz_ptr,
    partial_dw_ptr,
    n_rows: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row * dim + offs

    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    sig = tl.sigmoid(z)
    gate = z * sig
    dgate_dz = sig * (1.0 + z * (1.0 - sig))

    xhat = x * rstd

    # y = xhat * w * gate
    dweight_contrib = dy * xhat * gate
    dgate = dy * xhat * w
    dz = dgate * dgate_dz

    dxhat = dy * w * gate

    # xhat = x * rstd
    # rstd = rsqrt(mean(x^2) + eps)
    dot = tl.sum(dxhat * x, axis=0)
    dx = dxhat * rstd - x * (rstd * rstd * rstd) * dot / dim

    tl.store(dx_ptr + base, dx, mask=mask)
    tl.store(dz_ptr + base, dz, mask=mask)
    tl.store(partial_dw_ptr + base, dweight_contrib, mask=mask)


@triton.jit
def _reduce_dw_kernel(
    partial_dw_ptr,
    dw_ptr,
    n_rows: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_d = tl.program_id(0)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < dim

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for row_start in tl.range(0, n_rows, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < n_rows) & col_mask[None, :]

        vals = tl.load(
            partial_dw_ptr + rows[:, None] * dim + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(vals, axis=0)

    tl.store(dw_ptr + cols, acc, mask=col_mask)


@triton.jit
def _gated_rmsnorm_fwd_bwd_flat_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    dy_ptr,
    flat_out_ptr,
    partial_dw_ptr,
    n_rows: tl.constexpr,
    dim: tl.constexpr,
    n_elements: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK)
    mask = offs < dim
    base = row * dim + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)

    # Forward RMSNorm statistics.
    var = tl.sum(x * x, axis=0) / dim
    rstd = tl.rsqrt(var + eps)

    # Gating.
    sig = tl.sigmoid(z)
    gate = z * sig
    dgate_dz = sig * (1.0 + z * (1.0 - sig))

    xhat = x * rstd
    y = xhat * w * gate

    # Backward.
    dweight_contrib = dy * xhat * gate

    dgate = dy * xhat * w
    dz = dgate * dgate_dz

    dxhat = dy * w * gate
    dot = tl.sum(dxhat * x, axis=0)
    dx = dxhat * rstd - x * (rstd * rstd * rstd) * dot / dim

    # Flat layout: [y | dx | dz | dweight]
    tl.store(flat_out_ptr + base, y, mask=mask)
    tl.store(flat_out_ptr + n_elements + base, dx, mask=mask)
    tl.store(flat_out_ptr + 2 * n_elements + base, dz, mask=mask)

    tl.store(partial_dw_ptr + base, dweight_contrib, mask=mask)


@triton.jit
def _reduce_dw_to_flat_kernel(
    partial_dw_ptr,
    flat_out_ptr,
    n_rows: tl.constexpr,
    dim: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_d = tl.program_id(0)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < dim

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for row_start in tl.range(0, n_rows, BLOCK_M):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < n_rows) & col_mask[None, :]

        vals = tl.load(
            partial_dw_ptr + rows[:, None] * dim + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(vals, axis=0)

    tl.store(flat_out_ptr + 3 * n_elements + cols, acc, mask=col_mask)


def _block_size(dim: int) -> int:
    return triton.next_power_of_2(dim)


def _num_warps(block: int) -> int:
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    return 4


def gated_rmsnorm_forward_with_saved(x, z, weight, eps):
    """
    Forward pass for gated RMSNorm.

    Returns:
      y, saved

    saved contains only tensors/scalars needed by the Triton backward path.
    """
    assert x.is_cuda and z.is_cuda and weight.is_cuda
    assert x.is_contiguous() and z.is_contiguous() and weight.is_contiguous()
    assert x.shape == z.shape
    assert weight.numel() == x.shape[-1]

    dim = x.shape[-1]
    n_rows = x.numel() // dim
    block = _block_size(dim)

    y = torch.empty_like(x)
    rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    _gated_rmsnorm_fwd_kernel[(n_rows,)](
        x,
        z,
        weight,
        y,
        rstd,
        n_rows,
        dim,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )

    saved = (x, z, weight, rstd, eps)
    return y, saved


def _extract_saved_tuple(primary, extra):
    """
    Compatibility helper for evaluator deployment interfaces.

    Some harnesses call:
      backward_from_saved(upstream, saved)

    Others call:
      backward_from_saved(upstream, outputs, saved)

    This helper selects the saved tuple without doing tensor computation.
    """
    candidates = (primary,) + extra
    for cand in reversed(candidates):
        if isinstance(cand, (tuple, list)) and len(cand) >= 5:
            return cand
    return primary


def gated_rmsnorm_backward_from_saved(upstream, saved, *extra):
    """
    Backward pass from saved tensors.

    Accepts both deployment signatures:
      gated_rmsnorm_backward_from_saved(upstream, saved)
      gated_rmsnorm_backward_from_saved(upstream, outputs, saved)

    The additional positional argument is ignored except when it is the saved
    tuple. This fixes deployment-interface compatibility while preserving the
    Triton-only compute path.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    saved = _extract_saved_tuple(saved, extra)
    x, z, weight, rstd, eps = saved[:5]

    assert dy.is_cuda and x.is_cuda and z.is_cuda and weight.is_cuda
    assert dy.is_contiguous() and x.is_contiguous() and z.is_contiguous()
    assert weight.is_contiguous()
    assert dy.shape == x.shape
    assert x.shape == z.shape
    assert weight.numel() == x.shape[-1]

    dim = x.shape[-1]
    n_rows = x.numel() // dim
    block = _block_size(dim)

    dx = torch.empty_like(x)
    dz = torch.empty_like(z)
    dweight = torch.empty_like(weight)

    partial_dw = torch.empty((n_rows, dim), device=x.device, dtype=torch.float32)

    _gated_rmsnorm_bwd_rows_kernel[(n_rows,)](
        dy,
        x,
        z,
        weight,
        rstd,
        dx,
        dz,
        partial_dw,
        n_rows,
        dim,
        BLOCK=block,
        num_warps=_num_warps(block),
    )

    block_d = block
    block_m = 8

    _reduce_dw_kernel[(triton.cdiv(dim, block_d),)](
        partial_dw,
        dweight,
        n_rows,
        dim,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        num_warps=_num_warps(block_d),
    )

    return dx, dz, dweight


def kernel_function(x, z, weight, eps, upstream):
    """
    Evaluator entry point.

    Computes the full forward+backward pair and returns a single flattened
    float32 tensor equivalent to:

      torch.cat([
        y.float().reshape(-1),
        dx.float().reshape(-1),
        dz.float().reshape(-1),
        dweight.float().reshape(-1),
      ])

    Python only validates inputs, allocates tensors, and launches Triton kernels.
    All arithmetic is performed in Triton.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    assert x.is_cuda and z.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous() and z.is_contiguous() and weight.is_contiguous()
    assert dy.is_contiguous()
    assert x.shape == z.shape
    assert x.shape == dy.shape
    assert weight.numel() == x.shape[-1]

    dim = x.shape[-1]
    n_rows = x.numel() // dim
    n_elements = x.numel()
    block = _block_size(dim)

    # Flat output layout: [y | dx | dz | dweight]
    flat = torch.empty((3 * n_elements + dim,), device=x.device, dtype=torch.float32)
    partial_dw = torch.empty((n_rows, dim), device=x.device, dtype=torch.float32)

    _gated_rmsnorm_fwd_bwd_flat_kernel[(n_rows,)](
        x,
        z,
        weight,
        dy,
        flat,
        partial_dw,
        n_rows,
        dim,
        n_elements,
        eps,
        BLOCK=block,
        num_warps=_num_warps(block),
    )

    block_d = block
    block_m = 8

    _reduce_dw_to_flat_kernel[(triton.cdiv(dim, block_d),)](
        partial_dw,
        flat,
        n_rows,
        dim,
        n_elements,
        BLOCK_D=block_d,
        BLOCK_M=block_m,
        num_warps=_num_warps(block_d),
    )

    return flat
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gated_rmsnorm', 'arguments': ['x', 'z', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dz', 'dweight'], 'gradient_return_order': ['dx', 'dz', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'gated_rmsnorm_deployment', 'allowed_primitives': []}


class GatedRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `gated_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, z, weight, eps):
        y, saved = gated_rmsnorm_forward_with_saved(x, z, weight, eps)
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
        dx, dz, dweight = gated_rmsnorm_backward_from_saved(
            dy, saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dz, dweight, None


def gated_rmsnorm_deployment(x, z, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        z = z.reshape(-1, z.shape[-1])
    y = GatedRmsnormFunction.apply(x, z, weight, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class GatedRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, z):
        return gated_rmsnorm_deployment(x, z, self.weight, self.eps)


DEPLOYMENT_ENTRY = "gated_rmsnorm_deployment"
