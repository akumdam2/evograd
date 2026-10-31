"""adbench-tools artifact: task llama_3_2_1b.llama3_residual_rmsnorm.residual_rmsnorm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Fused Triton implementation for:
#   residual = x + r
#   y        = RMSNorm(residual, weight, eps)
#
# and its backward pass.  The public kernel_function computes the complete
# forward+backward "pair step" expected by the evaluator and returns a single
# flattened float32 tensor in the order:
#
#   if two upstream tensors are supplied:
#       cat([y, residual, dx, dr, dweight])
#   if one upstream tensor is supplied:
#       cat([y, dx, dr, dweight])
#
# Fusion note:
#   The row-wise forward computation, RMS reduction, backward row reduction,
#   dx/dr computation, and per-element dweight contribution are fused in one
#   Triton pass.  A second Triton kernel performs the required cross-row
#   reduction for dweight.  This separation is necessary because dweight is a
#   reduction over all rows.

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    residual_ptr,
    inv_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    residual = x + r
    mean_sq = tl.sum(residual * residual, axis=0) / N
    inv = 1.0 / tl.sqrt(mean_sq + eps)
    y = residual * inv * w

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(residual_ptr + offs, residual, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _backward_rows_kernel(
    residual_ptr,
    inv_ptr,
    w_ptr,
    dy_ptr,
    dres_out_ptr,
    dx_ptr,
    dr_ptr,
    dw_contrib_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols

    residual = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm backward:
    # y_i = residual_i * inv_rms * weight_i
    # dx_i = dy_i*w_i*inv - residual_i*inv^3/N*sum_j(dy_j*w_j*residual_j)
    dy_w = dy * w
    dot = tl.sum(dy_w * residual, axis=0)
    inv3 = inv * inv * inv
    dnorm = dy_w * inv - residual * (inv3 * dot / N)

    if HAS_DRES_OUT:
        dres_out = tl.load(dres_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dtotal = dnorm + dres_out
    else:
        dtotal = dnorm

    dw_contrib = dy * residual * inv

    tl.store(dx_ptr + offs, dtotal, mask=mask)
    tl.store(dr_ptr + offs, dtotal, mask=mask)
    tl.store(dw_contrib_ptr + offs, dw_contrib, mask=mask)


@triton.jit
def _dw_reduce_kernel(
    dw_contrib_ptr,
    dw_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, M, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & col_mask[None, :]
        vals = tl.load(
            dw_contrib_ptr + rows[:, None] * N + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    tl.store(dw_ptr + cols, acc, mask=col_mask)


@triton.jit
def _fused_pair_rows_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    dy_ptr,
    dres_out_ptr,
    flat_out_ptr,
    dw_contrib_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
    STORE_RESIDUAL_OUTPUT: tl.constexpr,
    DX_OFFSET: tl.constexpr,
    DR_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    offs = row * N + cols
    E: tl.constexpr = M * N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    residual = x + r
    mean_sq = tl.sum(residual * residual, axis=0) / N
    inv = 1.0 / tl.sqrt(mean_sq + eps)
    y = residual * inv * w

    # Store forward output y.
    tl.store(
        flat_out_ptr + offs,
        y.to(x_ptr.dtype.element_ty).to(tl.float32),
        mask=mask,
    )

    # Optional second forward output: residual.
    if STORE_RESIDUAL_OUTPUT:
        tl.store(
            flat_out_ptr + E + offs,
            residual.to(x_ptr.dtype.element_ty).to(tl.float32),
            mask=mask,
        )

    # Backward.
    dy_w = dy * w
    dot = tl.sum(dy_w * residual, axis=0)
    inv3 = inv * inv * inv
    dnorm = dy_w * inv - residual * (inv3 * dot / N)

    if HAS_DRES_OUT:
        dres_out = tl.load(dres_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dtotal = dnorm + dres_out
    else:
        dtotal = dnorm

    dw_contrib = dy * residual * inv

    tl.store(
        flat_out_ptr + DX_OFFSET + offs,
        dtotal.to(x_ptr.dtype.element_ty).to(tl.float32),
        mask=mask,
    )
    tl.store(
        flat_out_ptr + DR_OFFSET + offs,
        dtotal.to(x_ptr.dtype.element_ty).to(tl.float32),
        mask=mask,
    )
    tl.store(dw_contrib_ptr + offs, dw_contrib, mask=mask)


@triton.jit
def _dw_reduce_to_flat_kernel(
    dw_contrib_ptr,
    w_ptr,
    flat_out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    DW_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, M, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & col_mask[None, :]
        vals = tl.load(
            dw_contrib_ptr + rows[:, None] * N + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    tl.store(
        flat_out_ptr + DW_OFFSET + cols,
        acc.to(w_ptr.dtype.element_ty).to(tl.float32),
        mask=col_mask,
    )


def _block_n(n: int) -> int:
    # Metadata-only helper.  Hidden sizes for Llama RMSNorm are normally powers
    # of two, but this also handles non-powers of two via masking.
    return triton.next_power_of_2(n)


def llama3_residual_rmsnorm_forward_with_saved(x, r, weight, eps):
    """
    Forward pass for residual RMSNorm.

    Computes entirely in Triton:
        residual = x + r
        y = residual * rsqrt(mean(residual^2) + eps) * weight

    Returns:
        outputs: (y, residual)
        saved:   values needed by the Triton backward implementation
    """
    assert x.is_cuda and r.is_cuda and weight.is_cuda
    assert x.is_contiguous() and r.is_contiguous() and weight.is_contiguous()
    assert x.shape == r.shape
    assert weight.dim() == 1
    N = weight.shape[0]
    assert x.numel() % N == 0

    M = x.numel() // N
    BLOCK_N = _block_n(N)

    y = torch.empty_like(x)
    residual = torch.empty_like(x)
    inv = torch.empty((M,), device=x.device, dtype=torch.float32)

    _forward_kernel[(M,)](
        x,
        r,
        weight,
        y,
        residual,
        inv,
        M,
        N,
        eps,
        BLOCK_N,
        num_warps=8,
    )

    return (y, residual), (residual, inv, weight)


def llama3_residual_rmsnorm_backward_from_saved(upstream, saved):
    """
    Backward pass for residual RMSNorm, implemented with Triton kernels only.

    Args:
        upstream:
            Either dy, or the tuple (dy, d_residual_output) when the forward
            residual output participates in the loss.
        saved:
            Tuple produced by llama3_residual_rmsnorm_forward_with_saved.

    Returns:
        (dx, dr, dweight)
    """
    residual, inv, weight = saved

    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
        dres_out = upstream[1]
        has_dres_out = True
    else:
        dy = upstream
        dres_out = dy
        has_dres_out = False

    assert residual.is_cuda and dy.is_cuda and weight.is_cuda
    assert residual.is_contiguous() and dy.is_contiguous() and weight.is_contiguous()

    N = weight.shape[0]
    assert residual.numel() % N == 0
    M = residual.numel() // N

    BLOCK_N = _block_n(N)
    BLOCK_M = 32

    dx = torch.empty_like(residual)
    dr = torch.empty_like(residual)
    dweight = torch.empty_like(weight)
    dw_contrib = torch.empty_like(residual, dtype=torch.float32)

    _backward_rows_kernel[(M,)](
        residual,
        inv,
        weight,
        dy,
        dres_out,
        dx,
        dr,
        dw_contrib,
        M,
        N,
        has_dres_out,
        BLOCK_N,
        num_warps=8,
    )

    grid_dw = (triton.cdiv(N, BLOCK_N),)
    _dw_reduce_kernel[grid_dw](
        dw_contrib,
        dweight,
        M,
        N,
        BLOCK_N,
        BLOCK_M,
        num_warps=8,
    )

    return dx, dr, dweight


def kernel_function(x, r, weight, eps, *upstream):
    """
    Complete fused forward+backward step for the evaluator.

    All numerical work is performed by Triton kernels.  The Python wrapper only
    validates metadata, allocates buffers, computes launch grids, and launches
    kernels.

    Expected call forms:
        kernel_function(x, r, weight, eps, dy, dresidual)
            returns cat([y, residual, dx, dr, dweight]) as one float32 vector.

        kernel_function(x, r, weight, eps, dy)
            returns cat([y, dx, dr, dweight]) as one float32 vector.
    """
    assert x.is_cuda and r.is_cuda and weight.is_cuda
    assert x.is_contiguous() and r.is_contiguous() and weight.is_contiguous()
    assert x.shape == r.shape
    assert weight.dim() == 1
    assert len(upstream) == 1 or len(upstream) == 2

    dy = upstream[0]
    if len(upstream) == 2:
        dres_out = upstream[1]
        has_dres_out = True
        store_residual_output = True
    else:
        dres_out = dy
        has_dres_out = False
        store_residual_output = False

    assert dy.is_cuda and dy.is_contiguous()
    if has_dres_out:
        assert dres_out.is_cuda and dres_out.is_contiguous()

    N = weight.shape[0]
    assert x.numel() % N == 0
    M = x.numel() // N
    E = x.numel()

    if store_residual_output:
        # [y, residual, dx, dr, dweight]
        total = 4 * E + N
        dx_offset = 2 * E
        dr_offset = 3 * E
        dw_offset = 4 * E
    else:
        # [y, dx, dr, dweight]
        total = 3 * E + N
        dx_offset = E
        dr_offset = 2 * E
        dw_offset = 3 * E

    out = torch.empty((total,), device=x.device, dtype=torch.float32)
    dw_contrib = torch.empty_like(x, dtype=torch.float32)

    BLOCK_N = _block_n(N)
    BLOCK_M = 32

    _fused_pair_rows_kernel[(M,)](
        x,
        r,
        weight,
        dy,
        dres_out,
        out,
        dw_contrib,
        M,
        N,
        eps,
        has_dres_out,
        store_residual_output,
        dx_offset,
        dr_offset,
        BLOCK_N,
        num_warps=8,
    )

    _dw_reduce_to_flat_kernel[(triton.cdiv(N, BLOCK_N),)](
        dw_contrib,
        weight,
        out,
        M,
        N,
        dw_offset,
        BLOCK_N,
        BLOCK_M,
        num_warps=8,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_residual_rmsnorm', 'arguments': ['x', 'r', 'weight', 'eps'], 'outputs': ['out', 'summed'], 'upstream_grads': ['dout', 'dsummed'], 'input_grads': ['dx', 'dr', 'dweight'], 'gradient_return_order': ['dx', 'dr', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'llama3_residual_rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class Llama3ResidualRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_residual_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, r, weight, eps):
        (out, summed), saved = llama3_residual_rmsnorm_forward_with_saved(x, r, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return out, summed

    @staticmethod
    def backward(ctx, dout, dsummed):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dout = dout.contiguous() if (dout is not None and not dout.is_contiguous()) else dout
        dsummed = dsummed.contiguous() if (dsummed is not None and not dsummed.is_contiguous()) else dsummed
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dr, dweight = llama3_residual_rmsnorm_backward_from_saved(
            (dout, dsummed), saved, **_backward_scalars(llama3_residual_rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dr, dweight, None


def llama3_residual_rmsnorm_deployment(x, r, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        r = r.reshape(-1, r.shape[-1])
    out, summed = Llama3ResidualRmsnormFunction.apply(x, r, weight, eps)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return out, summed


class Llama3ResidualRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-05):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, r):
        return llama3_residual_rmsnorm_deployment(x, r, self.weight, self.eps)


DEPLOYMENT_ENTRY = "llama3_residual_rmsnorm_deployment"
