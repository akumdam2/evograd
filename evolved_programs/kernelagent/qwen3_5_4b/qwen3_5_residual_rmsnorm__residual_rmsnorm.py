"""adbench-tools artifact: task qwen3_5_4b.qwen3_5_residual_rmsnorm.residual_rmsnorm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for fused residual-add + RMSNorm with forward and backward.
#
# Operation:
#   summed = branch + residual
#   normed = summed * rsqrt(mean(summed^2) + eps) * (weight + 1)
#
# Note: this task's RMSNorm parameterization uses an implicit unit offset on the
# learned weight, i.e. effective_gamma = weight + 1.  The gradient w.r.t. weight
# is unchanged versus gamma because d(gamma)/d(weight) = 1.
#
# kernel_function returns:
#   cat([
#     normed.float().flatten(),
#     summed.float().flatten(),
#     d_branch.float().flatten(),
#     d_residual.float().flatten(),
#     d_weight.float().flatten(),
#   ])

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@triton.jit
def _forward_kernel(
    branch_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < n_cols
    idx = row * n_cols + offs

    b = tl.load(branch_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    x_sum = b + r
    x_mat = x_sum.to(residual_out_ptr.dtype.element_ty)
    x = x_mat.to(tl.float32)

    ss = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms = 1.0 / tl.sqrt(ss / n_cols + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = w + 1.0

    y = x * inv_rms * gamma

    tl.store(residual_out_ptr + idx, x_mat, mask=mask)
    tl.store(out_ptr + idx, y.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, inv_rms)


@triton.jit
def _backward_input_kernel(
    dy_ptr,
    dres_out_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dbranch_ptr,
    dresidual_ptr,
    n_cols: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < n_cols
    idx = row * n_cols + offs

    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(rstd_ptr + row).to(tl.float32)

    gamma = w + 1.0
    g = dy * gamma

    dot = tl.sum(tl.where(mask, g * x, 0.0), axis=0)
    inv3 = inv_rms * inv_rms * inv_rms

    dx = g * inv_rms - x * inv3 * (dot / n_cols)

    if HAS_DRES_OUT:
        dres_out = tl.load(dres_out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dx = dx + dres_out

    tl.store(dbranch_ptr + idx, dx.to(dbranch_ptr.dtype.element_ty), mask=mask)
    tl.store(dresidual_ptr + idx, dx.to(dresidual_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dweight_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    dweight_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, n_rows, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_rows

        elem = offs_m[:, None] * n_cols + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        dy = tl.load(dy_ptr + elem, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + elem, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(rstd_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        # d/dweight of (weight + 1) is 1, so no +1 appears here.
        acc += tl.sum(dy * x * inv[:, None], axis=0)

    tl.store(dweight_ptr + offs_n, acc.to(dweight_ptr.dtype.element_ty), mask=mask_n)


@triton.jit
def _flat_fused_forward_backward_kernel(
    branch_ptr,
    residual_ptr,
    weight_ptr,
    dy_ptr,
    dres_out_ptr,
    rstd_ptr,
    flat_ptr,
    n_cols: tl.constexpr,
    n_total: tl.constexpr,
    eps: tl.constexpr,
    HAS_DRES_OUT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < n_cols
    idx = row * n_cols + offs

    b = tl.load(branch_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    x_sum = b + r
    x_mat = x_sum.to(branch_ptr.dtype.element_ty)
    x = x_mat.to(tl.float32)

    ss = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms = 1.0 / tl.sqrt(ss / n_cols + eps)
    tl.store(rstd_ptr + row, inv_rms)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = w + 1.0

    y = x * inv_rms * gamma
    y_mat = y.to(branch_ptr.dtype.element_ty)

    # Forward output segments.
    tl.store(flat_ptr + idx, y_mat.to(tl.float32), mask=mask)
    tl.store(flat_ptr + n_total + idx, x_mat.to(tl.float32), mask=mask)

    # Backward input-gradient segment.
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = dy * gamma

    dot = tl.sum(tl.where(mask, g * x, 0.0), axis=0)
    inv3 = inv_rms * inv_rms * inv_rms

    dx = g * inv_rms - x * inv3 * (dot / n_cols)

    if HAS_DRES_OUT:
        dres_out = tl.load(dres_out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        dx = dx + dres_out

    dx_mat = dx.to(branch_ptr.dtype.element_ty).to(tl.float32)

    tl.store(flat_ptr + 2 * n_total + idx, dx_mat, mask=mask)
    tl.store(flat_ptr + 3 * n_total + idx, dx_mat, mask=mask)


@triton.jit
def _flat_dweight_kernel(
    dy_ptr,
    flat_ptr,
    rstd_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_total: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, n_rows, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n_rows

        elem = offs_m[:, None] * n_cols + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]

        # summed residual is stored in flat segment [n_total, 2 * n_total).
        x = tl.load(flat_ptr + n_total + elem, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + elem, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(rstd_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * inv[:, None], axis=0)

    # d_weight segment starts at 4 * n_total.
    tl.store(flat_ptr + 4 * n_total + offs_n, acc, mask=mask_n)


def qwen3_5_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps):
    assert branch.is_cuda and residual.is_cuda and weight.is_cuda
    assert branch.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert branch.shape == residual.shape

    n_cols = weight.numel()
    n_total = branch.numel()
    assert n_total % n_cols == 0
    n_rows = n_total // n_cols

    out = torch.empty_like(branch)
    residual_out = torch.empty_like(branch)
    rstd = torch.empty((n_rows,), device=branch.device, dtype=torch.float32)

    block_h = _next_power_of_2(n_cols)

    _forward_kernel[(n_rows,)](
        branch,
        residual,
        weight,
        out,
        residual_out,
        rstd,
        n_cols,
        eps,
        BLOCK_H=block_h,
        num_warps=8,
    )

    saved = (residual_out, weight, rstd)
    return (out, residual_out), saved


def qwen3_5_residual_rmsnorm_backward_from_saved(upstream, saved):
    x, weight, rstd = saved

    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
        has_dres_out = len(upstream) > 1
        dres_out = upstream[1] if has_dres_out else upstream[0]
    else:
        dy = upstream
        dres_out = upstream
        has_dres_out = False

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous() and weight.is_contiguous() and dy.is_contiguous()

    n_cols = weight.numel()
    n_total = x.numel()
    assert n_total % n_cols == 0
    n_rows = n_total // n_cols

    dbranch = torch.empty_like(x)
    dresidual = torch.empty_like(x)
    dweight = torch.empty_like(weight)

    block_h = _next_power_of_2(n_cols)

    _backward_input_kernel[(n_rows,)](
        dy,
        dres_out,
        x,
        weight,
        rstd,
        dbranch,
        dresidual,
        n_cols,
        HAS_DRES_OUT=has_dres_out,
        BLOCK_H=block_h,
        num_warps=8,
    )

    block_n = 256
    block_m = 16

    _dweight_kernel[(triton.cdiv(n_cols, block_n),)](
        dy,
        x,
        rstd,
        dweight,
        n_rows,
        n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )

    return dbranch, dresidual, dweight


def kernel_function(branch, residual, weight, eps, *upstream):
    """
    Complete fused forward+backward step.

    The row-local stages are fused into one Triton kernel:
      - residual add
      - RMS reduction / inverse RMS
      - RMSNorm output
      - d_branch and d_residual

    d_weight is a column-wise reduction across rows, so it is emitted by a
    second Triton reduction kernel to avoid inter-program atomics.
    """
    assert branch.is_cuda and residual.is_cuda and weight.is_cuda
    assert branch.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
    assert branch.shape == residual.shape

    if len(upstream) == 1 and isinstance(upstream[0], (tuple, list)):
        upstream = tuple(upstream[0])

    assert len(upstream) >= 1
    dy = upstream[0]
    has_dres_out = len(upstream) > 1
    dres_out = upstream[1] if has_dres_out else dy

    assert dy.is_cuda and dy.is_contiguous()

    n_cols = weight.numel()
    n_total = branch.numel()
    assert n_total % n_cols == 0
    n_rows = n_total // n_cols

    flat = torch.empty((4 * n_total + n_cols,), device=branch.device, dtype=torch.float32)
    rstd = torch.empty((n_rows,), device=branch.device, dtype=torch.float32)

    block_h = _next_power_of_2(n_cols)

    _flat_fused_forward_backward_kernel[(n_rows,)](
        branch,
        residual,
        weight,
        dy,
        dres_out,
        rstd,
        flat,
        n_cols,
        n_total,
        eps,
        HAS_DRES_OUT=has_dres_out,
        BLOCK_H=block_h,
        num_warps=8,
    )

    block_n = 256
    block_m = 16

    _flat_dweight_kernel[(triton.cdiv(n_cols, block_n),)](
        dy,
        flat,
        rstd,
        n_rows,
        n_cols,
        n_total,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_residual_rmsnorm', 'arguments': ['branch', 'residual', 'weight', 'eps'], 'outputs': ['normed', 'summed'], 'upstream_grads': ['dnormed', 'dsummed'], 'input_grads': ['dbranch', 'dresidual', 'dweight'], 'gradient_return_order': ['dbranch', 'dresidual', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'qwen3_5_residual_rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class Qwen35ResidualRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_residual_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, branch, residual, weight, eps):
        (normed, summed), saved = qwen3_5_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return normed, summed

    @staticmethod
    def backward(ctx, dnormed, dsummed):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dnormed = dnormed.contiguous() if (dnormed is not None and not dnormed.is_contiguous()) else dnormed
        dsummed = dsummed.contiguous() if (dsummed is not None and not dsummed.is_contiguous()) else dsummed
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dbranch, dresidual, dweight = qwen3_5_residual_rmsnorm_backward_from_saved(
            (dnormed, dsummed), saved, **_backward_scalars(qwen3_5_residual_rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dbranch, dresidual, dweight, None


def qwen3_5_residual_rmsnorm_deployment(branch, residual, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if branch.dim() > 3:
        _leading = branch.shape[:-1]
        branch = branch.reshape(-1, branch.shape[-1])
        residual = residual.reshape(-1, residual.shape[-1])
    normed, summed = Qwen35ResidualRmsnormFunction.apply(branch, residual, weight, eps)
    if _leading is not None:
        normed = normed.view(*_leading, normed.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return normed, summed


class Qwen35ResidualRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, branch, residual):
        return qwen3_5_residual_rmsnorm_deployment(branch, residual, self.weight, self.eps)


DEPLOYMENT_ENTRY = "qwen3_5_residual_rmsnorm_deployment"
