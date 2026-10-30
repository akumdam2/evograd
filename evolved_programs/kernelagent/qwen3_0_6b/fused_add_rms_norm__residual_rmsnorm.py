"""adbench-tools artifact: task qwen3_0_6b.fused_add_rms_norm.residual_rmsnorm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _num_warps_for_block(block_n: int) -> int:
    if block_n >= 2048:
        return 8
    if block_n >= 1024:
        return 4
    return 1


@triton.jit
def _forward_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    residual_ptr,
    inv_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

    residual_low = (x + r).to(x_ptr.dtype.element_ty)
    z = residual_low.to(tl.float32)

    mean_sq = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    inv = tl.rsqrt(mean_sq + eps)

    y = z * inv * w
    y_low = y.to(x_ptr.dtype.element_ty)

    tl.store(y_ptr + base, y_low, mask=mask)
    tl.store(residual_ptr + base, residual_low, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _bwd_dx_kernel(
    residual_ptr,
    w_ptr,
    dy_ptr,
    dres_ptr,
    inv_ptr,
    dx_ptr,
    dr_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL_GRAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    z = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)

    dot = tl.sum(tl.where(mask, dy * w * z, 0.0), axis=0)
    inv3 = inv * inv * inv
    dz = dy * w * inv - z * dot * inv3 / N

    if HAS_RESIDUAL_GRAD:
        dres = tl.load(dres_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dz += dres

    dz_low = dz.to(residual_ptr.dtype.element_ty)
    tl.store(dx_ptr + base, dz_low, mask=mask)
    tl.store(dr_ptr + base, dz_low, mask=mask)


@triton.jit
def _fwd_bwd_dx_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    dy_ptr,
    dres_ptr,
    out_ptr,
    inv_ptr,
    N: tl.constexpr,
    S: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL_GRAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    r = tl.load(r_ptr + base, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)

    residual_low = (x + r).to(x_ptr.dtype.element_ty)
    z = residual_low.to(tl.float32)

    mean_sq = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    inv = tl.rsqrt(mean_sq + eps)

    y = z * inv * w
    y_low_f32 = y.to(x_ptr.dtype.element_ty).to(tl.float32)

    dot = tl.sum(tl.where(mask, dy * w * z, 0.0), axis=0)
    inv3 = inv * inv * inv
    dz = dy * w * inv - z * dot * inv3 / N

    if HAS_RESIDUAL_GRAD:
        dres = tl.load(dres_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dz += dres

    dz_low_f32 = dz.to(x_ptr.dtype.element_ty).to(tl.float32)
    residual_f32 = residual_low.to(tl.float32)

    # Flat output layout:
    # [forward_y, forward_residual, grad_x, grad_r, grad_weight]
    tl.store(out_ptr + base, y_low_f32, mask=mask)
    tl.store(out_ptr + S + base, residual_f32, mask=mask)
    tl.store(out_ptr + 2 * S + base, dz_low_f32, mask=mask)
    tl.store(out_ptr + 3 * S + base, dz_low_f32, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _rms_norm_bwd_dw_partial_kernel(
    dy_ptr,
    residual_ptr,
    inv_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    z = tl.load(residual_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    inv = tl.load(inv_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    acc = tl.sum(dy * z * inv[:, None], axis=0)
    tl.store(partial_ptr + pid_m * N + offs_n, acc, mask=mask_n)


@triton.jit
def _rms_norm_bwd_dw_partial_from_flat_kernel(
    dy_ptr,
    out_ptr,
    inv_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]

    dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    z = tl.load(out_ptr + S + ptrs, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    inv = tl.load(inv_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    acc = tl.sum(dy * z * inv[:, None], axis=0)
    tl.store(partial_ptr + pid_m * N + offs_n, acc, mask=mask_n)


@triton.jit
def _rms_norm_bwd_dw_final_kernel(
    partial_ptr,
    w_ptr,
    dw_ptr,
    RB: tl.constexpr,
    N: tl.constexpr,
    BLOCK_RB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_rb = tl.arange(0, BLOCK_RB)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rb = offs_rb < RB
    mask_n = offs_n < N

    vals = tl.load(
        partial_ptr + offs_rb[:, None] * N + offs_n[None, :],
        mask=mask_rb[:, None] & mask_n[None, :],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    acc = tl.sum(vals, axis=0)
    acc_low = acc.to(w_ptr.dtype.element_ty)
    tl.store(dw_ptr + offs_n, acc_low, mask=mask_n)


@triton.jit
def _rms_norm_bwd_dw_final_to_flat_kernel(
    partial_ptr,
    w_ptr,
    out_ptr,
    RB: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_RB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_rb = tl.arange(0, BLOCK_RB)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rb = offs_rb < RB
    mask_n = offs_n < N

    vals = tl.load(
        partial_ptr + offs_rb[:, None] * N + offs_n[None, :],
        mask=mask_rb[:, None] & mask_n[None, :],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    acc = tl.sum(vals, axis=0)
    acc_low_f32 = acc.to(w_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + 4 * S + offs_n, acc_low_f32, mask=mask_n)


def fused_add_rms_norm_forward_with_saved(x, r, weight, eps):
    n = int(weight.numel())
    s = int(x.numel())
    m = s // n

    y = torch.empty_like(x)
    residual = torch.empty_like(x)
    inv = torch.empty((m,), device=x.device, dtype=torch.float32)

    block_n = _next_power_of_2(n)

    _forward_kernel[(m,)](
        x,
        r,
        weight,
        y,
        residual,
        inv,
        N=n,
        eps=float(eps),
        BLOCK_N=block_n,
        num_warps=_num_warps_for_block(block_n),
    )

    saved = (residual, inv, weight, float(eps))
    return (y, residual), saved


def fused_add_rms_norm_backward_from_saved(upstream, saved):
    residual, inv, weight, eps = saved

    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
        if len(upstream) > 1:
            dres = upstream[1]
            has_residual_grad = True
        else:
            dres = upstream[0]
            has_residual_grad = False
    else:
        dy = upstream
        dres = upstream
        has_residual_grad = False

    n = int(weight.numel())
    s = int(residual.numel())
    m = s // n

    dx = torch.empty_like(residual)
    dr = torch.empty_like(residual)
    dw = torch.empty_like(weight)

    block_n = _next_power_of_2(n)

    _bwd_dx_kernel[(m,)](
        residual,
        weight,
        dy,
        dres,
        inv,
        dx,
        dr,
        N=n,
        eps=float(eps),
        BLOCK_N=block_n,
        HAS_RESIDUAL_GRAD=has_residual_grad,
        num_warps=_num_warps_for_block(block_n),
    )

    block_m = 32
    row_blocks = triton.cdiv(m, block_m)
    partial = torch.empty((row_blocks, n), device=residual.device, dtype=torch.float32)

    _rms_norm_bwd_dw_partial_kernel[(triton.cdiv(n, block_n), row_blocks)](
        dy,
        residual,
        inv,
        partial,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )

    block_rb = _next_power_of_2(row_blocks)

    _rms_norm_bwd_dw_final_kernel[(triton.cdiv(n, block_n),)](
        partial,
        weight,
        dw,
        RB=row_blocks,
        N=n,
        BLOCK_RB=block_rb,
        BLOCK_N=block_n,
        num_warps=8,
    )

    return dx, dr, dw


def kernel_function(x, r, weight, eps, *upstream_grads):
    n = int(weight.numel())
    s = int(x.numel())
    m = s // n

    # Flat output layout matches:
    # torch.cat([forward_y, forward_residual, grad_x, grad_r, grad_weight])
    total = 4 * s + n
    out = torch.empty((total,), device=x.device, dtype=torch.float32)
    inv = torch.empty((m,), device=x.device, dtype=torch.float32)

    dy = upstream_grads[0]
    if len(upstream_grads) > 1:
        dres = upstream_grads[1]
        has_residual_grad = True
    else:
        dres = upstream_grads[0]
        has_residual_grad = False

    block_n = _next_power_of_2(n)

    # Fuses residual add, RMSNorm forward, saved inverse-RMS computation,
    # and input/residual-gradient computation in one pass. Weight-gradient
    # still needs a cross-row reduction, so it is completed by reduction
    # kernels below.
    _fwd_bwd_dx_kernel[(m,)](
        x,
        r,
        weight,
        dy,
        dres,
        out,
        inv,
        N=n,
        S=s,
        eps=float(eps),
        BLOCK_N=block_n,
        HAS_RESIDUAL_GRAD=has_residual_grad,
        num_warps=_num_warps_for_block(block_n),
    )

    block_m = 32
    row_blocks = triton.cdiv(m, block_m)
    partial = torch.empty((row_blocks, n), device=x.device, dtype=torch.float32)

    _rms_norm_bwd_dw_partial_from_flat_kernel[(triton.cdiv(n, block_n), row_blocks)](
        dy,
        out,
        inv,
        partial,
        M=m,
        N=n,
        S=s,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )

    block_rb = _next_power_of_2(row_blocks)

    _rms_norm_bwd_dw_final_to_flat_kernel[(triton.cdiv(n, block_n),)](
        partial,
        weight,
        out,
        RB=row_blocks,
        N=n,
        S=s,
        BLOCK_RB=block_rb,
        BLOCK_N=block_n,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'fused_add_rms_norm', 'arguments': ['x', 'r', 'weight', 'eps'], 'outputs': ['out', 'summed'], 'upstream_grads': ['dout', 'dsummed'], 'input_grads': ['dx', 'dr', 'dweight'], 'gradient_return_order': ['dx', 'dr', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'fused_add_rms_norm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class FusedAddRmsNormFunction(torch.autograd.Function):
    """Static autograd wiring for `fused_add_rms_norm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, r, weight, eps):
        (out, summed), saved = fused_add_rms_norm_forward_with_saved(x, r, weight, eps)
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
        dx, dr, dweight = fused_add_rms_norm_backward_from_saved(
            (dout, dsummed), saved, **_backward_scalars(fused_add_rms_norm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dr, dweight, None


def fused_add_rms_norm_deployment(x, r, weight, eps):
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
    out, summed = FusedAddRmsNormFunction.apply(x, r, weight, eps)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return out, summed


class FusedAddRmsNormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, r):
        return fused_add_rms_norm_deployment(x, r, self.weight, self.eps)


DEPLOYMENT_ENTRY = "fused_add_rms_norm_deployment"
