"""adbench-tools artifact: task olmoe_1b_7b.olmoe_residual_rmsnorm.residual_rmsnorm, produced by kernelagent.

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


@triton.jit
def _forward_save_kernel(
    branch_ptr,
    residual_ptr,
    weight_ptr,
    y_ptr,
    residual_out_ptr,
    x_save_ptr,
    rstd_ptr,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    b = tl.load(branch_ptr + base, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    x = b + r
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    y = x * rstd * w

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(residual_out_ptr + base, x, mask=mask)
    tl.store(x_save_ptr + base, x, mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def _backward_dx_kernel(
    dy_ptr,
    dresidual_out_ptr,
    x_ptr,
    weight_ptr,
    rstd_ptr,
    dbranch_ptr,
    dresidual_ptr,
    N: tl.constexpr,
    HAS_RESIDUAL_OUT_GRAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row).to(tl.float32)

    a = dy * w
    dot = tl.sum(a * x, axis=0)
    coeff = dot * rstd * rstd * rstd / N
    dx = a * rstd - x * coeff

    if HAS_RESIDUAL_OUT_GRAD:
        dres_out = tl.load(dresidual_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dx += dres_out

    tl.store(dbranch_ptr + base, dx, mask=mask)
    tl.store(dresidual_ptr + base, dx, mask=mask)


@triton.jit
def _fwd_bwd_dx_pack_kernel(
    branch_ptr,
    residual_ptr,
    weight_ptr,
    dy_ptr,
    dresidual_out_ptr,
    packed_ptr,
    rstd_ptr,
    N: tl.constexpr,
    P: tl.constexpr,
    eps,
    HAS_RESIDUAL_OUT_GRAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * N + offs

    b = tl.load(branch_ptr + base, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    x = b + r
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    y = x * rstd * w

    # Packed layout:
    # [0:P]       forward rmsnorm output
    # [P:2P]      residual output = branch + residual
    # [2P:3P]     d_branch
    # [3P:4P]     d_residual
    # [4P:4P+N]   d_weight
    tl.store(packed_ptr + base, y, mask=mask)
    tl.store(packed_ptr + P + base, x, mask=mask)
    tl.store(rstd_ptr + row, rstd)

    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    a = dy * w
    dot = tl.sum(a * x, axis=0)
    coeff = dot * rstd * rstd * rstd / N
    dx = a * rstd - x * coeff

    if HAS_RESIDUAL_OUT_GRAD:
        dres = tl.load(dresidual_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dx += dres

    tl.store(packed_ptr + 2 * P + base, dx, mask=mask)
    tl.store(packed_ptr + 3 * P + base, dx, mask=mask)


@triton.jit
def _dweight_partial_kernel(
    dy_ptr,
    x_ptr,
    rstd_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    X_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    dy = tl.load(dy_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + X_OFFSET + ptrs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)

    acc = tl.sum(dy * x * rstd[:, None], axis=0)
    tl.store(partial_ptr + pid_m * N + offs_n, acc, mask=offs_n < N)


@triton.jit
def _dweight_reduce_kernel(
    partial_ptr,
    out_ptr,
    N: tl.constexpr,
    OUT_OFFSET: tl.constexpr,
    NUM_BLOCKS_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_b = tl.arange(0, BLOCK_B)

    ptrs = offs_b[:, None] * N + offs_n[None, :]
    mask = (offs_b[:, None] < NUM_BLOCKS_M) & (offs_n[None, :] < N)

    vals = tl.load(partial_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)

    tl.store(out_ptr + OUT_OFFSET + offs_n, acc, mask=offs_n < N)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def olmoe_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps):
    N = weight.numel()
    P = branch.numel()
    M = P // N

    block_n = _next_power_of_2(N)
    row_warps = 8 if block_n >= 2048 else 4

    y = torch.empty_like(branch)
    residual_out = torch.empty_like(branch)
    x_save = torch.empty(branch.shape, device=branch.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=branch.device, dtype=torch.float32)

    _forward_save_kernel[(M,)](
        branch,
        residual,
        weight,
        y,
        residual_out,
        x_save,
        rstd,
        N,
        eps,
        block_n,
        num_warps=row_warps,
    )

    saved = (x_save, rstd, weight)
    return (y, residual_out), saved


def olmoe_residual_rmsnorm_backward_from_saved(upstream, saved):
    x_save, rstd, weight = saved

    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
        if len(upstream) > 1:
            dresidual_out = upstream[1]
            has_residual_out_grad = True
        else:
            dresidual_out = dy
            has_residual_out_grad = False
    else:
        dy = upstream
        dresidual_out = dy
        has_residual_out_grad = False

    N = weight.numel()
    P = x_save.numel()
    M = P // N

    block_n = _next_power_of_2(N)
    row_warps = 8 if block_n >= 2048 else 4

    dbranch = torch.empty_like(dy)
    dresidual = torch.empty_like(dy)
    dweight = torch.empty_like(weight)

    _backward_dx_kernel[(M,)](
        dy,
        dresidual_out,
        x_save,
        weight,
        rstd,
        dbranch,
        dresidual,
        N,
        has_residual_out_grad,
        block_n,
        num_warps=row_warps,
    )

    # dweight is a cross-row reduction, so it is kept as partial-reduce kernels.
    # The row-local forward + dx stages are fused above in kernel_function.
    block_dw_m = 32
    block_dw_n = 128
    num_blocks_m = triton.cdiv(M, block_dw_m)
    block_b = _next_power_of_2(num_blocks_m)

    partial = torch.empty((num_blocks_m, N), device=x_save.device, dtype=torch.float32)

    _dweight_partial_kernel[(triton.cdiv(N, block_dw_n), num_blocks_m)](
        dy,
        x_save,
        rstd,
        partial,
        M,
        N,
        0,
        block_dw_m,
        block_dw_n,
        num_warps=4,
    )

    _dweight_reduce_kernel[(triton.cdiv(N, block_dw_n),)](
        partial,
        dweight,
        N,
        0,
        num_blocks_m,
        block_b,
        block_dw_n,
        num_warps=4,
    )

    return dbranch, dresidual, dweight


def kernel_function(branch, residual, weight, eps, *upstream):
    N = weight.numel()
    P = branch.numel()
    M = P // N

    block_n = _next_power_of_2(N)
    row_warps = 8 if block_n >= 2048 else 4

    total = 4 * P + N
    packed = torch.empty((total,), device=branch.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=branch.device, dtype=torch.float32)

    dy = upstream[0]
    if len(upstream) > 1:
        dresidual_out = upstream[1]
        has_residual_out_grad = True
    else:
        dresidual_out = dy
        has_residual_out_grad = False

    _fwd_bwd_dx_pack_kernel[(M,)](
        branch,
        residual,
        weight,
        dy,
        dresidual_out,
        packed,
        rstd,
        N,
        P,
        eps,
        has_residual_out_grad,
        block_n,
        num_warps=row_warps,
    )

    block_dw_m = 32
    block_dw_n = 128
    num_blocks_m = triton.cdiv(M, block_dw_m)
    block_b = _next_power_of_2(num_blocks_m)

    partial = torch.empty((num_blocks_m, N), device=branch.device, dtype=torch.float32)

    _dweight_partial_kernel[(triton.cdiv(N, block_dw_n), num_blocks_m)](
        dy,
        packed,
        rstd,
        partial,
        M,
        N,
        P,
        block_dw_m,
        block_dw_n,
        num_warps=4,
    )

    _dweight_reduce_kernel[(triton.cdiv(N, block_dw_n),)](
        partial,
        packed,
        N,
        4 * P,
        num_blocks_m,
        block_b,
        block_dw_n,
        num_warps=4,
    )

    return packed
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'olmoe_residual_rmsnorm', 'arguments': ['branch', 'residual', 'weight', 'eps'], 'outputs': ['normed', 'summed'], 'upstream_grads': ['dnormed', 'dsummed'], 'input_grads': ['dbranch', 'dresidual', 'dweight'], 'gradient_return_order': ['dbranch', 'dresidual', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'olmoe_residual_rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class OlmoeResidualRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `olmoe_residual_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, branch, residual, weight, eps):
        (normed, summed), saved = olmoe_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps)
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
        dbranch, dresidual, dweight = olmoe_residual_rmsnorm_backward_from_saved(
            (dnormed, dsummed), saved, **_backward_scalars(olmoe_residual_rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dbranch, dresidual, dweight, None


def olmoe_residual_rmsnorm_deployment(branch, residual, weight, eps):
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
    normed, summed = OlmoeResidualRmsnormFunction.apply(branch, residual, weight, eps)
    if _leading is not None:
        normed = normed.view(*_leading, normed.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return normed, summed


class OlmoeResidualRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-05):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, branch, residual):
        return olmoe_residual_rmsnorm_deployment(branch, residual, self.weight, self.eps)


DEPLOYMENT_ENTRY = "olmoe_residual_rmsnorm_deployment"
