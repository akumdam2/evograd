"""adbench-tools artifact: task qwen2_vl_2b.rmsnorm.model.language_model.layers.input_layernorm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation of RMSNorm forward and backward.
#
# Fused path used by `kernel_function`:
#   1. Compute RMSNorm forward output.
#   2. Compute row-local backward terms and dx.
#   3. Compute partial reductions for dweight.
#   4. Reduce partial dweight.
#   5. Pack [forward_output, dx, dweight] into one flat float32 tensor.
#
# The wrappers only allocate tensors, validate metadata, and launch Triton kernels.
# All numerical computation is performed inside Triton kernels.

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    eps_arg,
    y_ptr,
    inv_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)

    if EPS_IS_TENSOR:
        eps_val = tl.load(eps_arg).to(tl.float32)
    else:
        eps_val = eps_arg

    inv = tl.rsqrt(ss / N + eps_val)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * inv * w

    tl.store(y_ptr + pid * N + cols, y, mask=mask)
    tl.store(inv_ptr + pid, inv)


@triton.jit
def _rmsnorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    inv_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + pid).to(tl.float32)

    # s = sum_j dy_j * weight_j * x_j
    s = tl.sum(dy * w * x, axis=0)

    # dx_i = inv * dy_i * w_i - x_i * inv^3 / N * s
    dx = inv * dy * w - x * (inv * inv * inv) * s / N
    tl.store(dx_ptr + pid * N + cols, dx, mask=mask)


@triton.jit
def _rmsnorm_dweight_partial_kernel(
    dy_ptr,
    x_ptr,
    inv_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rb = tl.program_id(0)

    rows = rb * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + rows, mask=rows < M, other=0.0).to(tl.float32)

    # dweight_col partial = sum_rows dy * x * inv
    part = tl.sum(dy * x * inv[:, None], axis=0)
    tl.store(partial_ptr + rb * N + cols, part, mask=cols < N)


@triton.jit
def _rmsnorm_dweight_reduce_kernel(
    partial_ptr,
    dweight_ptr,
    ROW_BLOCKS: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for rb in range(ROW_BLOCKS):
        acc += tl.load(partial_ptr + rb * N + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(dweight_ptr + cols, acc, mask=mask)


@triton.jit
def _rmsnorm_fused_fwd_bwd_partial_kernel(
    x_ptr,
    weight_ptr,
    eps_arg,
    dy_ptr,
    y_ptr,
    dx_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rb = tl.program_id(0)

    rows = rb * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offs = rows[:, None] * N + cols[None, :]

    if EPS_IS_TENSOR:
        eps_val = tl.load(eps_arg).to(tl.float32)
    else:
        eps_val = eps_arg

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=1)
    inv = tl.rsqrt(ss / N + eps_val)

    # Forward output.
    y = x * inv[:, None] * w[None, :]
    tl.store(y_ptr + offs, y, mask=mask)

    # Backward dx.
    s = tl.sum(dy * w[None, :] * x, axis=1)
    dx = inv[:, None] * dy * w[None, :] - x * (inv * inv * inv)[:, None] * s[:, None] / N
    tl.store(dx_ptr + offs, dx, mask=mask)

    # Partial dweight reduction over this row block.
    part = tl.sum(dy * x * inv[:, None], axis=0)
    tl.store(partial_ptr + rb * N + cols, part, mask=cols < N)


@triton.jit
def _pack_kernel(
    y_ptr,
    dx_ptr,
    dweight_ptr,
    out_ptr,
    y_numel: tl.constexpr,
    dx_numel: tl.constexpr,
    dw_numel: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    vals = tl.zeros((BLOCK,), dtype=tl.float32)

    y_mask = offs < y_numel
    dx_mask = (offs >= y_numel) & (offs < y_numel + dx_numel)
    dw_mask = offs >= y_numel + dx_numel

    y_vals = tl.load(y_ptr + offs, mask=mask & y_mask, other=0.0).to(tl.float32)
    dx_vals = tl.load(dx_ptr + (offs - y_numel), mask=mask & dx_mask, other=0.0).to(tl.float32)
    dw_vals = tl.load(
        dweight_ptr + (offs - y_numel - dx_numel),
        mask=mask & dw_mask,
        other=0.0,
    ).to(tl.float32)

    vals = tl.where(y_mask, y_vals, vals)
    vals = tl.where(dx_mask, dx_vals, vals)
    vals = tl.where(dw_mask, dw_vals, vals)

    tl.store(out_ptr + offs, vals, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _num_warps(block_n: int) -> int:
    if block_n >= 2048:
        return 8
    if block_n >= 1024:
        return 4
    return 4


def _out_dtype(x, weight):
    # Metadata-only dtype promotion.
    return torch.promote_types(x.dtype, weight.dtype)


def rmsnorm_forward_with_saved(x, weight, eps):
    """
    RMSNorm forward.

    Computes:
        inv = rsqrt(mean(x^2, dim=-1) + eps)
        y = x * inv * weight

    Returns:
        y, saved

    saved contains x, weight, inv, and eps metadata needed by backward.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.numel() % weight.numel() == 0

    N = weight.numel()
    M = x.numel() // N
    BLOCK_N = _next_power_of_2(N)

    y = torch.empty(x.shape, device=x.device, dtype=_out_dtype(x, weight))
    inv = torch.empty((M,), device=x.device, dtype=torch.float32)

    eps_is_tensor = isinstance(eps, torch.Tensor)
    grid = (M,)

    _rmsnorm_forward_kernel[grid](
        x,
        weight,
        eps,
        y,
        inv,
        M,
        N,
        eps_is_tensor,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    saved = (x, weight, inv, eps)
    return y, saved


def rmsnorm_backward_from_saved(upstream, saved):
    """
    RMSNorm backward from saved forward tensors.

    Given upstream gradient dy, returns:
        dx, dweight
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, weight, inv, _eps = saved

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.numel() == dy.numel()
    assert x.numel() % weight.numel() == 0

    N = weight.numel()
    M = x.numel() // N
    BLOCK_N = _next_power_of_2(N)
    BLOCK_M = 8 if BLOCK_N >= 1024 else 16
    ROW_BLOCKS = triton.cdiv(M, BLOCK_M)

    dx = torch.empty_like(x)
    partial = torch.empty((ROW_BLOCKS, N), device=x.device, dtype=torch.float32)
    dweight = torch.empty_like(weight)

    _rmsnorm_backward_dx_kernel[(M,)](
        dy,
        x,
        weight,
        inv,
        dx,
        M,
        N,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    _rmsnorm_dweight_partial_kernel[(ROW_BLOCKS,)](
        dy,
        x,
        inv,
        partial,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    _rmsnorm_dweight_reduce_kernel[(1,)](
        partial,
        dweight,
        ROW_BLOCKS,
        N,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    return dx, dweight


def kernel_function(x, weight, eps, upstream):
    """
    Complete forward+backward RMSNorm step.

    This function returns a single flat float32 tensor containing:
        concat([rmsnorm_output.flatten(), dx.flatten(), dweight.flatten()])

    Fusion note:
      The main Triton kernel fuses RMSNorm forward, row-local backward dx,
      and partial dweight accumulation in one pass over x/upstream/weight.
      A second reduction kernel finalizes dweight, and a final lightweight
      Triton kernel packs the outputs into the format required by the test.
    """
    assert x.is_cuda and weight.is_cuda and upstream.is_cuda
    assert x.numel() == upstream.numel()
    assert x.numel() % weight.numel() == 0

    N = weight.numel()
    M = x.numel() // N
    BLOCK_N = _next_power_of_2(N)
    BLOCK_M = 8 if BLOCK_N >= 1024 else 16
    ROW_BLOCKS = triton.cdiv(M, BLOCK_M)

    y = torch.empty(x.shape, device=x.device, dtype=_out_dtype(x, weight))
    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    partial = torch.empty((ROW_BLOCKS, N), device=x.device, dtype=torch.float32)

    eps_is_tensor = isinstance(eps, torch.Tensor)

    _rmsnorm_fused_fwd_bwd_partial_kernel[(ROW_BLOCKS,)](
        x,
        weight,
        eps,
        upstream,
        y,
        dx,
        partial,
        M,
        N,
        eps_is_tensor,
        BLOCK_M,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    _rmsnorm_dweight_reduce_kernel[(1,)](
        partial,
        dweight,
        ROW_BLOCKS,
        N,
        BLOCK_N,
        num_warps=_num_warps(BLOCK_N),
    )

    y_numel = y.numel()
    dx_numel = dx.numel()
    dw_numel = dweight.numel()
    total = y_numel + dx_numel + dw_numel

    out = torch.empty((total,), device=x.device, dtype=torch.float32)
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)

    _pack_kernel[grid](
        y,
        dx,
        dweight,
        out,
        y_numel,
        dx_numel,
        dw_numel,
        total,
        BLOCK,
        num_warps=4,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'rmsnorm', 'arguments': ['x', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight'], 'gradient_return_order': ['dx', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class RmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        y, saved = rmsnorm_forward_with_saved(x, weight, eps)
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
        dx, dweight = rmsnorm_backward_from_saved(
            dy, saved, **_backward_scalars(rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, None


def rmsnorm_deployment(x, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = RmsnormFunction.apply(x, weight, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class RmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return rmsnorm_deployment(x, self.weight, self.eps)


DEPLOYMENT_ENTRY = "rmsnorm_deployment"
