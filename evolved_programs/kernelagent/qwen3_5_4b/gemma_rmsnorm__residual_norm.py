"""adbench-tools artifact: task qwen3_5_4b.gemma_rmsnorm.residual_norm, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for Gemma RMSNorm plus its backward pass.
#
# Fused stages:
# - `kernel_function` performs the forward RMSNorm output and input-gradient
#   computation in one Triton row kernel, sharing the row reduction work where
#   possible, then computes the weight-gradient with Triton reduction kernels.
# - The wrapper only allocates tensors and launches Triton kernels; all numeric
#   computation is performed inside Triton.

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    inv_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward Gemma RMSNorm.

    y = x * rsqrt(mean(x^2) + eps) * (1 + weight)
    Reduction is over the last dimension.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(mean_sq + eps)

    y = x * inv * (w + 1.0)

    tl.store(y_ptr + row * N + offs, y, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _gemma_rmsnorm_grad_x_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    inv_ptr,
    dx_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute gradient w.r.t. x for Gemma RMSNorm."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)

    scale = w + 1.0
    dot = tl.sum(dy * scale * x, axis=0)

    # dx_i = dy_i * scale_i * inv - x_i * inv^3 / N * sum_j(dy_j * scale_j * x_j)
    inv3 = inv * inv * inv
    dx = dy * scale * inv - x * (inv3 * dot / N)

    tl.store(dx_ptr + row * N + offs, dx, mask=mask)


@triton.jit
def _gemma_rmsnorm_partial_dw_kernel(
    x_ptr,
    dy_ptr,
    inv_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    NUM_RB: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Partial reduction for gradient w.r.t. weight.

    dweight[n] = sum_m dy[m, n] * x[m, n] * inv[m]
    """
    pid_n = tl.program_id(0)
    pid_rb = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_rb * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_n = offs_n < N
    mask_m = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    dy = tl.load(
        dy_ptr + offs_m[:, None] * N + offs_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    inv = tl.load(inv_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    acc = tl.sum(dy * x * inv[:, None], axis=0)

    tl.store(partial_ptr + pid_rb * N + offs_n, acc, mask=mask_n)


@triton.jit
def _gemma_rmsnorm_reduce_dw_kernel(
    partial_ptr,
    dw_ptr,
    N: tl.constexpr,
    NUM_RB: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Final reduction for dweight."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for rb in tl.range(0, NUM_RB):
        vals = tl.load(partial_ptr + rb * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += vals

    tl.store(dw_ptr + offs_n, acc, mask=mask_n)


@triton.jit
def _gemma_rmsnorm_pair_row_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    packed_ptr,
    inv_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    eps: tl.constexpr,
    X_NUMEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused forward output + dx computation for `kernel_function`.

    Stores:
      packed[0:X_NUMEL]                 = forward output flattened as fp32
      packed[X_NUMEL:2*X_NUMEL]         = dx flattened as fp32
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    base = row * N + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N
    inv = tl.rsqrt(mean_sq + eps)
    scale = w + 1.0

    y = x * inv * scale

    dot = tl.sum(dy * scale * x, axis=0)
    inv3 = inv * inv * inv
    dx = dy * scale * inv - x * (inv3 * dot / N)

    tl.store(packed_ptr + base, y, mask=mask)
    tl.store(packed_ptr + X_NUMEL + base, dx, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _gemma_rmsnorm_reduce_dw_to_packed_kernel(
    partial_ptr,
    packed_ptr,
    N: tl.constexpr,
    NUM_RB: tl.constexpr,
    DW_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Final dweight reduction directly into packed output."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for rb in tl.range(0, NUM_RB):
        vals = tl.load(partial_ptr + rb * N + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += vals

    tl.store(packed_ptr + DW_OFFSET + offs_n, acc, mask=mask_n)


def _next_power_of_2(v: int) -> int:
    return 1 << (v - 1).bit_length()


def _num_warps(block_n: int) -> int:
    if block_n >= 4096:
        return 8
    if block_n >= 2048:
        return 4
    return 4


def gemma_rmsnorm_forward_with_saved(x, weight, eps):
    """Forward Gemma RMSNorm.

    Returns `(output, saved)` where `saved` is consumed by
    `gemma_rmsnorm_backward_from_saved`.

    Wrapper responsibilities are limited to shape checks, allocation, and launch.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()

    N = weight.numel()
    assert x.numel() % N == 0
    M = x.numel() // N

    y = torch.empty_like(x)
    inv = torch.empty((M,), device=x.device, dtype=torch.float32)

    block_n = _next_power_of_2(N)
    warps = _num_warps(block_n)

    _gemma_rmsnorm_forward_kernel[(M,)](
        x,
        weight,
        y,
        inv,
        M,
        N,
        eps,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    return y, (x, weight, inv)


def gemma_rmsnorm_backward_from_saved(upstream, saved):
    """Backward for Gemma RMSNorm.

    Computes gradients with respect to x and weight. `eps` is non-tensor metadata
    and has no returned tensor gradient.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, weight, inv = saved

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert dy.is_contiguous()

    N = weight.numel()
    assert x.numel() % N == 0
    M = x.numel() // N

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)

    block_n = _next_power_of_2(N)
    warps = _num_warps(block_n)

    _gemma_rmsnorm_grad_x_kernel[(M,)](
        x,
        weight,
        dy,
        inv,
        dx,
        M,
        N,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    block_m = 8
    num_rb = triton.cdiv(M, block_m)
    partial = torch.empty((num_rb, N), device=x.device, dtype=torch.float32)

    grid_partial = (1, num_rb)
    _gemma_rmsnorm_partial_dw_kernel[grid_partial](
        x,
        dy,
        inv,
        partial,
        M,
        N,
        num_rb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    _gemma_rmsnorm_reduce_dw_kernel[(1,)](
        partial,
        dw,
        N,
        num_rb,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    return dx, dw


def kernel_function(x, weight, eps, upstream):
    """Forward + backward fused step for Gemma RMSNorm.

    This returns a single flat fp32 tensor matching:

        torch.cat([
            forward_output.float().reshape(-1),
            grad_x.float().reshape(-1),
            grad_weight.float().reshape(-1),
        ])

    Fusion note:
    - The first Triton kernel fuses forward RMSNorm output computation and the
      row-local dx backward computation. Both need the same row RMS reduction, so
      the inverse RMS is computed once and reused.
    - dweight is a cross-row reduction, so it is implemented with Triton partial
      and final reduction kernels. No PyTorch math or concatenation is used.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert dy.is_contiguous()

    N = weight.numel()
    assert x.numel() % N == 0
    M = x.numel() // N

    x_numel = x.numel()
    total = x_numel + x_numel + N

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)
    inv = torch.empty((M,), device=x.device, dtype=torch.float32)

    block_n = _next_power_of_2(N)
    warps = _num_warps(block_n)

    _gemma_rmsnorm_pair_row_kernel[(M,)](
        x,
        weight,
        dy,
        packed,
        inv,
        M,
        N,
        eps,
        x_numel,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    block_m = 8
    num_rb = triton.cdiv(M, block_m)
    partial = torch.empty((num_rb, N), device=x.device, dtype=torch.float32)

    _gemma_rmsnorm_partial_dw_kernel[(1, num_rb)](
        x,
        dy,
        inv,
        partial,
        M,
        N,
        num_rb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
    )

    _gemma_rmsnorm_reduce_dw_to_packed_kernel[(1,)](
        partial,
        packed,
        N,
        num_rb,
        DW_OFFSET=2 * x_numel,
        BLOCK_N=block_n,
        num_warps=warps,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gemma_rmsnorm', 'arguments': ['x', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight'], 'gradient_return_order': ['dx', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'gemma_rmsnorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class GemmaRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `gemma_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        y, saved = gemma_rmsnorm_forward_with_saved(x, weight, eps)
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
        dx, dweight = gemma_rmsnorm_backward_from_saved(
            dy, saved, **_backward_scalars(gemma_rmsnorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, None


def gemma_rmsnorm_deployment(x, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = GemmaRmsnormFunction.apply(x, weight, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class GemmaRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return gemma_rmsnorm_deployment(x, self.weight, self.eps)


DEPLOYMENT_ENTRY = "gemma_rmsnorm_deployment"
