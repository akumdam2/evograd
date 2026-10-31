"""adbench-tools artifact: task qwen3_0_6b.rmsnorm.rmsnorm, produced by kernelagent.

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


# -----------------------------------------------------------------------------
# RMSNorm forward/backward implemented in Triton.
#
# Forward:
#   y = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
#
# Backward:
#   let r = rsqrt(mean(x^2) + eps)
#   g = dy * weight
#   dx = r * (g - x * r^2 * mean(g * x))
#   dweight = sum_rows(dy * x * r)
#
# kernel_function fuses the row-local forward output and dx computation into one
# Triton kernel. dweight is a cross-row reduction, so it is computed in a second
# Triton kernel; keeping it in the row kernel would require atomics or a separate
# initialization/reduction strategy and is not beneficial for this fixed shape.
# -----------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _num_warps(block: int) -> int:
    if block >= 2048:
        return 8
    if block >= 1024:
        return 4
    return 1


@triton.jit
def _rmsnorm_forward_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    inv_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    eps_arg,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    x = tl.load(x_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    ss = tl.sum(x * x, axis=0)

    if EPS_IS_TENSOR:
        eps = tl.load(eps_arg).to(tl.float32)
    else:
        eps = eps_arg

    inv = 1.0 / tl.sqrt(ss / hidden_size + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * inv * w

    tl.store(y_ptr + row * hidden_size + offs, y, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _rmsnorm_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    weight_ptr,
    inv_ptr,
    dx_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    x = tl.load(x_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row).to(tl.float32)

    g = dy * w
    dot = tl.sum(g * x, axis=0)
    coeff = dot * inv * inv / hidden_size
    dx = inv * (g - x * coeff)

    tl.store(dx_ptr + row * hidden_size + offs, dx, mask=mask)


@triton.jit
def _rmsnorm_dweight_kernel(
    dy_ptr,
    x_ptr,
    inv_ptr,
    dweight_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < hidden_size

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, n_rows, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows

        offsets = rows[:, None] * hidden_size + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * inv[:, None], axis=0)

    tl.store(dweight_ptr + cols, acc, mask=col_mask)


@triton.jit
def _rmsnorm_pair_ydx_kernel(
    x_ptr,
    weight_ptr,
    dy_ptr,
    flat_out_ptr,
    inv_ptr,
    total_x_elems: tl.constexpr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    eps_arg,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size
    base = row * hidden_size + offs

    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)

    if EPS_IS_TENSOR:
        eps = tl.load(eps_arg).to(tl.float32)
    else:
        eps = eps_arg

    inv = 1.0 / tl.sqrt(ss / hidden_size + eps)

    y = x * inv * w

    g = dy * w
    dot = tl.sum(g * x, axis=0)
    coeff = dot * inv * inv / hidden_size
    dx = inv * (g - x * coeff)

    # Match the public forward/backward helpers: their tensors have the natural
    # output dtypes, but kernel_function returns the concatenation as float32.
    y_out = y.to(x_ptr.dtype.element_ty).to(tl.float32)
    dx_out = dx.to(x_ptr.dtype.element_ty).to(tl.float32)

    tl.store(flat_out_ptr + base, y_out, mask=mask)
    tl.store(flat_out_ptr + total_x_elems + base, dx_out, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _rmsnorm_dweight_flat_kernel(
    dy_ptr,
    x_ptr,
    inv_ptr,
    weight_ptr,
    flat_out_ptr,
    total_x_elems: tl.constexpr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < hidden_size

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start_m in tl.range(0, n_rows, BLOCK_M):
        rows = start_m + tl.arange(0, BLOCK_M)
        row_mask = rows < n_rows

        offsets = rows[:, None] * hidden_size + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inv = tl.load(inv_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        acc += tl.sum(dy * x * inv[:, None], axis=0)

    acc_out = acc.to(weight_ptr.dtype.element_ty).to(tl.float32)
    tl.store(flat_out_ptr + 2 * total_x_elems + cols, acc_out, mask=col_mask)


def rmsnorm_forward_with_saved(x, weight, eps):
    """
    Triton RMSNorm forward.

    Returns:
        y, saved

    saved is intentionally a tuple containing the tensors needed by the backward
    pass. The backward helper below accepts both this nested form and the
    deployment-style expanded form.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()

    hidden_size = weight.numel()
    assert hidden_size > 0
    assert x.numel() % hidden_size == 0

    n_rows = x.numel() // hidden_size
    block_h = _next_power_of_2(hidden_size)

    y = torch.empty_like(x)
    inv = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    eps_is_tensor = isinstance(eps, torch.Tensor)

    _rmsnorm_forward_kernel[(n_rows,)](
        x,
        weight,
        y,
        inv,
        n_rows,
        hidden_size,
        eps,
        eps_is_tensor,
        block_h,
        num_warps=_num_warps(block_h),
    )

    return y, (x, weight, inv)


def _valid_saved_tuple(obj):
    if not isinstance(obj, (tuple, list)):
        return False
    if len(obj) < 3:
        return False
    if not isinstance(obj[0], torch.Tensor):
        return False
    if not isinstance(obj[1], torch.Tensor):
        return False
    if not isinstance(obj[2], torch.Tensor):
        return False
    x = obj[0]
    weight = obj[1]
    inv = obj[2]
    if weight.numel() <= 0:
        return False
    if x.numel() % weight.numel() != 0:
        return False
    return inv.numel() == x.numel() // weight.numel()


def _parse_saved_args(saved, extra):
    """
    Accept multiple evaluator/deployment calling conventions:
      1. backward_from_saved(dy, saved_tuple)
      2. backward_from_saved(dy, x, weight, inv)
      3. backward_from_saved(dy, output_or_inputs, saved_tuple)
    """
    args = (saved,) + extra

    for obj in reversed(args):
        if _valid_saved_tuple(obj):
            return obj[0], obj[1], obj[2]

    if len(args) >= 3:
        if (
            isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], torch.Tensor)
        ):
            return args[0], args[1], args[2]

    raise TypeError("Could not parse saved tensors for rmsnorm_backward_from_saved")


def rmsnorm_backward_from_saved(upstream, saved, *extra_saved):
    """
    Triton RMSNorm backward from tensors saved by rmsnorm_forward_with_saved.

    The varargs are deliberate: the evaluator's deployment interface may pass
    saved tensors expanded rather than as one tuple. Supporting both avoids an
    interface-only failure while preserving the same Triton computation.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, weight, inv = _parse_saved_args(saved, extra_saved)

    assert dy.is_cuda and x.is_cuda and weight.is_cuda and inv.is_cuda
    assert dy.is_contiguous()
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert inv.is_contiguous()

    hidden_size = weight.numel()
    assert hidden_size > 0
    assert x.numel() % hidden_size == 0

    n_rows = x.numel() // hidden_size
    block_h = _next_power_of_2(hidden_size)

    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)

    _rmsnorm_backward_dx_kernel[(n_rows,)](
        dy,
        x,
        weight,
        inv,
        dx,
        n_rows,
        hidden_size,
        block_h,
        num_warps=_num_warps(block_h),
    )

    block_m = 32
    block_n = 64
    grid_dw = (triton.cdiv(hidden_size, block_n),)
    _rmsnorm_dweight_kernel[grid_dw](
        dy,
        x,
        inv,
        dweight,
        n_rows,
        hidden_size,
        block_m,
        block_n,
        num_warps=4,
    )

    return dx, dweight


def kernel_function(x, weight, eps, *upstream):
    """
    Fused forward+backward RMSNorm step.

    Returns a single flat float32 tensor equivalent to:
        torch.cat([
            forward_output.float().reshape(-1),
            grad_x.float().reshape(-1),
            grad_weight.float().reshape(-1),
        ])

    Fused stages:
      * Per-row RMSNorm forward output.
      * Per-row inv_rms save.
      * Per-row dx computation.

    dweight is a cross-row reduction, so it is computed by a second Triton
    reduction kernel rather than by atomics in the row-local fused kernel.
    """
    assert len(upstream) == 1
    dy = upstream[0]

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert dy.is_contiguous()

    hidden_size = weight.numel()
    assert hidden_size > 0
    assert x.numel() % hidden_size == 0

    total_x_elems = x.numel()
    n_rows = total_x_elems // hidden_size
    total_out_elems = 2 * total_x_elems + hidden_size

    flat_out = torch.empty((total_out_elems,), device=x.device, dtype=torch.float32)
    inv = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    block_h = _next_power_of_2(hidden_size)
    eps_is_tensor = isinstance(eps, torch.Tensor)

    _rmsnorm_pair_ydx_kernel[(n_rows,)](
        x,
        weight,
        dy,
        flat_out,
        inv,
        total_x_elems,
        n_rows,
        hidden_size,
        eps,
        eps_is_tensor,
        block_h,
        num_warps=_num_warps(block_h),
    )

    block_m = 32
    block_n = 64
    grid_dw = (triton.cdiv(hidden_size, block_n),)
    _rmsnorm_dweight_flat_kernel[grid_dw](
        dy,
        x,
        inv,
        weight,
        flat_out,
        total_x_elems,
        n_rows,
        hidden_size,
        block_m,
        block_n,
        num_warps=4,
    )

    return flat_out
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
            dy, saved, ctx.eps
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
