"""adbench-tools artifact: task qwen3_5_4b.l2norm.gdn_qk_l2norm, produced by kernelagent.

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
def _load_eps(eps, EPS_IS_TENSOR: tl.constexpr):
    if EPS_IS_TENSOR:
        return tl.load(eps).to(tl.float32)
    else:
        return eps


@triton.jit
def _l2norm_forward_kernel(
    x_ptr,
    y_ptr,
    inv_ptr,
    eps,
    n_rows: tl.constexpr,
    D: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward L2 normalization over the last dimension.

    y[row, :] = x[row, :] / (sqrt(sum(x^2)) + eps)

    Also saves inv = 1 / (sqrt(sum(x^2)) + eps) per row for backward.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    base = row * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)
    eps_val = _load_eps(eps, EPS_IS_TENSOR)
    norm = tl.sqrt(ss)
    inv = 1.0 / (norm + eps_val)

    y = x * inv

    tl.store(y_ptr + base, y, mask=mask)
    tl.store(inv_ptr + row, inv)


@triton.jit
def _l2norm_backward_kernel(
    x_ptr,
    dy_ptr,
    inv_ptr,
    dx_ptr,
    eps,
    n_rows: tl.constexpr,
    D: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for y = x / (||x||_2 + eps), normalized over last dim."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    base = row * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)

    inv = tl.load(inv_ptr + row).to(tl.float32)
    eps_val = _load_eps(eps, EPS_IS_TENSOR)

    dot_gx = tl.sum(dy * x, axis=0)

    # inv = 1 / (norm + eps) -> norm = 1 / inv - eps.
    norm = (1.0 / inv) - eps_val

    # d/dx x/(norm+eps):
    # dx = dy * inv - x * sum(dy*x) / (norm * (norm+eps)^2)
    # Guard norm == 0 to match the usual PyTorch subgradient behavior at zero.
    coeff = tl.where(norm > 0.0, inv * inv / norm, 0.0)
    dx = dy * inv - x * dot_gx * coeff

    tl.store(dx_ptr + base, dx, mask=mask)


@triton.jit
def _l2norm_fwd_bwd_concat_kernel(
    x_ptr,
    dy_ptr,
    out_ptr,
    eps,
    total_elements: tl.constexpr,
    n_rows: tl.constexpr,
    D: tl.constexpr,
    EPS_IS_TENSOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused forward + backward + packing kernel.

    Computes:
      1. y  = x / (||x||_2 + eps)
      2. dx = backward(y, dy)
      3. stores flattened [y, dx] into one output tensor

    This is the fused path used by kernel_function to avoid separate forward,
    backward, and concat kernels.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    base = row * D + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(x * x, axis=0)
    eps_val = _load_eps(eps, EPS_IS_TENSOR)
    norm = tl.sqrt(ss)
    inv = 1.0 / (norm + eps_val)

    y = x * inv

    dot_gx = tl.sum(dy * x, axis=0)
    coeff = tl.where(norm > 0.0, inv * inv / norm, 0.0)
    dx = dy * inv - x * dot_gx * coeff

    tl.store(out_ptr + base, y, mask=mask)
    tl.store(out_ptr + total_elements + base, dx, mask=mask)


def _launch_params(x):
    D = x.shape[-1]
    n_elements = x.numel()
    assert D > 0
    assert n_elements % D == 0
    n_rows = n_elements // D
    block = triton.next_power_of_2(D)

    if block <= 64:
        num_warps = 1
    elif block <= 256:
        num_warps = 4
    else:
        num_warps = 8

    return D, n_rows, block, num_warps


def l2norm_forward_with_saved(x, eps):
    """Forward helper used by the evaluator.

    All numerical work is performed by Triton. The normalization is applied
    row-wise over the final dimension of `x`.
    """
    assert x.is_cuda
    assert x.is_contiguous()

    D, n_rows, block, num_warps = _launch_params(x)
    eps_is_tensor = isinstance(eps, torch.Tensor)

    y = torch.empty_like(x)
    inv = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    grid = (n_rows,)
    _l2norm_forward_kernel[grid](
        x,
        y,
        inv,
        eps,
        n_rows,
        D,
        eps_is_tensor,
        block,
        num_warps=num_warps,
    )

    saved = (x, inv, eps, eps_is_tensor, D, n_rows, block, num_warps)
    return y, saved


def l2norm_backward_from_saved(upstream, saved):
    """Backward helper used by the evaluator.

    Returns the gradient with respect to `x`. `eps` is treated as a numerical
    stability constant, not a differentiable tensor.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, inv, eps, eps_is_tensor, D, n_rows, block, num_warps = saved

    assert dy.is_cuda
    assert dy.is_contiguous()

    dx = torch.empty_like(x)

    grid = (n_rows,)
    _l2norm_backward_kernel[grid](
        x,
        dy,
        inv,
        dx,
        eps,
        n_rows,
        D,
        eps_is_tensor,
        block,
        num_warps=num_warps,
    )

    return dx


def kernel_function(x, eps, *upstreams):
    """Fused forward+backward step for the l2norm task.

    This wrapper launches a single Triton kernel that fuses:
      - forward L2 normalization over the last dimension,
      - backward gradient computation with respect to `x`,
      - flattening/packing of `[forward_output, input_gradient]`.

    The wrapper performs only validation, allocation, and launch setup.
    """
    assert x.is_cuda
    assert x.is_contiguous()
    assert len(upstreams) == 1

    dy = upstreams[0]
    if isinstance(dy, (tuple, list)):
        dy = dy[0]

    assert dy.is_cuda
    assert dy.is_contiguous()
    assert dy.numel() == x.numel()

    D, n_rows, block, num_warps = _launch_params(x)
    n_elements = x.numel()
    eps_is_tensor = isinstance(eps, torch.Tensor)

    out = torch.empty((2 * n_elements,), device=x.device, dtype=x.dtype)

    grid = (n_rows,)
    _l2norm_fwd_bwd_concat_kernel[grid](
        x,
        dy,
        out,
        eps,
        n_elements,
        n_rows,
        D,
        eps_is_tensor,
        block,
        num_warps=num_warps,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'l2norm', 'arguments': ['x', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx'], 'gradient_return_order': ['dx', 'None'], 'scalars': ['eps'], 'deployment_entry': 'l2norm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class L2NormFunction(torch.autograd.Function):
    """Static autograd wiring for `l2norm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, eps):
        y, saved = l2norm_forward_with_saved(x, eps)
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
        dx = l2norm_backward_from_saved(
            dy, saved, **_backward_scalars(l2norm_backward_from_saved, eps=ctx.eps)
        )
        if isinstance(dx, (tuple, list)):
            dx, = dx
        # One value per forward argument, `None` for every inactive one.
        return dx, None


def l2norm_deployment(x, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = L2NormFunction.apply(x, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class L2NormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, eps=1e-06):
        super().__init__()
        pass
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return l2norm_deployment(x, self.eps)


DEPLOYMENT_ENTRY = "l2norm_deployment"
