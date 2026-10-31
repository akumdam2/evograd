"""adbench-tools artifact: task qwen2_vl_2b.quick_gelu.model.visual.blocks.mlp.act, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation of QuickGELU forward, backward, and the fused
# forward+backward "pair step" required by the evaluator.
#
# QuickGELU:
#     y = x * sigmoid(1.702 * x)
#
# Backward:
#     dy/dx = sigmoid(1.702x) + 1.702 * x * sigmoid(1.702x) * (1 - sigmoid(1.702x))
#     grad_x = grad_out * dy/dx
#
# Fusion note:
# - quick_gelu_forward_with_saved launches a forward-only Triton kernel.
# - quick_gelu_backward_from_saved launches a backward-only Triton kernel.
# - kernel_function fuses forward and backward in a single Triton pass and writes:
#       concat([forward_output.float().reshape(-1),
#               grad_x.float().reshape(-1)])
#   into one flat output tensor, avoiding intermediate global-memory writes.

import torch
import triton
import triton.language as tl


_QUICK_GELU_ALPHA = 1.702


@triton.jit
def _quick_gelu_forward_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-1.702 * x))
    y = x * sig

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _quick_gelu_backward_kernel(
    grad_out_ptr,
    x_ptr,
    grad_x_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-1.702 * x))
    deriv = sig + 1.702 * x * sig * (1.0 - sig)
    grad_x = grad_out * deriv

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


@triton.jit
def _quick_gelu_fused_forward_backward_concat_kernel(
    x_ptr,
    grad_out_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused QuickGELU forward + backward.

    Stores a flat float32 tensor:
        out[0:n]   = QuickGELU(x)
        out[n:2n]  = grad_out * d QuickGELU(x) / dx
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-1.702 * x))

    y = x * sig
    deriv = sig + 1.702 * x * sig * (1.0 - sig)
    grad_x = grad_out * deriv

    tl.store(out_ptr + offsets, y, mask=mask)
    tl.store(out_ptr + n_elements + offsets, grad_x, mask=mask)


def quick_gelu_forward_with_saved(x):
    """
    Forward pass for QuickGELU.

    The wrapper only validates, allocates, and launches the Triton kernel.
    The saved object contains x, which is sufficient for the analytic backward.
    """
    assert x.is_cuda, "Triton implementation requires CUDA tensors"
    assert x.is_contiguous(), "This implementation expects contiguous input"

    y = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements > 0:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _quick_gelu_forward_kernel[grid](
            x,
            y,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

    return y, (x,)


def quick_gelu_backward_from_saved(upstream, saved):
    """
    Backward pass for QuickGELU from saved forward tensors.

    Computes grad_x using Triton only:
        grad_x = upstream * d/dx[x * sigmoid(1.702x)]
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    x = saved[0]

    assert x.is_cuda and upstream.is_cuda, "Triton implementation requires CUDA tensors"
    assert x.is_contiguous(), "This implementation expects contiguous input"
    assert upstream.is_contiguous(), "This implementation expects contiguous upstream gradient"
    assert upstream.shape == x.shape, "upstream gradient must have the same shape as x"

    grad_x = torch.empty_like(x)
    n_elements = x.numel()

    if n_elements > 0:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _quick_gelu_backward_kernel[grid](
            upstream,
            x,
            grad_x,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

    return grad_x


def kernel_function(x, upstream):
    """
    Fused evaluator entry point.

    Given x and the upstream gradient, this performs both:
      1. QuickGELU forward:  y = x * sigmoid(1.702x)
      2. QuickGELU backward: grad_x = upstream * dy/dx

    in one Triton kernel and returns a single flat float32 tensor:
        concat([y.float().reshape(-1), grad_x.float().reshape(-1)])

    The Python wrapper performs no numerical tensor computation; it only checks
    inputs, allocates the output buffer, computes the launch grid, and launches
    the Triton kernel.
    """
    assert x.is_cuda and upstream.is_cuda, "Triton implementation requires CUDA tensors"
    assert x.is_contiguous(), "This implementation expects contiguous input"
    assert upstream.is_contiguous(), "This implementation expects contiguous upstream gradient"
    assert upstream.shape == x.shape, "upstream gradient must have the same shape as x"

    n_elements = x.numel()
    out = torch.empty((n_elements * 2,), device=x.device, dtype=torch.float32)

    if n_elements > 0:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _quick_gelu_fused_forward_backward_concat_kernel[grid](
            x,
            upstream,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'quick_gelu', 'arguments': ['x'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx'], 'gradient_return_order': ['dx'], 'scalars': [], 'deployment_entry': 'quick_gelu_deployment', 'allowed_primitives': []}


class QuickGeluFunction(torch.autograd.Function):
    """Static autograd wiring for `quick_gelu`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x):
        out, saved = quick_gelu_forward_with_saved(x)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return out

    @staticmethod
    def backward(ctx, dout):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dout = dout.contiguous() if (dout is not None and not dout.is_contiguous()) else dout
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx = quick_gelu_backward_from_saved(
            dout, saved
        )
        if isinstance(dx, (tuple, list)):
            dx, = dx
        # One value per forward argument, `None` for every inactive one.
        return dx,


def quick_gelu_deployment(x):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = QuickGeluFunction.apply(x)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class QuickGeluModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return quick_gelu_deployment(x)


DEPLOYMENT_ENTRY = "quick_gelu_deployment"
