"""adbench-tools artifact: task qwen3_5_4b.swiglu.mlp_activation, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_forward_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward SwiGLU activation:
        out = silu(a) * b
            = a * sigmoid(a) * b
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    out = a * sigmoid_a * b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_out_ptr,
    a_ptr,
    b_ptr,
    grad_a_ptr,
    grad_b_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward SwiGLU activation.

    Forward:
        y = silu(a) * b
        silu(a) = a * sigmoid(a)

    Gradients:
        dy/da = b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
        dy/db = silu(a)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sigmoid_a
    dsilu_da = sigmoid_a * (1.0 + a * (1.0 - sigmoid_a))

    grad_a = grad_out * b * dsilu_da
    grad_b = grad_out * silu_a

    tl.store(grad_a_ptr + offsets, grad_a, mask=mask)
    tl.store(grad_b_ptr + offsets, grad_b, mask=mask)


@triton.jit
def _swiglu_fused_forward_backward_flat_kernel(
    a_ptr,
    b_ptr,
    grad_out_ptr,
    flat_out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused forward + backward kernel for the evaluator's kernel_function.

    Computes in one pass:
      1. forward output: y = silu(a) * b
      2. grad_a
      3. grad_b

    The results are written into a single flat float32 tensor laid out as:
        [ y.flatten(), grad_a.flatten(), grad_b.flatten() ]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sigmoid_a

    y = silu_a * b
    dsilu_da = sigmoid_a * (1.0 + a * (1.0 - sigmoid_a))

    grad_a = grad_out * b * dsilu_da
    grad_b = grad_out * silu_a

    tl.store(flat_out_ptr + offsets, y, mask=mask)
    tl.store(flat_out_ptr + n_elements + offsets, grad_a, mask=mask)
    tl.store(flat_out_ptr + 2 * n_elements + offsets, grad_b, mask=mask)


def _launch_grid(n_elements, block_size):
    return (triton.cdiv(n_elements, block_size),)


def swiglu_forward_with_saved(a, b):
    """
    Forward wrapper for:
        out = silu(a) * b

    Saves the original inputs because the backward formula needs both `a`
    and `b`. No PyTorch math is performed here; all numerical work is done
    by the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.is_contiguous() and b.is_contiguous()

    out = torch.empty_like(a)
    n_elements = a.numel()

    if n_elements > 0:
        BLOCK_SIZE = 1024
        _swiglu_forward_kernel[_launch_grid(n_elements, BLOCK_SIZE)](
            a,
            b,
            out,
            n_elements,
            BLOCK_SIZE,
        )

    return out, (a, b)


def swiglu_backward_from_saved(upstream, saved):
    """
    Backward wrapper for SwiGLU.

    Args:
        upstream: gradient of the loss w.r.t. the SwiGLU output.
        saved: tuple `(a, b)` from `swiglu_forward_with_saved`.

    Returns:
        Tuple `(grad_a, grad_b)`.
    """
    if isinstance(upstream, (tuple, list)):
        grad_out = upstream[0]
    else:
        grad_out = upstream

    a, b = saved

    assert grad_out.is_cuda and a.is_cuda and b.is_cuda
    assert grad_out.shape == a.shape
    assert b.shape == a.shape
    assert grad_out.is_contiguous() and a.is_contiguous() and b.is_contiguous()

    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    n_elements = a.numel()

    if n_elements > 0:
        BLOCK_SIZE = 1024
        _swiglu_backward_kernel[_launch_grid(n_elements, BLOCK_SIZE)](
            grad_out,
            a,
            b,
            grad_a,
            grad_b,
            n_elements,
            BLOCK_SIZE,
        )

    return grad_a, grad_b


def kernel_function(a, b, *upstream):
    """
    Evaluator entry point.

    This function fuses the full forward+backward SwiGLU step into a single
    Triton kernel launch. It computes:
        output = silu(a) * b
        grad_a = upstream * b * d/da(silu(a))
        grad_b = upstream * silu(a)

    and returns one flat float32 tensor equivalent to:
        torch.cat([
            output.float().reshape(-1),
            grad_a.float().reshape(-1),
            grad_b.float().reshape(-1),
        ])

    The wrapper only validates inputs, allocates the output buffer, configures
    the Triton grid, and launches the fused kernel. All numerical computation
    is performed inside Triton.
    """
    grad_out = upstream[0]

    assert a.is_cuda and b.is_cuda and grad_out.is_cuda
    assert a.shape == b.shape
    assert a.shape == grad_out.shape
    assert a.is_contiguous() and b.is_contiguous() and grad_out.is_contiguous()

    n_elements = a.numel()
    flat_out = torch.empty((n_elements * 3,), device=a.device, dtype=torch.float32)

    if n_elements > 0:
        BLOCK_SIZE = 1024
        _swiglu_fused_forward_backward_flat_kernel[_launch_grid(n_elements, BLOCK_SIZE)](
            a,
            b,
            grad_out,
            flat_out,
            n_elements,
            BLOCK_SIZE,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'swiglu', 'arguments': ['a', 'b'], 'outputs': ['c'], 'upstream_grads': ['dc'], 'input_grads': ['da', 'db'], 'gradient_return_order': ['da', 'db'], 'scalars': [], 'deployment_entry': 'swiglu_deployment', 'allowed_primitives': []}


class SwigluFunction(torch.autograd.Function):
    """Static autograd wiring for `swiglu`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, a, b):
        c, saved = swiglu_forward_with_saved(a, b)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return c

    @staticmethod
    def backward(ctx, dc):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dc = dc.contiguous() if (dc is not None and not dc.is_contiguous()) else dc
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        da, db = swiglu_backward_from_saved(
            dc, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return da, db


def swiglu_deployment(a, b):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if a.dim() > 2:
        _leading = a.shape[:-1]
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    c = SwigluFunction.apply(a, b)
    if _leading is not None:
        c = c.view(*_leading, c.shape[-1])
    return c


class SwigluModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, a, b):
        return swiglu_deployment(a, b)


DEPLOYMENT_ENTRY = "swiglu_deployment"
