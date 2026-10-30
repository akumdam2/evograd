"""adbench-tools artifact: task llama_3_2_1b.swiglu.mlp_activation, produced by kernelagent.

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
def _swiglu_forward_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward SwiGLU activation:
        out = silu(a) * b = a * sigmoid(a) * b
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-a))
    out = a * sig * b

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
    Backward for:
        y = silu(a) * b

    Gradients:
        d/db = grad_out * silu(a)
        d/da = grad_out * b * d(silu(a))/da

    where:
        silu(a) = a * sigmoid(a)
        d(silu(a))/da = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sig
    dsilu = sig * (1.0 + a * (1.0 - sig))

    grad_a = go * b * dsilu
    grad_b = go * silu

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
    Fully fused forward + backward step for the evaluator's kernel_function.

    Computes in one pass:
        forward: out = silu(a) * b
        backward: grad_a, grad_b

    Stores a single flat float32 vector laid out exactly as:
        [out.flatten(), grad_a.flatten(), grad_b.flatten()]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-a))
    silu = a * sig
    out = silu * b

    dsilu = sig * (1.0 + a * (1.0 - sig))
    grad_a = go * b * dsilu
    grad_b = go * silu

    tl.store(flat_out_ptr + offsets, out.to(tl.float32), mask=mask)
    tl.store(flat_out_ptr + n_elements + offsets, grad_a.to(tl.float32), mask=mask)
    tl.store(flat_out_ptr + 2 * n_elements + offsets, grad_b.to(tl.float32), mask=mask)


def _validate_inputs(a, b):
    assert a.is_cuda and b.is_cuda, "SwiGLU Triton kernels require CUDA tensors"
    assert a.shape == b.shape, "a and b must have the same shape"
    assert a.is_contiguous() and b.is_contiguous(), "a and b must be contiguous"


def swiglu_forward_with_saved(a, b):
    """
    Forward half of the SwiGLU pair.

    This wrapper performs only validation/allocation/launch.  The math is inside
    the Triton kernel.  The saved tensors are the original inputs, allowing the
    backward kernel to recompute sigmoid/silu without storing extra activations.
    """
    _validate_inputs(a, b)

    out = torch.empty_like(a)
    n_elements = a.numel()

    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _swiglu_forward_kernel[grid](
            a,
            b,
            out,
            n_elements,
            BLOCK_SIZE,
            num_warps=4,
        )

    saved = (a, b)
    return out, saved


def swiglu_backward_from_saved(upstream, saved):
    """
    Backward half of the SwiGLU pair.

    Computes gradients with respect to both inputs:
        grad_a, grad_b

    The wrapper does no numerical work; all derivative math is fused inside the
    Triton kernel.
    """
    grad_out = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    a, b = saved

    _validate_inputs(a, b)
    assert grad_out.is_cuda, "upstream gradient must be a CUDA tensor"
    assert grad_out.shape == a.shape, "upstream gradient shape must match inputs"
    assert grad_out.is_contiguous(), "upstream gradient must be contiguous"

    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    n_elements = a.numel()

    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _swiglu_backward_kernel[grid](
            grad_out,
            a,
            b,
            grad_a,
            grad_b,
            n_elements,
            BLOCK_SIZE,
            num_warps=4,
        )

    return grad_a, grad_b


def kernel_function(a, b, grad_out):
    """
    Fused forward+backward SwiGLU step.

    Implements the evaluator-required combined operation in a single Triton pass:
        1. out    = silu(a) * b
        2. grad_a = grad_out * b * d(silu(a))/da
        3. grad_b = grad_out * silu(a)

    Returns one flat float32 tensor:
        cat([out.float().reshape(-1),
             grad_a.float().reshape(-1),
             grad_b.float().reshape(-1)])

    The wrapper only validates inputs, allocates the flat output, configures the
    grid, and launches Triton.  No PyTorch compute primitives are used.
    """
    _validate_inputs(a, b)
    assert grad_out.is_cuda, "grad_out must be a CUDA tensor"
    assert grad_out.shape == a.shape, "grad_out shape must match inputs"
    assert grad_out.is_contiguous(), "grad_out must be contiguous"

    n_elements = a.numel()
    flat = torch.empty((3 * n_elements,), device=a.device, dtype=torch.float32)

    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _swiglu_fused_forward_backward_flat_kernel[grid](
            a,
            b,
            grad_out,
            flat,
            n_elements,
            BLOCK_SIZE,
            num_warps=4,
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
