"""adbench-tools artifact: task qwen2_vl_2b.gelu.model.visual.merger.mlp, produced by kernelagent.

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


# Exact GELU:
#   y = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
# Backward:
#   dy/dx = Phi(x) + x * phi(x)
# where:
#   phi(x) = exp(-0.5*x*x) / sqrt(2*pi)
#
# The public kernel_function fuses the forward GELU and backward-gradient
# computation in one Triton pass and returns the flattened concatenation:
#   cat([gelu(x).reshape(-1), grad_x.reshape(-1)])
# This matches the evaluator's required "forward + backward step" output.


@triton.jit
def _gelu_forward_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _gelu_backward_kernel(
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

    inv_sqrt2 = 0.7071067811865476
    inv_sqrt_2pi = 0.3989422804014327

    cdf = 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
    pdf_term = x * inv_sqrt_2pi * tl.exp(-0.5 * x * x)
    grad = grad_out * (cdf + pdf_term)

    tl.store(grad_x_ptr + offsets, grad, mask=mask)


@triton.jit
def _gelu_fused_forward_backward_flat_kernel(
    x_ptr,
    grad_out_ptr,
    flat_out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU forward + backward.

    Stores:
      flat_out[0:n]   = gelu(x)
      flat_out[n:2*n] = grad_out * dgelu(x)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.7071067811865476
    inv_sqrt_2pi = 0.3989422804014327

    cdf = 0.5 * (1.0 + tl.erf(x * inv_sqrt2))
    y = x * cdf

    pdf_term = x * inv_sqrt_2pi * tl.exp(-0.5 * x * x)
    grad_x = grad_out * (cdf + pdf_term)

    tl.store(flat_out_ptr + offsets, y, mask=mask)
    tl.store(flat_out_ptr + n_elements + offsets, grad_x, mask=mask)


def _launch_grid(n_elements, block_size):
    return (triton.cdiv(n_elements, block_size),)


def gelu_forward_with_saved(x):
    """
    Forward GELU implemented in Triton.

    Returns:
        output: GELU(x), same shape/device/dtype as x
        saved: tuple containing x for backward
    """
    assert x.is_cuda, "Triton GELU kernel requires a CUDA tensor"
    assert x.is_contiguous(), "This implementation expects contiguous input"

    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n_elements = x.numel()

    if n_elements > 0:
        block_size = 1024
        _gelu_forward_kernel[_launch_grid(n_elements, block_size)](
            x,
            y,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )

    return y, (x,)


def gelu_backward_from_saved(upstream, saved):
    """
    Backward GELU implemented in Triton.

    Args:
        upstream: gradient of loss with respect to GELU output
        saved: saved tuple from gelu_forward_with_saved

    Returns:
        grad_x
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    x = saved[0] if isinstance(saved, (tuple, list)) else saved

    assert x.is_cuda and upstream.is_cuda, "Triton GELU backward requires CUDA tensors"
    assert x.is_contiguous(), "This implementation expects contiguous x"
    assert upstream.is_contiguous(), "This implementation expects contiguous upstream gradient"
    assert x.numel() == upstream.numel(), "upstream gradient must have the same number of elements as x"

    grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    n_elements = x.numel()

    if n_elements > 0:
        block_size = 1024
        _gelu_backward_kernel[_launch_grid(n_elements, block_size)](
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
    Fused forward+backward GELU step required by the evaluator.

    This wrapper performs only validation, allocation, and Triton launch setup.
    The single Triton kernel fuses:
      1. GELU forward: y = 0.5 * x * (1 + erf(x / sqrt(2)))
      2. GELU backward: grad_x = upstream * dgelu(x)
      3. Flattened concatenated storage of [y, grad_x]

    Returns:
        A flat tensor of shape [2 * x.numel()] containing:
          first half  = GELU(x)
          second half = gradient with respect to x
    """
    assert x.is_cuda and upstream.is_cuda, "Triton GELU kernel requires CUDA tensors"
    assert x.is_contiguous(), "This implementation expects contiguous x"
    assert upstream.is_contiguous(), "This implementation expects contiguous upstream gradient"
    assert x.numel() == upstream.numel(), "upstream gradient must have the same number of elements as x"

    n_elements = x.numel()
    out = torch.empty((2 * n_elements,), device=x.device, dtype=x.dtype)

    if n_elements > 0:
        block_size = 1024
        _gelu_fused_forward_backward_flat_kernel[_launch_grid(n_elements, block_size)](
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gelu', 'arguments': ['x'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx'], 'gradient_return_order': ['dx'], 'scalars': [], 'deployment_entry': 'gelu_deployment', 'allowed_primitives': []}


class GeluFunction(torch.autograd.Function):
    """Static autograd wiring for `gelu`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x):
        out, saved = gelu_forward_with_saved(x)
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
        dx = gelu_backward_from_saved(
            dout, saved
        )
        if isinstance(dx, (tuple, list)):
            dx, = dx
        # One value per forward argument, `None` for every inactive one.
        return dx,


def gelu_deployment(x):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = GeluFunction.apply(x)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class GeluModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return gelu_deployment(x)


DEPLOYMENT_ENTRY = "gelu_deployment"
