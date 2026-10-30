"""adbench-tools artifact: task olmoe_1b_7b.swiglu.moe_activation, produced by kernelagent.

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
    rows,
    cols,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    out_s0,
    out_s1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward SwiGLU with arbitrary 1D/2D logical strides:
        out = silu(a) * b = a * sigmoid(a) * b
    """
    row = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)

    cols_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < rows) & (cols_offsets < cols)

    a_offsets = row * a_s0 + cols_offsets * a_s1
    b_offsets = row * b_s0 + cols_offsets * b_s1
    out_offsets = row * out_s0 + cols_offsets * out_s1

    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    out = a * sigmoid_a * b

    tl.store(out_ptr + out_offsets, out, mask=mask)


@triton.jit
def _swiglu_backward_kernel(
    grad_out_ptr,
    a_ptr,
    b_ptr,
    grad_a_ptr,
    grad_b_ptr,
    rows,
    cols,
    go_s0,
    go_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    ga_s0,
    ga_s1,
    gb_s0,
    gb_s1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward for:
        y = silu(a) * b

    Gradients:
        d/da silu(a) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
        grad_a = grad_out * b * d/da silu(a)
        grad_b = grad_out * silu(a)
    """
    row = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)

    cols_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < rows) & (cols_offsets < cols)

    go_offsets = row * go_s0 + cols_offsets * go_s1
    a_offsets = row * a_s0 + cols_offsets * a_s1
    b_offsets = row * b_s0 + cols_offsets * b_s1
    ga_offsets = row * ga_s0 + cols_offsets * ga_s1
    gb_offsets = row * gb_s0 + cols_offsets * gb_s1

    go = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sigmoid_a
    dsilu_da = sigmoid_a * (1.0 + a * (1.0 - sigmoid_a))

    grad_a = go * b * dsilu_da
    grad_b = go * silu_a

    tl.store(grad_a_ptr + ga_offsets, grad_a, mask=mask)
    tl.store(grad_b_ptr + gb_offsets, grad_b, mask=mask)


@triton.jit
def _swiglu_fused_forward_backward_concat_kernel(
    a_ptr,
    b_ptr,
    grad_out_ptr,
    result_ptr,
    rows,
    cols,
    n_elements,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    go_s0,
    go_s1,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fully fused forward + backward + concatenation kernel.

    Computes all dependent stages in one pass:
      1. y      = silu(a) * b
      2. grad_a = upstream * b * d/da silu(a)
      3. grad_b = upstream * silu(a)

    Stores the evaluator-required flattened concatenation:
      result[0:n]     = y
      result[n:2n]    = grad_a
      result[2n:3n]   = grad_b

    Input tensors may be non-contiguous; logical row-major output order is used.
    """
    row = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)

    cols_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (row < rows) & (cols_offsets < cols)

    logical_offsets = row * cols + cols_offsets

    a_offsets = row * a_s0 + cols_offsets * a_s1
    b_offsets = row * b_s0 + cols_offsets * b_s1
    go_offsets = row * go_s0 + cols_offsets * go_s1

    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0).to(tl.float32)
    go = tl.load(grad_out_ptr + go_offsets, mask=mask, other=0.0).to(tl.float32)

    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    silu_a = a * sigmoid_a

    y = silu_a * b
    dsilu_da = sigmoid_a * (1.0 + a * (1.0 - sigmoid_a))
    grad_a = go * b * dsilu_da
    grad_b = go * silu_a

    tl.store(result_ptr + logical_offsets, y, mask=mask)
    tl.store(result_ptr + n_elements + logical_offsets, grad_a, mask=mask)
    tl.store(result_ptr + 2 * n_elements + logical_offsets, grad_b, mask=mask)


def _check_common(a, b):
    assert a.is_cuda and b.is_cuda, "inputs must be CUDA tensors"
    assert a.shape == b.shape, "a and b must have the same shape"
    assert a.dtype == b.dtype, "a and b must have the same dtype"
    assert a.dim() in (1, 2), "this Triton implementation expects 1D or 2D tensors"


def _meta_2d(x):
    """
    Return logical 2D metadata for a 1D or 2D tensor.

    This uses shape/stride metadata only. It does not perform tensor computation.
    """
    if x.dim() == 1:
        rows = 1
        cols = x.shape[0]
        s0 = 0
        s1 = x.stride(0)
    else:
        rows = x.shape[0]
        cols = x.shape[1]
        s0 = x.stride(0)
        s1 = x.stride(1)
    return rows, cols, s0, s1


def swiglu_forward_with_saved(a, b):
    """
    Forward half of the pair used by the evaluator.

    Computes:
        output = silu(a) * b = a * sigmoid(a) * b

    Supports non-contiguous input tensors by using their logical strides.
    """
    _check_common(a, b)

    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)

    rows, cols, a_s0, a_s1 = _meta_2d(a)
    _, _, b_s0, b_s1 = _meta_2d(b)
    _, _, out_s0, out_s1 = _meta_2d(out)

    n_elements = a.numel()
    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (rows, triton.cdiv(cols, BLOCK_SIZE))
        _swiglu_forward_kernel[grid](
            a,
            b,
            out,
            rows,
            cols,
            a_s0,
            a_s1,
            b_s0,
            b_s1,
            out_s0,
            out_s1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    saved = (a, b)
    return out, saved


def swiglu_backward_from_saved(upstream, saved):
    """
    Backward half of the pair used by the evaluator.

    Given upstream dL/dout, computes gradients with respect to a and b.
    Supports non-contiguous inputs and upstream gradients via strides.
    """
    if isinstance(upstream, (tuple, list)):
        grad_out = upstream[0]
    else:
        grad_out = upstream

    a, b = saved
    _check_common(a, b)
    assert grad_out.is_cuda, "upstream gradient must be a CUDA tensor"
    assert grad_out.shape == a.shape, "upstream gradient shape must match inputs"
    assert grad_out.dim() in (1, 2), "upstream gradient must be 1D or 2D"

    grad_a = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    grad_b = torch.empty(b.shape, device=b.device, dtype=b.dtype)

    rows, cols, a_s0, a_s1 = _meta_2d(a)
    _, _, b_s0, b_s1 = _meta_2d(b)
    _, _, go_s0, go_s1 = _meta_2d(grad_out)
    _, _, ga_s0, ga_s1 = _meta_2d(grad_a)
    _, _, gb_s0, gb_s1 = _meta_2d(grad_b)

    n_elements = a.numel()
    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (rows, triton.cdiv(cols, BLOCK_SIZE))
        _swiglu_backward_kernel[grid](
            grad_out,
            a,
            b,
            grad_a,
            grad_b,
            rows,
            cols,
            go_s0,
            go_s1,
            a_s0,
            a_s1,
            b_s0,
            b_s1,
            ga_s0,
            ga_s1,
            gb_s0,
            gb_s1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return grad_a, grad_b


def kernel_function(a, b, upstream):
    """
    Fused forward+backward SwiGLU step.

    Returns:
        cat([
            y.float().reshape(-1),
            grad_a.float().reshape(-1),
            grad_b.float().reshape(-1),
        ])

    The actual forward, backward, and result packaging are fused into a single
    Triton kernel. Non-contiguous inputs are handled through explicit strides.
    """
    _check_common(a, b)
    assert upstream.is_cuda, "upstream gradient must be a CUDA tensor"
    assert upstream.shape == a.shape, "upstream gradient shape must match inputs"
    assert upstream.dim() in (1, 2), "upstream gradient must be 1D or 2D"

    rows, cols, a_s0, a_s1 = _meta_2d(a)
    _, _, b_s0, b_s1 = _meta_2d(b)
    _, _, go_s0, go_s1 = _meta_2d(upstream)

    n_elements = a.numel()
    result = torch.empty((3 * n_elements,), device=a.device, dtype=torch.float32)

    if n_elements > 0:
        BLOCK_SIZE = 1024
        grid = (rows, triton.cdiv(cols, BLOCK_SIZE))
        _swiglu_fused_forward_backward_concat_kernel[grid](
            a,
            b,
            upstream,
            result,
            rows,
            cols,
            n_elements,
            a_s0,
            a_s1,
            b_s0,
            b_s1,
            go_s0,
            go_s1,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return result
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
