"""adbench-tools artifact: task convnextv2_tiny.gelu.convnextv2.encoder.stages.0.layers.act, produced by kernelagent.

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
def _gelu_forward_contig_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    y = x * (0.5 * (1.0 + erf_x))
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_forward_strided_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    sx3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    i3 = offs % d3
    q = offs // d3
    i2 = q % d2
    q = q // d2
    i1 = q % d1
    i0 = q // d1

    x_offs = i0 * sx0 + i1 * sx1 + i2 * sx2 + i3 * sx3
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    y = x * (0.5 * (1.0 + erf_x))
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_backward_contig_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    cdf = 0.5 * (1.0 + erf_x)
    dx = dy * (cdf + x * (0.3989422804014327 * e))

    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_backward_strided_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    n_elements,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    sx3: tl.constexpr,
    sd0: tl.constexpr,
    sd1: tl.constexpr,
    sd2: tl.constexpr,
    sd3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    i3 = offs % d3
    q = offs // d3
    i2 = q % d2
    q = q // d2
    i1 = q % d1
    i0 = q // d1

    x_offs = i0 * sx0 + i1 * sx1 + i2 * sx2 + i3 * sx3
    dy_offs = i0 * sd0 + i1 * sd1 + i2 * sd2 + i3 * sd3

    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + dy_offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    cdf = 0.5 * (1.0 + erf_x)
    dx = dy * (cdf + x * (0.3989422804014327 * e))

    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gelu_fused_contig_kernel(
    x_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    cdf = 0.5 * (1.0 + erf_x)
    y = x * cdf
    dx = dy * (cdf + x * (0.3989422804014327 * e))

    # Match helper API tensors: forward output and grad are rounded to input dtype,
    # then kernel_function packages them as a single float32 vector.
    y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
    dx = dx.to(x_ptr.dtype.element_ty).to(tl.float32)

    tl.store(out_ptr + offs, y, mask=mask)
    tl.store(out_ptr + n_elements + offs, dx, mask=mask)


@triton.jit
def _gelu_fused_strided_kernel(
    x_ptr,
    dy_ptr,
    out_ptr,
    n_elements,
    d0: tl.constexpr,
    d1: tl.constexpr,
    d2: tl.constexpr,
    d3: tl.constexpr,
    sx0: tl.constexpr,
    sx1: tl.constexpr,
    sx2: tl.constexpr,
    sx3: tl.constexpr,
    sd0: tl.constexpr,
    sd1: tl.constexpr,
    sd2: tl.constexpr,
    sd3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    i3 = offs % d3
    q = offs // d3
    i2 = q % d2
    q = q // d2
    i1 = q % d1
    i0 = q // d1

    x_offs = i0 * sx0 + i1 * sx1 + i2 * sx2 + i3 * sx3
    dy_offs = i0 * sd0 + i1 * sd1 + i2 * sd2 + i3 * sd3

    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + dy_offs, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    z = ax * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    e = tl.exp(-0.5 * x * x)
    p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
    erf_abs = 1.0 - p * e
    erf_x = tl.where(x < 0.0, -erf_abs, erf_abs)

    cdf = 0.5 * (1.0 + erf_x)
    y = x * cdf
    dx = dy * (cdf + x * (0.3989422804014327 * e))

    y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
    dx = dx.to(x_ptr.dtype.element_ty).to(tl.float32)

    tl.store(out_ptr + offs, y, mask=mask)
    tl.store(out_ptr + n_elements + offs, dx, mask=mask)


def _shape4_and_strides4(t):
    dims = list(t.shape)
    strides = list(t.stride())
    while len(dims) < 4:
        dims.insert(0, 1)
        strides.insert(0, 0)
    return dims, strides


def gelu_forward_with_saved(x):
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    n_elements = x.numel()
    if n_elements == 0:
        return y, (x,)

    block = 1024
    grid = (triton.cdiv(n_elements, block),)

    if x.is_contiguous():
        _gelu_forward_contig_kernel[grid](
            x,
            y,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=4,
        )
    else:
        dims, sx = _shape4_and_strides4(x)
        _gelu_forward_strided_kernel[grid](
            x,
            y,
            n_elements,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            sx[0],
            sx[1],
            sx[2],
            sx[3],
            BLOCK_SIZE=block,
            num_warps=4,
        )

    return y, (x,)


def gelu_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]
    if isinstance(saved, (tuple, list)):
        x = saved[0]
    else:
        x = saved

    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    n_elements = x.numel()
    if n_elements == 0:
        return dx

    block = 1024
    grid = (triton.cdiv(n_elements, block),)

    if x.is_contiguous() and upstream.is_contiguous():
        _gelu_backward_contig_kernel[grid](
            x,
            upstream,
            dx,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=4,
        )
    else:
        dims, sx = _shape4_and_strides4(x)
        _, sd = _shape4_and_strides4(upstream)
        _gelu_backward_strided_kernel[grid](
            x,
            upstream,
            dx,
            n_elements,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            sx[0],
            sx[1],
            sx[2],
            sx[3],
            sd[0],
            sd[1],
            sd[2],
            sd[3],
            BLOCK_SIZE=block,
            num_warps=4,
        )

    return dx


def kernel_function(x, upstream):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    n_elements = x.numel()
    out = torch.empty((2 * n_elements,), device=x.device, dtype=torch.float32)
    if n_elements == 0:
        return out

    block = 1024
    grid = (triton.cdiv(n_elements, block),)

    # Fused forward GELU + backward input-gradient in one pass.
    if x.is_contiguous() and upstream.is_contiguous():
        _gelu_fused_contig_kernel[grid](
            x,
            upstream,
            out,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=4,
        )
    else:
        dims, sx = _shape4_and_strides4(x)
        _, sd = _shape4_and_strides4(upstream)
        _gelu_fused_strided_kernel[grid](
            x,
            upstream,
            out,
            n_elements,
            dims[0],
            dims[1],
            dims[2],
            dims[3],
            sx[0],
            sx[1],
            sx[2],
            sx[3],
            sd[0],
            sd[1],
            sd[2],
            sd[3],
            BLOCK_SIZE=block,
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
