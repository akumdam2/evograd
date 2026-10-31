"""adbench-tools artifact: task llama_3_2_1b.rope.rope, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton RoPE forward + backward implementation.
#
# Contract for this task:
#   forward:  y = rope(x, cos, sin)
#   backward: returns only dx. cos/sin are non-differentiated buffers.
#
# kernel_function returns the evaluator-required flattened vector:
#   concat([forward_output.float().reshape(-1), dx.float().reshape(-1)])
#
# Fusion note:
#   kernel_function fuses forward and dx computation in a single Triton kernel.
#   The standalone forward/backward helpers are kept for the evaluator API.

import torch
import triton
import triton.language as tl


@triton.jit
def _x_offset(
    b,
    h,
    s,
    d,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    XDIM: tl.constexpr,
    ORDER: tl.constexpr,
):
    # ORDER == 0: [B, H, S, D]
    # ORDER == 1: [B, S, H, D]
    # XDIM == 3: [B, S, D]
    # XDIM == 2: [S, D]
    if XDIM == 4:
        if ORDER == 0:
            return b * xs0 + h * xs1 + s * xs2 + d * xs3
        else:
            return b * xs0 + s * xs1 + h * xs2 + d * xs3
    elif XDIM == 3:
        return b * xs0 + s * xs1 + d * xs2
    else:
        return s * xs0 + d * xs1


@triton.jit
def _linear_to_bhsd(
    n,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    XDIM: tl.constexpr,
    ORDER: tl.constexpr,
):
    d = n % D
    outer = n // D

    if XDIM == 4:
        if ORDER == 0:
            # [B, H, S, D]
            s = outer % S
            tmp = outer // S
            h = tmp % H
            b = tmp // H
        else:
            # [B, S, H, D]
            h = outer % H
            tmp = outer // H
            s = tmp % S
            b = tmp // S
    elif XDIM == 3:
        # [B, S, D]
        s = outer % S
        b = outer // S
        h = outer * 0
    else:
        # [S, D]
        s = outer
        b = outer * 0
        h = outer * 0

    return b, h, s, d


@triton.jit
def _cos_storage_offset(
    b,
    h,
    s,
    k,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    CNDIM: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
):
    # Supports [K], [S,K], [B,S,K]/[1,S,K]/[S,1,K],
    # and [B,H,S,K]/broadcast variants.
    if CNDIM == 1:
        return k * cs0
    elif CNDIM == 2:
        return s * cs0 + k * cs1
    elif CNDIM == 3:
        if C0 == S:
            # [S, *, K]
            return s * cs0 + k * cs2
        else:
            bb = b
            if C0 == 1:
                bb = b * 0
            return bb * cs0 + s * cs1 + k * cs2
    else:
        bb = b
        hh = h
        if C0 == 1:
            bb = b * 0
        if C1 == 1:
            hh = h * 0
        return bb * cs0 + hh * cs1 + s * cs2 + k * cs3


@triton.jit
def _rope_forward_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    ys0: tl.constexpr,
    ys1: tl.constexpr,
    ys2: tl.constexpr,
    ys3: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    XDIM: tl.constexpr,
    ORDER: tl.constexpr,
    CNDIM: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    STYLE: tl.constexpr,  # 0: half-split, 1: interleaved even/odd pairs
    FULL_COS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    b, h, s, d = _linear_to_bhsd(offs, B, H, S, D, XDIM, ORDER)

    if STYLE == 1:
        # Interleaved RoPE:
        #   y_even = x_even * cos - x_odd  * sin
        #   y_odd  = x_odd  * cos + x_even * sin
        is_second = (d % 2) == 1
        pd = tl.where(is_second, d - 1, d + 1)
        k = d // 2
    else:
        # Half-split RoPE:
        #   y_lo = x_lo * cos - x_hi * sin
        #   y_hi = x_hi * cos + x_lo * sin
        half = D // 2
        is_second = d >= half
        pd = tl.where(is_second, d - half, d + half)
        k = d if FULL_COS else tl.where(is_second, d - half, d)

    xoff = _x_offset(b, h, s, d, xs0, xs1, xs2, xs3, XDIM, ORDER)
    poff = _x_offset(b, h, s, pd, xs0, xs1, xs2, xs3, XDIM, ORDER)
    yoff = _x_offset(b, h, s, d, ys0, ys1, ys2, ys3, XDIM, ORDER)

    coff = _cos_storage_offset(
        b, h, s, k,
        cs0, cs1, cs2, cs3,
        B, H, S,
        CNDIM, C0, C1, C2, C3,
    )

    xv = tl.load(x_ptr + xoff, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + poff, mask=mask, other=0.0).to(tl.float32)
    cv = tl.load(cos_ptr + coff, mask=mask, other=0.0).to(tl.float32)
    sv = tl.load(sin_ptr + coff, mask=mask, other=0.0).to(tl.float32)

    yv = tl.where(is_second, xv * cv + xp * sv, xv * cv - xp * sv)
    tl.store(y_ptr + yoff, yv, mask=mask)


@triton.jit
def _rope_backward_dx_kernel(
    cos_ptr,
    sin_ptr,
    grad_ptr,
    dx_ptr,
    n_elements: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    dxs0: tl.constexpr,
    dxs1: tl.constexpr,
    dxs2: tl.constexpr,
    dxs3: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    XDIM: tl.constexpr,
    ORDER: tl.constexpr,
    CNDIM: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    STYLE: tl.constexpr,
    FULL_COS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    b, h, s, d = _linear_to_bhsd(offs, B, H, S, D, XDIM, ORDER)

    if STYLE == 1:
        is_second = (d % 2) == 1
        pd = tl.where(is_second, d - 1, d + 1)
        k = d // 2
        pk = k
    else:
        half = D // 2
        is_second = d >= half
        pd = tl.where(is_second, d - half, d + half)
        k = d if FULL_COS else tl.where(is_second, d - half, d)
        pk = pd if FULL_COS else tl.where(is_second, d - half, d)

    goff = _x_offset(b, h, s, d, gs0, gs1, gs2, gs3, XDIM, ORDER)
    gpoff = _x_offset(b, h, s, pd, gs0, gs1, gs2, gs3, XDIM, ORDER)
    dxoff = _x_offset(b, h, s, d, dxs0, dxs1, dxs2, dxs3, XDIM, ORDER)

    coff = _cos_storage_offset(
        b, h, s, k,
        cs0, cs1, cs2, cs3,
        B, H, S,
        CNDIM, C0, C1, C2, C3,
    )
    pcoff = _cos_storage_offset(
        b, h, s, pk,
        cs0, cs1, cs2, cs3,
        B, H, S,
        CNDIM, C0, C1, C2, C3,
    )

    gv = tl.load(grad_ptr + goff, mask=mask, other=0.0).to(tl.float32)
    gp = tl.load(grad_ptr + gpoff, mask=mask, other=0.0).to(tl.float32)
    cv = tl.load(cos_ptr + coff, mask=mask, other=0.0).to(tl.float32)
    sp = tl.load(sin_ptr + pcoff, mask=mask, other=0.0).to(tl.float32)

    # dx_first  = gy_first  * cos + gy_second * sin
    # dx_second = gy_second * cos - gy_first  * sin
    dxv = tl.where(is_second, gv * cv - gp * sp, gv * cv + gp * sp)
    tl.store(dx_ptr + dxoff, dxv, mask=mask)


@triton.jit
def _rope_fused_ydx_vector_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    grad_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    gs0: tl.constexpr,
    gs1: tl.constexpr,
    gs2: tl.constexpr,
    gs3: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    XDIM: tl.constexpr,
    ORDER: tl.constexpr,
    CNDIM: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    STYLE: tl.constexpr,
    FULL_COS: tl.constexpr,
    Y_IS_X_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    b, h, s, d = _linear_to_bhsd(offs, B, H, S, D, XDIM, ORDER)

    if STYLE == 1:
        is_second = (d % 2) == 1
        pd = tl.where(is_second, d - 1, d + 1)
        k = d // 2
        pk = k
    else:
        half = D // 2
        is_second = d >= half
        pd = tl.where(is_second, d - half, d + half)
        k = d if FULL_COS else tl.where(is_second, d - half, d)
        pk = pd if FULL_COS else tl.where(is_second, d - half, d)

    xoff = _x_offset(b, h, s, d, xs0, xs1, xs2, xs3, XDIM, ORDER)
    poff = _x_offset(b, h, s, pd, xs0, xs1, xs2, xs3, XDIM, ORDER)
    goff = _x_offset(b, h, s, d, gs0, gs1, gs2, gs3, XDIM, ORDER)
    gpoff = _x_offset(b, h, s, pd, gs0, gs1, gs2, gs3, XDIM, ORDER)

    coff = _cos_storage_offset(
        b, h, s, k,
        cs0, cs1, cs2, cs3,
        B, H, S,
        CNDIM, C0, C1, C2, C3,
    )
    pcoff = _cos_storage_offset(
        b, h, s, pk,
        cs0, cs1, cs2, cs3,
        B, H, S,
        CNDIM, C0, C1, C2, C3,
    )

    xv = tl.load(x_ptr + xoff, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + poff, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(grad_ptr + goff, mask=mask, other=0.0).to(tl.float32)
    gp = tl.load(grad_ptr + gpoff, mask=mask, other=0.0).to(tl.float32)

    cv = tl.load(cos_ptr + coff, mask=mask, other=0.0).to(tl.float32)
    sv = tl.load(sin_ptr + coff, mask=mask, other=0.0).to(tl.float32)
    sp = tl.load(sin_ptr + pcoff, mask=mask, other=0.0).to(tl.float32)

    yv = tl.where(is_second, xv * cv + xp * sv, xv * cv - xp * sv)
    dxv = tl.where(is_second, gv * cv - gp * sp, gv * cv + gp * sp)

    # Match standalone helper semantics:
    # - forward output may be x dtype or promoted dtype;
    # - dx is gradient wrt x and is stored as x dtype by backward helper.
    if Y_IS_X_DTYPE:
        y_store = yv.to(x_ptr.dtype.element_ty).to(tl.float32)
    else:
        y_store = yv

    dx_store = dxv.to(x_ptr.dtype.element_ty).to(tl.float32)

    tl.store(out_ptr + offs, y_store, mask=mask)
    tl.store(out_ptr + n_elements + offs, dx_store, mask=mask)


def _result_dtype(x, cos, sin):
    # Allocation/dtype metadata only.
    return torch.promote_types(torch.promote_types(x.dtype, cos.dtype), sin.dtype)


def _meta(x, cos):
    """Shape/stride metadata only; no tensor math."""
    xdim = x.dim()
    if xdim == 4:
        seq_from_cos = cos.shape[-2] if cos.dim() >= 2 else x.shape[2]
        if x.shape[2] == seq_from_cos:
            order = 0  # [B, H, S, D]
            B, H, S, D = x.shape
        else:
            order = 1  # [B, S, H, D]
            B, S, H, D = x.shape
    elif xdim == 3:
        order = 0
        B, S, D = x.shape
        H = 1
    elif xdim == 2:
        order = 0
        S, D = x.shape
        B = 1
        H = 1
    else:
        raise AssertionError("RoPE kernel expects x rank 2, 3, or 4")

    cndim = cos.dim()
    if cndim < 1 or cndim > 4:
        raise AssertionError("cos/sin rank must be between 1 and 4")

    cshape = list(cos.shape)
    cstride = list(cos.stride())
    while len(cshape) < 4:
        cshape.append(1)
        cstride.append(0)

    xstride = list(x.stride())
    while len(xstride) < 4:
        xstride.append(0)

    full_cos = cos.shape[-1] == D

    # LLaMA-style implementations commonly use:
    #   - cos/sin last dim D     : half-split rotate_half layout
    #   - cos/sin last dim D / 2 : interleaved even/odd complex-pair layout
    style = 1 if cos.shape[-1] * 2 == D else 0

    return (
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape,
        cstride,
        xstride,
        full_cos,
        style,
    )


def rope_forward_with_saved(x, cos, sin):
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.shape == sin.shape

    out_dtype = _result_dtype(x, cos, sin)
    y = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=out_dtype)

    (
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape,
        cstride,
        xstride,
        full_cos,
        style,
    ) = _meta(x, cos)

    ystride = list(y.stride())
    while len(ystride) < 4:
        ystride.append(0)

    n = x.numel()
    block = 256
    grid = (triton.cdiv(n, block),)

    _rope_forward_kernel[grid](
        x,
        cos,
        sin,
        y,
        n,
        xstride[0],
        xstride[1],
        xstride[2],
        xstride[3],
        ystride[0],
        ystride[1],
        ystride[2],
        ystride[3],
        cstride[0],
        cstride[1],
        cstride[2],
        cstride[3],
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape[0],
        cshape[1],
        cshape[2],
        cshape[3],
        style,
        full_cos,
        BLOCK_SIZE=block,
    )

    return y, (x, cos, sin)


def rope_backward_from_saved(upstream, saved):
    """Backward for RoPE. Contract requires only dx."""
    if isinstance(upstream, (tuple, list)):
        grad = upstream[0]
    else:
        grad = upstream

    x, cos, sin = saved
    assert x.is_cuda and cos.is_cuda and sin.is_cuda and grad.is_cuda
    assert x.shape == grad.shape

    dx = torch.empty_like(x)

    (
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape,
        cstride,
        xstride,
        full_cos,
        style,
    ) = _meta(x, cos)

    gstride = list(grad.stride())
    dxstride = list(dx.stride())
    while len(gstride) < 4:
        gstride.append(0)
    while len(dxstride) < 4:
        dxstride.append(0)

    n = x.numel()
    block = 256

    _rope_backward_dx_kernel[(triton.cdiv(n, block),)](
        cos,
        sin,
        grad,
        dx,
        n,
        gstride[0],
        gstride[1],
        gstride[2],
        gstride[3],
        dxstride[0],
        dxstride[1],
        dxstride[2],
        dxstride[3],
        cstride[0],
        cstride[1],
        cstride[2],
        cstride[3],
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape[0],
        cshape[1],
        cshape[2],
        cshape[3],
        style,
        full_cos,
        BLOCK_SIZE=block,
    )

    return dx


def kernel_function(x, cos, sin, upstream):
    """
    Fused forward+backward evaluator entry point.

    Returns:
        1D float32 tensor:
        concat([rope(x, cos, sin).float().reshape(-1),
                dx.float().reshape(-1)])
    """
    if isinstance(upstream, (tuple, list)):
        grad = upstream[0]
    else:
        grad = upstream

    assert x.is_cuda and cos.is_cuda and sin.is_cuda and grad.is_cuda
    assert cos.shape == sin.shape
    assert x.shape == grad.shape

    (
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape,
        cstride,
        xstride,
        full_cos,
        style,
    ) = _meta(x, cos)

    gstride = list(grad.stride())
    while len(gstride) < 4:
        gstride.append(0)

    n = x.numel()
    out = torch.empty((2 * n,), device=x.device, dtype=torch.float32)

    out_dtype = _result_dtype(x, cos, sin)
    y_is_x_dtype = out_dtype == x.dtype

    block = 256

    _rope_fused_ydx_vector_kernel[(triton.cdiv(n, block),)](
        x,
        cos,
        sin,
        grad,
        out,
        n,
        xstride[0],
        xstride[1],
        xstride[2],
        xstride[3],
        gstride[0],
        gstride[1],
        gstride[2],
        gstride[3],
        cstride[0],
        cstride[1],
        cstride[2],
        cstride[3],
        B,
        H,
        S,
        D,
        xdim,
        order,
        cndim,
        cshape[0],
        cshape[1],
        cshape[2],
        cshape[3],
        style,
        full_cos,
        y_is_x_dtype,
        BLOCK_SIZE=block,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'rope', 'arguments': ['x', 'cos', 'sin'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx'], 'gradient_return_order': ['dx', 'None', 'None'], 'scalars': [], 'deployment_entry': 'rope_deployment', 'allowed_primitives': []}


class RopeFunction(torch.autograd.Function):
    """Static autograd wiring for `rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        y, saved = rope_forward_with_saved(x, cos, sin)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
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
        dx = rope_backward_from_saved(
            dy, saved
        )
        if isinstance(dx, (tuple, list)):
            dx, = dx
        # One value per forward argument, `None` for every inactive one.
        return dx, None, None


def rope_deployment(x, cos, sin):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = RopeFunction.apply(x, cos, sin)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class RopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return rope_deployment(x, cos, sin)


DEPLOYMENT_ENTRY = "rope_deployment"
