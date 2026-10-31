"""adbench-tools artifact: task olmoe_1b_7b.rope.rope, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the RoPE forward + backward pair used by this task.
#
# Contract for this extracted pair:
#   inputs:  x, cos, sin
#   output:  y = x * cos + rotate_half(x) * sin
#   grads:   dx only
#
# cos/sin are non-differentiated constants for this task, so backward must return
# exactly one gradient tensor: dx.
#
# Fusion note:
#   kernel_function fuses forward y and backward dx into a single Triton
#   elementwise pass and returns the evaluator-required flat concatenation:
#       cat([y.float().reshape(-1), dx.float().reshape(-1)])
#
# Python wrappers only validate metadata, allocate tensors, and launch Triton.
# All numerical computation is performed in Triton kernels.

import torch
import triton
import triton.language as tl


@triton.jit
def _offset4(
    i0, i1, i2, i3,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
):
    return i0 * S0 + i1 * S1 + i2 * S2 + i3 * S3


@triton.jit
def _broadcast_offset4(
    i0, i1, i2, i3,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
):
    j0 = 0 if C0 == 1 else i0
    j1 = 0 if C1 == 1 else i1
    j2 = 0 if C2 == 1 else i2
    j3 = 0 if C3 == 1 else i3
    return j0 * S0 + j1 * S1 + j2 * S2 + j3 * S3


@triton.jit
def _rope_forward_kernel(
    x_ptr, cos_ptr, sin_ptr, y_ptr,
    N: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr, SX3: tl.constexpr,
    SC0: tl.constexpr, SC1: tl.constexpr, SC2: tl.constexpr, SC3: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
    SY0: tl.constexpr, SY1: tl.constexpr, SY2: tl.constexpr, SY3: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    i3 = offs % X3
    t = offs // X3
    i2 = t % X2
    t = t // X2
    i1 = t % X1
    i0 = t // X1

    half: tl.constexpr = X3 // 2
    p3 = tl.where(i3 < half, i3 + half, i3 - half)

    x_off = _offset4(i0, i1, i2, i3, SX0, SX1, SX2, SX3)
    xp_off = _offset4(i0, i1, i2, p3, SX0, SX1, SX2, SX3)
    y_off = _offset4(i0, i1, i2, i3, SY0, SY1, SY2, SY3)

    c_off = _broadcast_offset4(i0, i1, i2, i3, C0, C1, C2, C3, SC0, SC1, SC2, SC3)
    s_off = _broadcast_offset4(i0, i1, i2, i3, C0, C1, C2, C3, SS0, SS1, SS2, SS3)

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + xp_off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + c_off, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + s_off, mask=mask, other=0.0).to(tl.float32)

    # rotate_half(x): concat(-x[..., half:], x[..., :half])
    rot = tl.where(i3 < half, -xp, xp)
    y = x * c + rot * s

    tl.store(y_ptr + y_off, y, mask=mask)


@triton.jit
def _rope_backward_x_kernel(
    dy_ptr, cos_ptr, sin_ptr, dx_ptr,
    N: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
    SDY0: tl.constexpr, SDY1: tl.constexpr, SDY2: tl.constexpr, SDY3: tl.constexpr,
    SDX0: tl.constexpr, SDX1: tl.constexpr, SDX2: tl.constexpr, SDX3: tl.constexpr,
    SC0: tl.constexpr, SC1: tl.constexpr, SC2: tl.constexpr, SC3: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    i3 = offs % X3
    t = offs // X3
    i2 = t % X2
    t = t // X2
    i1 = t % X1
    i0 = t // X1

    half: tl.constexpr = X3 // 2
    p3 = tl.where(i3 < half, i3 + half, i3 - half)

    dy_off = _offset4(i0, i1, i2, i3, SDY0, SDY1, SDY2, SDY3)
    dyp_off = _offset4(i0, i1, i2, p3, SDY0, SDY1, SDY2, SDY3)
    dx_off = _offset4(i0, i1, i2, i3, SDX0, SDX1, SDX2, SDX3)

    c_off = _broadcast_offset4(i0, i1, i2, i3, C0, C1, C2, C3, SC0, SC1, SC2, SC3)
    sp_off = _broadcast_offset4(i0, i1, i2, p3, C0, C1, C2, C3, SS0, SS1, SS2, SS3)

    dy = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dyp = tl.load(dy_ptr + dyp_off, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + c_off, mask=mask, other=0.0).to(tl.float32)
    sp = tl.load(sin_ptr + sp_off, mask=mask, other=0.0).to(tl.float32)

    # For first half:
    #   y_i = x_i*c_i - x_p*s_i
    #   y_p = x_p*c_p + x_i*s_p
    #   dx_i = dy_i*c_i + dy_p*s_p
    #
    # For second half:
    #   y_i = x_i*c_i + x_p*s_i
    #   y_p = x_p*c_p - x_i*s_p
    #   dx_i = dy_i*c_i - dy_p*s_p
    sign = tl.where(i3 < half, 1.0, -1.0)
    dx = dy * c + sign * dyp * sp

    tl.store(dx_ptr + dx_off, dx, mask=mask)


@triton.jit
def _rope_pair_flat_kernel(
    x_ptr, cos_ptr, sin_ptr, dy_ptr, out_ptr,
    N: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr, X3: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr, SX3: tl.constexpr,
    SDY0: tl.constexpr, SDY1: tl.constexpr, SDY2: tl.constexpr, SDY3: tl.constexpr,
    SC0: tl.constexpr, SC1: tl.constexpr, SC2: tl.constexpr, SC3: tl.constexpr,
    SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    i3 = offs % X3
    t = offs // X3
    i2 = t % X2
    t = t // X2
    i1 = t % X1
    i0 = t // X1

    half: tl.constexpr = X3 // 2
    p3 = tl.where(i3 < half, i3 + half, i3 - half)

    x_off = _offset4(i0, i1, i2, i3, SX0, SX1, SX2, SX3)
    xp_off = _offset4(i0, i1, i2, p3, SX0, SX1, SX2, SX3)

    dy_off = _offset4(i0, i1, i2, i3, SDY0, SDY1, SDY2, SDY3)
    dyp_off = _offset4(i0, i1, i2, p3, SDY0, SDY1, SDY2, SDY3)

    c_off = _broadcast_offset4(i0, i1, i2, i3, C0, C1, C2, C3, SC0, SC1, SC2, SC3)
    s_off = _broadcast_offset4(i0, i1, i2, i3, C0, C1, C2, C3, SS0, SS1, SS2, SS3)
    sp_off = _broadcast_offset4(i0, i1, i2, p3, C0, C1, C2, C3, SS0, SS1, SS2, SS3)

    x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    xp = tl.load(x_ptr + xp_off, mask=mask, other=0.0).to(tl.float32)

    dy = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dyp = tl.load(dy_ptr + dyp_off, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(cos_ptr + c_off, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + s_off, mask=mask, other=0.0).to(tl.float32)
    sp = tl.load(sin_ptr + sp_off, mask=mask, other=0.0).to(tl.float32)

    rot = tl.where(i3 < half, -xp, xp)
    y = x * c + rot * s

    sign = tl.where(i3 < half, 1.0, -1.0)
    dx = dy * c + sign * dyp * sp

    # Evaluator flat order:
    #   cat([y.float().reshape(-1), dx.float().reshape(-1)])
    tl.store(out_ptr + offs, y, mask=mask)
    tl.store(out_ptr + N + offs, dx, mask=mask)


def _pad4_shape_and_stride(t, x_ndim=None):
    shape = list(t.shape)
    stride = list(t.stride())

    if x_ndim is not None:
        assert len(shape) <= x_ndim, "cos/sin rank must be <= x rank for right-aligned broadcasting"
        add = x_ndim - len(shape)
        shape = [1] * add + shape
        stride = [0] * add + stride

    assert len(shape) <= 4, "this RoPE kernel supports tensors up to rank 4"
    add4 = 4 - len(shape)
    return [1] * add4 + shape, [0] * add4 + stride


def _promoted_output_dtype(x, cos, sin):
    # Allocation-only dtype selection matching PyTorch floating promotion for
    # the dtypes relevant to this task.
    dtype = torch.promote_types(x.dtype, cos.dtype)
    dtype = torch.promote_types(dtype, sin.dtype)
    return dtype


def _meta(x, cos, sin):
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.shape == sin.shape, "cos and sin must have the same shape"
    assert x.ndim <= 4, "this RoPE kernel supports tensors up to rank 4"
    assert x.shape[-1] % 2 == 0, "RoPE hidden dimension must be even"

    x_shape4, x_stride4 = _pad4_shape_and_stride(x)
    c_shape4, c_stride4 = _pad4_shape_and_stride(cos, x.ndim)
    s_shape4, s_stride4 = _pad4_shape_and_stride(sin, x.ndim)

    for xd, cd in zip(x_shape4, c_shape4):
        assert cd == 1 or cd == xd, "cos/sin must be broadcastable to x"

    return x_shape4, x_stride4, c_shape4, c_stride4, s_stride4


def rope_forward_with_saved(x, cos, sin):
    """
    Forward RoPE:
        y = x * cos + rotate_half(x) * sin

    Returns:
        y, saved
    """
    x_shape4, x_stride4, c_shape4, c_stride4, s_stride4 = _meta(x, cos, sin)

    # Important: if cos/sin are fp32 and x is bf16, PyTorch promotes the output
    # to fp32. Using empty_like(x) would incorrectly quantize the forward result.
    y = torch.empty(x.shape, device=x.device, dtype=_promoted_output_dtype(x, cos, sin))

    n = x.numel()
    if n > 0:
        y_shape4, y_stride4 = _pad4_shape_and_stride(y)
        block = 256
        grid = (triton.cdiv(n, block),)

        _rope_forward_kernel[grid](
            x, cos, sin, y,
            N=n,
            X0=x_shape4[0], X1=x_shape4[1], X2=x_shape4[2], X3=x_shape4[3],
            SX0=x_stride4[0], SX1=x_stride4[1], SX2=x_stride4[2], SX3=x_stride4[3],
            SC0=c_stride4[0], SC1=c_stride4[1], SC2=c_stride4[2], SC3=c_stride4[3],
            SS0=s_stride4[0], SS1=s_stride4[1], SS2=s_stride4[2], SS3=s_stride4[3],
            SY0=y_stride4[0], SY1=y_stride4[1], SY2=y_stride4[2], SY3=y_stride4[3],
            C0=c_shape4[0], C1=c_shape4[1], C2=c_shape4[2], C3=c_shape4[3],
            BLOCK_SIZE=block,
        )

    return y, (x, cos, sin)


def rope_backward_from_saved(upstream, saved):
    """
    Backward for this task's RoPE pair.

    Returns exactly one gradient:
        dx

    cos/sin are constants in the extracted graph and must not be returned as
    differentiable gradients.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, cos, sin = saved

    x_shape4, x_stride4, c_shape4, c_stride4, s_stride4 = _meta(x, cos, sin)
    dy_shape4, dy_stride4 = _pad4_shape_and_stride(dy)

    dx = torch.empty_like(x)

    n = x.numel()
    if n > 0:
        dx_shape4, dx_stride4 = _pad4_shape_and_stride(dx)
        block = 256
        grid = (triton.cdiv(n, block),)

        _rope_backward_x_kernel[grid](
            dy, cos, sin, dx,
            N=n,
            X0=x_shape4[0], X1=x_shape4[1], X2=x_shape4[2], X3=x_shape4[3],
            SDY0=dy_stride4[0], SDY1=dy_stride4[1], SDY2=dy_stride4[2], SDY3=dy_stride4[3],
            SDX0=dx_stride4[0], SDX1=dx_stride4[1], SDX2=dx_stride4[2], SDX3=dx_stride4[3],
            SC0=c_stride4[0], SC1=c_stride4[1], SC2=c_stride4[2], SC3=c_stride4[3],
            SS0=s_stride4[0], SS1=s_stride4[1], SS2=s_stride4[2], SS3=s_stride4[3],
            C0=c_shape4[0], C1=c_shape4[1], C2=c_shape4[2], C3=c_shape4[3],
            BLOCK_SIZE=block,
        )

    return dx


def kernel_function(x, cos, sin, upstream):
    """
    Fused forward+backward step.

    Returns a flat float32 tensor equivalent to:
        y, saved = rope_forward_with_saved(x, cos, sin)
        dx = rope_backward_from_saved(upstream, saved)
        torch.cat([y.float().reshape(-1), dx.float().reshape(-1)])

    Forward y and dx are computed in one Triton kernel.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    x_shape4, x_stride4, c_shape4, c_stride4, s_stride4 = _meta(x, cos, sin)
    dy_shape4, dy_stride4 = _pad4_shape_and_stride(dy)

    n = x.numel()
    total = 2 * n
    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    if n > 0:
        block = 256
        grid = (triton.cdiv(n, block),)

        _rope_pair_flat_kernel[grid](
            x, cos, sin, dy, out,
            N=n,
            X0=x_shape4[0], X1=x_shape4[1], X2=x_shape4[2], X3=x_shape4[3],
            SX0=x_stride4[0], SX1=x_stride4[1], SX2=x_stride4[2], SX3=x_stride4[3],
            SDY0=dy_stride4[0], SDY1=dy_stride4[1], SDY2=dy_stride4[2], SDY3=dy_stride4[3],
            SC0=c_stride4[0], SC1=c_stride4[1], SC2=c_stride4[2], SC3=c_stride4[3],
            SS0=s_stride4[0], SS1=s_stride4[1], SS2=s_stride4[2], SS3=s_stride4[3],
            C0=c_shape4[0], C1=c_shape4[1], C2=c_shape4[2], C3=c_shape4[3],
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
