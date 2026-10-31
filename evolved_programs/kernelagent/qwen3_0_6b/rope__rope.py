"""adbench-tools artifact: task qwen3_0_6b.rope.rope, produced by kernelagent.

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
def _logical_4d_indices(
    offsets,
    D: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
):
    d = offsets % D
    t0 = offsets // D
    s = t0 % S
    t1 = t0 // S
    h = t1 % H
    b = t1 // H
    return b, h, s, d


@triton.jit
def _strided_offset_4d(
    b,
    h,
    s,
    d,
    stride_b,
    stride_h,
    stride_s,
    stride_d,
):
    return b * stride_b + h * stride_h + s * stride_s + d * stride_d


@triton.jit
def _rope_table_offset(
    b,
    h,
    s,
    d,
    COS_MODE: tl.constexpr,
    cos_stride_b,
    cos_stride_h,
    cos_stride_s,
    cos_stride_d,
):
    """
    Supported cos/sin logical layouts:
      COS_MODE 1: [D]
      COS_MODE 2: [S, D]
      COS_MODE 3: [B, S, D]
      COS_MODE 4: [B, 1, S, D] / broadcast over heads
      COS_MODE 0: [B, H, S, D]
    """
    if COS_MODE == 1:
        return d * cos_stride_d
    elif COS_MODE == 2:
        return s * cos_stride_s + d * cos_stride_d
    elif COS_MODE == 3:
        return b * cos_stride_b + s * cos_stride_s + d * cos_stride_d
    elif COS_MODE == 4:
        return b * cos_stride_b + s * cos_stride_s + d * cos_stride_d
    else:
        return (
            b * cos_stride_b
            + h * cos_stride_h
            + s * cos_stride_s
            + d * cos_stride_d
        )


@triton.jit
def _rope_forward_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    x_stride_d,
    y_stride_b,
    y_stride_h,
    y_stride_s,
    y_stride_d,
    cos_stride_b,
    cos_stride_h,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_h,
    sin_stride_s,
    sin_stride_d,
    COS_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward RoPE:
        y = x * cos + rotate_half(x) * sin

    rotate_half([x0, x1]) = [-x1, x0] over the final dimension split in half.

    This kernel uses logical row-major indexing for [B, H, S, D], but loads and
    stores through explicit strides. This is required for Qwen attention tensors,
    which are typically non-contiguous after transpose.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    half: tl.constexpr = D // 2

    b, h, s, d = _logical_4d_indices(offsets, D, S, H)
    pair_d = tl.where(d < half, d + half, d - half)

    x_off = _strided_offset_4d(
        b, h, s, d, x_stride_b, x_stride_h, x_stride_s, x_stride_d
    )
    x_pair_off = _strided_offset_4d(
        b, h, s, pair_d, x_stride_b, x_stride_h, x_stride_s, x_stride_d
    )
    y_off = _strided_offset_4d(
        b, h, s, d, y_stride_b, y_stride_h, y_stride_s, y_stride_d
    )

    cs_off = _rope_table_offset(
        b,
        h,
        s,
        d,
        COS_MODE,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    )
    ss_off = _rope_table_offset(
        b,
        h,
        s,
        d,
        COS_MODE,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    )

    x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_pair_off, mask=mask, other=0.0).to(tl.float32)
    cos_val = tl.load(cos_ptr + cs_off, mask=mask, other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + ss_off, mask=mask, other=0.0).to(tl.float32)

    rot_sign = tl.where(d < half, -1.0, 1.0)
    y = x_val * cos_val + x_pair * rot_sign * sin_val

    tl.store(y_ptr + y_off, y, mask=mask)


@triton.jit
def _rope_backward_dx_kernel(
    dy_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    dx_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    dy_stride_b,
    dy_stride_h,
    dy_stride_s,
    dy_stride_d,
    dx_stride_b,
    dx_stride_h,
    dx_stride_s,
    dx_stride_d,
    cos_stride_b,
    cos_stride_h,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_h,
    sin_stride_s,
    sin_stride_d,
    COS_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward for RoPE with respect to x only.

    The task contract requires only dx; cos/sin are rotary tables and are not
    differentiated for this pair.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    half: tl.constexpr = D // 2

    b, h, s, d = _logical_4d_indices(offsets, D, S, H)
    pair_d = tl.where(d < half, d + half, d - half)

    dy_off = _strided_offset_4d(
        b, h, s, d, dy_stride_b, dy_stride_h, dy_stride_s, dy_stride_d
    )
    dy_pair_off = _strided_offset_4d(
        b, h, s, pair_d, dy_stride_b, dy_stride_h, dy_stride_s, dy_stride_d
    )
    dx_off = _strided_offset_4d(
        b, h, s, d, dx_stride_b, dx_stride_h, dx_stride_s, dx_stride_d
    )

    cos_off = _rope_table_offset(
        b,
        h,
        s,
        d,
        COS_MODE,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    )
    sin_pair_off = _rope_table_offset(
        b,
        h,
        s,
        pair_d,
        COS_MODE,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    )

    dy = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dy_pair = tl.load(dy_ptr + dy_pair_off, mask=mask, other=0.0).to(tl.float32)
    cos_val = tl.load(cos_ptr + cos_off, mask=mask, other=0.0).to(tl.float32)
    sin_pair = tl.load(sin_ptr + sin_pair_off, mask=mask, other=0.0).to(tl.float32)

    cross_sign = tl.where(d < half, 1.0, -1.0)
    dx = dy * cos_val + dy_pair * sin_pair * cross_sign

    tl.store(dx_ptr + dx_off, dx, mask=mask)


@triton.jit
def _rope_pair_flat_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    dy_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    x_stride_d,
    dy_stride_b,
    dy_stride_h,
    dy_stride_s,
    dy_stride_d,
    cos_stride_b,
    cos_stride_h,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_h,
    sin_stride_s,
    sin_stride_d,
    COS_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused forward + backward(dx) step for kernel_function.

    Writes a flat fp32 tensor:
        [rope_forward(x, cos, sin).reshape(-1), dx.reshape(-1)]

    Forward and dx are fused because both need the same logical indexing and
    rotary table loads. Gradients for cos/sin are intentionally omitted because
    the pair contract only requires dx.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    half: tl.constexpr = D // 2

    b, h, s, d = _logical_4d_indices(offsets, D, S, H)
    pair_d = tl.where(d < half, d + half, d - half)

    x_off = _strided_offset_4d(
        b, h, s, d, x_stride_b, x_stride_h, x_stride_s, x_stride_d
    )
    x_pair_off = _strided_offset_4d(
        b, h, s, pair_d, x_stride_b, x_stride_h, x_stride_s, x_stride_d
    )

    dy_off = _strided_offset_4d(
        b, h, s, d, dy_stride_b, dy_stride_h, dy_stride_s, dy_stride_d
    )
    dy_pair_off = _strided_offset_4d(
        b, h, s, pair_d, dy_stride_b, dy_stride_h, dy_stride_s, dy_stride_d
    )

    cos_off = _rope_table_offset(
        b,
        h,
        s,
        d,
        COS_MODE,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    )
    sin_off = _rope_table_offset(
        b,
        h,
        s,
        d,
        COS_MODE,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    )
    sin_pair_off = _rope_table_offset(
        b,
        h,
        s,
        pair_d,
        COS_MODE,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    )

    x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_pair_off, mask=mask, other=0.0).to(tl.float32)

    dy = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dy_pair = tl.load(dy_ptr + dy_pair_off, mask=mask, other=0.0).to(tl.float32)

    cos_val = tl.load(cos_ptr + cos_off, mask=mask, other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + sin_off, mask=mask, other=0.0).to(tl.float32)
    sin_pair = tl.load(sin_ptr + sin_pair_off, mask=mask, other=0.0).to(tl.float32)

    rot_sign = tl.where(d < half, -1.0, 1.0)
    y = x_val * cos_val + x_pair * rot_sign * sin_val

    cross_sign = tl.where(d < half, 1.0, -1.0)
    dx = dy * cos_val + dy_pair * sin_pair * cross_sign

    tl.store(out_ptr + offsets, y, mask=mask)
    tl.store(out_ptr + N + offsets, dx, mask=mask)


def _x_meta(x):
    assert x.is_cuda
    assert x.dim() >= 2
    assert x.shape[-1] % 2 == 0

    d = x.shape[-1]
    s = x.shape[-2]

    if x.dim() >= 4:
        h = x.shape[-3]
        b = x.numel() // (h * s * d)
        stride_b = x.stride(-4)
        stride_h = x.stride(-3)
        stride_s = x.stride(-2)
        stride_d = x.stride(-1)
    elif x.dim() == 3:
        b = x.shape[0]
        h = 1
        stride_b = x.stride(0)
        stride_h = 0
        stride_s = x.stride(1)
        stride_d = x.stride(2)
    else:
        b = 1
        h = 1
        stride_b = 0
        stride_h = 0
        stride_s = x.stride(0)
        stride_d = x.stride(1)

    return x.numel(), b, h, s, d, stride_b, stride_h, stride_s, stride_d


def _table_meta(table, b, h, s, d):
    assert table.is_cuda
    assert table.shape[-1] == d

    if table.dim() == 1:
        mode = 1
        stride_b = 0
        stride_h = 0
        stride_s = 0
        stride_d = table.stride(0)
    elif table.dim() == 2:
        mode = 2
        stride_b = 0
        stride_h = 0
        stride_s = table.stride(0)
        stride_d = table.stride(1)
    elif table.dim() == 3:
        mode = 3
        stride_b = table.stride(0)
        stride_h = 0
        stride_s = table.stride(1)
        stride_d = table.stride(2)
    else:
        if table.shape[-3] == h and table.shape[-4] == b:
            mode = 0
            stride_b = table.stride(-4)
            stride_h = table.stride(-3)
            stride_s = table.stride(-2)
            stride_d = table.stride(-1)
        else:
            mode = 4
            stride_b = table.stride(-4)
            stride_h = 0
            stride_s = table.stride(-2)
            stride_d = table.stride(-1)

    return mode, stride_b, stride_h, stride_s, stride_d


def rope_forward_with_saved(x, cos, sin):
    """
    Triton RoPE forward.

    Computes:
        y = x * cos + rotate_half(x) * sin
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.numel() == sin.numel()

    (
        n,
        b,
        h,
        s,
        d,
        x_stride_b,
        x_stride_h,
        x_stride_s,
        x_stride_d,
    ) = _x_meta(x)

    (
        cos_mode,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    ) = _table_meta(cos, b, h, s, d)
    (
        sin_mode,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    ) = _table_meta(sin, b, h, s, d)

    assert cos_mode == sin_mode

    y = torch.empty_like(x)

    block = 256
    grid = (triton.cdiv(n, block),)
    _rope_forward_kernel[grid](
        x,
        cos,
        sin,
        y,
        N=n,
        D=d,
        S=s,
        H=h,
        x_stride_b=x_stride_b,
        x_stride_h=x_stride_h,
        x_stride_s=x_stride_s,
        x_stride_d=x_stride_d,
        y_stride_b=y.stride(-4) if y.dim() >= 4 else (y.stride(0) if y.dim() == 3 else 0),
        y_stride_h=y.stride(-3) if y.dim() >= 4 else 0,
        y_stride_s=y.stride(-2),
        y_stride_d=y.stride(-1),
        cos_stride_b=cos_stride_b,
        cos_stride_h=cos_stride_h,
        cos_stride_s=cos_stride_s,
        cos_stride_d=cos_stride_d,
        sin_stride_b=sin_stride_b,
        sin_stride_h=sin_stride_h,
        sin_stride_s=sin_stride_s,
        sin_stride_d=sin_stride_d,
        COS_MODE=cos_mode,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return y, (x, cos, sin)


def rope_backward_from_saved(upstream, saved):
    """
    Triton RoPE backward.

    Returns only dx, matching the task contract.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, cos, sin = saved
    assert dy.is_cuda and x.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.numel() == sin.numel()

    (
        n,
        b,
        h,
        s,
        d,
        _x_stride_b,
        _x_stride_h,
        _x_stride_s,
        _x_stride_d,
    ) = _x_meta(x)

    (
        cos_mode,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    ) = _table_meta(cos, b, h, s, d)
    (
        sin_mode,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    ) = _table_meta(sin, b, h, s, d)

    assert cos_mode == sin_mode

    dx = torch.empty_like(x)

    block = 256
    grid = (triton.cdiv(n, block),)
    _rope_backward_dx_kernel[grid](
        dy,
        x,
        cos,
        sin,
        dx,
        N=n,
        D=d,
        S=s,
        H=h,
        dy_stride_b=dy.stride(-4) if dy.dim() >= 4 else (dy.stride(0) if dy.dim() == 3 else 0),
        dy_stride_h=dy.stride(-3) if dy.dim() >= 4 else 0,
        dy_stride_s=dy.stride(-2),
        dy_stride_d=dy.stride(-1),
        dx_stride_b=dx.stride(-4) if dx.dim() >= 4 else (dx.stride(0) if dx.dim() == 3 else 0),
        dx_stride_h=dx.stride(-3) if dx.dim() >= 4 else 0,
        dx_stride_s=dx.stride(-2),
        dx_stride_d=dx.stride(-1),
        cos_stride_b=cos_stride_b,
        cos_stride_h=cos_stride_h,
        cos_stride_s=cos_stride_s,
        cos_stride_d=cos_stride_d,
        sin_stride_b=sin_stride_b,
        sin_stride_h=sin_stride_h,
        sin_stride_s=sin_stride_s,
        sin_stride_d=sin_stride_d,
        COS_MODE=cos_mode,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return dx


def kernel_function(x, cos, sin, upstream):
    """
    Fused forward+backward RoPE step required by the evaluator.

    Returns:
        cat([
            rope_forward(x, cos, sin).float().reshape(-1),
            grad_x.float().reshape(-1),
        ])

    All numerical work is performed in one Triton kernel.
    """
    assert x.is_cuda and cos.is_cuda and sin.is_cuda and upstream.is_cuda
    assert cos.numel() == sin.numel()

    (
        n,
        b,
        h,
        s,
        d,
        x_stride_b,
        x_stride_h,
        x_stride_s,
        x_stride_d,
    ) = _x_meta(x)

    (
        cos_mode,
        cos_stride_b,
        cos_stride_h,
        cos_stride_s,
        cos_stride_d,
    ) = _table_meta(cos, b, h, s, d)
    (
        sin_mode,
        sin_stride_b,
        sin_stride_h,
        sin_stride_s,
        sin_stride_d,
    ) = _table_meta(sin, b, h, s, d)

    assert cos_mode == sin_mode

    out = torch.empty((2 * n,), device=x.device, dtype=torch.float32)

    block = 256
    grid = (triton.cdiv(n, block),)
    _rope_pair_flat_kernel[grid](
        x,
        cos,
        sin,
        upstream,
        out,
        N=n,
        D=d,
        S=s,
        H=h,
        x_stride_b=x_stride_b,
        x_stride_h=x_stride_h,
        x_stride_s=x_stride_s,
        x_stride_d=x_stride_d,
        dy_stride_b=upstream.stride(-4)
        if upstream.dim() >= 4
        else (upstream.stride(0) if upstream.dim() == 3 else 0),
        dy_stride_h=upstream.stride(-3) if upstream.dim() >= 4 else 0,
        dy_stride_s=upstream.stride(-2),
        dy_stride_d=upstream.stride(-1),
        cos_stride_b=cos_stride_b,
        cos_stride_h=cos_stride_h,
        cos_stride_s=cos_stride_s,
        cos_stride_d=cos_stride_d,
        sin_stride_b=sin_stride_b,
        sin_stride_h=sin_stride_h,
        sin_stride_s=sin_stride_s,
        sin_stride_d=sin_stride_d,
        COS_MODE=cos_mode,
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
