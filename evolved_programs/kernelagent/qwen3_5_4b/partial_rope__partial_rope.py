"""adbench-tools artifact: task qwen3_5_4b.partial_rope.partial_rope, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the ADBench "partial_rope" forward + backward pair.
#
# Fixes:
#   - The task contract requires only the gradient w.r.t. x (`dx`).  Gradients
#     for cos/sin must not be returned.
#   - kernel_function now returns concat(forward.flatten(), dx.flatten()).
#   - The fused kernel preserves fusion of forward + dx backward in one pass.
#   - For bf16 paths, intermediate/product and final flat-buffer values are
#     rounded to bf16 where needed to match eager PyTorch bf16 behavior.

import torch
import triton
import triton.language as tl


def _pad4_shape_stride(t):
    """Right-align a tensor's shape/stride into a 4D logical layout."""
    shape = list(t.shape)
    stride = list(t.stride())
    if len(shape) > 4:
        raise AssertionError("This implementation supports tensors with rank <= 4")
    pad = 4 - len(shape)
    return [1] * pad + shape, [0] * pad + stride


@triton.jit
def _partial_rope_forward_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    # padded logical x shape
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    XD: tl.constexpr,
    # input x strides
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    # cos/sin padded shape
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    # cos strides
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    # sin strides
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    # output y strides
    ys0: tl.constexpr,
    ys1: tl.constexpr,
    ys2: tl.constexpr,
    ys3: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    d = offs % XD
    tmp = offs // XD
    i2 = tmp % X2
    tmp = tmp // X2
    i1 = tmp % X1
    i0 = tmp // X1

    rot_mask = mask & (d < ROTARY_DIM)

    x_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + d * xs3
    y_off = i0 * ys0 + i1 * ys1 + i2 * ys2 + d * ys3

    pair_d = tl.where(d < HALF, d + HALF, d - HALF)
    x_pair_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + pair_d * xs3

    c0 = tl.where(C0 == 1, 0, i0)
    c1 = tl.where(C1 == 1, 0, i1)
    c2 = tl.where(C2 == 1, 0, i2)

    c_off = c0 * cs0 + c1 * cs1 + c2 * cs2 + d * cs3
    s_off = c0 * ss0 + c1 * ss1 + c2 * ss2 + d * ss3

    x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_pair_off, mask=rot_mask, other=0.0).to(tl.float32)
    cos_val = tl.load(cos_ptr + c_off, mask=rot_mask, other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s_off, mask=rot_mask, other=0.0).to(tl.float32)

    rotated_x = tl.where(d < HALF, -x_pair, x_pair)

    prod0 = x_val * cos_val
    prod1 = rotated_x * sin_val
    if ROUND_BF16:
        prod0 = prod0.to(tl.bfloat16).to(tl.float32)
        prod1 = prod1.to(tl.bfloat16).to(tl.float32)

    y_rot = prod0 + prod1
    y_val = tl.where(d < ROTARY_DIM, y_rot, x_val)

    tl.store(y_ptr + y_off, y_val, mask=mask)


@triton.jit
def _partial_rope_backward_dx_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    dy_ptr,
    dx_ptr,
    n_elements: tl.constexpr,
    # padded logical x shape
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    XD: tl.constexpr,
    # input x strides
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    # upstream dy strides
    dys0: tl.constexpr,
    dys1: tl.constexpr,
    dys2: tl.constexpr,
    dys3: tl.constexpr,
    # output dx strides
    dxs0: tl.constexpr,
    dxs1: tl.constexpr,
    dxs2: tl.constexpr,
    dxs3: tl.constexpr,
    # cos/sin padded shape
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    # cos input strides
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    # sin input strides
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    d = offs % XD
    tmp = offs // XD
    i2 = tmp % X2
    tmp = tmp // X2
    i1 = tmp % X1
    i0 = tmp // X1

    rot_mask = mask & (d < ROTARY_DIM)

    dy_off = i0 * dys0 + i1 * dys1 + i2 * dys2 + d * dys3
    dx_off = i0 * dxs0 + i1 * dxs1 + i2 * dxs2 + d * dxs3

    pair_d = tl.where(d < HALF, d + HALF, d - HALF)
    dy_pair_off = i0 * dys0 + i1 * dys1 + i2 * dys2 + pair_d * dys3

    c0 = tl.where(C0 == 1, 0, i0)
    c1 = tl.where(C1 == 1, 0, i1)
    c2 = tl.where(C2 == 1, 0, i2)

    c_off = c0 * cs0 + c1 * cs1 + c2 * cs2 + d * cs3
    s_pair_off = c0 * ss0 + c1 * ss1 + c2 * ss2 + pair_d * ss3

    dy_val = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dy_pair = tl.load(dy_ptr + dy_pair_off, mask=rot_mask, other=0.0).to(tl.float32)
    cos_val = tl.load(cos_ptr + c_off, mask=rot_mask, other=0.0).to(tl.float32)
    sin_pair = tl.load(sin_ptr + s_pair_off, mask=rot_mask, other=0.0).to(tl.float32)

    prod0 = dy_val * cos_val
    prod1 = dy_pair * sin_pair
    if ROUND_BF16:
        prod0 = prod0.to(tl.bfloat16).to(tl.float32)
        prod1 = prod1.to(tl.bfloat16).to(tl.float32)

    sign = tl.where(d < HALF, 1.0, -1.0)
    dx_rot = prod0 + sign * prod1
    dx_val = tl.where(d < ROTARY_DIM, dx_rot, dy_val)

    tl.store(dx_ptr + dx_off, dx_val, mask=mask)


@triton.jit
def _fused_pair_flat_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    dy_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    base_dx: tl.constexpr,
    # padded logical x shape
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    XD: tl.constexpr,
    # input x strides
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    xs3: tl.constexpr,
    # upstream dy strides
    dys0: tl.constexpr,
    dys1: tl.constexpr,
    dys2: tl.constexpr,
    dys3: tl.constexpr,
    # cos/sin padded shape
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    # cos input strides
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    cs3: tl.constexpr,
    # sin input strides
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    ss3: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF: tl.constexpr,
    ROUND_BF16: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_FP16: tl.constexpr,
    DX_BF16: tl.constexpr,
    DX_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fused stages:
    #   1. partial RoPE forward, stored in flat out[0:n_elements]
    #   2. backward dx, stored in flat out[base_dx:base_dx+n_elements]
    #
    # cos/sin gradients are intentionally not computed because this task's
    # backward contract exposes only dx.
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    d = offs % XD
    tmp = offs // XD
    i2 = tmp % X2
    tmp = tmp // X2
    i1 = tmp % X1
    i0 = tmp // X1

    rot_mask = mask & (d < ROTARY_DIM)

    x_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + d * xs3
    dy_off = i0 * dys0 + i1 * dys1 + i2 * dys2 + d * dys3

    pair_d = tl.where(d < HALF, d + HALF, d - HALF)
    x_pair_off = i0 * xs0 + i1 * xs1 + i2 * xs2 + pair_d * xs3
    dy_pair_off = i0 * dys0 + i1 * dys1 + i2 * dys2 + pair_d * dys3

    c0 = tl.where(C0 == 1, 0, i0)
    c1 = tl.where(C1 == 1, 0, i1)
    c2 = tl.where(C2 == 1, 0, i2)

    c_off = c0 * cs0 + c1 * cs1 + c2 * cs2 + d * cs3
    s_off = c0 * ss0 + c1 * ss1 + c2 * ss2 + d * ss3
    s_pair_off = c0 * ss0 + c1 * ss1 + c2 * ss2 + pair_d * ss3

    x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
    x_pair = tl.load(x_ptr + x_pair_off, mask=rot_mask, other=0.0).to(tl.float32)
    dy_val = tl.load(dy_ptr + dy_off, mask=mask, other=0.0).to(tl.float32)
    dy_pair = tl.load(dy_ptr + dy_pair_off, mask=rot_mask, other=0.0).to(tl.float32)

    cos_val = tl.load(cos_ptr + c_off, mask=rot_mask, other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + s_off, mask=rot_mask, other=0.0).to(tl.float32)
    sin_pair = tl.load(sin_ptr + s_pair_off, mask=rot_mask, other=0.0).to(tl.float32)

    # Forward output.
    rotated_x = tl.where(d < HALF, -x_pair, x_pair)
    f_prod0 = x_val * cos_val
    f_prod1 = rotated_x * sin_val
    if ROUND_BF16:
        f_prod0 = f_prod0.to(tl.bfloat16).to(tl.float32)
        f_prod1 = f_prod1.to(tl.bfloat16).to(tl.float32)

    y_rot = f_prod0 + f_prod1
    y_val = tl.where(d < ROTARY_DIM, y_rot, x_val)

    # The flat result is fp32, but expected values are forward_output.float().
    # Therefore explicitly round to the actual forward output dtype first.
    if OUT_BF16:
        y_val = y_val.to(tl.bfloat16).to(tl.float32)
    elif OUT_FP16:
        y_val = y_val.to(tl.float16).to(tl.float32)

    tl.store(out_ptr + offs, y_val, mask=mask)

    # Backward dx.
    b_prod0 = dy_val * cos_val
    b_prod1 = dy_pair * sin_pair
    if ROUND_BF16:
        b_prod0 = b_prod0.to(tl.bfloat16).to(tl.float32)
        b_prod1 = b_prod1.to(tl.bfloat16).to(tl.float32)

    sign = tl.where(d < HALF, 1.0, -1.0)
    dx_rot = b_prod0 + sign * b_prod1
    dx_val = tl.where(d < ROTARY_DIM, dx_rot, dy_val)

    # The flat result is fp32, but expected values are dx.float().
    if DX_BF16:
        dx_val = dx_val.to(tl.bfloat16).to(tl.float32)
    elif DX_FP16:
        dx_val = dx_val.to(tl.float16).to(tl.float32)

    tl.store(out_ptr + base_dx + offs, dx_val, mask=mask)


def _common_meta(x, cos, sin):
    assert x.is_cuda and cos.is_cuda and sin.is_cuda
    assert cos.shape == sin.shape
    assert x.shape[-1] >= cos.shape[-1]
    assert cos.shape[-1] % 2 == 0

    xs, xst = _pad4_shape_stride(x)
    cs, cst = _pad4_shape_stride(cos)
    _, sst = _pad4_shape_stride(sin)
    return xs, xst, cs, cst, sst


def partial_rope_forward_with_saved(x, cos, sin):
    """
    Forward partial RoPE.

    Applies rotary embedding to the first cos.shape[-1] channels of x and
    leaves the remaining channels unchanged.
    """
    xs, xst, cs, cst, sst = _common_meta(x, cos, sin)

    out_dtype = torch.promote_types(torch.promote_types(x.dtype, cos.dtype), sin.dtype)
    y = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    _, yst = _pad4_shape_stride(y)

    n = x.numel()
    block = 256
    grid = (triton.cdiv(n, block),)

    round_bf16 = x.dtype is torch.bfloat16 and cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16

    _partial_rope_forward_kernel[grid](
        x,
        cos,
        sin,
        y,
        n,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xst[0],
        xst[1],
        xst[2],
        xst[3],
        cs[0],
        cs[1],
        cs[2],
        cs[3],
        cst[0],
        cst[1],
        cst[2],
        cst[3],
        sst[0],
        sst[1],
        sst[2],
        sst[3],
        yst[0],
        yst[1],
        yst[2],
        yst[3],
        ROTARY_DIM=cs[3],
        HALF=cs[3] // 2,
        ROUND_BF16=round_bf16,
        BLOCK_SIZE=block,
    )
    return y, (x, cos, sin)


def partial_rope_backward_from_saved(upstream, saved):
    """
    Backward for partial RoPE.

    Task contract requires only gradient w.r.t. x, so this returns dx only.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, cos, sin = saved

    xs, xst, cs, cst, sst = _common_meta(x, cos, sin)
    _, dyst = _pad4_shape_stride(dy)

    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _, dxst = _pad4_shape_stride(dx)

    n = x.numel()
    block = 256
    grid = (triton.cdiv(n, block),)

    round_bf16 = dy.dtype is torch.bfloat16 and cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16

    _partial_rope_backward_dx_kernel[grid](
        x,
        cos,
        sin,
        dy,
        dx,
        n,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xst[0],
        xst[1],
        xst[2],
        xst[3],
        dyst[0],
        dyst[1],
        dyst[2],
        dyst[3],
        dxst[0],
        dxst[1],
        dxst[2],
        dxst[3],
        cs[0],
        cs[1],
        cs[2],
        cs[3],
        cst[0],
        cst[1],
        cst[2],
        cst[3],
        sst[0],
        sst[1],
        sst[2],
        sst[3],
        ROTARY_DIM=cs[3],
        HALF=cs[3] // 2,
        ROUND_BF16=round_bf16,
        BLOCK_SIZE=block,
    )
    return dx


def kernel_function(x, cos, sin, *upstream):
    """
    Return a single flat fp32 tensor containing:
        concat(forward_output.float().flatten(), dx.float().flatten())

    This launches one fused Triton kernel for the forward and dx backward
    computations.  No cos/sin gradients are produced because the task contract
    requires only dx.
    """
    dy = upstream[0] if len(upstream) == 1 else upstream[0][0]

    xs, xst, cs, cst, sst = _common_meta(x, cos, sin)
    _, dyst = _pad4_shape_stride(dy)

    n_x = x.numel()
    base_dx = n_x
    total = n_x + n_x

    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    out_dtype = torch.promote_types(torch.promote_types(x.dtype, cos.dtype), sin.dtype)

    round_bf16 = x.dtype is torch.bfloat16 and cos.dtype is torch.bfloat16 and sin.dtype is torch.bfloat16

    block = 256
    grid = (triton.cdiv(n_x, block),)

    _fused_pair_flat_kernel[grid](
        x,
        cos,
        sin,
        dy,
        out,
        n_x,
        base_dx,
        xs[0],
        xs[1],
        xs[2],
        xs[3],
        xst[0],
        xst[1],
        xst[2],
        xst[3],
        dyst[0],
        dyst[1],
        dyst[2],
        dyst[3],
        cs[0],
        cs[1],
        cs[2],
        cs[3],
        cst[0],
        cst[1],
        cst[2],
        cst[3],
        sst[0],
        sst[1],
        sst[2],
        sst[3],
        ROTARY_DIM=cs[3],
        HALF=cs[3] // 2,
        ROUND_BF16=round_bf16,
        OUT_BF16=out_dtype is torch.bfloat16,
        OUT_FP16=out_dtype is torch.float16,
        DX_BF16=x.dtype is torch.bfloat16,
        DX_FP16=x.dtype is torch.float16,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'partial_rope', 'arguments': ['x', 'cos', 'sin'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx'], 'gradient_return_order': ['dx', 'None', 'None'], 'scalars': [], 'deployment_entry': 'partial_rope_deployment', 'allowed_primitives': []}


class PartialRopeFunction(torch.autograd.Function):
    """Static autograd wiring for `partial_rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        y, saved = partial_rope_forward_with_saved(x, cos, sin)
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
        dx = partial_rope_backward_from_saved(
            dy, saved
        )
        if isinstance(dx, (tuple, list)):
            dx, = dx
        # One value per forward argument, `None` for every inactive one.
        return dx, None, None


def partial_rope_deployment(x, cos, sin):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = PartialRopeFunction.apply(x, cos, sin)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class PartialRopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return partial_rope_deployment(x, cos, sin)


DEPLOYMENT_ENTRY = "partial_rope_deployment"
