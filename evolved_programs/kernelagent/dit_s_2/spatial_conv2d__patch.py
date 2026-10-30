"""adbench-tools artifact: task dit_s_2.spatial_conv2d.patch, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation of a spatial Conv2d forward + backward pair.
#
# Fusion note:
#   The public kernel_function returns the evaluator-required flattened
#   concatenation of:
#       forward output, grad_x, grad_weight, grad_bias
#   without using torch.cat or PyTorch math.  The forward, input-gradient, and
#   weight/bias-gradient reductions have different reduction domains, so they
#   are launched as separate Triton kernels.  The weight-gradient kernel also
#   computes bias-gradient, fusing those two backward reductions.

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    OUT_OFFSET: tl.constexpr,
    N: tl.constexpr,
    CIN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    COUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    xs_n: tl.constexpr,
    xs_c: tl.constexpr,
    xs_h: tl.constexpr,
    xs_w: tl.constexpr,
    ws_o: tl.constexpr,
    ws_i: tl.constexpr,
    ws_h: tl.constexpr,
    ws_w: tl.constexpr,
    os_n: tl.constexpr,
    os_c: tl.constexpr,
    os_h: tl.constexpr,
    os_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    n = offs_m // (OH * OW)

    oc_group = offs_oc // COUT_PER_GROUP

    acc = tl.zeros((BLOCK_M, BLOCK_OC), dtype=tl.float32)

    for kk in range(CIN_PER_GROUP * KH * KW):
        ic_l = kk // (KH * KW)
        rem = kk - ic_l * KH * KW
        kh = rem // KW
        kw = rem - kh * KW

        ih = oh * SH + kh - PH
        iw = ow * SW + kw - PW

        ic_g = oc_group * CIN_PER_GROUP + ic_l

        x_ptrs = (
            x_ptr
            + n[:, None] * xs_n
            + ic_g[None, :] * xs_c
            + ih[:, None] * xs_h
            + iw[:, None] * xs_w
        )
        w_ptrs = (
            w_ptr
            + offs_oc[None, :] * ws_o
            + ic_l * ws_i
            + kh * ws_h
            + kw * ws_w
        )

        mask_x = (
            (offs_m[:, None] < N * OH * OW)
            & (offs_oc[None, :] < COUT)
            & (ih[:, None] >= 0)
            & (ih[:, None] < H)
            & (iw[:, None] >= 0)
            & (iw[:, None] < W)
        )

        xv = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)
        wv = tl.load(w_ptrs, mask=offs_oc[None, :] < COUT, other=0.0).to(tl.float32)
        acc += xv * wv

    if HAS_BIAS:
        bv = tl.load(b_ptr + offs_oc, mask=offs_oc < COUT, other=0.0).to(tl.float32)
        acc += bv[None, :]

    out_ptrs = (
        out_ptr
        + OUT_OFFSET
        + n[:, None] * os_n
        + offs_oc[None, :] * os_c
        + oh[:, None] * os_h
        + ow[:, None] * os_w
    )
    mask_o = (offs_m[:, None] < N * OH * OW) & (offs_oc[None, :] < COUT)
    tl.store(out_ptrs, acc, mask=mask_o)


@triton.jit
def _conv2d_dx_kernel(
    dy_ptr,
    w_ptr,
    dx_ptr,
    DX_OFFSET: tl.constexpr,
    N: tl.constexpr,
    CIN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    COUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    dys_n: tl.constexpr,
    dys_c: tl.constexpr,
    dys_h: tl.constexpr,
    dys_w: tl.constexpr,
    ws_o: tl.constexpr,
    ws_i: tl.constexpr,
    ws_h: tl.constexpr,
    ws_w: tl.constexpr,
    dxs_n: tl.constexpr,
    dxs_c: tl.constexpr,
    dxs_h: tl.constexpr,
    dxs_w: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_X + tl.arange(0, BLOCK_X)

    widx = offs % W
    hidx = (offs // W) % H
    cidx = (offs // (H * W)) % CIN
    nidx = offs // (CIN * H * W)

    group = cidx // CIN_PER_GROUP
    ic_l = cidx - group * CIN_PER_GROUP

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)

    for kh in range(KH):
        th = hidx + PH - kh
        valid_h = (th >= 0) & ((th % SH) == 0)
        oh = th // SH

        for kw in range(KW):
            tw = widx + PW - kw
            valid_w = (tw >= 0) & ((tw % SW) == 0)
            ow = tw // SW

            valid_spatial = (
                (offs < N * CIN * H * W)
                & valid_h
                & valid_w
                & (oh >= 0)
                & (oh < OH)
                & (ow >= 0)
                & (ow < OW)
            )

            for ocb in tl.range(0, COUT_PER_GROUP, BLOCK_OC):
                offs_oc_l = ocb + tl.arange(0, BLOCK_OC)
                oc = group[:, None] * COUT_PER_GROUP + offs_oc_l[None, :]

                dy_ptrs = (
                    dy_ptr
                    + nidx[:, None] * dys_n
                    + oc * dys_c
                    + oh[:, None] * dys_h
                    + ow[:, None] * dys_w
                )
                w_ptrs = (
                    w_ptr
                    + oc * ws_o
                    + ic_l[:, None] * ws_i
                    + kh * ws_h
                    + kw * ws_w
                )

                mask = (
                    valid_spatial[:, None]
                    & (offs_oc_l[None, :] < COUT_PER_GROUP)
                    & (oc < COUT)
                )

                dyv = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)
                wv = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(dyv * wv, axis=1)

    dx_ptrs = (
        dx_ptr
        + DX_OFFSET
        + nidx * dxs_n
        + cidx * dxs_c
        + hidx * dxs_h
        + widx * dxs_w
    )
    tl.store(dx_ptrs, acc, mask=offs < N * CIN * H * W)


@triton.jit
def _conv2d_dw_db_kernel(
    x_ptr,
    dy_ptr,
    dw_ptr,
    db_ptr,
    DW_OFFSET: tl.constexpr,
    DB_OFFSET: tl.constexpr,
    N: tl.constexpr,
    CIN: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    COUT: tl.constexpr,
    CIN_PER_GROUP: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    COUT_PER_GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    xs_n: tl.constexpr,
    xs_c: tl.constexpr,
    xs_h: tl.constexpr,
    xs_w: tl.constexpr,
    dys_n: tl.constexpr,
    dys_c: tl.constexpr,
    dys_h: tl.constexpr,
    dys_w: tl.constexpr,
    dws_o: tl.constexpr,
    dws_i: tl.constexpr,
    dws_h: tl.constexpr,
    dws_w: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)

    kw = pid % KW
    tmp0 = pid // KW
    kh = tmp0 % KH
    tmp1 = tmp0 // KH
    ic_l = tmp1 % CIN_PER_GROUP
    oc = tmp1 // CIN_PER_GROUP

    group = oc // COUT_PER_GROUP
    ic_g = group * CIN_PER_GROUP + ic_l

    offs = tl.arange(0, BLOCK_R)
    acc_w = tl.zeros((BLOCK_R,), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_R,), dtype=tl.float32)

    total_r = N * OH * OW

    for base in tl.range(0, total_r, BLOCK_R):
        r = base + offs
        ow = r % OW
        oh = (r // OW) % OH
        n = r // (OH * OW)

        ih = oh * SH + kh - PH
        iw = ow * SW + kw - PW

        valid = (
            (r < total_r)
            & (ih >= 0)
            & (ih < H)
            & (iw >= 0)
            & (iw < W)
            & (oc < COUT)
        )

        x_vals = tl.load(
            x_ptr + n * xs_n + ic_g * xs_c + ih * xs_h + iw * xs_w,
            mask=valid,
            other=0.0,
        ).to(tl.float32)

        dy_vals = tl.load(
            dy_ptr + n * dys_n + oc * dys_c + oh * dys_h + ow * dys_w,
            mask=(r < total_r) & (oc < COUT),
            other=0.0,
        ).to(tl.float32)

        acc_w += x_vals * dy_vals
        acc_b += dy_vals

    grad_w = tl.sum(acc_w, axis=0)
    grad_b = tl.sum(acc_b, axis=0)

    tl.store(
        dw_ptr
        + DW_OFFSET
        + oc * dws_o
        + ic_l * dws_i
        + kh * dws_h
        + kw * dws_w,
        grad_w,
        mask=pid < COUT * CIN_PER_GROUP * KH * KW,
    )

    if HAS_BIAS:
        first_weight_for_oc = (ic_l == 0) & (kh == 0) & (kw == 0)
        tl.store(db_ptr + DB_OFFSET + oc, grad_b, mask=first_weight_for_oc & (oc < COUT))


def _pair(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def spatial_conv2d_forward_with_saved(x, weight, bias, stride, padding, groups):
    """Forward Conv2d implemented in Triton; returns (output, saved)."""
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    n, cin, h, w = x.shape
    cout, cin_per_group, kh, kw = weight.shape
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    cout_per_group = cout // int(groups)

    y = torch.empty((n, cout, oh, ow), device=x.device, dtype=x.dtype)

    block_m = 16
    block_oc = 32
    grid = (triton.cdiv(n * oh * ow, block_m), triton.cdiv(cout, block_oc))

    _conv2d_forward_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        y,
        0,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        bias is not None,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_M=block_m,
        BLOCK_OC=block_oc,
        num_warps=4,
    )

    saved = (x, weight, bias, sh, sw, ph, pw, int(groups), oh, ow)
    return y, saved


def spatial_conv2d_backward_from_saved(upstream, saved):
    """Backward Conv2d implemented in Triton; returns gradients for tensor inputs."""
    if isinstance(upstream, tuple) or isinstance(upstream, list):
        dy = upstream[0]
    else:
        dy = upstream

    x, weight, bias, sh, sw, ph, pw, groups, oh, ow = saved

    n, cin, h, w = x.shape
    cout, cin_per_group, kh, kw = weight.shape
    cout_per_group = cout // int(groups)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias) if bias is not None else None

    block_x = 128
    block_oc = 32
    grid_dx = (triton.cdiv(n * cin * h * w, block_x),)

    _conv2d_dx_kernel[grid_dx](
        dy,
        weight,
        dx,
        0,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dy.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        dx.stride(3),
        BLOCK_X=block_x,
        BLOCK_OC=block_oc,
        num_warps=4,
    )

    block_r = 256
    grid_dw = (cout * cin_per_group * kh * kw,)

    _conv2d_dw_db_kernel[grid_dw](
        x,
        dy,
        dw,
        db if db is not None else dw,
        0,
        0,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        bias is not None,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dy.stride(3),
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        dw.stride(3),
        BLOCK_R=block_r,
        num_warps=8,
    )

    if bias is not None:
        return dx, dw, db
    return dx, dw


def kernel_function(x, weight, bias, stride, padding, groups, *upstream):
    """
    Evaluator-facing fused forward+backward step.

    Produces one flat float32 tensor containing:
        conv2d(x, weight, bias).reshape(-1),
        grad_x.reshape(-1),
        grad_weight.reshape(-1),
        grad_bias.reshape(-1)  [when bias exists]

    All numerical work, including the final flattened layout writes, is done by
    Triton kernels.  The Python wrapper only computes metadata, allocates the
    output buffer, and launches kernels.
    """
    dy = upstream[0] if len(upstream) == 1 else upstream[0][0]

    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    n, cin, h, w = x.shape
    cout, cin_per_group, kh, kw = weight.shape
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    cout_per_group = cout // int(groups)

    num_y = n * cout * oh * ow
    num_dx = n * cin * h * w
    num_dw = cout * cin_per_group * kh * kw
    num_db = cout if bias is not None else 0
    total = num_y + num_dx + num_dw + num_db

    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    # Forward output segment.
    block_m = 16
    block_oc = 32
    grid_fwd = (triton.cdiv(n * oh * ow, block_m), triton.cdiv(cout, block_oc))
    _conv2d_forward_kernel[grid_fwd](
        x,
        weight,
        bias if bias is not None else x,
        out,
        0,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        bias is not None,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        cout * oh * ow,
        oh * ow,
        ow,
        1,
        BLOCK_M=block_m,
        BLOCK_OC=block_oc,
        num_warps=4,
    )

    # grad_x segment.
    block_x = 128
    grid_dx = (triton.cdiv(n * cin * h * w, block_x),)
    _conv2d_dx_kernel[grid_dx](
        dy,
        weight,
        out,
        num_y,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dy.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        cin * h * w,
        h * w,
        w,
        1,
        BLOCK_X=block_x,
        BLOCK_OC=block_oc,
        num_warps=4,
    )

    # grad_weight and grad_bias segments.
    block_r = 256
    grid_dw = (cout * cin_per_group * kh * kw,)
    _conv2d_dw_db_kernel[grid_dw](
        x,
        dy,
        out,
        out,
        num_y + num_dx,
        num_y + num_dx + num_dw,
        n,
        cin,
        h,
        w,
        cout,
        cin_per_group,
        kh,
        kw,
        oh,
        ow,
        sh,
        sw,
        ph,
        pw,
        cout_per_group,
        bias is not None,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dy.stride(3),
        cin_per_group * kh * kw,
        kh * kw,
        kw,
        1,
        BLOCK_R=block_r,
        num_warps=8,
    )

    return out
# EVOLVE-BLOCK-END


# ── trusted primitive: vendor_gemm ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it.

def vendor_gemm(a, b):
    """Row-major 2-D GEMM through the vendor library (cuBLAS via ``torch.mm``).

    Two dimensions only, so the caller owns every reshape and the primitive
    cannot be handed a batched contraction it would silently loop. A
    ``.t()`` argument stays a metadata-only view -- cuBLAS consumes the
    transposed layout directly -- so ``vendor_gemm(x, w.t())`` costs one
    kernel and no copy.

    ``no_grad`` is not an optimisation: the pair differentiates itself, and a
    graph built here would be a second, hidden derivative.
    """
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"vendor_gemm is 2-D; got {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"vendor_gemm shape mismatch: {tuple(a.shape)} @ {tuple(b.shape)}"
        )
    with torch.no_grad():
        return torch.mm(a, b)


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'spatial_conv2d', 'arguments': ['x', 'weight', 'bias', 'stride', 'padding', 'groups'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias', 'None', 'None', 'None'], 'scalars': ['stride', 'padding', 'groups'], 'deployment_entry': 'spatial_conv2d_deployment', 'allowed_primitives': ['vendor_gemm']}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class SpatialConv2DFunction(torch.autograd.Function):
    """Static autograd wiring for `spatial_conv2d`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, groups):
        out, saved = spatial_conv2d_forward_with_saved(x, weight, bias, stride, padding, groups)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
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
        dx, dweight, dbias = spatial_conv2d_backward_from_saved(
            dout, saved, **_backward_scalars(spatial_conv2d_backward_from_saved, stride=ctx.stride, padding=ctx.padding, groups=ctx.groups)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias, None, None, None


def spatial_conv2d_deployment(x, weight, bias, stride, padding, groups):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        weight = weight.reshape(-1, weight.shape[-1])
    out = SpatialConv2DFunction.apply(x, weight, bias, stride, padding, groups)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class SpatialConv2DModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias, stride=2, padding=0, groups=1):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return spatial_conv2d_deployment(x, self.weight, self.bias, self.stride, self.padding, self.groups)


DEPLOYMENT_ENTRY = "spatial_conv2d_deployment"
