"""adbench-tools artifact: task convnextv2_tiny.spatial_conv2d.convnextv2.embeddings.patch_embeddings, produced by kernelagent.

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


def _pair2(v):
    if isinstance(v, int):
        return int(v), int(v)
    return int(v[0]), int(v[1])


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


@triton.jit
def _round_like_ptr(v, ptr):
    if ptr.dtype.element_ty == tl.bfloat16:
        return v.to(tl.bfloat16).to(tl.float32)
    elif ptr.dtype.element_ty == tl.float16:
        return v.to(tl.float16).to(tl.float32)
    else:
        return v.to(tl.float32)


@triton.jit
def _conv2d_forward_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    flat_ptr,
    TOTAL: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OC: tl.constexpr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    W_SO: tl.constexpr,
    W_SCI: tl.constexpr,
    W_SKH: tl.constexpr,
    W_SKW: tl.constexpr,
    B_S: tl.constexpr,
    FLAT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    ow = offs % OW
    t = offs // OW
    oh = t % OH
    t = t // OH
    oc = t % OC
    n = t // OC

    acc = tl.load(b_ptr + oc * B_S, mask=mask, other=0.0).to(tl.float32)

    group = oc // OPG
    c_base = group * CPG

    for ci in tl.static_range(0, CPG):
        ic = c_base + ci
        for kh in tl.static_range(0, KH):
            ih = oh * STRIDE_H + kh - PAD_H
            vh = (ih >= 0) & (ih < H)
            ihs = tl.where(vh, ih, 0)
            for kw in tl.static_range(0, KW):
                iw = ow * STRIDE_W + kw - PAD_W
                vw = (iw >= 0) & (iw < W)
                iws = tl.where(vw, iw, 0)

                x_idx = n * X_SN + ic * X_SC + ihs * X_SH + iws * X_SW
                w_idx = oc * W_SO + ci * W_SCI + kh * W_SKH + kw * W_SKW

                xv = tl.load(x_ptr + x_idx, mask=mask & vh & vw, other=0.0).to(tl.float32)
                wv = tl.load(w_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
                acc += xv * wv

    tl.store(flat_ptr + FLAT_OFFSET + offs, _round_like_ptr(acc, x_ptr), mask=mask)


@triton.jit
def _conv2d_dx_nonoverlap_kernel(
    dout_ptr,
    w_ptr,
    flat_ptr,
    TOTAL_X: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    D_SN: tl.constexpr,
    D_SO: tl.constexpr,
    D_SH: tl.constexpr,
    D_SW: tl.constexpr,
    W_SO: tl.constexpr,
    W_SCI: tl.constexpr,
    W_SKH: tl.constexpr,
    W_SKW: tl.constexpr,
    FLAT_OFFSET: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_x = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    mask_x = offs_x < TOTAL_X

    iw = offs_x % W
    t = offs_x // W
    ih = t % H
    t = t // H
    c = t % C
    n = t // C

    group = c // CPG
    ci = c - group * CPG

    th = ih + PAD_H
    tw = iw + PAD_W

    oh = th // STRIDE_H
    ow = tw // STRIDE_W
    kh = th - oh * STRIDE_H
    kw = tw - ow * STRIDE_W

    valid = (
        mask_x
        & (oh >= 0)
        & (oh < OH)
        & (ow >= 0)
        & (ow < OW)
        & (kh >= 0)
        & (kh < KH)
        & (kw >= 0)
        & (kw < KW)
    )

    ohs = tl.where(valid, oh, 0)
    ows = tl.where(valid, ow, 0)
    khs = tl.where(valid, kh, 0)
    kws = tl.where(valid, kw, 0)

    oc_offsets = tl.arange(0, BLOCK_OC)
    oc = group[:, None] * OPG + oc_offsets[None, :]
    oc_mask = oc_offsets[None, :] < OPG

    dout_idx = n[:, None] * D_SN + oc * D_SO + ohs[:, None] * D_SH + ows[:, None] * D_SW
    w_idx = oc * W_SO + ci[:, None] * W_SCI + khs[:, None] * W_SKH + kws[:, None] * W_SKW

    vals = tl.load(dout_ptr + dout_idx, mask=valid[:, None] & oc_mask, other=0.0).to(tl.float32) * tl.load(
        w_ptr + w_idx, mask=valid[:, None] & oc_mask, other=0.0
    ).to(tl.float32)

    acc = tl.sum(vals, axis=1)
    tl.store(flat_ptr + FLAT_OFFSET + offs_x, _round_like_ptr(acc, dout_ptr), mask=mask_x)


@triton.jit
def _conv2d_dx_general_kernel(
    dout_ptr,
    w_ptr,
    flat_ptr,
    TOTAL_X: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    D_SN: tl.constexpr,
    D_SO: tl.constexpr,
    D_SH: tl.constexpr,
    D_SW: tl.constexpr,
    W_SO: tl.constexpr,
    W_SCI: tl.constexpr,
    W_SKH: tl.constexpr,
    W_SKW: tl.constexpr,
    FLAT_OFFSET: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_x = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    mask_x = offs_x < TOTAL_X

    iw = offs_x % W
    t = offs_x // W
    ih = t % H
    t = t // H
    c = t % C
    n = t // C

    group = c // CPG
    ci = c - group * CPG

    oc_offsets = tl.arange(0, BLOCK_OC)
    oc = group[:, None] * OPG + oc_offsets[None, :]
    oc_mask = oc_offsets[None, :] < OPG

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        oh_num = ih + PAD_H - kh
        ok_h = oh_num >= 0
        oh = oh_num // STRIDE_H
        ok_h = ok_h & (oh * STRIDE_H == oh_num) & (oh >= 0) & (oh < OH)
        ohs = tl.where(ok_h, oh, 0)

        for kw in tl.static_range(0, KW):
            ow_num = iw + PAD_W - kw
            ok_w = ow_num >= 0
            ow = ow_num // STRIDE_W
            ok_w = ok_w & (ow * STRIDE_W == ow_num) & (ow >= 0) & (ow < OW)
            ows = tl.where(ok_w, ow, 0)

            valid = mask_x & ok_h & ok_w

            dout_idx = n[:, None] * D_SN + oc * D_SO + ohs[:, None] * D_SH + ows[:, None] * D_SW
            w_idx = oc * W_SO + ci[:, None] * W_SCI + kh * W_SKH + kw * W_SKW

            vals = tl.load(dout_ptr + dout_idx, mask=valid[:, None] & oc_mask, other=0.0).to(tl.float32) * tl.load(
                w_ptr + w_idx, mask=valid[:, None] & oc_mask, other=0.0
            ).to(tl.float32)
            acc += tl.sum(vals, axis=1)

    tl.store(flat_ptr + FLAT_OFFSET + offs_x, _round_like_ptr(acc, dout_ptr), mask=mask_x)


@triton.jit
def _conv2d_dw_db_fused_flat_kernel(
    x_ptr,
    dout_ptr,
    dw_ptr,
    db_ptr,
    flat_ptr,
    TOTAL_W: tl.constexpr,
    TOTAL_POS: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    D_SN: tl.constexpr,
    D_SO: tl.constexpr,
    D_SH: tl.constexpr,
    D_SW: tl.constexpr,
    DW_FLAT_OFFSET: tl.constexpr,
    DB_FLAT_OFFSET: tl.constexpr,
    BLOCK_POS: tl.constexpr,
):
    w_linear = tl.program_id(0)

    kw = w_linear % KW
    t = w_linear // KW
    kh = t % KH
    t = t // KH
    ci = t % CPG
    oc = t // CPG

    group = oc // OPG
    c = group * CPG + ci

    offs = tl.arange(0, BLOCK_POS)
    acc_dw = tl.zeros((BLOCK_POS,), dtype=tl.float32)
    acc_db = tl.zeros((BLOCK_POS,), dtype=tl.float32)

    for start in tl.range(0, TOTAL_POS, BLOCK_POS):
        pos = start + offs
        mask = pos < TOTAL_POS

        ow = pos % OW
        tt = pos // OW
        oh = tt % OH
        n = tt // OH

        ih = oh * STRIDE_H + kh - PAD_H
        iw = ow * STRIDE_W + kw - PAD_W

        valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        ihs = tl.where(valid, ih, 0)
        iws = tl.where(valid, iw, 0)

        x_idx = n * X_SN + c * X_SC + ihs * X_SH + iws * X_SW
        dout_idx = n * D_SN + oc * D_SO + oh * D_SH + ow * D_SW

        xv = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
        dv = tl.load(dout_ptr + dout_idx, mask=mask, other=0.0).to(tl.float32)

        acc_dw += xv * dv
        acc_db += dv

    result_dw = tl.sum(acc_dw, axis=0)
    tl.store(flat_ptr + DW_FLAT_OFFSET + w_linear, _round_like_ptr(result_dw, dw_ptr), mask=w_linear < TOTAL_W)

    result_db = tl.sum(acc_db, axis=0)
    sel = (ci == 0) & (kh == 0) & (kw == 0)
    tl.store(flat_ptr + DB_FLAT_OFFSET + oc, _round_like_ptr(result_db, db_ptr), mask=sel)


@triton.jit
def _conv2d_dw_db_fused_separate_kernel(
    x_ptr,
    dout_ptr,
    dw_out_ptr,
    db_out_ptr,
    TOTAL_W: tl.constexpr,
    TOTAL_POS: tl.constexpr,
    CPG: tl.constexpr,
    OPG: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    D_SN: tl.constexpr,
    D_SO: tl.constexpr,
    D_SH: tl.constexpr,
    D_SW: tl.constexpr,
    BLOCK_POS: tl.constexpr,
):
    w_linear = tl.program_id(0)

    kw = w_linear % KW
    t = w_linear // KW
    kh = t % KH
    t = t // KH
    ci = t % CPG
    oc = t // CPG

    group = oc // OPG
    c = group * CPG + ci

    offs = tl.arange(0, BLOCK_POS)
    acc_dw = tl.zeros((BLOCK_POS,), dtype=tl.float32)
    acc_db = tl.zeros((BLOCK_POS,), dtype=tl.float32)

    for start in tl.range(0, TOTAL_POS, BLOCK_POS):
        pos = start + offs
        mask = pos < TOTAL_POS

        ow = pos % OW
        tt = pos // OW
        oh = tt % OH
        n = tt // OH

        ih = oh * STRIDE_H + kh - PAD_H
        iw = ow * STRIDE_W + kw - PAD_W

        valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

        ihs = tl.where(valid, ih, 0)
        iws = tl.where(valid, iw, 0)

        x_idx = n * X_SN + c * X_SC + ihs * X_SH + iws * X_SW
        dout_idx = n * D_SN + oc * D_SO + oh * D_SH + ow * D_SW

        xv = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
        dv = tl.load(dout_ptr + dout_idx, mask=mask, other=0.0).to(tl.float32)

        acc_dw += xv * dv
        acc_db += dv

    result_dw = tl.sum(acc_dw, axis=0)
    tl.store(dw_out_ptr + w_linear, _round_like_ptr(result_dw, dw_out_ptr), mask=w_linear < TOTAL_W)

    result_db = tl.sum(acc_db, axis=0)
    sel = (ci == 0) & (kh == 0) & (kw == 0)
    tl.store(db_out_ptr + oc, _round_like_ptr(result_db, db_out_ptr), mask=sel)


def spatial_conv2d_forward_with_saved(x, weight, bias, stride, padding, groups):
    sh, sw = _pair2(stride)
    ph, pw = _pair2(padding)

    n, c, h, w = x.shape
    oc, cpg, kh, kw = weight.shape
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    opg = oc // int(groups)

    out = torch.empty((n, oc, oh, ow), device=x.device, dtype=x.dtype)

    total = n * oc * oh * ow
    block = 256
    _conv2d_forward_kernel[(triton.cdiv(total, block),)](
        x,
        weight,
        bias,
        out,
        TOTAL=total,
        CPG=cpg,
        OPG=opg,
        KH=kh,
        KW=kw,
        OH=oh,
        OW=ow,
        STRIDE_H=sh,
        STRIDE_W=sw,
        PAD_H=ph,
        PAD_W=pw,
        H=h,
        W=w,
        OC=oc,
        X_SN=x.stride(0),
        X_SC=x.stride(1),
        X_SH=x.stride(2),
        X_SW=x.stride(3),
        W_SO=weight.stride(0),
        W_SCI=weight.stride(1),
        W_SKH=weight.stride(2),
        W_SKW=weight.stride(3),
        B_S=bias.stride(0),
        FLAT_OFFSET=0,
        BLOCK=block,
        num_warps=4,
    )

    saved = (x, weight, bias, stride, padding, groups)
    return out, saved


def spatial_conv2d_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    x, weight, bias, stride, padding, groups = saved

    sh, sw = _pair2(stride)
    ph, pw = _pair2(padding)

    n, c, h, w = x.shape
    oc, cpg, kh, kw = weight.shape
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    opg = oc // int(groups)

    dx = torch.empty((n, c, h, w), device=x.device, dtype=x.dtype)
    dw = torch.empty((oc, cpg, kh, kw), device=weight.device, dtype=weight.dtype)
    db = torch.empty((bias.shape[0],), device=bias.device, dtype=bias.dtype)

    x_num = n * c * h * w
    w_num = oc * cpg * kh * kw

    dx_block_x = 16
    dx_block_oc = _next_power_of_2(opg)
    dx_grid = (triton.cdiv(x_num, dx_block_x),)

    if sh >= kh and sw >= kw:
        _conv2d_dx_nonoverlap_kernel[dx_grid](
            dout,
            weight,
            dx,
            TOTAL_X=x_num,
            C=c,
            H=h,
            W=w,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            OH=oh,
            OW=ow,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            D_SN=dout.stride(0),
            D_SO=dout.stride(1),
            D_SH=dout.stride(2),
            D_SW=dout.stride(3),
            W_SO=weight.stride(0),
            W_SCI=weight.stride(1),
            W_SKH=weight.stride(2),
            W_SKW=weight.stride(3),
            FLAT_OFFSET=0,
            BLOCK_X=dx_block_x,
            BLOCK_OC=dx_block_oc,
            num_warps=4,
        )
    else:
        _conv2d_dx_general_kernel[dx_grid](
            dout,
            weight,
            dx,
            TOTAL_X=x_num,
            C=c,
            H=h,
            W=w,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            OH=oh,
            OW=ow,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            D_SN=dout.stride(0),
            D_SO=dout.stride(1),
            D_SH=dout.stride(2),
            D_SW=dout.stride(3),
            W_SO=weight.stride(0),
            W_SCI=weight.stride(1),
            W_SKH=weight.stride(2),
            W_SKW=weight.stride(3),
            FLAT_OFFSET=0,
            BLOCK_X=dx_block_x,
            BLOCK_OC=dx_block_oc,
            num_warps=4,
        )

    block_pos = 1024
    _conv2d_dw_db_fused_separate_kernel[(w_num,)](
        x,
        dout,
        dw,
        db,
        TOTAL_W=w_num,
        TOTAL_POS=n * oh * ow,
        CPG=cpg,
        OPG=opg,
        KH=kh,
        KW=kw,
        H=h,
        W=w,
        OH=oh,
        OW=ow,
        STRIDE_H=sh,
        STRIDE_W=sw,
        PAD_H=ph,
        PAD_W=pw,
        X_SN=x.stride(0),
        X_SC=x.stride(1),
        X_SH=x.stride(2),
        X_SW=x.stride(3),
        D_SN=dout.stride(0),
        D_SO=dout.stride(1),
        D_SH=dout.stride(2),
        D_SW=dout.stride(3),
        BLOCK_POS=block_pos,
        num_warps=8,
    )

    return dx, dw, db


def kernel_function(x, weight, bias, stride, padding, groups, upstream):
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    sh, sw = _pair2(stride)
    ph, pw = _pair2(padding)

    n, c, h, w = x.shape
    oc, cpg, kh, kw = weight.shape
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    opg = oc // int(groups)

    out_num = n * oc * oh * ow
    x_num = x.numel()
    w_num = weight.numel()
    b_num = bias.numel()
    total = out_num + x_num + w_num + b_num

    result = torch.empty((total,), device=x.device, dtype=torch.float32)

    # Forward, dX, and fused dW+dBias are separate launches because dW/dBias are
    # reductions over N/OH/OW while forward/dX are pointwise over different output
    # domains. dW and dBias remain fused to avoid an extra reduction pass.
    f_block = 256
    _conv2d_forward_kernel[(triton.cdiv(out_num, f_block),)](
        x,
        weight,
        bias,
        result,
        TOTAL=out_num,
        CPG=cpg,
        OPG=opg,
        KH=kh,
        KW=kw,
        OH=oh,
        OW=ow,
        STRIDE_H=sh,
        STRIDE_W=sw,
        PAD_H=ph,
        PAD_W=pw,
        H=h,
        W=w,
        OC=oc,
        X_SN=x.stride(0),
        X_SC=x.stride(1),
        X_SH=x.stride(2),
        X_SW=x.stride(3),
        W_SO=weight.stride(0),
        W_SCI=weight.stride(1),
        W_SKH=weight.stride(2),
        W_SKW=weight.stride(3),
        B_S=bias.stride(0),
        FLAT_OFFSET=0,
        BLOCK=f_block,
        num_warps=4,
    )

    dx_block_x = 16
    dx_block_oc = _next_power_of_2(opg)
    dx_grid = (triton.cdiv(x_num, dx_block_x),)

    if sh >= kh and sw >= kw:
        _conv2d_dx_nonoverlap_kernel[dx_grid](
            dout,
            weight,
            result,
            TOTAL_X=x_num,
            C=c,
            H=h,
            W=w,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            OH=oh,
            OW=ow,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            D_SN=dout.stride(0),
            D_SO=dout.stride(1),
            D_SH=dout.stride(2),
            D_SW=dout.stride(3),
            W_SO=weight.stride(0),
            W_SCI=weight.stride(1),
            W_SKH=weight.stride(2),
            W_SKW=weight.stride(3),
            FLAT_OFFSET=out_num,
            BLOCK_X=dx_block_x,
            BLOCK_OC=dx_block_oc,
            num_warps=4,
        )
    else:
        _conv2d_dx_general_kernel[dx_grid](
            dout,
            weight,
            result,
            TOTAL_X=x_num,
            C=c,
            H=h,
            W=w,
            CPG=cpg,
            OPG=opg,
            KH=kh,
            KW=kw,
            OH=oh,
            OW=ow,
            STRIDE_H=sh,
            STRIDE_W=sw,
            PAD_H=ph,
            PAD_W=pw,
            D_SN=dout.stride(0),
            D_SO=dout.stride(1),
            D_SH=dout.stride(2),
            D_SW=dout.stride(3),
            W_SO=weight.stride(0),
            W_SCI=weight.stride(1),
            W_SKH=weight.stride(2),
            W_SKW=weight.stride(3),
            FLAT_OFFSET=out_num,
            BLOCK_X=dx_block_x,
            BLOCK_OC=dx_block_oc,
            num_warps=4,
        )

    block_pos = 1024
    _conv2d_dw_db_fused_flat_kernel[(w_num,)](
        x,
        dout,
        weight,
        bias,
        result,
        TOTAL_W=w_num,
        TOTAL_POS=n * oh * ow,
        CPG=cpg,
        OPG=opg,
        KH=kh,
        KW=kw,
        H=h,
        W=w,
        OH=oh,
        OW=ow,
        STRIDE_H=sh,
        STRIDE_W=sw,
        PAD_H=ph,
        PAD_W=pw,
        X_SN=x.stride(0),
        X_SC=x.stride(1),
        X_SH=x.stride(2),
        X_SW=x.stride(3),
        D_SN=dout.stride(0),
        D_SO=dout.stride(1),
        D_SH=dout.stride(2),
        D_SW=dout.stride(3),
        DW_FLAT_OFFSET=out_num + x_num,
        DB_FLAT_OFFSET=out_num + x_num + w_num,
        BLOCK_POS=block_pos,
        num_warps=8,
    )

    return result
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

    def __init__(self, weight, bias, stride=4, padding=0, groups=1):
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
