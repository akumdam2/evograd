import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _fwd_conv_silu_kernel(qkv, cw, y, BT: tl.constexpr, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tt = offs_t % T
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for j in tl.static_range(0, K):
        in_t = tt - (K - 1) + j
        in_row = offs_t - (K - 1) + j
        m = (offs_t[:, None] < BT) & (offs_c[None, :] < C) & (in_t[:, None] >= 0)
        xv = tl.load(qkv + in_row[:, None] * C + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wv = tl.load(cw + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += xv * wv[None, :]
    sig = 1.0 / (1.0 + tl.exp(-acc))
    out = acc * sig
    tl.store(y + offs_t[:, None] * C + offs_c[None, :], out, mask=(offs_t[:, None] < BT) & (offs_c[None, :] < C))


_fwd_conv_silu_kernel = triton.jit(_fwd_conv_silu_kernel)


def _fwd_repeat_qk_kernel(y, q, k, N: tl.constexpr, C: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = offs % D
    hv = (offs // D) % HV
    row = offs // (HV * D)
    h0 = hv // R
    qch = h0 * D + d
    kch = HK * D + h0 * D + d
    qv = tl.load(y + row * C + qch, mask=offs < N, other=0.0)
    kv = tl.load(y + row * C + kch, mask=offs < N, other=0.0)
    tl.store(q + offs, qv, mask=offs < N)
    tl.store(k + offs, kv, mask=offs < N)


_fwd_repeat_qk_kernel = triton.jit(_fwd_repeat_qk_kernel)


def _fwd_beta_g_kernel(b, a, alog, dt, beta, g, N: tl.constexpr, HV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = offs % HV
    mask = offs < N
    bv = tl.load(b + offs, mask=mask, other=0.0).to(tl.float32)
    av = tl.load(a + offs, mask=mask, other=0.0).to(tl.float32)
    al = tl.load(alog + h, mask=mask, other=0.0).to(tl.float32)
    dv = tl.load(dt + h, mask=mask, other=0.0).to(tl.float32)
    bet = 1.0 / (1.0 + tl.exp(-bv))
    u = av + dv
    sp = tl.where(u > 20.0, u, tl.log(1.0 + tl.exp(u)))
    gv = -tl.exp(al) * sp
    tl.store(beta + offs, bet, mask=mask)
    tl.store(g + offs, gv, mask=mask)


_fwd_beta_g_kernel = triton.jit(_fwd_beta_g_kernel)


def _bwd_gate_kernel(dbeta, dg, beta, g, a, alog, dt, db, da, pdt, pa, s_db0: tl.constexpr, s_db1: tl.constexpr, s_db2: tl.constexpr, s_dg0: tl.constexpr, s_dg1: tl.constexpr, s_dg2: tl.constexpr, BT: tl.constexpr, T: tl.constexpr, HV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    bidx = rows // T
    tidx = rows - bidx * T
    flat_offs = rows[:, None] * HV + hs[None, :]
    off_dbeta = bidx[:, None] * s_db0 + tidx[:, None] * s_db1 + hs[None, :] * s_db2
    off_dg = bidx[:, None] * s_dg0 + tidx[:, None] * s_dg1 + hs[None, :] * s_dg2
    mask = (rows[:, None] < BT) & (hs[None, :] < HV)
    bet = tl.load(beta + flat_offs, mask=mask, other=0.0).to(tl.float32)
    dbet = tl.load(dbeta + off_dbeta, mask=mask, other=0.0).to(tl.float32)
    dbv = dbet * bet * (1.0 - bet)
    tl.store(db + flat_offs, dbv, mask=mask)
    dgv = tl.load(dg + off_dg, mask=mask, other=0.0).to(tl.float32)
    av = tl.load(a + flat_offs, mask=mask, other=0.0).to(tl.float32)
    al = tl.load(alog + hs, mask=hs < HV, other=0.0).to(tl.float32)
    dv = tl.load(dt + hs, mask=hs < HV, other=0.0).to(tl.float32)
    u = av + dv[None, :]
    sigu = 1.0 / (1.0 + tl.exp(-u))
    eal = tl.exp(al)
    dav = dgv * (-eal[None, :]) * sigu
    tl.store(da + flat_offs, dav, mask=mask)
    gv = tl.load(g + flat_offs, mask=mask, other=0.0).to(tl.float32)
    sdt = tl.sum(tl.where(mask, dav, 0.0), axis=0)
    sa = tl.sum(tl.where(mask, dgv * gv, 0.0), axis=0)
    tl.store(pdt + pid_m * HV + hs, sdt, mask=hs < HV)
    tl.store(pa + pid_m * HV + hs, sa, mask=hs < HV)


_bwd_gate_kernel = triton.jit(_bwd_gate_kernel)


def _reduce_gate_kernel(pdt, pa, ddt, dalog, NB: tl.constexpr, HV: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_h = tl.program_id(0)
    ns = tl.arange(0, BLOCK_N)
    hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (ns[:, None] < NB) & (hs[None, :] < HV)
    vd = tl.load(pdt + ns[:, None] * HV + hs[None, :], mask=mask, other=0.0).to(tl.float32)
    va = tl.load(pa + ns[:, None] * HV + hs[None, :], mask=mask, other=0.0).to(tl.float32)
    sd = tl.sum(vd, axis=0)
    sa = tl.sum(va, axis=0)
    tl.store(ddt + hs, sd, mask=hs < HV)
    tl.store(dalog + hs, sa, mask=hs < HV)


_reduce_gate_kernel = triton.jit(_reduce_gate_kernel)


def _pack_dy_kernel(dq, dk, dv, dy, s_q0: tl.constexpr, s_q1: tl.constexpr, s_q2: tl.constexpr, s_q3: tl.constexpr, s_k0: tl.constexpr, s_k1: tl.constexpr, s_k2: tl.constexpr, s_k3: tl.constexpr, s_v0: tl.constexpr, s_v1: tl.constexpr, s_v2: tl.constexpr, s_v3: tl.constexpr, N: tl.constexpr, C: tl.constexpr, T: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = offs % C
    row = offs // C
    bidx = row // T
    tidx = row - bidx * T
    hkd = HK * D
    vd = HV * D
    val = tl.zeros((BLOCK,), tl.float32)
    is_q = c < hkd
    is_k = (c >= hkd) & (c < 2 * hkd)
    is_v = c >= 2 * hkd
    cq = c
    hq = cq // D
    dq_d = cq - hq * D
    ck = c - hkd
    hk = ck // D
    dk_d = ck - hk * D
    cv = c - 2 * hkd
    vh = cv // D
    vd_d = cv - vh * D
    for r in tl.static_range(0, R):
        qh = hq * R + r
        kh = hk * R + r
        qidx = bidx * s_q0 + tidx * s_q1 + qh * s_q2 + dq_d * s_q3
        kidx = bidx * s_k0 + tidx * s_k1 + kh * s_k2 + dk_d * s_k3
        qv = tl.load(dq + qidx, mask=(offs < N) & is_q, other=0.0).to(tl.float32)
        kv = tl.load(dk + kidx, mask=(offs < N) & is_k, other=0.0).to(tl.float32)
        val += qv + kv
    vidx = bidx * s_v0 + tidx * s_v1 + vh * s_v2 + vd_d * s_v3
    vv = tl.load(dv + vidx, mask=(offs < N) & is_v, other=0.0).to(tl.float32)
    val += vv
    tl.store(dy + offs, val, mask=offs < N)


_pack_dy_kernel = triton.jit(_pack_dy_kernel)


def _copy_dz_kernel(dz, out, s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr, N: tl.constexpr, T: tl.constexpr, HV: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = offs % D
    h = (offs // D) % HV
    row = offs // (HV * D)
    bidx = row // T
    tidx = row - bidx * T
    src = bidx * s0 + tidx * s1 + h * s2 + d * s3
    v = tl.load(dz + src, mask=offs < N, other=0.0)
    tl.store(out + offs, v, mask=offs < N)


_copy_dz_kernel = triton.jit(_copy_dz_kernel)


def _pack_dba_kernel(db, da, out, N: tl.constexpr, HV: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    h = offs % (2 * HV)
    row = offs // (2 * HV)
    is_b = h < HV
    src_h = tl.where(is_b, h, h - HV)
    v = tl.load(tl.where(is_b, db, da) + row * HV + src_h, mask=offs < N, other=0.0)
    tl.store(out + offs, v, mask=offs < N)


_pack_dba_kernel = triton.jit(_pack_dba_kernel)


def _pack_wba_kernel(wb, wa, out, N: tl.constexpr, HV: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = offs % E
    r = offs // E
    is_b = r < HV
    rr = tl.where(is_b, r, r - HV)
    v = tl.load(tl.where(is_b, wb, wa) + rr * E + e, mask=offs < N, other=0.0)
    tl.store(out + offs, v, mask=offs < N)


_pack_wba_kernel = triton.jit(_pack_wba_kernel)


def _bwd_silu_conv_ds_kernel(dy, qkv, cw, ds, BT: tl.constexpr, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tt = offs_t % T
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for j in tl.static_range(0, K):
        in_t = tt - (K - 1) + j
        in_row = offs_t - (K - 1) + j
        m = (offs_t[:, None] < BT) & (offs_c[None, :] < C) & (in_t[:, None] >= 0)
        xv = tl.load(qkv + in_row[:, None] * C + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wv = tl.load(cw + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += xv * wv[None, :]
    sig = 1.0 / (1.0 + tl.exp(-acc))
    prime = sig * (1.0 + acc * (1.0 - sig))
    dyv = tl.load(dy + offs_t[:, None] * C + offs_c[None, :], mask=(offs_t[:, None] < BT) & (offs_c[None, :] < C), other=0.0).to(tl.float32)
    tl.store(ds + offs_t[:, None] * C + offs_c[None, :], dyv * prime, mask=(offs_t[:, None] < BT) & (offs_c[None, :] < C))


_bwd_silu_conv_ds_kernel = triton.jit(_bwd_silu_conv_ds_kernel)


def _bwd_silu_conv_ds_from_grads_kernel(dq, dk, dv, qkv, cw, ds, s_q0: tl.constexpr, s_q1: tl.constexpr, s_q2: tl.constexpr, s_q3: tl.constexpr, s_k0: tl.constexpr, s_k1: tl.constexpr, s_k2: tl.constexpr, s_k3: tl.constexpr, s_v0: tl.constexpr, s_v1: tl.constexpr, s_v2: tl.constexpr, s_v3: tl.constexpr, BT: tl.constexpr, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, R: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tt = offs_t % T

    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for j in tl.static_range(0, K):
        in_t = tt - (K - 1) + j
        in_row = offs_t - (K - 1) + j
        m = (offs_t[:, None] < BT) & (offs_c[None, :] < C) & (in_t[:, None] >= 0)
        xv = tl.load(qkv + in_row[:, None] * C + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wv = tl.load(cw + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += xv * wv[None, :]

    sig = 1.0 / (1.0 + tl.exp(-acc))
    prime = sig * (1.0 + acc * (1.0 - sig))

    rows = offs_t[:, None]
    cs = offs_c[None, :]
    bidx = rows // T
    tidx = rows - bidx * T
    mask = (rows < BT) & (cs < C)

    hkd = HK * D
    is_q = cs < hkd
    is_k = (cs >= hkd) & (cs < 2 * hkd)
    is_v = cs >= 2 * hkd

    val = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)

    hq = cs // D
    dq_d = cs - hq * D
    ck = cs - hkd
    hk = ck // D
    dk_d = ck - hk * D

    for r in tl.static_range(0, R):
        qh = hq * R + r
        kh = hk * R + r
        qidx = bidx * s_q0 + tidx * s_q1 + qh * s_q2 + dq_d * s_q3
        kidx = bidx * s_k0 + tidx * s_k1 + kh * s_k2 + dk_d * s_k3
        qv = tl.load(dq + qidx, mask=mask & is_q, other=0.0).to(tl.float32)
        kv = tl.load(dk + kidx, mask=mask & is_k, other=0.0).to(tl.float32)
        val += qv + kv

    cv = cs - 2 * hkd
    vh = cv // D
    vd_d = cv - vh * D
    vidx = bidx * s_v0 + tidx * s_v1 + vh * s_v2 + vd_d * s_v3
    vv = tl.load(dv + vidx, mask=mask & is_v, other=0.0).to(tl.float32)
    val += vv

    tl.store(ds + rows * C + cs, val * prime, mask=mask)


_bwd_silu_conv_ds_from_grads_kernel = triton.jit(_bwd_silu_conv_ds_from_grads_kernel)


def _bwd_conv_input_kernel(ds, cw, dqkv, BT: tl.constexpr, C: tl.constexpr, T: tl.constexpr, K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    tt = offs_t % T
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for j in tl.static_range(0, K):
        out_t = tt + (K - 1) - j
        out_row = offs_t + (K - 1) - j
        m = (offs_t[:, None] < BT) & (offs_c[None, :] < C) & (out_t[:, None] < T)
        dsv = tl.load(ds + out_row[:, None] * C + offs_c[None, :], mask=m, other=0.0).to(tl.float32)
        wv = tl.load(cw + offs_c * K + j, mask=offs_c < C, other=0.0).to(tl.float32)
        acc += dsv * wv[None, :]
    tl.store(dqkv + offs_t[:, None] * C + offs_c[None, :], acc, mask=(offs_t[:, None] < BT) & (offs_c[None, :] < C))


_bwd_conv_input_kernel = triton.jit(_bwd_conv_input_kernel)


def _conv_w_partial_kernel(ds, qkv, partial, B: tl.constexpr, T: tl.constexpr, C: tl.constexpr, K: tl.constexpr, NTB: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_bt = tl.program_id(1)
    b = pid_bt // NTB
    tb = pid_bt - b * NTB
    ts = tb * BLOCK_T + tl.arange(0, BLOCK_T)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    row = b * T + ts
    valid_ds = (b < B) & (ts[:, None] < T) & (cs[None, :] < C)
    dsv = tl.load(ds + row[:, None] * C + cs[None, :], mask=valid_ds, other=0.0).to(tl.float32)
    for j in tl.static_range(0, K):
        it = ts - (K - 1) + j
        irow = b * T + it
        mask_x = (b < B) & (ts[:, None] < T) & (it[:, None] >= 0) & (cs[None, :] < C)
        xv = tl.load(qkv + irow[:, None] * C + cs[None, :], mask=mask_x, other=0.0).to(tl.float32)
        s = tl.sum(dsv * xv, axis=0)
        tl.store(partial + pid_bt * C * K + cs * K + j, s, mask=cs < C)


_conv_w_partial_kernel = triton.jit(_conv_w_partial_kernel)


def _conv_w_reduce_kernel(partial, dw, NB: tl.constexpr, C: tl.constexpr, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_c = tl.program_id(0)
    ns = tl.arange(0, BLOCK_N)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    for j in tl.static_range(0, K):
        vals = tl.load(partial + ns[:, None] * C * K + cs[None, :] * K + j, mask=(ns[:, None] < NB) & (cs[None, :] < C), other=0.0).to(tl.float32)
        s = tl.sum(vals, axis=0)
        tl.store(dw + cs * K + j, s, mask=cs < C)


_conv_w_reduce_kernel = triton.jit(_conv_w_reduce_kernel)


def _dx_sum_kernel(a, b, c, d, out, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(a + offs, mask=offs < N, other=0.0).to(tl.float32)
    bv = tl.load(b + offs, mask=offs < N, other=0.0).to(tl.float32)
    cv = tl.load(c + offs, mask=offs < N, other=0.0).to(tl.float32)
    dv = tl.load(d + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(out + offs, av + bv + cv + dv, mask=offs < N)


_dx_sum_kernel = triton.jit(_dx_sum_kernel)


def _dx_sum3_kernel(a, b, c, out, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    av = tl.load(a + offs, mask=offs < N, other=0.0).to(tl.float32)
    bv = tl.load(b + offs, mask=offs < N, other=0.0).to(tl.float32)
    cv = tl.load(c + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(out + offs, av + bv + cv, mask=offs < N)


_dx_sum3_kernel = triton.jit(_dx_sum3_kernel)


def qwen3_5_gdn_in_proj_forward_with_saved(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
    if x.dim() != 3:
        raise ValueError("x must be rank 3")
    if w_qkv.dim() != 2 or w_z.dim() != 2 or w_b.dim() != 2 or w_a.dim() != 2 or conv_w.dim() != 2:
        raise ValueError("weights have wrong rank")
    if A_log.dim() != 1 or dt_bias.dim() != 1:
        raise ValueError("gate vectors have wrong rank")
    if not x.is_cuda:
        raise ValueError("cuda tensors required")
    tensors = (w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias)
    for ten in tensors:
        if not ten.is_cuda:
            raise ValueError("cuda tensors required")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not w_qkv.is_contiguous() or not w_z.is_contiguous() or not w_b.is_contiguous() or not w_a.is_contiguous() or not conv_w.is_contiguous():
        raise ValueError("weights must be contiguous")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("gate vectors must be contiguous")
    B = x.shape[0]
    T = x.shape[1]
    E = x.shape[2]
    C = w_qkv.shape[0]
    VD = w_z.shape[0]
    HV = A_log.shape[0]
    K = conv_w.shape[1]
    if w_qkv.shape[1] != E or w_z.shape[1] != E or w_b.shape[1] != E or w_a.shape[1] != E:
        raise ValueError("input widths mismatch")
    if w_b.shape[0] != HV or w_a.shape[0] != HV or dt_bias.shape[0] != HV:
        raise ValueError("head vector mismatch")
    if conv_w.shape[0] != C:
        raise ValueError("conv width mismatch")
    if HV <= 0 or VD % HV != 0:
        raise ValueError("invalid value heads")
    D = VD // HV
    if D <= 0:
        raise ValueError("invalid head dim")
    rem = C - VD
    if rem <= 0 or rem % (2 * D) != 0:
        raise ValueError("invalid qkv width")
    HK = rem // (2 * D)
    if HK <= 0 or HV % HK != 0:
        raise ValueError("invalid key heads")
    R = HV // HK
    if 2 * HK * D + VD != C:
        raise ValueError("invalid qkv layout")
    if K <= 0 or K > 16:
        raise ValueError("unsupported conv width")
    BT = B * T
    x_flat = x.reshape(BT, E)
    qkv_flat = vendor_gemm(x_flat, w_qkv.t())
    z_flat = vendor_gemm(x_flat, w_z.t())
    b_flat = vendor_gemm(x_flat, w_b.t())
    a_flat = vendor_gemm(x_flat, w_a.t())
    y_flat = torch.empty((BT, C), device=x.device, dtype=x.dtype)
    btile = 16
    cblock = 32
    grid_conv = (triton.cdiv(BT, btile), triton.cdiv(C, cblock))
    _fwd_conv_silu_kernel[grid_conv](qkv_flat, conv_w, y_flat, BT, C, T, K, BLOCK_T=btile, BLOCK_C=cblock)
    q = torch.empty((B, T, HV, D), device=x.device, dtype=x.dtype)
    k = torch.empty((B, T, HV, D), device=x.device, dtype=x.dtype)
    n_qk = BT * HV * D
    _fwd_repeat_qk_kernel[(triton.cdiv(n_qk, 256),)](y_flat, q, k, n_qk, C, HV, HK, D, R, BLOCK=256)
    start_v = 2 * HK * D
    v = y_flat[:, start_v:start_v + VD].view(B, T, HV, D)
    z = z_flat.view(B, T, HV, D)
    beta_flat = torch.empty((BT, HV), device=x.device, dtype=x.dtype)
    g_flat = torch.empty((BT, HV), device=x.device, dtype=torch.float32)
    n_gate = BT * HV
    _fwd_beta_g_kernel[(triton.cdiv(n_gate, 256),)](b_flat, a_flat, A_log, dt_bias, beta_flat, g_flat, n_gate, HV, BLOCK=256)
    beta = beta_flat.view(B, T, HV)
    g = g_flat.view(B, T, HV)
    saved_tensors = (x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias, qkv_flat, a_flat, beta, g, B, T, E, C, VD, HV, HK, D, R, K)
    return (q, k, v, z, beta, g), saved_tensors


def qwen3_5_gdn_in_proj_backward_from_saved(output_grads, saved_tensors):
    dq, dk, dv, dz, dbeta, dg = output_grads
    x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias, qkv_flat, a_flat, beta, g, B, T, E, C, VD, HV, HK, D, R, K = saved_tensors
    if dq is None or dk is None or dv is None or dz is None or dbeta is None or dg is None:
        raise ValueError("all output gradients are required")
    if not dq.is_cuda or not dk.is_cuda or not dv.is_cuda or not dz.is_cuda or not dbeta.is_cuda or not dg.is_cuda:
        raise ValueError("cuda gradients required")
    if dq.dim() != 4 or dk.dim() != 4 or dv.dim() != 4 or dz.dim() != 4:
        raise ValueError("rank 4 qk vz gradients required")
    if dbeta.dim() != 3 or dg.dim() != 3:
        raise ValueError("rank 3 gate gradients required")
    if dq.shape[0] != B or dq.shape[1] != T or dq.shape[2] != HV or dq.shape[3] != D:
        raise ValueError("dq shape mismatch")
    if dk.shape[0] != B or dk.shape[1] != T or dk.shape[2] != HV or dk.shape[3] != D:
        raise ValueError("dk shape mismatch")
    if dv.shape[0] != B or dv.shape[1] != T or dv.shape[2] != HV or dv.shape[3] != D:
        raise ValueError("dv shape mismatch")
    if dz.shape[0] != B or dz.shape[1] != T or dz.shape[2] != HV or dz.shape[3] != D:
        raise ValueError("dz shape mismatch")
    if dbeta.shape[0] != B or dbeta.shape[1] != T or dbeta.shape[2] != HV:
        raise ValueError("dbeta shape mismatch")
    if dg.shape[0] != B or dg.shape[1] != T or dg.shape[2] != HV:
        raise ValueError("dg shape mismatch")
    BT = B * T
    x_flat = x.reshape(BT, E)
    if dz.is_contiguous():
        dz_flat = dz.reshape(BT, VD)
    else:
        dz_flat = torch.empty((BT, VD), device=x.device, dtype=x.dtype)
        _copy_dz_kernel[(triton.cdiv(BT * VD, 256),)](dz, dz_flat, dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3), BT * VD, T, HV, D, BLOCK=256)
    db_flat = torch.empty((BT, HV), device=x.device, dtype=x.dtype)
    da_flat = torch.empty((BT, HV), device=x.device, dtype=x.dtype)
    gate_bm = 64
    gate_bh = triton.next_power_of_2(HV)
    if gate_bh > 1024:
        raise ValueError("too many heads")
    gate_blocks = triton.cdiv(BT, gate_bm)
    pdt = torch.empty((gate_blocks, HV), device=x.device, dtype=torch.float32)
    pa = torch.empty((gate_blocks, HV), device=x.device, dtype=torch.float32)
    _bwd_gate_kernel[(triton.cdiv(HV, gate_bh), gate_blocks)](dbeta, dg, beta, g, a_flat, A_log, dt_bias, db_flat, da_flat, pdt, pa, dbeta.stride(0), dbeta.stride(1), dbeta.stride(2), dg.stride(0), dg.stride(1), dg.stride(2), BT, T, HV, BLOCK_M=gate_bm, BLOCK_H=gate_bh)
    ddt_bias = torch.empty((HV,), device=x.device, dtype=dt_bias.dtype)
    dA_log = torch.empty((HV,), device=x.device, dtype=A_log.dtype)
    red_n = triton.next_power_of_2(gate_blocks)
    if red_n > 4096:
        raise ValueError("gate reduction too large")
    _reduce_gate_kernel[(triton.cdiv(HV, gate_bh),)](pdt, pa, ddt_bias, dA_log, gate_blocks, HV, BLOCK_N=red_n, BLOCK_H=gate_bh)
    ds_flat = torch.empty((BT, C), device=x.device, dtype=x.dtype)
    btile = 16
    cblock = 32
    grid_conv = (triton.cdiv(BT, btile), triton.cdiv(C, cblock))
    _bwd_silu_conv_ds_from_grads_kernel[grid_conv](dq, dk, dv, qkv_flat, conv_w, ds_flat, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3), dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3), dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3), BT, C, T, K, HV, HK, D, R, BLOCK_T=btile, BLOCK_C=cblock)
    dqkv_flat = torch.empty((BT, C), device=x.device, dtype=x.dtype)
    _bwd_conv_input_kernel[grid_conv](ds_flat, conv_w, dqkv_flat, BT, C, T, K, BLOCK_T=btile, BLOCK_C=cblock)
    wt_block_t = 32
    ntb = triton.cdiv(T, wt_block_t)
    nb_partial = B * ntb
    partial = torch.empty((nb_partial, C, K), device=x.device, dtype=torch.float32)
    _conv_w_partial_kernel[(triton.cdiv(C, cblock), nb_partial)](ds_flat, qkv_flat, partial, B, T, C, K, ntb, BLOCK_T=wt_block_t, BLOCK_C=cblock)
    dconv_w = torch.empty((C, K), device=x.device, dtype=conv_w.dtype)
    red_conv = triton.next_power_of_2(nb_partial)
    if red_conv > 4096:
        raise ValueError("conv reduction too large")
    _conv_w_reduce_kernel[(triton.cdiv(C, cblock),)](partial, dconv_w, nb_partial, C, K, BLOCK_N=red_conv, BLOCK_C=cblock)
    dx_qkv = vendor_gemm(dqkv_flat, w_qkv)
    dw_qkv = vendor_gemm(dqkv_flat.t(), x_flat)
    dx_z = vendor_gemm(dz_flat, w_z)
    dw_z = vendor_gemm(dz_flat.t(), x_flat)

    dba_flat = torch.empty((BT, 2 * HV), device=x.device, dtype=x.dtype)
    _pack_dba_kernel[(triton.cdiv(BT * 2 * HV, 256),)](db_flat, da_flat, dba_flat, BT * 2 * HV, HV, BLOCK=256)

    w_ba = torch.empty((2 * HV, E), device=x.device, dtype=x.dtype)
    _pack_wba_kernel[(triton.cdiv(2 * HV * E, 256),)](w_b, w_a, w_ba, 2 * HV * E, HV, E, BLOCK=256)

    dx_ba = vendor_gemm(dba_flat, w_ba)
    dw_ba = vendor_gemm(dba_flat.t(), x_flat)
    dw_b = dw_ba[:HV, :]
    dw_a = dw_ba[HV:, :]

    dx = torch.empty((B, T, E), device=x.device, dtype=x.dtype)
    _dx_sum3_kernel[(triton.cdiv(BT * E, 256),)](dx_qkv, dx_z, dx_ba, dx, BT * E, BLOCK=256)
    return dx, dw_qkv, dw_z, dw_b, dw_a, dconv_w, dA_log, ddt_bias
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_gdn_in_proj', 'arguments': ['x', 'w_qkv', 'w_z', 'w_b', 'w_a', 'conv_w', 'A_log', 'dt_bias'], 'outputs': ['q', 'k', 'v', 'z', 'beta', 'g'], 'upstream_grads': ['dq', 'dk', 'dv', 'dz', 'dbeta', 'dg'], 'input_grads': ['dx', 'dw_qkv', 'dw_z', 'dw_b', 'dw_a', 'dconv_w', 'dA_log', 'ddt_bias'], 'gradient_return_order': ['dx', 'dw_qkv', 'dw_z', 'dw_b', 'dw_a', 'dconv_w', 'dA_log', 'ddt_bias'], 'scalars': [], 'deployment_entry': 'qwen3_5_gdn_in_proj_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen35GdnInProjFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_gdn_in_proj`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
        (q, k, v, z, beta, g), saved = qwen3_5_gdn_in_proj_forward_with_saved(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return q, k, v, z, beta, g

    @staticmethod
    def backward(ctx, dq, dk, dv, dz, dbeta, dg):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dq = dq.contiguous() if (dq is not None and not dq.is_contiguous()) else dq
        dk = dk.contiguous() if (dk is not None and not dk.is_contiguous()) else dk
        dv = dv.contiguous() if (dv is not None and not dv.is_contiguous()) else dv
        dz = dz.contiguous() if (dz is not None and not dz.is_contiguous()) else dz
        dbeta = dbeta.contiguous() if (dbeta is not None and not dbeta.is_contiguous()) else dbeta
        dg = dg.contiguous() if (dg is not None and not dg.is_contiguous()) else dg
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dw_qkv, dw_z, dw_b, dw_a, dconv_w, dA_log, ddt_bias = qwen3_5_gdn_in_proj_backward_from_saved(
            (dq, dk, dv, dz, dbeta, dg), saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dw_qkv, dw_z, dw_b, dw_a, dconv_w, dA_log, ddt_bias


def qwen3_5_gdn_in_proj_deployment(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen35GdnInProjFunction.apply(x, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias)


class Qwen35GdnInProjModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, w_qkv, w_z, w_b, w_a, conv_w, A_log, dt_bias):
        super().__init__()
        self.w_qkv = (w_qkv if isinstance(w_qkv, torch.nn.Parameter)
                  else torch.nn.Parameter(w_qkv))
        self.w_z = (w_z if isinstance(w_z, torch.nn.Parameter)
                  else torch.nn.Parameter(w_z))
        self.w_b = (w_b if isinstance(w_b, torch.nn.Parameter)
                  else torch.nn.Parameter(w_b))
        self.w_a = (w_a if isinstance(w_a, torch.nn.Parameter)
                  else torch.nn.Parameter(w_a))
        self.conv_w = (conv_w if isinstance(conv_w, torch.nn.Parameter)
                  else torch.nn.Parameter(conv_w))
        self.A_log = (A_log if isinstance(A_log, torch.nn.Parameter)
                  else torch.nn.Parameter(A_log))
        self.dt_bias = (dt_bias if isinstance(dt_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(dt_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return qwen3_5_gdn_in_proj_deployment(x, self.w_qkv, self.w_z, self.w_b, self.w_a, self.conv_w, self.A_log, self.dt_bias)


DEPLOYMENT_ENTRY = "qwen3_5_gdn_in_proj_deployment"
