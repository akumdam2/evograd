import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _ceil_div(a, b):
    return triton.cdiv(a, b)


def _check_forward_inputs(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    if not x.is_cuda or not gate_up_proj.is_cuda or not down_proj.is_cuda or not top_k_index.is_cuda or not top_k_weights.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")
    if not x.is_contiguous() or not gate_up_proj.is_contiguous() or not down_proj.is_contiguous() or not top_k_index.is_contiguous() or not top_k_weights.is_contiguous():
        raise ValueError("all tensors must be contiguous")
    if x.ndim != 2:
        raise ValueError("x must be rank 2")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert weights must be rank 3")
    if top_k_index.ndim != 2 or top_k_weights.ndim != 2:
        raise ValueError("routing tensors must be rank 2")
    if top_k_index.shape != top_k_weights.shape:
        raise ValueError("top_k_index and top_k_weights must have the same shape")
    T = x.shape[0]
    H = x.shape[1]
    E = gate_up_proj.shape[0]
    G = gate_up_proj.shape[1]
    H2 = gate_up_proj.shape[2]
    if H2 != H:
        raise ValueError("gate_up_proj has incompatible hidden width")
    if G % 2 != 0:
        raise ValueError("gate_up_proj second dimension must be even")
    I = G // 2
    if down_proj.shape[0] != E or down_proj.shape[1] != H or down_proj.shape[2] != I:
        raise ValueError("down_proj has incompatible shape")
    if top_k_index.shape[0] != T:
        raise ValueError("top_k_index must have one row per token")
    if top_k_index.dtype not in (torch.int64, torch.int32):
        raise ValueError("top_k_index must be int64 or int32")
    if gate_up_proj.dtype != x.dtype or down_proj.dtype != x.dtype or top_k_weights.dtype != x.dtype:
        raise ValueError("floating tensors must have one dtype")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("unsupported floating dtype")
    return T, H, I, E, top_k_index.shape[1]


def _check_backward_inputs(dout, saved_tensors):
    if not isinstance(saved_tensors, (tuple, list)):
        raise ValueError("saved_tensors must be a tuple or list")
    if len(saved_tensors) != 12:
        raise ValueError("unexpected saved state")
    x, gate_up_proj, down_proj, top_k_weights, hidden, route_ids, expert_offsets, T, H, I, E, K = saved_tensors
    if not dout.is_cuda or not dout.is_contiguous():
        raise ValueError("dout must be contiguous CUDA")
    if dout.shape != (T, H):
        raise ValueError("dout has incompatible shape")
    if dout.dtype != x.dtype:
        raise ValueError("dout has incompatible dtype")
    return x, gate_up_proj, down_proj, top_k_weights, hidden, route_ids, expert_offsets, T, H, I, E, K


def _zero_i32_kernel(ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=offs < n)


_zero_i32_kernel = triton.jit(_zero_i32_kernel)


def _zero_f32_kernel(ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros((BLOCK,), tl.float32), mask=offs < n)


_zero_f32_kernel = triton.jit(_zero_f32_kernel)


def _count_experts_kernel(top_idx, counts, P: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P
    e = tl.load(top_idx + offs, mask=mask, other=0)
    tl.atomic_add(counts + e, tl.full((BLOCK,), 1, tl.int32), sem="relaxed", mask=mask)


_count_experts_kernel = triton.jit(_count_experts_kernel)


def _prefix_offsets_kernel(counts, offsets, cursor, E: tl.constexpr, BLOCK_E: tl.constexpr):
    idx = tl.arange(0, BLOCK_E)
    vals = tl.load(counts + idx, mask=idx < E, other=0)
    cs = tl.cumsum(vals, 0)
    tl.store(offsets, tl.full((), 0, tl.int32))
    tl.store(offsets + idx + 1, cs, mask=idx < E)
    prev = cs - vals
    tl.store(cursor + idx, prev, mask=idx < E)


_prefix_offsets_kernel = triton.jit(_prefix_offsets_kernel)


def _fill_route_ids_kernel(top_idx, cursor, route_ids, P: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P
    e = tl.load(top_idx + offs, mask=mask, other=0)
    slot = tl.atomic_add(cursor + e, tl.full((BLOCK,), 1, tl.int32), sem="relaxed", mask=mask)
    tl.store(route_ids + slot, offs.to(tl.int32), mask=mask)


_fill_route_ids_kernel = triton.jit(_fill_route_ids_kernel)


def _fwd_hidden_kernel(x, gate_up, route_ids, offsets, hidden, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    rb = tl.program_id(1)
    ib = tl.program_id(2)
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    block_start = start + rb * BLOCK_M
    if block_start >= end:
        return
    rm = tl.arange(0, BLOCK_M)
    ri = tl.arange(0, BLOCK_I)
    rh = tl.arange(0, BLOCK_H)
    q = block_start + rm
    m_mask = q < end
    p = tl.load(route_ids + q, mask=m_mask, other=0)
    t = p // K
    ioffs = ib * BLOCK_I + ri
    acc_g = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + rh
        xm = tl.load(x + t[:, None] * H + h[None, :], mask=m_mask[:, None] & (h[None, :] < H), other=0.0)
        wg = tl.load(gate_up + e * (2 * I) * H + ioffs[:, None] * H + h[None, :], mask=(ioffs[:, None] < I) & (h[None, :] < H), other=0.0)
        wu = tl.load(gate_up + e * (2 * I) * H + (I + ioffs)[:, None] * H + h[None, :], mask=(ioffs[:, None] < I) & (h[None, :] < H), other=0.0)
        acc_g += tl.dot(xm, tl.trans(wg), out_dtype=tl.float32)
        acc_u += tl.dot(xm, tl.trans(wu), out_dtype=tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-acc_g))
    hv = acc_g * sig * acc_u
    tl.store(hidden + p[:, None] * I + ioffs[None, :], hv, mask=m_mask[:, None] & (ioffs[None, :] < I))


_fwd_hidden_kernel = triton.jit(_fwd_hidden_kernel)


def _fwd_down_scatter_kernel(hidden, down, route_ids, offsets, topw, out_accum, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_HO: tl.constexpr, BLOCK_I: tl.constexpr):
    e = tl.program_id(0)
    rb = tl.program_id(1)
    hb = tl.program_id(2)
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    block_start = start + rb * BLOCK_M
    if block_start >= end:
        return
    rm = tl.arange(0, BLOCK_M)
    rh = tl.arange(0, BLOCK_HO)
    ri = tl.arange(0, BLOCK_I)
    q = block_start + rm
    m_mask = q < end
    p = tl.load(route_ids + q, mask=m_mask, other=0)
    t = p // K
    j = p - t * K
    ho = hb * BLOCK_HO + rh
    acc = tl.zeros((BLOCK_M, BLOCK_HO), tl.float32)
    for i0 in range(0, I, BLOCK_I):
        ii = i0 + ri
        hm = tl.load(hidden + p[:, None] * I + ii[None, :], mask=m_mask[:, None] & (ii[None, :] < I), other=0.0)
        dw = tl.load(down + e * H * I + ho[:, None] * I + ii[None, :], mask=(ho[:, None] < H) & (ii[None, :] < I), other=0.0).to(tl.float32)
        acc += tl.dot(hm, tl.trans(dw), input_precision="ieee", out_dtype=tl.float32)
    w = tl.load(topw + t * K + j, mask=m_mask, other=0.0).to(tl.float32)
    tl.atomic_add(out_accum + t[:, None] * H + ho[None, :], acc * w[:, None], sem="relaxed", mask=m_mask[:, None] & (ho[None, :] < H))


_fwd_down_scatter_kernel = triton.jit(_fwd_down_scatter_kernel)


def _cast_kernel(src, dst, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(src + offs, mask=offs < n, other=0.0)
    tl.store(dst + offs, v, mask=offs < n)


_cast_kernel = triton.jit(_cast_kernel)


def _bwd_down_input_dtop_kernel(dout, down, hidden, route_ids, offsets, topw, dhidden, dtop_accum, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    rb = tl.program_id(1)
    ib = tl.program_id(2)
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    block_start = start + rb * BLOCK_M
    if block_start >= end:
        return
    rm = tl.arange(0, BLOCK_M)
    ri = tl.arange(0, BLOCK_I)
    rh = tl.arange(0, BLOCK_H)
    q = block_start + rm
    m_mask = q < end
    p = tl.load(route_ids + q, mask=m_mask, other=0)
    t = p // K
    j = p - t * K
    ii = ib * BLOCK_I + ri
    z = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        hh = h0 + rh
        dm = tl.load(dout + t[:, None] * H + hh[None, :], mask=m_mask[:, None] & (hh[None, :] < H), other=0.0)
        wd = tl.load(down + e * H * I + hh[:, None] * I + ii[None, :], mask=(hh[:, None] < H) & (ii[None, :] < I), other=0.0)
        z += tl.dot(dm, wd, out_dtype=tl.float32)
    w = tl.load(topw + t * K + j, mask=m_mask, other=0.0).to(tl.float32)
    tl.store(dhidden + p[:, None] * I + ii[None, :], z * w[:, None], mask=m_mask[:, None] & (ii[None, :] < I))
    hv = tl.load(hidden + p[:, None] * I + ii[None, :], mask=m_mask[:, None] & (ii[None, :] < I), other=0.0)
    part = tl.sum(hv * z, axis=1)
    tl.atomic_add(dtop_accum + p, part, sem="relaxed", mask=m_mask)


_bwd_down_input_dtop_kernel = triton.jit(_bwd_down_input_dtop_kernel)


def _bwd_ddown_kernel(dout, hidden, route_ids, offsets, topw, ddown, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_I: tl.constexpr):
    e = tl.program_id(0)
    hb = tl.program_id(1)
    ib = tl.program_id(2)
    rh = tl.arange(0, BLOCK_H)
    ri = tl.arange(0, BLOCK_I)
    rm = tl.arange(0, BLOCK_M)
    ho = hb * BLOCK_H + rh
    ii = ib * BLOCK_I + ri
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    acc = tl.zeros((BLOCK_H, BLOCK_I), tl.float32)
    base = start
    while base < end:
        q = base + rm
        mask = q < end
        p = tl.load(route_ids + q, mask=mask, other=0)
        t = p // K
        j = p - t * K
        w = tl.load(topw + t * K + j, mask=mask, other=0.0).to(tl.float32)
        dm = tl.load(dout + t[:, None] * H + ho[None, :], mask=mask[:, None] & (ho[None, :] < H), other=0.0).to(tl.float32)
        dy = dm * w[:, None]
        hv = tl.load(hidden + p[:, None] * I + ii[None, :], mask=mask[:, None] & (ii[None, :] < I), other=0.0)
        acc += tl.dot(tl.trans(dy), hv, input_precision="ieee", out_dtype=tl.float32)
        base += BLOCK_M
    tl.store(ddown + e * H * I + ho[:, None] * I + ii[None, :], acc, mask=(ho[:, None] < H) & (ii[None, :] < I))


_bwd_ddown_kernel = triton.jit(_bwd_ddown_kernel)


def _bwd_dgate_dup_kernel(x, gate_up, dhidden, route_ids, offsets, dgu, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    rb = tl.program_id(1)
    ib = tl.program_id(2)
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    block_start = start + rb * BLOCK_M
    if block_start >= end:
        return
    rm = tl.arange(0, BLOCK_M)
    ri = tl.arange(0, BLOCK_I)
    rh = tl.arange(0, BLOCK_H)
    q = block_start + rm
    m_mask = q < end
    p = tl.load(route_ids + q, mask=m_mask, other=0)
    t = p // K
    ioffs = ib * BLOCK_I + ri
    acc_g = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_I), tl.float32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + rh
        xm = tl.load(x + t[:, None] * H + h[None, :], mask=m_mask[:, None] & (h[None, :] < H), other=0.0)
        wg = tl.load(gate_up + e * (2 * I) * H + ioffs[:, None] * H + h[None, :], mask=(ioffs[:, None] < I) & (h[None, :] < H), other=0.0)
        wu = tl.load(gate_up + e * (2 * I) * H + (I + ioffs)[:, None] * H + h[None, :], mask=(ioffs[:, None] < I) & (h[None, :] < H), other=0.0)
        acc_g += tl.dot(xm, tl.trans(wg), out_dtype=tl.float32)
        acc_u += tl.dot(xm, tl.trans(wu), out_dtype=tl.float32)
    dh = tl.load(dhidden + p[:, None] * I + ioffs[None, :], mask=m_mask[:, None] & (ioffs[None, :] < I), other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-acc_g))
    silu = acc_g * sig
    dsilu = sig * (1.0 + acc_g * (1.0 - sig))
    dg = dh * acc_u * dsilu
    du = dh * silu
    tl.store(dgu + p[:, None] * (2 * I) + ioffs[None, :], dg, mask=m_mask[:, None] & (ioffs[None, :] < I))
    tl.store(dgu + p[:, None] * (2 * I) + (I + ioffs)[None, :], du, mask=m_mask[:, None] & (ioffs[None, :] < I))


_bwd_dgate_dup_kernel = triton.jit(_bwd_dgate_dup_kernel)


def _bwd_dgate_up_kernel(dgu, x, route_ids, offsets, dgate_up, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_H: tl.constexpr):
    e = tl.program_id(0)
    gb = tl.program_id(1)
    hb = tl.program_id(2)
    rg = tl.arange(0, BLOCK_G)
    rh = tl.arange(0, BLOCK_H)
    rm = tl.arange(0, BLOCK_M)
    go = gb * BLOCK_G + rg
    ho = hb * BLOCK_H + rh
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    acc = tl.zeros((BLOCK_G, BLOCK_H), tl.float32)
    base = start
    while base < end:
        q = base + rm
        mask = q < end
        p = tl.load(route_ids + q, mask=mask, other=0)
        t = p // K
        gm = tl.load(dgu + p[:, None] * (2 * I) + go[None, :], mask=mask[:, None] & (go[None, :] < 2 * I), other=0.0)
        xm = tl.load(x + t[:, None] * H + ho[None, :], mask=mask[:, None] & (ho[None, :] < H), other=0.0)
        acc += tl.dot(tl.trans(gm), xm, input_precision="ieee", out_dtype=tl.float32)
        base += BLOCK_M
    tl.store(dgate_up + e * (2 * I) * H + go[:, None] * H + ho[None, :], acc, mask=(go[:, None] < 2 * I) & (ho[None, :] < H))


_bwd_dgate_up_kernel = triton.jit(_bwd_dgate_up_kernel)


def _bwd_dx_scatter_kernel(dgu, gate_up, route_ids, offsets, dx_accum, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, P: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_HO: tl.constexpr, BLOCK_G: tl.constexpr):
    e = tl.program_id(0)
    rb = tl.program_id(1)
    hb = tl.program_id(2)
    start = tl.load(offsets + e)
    end = tl.load(offsets + e + 1)
    block_start = start + rb * BLOCK_M
    if block_start >= end:
        return
    rm = tl.arange(0, BLOCK_M)
    rh = tl.arange(0, BLOCK_HO)
    rg = tl.arange(0, BLOCK_G)
    q = block_start + rm
    m_mask = q < end
    p = tl.load(route_ids + q, mask=m_mask, other=0)
    t = p // K
    ho = hb * BLOCK_HO + rh
    acc = tl.zeros((BLOCK_M, BLOCK_HO), tl.float32)
    for g0 in range(0, 2 * I, BLOCK_G):
        gg = g0 + rg
        gm = tl.load(dgu + p[:, None] * (2 * I) + gg[None, :], mask=m_mask[:, None] & (gg[None, :] < 2 * I), other=0.0)
        wg = tl.load(gate_up + e * (2 * I) * H + gg[:, None] * H + ho[None, :], mask=(gg[:, None] < 2 * I) & (ho[None, :] < H), other=0.0)
        acc += tl.dot(gm, wg, input_precision="ieee", out_dtype=tl.float32)
    tl.atomic_add(dx_accum + t[:, None] * H + ho[None, :], acc, sem="relaxed", mask=m_mask[:, None] & (ho[None, :] < H))


_bwd_dx_scatter_kernel = triton.jit(_bwd_dx_scatter_kernel)


def _build_routes(top_k_index, T, E, K):
    P = T * K
    counts = torch.empty((E,), device=top_k_index.device, dtype=torch.int32)
    offsets = torch.empty((E + 1,), device=top_k_index.device, dtype=torch.int32)
    cursor = torch.empty((E,), device=top_k_index.device, dtype=torch.int32)
    route_ids = torch.empty((P,), device=top_k_index.device, dtype=torch.int32)
    block = 256
    _zero_i32_kernel[(_ceil_div(E, block),)](counts, E, BLOCK=block)
    _count_experts_kernel[(_ceil_div(P, block),)](top_k_index, counts, P, BLOCK=block)
    be = triton.next_power_of_2(E)
    _prefix_offsets_kernel[(1,)](counts, offsets, cursor, E, BLOCK_E=be)
    _fill_route_ids_kernel[(_ceil_div(P, block),)](top_k_index, cursor, route_ids, P, BLOCK=block)
    return route_ids, offsets


def olmoe_moe_experts_forward_with_saved(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    T, H, I, E, K = _check_forward_inputs(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    P = T * K
    route_ids, expert_offsets = _build_routes(top_k_index, T, E, K)
    hidden = torch.empty((P, I), device=x.device, dtype=torch.float32)
    out_accum = torch.empty((T, H), device=x.device, dtype=torch.float32)
    out = torch.empty((T, H), device=x.device, dtype=x.dtype)
    block_flat = 256
    _zero_f32_kernel[(_ceil_div(T * H, block_flat),)](out_accum, T * H, BLOCK=block_flat)
    bm = 32
    bi = 32
    bh = 128
    # Each expert can appear at most once per token because top_k experts are distinct,
    # so per-expert route count is bounded by T, not P=T*K.  This avoids launching
    # many empty programs for every expert.
    route_blocks = _ceil_div(T, bm)
    _fwd_hidden_kernel[(E, route_blocks, _ceil_div(I, bi))](x, gate_up_proj, route_ids, expert_offsets, hidden, H, I, K, P, BLOCK_M=bm, BLOCK_I=bi, BLOCK_H=bh)
    bho = 32
    _fwd_down_scatter_kernel[(E, route_blocks, _ceil_div(H, bho))](hidden, down_proj, route_ids, expert_offsets, top_k_weights, out_accum, H, I, K, P, BLOCK_M=bm, BLOCK_HO=bho, BLOCK_I=bi)
    _cast_kernel[(_ceil_div(T * H, block_flat),)](out_accum, out, T * H, BLOCK=block_flat)
    saved_tensors = (x, gate_up_proj, down_proj, top_k_weights, hidden, route_ids, expert_offsets, T, H, I, E, K)
    return out, saved_tensors


def olmoe_moe_experts_backward_from_saved(dout, saved_tensors):
    x, gate_up_proj, down_proj, top_k_weights, hidden, route_ids, expert_offsets, T, H, I, E, K = _check_backward_inputs(dout, saved_tensors)
    P = T * K
    dhidden = torch.empty((P, I), device=x.device, dtype=torch.float32)
    # Internal per-route gate/up gradient.  It is consumed only by tensor-core
    # GEMMs that accumulate in fp32 before casting final gradients to input dtype.
    # Keeping it in the activation dtype greatly reduces bandwidth vs fp32.
    dgu = torch.empty((P, 2 * I), device=x.device, dtype=x.dtype)
    dx_accum = torch.empty((T, H), device=x.device, dtype=torch.float32)
    dtop_accum = torch.empty((T, K), device=x.device, dtype=torch.float32)
    dx = torch.empty((T, H), device=x.device, dtype=x.dtype)
    dtop = torch.empty((T, K), device=x.device, dtype=top_k_weights.dtype)
    dgate_up = torch.empty_like(gate_up_proj)
    ddown = torch.empty_like(down_proj)
    block_flat = 256
    _zero_f32_kernel[(_ceil_div(T * H, block_flat),)](dx_accum, T * H, BLOCK=block_flat)
    _zero_f32_kernel[(_ceil_div(P, block_flat),)](dtop_accum, P, BLOCK=block_flat)
    bm = 32
    bi = 32
    bh = 128
    # Same bound as forward: no expert can receive more than T routes.
    route_blocks = _ceil_div(T, bm)
    _bwd_down_input_dtop_kernel[(E, route_blocks, _ceil_div(I, bi))](dout, down_proj, hidden, route_ids, expert_offsets, top_k_weights, dhidden, dtop_accum, H, I, K, P, BLOCK_M=bm, BLOCK_I=bi, BLOCK_H=bh)
    bho = 32
    _bwd_ddown_kernel[(E, _ceil_div(H, bho), _ceil_div(I, bi))](dout, hidden, route_ids, expert_offsets, top_k_weights, ddown, H, I, K, BLOCK_M=bm, BLOCK_H=bho, BLOCK_I=bi)
    _bwd_dgate_dup_kernel[(E, route_blocks, _ceil_div(I, bi))](x, gate_up_proj, dhidden, route_ids, expert_offsets, dgu, H, I, K, P, BLOCK_M=bm, BLOCK_I=bi, BLOCK_H=bh)
    bg = 32
    _bwd_dgate_up_kernel[(E, _ceil_div(2 * I, bg), _ceil_div(H, bho))](dgu, x, route_ids, expert_offsets, dgate_up, H, I, K, BLOCK_M=bm, BLOCK_G=bg, BLOCK_H=bho)
    _bwd_dx_scatter_kernel[(E, route_blocks, _ceil_div(H, bho))](dgu, gate_up_proj, route_ids, expert_offsets, dx_accum, H, I, K, P, BLOCK_M=bm, BLOCK_HO=bho, BLOCK_G=bg)
    _cast_kernel[(_ceil_div(T * H, block_flat),)](dx_accum, dx, T * H, BLOCK=block_flat)
    _cast_kernel[(_ceil_div(P, block_flat),)](dtop_accum, dtop, P, BLOCK=block_flat)
    return dx, dgate_up, ddown, dtop
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'olmoe_moe_experts', 'arguments': ['x', 'gate_up_proj', 'down_proj', 'top_k_index', 'top_k_weights'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_up_proj', 'ddown_proj', 'dtop_k_weights'], 'gradient_return_order': ['dx', 'dgate_up_proj', 'ddown_proj', 'None', 'dtop_k_weights'], 'scalars': [], 'deployment_entry': 'olmoe_moe_experts_deployment', 'allowed_primitives': []}


class OlmoeMoeExpertsFunction(torch.autograd.Function):
    """Static autograd wiring for `olmoe_moe_experts`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        out, saved = olmoe_moe_experts_forward_with_saved(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
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
        dx, dgate_up_proj, ddown_proj, dtop_k_weights = olmoe_moe_experts_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_up_proj, ddown_proj, None, dtop_k_weights


def olmoe_moe_experts_deployment(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return OlmoeMoeExpertsFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)


class OlmoeMoeExpertsModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, gate_up_proj, down_proj):
        super().__init__()
        self.gate_up_proj = (gate_up_proj if isinstance(gate_up_proj, torch.nn.Parameter)
                  else torch.nn.Parameter(gate_up_proj))
        self.down_proj = (down_proj if isinstance(down_proj, torch.nn.Parameter)
                  else torch.nn.Parameter(down_proj))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, top_k_index, top_k_weights):
        return olmoe_moe_experts_deployment(x, self.gate_up_proj, self.down_proj, top_k_index, top_k_weights)


DEPLOYMENT_ENTRY = "olmoe_moe_experts_deployment"
