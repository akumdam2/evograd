"""adbench-tools artifact: task convnextv2_tiny.grn.convnextv2.encoder.stages.0.layers.grn, produced by kernelagent.

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


# ---------------------------
# Forward reduction kernels
# ---------------------------

@triton.jit
def _forward_sx2_part_kernel(
    x_ptr,
    sx2_part_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    hb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    sx2 = tl.sum(x * x, axis=0)

    ci = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    out = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + hb) * BLOCK_C + ci
    tl.store(sx2_part_ptr + out, sx2, mask=c_mask)


@triton.jit
def _finalize_g_kernel(
    sx2_part_ptr,
    g_ptr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HWB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)

    offs_hb = tl.arange(0, BLOCK_HWB)
    ci = tl.arange(0, BLOCK_C)
    c = cb * BLOCK_C + ci

    pidx = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + offs_hb[:, None]) * BLOCK_C + ci[None, :]
    mask = (offs_hb[:, None] < NUM_HW_BLOCKS) & (c[None, :] < C)

    sx2p = tl.load(sx2_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)
    sx2 = tl.sum(sx2p, axis=0)
    g = tl.sqrt(sx2)

    tl.store(g_ptr + n * C + c, g, mask=c < C)


@triton.jit
def _invd_only_kernel(
    g_ptr,
    inv_d_ptr,
    C: tl.constexpr,
    BLOCK_C_ALL: tl.constexpr,
):
    n = tl.program_id(0)

    offs = tl.arange(0, BLOCK_C_ALL)
    mask = offs < C

    g = tl.load(g_ptr + n * C + offs, mask=mask, other=0.0).to(tl.float32)
    sum_g = tl.sum(g, axis=0)
    inv_d = 1.0 / (sum_g / C + 1.0e-6)

    tl.store(inv_d_ptr + n, inv_d)


@triton.jit
def _forward_y_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    g_ptr,
    inv_d_ptr,
    y_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hb = tl.program_id(1)
    cb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    c_mask = offs_c < C
    w = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_d_ptr + n).to(tl.float32)

    nx = g * inv_d
    y = x + x * w[None, :] * nx[None, :] + b[None, :]

    tl.store(y_ptr + idx, y, mask=mask)


# ---------------------------
# Backward reduction kernels
# ---------------------------

@triton.jit
def _backward_t_dbn_part_kernel(
    x_ptr,
    dy_ptr,
    t_part_ptr,
    dbn_part_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    hb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    t = tl.sum(dy * x, axis=0)
    dbn = tl.sum(dy, axis=0)

    ci = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    out = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + hb) * BLOCK_C + ci

    tl.store(t_part_ptr + out, t, mask=c_mask)
    tl.store(dbn_part_ptr + out, dbn, mask=c_mask)


@triton.jit
def _finalize_t_dbn_kernel(
    t_part_ptr,
    dbn_part_ptr,
    t_ptr,
    dbn_ptr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HWB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)

    offs_hb = tl.arange(0, BLOCK_HWB)
    ci = tl.arange(0, BLOCK_C)
    c = cb * BLOCK_C + ci

    pidx = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + offs_hb[:, None]) * BLOCK_C + ci[None, :]
    mask = (offs_hb[:, None] < NUM_HW_BLOCKS) & (c[None, :] < C)

    tp = tl.load(t_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)
    dbnp = tl.load(dbn_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)

    t = tl.sum(tp, axis=0)
    dbn = tl.sum(dbnp, axis=0)

    out = n * C + c
    tl.store(t_ptr + out, t, mask=c < C)
    tl.store(dbn_ptr + out, dbn, mask=c < C)


@triton.jit
def _q_only_kernel(
    g_ptr,
    t_ptr,
    weight_ptr,
    q_ptr,
    C: tl.constexpr,
    BLOCK_C_ALL: tl.constexpr,
):
    n = tl.program_id(0)

    offs = tl.arange(0, BLOCK_C_ALL)
    mask = offs < C

    g = tl.load(g_ptr + n * C + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + n * C + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    q = tl.sum(w * t * g, axis=0)
    tl.store(q_ptr + n, q)


@triton.jit
def _dx_only_kernel(
    x_ptr,
    dy_ptr,
    weight_ptr,
    g_ptr,
    inv_d_ptr,
    t_ptr,
    q_ptr,
    dx_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hb = tl.program_id(1)
    cb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    c_mask = offs_c < C
    w = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_d_ptr + n).to(tl.float32)
    q = tl.load(q_ptr + n).to(tl.float32)

    nx = g * inv_d
    d_g = w * t * inv_d - q * inv_d * inv_d / C
    via_g = tl.where(g > 0.0, d_g / g, 0.0)

    dx = dy * (1.0 + w[None, :] * nx[None, :]) + x * via_g[None, :]
    tl.store(dx_ptr + idx, dx, mask=mask)


@triton.jit
def _param_grad_kernel(
    t_ptr,
    dbn_ptr,
    g_ptr,
    inv_d_ptr,
    result_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    RESULT_DW_OFFSET: tl.constexpr,
    RESULT_DB_OFFSET: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cb = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = offs_n[:, None] * C + offs_c[None, :]
    mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)

    t = tl.load(t_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dbn = tl.load(dbn_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_d_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    dw = tl.sum(t * g * inv_d[:, None], axis=0)
    db = tl.sum(dbn, axis=0)

    c_mask = offs_c < C
    tl.store(result_ptr + RESULT_DW_OFFSET + offs_c, dw, mask=c_mask)
    tl.store(result_ptr + RESULT_DB_OFFSET + offs_c, db, mask=c_mask)


@triton.jit
def _param_grad_separate_kernel(
    t_ptr,
    dbn_ptr,
    g_ptr,
    inv_d_ptr,
    dw_ptr,
    db_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cb = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = offs_n[:, None] * C + offs_c[None, :]
    mask = (offs_n[:, None] < N) & (offs_c[None, :] < C)

    t = tl.load(t_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dbn = tl.load(dbn_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_d_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)

    dw = tl.sum(t * g * inv_d[:, None], axis=0)
    db = tl.sum(dbn, axis=0)

    c_mask = offs_c < C
    tl.store(dw_ptr + offs_c, dw, mask=c_mask)
    tl.store(db_ptr + offs_c, db, mask=c_mask)


# ---------------------------
# Combined forward/backward kernels used by kernel_function
# ---------------------------

@triton.jit
def _partial_reduce_kernel(
    x_ptr,
    dy_ptr,
    sx2_part_ptr,
    t_part_ptr,
    dbn_part_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)
    hb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    sx2 = tl.sum(x * x, axis=0)
    t = tl.sum(dy * x, axis=0)
    dbn = tl.sum(dy, axis=0)

    ci = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    out = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + hb) * BLOCK_C + ci

    tl.store(sx2_part_ptr + out, sx2, mask=c_mask)
    tl.store(t_part_ptr + out, t, mask=c_mask)
    tl.store(dbn_part_ptr + out, dbn, mask=c_mask)


@triton.jit
def _finalize_reduce_kernel(
    sx2_part_ptr,
    t_part_ptr,
    dbn_part_ptr,
    g_ptr,
    t_ptr,
    dbn_ptr,
    C: tl.constexpr,
    NUM_HW_BLOCKS: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    BLOCK_HWB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cb = tl.program_id(1)

    offs_hb = tl.arange(0, BLOCK_HWB)
    ci = tl.arange(0, BLOCK_C)
    c = cb * BLOCK_C + ci

    pidx = ((n * NUM_C_BLOCKS + cb) * NUM_HW_BLOCKS + offs_hb[:, None]) * BLOCK_C + ci[None, :]
    mask = (offs_hb[:, None] < NUM_HW_BLOCKS) & (c[None, :] < C)

    sx2p = tl.load(sx2_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)
    tp = tl.load(t_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)
    dbnp = tl.load(dbn_part_ptr + pidx, mask=mask, other=0.0).to(tl.float32)

    sx2 = tl.sum(sx2p, axis=0)
    t = tl.sum(tp, axis=0)
    dbn = tl.sum(dbnp, axis=0)
    g = tl.sqrt(sx2)

    out = n * C + c
    c_mask = c < C

    tl.store(g_ptr + out, g, mask=c_mask)
    tl.store(t_ptr + out, t, mask=c_mask)
    tl.store(dbn_ptr + out, dbn, mask=c_mask)


@triton.jit
def _invd_q_kernel(
    g_ptr,
    t_ptr,
    weight_ptr,
    inv_d_ptr,
    q_ptr,
    C: tl.constexpr,
    BLOCK_C_ALL: tl.constexpr,
):
    n = tl.program_id(0)

    offs = tl.arange(0, BLOCK_C_ALL)
    mask = offs < C

    g = tl.load(g_ptr + n * C + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + n * C + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sum_g = tl.sum(g, axis=0)
    q = tl.sum(w * t * g, axis=0)
    inv_d = 1.0 / (sum_g / C + 1.0e-6)

    tl.store(inv_d_ptr + n, inv_d)
    tl.store(q_ptr + n, q)


@triton.jit
def _ydx_kernel(
    x_ptr,
    dy_ptr,
    weight_ptr,
    bias_ptr,
    g_ptr,
    inv_d_ptr,
    t_ptr,
    q_ptr,
    result_ptr,
    HW: tl.constexpr,
    C: tl.constexpr,
    RESULT_DX_OFFSET: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    hb = tl.program_id(1)
    cb = tl.program_id(2)

    offs_hw = hb * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)

    idx = n * HW * C + offs_hw[:, None] * C + offs_c[None, :]
    mask = (offs_hw[:, None] < HW) & (offs_c[None, :] < C)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    c_mask = offs_c < C
    w = tl.load(weight_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + n * C + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    inv_d = tl.load(inv_d_ptr + n).to(tl.float32)
    q = tl.load(q_ptr + n).to(tl.float32)

    nx = g * inv_d
    y = x + x * w[None, :] * nx[None, :] + b[None, :]

    d_g = w * t * inv_d - q * inv_d * inv_d / C
    via_g = tl.where(g > 0.0, d_g / g, 0.0)
    dx = dy * (1.0 + w[None, :] * nx[None, :]) + x * via_g[None, :]

    tl.store(result_ptr + idx, y, mask=mask)
    tl.store(result_ptr + RESULT_DX_OFFSET + idx, dx, mask=mask)


# ---------------------------
# Public pair API required by evaluator
# ---------------------------

def grn_forward_with_saved(x, weight, bias):
    N = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    HW = H * W

    BLOCK_C = 32
    BLOCK_HW_RED = 32
    BLOCK_HW_ELEM = 16

    num_hw_blocks = triton.cdiv(HW, BLOCK_HW_RED)
    num_c_blocks = triton.cdiv(C, BLOCK_C)
    part_size = N * num_c_blocks * num_hw_blocks * BLOCK_C

    y = torch.empty_like(x)
    sx2_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)
    g = torch.empty((N, C), device=x.device, dtype=torch.float32)
    inv_d = torch.empty((N,), device=x.device, dtype=torch.float32)

    block_hwb = triton.next_power_of_2(num_hw_blocks)
    block_c_all = triton.next_power_of_2(C)

    _forward_sx2_part_kernel[(N, num_c_blocks, num_hw_blocks)](
        x,
        sx2_part,
        HW,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HW=BLOCK_HW_RED,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _finalize_g_kernel[(N, num_c_blocks)](
        sx2_part,
        g,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HWB=block_hwb,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _invd_only_kernel[(N,)](
        g,
        inv_d,
        C,
        BLOCK_C_ALL=block_c_all,
        num_warps=4,
    )

    _forward_y_kernel[(N, triton.cdiv(HW, BLOCK_HW_ELEM), num_c_blocks)](
        x,
        weight,
        bias,
        g,
        inv_d,
        y,
        HW,
        C,
        BLOCK_HW=BLOCK_HW_ELEM,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    saved = (x, weight, g, inv_d)
    return y, saved


def grn_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, weight, g, inv_d = saved

    N = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    HW = H * W

    BLOCK_C = 32
    BLOCK_HW_RED = 32
    BLOCK_HW_ELEM = 16

    num_hw_blocks = triton.cdiv(HW, BLOCK_HW_RED)
    num_c_blocks = triton.cdiv(C, BLOCK_C)
    part_size = N * num_c_blocks * num_hw_blocks * BLOCK_C

    t_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)
    dbn_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)
    t = torch.empty((N, C), device=x.device, dtype=torch.float32)
    dbn = torch.empty((N, C), device=x.device, dtype=torch.float32)
    q = torch.empty((N,), device=x.device, dtype=torch.float32)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(weight)

    block_hwb = triton.next_power_of_2(num_hw_blocks)
    block_c_all = triton.next_power_of_2(C)
    block_n = triton.next_power_of_2(N)

    _backward_t_dbn_part_kernel[(N, num_c_blocks, num_hw_blocks)](
        x,
        dy,
        t_part,
        dbn_part,
        HW,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HW=BLOCK_HW_RED,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _finalize_t_dbn_kernel[(N, num_c_blocks)](
        t_part,
        dbn_part,
        t,
        dbn,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HWB=block_hwb,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _q_only_kernel[(N,)](
        g,
        t,
        weight,
        q,
        C,
        BLOCK_C_ALL=block_c_all,
        num_warps=4,
    )

    _dx_only_kernel[(N, triton.cdiv(HW, BLOCK_HW_ELEM), num_c_blocks)](
        x,
        dy,
        weight,
        g,
        inv_d,
        t,
        q,
        dx,
        HW,
        C,
        BLOCK_HW=BLOCK_HW_ELEM,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _param_grad_separate_kernel[(num_c_blocks,)](
        t,
        dbn,
        g,
        inv_d,
        dw,
        db,
        N,
        C,
        BLOCK_N=block_n,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    return dx, dw, db


# ---------------------------
# Required fused forward+backward step
# ---------------------------

def kernel_function(x, weight, bias, upstream):
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    N = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    HW = H * W
    numel = x.numel()

    w_numel = weight.numel()
    b_numel = bias.numel()
    total = numel + numel + w_numel + b_numel

    result = torch.empty((total,), device=x.device, dtype=torch.float32)

    BLOCK_C = 32
    BLOCK_HW_RED = 32
    BLOCK_HW_ELEM = 16

    num_hw_blocks = triton.cdiv(HW, BLOCK_HW_RED)
    num_c_blocks = triton.cdiv(C, BLOCK_C)

    part_size = N * num_c_blocks * num_hw_blocks * BLOCK_C

    sx2_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)
    t_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)
    dbn_part = torch.empty((part_size,), device=x.device, dtype=torch.float32)

    g = torch.empty((N, C), device=x.device, dtype=torch.float32)
    t = torch.empty((N, C), device=x.device, dtype=torch.float32)
    dbn = torch.empty((N, C), device=x.device, dtype=torch.float32)
    inv_d = torch.empty((N,), device=x.device, dtype=torch.float32)
    q = torch.empty((N,), device=x.device, dtype=torch.float32)

    block_hwb = triton.next_power_of_2(num_hw_blocks)
    block_c_all = triton.next_power_of_2(C)
    block_n = triton.next_power_of_2(N)

    # Full single-pass fusion is not possible here without inter-block synchronization:
    # GRN needs reductions over HW and C before the elementwise y/dx epilogue.
    # We still fuse compatible stages: x^2, dy*x, and dy reductions share one pass;
    # y and dx are emitted in one epilogue pass.
    _partial_reduce_kernel[(N, num_c_blocks, num_hw_blocks)](
        x,
        dy,
        sx2_part,
        t_part,
        dbn_part,
        HW,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HW=BLOCK_HW_RED,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _finalize_reduce_kernel[(N, num_c_blocks)](
        sx2_part,
        t_part,
        dbn_part,
        g,
        t,
        dbn,
        C,
        num_hw_blocks,
        num_c_blocks,
        BLOCK_HWB=block_hwb,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _invd_q_kernel[(N,)](
        g,
        t,
        weight,
        inv_d,
        q,
        C,
        BLOCK_C_ALL=block_c_all,
        num_warps=4,
    )

    _ydx_kernel[(N, triton.cdiv(HW, BLOCK_HW_ELEM), num_c_blocks)](
        x,
        dy,
        weight,
        bias,
        g,
        inv_d,
        t,
        q,
        result,
        HW,
        C,
        numel,
        BLOCK_HW=BLOCK_HW_ELEM,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _param_grad_kernel[(num_c_blocks,)](
        t,
        dbn,
        g,
        inv_d,
        result,
        N,
        C,
        2 * numel,
        2 * numel + w_numel,
        BLOCK_N=block_n,
        BLOCK_C=BLOCK_C,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'grn', 'arguments': ['x', 'weight', 'bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias'], 'scalars': [], 'deployment_entry': 'grn_deployment', 'allowed_primitives': []}


class GrnFunction(torch.autograd.Function):
    """Static autograd wiring for `grn`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        out, saved = grn_forward_with_saved(x, weight, bias)
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
        dx, dweight, dbias = grn_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias


def grn_deployment(x, weight, bias):
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
        bias = bias.reshape(-1, bias.shape[-1])
    out = GrnFunction.apply(x, weight, bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class GrnModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return grn_deployment(x, self.weight, self.bias)


DEPLOYMENT_ENTRY = "grn_deployment"
