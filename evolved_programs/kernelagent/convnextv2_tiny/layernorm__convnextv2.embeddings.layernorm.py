"""adbench-tools artifact: task convnextv2_tiny.layernorm.convnextv2.embeddings.layernorm, produced by kernelagent.

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
def _ln_forward_channels_first_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    c = tl.arange(0, BLOCK_C)
    p = tl.arange(0, BLOCK_HW)
    hw = tid * BLOCK_HW + p

    mask = (c[:, None] < C) & (hw[None, :] < HW)
    idx = bid * C * HW + c[:, None] * HW + hw[None, :]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + EPS)
    xhat = xc * rstd[None, :]

    w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    y = xhat * w[:, None] + b[:, None]
    tl.store(y_ptr + idx, y, mask=mask)

    mh = bid * HW + hw
    tl.store(mean_ptr + mh, mean, mask=hw < HW)
    tl.store(rstd_ptr + mh, rstd, mask=hw < HW)


@triton.jit
def _ln_forward_lastdim_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    mean_ptr,
    rstd_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tid = tl.program_id(0)

    r = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_N)

    mask = (r[:, None] < M) & (c[None, :] < N)
    idx = r[:, None] * N + c[None, :]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    mean = tl.sum(x, axis=1) / N
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = tl.rsqrt(var + EPS)
    xhat = xc * rstd[:, None]

    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + c, mask=c < N, other=0.0).to(tl.float32)

    y = xhat * w[None, :] + b[None, :]
    tl.store(y_ptr + idx, y, mask=mask)

    tl.store(mean_ptr + r, mean, mask=r < M)
    tl.store(rstd_ptr + r, rstd, mask=r < M)


@triton.jit
def _ln_backward_channels_first_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TILES_PER_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    c = tl.arange(0, BLOCK_C)
    p = tl.arange(0, BLOCK_HW)
    hw = tid * BLOCK_HW + p
    tile_id = bid * TILES_PER_B + tid

    mask = (c[:, None] < C) & (hw[None, :] < HW)
    idx = bid * C * HW + c[:, None] * HW + hw[None, :]

    mh = bid * HW + hw
    mean = tl.load(mean_ptr + mh, mask=hw < HW, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + mh, mask=hw < HW, other=0.0).to(tl.float32)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    dy = tl.where(mask, dy, 0.0)

    xhat = (x - mean[None, :]) * rstd[None, :]
    xhat = tl.where(mask, xhat, 0.0)

    w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    dxhat = dy * w[:, None]

    s1 = tl.sum(dxhat, axis=0)
    s2 = tl.sum(dxhat * xhat, axis=0)
    dx = (dxhat * C - s1[None, :] - xhat * s2[None, :]) * (rstd[None, :] / C)

    tl.store(dx_ptr + idx, dx, mask=mask)

    dw = tl.sum(dy * xhat, axis=1)
    db = tl.sum(dy, axis=1)

    tl.store(partial_dw_ptr + tile_id * C + c, dw, mask=c < C)
    tl.store(partial_db_ptr + tile_id * C + c, db, mask=c < C)


@triton.jit
def _ln_backward_lastdim_kernel(
    x_ptr,
    w_ptr,
    dy_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tid = tl.program_id(0)

    r = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_N)

    mask = (r[:, None] < M) & (c[None, :] < N)
    idx = r[:, None] * N + c[None, :]

    mean = tl.load(mean_ptr + r, mask=r < M, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + r, mask=r < M, other=0.0).to(tl.float32)

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    dy = tl.where(mask, dy, 0.0)

    xhat = (x - mean[:, None]) * rstd[:, None]
    xhat = tl.where(mask, xhat, 0.0)

    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    dxhat = dy * w[None, :]

    s1 = tl.sum(dxhat, axis=1)
    s2 = tl.sum(dxhat * xhat, axis=1)
    dx = (dxhat * N - s1[:, None] - xhat * s2[:, None]) * (rstd[:, None] / N)

    tl.store(dx_ptr + idx, dx, mask=mask)

    dw = tl.sum(dy * xhat, axis=0)
    db = tl.sum(dy, axis=0)

    tl.store(partial_dw_ptr + tid * N + c, dw, mask=c < N)
    tl.store(partial_db_ptr + tid * N + c, db, mask=c < N)


@triton.jit
def _ln_fused_channels_first_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    dy_ptr,
    out_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TILES_PER_B: tl.constexpr,
    NUM_TILES: tl.constexpr,
    EPS: tl.constexpr,
    DX_BASE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    c = tl.arange(0, BLOCK_C)
    p = tl.arange(0, BLOCK_HW)
    hw = tid * BLOCK_HW + p
    tile_id = bid * TILES_PER_B + tid

    mask = (c[:, None] < C) & (hw[None, :] < HW)
    idx = bid * C * HW + c[:, None] * HW + hw[None, :]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean[None, :], 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = tl.rsqrt(var + EPS)
    xhat = xc * rstd[None, :]

    w = tl.load(w_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    y = xhat * w[:, None] + bb[:, None]
    tl.store(out_ptr + idx, y, mask=mask)

    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.where(mask, dy, 0.0)

    dxhat = dy * w[:, None]
    s1 = tl.sum(dxhat, axis=0)
    s2 = tl.sum(dxhat * xhat, axis=0)
    dx = (dxhat * C - s1[None, :] - xhat * s2[None, :]) * (rstd[None, :] / C)

    tl.store(out_ptr + DX_BASE + idx, dx, mask=mask)

    dw = tl.sum(dy * xhat, axis=1)
    db = tl.sum(dy, axis=1)

    tl.store(partial_dw_ptr + tile_id * C + c, dw, mask=c < C)
    tl.store(partial_db_ptr + tile_id * C + c, db, mask=c < C)


@triton.jit
def _ln_fused_lastdim_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    dy_ptr,
    out_ptr,
    partial_dw_ptr,
    partial_db_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    NUM_TILES: tl.constexpr,
    EPS: tl.constexpr,
    DX_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tid = tl.program_id(0)

    r = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_N)

    mask = (r[:, None] < M) & (c[None, :] < N)
    idx = r[:, None] * N + c[None, :]

    x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    mean = tl.sum(x, axis=1) / N
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = tl.rsqrt(var + EPS)
    xhat = xc * rstd[:, None]

    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    bb = tl.load(b_ptr + c, mask=c < N, other=0.0).to(tl.float32)

    y = xhat * w[None, :] + bb[None, :]
    tl.store(out_ptr + idx, y, mask=mask)

    dy = tl.load(dy_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    dy = tl.where(mask, dy, 0.0)

    dxhat = dy * w[None, :]
    s1 = tl.sum(dxhat, axis=1)
    s2 = tl.sum(dxhat * xhat, axis=1)
    dx = (dxhat * N - s1[:, None] - xhat * s2[:, None]) * (rstd[:, None] / N)

    tl.store(out_ptr + DX_BASE + idx, dx, mask=mask)

    dw = tl.sum(dy * xhat, axis=0)
    db = tl.sum(dy, axis=0)

    tl.store(partial_dw_ptr + tid * N + c, dw, mask=c < N)
    tl.store(partial_db_ptr + tid * N + c, db, mask=c < N)


@triton.jit
def _ln_reduce_stage_kernel(
    partial_dw_ptr,
    partial_db_ptr,
    tmp_dw_ptr,
    tmp_db_ptr,
    NUM_TILES: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    gid = tl.program_id(0)

    r = gid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)

    mask = (r[:, None] < NUM_TILES) & (c[None, :] < C)

    dw = tl.load(partial_dw_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    db = tl.load(partial_db_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)

    sdw = tl.sum(dw, axis=0)
    sdb = tl.sum(db, axis=0)

    tl.store(tmp_dw_ptr + gid * C + c, sdw, mask=c < C)
    tl.store(tmp_db_ptr + gid * C + c, sdb, mask=c < C)


@triton.jit
def _ln_reduce_to_out_kernel(
    src_dw_ptr,
    src_db_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    DW_BASE: tl.constexpr,
    DB_BASE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)

    mask = (r[:, None] < ROWS) & (c[None, :] < C)

    dw = tl.load(src_dw_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    db = tl.load(src_db_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)

    sdw = tl.sum(dw, axis=0)
    sdb = tl.sum(db, axis=0)

    tl.store(out_ptr + DW_BASE + c, sdw, mask=c < C)
    tl.store(out_ptr + DB_BASE + c, sdb, mask=c < C)


@triton.jit
def _ln_reduce_to_grads_kernel(
    src_dw_ptr,
    src_db_ptr,
    dw_ptr,
    db_ptr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_C)

    mask = (r[:, None] < ROWS) & (c[None, :] < C)

    dw = tl.load(src_dw_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    db = tl.load(src_db_ptr + r[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)

    sdw = tl.sum(dw, axis=0)
    sdb = tl.sum(db, axis=0)

    tl.store(dw_ptr + c, sdw, mask=c < C)
    tl.store(db_ptr + c, sdb, mask=c < C)


def _next_power_of_2_at_least_16(n):
    block = 1 << (n - 1).bit_length()
    if block < 16:
        block = 16
    return block


def layernorm_forward_with_saved(x, weight, bias, eps):
    n = weight.numel()
    block_c = _next_power_of_2_at_least_16(n)

    y = torch.empty_like(x)

    if x.dim() == 4 and x.shape[1] == n:
        bsz = x.shape[0]
        hw = x.shape[2] * x.shape[3]
        block_hw = 2048 // block_c
        if block_hw > 16:
            block_hw = 16
        if block_hw < 1:
            block_hw = 1
        tiles_per_b = triton.cdiv(hw, block_hw)

        mean = torch.empty((bsz * hw,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((bsz * hw,), device=x.device, dtype=torch.float32)

        _ln_forward_channels_first_kernel[(bsz, tiles_per_b)](
            x,
            weight,
            bias,
            y,
            mean,
            rstd,
            C=n,
            HW=hw,
            EPS=eps,
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            num_warps=8,
        )
    else:
        m = x.numel() // n
        block_m = 16
        num_tiles = triton.cdiv(m, block_m)

        mean = torch.empty((m,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((m,), device=x.device, dtype=torch.float32)

        _ln_forward_lastdim_kernel[(num_tiles,)](
            x,
            weight,
            bias,
            y,
            mean,
            rstd,
            M=m,
            N=n,
            EPS=eps,
            BLOCK_M=block_m,
            BLOCK_N=block_c,
            num_warps=8,
        )

    saved = (x, weight, mean, rstd, eps)
    return y, saved


def layernorm_backward_from_saved(upstream, saved):
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, weight, mean, rstd, eps = saved

    n = weight.numel()
    block_c = _next_power_of_2_at_least_16(n)

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(weight)

    if x.dim() == 4 and x.shape[1] == n:
        bsz = x.shape[0]
        hw = x.shape[2] * x.shape[3]
        block_hw = 2048 // block_c
        if block_hw > 16:
            block_hw = 16
        if block_hw < 1:
            block_hw = 1
        tiles_per_b = triton.cdiv(hw, block_hw)
        num_tiles = bsz * tiles_per_b

        partial_dw = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)
        partial_db = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)

        _ln_backward_channels_first_kernel[(bsz, tiles_per_b)](
            x,
            weight,
            dy,
            mean,
            rstd,
            dx,
            partial_dw,
            partial_db,
            C=n,
            HW=hw,
            TILES_PER_B=tiles_per_b,
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            num_warps=8,
        )
    else:
        m = x.numel() // n
        block_m = 16
        num_tiles = triton.cdiv(m, block_m)

        partial_dw = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)
        partial_db = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)

        _ln_backward_lastdim_kernel[(num_tiles,)](
            x,
            weight,
            dy,
            mean,
            rstd,
            dx,
            partial_dw,
            partial_db,
            M=m,
            N=n,
            BLOCK_M=block_m,
            BLOCK_N=block_c,
            num_warps=8,
        )

    reduce_block = 256

    if num_tiles <= reduce_block:
        br = 1 << (num_tiles - 1).bit_length()
        _ln_reduce_to_grads_kernel[(1,)](
            partial_dw,
            partial_db,
            dw,
            db,
            ROWS=num_tiles,
            C=n,
            BLOCK_R=br,
            BLOCK_C=block_c,
            num_warps=8,
        )
    else:
        groups = triton.cdiv(num_tiles, reduce_block)
        tmp_dw = torch.empty((groups, n), device=x.device, dtype=torch.float32)
        tmp_db = torch.empty((groups, n), device=x.device, dtype=torch.float32)

        _ln_reduce_stage_kernel[(groups,)](
            partial_dw,
            partial_db,
            tmp_dw,
            tmp_db,
            NUM_TILES=num_tiles,
            C=n,
            BLOCK_R=reduce_block,
            BLOCK_C=block_c,
            num_warps=8,
        )

        br = 1 << (groups - 1).bit_length()
        _ln_reduce_to_grads_kernel[(1,)](
            tmp_dw,
            tmp_db,
            dw,
            db,
            ROWS=groups,
            C=n,
            BLOCK_R=br,
            BLOCK_C=block_c,
            num_warps=8,
        )

    return dx, dw, db


def kernel_function(x, weight, bias, eps, upstream):
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    n = weight.numel()
    block_c = _next_power_of_2_at_least_16(n)

    x_numel = x.numel()
    total = x_numel + x_numel + n + n
    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    if x.dim() == 4 and x.shape[1] == n:
        bsz = x.shape[0]
        hw = x.shape[2] * x.shape[3]
        block_hw = 2048 // block_c
        if block_hw > 16:
            block_hw = 16
        if block_hw < 1:
            block_hw = 1
        tiles_per_b = triton.cdiv(hw, block_hw)
        num_tiles = bsz * tiles_per_b

        partial_dw = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)
        partial_db = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)

        _ln_fused_channels_first_kernel[(bsz, tiles_per_b)](
            x,
            weight,
            bias,
            dy,
            out,
            partial_dw,
            partial_db,
            C=n,
            HW=hw,
            TILES_PER_B=tiles_per_b,
            NUM_TILES=num_tiles,
            EPS=eps,
            DX_BASE=x_numel,
            BLOCK_C=block_c,
            BLOCK_HW=block_hw,
            num_warps=8,
        )
    else:
        m = x_numel // n
        block_m = 16
        num_tiles = triton.cdiv(m, block_m)

        partial_dw = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)
        partial_db = torch.empty((num_tiles, n), device=x.device, dtype=torch.float32)

        _ln_fused_lastdim_kernel[(num_tiles,)](
            x,
            weight,
            bias,
            dy,
            out,
            partial_dw,
            partial_db,
            M=m,
            N=n,
            NUM_TILES=num_tiles,
            EPS=eps,
            DX_BASE=x_numel,
            BLOCK_M=block_m,
            BLOCK_N=block_c,
            num_warps=8,
        )

    reduce_block = 256
    dw_base = 2 * x_numel
    db_base = 2 * x_numel + n

    if num_tiles <= reduce_block:
        br = 1 << (num_tiles - 1).bit_length()
        _ln_reduce_to_out_kernel[(1,)](
            partial_dw,
            partial_db,
            out,
            ROWS=num_tiles,
            C=n,
            DW_BASE=dw_base,
            DB_BASE=db_base,
            BLOCK_R=br,
            BLOCK_C=block_c,
            num_warps=8,
        )
    else:
        groups = triton.cdiv(num_tiles, reduce_block)
        tmp_dw = torch.empty((groups, n), device=x.device, dtype=torch.float32)
        tmp_db = torch.empty((groups, n), device=x.device, dtype=torch.float32)

        _ln_reduce_stage_kernel[(groups,)](
            partial_dw,
            partial_db,
            tmp_dw,
            tmp_db,
            NUM_TILES=num_tiles,
            C=n,
            BLOCK_R=reduce_block,
            BLOCK_C=block_c,
            num_warps=8,
        )

        br = 1 << (groups - 1).bit_length()
        _ln_reduce_to_out_kernel[(1,)](
            tmp_dw,
            tmp_db,
            out,
            ROWS=groups,
            C=n,
            DW_BASE=dw_base,
            DB_BASE=db_base,
            BLOCK_R=br,
            BLOCK_C=block_c,
            num_warps=8,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'layernorm', 'arguments': ['x', 'weight', 'bias', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias', 'None'], 'scalars': ['eps'], 'deployment_entry': 'layernorm_deployment', 'allowed_primitives': []}


def _backward_scalars(backward, **scalars):
    """The declared scalars `backward` accepts: those it names, or all of them
    when it takes **kwargs -- as Gate 1 passes them."""
    import inspect

    params = inspect.signature(backward).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return scalars
    return {name: value for name, value in scalars.items() if name in params}


class LayernormFunction(torch.autograd.Function):
    """Static autograd wiring for `layernorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, saved = layernorm_forward_with_saved(x, weight, bias, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
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
        dx, dweight, dbias = layernorm_backward_from_saved(
            dy, saved, **_backward_scalars(layernorm_backward_from_saved, eps=ctx.eps)
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias, None


def layernorm_deployment(x, weight, bias, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = LayernormFunction.apply(x, weight, bias, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class LayernormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return layernorm_deployment(x, self.weight, self.bias, self.eps)


DEPLOYMENT_ENTRY = "layernorm_deployment"
