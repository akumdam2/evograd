import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _causal_conv1d_fwd_kernel_fn(
    x,
    weight,
    y,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    ws0: tl.constexpr,
    ws1: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_C, BLOCK_T), tl.float32)

    for jj in range(0, K):
        in_t = offs_t - (K - 1) + jj
        x_ptrs = x + pid_b * xs0 + offs_c[:, None] * xs1 + in_t[None, :] * xs2
        w_ptrs = weight + offs_c * ws0 + jj * ws1

        x_mask = (offs_c[:, None] < C) & (in_t[None, :] >= 0)
        w_mask = offs_c < C

        xv = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc += xv * wv[:, None]

    out = acc
    if SILU != 0:
        sig = 1.0 / (1.0 + tl.exp(-acc))
        out = acc * sig

    y_ptrs = y + (pid_b * C + offs_c[:, None]) * T + offs_t[None, :]
    y_mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    tl.store(y_ptrs, out, mask=y_mask)


_causal_conv1d_fwd_kernel = triton.jit(_causal_conv1d_fwd_kernel_fn)


def _causal_conv1d_ds_kernel_fn(
    x,
    weight,
    dy,
    ds,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    ws0: tl.constexpr,
    ws1: tl.constexpr,
    dys0: tl.constexpr,
    dys1: tl.constexpr,
    dys2: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_C, BLOCK_T), tl.float32)

    for jj in range(0, K):
        in_t = offs_t - (K - 1) + jj
        x_ptrs = x + pid_b * xs0 + offs_c[:, None] * xs1 + in_t[None, :] * xs2
        w_ptrs = weight + offs_c * ws0 + jj * ws1

        x_mask = (offs_c[:, None] < C) & (in_t[None, :] >= 0) & (in_t[None, :] < T)
        w_mask = offs_c < C

        xv = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc += xv * wv[:, None]

    dy_ptrs = dy + pid_b * dys0 + offs_c[:, None] * dys1 + offs_t[None, :] * dys2
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    dyv = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    dsv = dyv
    if SILU != 0:
        sig = 1.0 / (1.0 + tl.exp(-acc))
        dsv = dyv * sig * (1.0 + acc * (1.0 - sig))

    ds_ptrs = ds + (pid_b * C + offs_c[:, None]) * T + offs_t[None, :]
    tl.store(ds_ptrs, dsv, mask=mask)


_causal_conv1d_ds_kernel = triton.jit(_causal_conv1d_ds_kernel_fn)


def _causal_conv1d_dx_kernel_fn(
    weight,
    ds,
    dx,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    ws0: tl.constexpr,
    ws1: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    acc = tl.zeros((BLOCK_C, BLOCK_T), tl.float32)

    for jj in range(0, K):
        out_t = offs_t + (K - 1) - jj
        ds_ptrs = ds + (pid_b * C + offs_c[:, None]) * T + out_t[None, :]
        w_ptrs = weight + offs_c * ws0 + jj * ws1

        ds_mask = (offs_c[:, None] < C) & (out_t[None, :] < T)
        w_mask = offs_c < C

        dsv = tl.load(ds_ptrs, mask=ds_mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
        acc += dsv * wv[:, None]

    dx_ptrs = dx + (pid_b * C + offs_c[:, None]) * T + offs_t[None, :]
    mask = (offs_c[:, None] < C) & (offs_t[None, :] < T)
    tl.store(dx_ptrs, acc, mask=mask)


_causal_conv1d_dx_kernel = triton.jit(_causal_conv1d_dx_kernel_fn)


def _causal_conv1d_dw_partial_kernel_fn(
    x,
    ds,
    partial,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BTOTAL: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_j = tl.program_id(2)

    offs_n = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    b = offs_n // T
    t = offs_n - b * T
    in_t = t - (K - 1) + pid_j

    ds_ptrs = ds + (b[None, :] * C + offs_c[:, None]) * T + t[None, :]
    x_ptrs = x + b[None, :] * xs0 + offs_c[:, None] * xs1 + in_t[None, :] * xs2

    mask = (
        (offs_n[None, :] < BTOTAL)
        & (offs_c[:, None] < C)
        & (in_t[None, :] >= 0)
        & (in_t[None, :] < T)
    )

    dsv = tl.load(ds_ptrs, mask=mask, other=0.0).to(tl.float32)
    xv = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    vals = dsv * xv
    sums = tl.sum(vals, axis=1)

    partial_ptrs = partial + pid_bt * C * K + offs_c * K + pid_j
    tl.store(partial_ptrs, sums, mask=offs_c < C)


_causal_conv1d_dw_partial_kernel = triton.jit(_causal_conv1d_dw_partial_kernel_fn)


def _causal_conv1d_dw_partial4_kernel_fn(
    x,
    ds,
    partial,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BTOTAL: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    b = offs_n // T
    t = offs_n - b * T

    ds_mask = (offs_n[None, :] < BTOTAL) & (offs_c[:, None] < C)
    dsv = tl.load(
        ds + (b[None, :] * C + offs_c[:, None]) * T + t[None, :],
        mask=ds_mask,
        other=0.0,
    ).to(tl.float32)

    base = b[None, :] * xs0 + offs_c[:, None] * xs1
    out_base = partial + pid_bt * C * K + offs_c * K

    in_t = t - 3
    xv = tl.load(
        x + base + in_t[None, :] * xs2,
        mask=ds_mask & (in_t[None, :] >= 0),
        other=0.0,
    ).to(tl.float32)
    tl.store(out_base + 0, tl.sum(dsv * xv, axis=1), mask=offs_c < C)

    in_t = t - 2
    xv = tl.load(
        x + base + in_t[None, :] * xs2,
        mask=ds_mask & (in_t[None, :] >= 0),
        other=0.0,
    ).to(tl.float32)
    tl.store(out_base + 1, tl.sum(dsv * xv, axis=1), mask=offs_c < C)

    in_t = t - 1
    xv = tl.load(
        x + base + in_t[None, :] * xs2,
        mask=ds_mask & (in_t[None, :] >= 0),
        other=0.0,
    ).to(tl.float32)
    tl.store(out_base + 2, tl.sum(dsv * xv, axis=1), mask=offs_c < C)

    xv = tl.load(
        x + base + t[None, :] * xs2,
        mask=ds_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(out_base + 3, tl.sum(dsv * xv, axis=1), mask=offs_c < C)


_causal_conv1d_dw_partial4_kernel = triton.jit(_causal_conv1d_dw_partial4_kernel_fn)


def _causal_conv1d_dx_dw_partial4_kernel_fn(
    x,
    weight,
    ds,
    dx,
    partial,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BTOTAL: tl.constexpr,
    xs0: tl.constexpr,
    xs1: tl.constexpr,
    xs2: tl.constexpr,
    ws0: tl.constexpr,
    ws1: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    b = offs_n // T
    t = offs_n - b * T
    cmask = offs_c < C
    mask = (offs_n[None, :] < BTOTAL) & cmask[:, None]

    w0 = tl.load(weight + offs_c * ws0 + 0 * ws1, mask=cmask, other=0.0).to(tl.float32)
    w1 = tl.load(weight + offs_c * ws0 + 1 * ws1, mask=cmask, other=0.0).to(tl.float32)
    w2 = tl.load(weight + offs_c * ws0 + 2 * ws1, mask=cmask, other=0.0).to(tl.float32)
    w3 = tl.load(weight + offs_c * ws0 + 3 * ws1, mask=cmask, other=0.0).to(tl.float32)

    ds_base = (b[None, :] * C + offs_c[:, None]) * T
    acc = tl.zeros((BLOCK_C, BLOCK_BT), tl.float32)

    out_t = t + 3
    d0 = tl.load(ds + ds_base + out_t[None, :], mask=mask & (out_t[None, :] < T), other=0.0).to(tl.float32)
    acc += d0 * w0[:, None]

    out_t = t + 2
    d1 = tl.load(ds + ds_base + out_t[None, :], mask=mask & (out_t[None, :] < T), other=0.0).to(tl.float32)
    acc += d1 * w1[:, None]

    out_t = t + 1
    d2 = tl.load(ds + ds_base + out_t[None, :], mask=mask & (out_t[None, :] < T), other=0.0).to(tl.float32)
    acc += d2 * w2[:, None]

    d3 = tl.load(ds + ds_base + t[None, :], mask=mask, other=0.0).to(tl.float32)
    acc += d3 * w3[:, None]

    tl.store(dx + ds_base + t[None, :], acc, mask=mask)

    x_base = b[None, :] * xs0 + offs_c[:, None] * xs1
    out_base = partial + pid_bt * C * 4 + offs_c * 4

    in_t = t - 3
    xv = tl.load(x + x_base + in_t[None, :] * xs2, mask=mask & (in_t[None, :] >= 0), other=0.0).to(tl.float32)
    tl.store(out_base + 0, tl.sum(d3 * xv, axis=1), mask=cmask)

    in_t = t - 2
    xv = tl.load(x + x_base + in_t[None, :] * xs2, mask=mask & (in_t[None, :] >= 0), other=0.0).to(tl.float32)
    tl.store(out_base + 1, tl.sum(d3 * xv, axis=1), mask=cmask)

    in_t = t - 1
    xv = tl.load(x + x_base + in_t[None, :] * xs2, mask=mask & (in_t[None, :] >= 0), other=0.0).to(tl.float32)
    tl.store(out_base + 2, tl.sum(d3 * xv, axis=1), mask=cmask)

    xv = tl.load(x + x_base + t[None, :] * xs2, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_base + 3, tl.sum(d3 * xv, axis=1), mask=cmask)


_causal_conv1d_dx_dw_partial4_kernel = triton.jit(_causal_conv1d_dx_dw_partial4_kernel_fn)


def _causal_conv1d_dw_reduce_kernel_fn(
    partial,
    dweight,
    NUM_BT_TILES: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    dws0: tl.constexpr,
    dws1: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_p = tl.arange(0, BLOCK_P)

    ptrs = partial + offs_p[None, :] * C * K + offs_c[:, None] * K + pid_j
    mask = (offs_c[:, None] < C) & (offs_p[None, :] < NUM_BT_TILES)

    vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    sums = tl.sum(vals, axis=1)

    out_ptrs = dweight + offs_c * dws0 + pid_j * dws1
    tl.store(out_ptrs, sums, mask=offs_c < C)


_causal_conv1d_dw_reduce_kernel = triton.jit(_causal_conv1d_dw_reduce_kernel_fn)


def _validate_inputs(x, weight):
    if x.ndim != 3:
        raise ValueError(f"x must be [B, C, T], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be [C, K], got {tuple(weight.shape)}")
    if weight.shape[0] != x.shape[1]:
        raise ValueError(
            f"weight must be [C, K] with C={x.shape[1]}, got {tuple(weight.shape)}"
        )
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same CUDA device")
    if not torch.is_floating_point(x) or not torch.is_floating_point(weight):
        raise ValueError("x and weight must be floating point tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x dtype must be float16, bfloat16, or float32")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("weight dtype must be float16, bfloat16, or float32")
    B = x.shape[0]
    C = x.shape[1]
    T = x.shape[2]
    K = weight.shape[1]
    if B <= 0 or C <= 0 or T <= 0 or K <= 0:
        raise ValueError("B, C, T, and K must be positive")
    if K > 32:
        raise ValueError("K greater than 32 is unsupported")
    return B, C, T, K


def causal_conv1d_forward_with_saved(x, weight, silu=1):
    B, C, T, K = _validate_inputs(x, weight)

    y = torch.empty((B, C, T), device=x.device, dtype=x.dtype)

    xs0 = x.stride(0)
    xs1 = x.stride(1)
    xs2 = x.stride(2)
    ws0 = weight.stride(0)
    ws1 = weight.stride(1)

    block_c = 8
    block_t = 128
    grid = (triton.cdiv(C, block_c), triton.cdiv(T, block_t), B)
    silu_flag = 1 if silu else 0

    _causal_conv1d_fwd_kernel[grid](
        x,
        weight,
        y,
        B,
        C,
        T,
        K,
        xs0,
        xs1,
        xs2,
        ws0,
        ws1,
        silu_flag,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
    )

    saved_tensors = (x, weight)
    return y, saved_tensors


def causal_conv1d_backward_from_saved(dy, saved_tensors, silu=1):
    if not isinstance(saved_tensors, (tuple, list)):
        raise ValueError("saved_tensors must be a tuple or list")
    if len(saved_tensors) != 2:
        raise ValueError("saved_tensors has unexpected length")

    x = saved_tensors[0]
    weight = saved_tensors[1]
    B = x.shape[0]
    C = x.shape[1]
    T = x.shape[2]
    K = weight.shape[1]
    xs0 = x.stride(0)
    xs1 = x.stride(1)
    xs2 = x.stride(2)
    ws0 = weight.stride(0)
    ws1 = weight.stride(1)

    if dy.ndim != 3:
        raise ValueError(f"dy must be [B, C, T], got {tuple(dy.shape)}")
    if dy.shape[0] != B or dy.shape[1] != C or dy.shape[2] != T:
        raise ValueError("dy shape does not match saved forward shape")
    if not dy.is_cuda:
        raise ValueError("dy must be a CUDA tensor")
    if dy.device != x.device or dy.device != weight.device:
        raise ValueError("dy, x, and weight must be on the same CUDA device")
    if not torch.is_floating_point(dy):
        raise ValueError("dy must be floating point")
    if dy.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("dy dtype must be float16, bfloat16, or float32")
    if x.shape[0] != B or x.shape[1] != C or x.shape[2] != T:
        raise ValueError("saved x shape is inconsistent")
    if weight.shape[0] != C or weight.shape[1] != K:
        raise ValueError("saved weight shape is inconsistent")
    if K > 32:
        raise ValueError("K greater than 32 is unsupported")

    ds = torch.empty((B, C, T), device=dy.device, dtype=dy.dtype)
    dx = torch.empty((B, C, T), device=x.device, dtype=x.dtype)
    dweight = torch.empty((C, K), device=weight.device, dtype=weight.dtype)

    dys0 = dy.stride(0)
    dys1 = dy.stride(1)
    dys2 = dy.stride(2)

    block_c = 8
    block_t = 128
    silu_flag = 1 if silu else 0

    grid_bct = (triton.cdiv(C, block_c), triton.cdiv(T, block_t), B)

    _causal_conv1d_ds_kernel[grid_bct](
        x,
        weight,
        dy,
        ds,
        B,
        C,
        T,
        K,
        xs0,
        xs1,
        xs2,
        ws0,
        ws1,
        dys0,
        dys1,
        dys2,
        silu_flag,
        BLOCK_C=block_c,
        BLOCK_T=block_t,
        num_warps=4,
    )

    block_bt = 128
    btotal = B * T
    num_bt_tiles = triton.cdiv(btotal, block_bt)
    block_p = 1 << (int(num_bt_tiles) - 1).bit_length()
    if block_p > 4096:
        raise ValueError("B*T too large for this dweight reduction")

    partial = torch.empty((num_bt_tiles, C, K), device=x.device, dtype=torch.float32)

    if K == 4:
        grid_partial = (num_bt_tiles, triton.cdiv(C, block_c))
        _causal_conv1d_dx_dw_partial4_kernel[grid_partial](
            x,
            weight,
            ds,
            dx,
            partial,
            B,
            C,
            T,
            btotal,
            xs0,
            xs1,
            xs2,
            ws0,
            ws1,
            BLOCK_C=block_c,
            BLOCK_BT=block_bt,
            num_warps=4,
        )
    else:
        _causal_conv1d_dx_kernel[grid_bct](
            weight,
            ds,
            dx,
            B,
            C,
            T,
            K,
            ws0,
            ws1,
            BLOCK_C=block_c,
            BLOCK_T=block_t,
            num_warps=4,
        )

        grid_partial = (num_bt_tiles, triton.cdiv(C, block_c), K)
        _causal_conv1d_dw_partial_kernel[grid_partial](
            x,
            ds,
            partial,
            B,
            C,
            T,
            K,
            btotal,
            xs0,
            xs1,
            xs2,
            BLOCK_C=block_c,
            BLOCK_BT=block_bt,
            num_warps=4,
        )

    dws0 = dweight.stride(0)
    dws1 = dweight.stride(1)
    grid_reduce = (triton.cdiv(C, block_c), K)

    reduce_warps = 1
    if block_p >= 2048:
        reduce_warps = 8
    elif block_p >= 512:
        reduce_warps = 4
    elif block_p >= 128:
        reduce_warps = 2

    _causal_conv1d_dw_reduce_kernel[grid_reduce](
        partial,
        dweight,
        num_bt_tiles,
        C,
        K,
        dws0,
        dws1,
        BLOCK_C=block_c,
        BLOCK_P=block_p,
        num_warps=reduce_warps,
    )

    return dx, dweight
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'causal_conv1d', 'arguments': ['x', 'weight', 'silu'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight'], 'gradient_return_order': ['dx', 'dweight', 'None'], 'scalars': ['silu'], 'deployment_entry': 'causal_conv1d_deployment', 'allowed_primitives': ['vendor_gemm']}


class CausalConv1DFunction(torch.autograd.Function):
    """Static autograd wiring for `causal_conv1d`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, silu):
        y, saved = causal_conv1d_forward_with_saved(x, weight, silu)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.silu = silu
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
        dx, dweight = causal_conv1d_backward_from_saved(
            dy, saved, ctx.silu
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, None


def causal_conv1d_deployment(x, weight, silu):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = CausalConv1DFunction.apply(x, weight, silu)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class CausalConv1DModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, silu=1):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.silu = silu
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return causal_conv1d_deployment(x, self.weight, self.silu)


DEPLOYMENT_ENTRY = "causal_conv1d_deployment"
