"""Handwritten Triton primitives for dense NCHW convolution.

The initial contract intentionally covers the shape-specialized subset emitted
by ``ops.conv2d``: groups=1, dilation=1, stride=1, padding=0, non-transposed.
Forward, dX, and dWeight use implicit-GEMM layouts without materializing
im2col; dBias is a tiled reduction.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _conv2d_fwd_kernel(
        x,
        weight,
        bias,
        y,
        B,
        C,
        H,
        W,
        O,
        KH,
        KW,
        OH,
        OW,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        M = B * OH * OW
        K = C * KH * KW
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(O, BLOCK_N)
        width = GROUP_M * grid_n
        group = pid // width
        group_size = min(grid_m - group * GROUP_M, GROUP_M)
        pid_m = group * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_o = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        batch = offs_m // (OH * OW)
        spatial = offs_m % (OH * OW)
        oh = spatial // OW
        ow = spatial % OW
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for start in tl.range(0, K, BLOCK_K):
            offs_k = start + tl.arange(0, BLOCK_K)
            channel = offs_k // (KH * KW)
            kernel_pos = offs_k % (KH * KW)
            kh = kernel_pos // KW
            kw = kernel_pos % KW
            x_offsets = (
                batch[:, None] * C * H * W
                + channel[None, :] * H * W
                + (oh[:, None] + kh[None, :]) * W
                + ow[:, None]
                + kw[None, :]
            )
            w_offsets = offs_o[None, :] * K + offs_k[:, None]
            x_tile = tl.load(
                x + x_offsets,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            w_tile = tl.load(
                weight + w_offsets,
                mask=(offs_k[:, None] < K) & (offs_o[None, :] < O),
                other=0.0,
            )
            accumulator += tl.dot(x_tile, w_tile, input_precision="ieee")

        accumulator += tl.load(
            bias + offs_o[None, :],
            mask=offs_o[None, :] < O,
            other=0.0,
        )
        out_offsets = (
            batch[:, None] * O * OH * OW
            + offs_o[None, :] * OH * OW
            + oh[:, None] * OW
            + ow[:, None]
        )
        tl.store(
            y + out_offsets,
            accumulator,
            mask=(offs_m[:, None] < M) & (offs_o[None, :] < O),
        )

    @triton.jit
    def _conv2d_dx_kernel(
        dy,
        weight,
        dx,
        B,
        C,
        H,
        W,
        O,
        KH,
        KW,
        OH,
        OW,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        M = B * H * W
        R = O * KH * KW
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(C, BLOCK_N)
        width = GROUP_M * grid_n
        group = pid // width
        group_size = min(grid_m - group * GROUP_M, GROUP_M)
        pid_m = group * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        batch = offs_m // (H * W)
        spatial = offs_m % (H * W)
        ih = spatial // W
        iw = spatial % W
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for start in tl.range(0, R, BLOCK_K):
            offs_r = start + tl.arange(0, BLOCK_K)
            out_channel = offs_r // (KH * KW)
            kernel_pos = offs_r % (KH * KW)
            kh = kernel_pos // KW
            kw = kernel_pos % KW
            oh = ih[:, None] - kh[None, :]
            ow = iw[:, None] - kw[None, :]
            valid_spatial = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW)
            dy_offsets = (
                batch[:, None] * O * OH * OW
                + out_channel[None, :] * OH * OW
                + oh * OW
                + ow
            )
            w_offsets = (
                out_channel[:, None] * C * KH * KW
                + offs_c[None, :] * KH * KW
                + kh[:, None] * KW
                + kw[:, None]
            )
            dy_tile = tl.load(
                dy + dy_offsets,
                mask=(offs_m[:, None] < M)
                & (offs_r[None, :] < R)
                & valid_spatial,
                other=0.0,
            )
            w_tile = tl.load(
                weight + w_offsets,
                mask=(offs_r[:, None] < R) & (offs_c[None, :] < C),
                other=0.0,
            )
            accumulator += tl.dot(dy_tile, w_tile, input_precision="ieee")

        dx_offsets = (
            batch[:, None] * C * H * W
            + offs_c[None, :] * H * W
            + ih[:, None] * W
            + iw[:, None]
        )
        tl.store(
            dx + dx_offsets,
            accumulator,
            mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
        )

    @triton.jit
    def _conv2d_dw_kernel(
        dy,
        x,
        dw,
        B,
        C,
        H,
        W,
        O,
        KH,
        KW,
        OH,
        OW,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        N = C * KH * KW
        R = B * OH * OW
        grid_m = tl.cdiv(O, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group = pid // width
        group_size = min(grid_m - group * GROUP_M, GROUP_M)
        pid_m = group * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_o = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = offs_n // (KH * KW)
        kernel_pos = offs_n % (KH * KW)
        kh = kernel_pos // KW
        kw = kernel_pos % KW
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for start in tl.range(0, R, BLOCK_K):
            offs_r = start + tl.arange(0, BLOCK_K)
            batch = offs_r // (OH * OW)
            spatial = offs_r % (OH * OW)
            oh = spatial // OW
            ow = spatial % OW
            dy_offsets = (
                batch[None, :] * O * OH * OW
                + offs_o[:, None] * OH * OW
                + oh[None, :] * OW
                + ow[None, :]
            )
            x_offsets = (
                batch[:, None] * C * H * W
                + channel[None, :] * H * W
                + (oh[:, None] + kh[None, :]) * W
                + ow[:, None]
                + kw[None, :]
            )
            dy_tile = tl.load(
                dy + dy_offsets,
                mask=(offs_o[:, None] < O) & (offs_r[None, :] < R),
                other=0.0,
            )
            x_tile = tl.load(
                x + x_offsets,
                mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(dy_tile, x_tile, input_precision="ieee")

        out_offsets = offs_o[:, None] * N + offs_n[None, :]
        tl.store(
            dw + out_offsets,
            accumulator,
            mask=(offs_o[:, None] < O) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _conv2d_db_kernel(
        dy,
        db,
        R,
        O,
        SPATIAL,
        BLOCK_R: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        offs_o = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
        accumulator = tl.zeros((BLOCK_O,), tl.float32)
        for start in tl.range(0, R, BLOCK_R):
            offs_r = start + tl.arange(0, BLOCK_R)
            batch = offs_r // SPATIAL
            spatial = offs_r % SPATIAL
            values = tl.load(
                dy
                + batch[None, :] * O * SPATIAL
                + offs_o[:, None] * SPATIAL
                + spatial[None, :],
                mask=(offs_o[:, None] < O) & (offs_r[None, :] < R),
                other=0.0,
            )
            accumulator += tl.sum(values.to(tl.float32), axis=1)
        tl.store(db + offs_o, accumulator, mask=offs_o < O)


def _validate(x, weight):
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d expects NCHW input and OIHW weight")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("input and weight channel dimensions must match")


def conv2d_forward(x, weight, bias):
    _validate(x, weight)
    x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
    b, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    oh, ow = h - kh + 1, w - kw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel must fit inside the input")
    output = torch.empty((b, o, oh, ow), device=x.device, dtype=x.dtype)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(b * oh * ow, block_m) * triton.cdiv(o, block_n),)
    _conv2d_fwd_kernel[grid](
        x,
        weight,
        bias,
        output,
        b,
        c,
        h,
        w,
        o,
        kh,
        kw,
        oh,
        ow,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    return output


def conv2d_backward(dy, x, weight, bias_sizes):
    _validate(x, weight)
    dy, x, weight = dy.contiguous(), x.contiguous(), weight.contiguous()
    b, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    oh, ow = h - kh + 1, w - kw + 1
    if tuple(dy.shape) != (b, o, oh, ow):
        raise ValueError("grad output shape does not match convolution output")
    if tuple(int(size) for size in bias_sizes) != (o,):
        raise ValueError("bias_sizes must contain the output-channel count")

    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty((o,), device=dy.device, dtype=dy.dtype)
    block_m, block_n, block_k = 64, 64, 32
    dx_grid = (triton.cdiv(b * h * w, block_m) * triton.cdiv(c, block_n),)
    _conv2d_dx_kernel[dx_grid](
        dy,
        weight,
        dx,
        b,
        c,
        h,
        w,
        o,
        kh,
        kw,
        oh,
        ow,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    dw_grid = (
        triton.cdiv(o, block_m)
        * triton.cdiv(c * kh * kw, block_n),
    )
    _conv2d_dw_kernel[dw_grid](
        dy,
        x,
        dw,
        b,
        c,
        h,
        w,
        o,
        kh,
        kw,
        oh,
        ow,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=4,
        num_stages=3,
    )
    _conv2d_db_kernel[(triton.cdiv(o, 64),)](
        dy,
        db,
        b * oh * ow,
        o,
        oh * ow,
        BLOCK_R=128,
        BLOCK_O=64,
        num_warps=4,
    )
    return dx, dw, db
