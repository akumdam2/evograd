"""adbench-tools artifact: task convnextv2_tiny.convnextv2_tiny_convnext_mlp.convnextv2.encoder.stages.1.layers, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for the ConvNeXtV2 MLP pair:
#   up linear -> GELU -> GRN -> down linear
# and its backward pass.
#
# Fusion notes:
# - The first projection and GELU are fused in one Triton matmul kernel.
# - GRN normalization is split into reductions because it depends on per-batch
#   spatial/channel norms.
# - GRN affine/residual and the down projection are fused: the intermediate GRN
#   output is not materialized in forward.
# - Backward similarly fuses GRN backward with GELU backward before the up-linear
#   gradients.
# - kernel_function returns a single flattened fp32 tensor containing:
#     forward output, then gradients for
#     x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias.
#
# No torch compute operators are used; torch is only used for allocation,
# shape/dtype/device handling, and launching Triton kernels.

import torch
import triton
import triton.language as tl


@triton.jit
def _up_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, z_ptr, a_ptr,
    M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)

        x = tl.load(
            x_ptr + offs_m[:, None] * C + offs_c[None, :],
            mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            w_ptr + offs_k[None, :] * C + offs_c[:, None],
            mask=(offs_k[None, :] < K) & (offs_c[:, None] < C),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(x, w, acc, input_precision="tf32")

    b = tl.load(bias_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
    z = acc + b[None, :]

    # Exact GELU: 0.5*x*(1 + erf(x/sqrt(2))).
    a = 0.5 * z * (1.0 + tl.erf(z * 0.7071067811865476))

    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(z_ptr + offs_m[:, None] * K + offs_k[None, :], z, mask=mask)
    tl.store(a_ptr + offs_m[:, None] * K + offs_k[None, :], a, mask=mask)


@triton.jit
def _grn_gx_kernel(
    a_ptr, gx_ptr,
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    mask = offs_s < S

    vals = tl.load(
        a_ptr + (pid_b * S + offs_s) * K + pid_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    ss = tl.sum(vals * vals, axis=0)
    gx = tl.sqrt(ss)
    tl.store(gx_ptr + pid_b * K + pid_k, gx)


@triton.jit
def _grn_mean_kernel(
    gx_ptr, mean_ptr,
    B: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    gx = tl.load(gx_ptr + pid_b * K + offs_k, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(gx, axis=0)
    tl.store(mean_ptr + pid_b, s / K)


@triton.jit
def _down_forward_kernel(
    a_ptr, gx_ptr, mean_ptr, gamma_ptr, beta_ptr,
    down_w_ptr, down_b_ptr, y_ptr,
    M: tl.constexpr, S: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    batch = offs_m // S

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        gx = tl.load(
            gx_ptr + batch[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(mean_ptr + batch, mask=offs_m < M, other=0.0).to(tl.float32)
        nx = gx / (mean[:, None] + 1.0e-6)

        gamma = tl.load(gamma_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)

        grn = a * (1.0 + gamma[None, :] * nx) + beta[None, :]

        w = tl.load(
            down_w_ptr + offs_c[None, :] * K + offs_k[:, None],
            mask=(offs_c[None, :] < C) & (offs_k[:, None] < K),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(grn, w, acc, input_precision="tf32")

    db = tl.load(down_b_ptr + offs_c, mask=offs_c < C, other=0.0).to(tl.float32)
    out = acc + db[None, :]

    tl.store(
        y_ptr + offs_m[:, None] * C + offs_c[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
    )


@triton.jit
def _down_db_kernel(
    dy_ptr, down_w_ptr, db_grn_ptr,
    M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)

        dy = tl.load(
            dy_ptr + offs_m[:, None] * C + offs_c[None, :],
            mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            down_w_ptr + offs_c[:, None] * K + offs_k[None, :],
            mask=(offs_c[:, None] < C) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(dy, w, acc, input_precision="tf32")

    tl.store(
        db_grn_ptr + offs_m[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _down_weight_grad_kernel(
    dy_ptr, a_ptr, gx_ptr, mean_ptr, gamma_ptr, beta_ptr, gdw_ptr,
    M: tl.constexpr, S: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_C, BLOCK_K), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        batch = offs_m // S

        dy = tl.load(
            dy_ptr + offs_m[None, :] * C + offs_c[:, None],
            mask=(offs_m[None, :] < M) & (offs_c[:, None] < C),
            other=0.0,
        ).to(tl.float32)

        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        gx = tl.load(
            gx_ptr + batch[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(mean_ptr + batch, mask=offs_m < M, other=0.0).to(tl.float32)
        nx = gx / (mean[:, None] + 1.0e-6)

        gamma = tl.load(gamma_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)

        grn = a * (1.0 + gamma[None, :] * nx) + beta[None, :]
        acc = tl.dot(dy, grn, acc, input_precision="tf32")

    tl.store(
        gdw_ptr + offs_c[:, None] * K + offs_k[None, :],
        acc,
        mask=(offs_c[:, None] < C) & (offs_k[None, :] < K),
    )


@triton.jit
def _bias_reduce_kernel(
    inp_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        vals = tl.load(
            inp_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + offs_n, acc, mask=offs_n < N)


@triton.jit
def _grn_param_and_q_kernel(
    db_ptr, a_ptr, gx_ptr, mean_ptr, gamma_ptr,
    q_ptr, ggamma_ptr, gbeta_ptr,
    B: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    mask = offs_s < S
    base = (pid_b * S + offs_s) * K + pid_k

    db = tl.load(db_ptr + base, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + base, mask=mask, other=0.0).to(tl.float32)

    gamma = tl.load(gamma_ptr + pid_k).to(tl.float32)
    gx = tl.load(gx_ptr + pid_b * K + pid_k).to(tl.float32)
    mean = tl.load(mean_ptr + pid_b).to(tl.float32)
    nx = gx / (mean + 1.0e-6)

    q = tl.sum(db * gamma * a, axis=0)
    gg = tl.sum(db * a * nx, axis=0)
    gb = tl.sum(db, axis=0)

    tl.store(q_ptr + pid_b * K + pid_k, q)
    tl.store(ggamma_ptr + pid_b * K + pid_k, gg)
    tl.store(gbeta_ptr + pid_b * K + pid_k, gb)


@triton.jit
def _sum_over_batch_kernel(
    tmp_ptr, out_ptr,
    B: tl.constexpr, K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_B)
    mask = offs_b < B
    vals = tl.load(tmp_ptr + offs_b * K + pid_k, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid_k, s)


@triton.jit
def _qg_sum_kernel(
    q_ptr, gx_ptr, qg_ptr,
    B: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K

    q = tl.load(q_ptr + pid_b * K + offs_k, mask=mask, other=0.0).to(tl.float32)
    gx = tl.load(gx_ptr + pid_b * K + offs_k, mask=mask, other=0.0).to(tl.float32)

    s = tl.sum(q * gx, axis=0)
    tl.store(qg_ptr + pid_b, s)


@triton.jit
def _grn_gelu_backward_kernel(
    db_ptr, a_ptr, z_ptr, gx_ptr, mean_ptr, gamma_ptr, q_ptr, qg_ptr, dz_ptr,
    M: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    batch = offs_m // S

    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    db = tl.load(db_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)

    gx = tl.load(gx_ptr + batch[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + batch, mask=offs_m < M, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)

    q = tl.load(q_ptr + batch[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)
    qg = tl.load(qg_ptr + batch, mask=offs_m < M, other=0.0).to(tl.float32)

    d = mean[:, None] + 1.0e-6
    nx = gx / d
    safe_gx = tl.maximum(gx, 1.0e-12)

    r = q / d - qg[:, None] / (K * d * d)
    da = db * (1.0 + gamma[None, :] * nx) + r * a / safe_gx

    # d/dx exact GELU = Phi(x) + x * phi(x)
    gelu_grad = 0.5 * (1.0 + tl.erf(z * 0.7071067811865476)) + z * 0.3989422804014327 * tl.exp(-0.5 * z * z)
    dz = da * gelu_grad

    tl.store(dz_ptr + offs_m[:, None] * K + offs_k[None, :], dz, mask=mask)


@triton.jit
def _x_grad_kernel(
    dz_ptr, up_w_ptr, gx_ptr,
    M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        dz = tl.load(
            dz_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            up_w_ptr + offs_k[:, None] * C + offs_c[None, :],
            mask=(offs_k[:, None] < K) & (offs_c[None, :] < C),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(dz, w, acc, input_precision="tf32")

    tl.store(
        gx_ptr + offs_m[:, None] * C + offs_c[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
    )


@triton.jit
def _up_weight_grad_kernel(
    dz_ptr, x_ptr, guw_ptr,
    M: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_K, BLOCK_C), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)

        dz = tl.load(
            dz_ptr + offs_m[None, :] * K + offs_k[:, None],
            mask=(offs_m[None, :] < M) & (offs_k[:, None] < K),
            other=0.0,
        ).to(tl.float32)

        x = tl.load(
            x_ptr + offs_m[:, None] * C + offs_c[None, :],
            mask=(offs_m[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        ).to(tl.float32)

        acc = tl.dot(dz, x, acc, input_precision="tf32")

    tl.store(
        guw_ptr + offs_k[:, None] * C + offs_c[None, :],
        acc,
        mask=(offs_k[:, None] < K) & (offs_c[None, :] < C),
    )


@triton.jit
def _concat_kernel(
    y, gx, guw, gub, ggw, ggb, gdw, gdb, out,
    n_y: tl.constexpr, n_gx: tl.constexpr, n_guw: tl.constexpr, n_gub: tl.constexpr,
    n_ggw: tl.constexpr, n_ggb: tl.constexpr, n_gdw: tl.constexpr, n_gdb: tl.constexpr,
    total: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    o0 = n_y
    o1 = o0 + n_gx
    o2 = o1 + n_guw
    o3 = o2 + n_gub
    o4 = o3 + n_ggw
    o5 = o4 + n_ggb
    o6 = o5 + n_gdw

    vals = tl.zeros((BLOCK,), dtype=tl.float32)

    vals = tl.where(offs < o0, tl.load(y + offs, mask=mask & (offs < o0), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o0) & (offs < o1), tl.load(gx + (offs - o0), mask=mask & (offs >= o0) & (offs < o1), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o1) & (offs < o2), tl.load(guw + (offs - o1), mask=mask & (offs >= o1) & (offs < o2), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o2) & (offs < o3), tl.load(gub + (offs - o2), mask=mask & (offs >= o2) & (offs < o3), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o3) & (offs < o4), tl.load(ggw + (offs - o3), mask=mask & (offs >= o3) & (offs < o4), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o4) & (offs < o5), tl.load(ggb + (offs - o4), mask=mask & (offs >= o4) & (offs < o5), other=0.0).to(tl.float32), vals)
    vals = tl.where((offs >= o5) & (offs < o6), tl.load(gdw + (offs - o5), mask=mask & (offs >= o5) & (offs < o6), other=0.0).to(tl.float32), vals)
    vals = tl.where(offs >= o6, tl.load(gdb + (offs - o6), mask=mask & (offs >= o6), other=0.0).to(tl.float32), vals)

    tl.store(out + offs, vals, mask=mask)


def _dims(x, up_weight, down_weight):
    C = x.shape[-1]
    M = x.numel() // C
    B = x.shape[0] if x.dim() >= 3 else 1
    S = M // B
    K = up_weight.shape[0]
    Cout = down_weight.shape[0]
    assert Cout == C
    assert up_weight.shape[1] == C
    assert down_weight.shape[1] == K
    return M, B, S, C, K


def convnextv2_tiny_convnext_mlp_forward_with_saved(
    x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias
):
    """Forward pass for ConvNeXtV2 MLP, implemented entirely with Triton kernels."""
    M, B, S, C, K = _dims(x, up_weight, down_weight)

    z = torch.empty((M, K), device=x.device, dtype=torch.float32)
    a = torch.empty((M, K), device=x.device, dtype=torch.float32)
    gx = torch.empty((B, K), device=x.device, dtype=torch.float32)
    mean = torch.empty((B,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)

    BM = 16
    BN = 32
    BCRED = 64
    _up_gelu_kernel[(triton.cdiv(M, BM), triton.cdiv(K, BN))](
        x, up_weight, up_bias, z, a,
        M, C, K,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_C=BCRED,
    )

    BS = triton.next_power_of_2(S)
    _grn_gx_kernel[(B, K)](
        a, gx,
        B, S, K,
        BLOCK_S=BS,
    )

    BK_ALL = triton.next_power_of_2(K)
    _grn_mean_kernel[(B,)](
        gx, mean,
        B, K,
        BLOCK_K=BK_ALL,
    )

    _down_forward_kernel[(triton.cdiv(M, BM), triton.cdiv(C, BN))](
        a, gx, mean, grn_weight, grn_bias,
        down_weight, down_bias, y,
        M, S, C, K,
        BLOCK_M=BM, BLOCK_C=BN, BLOCK_K=64,
    )

    saved = (x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias, z, a, gx, mean)
    return y, saved


def convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved):
    """Backward pass for the ConvNeXtV2 MLP, implemented with Triton kernels."""
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias, z, a, gx_norm, mean = saved
    M, B, S, C, K = _dims(x, up_weight, down_weight)

    db_grn = torch.empty((M, K), device=x.device, dtype=torch.float32)
    dz = torch.empty((M, K), device=x.device, dtype=torch.float32)

    grad_x = torch.empty_like(x)
    grad_up_weight = torch.empty_like(up_weight)
    grad_up_bias = torch.empty_like(up_bias)
    grad_grn_weight = torch.empty_like(grn_weight)
    grad_grn_bias = torch.empty_like(grn_bias)
    grad_down_weight = torch.empty_like(down_weight)
    grad_down_bias = torch.empty_like(down_bias)

    q = torch.empty((B, K), device=x.device, dtype=torch.float32)
    tmp_ggamma = torch.empty((B, K), device=x.device, dtype=torch.float32)
    tmp_gbeta = torch.empty((B, K), device=x.device, dtype=torch.float32)
    qg = torch.empty((B,), device=x.device, dtype=torch.float32)

    BM = 16
    BN = 32

    _down_db_kernel[(triton.cdiv(M, BM), triton.cdiv(K, BN))](
        dy, down_weight, db_grn,
        M, C, K,
        BLOCK_M=BM, BLOCK_K=BN, BLOCK_C=64,
    )

    _down_weight_grad_kernel[(triton.cdiv(C, BN), triton.cdiv(K, BN))](
        dy, a, gx_norm, mean, grn_weight, grn_bias, grad_down_weight,
        M, S, C, K,
        BLOCK_C=BN, BLOCK_K=BN, BLOCK_M=32,
    )

    _bias_reduce_kernel[(triton.cdiv(C, BN),)](
        dy, grad_down_bias,
        M, C,
        BLOCK_N=BN, BLOCK_M=64,
    )

    BS = triton.next_power_of_2(S)
    _grn_param_and_q_kernel[(B, K)](
        db_grn, a, gx_norm, mean, grn_weight,
        q, tmp_ggamma, tmp_gbeta,
        B, S, K,
        BLOCK_S=BS,
    )

    BB = triton.next_power_of_2(B)
    _sum_over_batch_kernel[(K,)](
        tmp_ggamma, grad_grn_weight,
        B, K,
        BLOCK_B=BB,
    )
    _sum_over_batch_kernel[(K,)](
        tmp_gbeta, grad_grn_bias,
        B, K,
        BLOCK_B=BB,
    )

    BK_ALL = triton.next_power_of_2(K)
    _qg_sum_kernel[(B,)](
        q, gx_norm, qg,
        B, K,
        BLOCK_K=BK_ALL,
    )

    _grn_gelu_backward_kernel[(triton.cdiv(M, BM), triton.cdiv(K, BN))](
        db_grn, a, z, gx_norm, mean, grn_weight, q, qg, dz,
        M, S, K,
        BLOCK_M=BM, BLOCK_K=BN,
    )

    _x_grad_kernel[(triton.cdiv(M, BM), triton.cdiv(C, BN))](
        dz, up_weight, grad_x,
        M, C, K,
        BLOCK_M=BM, BLOCK_C=BN, BLOCK_K=64,
    )

    _up_weight_grad_kernel[(triton.cdiv(K, BN), triton.cdiv(C, BN))](
        dz, x, grad_up_weight,
        M, C, K,
        BLOCK_K=BN, BLOCK_C=BN, BLOCK_M=32,
    )

    _bias_reduce_kernel[(triton.cdiv(K, BN),)](
        dz, grad_up_bias,
        M, K,
        BLOCK_N=BN, BLOCK_M=64,
    )

    return (
        grad_x,
        grad_up_weight,
        grad_up_bias,
        grad_grn_weight,
        grad_grn_bias,
        grad_down_weight,
        grad_down_bias,
    )


def kernel_function(
    x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias, upstream
):
    """
    Execute the complete forward+backward step for the ConvNeXtV2 MLP.

    Returns a single fp32 vector equivalent to:
        torch.cat([t.float().reshape(-1) for t in
          (output, grad_x, grad_up_weight, grad_up_bias,
           grad_grn_weight, grad_grn_bias, grad_down_weight, grad_down_bias)])
    but the concatenation/casting is done by a Triton copy kernel rather than
    torch.cat or other PyTorch compute.
    """
    y, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(
        x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias
    )
    grads = convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved)
    grad_x, grad_up_weight, grad_up_bias, grad_grn_weight, grad_grn_bias, grad_down_weight, grad_down_bias = grads

    n_y = y.numel()
    n_gx = grad_x.numel()
    n_guw = grad_up_weight.numel()
    n_gub = grad_up_bias.numel()
    n_ggw = grad_grn_weight.numel()
    n_ggb = grad_grn_bias.numel()
    n_gdw = grad_down_weight.numel()
    n_gdb = grad_down_bias.numel()
    total = n_y + n_gx + n_guw + n_gub + n_ggw + n_ggb + n_gdw + n_gdb

    out = torch.empty((total,), device=x.device, dtype=torch.float32)
    BLOCK = 1024
    _concat_kernel[(triton.cdiv(total, BLOCK),)](
        y, grad_x, grad_up_weight, grad_up_bias,
        grad_grn_weight, grad_grn_bias, grad_down_weight, grad_down_bias,
        out,
        n_y, n_gx, n_guw, n_gub, n_ggw, n_ggb, n_gdw, n_gdb,
        total,
        BLOCK=BLOCK,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'convnextv2_tiny_convnext_mlp', 'arguments': ['x', 'up_weight', 'up_bias', 'grn_weight', 'grn_bias', 'down_weight', 'down_bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'gradient_return_order': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'scalars': [], 'deployment_entry': 'convnextv2_tiny_convnext_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Convnextv2TinyConvnextMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `convnextv2_tiny_convnext_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        out, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
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
        dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias = convnextv2_tiny_convnext_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias


def convnextv2_tiny_convnext_mlp_deployment(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        grn_weight = grn_weight.reshape(-1, grn_weight.shape[-1])
        grn_bias = grn_bias.reshape(-1, grn_bias.shape[-1])
    out = Convnextv2TinyConvnextMlpFunction.apply(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Convnextv2TinyConvnextMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        super().__init__()
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.up_bias = (up_bias if isinstance(up_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(up_bias))
        self.grn_weight = (grn_weight if isinstance(grn_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_weight))
        self.grn_bias = (grn_bias if isinstance(grn_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_bias))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))
        self.down_bias = (down_bias if isinstance(down_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(down_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return convnextv2_tiny_convnext_mlp_deployment(x, self.up_weight, self.up_bias, self.grn_weight, self.grn_bias, self.down_weight, self.down_bias)


DEPLOYMENT_ENTRY = "convnextv2_tiny_convnext_mlp_deployment"
