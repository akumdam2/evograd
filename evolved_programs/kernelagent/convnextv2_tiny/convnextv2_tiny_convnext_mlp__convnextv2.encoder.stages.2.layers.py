"""adbench-tools artifact: task convnextv2_tiny.convnextv2_tiny_convnext_mlp.convnextv2.encoder.stages.2.layers, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
import torch
import triton
import triton.language as tl


# NOTE:
# The previous implementation computed correct values for dup_bias/ddown_bias but
# allocated them as float32. The reference gradients for bf16 bias tensors are
# bf16, so the evaluator rejected them on dtype. Bias-gradient buffers are now
# allocated with torch.empty_like(their corresponding bias tensors). Accumulation
# still happens in fp32 inside Triton and is cast on store to the output dtype.


@triton.jit
def _erf_approx(x):
    sign = tl.where(x < 0.0, -1.0, 1.0)
    ax = tl.abs(x)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
    y = 1.0 - poly * tl.exp(-(ax * ax))
    return sign * y


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + _erf_approx(x * 0.7071067811865476))


@triton.jit
def _gelu_grad(x):
    erfv = _erf_approx(x * 0.7071067811865476)
    pdf_term = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    return 0.5 * (1.0 + erfv) + pdf_term


@triton.jit
def _up_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    u_ptr,
    a_ptr,
    P: tl.constexpr,
    C: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for c0 in tl.range(0, C, BLOCK_C):
        cidx = c0 + offs_c
        x_vals = tl.load(
            x_ptr + offs_m[:, None] * C + cidx[None, :],
            mask=(offs_m[:, None] < P) & (cidx[None, :] < C),
            other=0.0,
        )
        w_vals = tl.load(
            w_ptr + offs_n[None, :] * C + cidx[:, None],
            mask=(offs_n[None, :] < K) & (cidx[:, None] < C),
            other=0.0,
        )
        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < K, other=0.0).to(tl.float32)
    u = acc + bias[None, :]
    a = _gelu(u)

    mask = (offs_m[:, None] < P) & (offs_n[None, :] < K)
    tl.store(u_ptr + offs_m[:, None] * K + offs_n[None, :], u, mask=mask)
    tl.store(a_ptr + offs_m[:, None] * K + offs_n[None, :], a, mask=mask)


@triton.jit
def _norm_kernel(
    a_ptr,
    g_ptr,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    vals = tl.load(
        a_ptr + (b * S + offs_s[:, None]) * K + offs_k[None, :],
        mask=(offs_s[:, None] < S) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    ss = tl.sum(vals * vals, axis=0)
    g = tl.sqrt(ss)
    tl.store(g_ptr + b * K + offs_k, g, mask=offs_k < K)


@triton.jit
def _mean_kernel(
    g_ptr,
    mean_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    vals = tl.load(g_ptr + b * K + offs, mask=offs < K, other=0.0).to(tl.float32)
    s = tl.sum(vals, axis=0)
    tl.store(mean_ptr + b, s / K)


@triton.jit
def _grn_forward_kernel(
    a_ptr,
    g_ptr,
    mean_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    P: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P * K

    k = offs % K
    p = offs // K
    b = p // S

    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * K + k, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mean_ptr + b, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + k, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + k, mask=mask, other=0.0).to(tl.float32)

    q = g / (m + EPS)
    y = a + beta + gamma * a * q
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _down_kernel(
    y_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    P: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        kk = k0 + offs_k
        y = tl.load(
            y_ptr + offs_m[:, None] * K + kk[None, :],
            mask=(offs_m[:, None] < P) & (kk[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + kk[:, None],
            mask=(offs_n[None, :] < C) & (kk[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(y, w)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < C, other=0.0).to(tl.float32)
    out = acc + bias[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * C + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < P) & (offs_n[None, :] < C),
    )


@triton.jit
def _linear_backward_input_kernel(
    grad_out_ptr,
    w_ptr,
    grad_in_ptr,
    P: tl.constexpr,
    OUT_D: tl.constexpr,
    IN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grad_in[P, IN_D] = grad_out[P, OUT_D] @ w[OUT_D, IN_D]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, OUT_D, BLOCK_K):
        kk = k0 + offs_k
        go = tl.load(
            grad_out_ptr + offs_m[:, None] * OUT_D + kk[None, :],
            mask=(offs_m[:, None] < P) & (kk[None, :] < OUT_D),
            other=0.0,
        )
        w = tl.load(
            w_ptr + kk[:, None] * IN_D + offs_n[None, :],
            mask=(kk[:, None] < OUT_D) & (offs_n[None, :] < IN_D),
            other=0.0,
        )
        acc += tl.dot(go, w)

    tl.store(
        grad_in_ptr + offs_m[:, None] * IN_D + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < P) & (offs_n[None, :] < IN_D),
    )


@triton.jit
def _weight_grad_kernel(
    lhs_ptr,
    rhs_ptr,
    grad_w_ptr,
    P: tl.constexpr,
    ROW_D: tl.constexpr,
    COL_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # grad_w[ROW_D, COL_D] = lhs[P, ROW_D]^T @ rhs[P, COL_D]
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_p = tl.arange(0, BLOCK_P)

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for p0 in tl.range(0, P, BLOCK_P):
        pp = p0 + offs_p
        lhs = tl.load(
            lhs_ptr + pp[None, :] * ROW_D + offs_r[:, None],
            mask=(pp[None, :] < P) & (offs_r[:, None] < ROW_D),
            other=0.0,
        )
        rhs = tl.load(
            rhs_ptr + pp[:, None] * COL_D + offs_c[None, :],
            mask=(pp[:, None] < P) & (offs_c[None, :] < COL_D),
            other=0.0,
        )
        acc += tl.dot(lhs, rhs)

    tl.store(
        grad_w_ptr + offs_r[:, None] * COL_D + offs_c[None, :],
        acc,
        mask=(offs_r[:, None] < ROW_D) & (offs_c[None, :] < COL_D),
    )


@triton.jit
def _bias_grad_kernel(
    grad_out_ptr,
    grad_b_ptr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_p = tl.arange(0, BLOCK_P)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for p0 in tl.range(0, P, BLOCK_P):
        pp = p0 + offs_p
        vals = tl.load(
            grad_out_ptr + pp[:, None] * D + offs_d[None, :],
            mask=(pp[:, None] < P) & (offs_d[None, :] < D),
            other=0.0,
        )
        acc += tl.sum(vals, axis=0)

    # Store casts to grad_b_ptr dtype, matching autograd's bias grad dtype.
    tl.store(grad_b_ptr + offs_d, acc, mask=offs_d < D)


@triton.jit
def _grn_partials_kernel(
    a_ptr,
    dy_ptr,
    g_ptr,
    mean_ptr,
    gamma_ptr,
    r_ptr,
    dgamma_part_ptr,
    dbeta_part_ptr,
    S: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    a = tl.load(
        a_ptr + (b * S + offs_s[:, None]) * K + offs_k[None, :],
        mask=(offs_s[:, None] < S) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    dy = tl.load(
        dy_ptr + (b * S + offs_s[:, None]) * K + offs_k[None, :],
        mask=(offs_s[:, None] < S) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    sum_dy = tl.sum(dy, axis=0)
    sum_dya = tl.sum(dy * a, axis=0)

    g = tl.load(g_ptr + b * K + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
    m = tl.load(mean_ptr + b).to(tl.float32)
    gamma = tl.load(gamma_ptr + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)

    q = g / (m + EPS)
    r = gamma * sum_dya

    tl.store(r_ptr + b * K + offs_k, r, mask=offs_k < K)
    tl.store(dgamma_part_ptr + b * K + offs_k, sum_dya * q, mask=offs_k < K)
    tl.store(dbeta_part_ptr + b * K + offs_k, sum_dy, mask=offs_k < K)


@triton.jit
def _sum_rg_kernel(
    r_ptr,
    g_ptr,
    sum_rg_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)

    r = tl.load(r_ptr + b * K + offs, mask=offs < K, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * K + offs, mask=offs < K, other=0.0).to(tl.float32)

    s = tl.sum(r * g, axis=0)
    tl.store(sum_rg_ptr + b, s)


@triton.jit
def _grn_reduce_batch_kernel(
    part_ptr,
    out_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)

    offs_b = tl.arange(0, BLOCK_B)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    vals = tl.load(
        part_ptr + offs_b[:, None] * K + offs_k[None, :],
        mask=(offs_b[:, None] < B) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    s = tl.sum(vals, axis=0)
    tl.store(out_ptr + offs_k, s, mask=offs_k < K)


@triton.jit
def _grn_backward_elem_kernel(
    a_ptr,
    dy_ptr,
    g_ptr,
    mean_ptr,
    gamma_ptr,
    r_ptr,
    sum_rg_ptr,
    da_ptr,
    P: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P * K

    k = offs % K
    p = offs // K
    b = p // S

    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + b * K + k, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(mean_ptr + b, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + k, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + b * K + k, mask=mask, other=0.0).to(tl.float32)
    sum_rg = tl.load(sum_rg_ptr + b, mask=mask, other=0.0).to(tl.float32)

    d = m + EPS
    q = g / d

    # q_k = g_k / d, d = mean(g) + eps.
    # r_k = gamma_k * sum_s(dy_s,k * a_s,k)
    # sum_rg = sum_j r_j * g_j
    dg = r / d - sum_rg / (d * d * K)

    a_over_g = tl.where(g > 0.0, a / g, 0.0)
    da = dy * (1.0 + gamma * q) + dg * a_over_g

    tl.store(da_ptr + offs, da, mask=mask)


@triton.jit
def _gelu_backward_kernel(
    da_ptr,
    u_ptr,
    du_ptr,
    P: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < P * K

    da = tl.load(da_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    du = da * _gelu_grad(u)
    tl.store(du_ptr + offs, du, mask=mask)


@triton.jit
def _copy_to_flat_kernel(
    src_ptr,
    dst_ptr,
    dst_offset: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst_ptr + dst_offset + offs, vals, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def convnextv2_tiny_convnext_mlp_forward_with_saved(
    x,
    up_weight,
    up_bias,
    grn_weight,
    grn_bias,
    down_weight,
    down_bias,
):
    """
    Forward:
      up  = x @ up_weight.T + up_bias
      act = GELU(up)
      GRN = act + grn_bias + grn_weight * act * Gx / (mean_c(Gx) + eps)
      out = GRN @ down_weight.T + down_bias

    Layout:
      x:           [B, Y, X, C]
      up_weight:   [K, C]
      up_bias:     [K]
      grn_weight:  [K]
      grn_bias:    [K]
      down_weight: [C, K]
      down_bias:   [C]
    """
    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    K = up_weight.shape[0]
    S = H * W
    P = B * S

    u = torch.empty((P, K), device=x.device, dtype=x.dtype)
    a = torch.empty((P, K), device=x.device, dtype=x.dtype)
    g = torch.empty((B, K), device=x.device, dtype=torch.float32)
    mean = torch.empty((B,), device=x.device, dtype=torch.float32)
    y = torch.empty((P, K), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)

    BM = 16
    BN = 32
    BK = 64

    _up_gelu_kernel[(triton.cdiv(P, BM), triton.cdiv(K, BN))](
        x,
        up_weight,
        up_bias,
        u,
        a,
        P,
        C,
        K,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_C=BK,
        num_warps=4,
        num_stages=3,
    )

    BS = _next_power_of_2(S)
    _norm_kernel[(B, triton.cdiv(K, 16))](
        a,
        g,
        S,
        K,
        BLOCK_S=BS,
        BLOCK_K=16,
        num_warps=4,
    )

    BK_MEAN = _next_power_of_2(K)
    _mean_kernel[(B,)](
        g,
        mean,
        K,
        BLOCK_K=BK_MEAN,
        num_warps=8,
    )

    total_hidden = P * K
    _grn_forward_kernel[(triton.cdiv(total_hidden, 256),)](
        a,
        g,
        mean,
        grn_weight,
        grn_bias,
        y,
        P,
        S,
        K,
        EPS=1.0e-6,
        BLOCK=256,
        num_warps=4,
    )

    _down_kernel[(triton.cdiv(P, BM), triton.cdiv(C, BN))](
        y,
        down_weight,
        down_bias,
        out,
        P,
        K,
        C,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    saved = (
        x,
        up_weight,
        grn_weight,
        down_weight,
        u,
        a,
        g,
        mean,
        y,
    )
    return out, saved


def convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved):
    """
    Backward pass. Matmul/reduction phases are kept as separate Triton kernels
    because GRN backward has both spatial and channel reductions that are reused
    by later elementwise stages; fusing all of them into one kernel would require
    cross-program synchronization, which Triton kernels do not provide.
    """
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    x, up_weight, grn_weight, down_weight, u, a, g, mean, y = saved

    B = x.shape[0]
    H = x.shape[1]
    W = x.shape[2]
    C = x.shape[3]
    K = up_weight.shape[0]
    S = H * W
    P = B * S

    dx = torch.empty_like(x)
    dup_weight = torch.empty_like(up_weight)

    # Critical dtype fix: bias grads must match corresponding bias dtype.
    dup_bias = torch.empty((K,), device=x.device, dtype=up_weight.dtype)
    ddown_bias = torch.empty((C,), device=x.device, dtype=down_weight.dtype)

    dgrn_weight = torch.empty_like(grn_weight)
    dgrn_bias = torch.empty_like(grn_weight)
    ddown_weight = torch.empty_like(down_weight)

    dy = torch.empty((P, K), device=x.device, dtype=x.dtype)
    da = torch.empty((P, K), device=x.device, dtype=x.dtype)
    du = torch.empty((P, K), device=x.device, dtype=x.dtype)

    r = torch.empty((B, K), device=x.device, dtype=torch.float32)
    dgamma_part = torch.empty((B, K), device=x.device, dtype=torch.float32)
    dbeta_part = torch.empty((B, K), device=x.device, dtype=torch.float32)
    sum_rg = torch.empty((B,), device=x.device, dtype=torch.float32)

    BM = 16
    BN = 32
    BK = 64

    # down_bias grad
    _bias_grad_kernel[(triton.cdiv(C, 32),)](
        dout,
        ddown_bias,
        P,
        C,
        BLOCK_D=32,
        BLOCK_P=1024,
        num_warps=4,
    )

    # down_weight grad: dout^T @ y
    _weight_grad_kernel[(triton.cdiv(C, 16), triton.cdiv(K, 32))](
        dout,
        y,
        ddown_weight,
        P,
        C,
        K,
        BLOCK_R=16,
        BLOCK_C=32,
        BLOCK_P=32,
        num_warps=4,
        num_stages=3,
    )

    # dy = dout @ down_weight
    _linear_backward_input_kernel[(triton.cdiv(P, BM), triton.cdiv(K, BN))](
        dout,
        down_weight,
        dy,
        P,
        C,
        K,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    # GRN backward partials
    BS = _next_power_of_2(S)
    _grn_partials_kernel[(B, triton.cdiv(K, 16))](
        a,
        dy,
        g,
        mean,
        grn_weight,
        r,
        dgamma_part,
        dbeta_part,
        S,
        K,
        EPS=1.0e-6,
        BLOCK_S=BS,
        BLOCK_K=16,
        num_warps=4,
    )

    BK_MEAN = _next_power_of_2(K)
    _sum_rg_kernel[(B,)](
        r,
        g,
        sum_rg,
        K,
        BLOCK_K=BK_MEAN,
        num_warps=8,
    )

    BB = _next_power_of_2(B)
    _grn_reduce_batch_kernel[(triton.cdiv(K, 32),)](
        dgamma_part,
        dgrn_weight,
        B,
        K,
        BLOCK_B=BB,
        BLOCK_K=32,
        num_warps=4,
    )

    _grn_reduce_batch_kernel[(triton.cdiv(K, 32),)](
        dbeta_part,
        dgrn_bias,
        B,
        K,
        BLOCK_B=BB,
        BLOCK_K=32,
        num_warps=4,
    )

    _grn_backward_elem_kernel[(triton.cdiv(P * K, 256),)](
        a,
        dy,
        g,
        mean,
        grn_weight,
        r,
        sum_rg,
        da,
        P,
        S,
        K,
        EPS=1.0e-6,
        BLOCK=256,
        num_warps=4,
    )

    # GELU backward
    _gelu_backward_kernel[(triton.cdiv(P * K, 256),)](
        da,
        u,
        du,
        P,
        K,
        BLOCK=256,
        num_warps=4,
    )

    # up_bias grad
    _bias_grad_kernel[(triton.cdiv(K, 32),)](
        du,
        dup_bias,
        P,
        K,
        BLOCK_D=32,
        BLOCK_P=1024,
        num_warps=4,
    )

    # up_weight grad: du^T @ x
    _weight_grad_kernel[(triton.cdiv(K, 16), triton.cdiv(C, 32))](
        du,
        x,
        dup_weight,
        P,
        K,
        C,
        BLOCK_R=16,
        BLOCK_C=32,
        BLOCK_P=32,
        num_warps=4,
        num_stages=3,
    )

    # dx = du @ up_weight
    _linear_backward_input_kernel[(triton.cdiv(P, BM), triton.cdiv(C, BN))](
        du,
        up_weight,
        dx,
        P,
        K,
        C,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=3,
    )

    return (
        dx,
        dup_weight,
        dup_bias,
        dgrn_weight,
        dgrn_bias,
        ddown_weight,
        ddown_bias,
    )


def _copy_flat(src, dst, offset):
    n = src.numel()
    if n > 0:
        _copy_to_flat_kernel[(triton.cdiv(n, 1024),)](
            src,
            dst,
            offset,
            n,
            BLOCK=1024,
            num_warps=4,
        )
    return offset + n


def kernel_function(
    x,
    up_weight,
    up_bias,
    grn_weight,
    grn_bias,
    down_weight,
    down_bias,
    upstream,
):
    """
    Evaluator entry point.

    Returns:
      flat fp32 concat of:
        out,
        dx,
        dup_weight,
        dup_bias,
        dgrn_weight,
        dgrn_bias,
        ddown_weight,
        ddown_bias
    """
    out, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(
        x,
        up_weight,
        up_bias,
        grn_weight,
        grn_bias,
        down_weight,
        down_bias,
    )

    grads = convnextv2_tiny_convnext_mlp_backward_from_saved(upstream, saved)

    total = out.numel()
    for t in grads:
        total += t.numel()

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)

    off = 0
    off = _copy_flat(out, packed, off)
    for t in grads:
        off = _copy_flat(t, packed, off)

    return packed
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
