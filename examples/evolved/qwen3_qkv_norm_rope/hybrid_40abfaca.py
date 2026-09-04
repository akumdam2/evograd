import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

def _qwen3_next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


def _qwen3_num_warps(block_d):
    if block_d <= 64:
        return 1
    if block_d <= 256:
        return 4
    return 8


def _qwen3_check_cuda_same_device(tensors):
    dev = None
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("all tensor inputs must be torch.Tensor")
        if not t.is_cuda:
            raise ValueError("only CUDA tensors are supported")
        if dev is None:
            dev = t.device
        elif t.device != dev:
            raise ValueError("all tensors must be on the same CUDA device")


def _qwen3_check_float_dtype(name, t):
    if t.dtype != torch.float32 and t.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be torch.float32 or torch.bfloat16")


@triton.jit
def _qwen3_zero_kernel(
    OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(OUT + offs, tl.zeros((BLOCK,), tl.float32), mask=mask)


@triton.jit
def _qwen3_zero2_kernel(
    A,
    B,
    NA: tl.constexpr,
    NB: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    z = tl.zeros((BLOCK,), tl.float32)
    tl.store(A + offs, z, mask=offs < NA)
    tl.store(B + offs, z, mask=offs < NB)


@triton.jit
def _qwen3_reduce_two_head_partials_kernel(
    QPART,
    KPART,
    QOUT,
    KOUT,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
):
    d = tl.program_id(0)

    hq = tl.arange(0, BLOCK_HQ)
    qmask = hq < HQ
    qv = tl.load(QPART + hq * D + d, mask=qmask, other=0.0).to(tl.float32)
    tl.store(QOUT + d, tl.sum(qv, axis=0))

    hk = tl.arange(0, BLOCK_HK)
    kmask = hk < HK
    kv = tl.load(KPART + hk * D + d, mask=kmask, other=0.0).to(tl.float32)
    tl.store(KOUT + d, tl.sum(kv, axis=0))


@triton.jit
def _qwen3_rms_rope_fwd_kernel(
    H,
    GAMMA,
    COS,
    SIN,
    RSTD,
    OUT_BASE,
    T: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    head = pid % NH
    bt = pid // NH
    b = bt // T
    t = bt - b * T

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    half: tl.constexpr = D // 2
    partner = tl.where(offs < half, offs + half, offs - half)
    sign = tl.where(offs < half, -1.0, 1.0)

    h_base = bt * (NH * D) + head * D
    h = tl.load(H + h_base + offs, mask=mask, other=0.0).to(tl.float32)

    ss = tl.sum(h * h, axis=0)
    r = tl.rsqrt(ss / D + EPS)

    gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
    y = h * r * gamma

    h_partner = tl.load(H + h_base + partner, mask=mask, other=0.0).to(tl.float32)
    gamma_partner = tl.load(GAMMA + partner, mask=mask, other=0.0).to(tl.float32)
    y_partner = h_partner * r * gamma_partner
    rot = sign * y_partner

    c = tl.load(COS + t * D + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(SIN + t * D + offs, mask=mask, other=0.0).to(tl.float32)

    out = y * c + rot * s

    out_off = ((b * T + t) * NH + head) * D + offs
    tl.store(OUT_BASE + out_off, out, mask=mask)
    tl.store(RSTD + pid, r)


@triton.jit
def _qwen3_unrope_rms_bwd_kernel(
    DOUT,
    H,
    GAMMA,
    COS,
    SIN,
    RSTD,
    DH,
    DGAMMA,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    T: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    OUT_OFFSET: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    head = pid % NH
    bt = pid // NH
    b = bt // T
    t = bt - b * T

    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    half: tl.constexpr = D // 2
    partner = tl.where(offs < half, offs + half, offs - half)

    dout_base = b * S0 + head * S1 + t * S2
    dy = tl.load(DOUT + dout_base + offs * S3, mask=mask, other=0.0).to(tl.float32)
    dy_partner = tl.load(DOUT + dout_base + partner * S3, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(COS + t * D + offs, mask=mask, other=0.0).to(tl.float32)
    s_partner = tl.load(SIN + t * D + partner, mask=mask, other=0.0).to(tl.float32)

    sign = tl.where(offs < half, 1.0, -1.0)
    dnorm = dy * c + sign * dy_partner * s_partner

    h_base = bt * (NH * D) + head * D
    h = tl.load(H + h_base + offs, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(GAMMA + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RSTD + pid).to(tl.float32)

    dg = dnorm * h * r
    tl.atomic_add(DGAMMA + head * D + offs, dg, sem="relaxed", mask=mask)

    inner = tl.sum(dnorm * gamma * h, axis=0) / D
    dh = dnorm * gamma * r - h * inner * r * r * r

    out_base = bt * OUT_STRIDE + OUT_OFFSET + head * D
    tl.store(DH + out_base + offs, dh, mask=mask)


@triton.jit
def _qwen3_unrope_rms_bwd_block_kernel(
    DOUT,
    H,
    GAMMA,
    COS,
    SIN,
    RSTD,
    DH,
    PARTIAL,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    T: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    NROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    offs = tl.arange(0, BLOCK_D)

    rows = pid * BLOCK_M + rm
    row_mask = rows < NROWS
    col_mask = offs < D

    head = rows % NH
    bt = rows // NH
    b = bt // T
    t = bt - b * T

    half: tl.constexpr = D // 2
    partner = tl.where(offs < half, offs + half, offs - half)

    base = b[:, None] * S0 + head[:, None] * S1 + t[:, None] * S2
    mask = row_mask[:, None] & col_mask[None, :]

    dy = tl.load(DOUT + base + offs[None, :] * S3, mask=mask, other=0.0).to(tl.float32)
    dyp = tl.load(DOUT + base + partner[None, :] * S3, mask=mask, other=0.0).to(tl.float32)

    c = tl.load(COS + t[:, None] * D + offs[None, :], mask=mask, other=0.0).to(tl.float32)
    sp = tl.load(SIN + t[:, None] * D + partner[None, :], mask=mask, other=0.0).to(tl.float32)

    sign = tl.where(offs < half, 1.0, -1.0)
    dnorm = dy * c + sign[None, :] * dyp * sp

    h = tl.load(H + rows[:, None] * D + offs[None, :], mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(GAMMA + offs, mask=col_mask, other=0.0).to(tl.float32)
    r = tl.load(RSTD + rows, mask=row_mask, other=0.0).to(tl.float32)

    inner = tl.sum(dnorm * gamma[None, :] * h, axis=1) / D
    dh = dnorm * gamma[None, :] * r[:, None] - h * inner[:, None] * r[:, None] * r[:, None] * r[:, None]
    tl.store(DH + rows[:, None] * D + offs[None, :], dh, mask=mask)

    dg = dnorm * h * r[:, None]
    part = tl.sum(tl.where(mask, dg, 0.0), axis=0)
    tl.store(PARTIAL + pid * D + offs, part, mask=col_mask)


@triton.jit
def _qwen3_reduce_partial_kernel(
    PARTIAL,
    OUT,
    NPART: tl.constexpr,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    d = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < NPART
    vals = tl.load(PARTIAL + offs * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + d, tl.sum(vals, axis=0))


@triton.jit
def _qwen3_bhtd_to_flat_kernel(
    IN,
    OUT,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    T: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    OUT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    d = offs % D
    tmp = offs // D
    head = tmp % NH
    bt = tmp // NH
    b = bt // T
    t = bt - b * T

    val = tl.load(IN + b * S0 + head * S1 + t * S2 + d * S3, mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + bt * OUT_STRIDE + OUT_OFFSET + head * D + d, val, mask=mask)


@triton.jit
def _qwen3_add3_kernel(
    A,
    B,
    C,
    OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(C + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(OUT + offs, a + b + c, mask=mask)


def qwen3_qkv_norm_rope_forward_with_saved(
    x,
    q_weight,
    k_weight,
    v_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    eps=1e-6,
):
    _qwen3_check_cuda_same_device(
        (x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
    )

    for name, t in (
        ("x", x),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos", cos),
        ("sin", sin),
    ):
        _qwen3_check_float_dtype(name, t)

    if q_weight.dtype != x.dtype or k_weight.dtype != x.dtype or v_weight.dtype != x.dtype:
        raise TypeError("projection weights must have the same dtype as x")

    if x.dim() != 3:
        raise ValueError("x must have shape [B, T, C]")
    if q_weight.dim() != 2 or k_weight.dim() != 2 or v_weight.dim() != 2:
        raise ValueError("projection weights must be 2-D")
    if q_norm_weight.dim() != 1 or k_norm_weight.dim() != 1:
        raise ValueError("norm weights must be 1-D")
    if cos.dim() != 3 or sin.dim() != 3:
        raise ValueError("cos and sin must have shape [1, T, D]")

    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    for name, t in (
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos", cos),
        ("sin", sin),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    B = x.shape[0]
    T = x.shape[1]
    C = x.shape[2]
    QO = q_weight.shape[0]
    KO = k_weight.shape[0]
    VO = v_weight.shape[0]
    D = q_norm_weight.shape[0]

    if B <= 0 or T <= 0 or C <= 0:
        raise ValueError("B, T and C must be positive")
    if D <= 0 or D % 2 != 0:
        raise ValueError("head dimension D must be positive and even")
    if D > 1024:
        raise ValueError("D > 1024 is not supported")
    if k_norm_weight.shape[0] != D:
        raise ValueError("q_norm_weight and k_norm_weight must have the same length")
    if cos.shape != (1, T, D) or sin.shape != (1, T, D):
        raise ValueError("cos and sin must have shape [1, T, D]")
    if q_weight.shape[1] != C or k_weight.shape[1] != C or v_weight.shape[1] != C:
        raise ValueError("projection weight input dimensions must match x.shape[-1]")
    if QO % D != 0 or KO % D != 0 or VO % D != 0:
        raise ValueError("projection fan-outs must be multiples of D")
    if KO != VO:
        raise ValueError("k_weight and v_weight fan-outs must match")

    HQ = QO // D
    HK = KO // D
    if HK <= 0 or HQ <= 0:
        raise ValueError("number of heads must be positive")
    if HQ % HK != 0:
        raise ValueError("HQ must be divisible by HK")

    x2 = x.reshape(B * T, C)

    if x.dtype == torch.bfloat16:
        w_all = torch.cat((q_weight, k_weight, v_weight), dim=0)
        proj_all = vendor_gemm(x2, w_all.t())
        if not proj_all.is_contiguous():
            raise RuntimeError("vendor_gemm returned a non-contiguous tensor, unsupported")
        q_proj = proj_all[:, :QO].contiguous()
        k_proj = proj_all[:, QO:QO + KO].contiguous()
        v_proj = proj_all[:, QO + KO:].contiguous()
    else:
        q_proj = vendor_gemm(x2, q_weight.t())
        k_proj = vendor_gemm(x2, k_weight.t())
        v_proj = vendor_gemm(x2, v_weight.t())

        if not q_proj.is_contiguous() or not k_proj.is_contiguous() or not v_proj.is_contiguous():
            raise RuntimeError("vendor_gemm returned a non-contiguous tensor, unsupported")

    q_base = torch.empty((B, T, HQ, D), device=x.device, dtype=x.dtype)
    k_base = torch.empty((B, T, HK, D), device=x.device, dtype=x.dtype)

    q_rstd = torch.empty((B * T * HQ,), device=x.device, dtype=torch.float32)
    k_rstd = torch.empty((B * T * HK,), device=x.device, dtype=torch.float32)

    block_d = _qwen3_next_power_of_2(D)
    warps = _qwen3_num_warps(block_d)

    _qwen3_rms_rope_fwd_kernel[(B * T * HQ,)](
        q_proj,
        q_norm_weight,
        cos,
        sin,
        q_rstd,
        q_base,
        T,
        HQ,
        D,
        float(eps),
        BLOCK_D=block_d,
        num_warps=warps,
    )

    _qwen3_rms_rope_fwd_kernel[(B * T * HK,)](
        k_proj,
        k_norm_weight,
        cos,
        sin,
        k_rstd,
        k_base,
        T,
        HK,
        D,
        float(eps),
        BLOCK_D=block_d,
        num_warps=warps,
    )

    v_base = v_proj.reshape(B, T, HK, D)

    q = q_base.transpose(1, 2)
    k = k_base.transpose(1, 2)
    v = v_base.transpose(1, 2)

    saved_tensors = (
        x,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        q_proj,
        k_proj,
        q_rstd,
        k_rstd,
    )
    return (q, k, v), saved_tensors


def qwen3_qkv_norm_rope_backward_from_saved(output_grads, saved_tensors, eps=1e-6):
    if not isinstance(output_grads, (tuple, list)) or len(output_grads) != 3:
        raise ValueError("output_grads must be a tuple/list (dq, dk, dv)")
    dq, dk, dv = output_grads
    if dq is None or dk is None or dv is None:
        raise ValueError("None output gradients are not supported")

    if not isinstance(saved_tensors, (tuple, list)) or len(saved_tensors) != 12:
        raise ValueError("saved_tensors has an unsupported format")

    (
        x,
        q_weight,
        k_weight,
        v_weight,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        q_proj,
        k_proj,
        q_rstd,
        k_rstd,
    ) = saved_tensors

    _qwen3_check_cuda_same_device(
        (
            dq,
            dk,
            dv,
            x,
            q_weight,
            k_weight,
            v_weight,
            q_norm_weight,
            k_norm_weight,
            cos,
            sin,
            q_proj,
            k_proj,
            q_rstd,
            k_rstd,
        )
    )

    for name, t in (
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
        ("x", x),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos", cos),
        ("sin", sin),
        ("q_proj", q_proj),
        ("k_proj", k_proj),
    ):
        _qwen3_check_float_dtype(name, t)

    if q_rstd.dtype != torch.float32 or k_rstd.dtype != torch.float32:
        raise TypeError("saved rstd tensors must be torch.float32")
    if q_weight.dtype != x.dtype or k_weight.dtype != x.dtype or v_weight.dtype != x.dtype:
        raise TypeError("projection weights must have the same dtype as x")
    if q_proj.dtype != x.dtype or k_proj.dtype != x.dtype:
        raise TypeError("saved projection tensors must have the same dtype as x")

    if x.dim() != 3:
        raise ValueError("saved x must have shape [B, T, C]")

    B = x.shape[0]
    T = x.shape[1]
    C = x.shape[2]
    D = q_norm_weight.shape[0]

    if B <= 0 or T <= 0 or C <= 0:
        raise ValueError("B, T and C must be positive")
    if D <= 0 or D % 2 != 0:
        raise ValueError("head dimension D must be positive and even")
    if D > 1024:
        raise ValueError("D > 1024 is not supported")

    QO = q_weight.shape[0]
    KO = k_weight.shape[0]
    VO = v_weight.shape[0]
    if q_weight.dim() != 2 or k_weight.dim() != 2 or v_weight.dim() != 2:
        raise ValueError("saved projection weights must be 2-D")
    if q_weight.shape[1] != C or k_weight.shape[1] != C or v_weight.shape[1] != C:
        raise ValueError("saved projection weight input dimensions must match x")
    if QO % D != 0 or KO % D != 0 or VO % D != 0:
        raise ValueError("projection fan-outs must be multiples of D")
    if KO != VO:
        raise ValueError("k and v fan-outs must match")

    HQ = QO // D
    HK = KO // D
    if HQ <= 0 or HK <= 0 or HQ % HK != 0:
        raise ValueError("invalid head counts")

    if q_norm_weight.shape != (D,) or k_norm_weight.shape != (D,):
        raise ValueError("norm weights have unsupported shapes")
    if cos.shape != (1, T, D) or sin.shape != (1, T, D):
        raise ValueError("cos and sin must have shape [1, T, D]")
    if dq.shape != (B, HQ, T, D):
        raise ValueError("dq has an unsupported shape")
    if dk.shape != (B, HK, T, D):
        raise ValueError("dk has an unsupported shape")
    if dv.shape != (B, HK, T, D):
        raise ValueError("dv has an unsupported shape")
    if q_proj.shape != (B * T, HQ * D) or k_proj.shape != (B * T, HK * D):
        raise ValueError("saved q_proj/k_proj have unsupported shapes")
    if q_rstd.shape != (B * T * HQ,) or k_rstd.shape != (B * T * HK,):
        raise ValueError("saved q_rstd/k_rstd have unsupported shapes")

    for name, t in (
        ("x", x),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("v_weight", v_weight),
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos", cos),
        ("sin", sin),
        ("q_proj", q_proj),
        ("k_proj", k_proj),
        ("q_rstd", q_rstd),
        ("k_rstd", k_rstd),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    x2 = x.reshape(B * T, C)

    if x.dtype == torch.float32:
        dq_proj = torch.empty((B * T, HQ * D), device=x.device, dtype=x.dtype)
        dk_proj = torch.empty((B * T, HK * D), device=x.device, dtype=x.dtype)
        dv_proj = torch.empty((B * T, HK * D), device=x.device, dtype=x.dtype)
        d_all = None
        q_store = dq_proj
        k_store = dk_proj
        v_store = dv_proj
        q_out_stride = QO
        k_out_stride = KO
        v_out_stride = VO
        q_out_offset = 0
        k_out_offset = 0
        v_out_offset = 0
    else:
        total_o = QO + KO + VO
        d_all = torch.empty((B * T, total_o), device=x.device, dtype=x.dtype)
        dq_proj = None
        dk_proj = None
        dv_proj = None
        q_store = d_all
        k_store = d_all
        v_store = d_all
        q_out_stride = total_o
        k_out_stride = total_o
        v_out_stride = total_o
        q_out_offset = 0
        k_out_offset = QO
        v_out_offset = QO + KO

    dq_norm_weight_part = torch.empty((HQ * D,), device=x.device, dtype=torch.float32)
    dk_norm_weight_part = torch.empty((HK * D,), device=x.device, dtype=torch.float32)
    dq_norm_weight_fp32 = torch.empty((D,), device=x.device, dtype=torch.float32)
    dk_norm_weight_fp32 = torch.empty((D,), device=x.device, dtype=torch.float32)

    block_d = _qwen3_next_power_of_2(D)
    warps = _qwen3_num_warps(block_d)
    zero_block = 1024
    nq_part = HQ * D
    nk_part = HK * D
    n_part = max(nq_part, nk_part)

    _qwen3_zero2_kernel[(triton.cdiv(n_part, zero_block),)](
        dq_norm_weight_part,
        dk_norm_weight_part,
        nq_part,
        nk_part,
        n_part,
        BLOCK=zero_block,
        num_warps=4,
    )

    _qwen3_unrope_rms_bwd_kernel[(B * T * HQ,)](
        dq,
        q_proj,
        q_norm_weight,
        cos,
        sin,
        q_rstd,
        q_store,
        dq_norm_weight_part,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        T,
        HQ,
        D,
        q_out_stride,
        q_out_offset,
        BLOCK_D=block_d,
        num_warps=warps,
    )

    _qwen3_unrope_rms_bwd_kernel[(B * T * HK,)](
        dk,
        k_proj,
        k_norm_weight,
        cos,
        sin,
        k_rstd,
        k_store,
        dk_norm_weight_part,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        T,
        HK,
        D,
        k_out_stride,
        k_out_offset,
        BLOCK_D=block_d,
        num_warps=warps,
    )

    _qwen3_reduce_two_head_partials_kernel[(D,)](
        dq_norm_weight_part,
        dk_norm_weight_part,
        dq_norm_weight_fp32,
        dk_norm_weight_fp32,
        HQ,
        HK,
        D,
        BLOCK_HQ=_qwen3_next_power_of_2(HQ),
        BLOCK_HK=_qwen3_next_power_of_2(HK),
        num_warps=1,
    )

    total_v = B * T * HK * D
    copy_block = 256
    _qwen3_bhtd_to_flat_kernel[(triton.cdiv(total_v, copy_block),)](
        dv,
        v_store,
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        T,
        HK,
        D,
        total_v,
        v_out_stride,
        v_out_offset,
        BLOCK=copy_block,
        num_warps=4,
    )

    if x.dtype == torch.float32:
        dq_weight = vendor_gemm(dq_proj.t(), x2)
        dk_weight = vendor_gemm(dk_proj.t(), x2)
        dv_weight = vendor_gemm(dv_proj.t(), x2)

        dx_q = vendor_gemm(dq_proj, q_weight)
        dx_k = vendor_gemm(dk_proj, k_weight)
        dx_v = vendor_gemm(dv_proj, v_weight)

        dx = torch.empty_like(x)
        n_dx = B * T * C
        add_block = 256
        _qwen3_add3_kernel[(triton.cdiv(n_dx, add_block),)](
            dx_q,
            dx_k,
            dx_v,
            dx,
            n_dx,
            BLOCK=add_block,
            num_warps=4,
        )
    else:
        w_all = torch.cat((q_weight, k_weight, v_weight), dim=0)

        d_weight_all = vendor_gemm(d_all.t(), x2)
        dx = vendor_gemm(d_all, w_all).reshape(B, T, C)

        dq_weight = d_weight_all[:QO, :]
        dk_weight = d_weight_all[QO:QO + KO, :]
        dv_weight = d_weight_all[QO + KO:, :]

    if q_norm_weight.dtype == torch.float32:
        dq_norm_weight = dq_norm_weight_fp32
    else:
        dq_norm_weight = dq_norm_weight_fp32.to(q_norm_weight.dtype)

    if k_norm_weight.dtype == torch.float32:
        dk_norm_weight = dk_norm_weight_fp32
    else:
        dk_norm_weight = dk_norm_weight_fp32.to(k_norm_weight.dtype)

    return dx, dq_weight, dk_weight, dv_weight, dq_norm_weight, dk_norm_weight


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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_qkv_norm_rope', 'arguments': ['x', 'q_weight', 'k_weight', 'v_weight', 'q_norm_weight', 'k_norm_weight', 'cos', 'sin', 'eps'], 'outputs': ['q', 'k', 'v'], 'upstream_grads': ['dq', 'dk', 'dv'], 'input_grads': ['dx', 'dq_weight', 'dk_weight', 'dv_weight', 'dq_norm_weight', 'dk_norm_weight'], 'gradient_return_order': ['dx', 'dq_weight', 'dk_weight', 'dv_weight', 'dq_norm_weight', 'dk_norm_weight', 'None', 'None', 'None'], 'scalars': ['eps'], 'deployment_entry': 'qwen3_qkv_norm_rope_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen3QkvNormRopeFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_qkv_norm_rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps):
        (q, k, v), saved = qwen3_qkv_norm_rope_forward_with_saved(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps)
        ctx.save_for_backward(*saved)
        ctx.eps = eps
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        dx, dq_weight, dk_weight, dv_weight, dq_norm_weight, dk_norm_weight = qwen3_qkv_norm_rope_backward_from_saved(
            (dq, dk, dv), ctx.saved_tensors, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dq_weight, dk_weight, dv_weight, dq_norm_weight, dk_norm_weight, None, None, None


def qwen3_qkv_norm_rope_deployment(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen3QkvNormRopeFunction.apply(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps)


class Qwen3QkvNormRopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, eps=1e-06):
        super().__init__()
        self.q_weight = (q_weight if isinstance(q_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(q_weight))
        self.k_weight = (k_weight if isinstance(k_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(k_weight))
        self.v_weight = (v_weight if isinstance(v_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(v_weight))
        self.q_norm_weight = (q_norm_weight if isinstance(q_norm_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(q_norm_weight))
        self.k_norm_weight = (k_norm_weight if isinstance(k_norm_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(k_norm_weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return qwen3_qkv_norm_rope_deployment(x, self.q_weight, self.k_weight, self.v_weight, self.q_norm_weight, self.k_norm_weight, cos, sin, self.eps)


DEPLOYMENT_ENTRY = "qwen3_qkv_norm_rope_deployment"
