"""adbench-tools artifact: task qwen2_vl_2b.qwen2_vl_2b_text_mlp.model.language_model.layers.mlp, produced by kernelagent.

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


# The MLP consists of several large GEMM/reduction stages with incompatible
# output shapes/reduction axes:
#   forward: x @ gate_w.T, x @ up_w.T, hidden @ down_w.T
#   backward: dy @ down_w, dgate.T @ x, dup.T @ x, dy.T @ hidden
# These cannot be collapsed into a single Triton program without either
# recomputing large reductions per element or requiring cross-program
# synchronization.  We therefore fuse compatible pointwise epilogues into the
# GEMM kernels and keep the independent GEMM reductions as separate kernels.


@triton.jit
def _gate_up_silu_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    hidden_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_k = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * H + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < H),
            other=0.0,
        )
        wg = tl.load(
            gate_w_ptr + offs_i[None, :] * H + k[:, None],
            mask=(offs_i[None, :] < I) & (k[:, None] < H),
            other=0.0,
        )
        wu = tl.load(
            up_w_ptr + offs_i[None, :] * H + k[:, None],
            mask=(offs_i[None, :] < I) & (k[:, None] < H),
            other=0.0,
        )

        acc_g = tl.dot(x, wg, acc_g)
        acc_u = tl.dot(x, wu, acc_u)

    if IS_BF16:
        gate = acc_g.to(tl.bfloat16).to(tl.float32)
        up = acc_u.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        gate = acc_g.to(tl.float16).to(tl.float32)
        up = acc_u.to(tl.float16).to(tl.float32)
    else:
        gate = acc_g
        up = acc_u

    sig = 1.0 / (1.0 + tl.exp(-gate))
    hidden = gate * sig * up

    offs = offs_m[:, None] * I + offs_i[None, :]
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)

    tl.store(gate_ptr + offs, gate, mask=mask)
    tl.store(up_ptr + offs, up, mask=mask)
    tl.store(hidden_ptr + offs, hidden, mask=mask)


@triton.jit
def _down_to_out_kernel(
    hidden_ptr,
    down_w_ptr,
    out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        k = k0 + offs_k

        a = tl.load(
            hidden_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        b = tl.load(
            down_w_ptr + offs_h[None, :] * I + k[:, None],
            mask=(offs_h[None, :] < H) & (k[:, None] < I),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16)
    elif IS_FP16:
        v = acc.to(tl.float16)
    else:
        v = acc

    tl.store(
        out_ptr + offs_m[:, None] * H + offs_h[None, :],
        v,
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
    )


@triton.jit
def _down_to_pack_kernel(
    hidden_ptr,
    down_w_ptr,
    pack_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        k = k0 + offs_k

        a = tl.load(
            hidden_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        b = tl.load(
            down_w_ptr + offs_h[None, :] * I + k[:, None],
            mask=(offs_h[None, :] < H) & (k[:, None] < I),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        v = acc.to(tl.float16).to(tl.float32)
    else:
        v = acc

    tl.store(
        pack_ptr + offs_m[:, None] * H + offs_h[None, :],
        v,
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
    )


@triton.jit
def _dhidden_to_dgate_dup_kernel(
    dy_ptr,
    down_w_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k

        a = tl.load(
            dy_ptr + offs_m[:, None] * H + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < H),
            other=0.0,
        )
        b = tl.load(
            down_w_ptr + k[:, None] * I + offs_i[None, :],
            mask=(k[:, None] < H) & (offs_i[None, :] < I),
            other=0.0,
        )

        acc = tl.dot(a, b, acc)

    if IS_BF16:
        dh = acc.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        dh = acc.to(tl.float16).to(tl.float32)
    else:
        dh = acc

    offs = offs_m[:, None] * I + offs_i[None, :]
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I)

    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    dsilu = sig * (1.0 + gate * (1.0 - sig))

    tl.store(dgate_ptr + offs, dh * up * dsilu, mask=mask)
    tl.store(dup_ptr + offs, dh * silu, mask=mask)


@triton.jit
def _dx_to_pack_kernel(
    dgate_ptr,
    dup_ptr,
    gate_w_ptr,
    up_w_ptr,
    pack_ptr,
    pack_offset: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        k = k0 + offs_k

        dg = tl.load(
            dgate_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        du = tl.load(
            dup_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        wg = tl.load(
            gate_w_ptr + k[:, None] * H + offs_h[None, :],
            mask=(k[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )
        wu = tl.load(
            up_w_ptr + k[:, None] * H + offs_h[None, :],
            mask=(k[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc = tl.dot(dg, wg, acc)
        acc = tl.dot(du, wu, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        v = acc.to(tl.float16).to(tl.float32)
    else:
        v = acc

    tl.store(
        pack_ptr + pack_offset + offs_m[:, None] * H + offs_h[None, :],
        v,
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
    )


@triton.jit
def _dx_to_tensor_kernel(
    dgate_ptr,
    dup_ptr,
    gate_w_ptr,
    up_w_ptr,
    dx_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in range(0, I, BLOCK_K):
        k = k0 + offs_k

        dg = tl.load(
            dgate_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        du = tl.load(
            dup_ptr + offs_m[:, None] * I + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < I),
            other=0.0,
        )
        wg = tl.load(
            gate_w_ptr + k[:, None] * H + offs_h[None, :],
            mask=(k[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )
        wu = tl.load(
            up_w_ptr + k[:, None] * H + offs_h[None, :],
            mask=(k[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc = tl.dot(dg, wg, acc)
        acc = tl.dot(du, wu, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16)
    elif IS_FP16:
        v = acc.to(tl.float16)
    else:
        v = acc

    tl.store(
        dx_ptr + offs_m[:, None] * H + offs_h[None, :],
        v,
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
    )


@triton.jit
def _gate_up_weight_to_pack_kernel(
    dgate_ptr,
    dup_ptr,
    x_ptr,
    pack_ptr,
    gate_offset: tl.constexpr,
    up_offset: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_I, BLOCK_H), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_I, BLOCK_H), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_K):
        m = m0 + offs_m

        dg = tl.load(
            dgate_ptr + m[None, :] * I + offs_i[:, None],
            mask=(m[None, :] < M) & (offs_i[:, None] < I),
            other=0.0,
        )
        du = tl.load(
            dup_ptr + m[None, :] * I + offs_i[:, None],
            mask=(m[None, :] < M) & (offs_i[:, None] < I),
            other=0.0,
        )
        x = tl.load(
            x_ptr + m[:, None] * H + offs_h[None, :],
            mask=(m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc_g = tl.dot(dg, x, acc_g)
        acc_u = tl.dot(du, x, acc_u)

    if IS_BF16:
        vg = acc_g.to(tl.bfloat16).to(tl.float32)
        vu = acc_u.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        vg = acc_g.to(tl.float16).to(tl.float32)
        vu = acc_u.to(tl.float16).to(tl.float32)
    else:
        vg = acc_g
        vu = acc_u

    mask = (offs_i[:, None] < I) & (offs_h[None, :] < H)
    offs = offs_i[:, None] * H + offs_h[None, :]

    tl.store(pack_ptr + gate_offset + offs, vg, mask=mask)
    tl.store(pack_ptr + up_offset + offs, vu, mask=mask)


@triton.jit
def _gate_up_weight_to_tensor_kernel(
    dgate_ptr,
    dup_ptr,
    x_ptr,
    dgw_ptr,
    duw_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = tl.arange(0, BLOCK_K)

    acc_g = tl.zeros((BLOCK_I, BLOCK_H), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_I, BLOCK_H), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_K):
        m = m0 + offs_m

        dg = tl.load(
            dgate_ptr + m[None, :] * I + offs_i[:, None],
            mask=(m[None, :] < M) & (offs_i[:, None] < I),
            other=0.0,
        )
        du = tl.load(
            dup_ptr + m[None, :] * I + offs_i[:, None],
            mask=(m[None, :] < M) & (offs_i[:, None] < I),
            other=0.0,
        )
        x = tl.load(
            x_ptr + m[:, None] * H + offs_h[None, :],
            mask=(m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc_g = tl.dot(dg, x, acc_g)
        acc_u = tl.dot(du, x, acc_u)

    if IS_BF16:
        vg = acc_g.to(tl.bfloat16)
        vu = acc_u.to(tl.bfloat16)
    elif IS_FP16:
        vg = acc_g.to(tl.float16)
        vu = acc_u.to(tl.float16)
    else:
        vg = acc_g
        vu = acc_u

    mask = (offs_i[:, None] < I) & (offs_h[None, :] < H)
    offs = offs_i[:, None] * H + offs_h[None, :]

    tl.store(dgw_ptr + offs, vg, mask=mask)
    tl.store(duw_ptr + offs, vu, mask=mask)


@triton.jit
def _ddown_weight_to_pack_kernel(
    dy_ptr,
    hidden_ptr,
    pack_ptr,
    pack_offset: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_m = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_H, BLOCK_I), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_K):
        m = m0 + offs_m

        dy = tl.load(
            dy_ptr + m[None, :] * H + offs_h[:, None],
            mask=(m[None, :] < M) & (offs_h[:, None] < H),
            other=0.0,
        )
        hid = tl.load(
            hidden_ptr + m[:, None] * I + offs_i[None, :],
            mask=(m[:, None] < M) & (offs_i[None, :] < I),
            other=0.0,
        )

        acc = tl.dot(dy, hid, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16).to(tl.float32)
    elif IS_FP16:
        v = acc.to(tl.float16).to(tl.float32)
    else:
        v = acc

    tl.store(
        pack_ptr + pack_offset + offs_h[:, None] * I + offs_i[None, :],
        v,
        mask=(offs_h[:, None] < H) & (offs_i[None, :] < I),
    )


@triton.jit
def _ddown_weight_to_tensor_kernel(
    dy_ptr,
    hidden_ptr,
    ddw_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_i = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_m = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_H, BLOCK_I), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_K):
        m = m0 + offs_m

        dy = tl.load(
            dy_ptr + m[None, :] * H + offs_h[:, None],
            mask=(m[None, :] < M) & (offs_h[:, None] < H),
            other=0.0,
        )
        hid = tl.load(
            hidden_ptr + m[:, None] * I + offs_i[None, :],
            mask=(m[:, None] < M) & (offs_i[None, :] < I),
            other=0.0,
        )

        acc = tl.dot(dy, hid, acc)

    if IS_BF16:
        v = acc.to(tl.bfloat16)
    elif IS_FP16:
        v = acc.to(tl.float16)
    else:
        v = acc

    tl.store(
        ddw_ptr + offs_h[:, None] * I + offs_i[None, :],
        v,
        mask=(offs_h[:, None] < H) & (offs_i[None, :] < I),
    )


def qwen2_vl_2b_text_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    H = x.shape[-1]
    I = gate_weight.shape[0]
    M = x.numel() // H

    is_bf16 = x.dtype == torch.bfloat16
    is_fp16 = x.dtype == torch.float16

    out = torch.empty_like(x)
    gate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    up = torch.empty((M, I), device=x.device, dtype=x.dtype)
    hidden = torch.empty((M, I), device=x.device, dtype=x.dtype)

    BM = 16
    BI = 64
    BH = 64
    BK = 64

    _gate_up_silu_kernel[(triton.cdiv(M, BM), triton.cdiv(I, BI))](
        x,
        gate_weight,
        up_weight,
        gate,
        up,
        hidden,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _down_to_out_kernel[(triton.cdiv(M, BM), triton.cdiv(H, BH))](
        hidden,
        down_weight,
        out,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_H=BH,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    saved = (x, gate_weight, up_weight, down_weight, gate, up, hidden)
    return out, saved


def qwen2_vl_2b_text_mlp_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    x, gate_weight, up_weight, down_weight, gate, up, hidden = saved

    H = x.shape[-1]
    I = gate_weight.shape[0]
    M = x.numel() // H

    is_bf16 = x.dtype == torch.bfloat16
    is_fp16 = x.dtype == torch.float16

    dx = torch.empty_like(x)
    dgate_weight = torch.empty_like(gate_weight)
    dup_weight = torch.empty_like(up_weight)
    ddown_weight = torch.empty_like(down_weight)

    dgate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    dup = torch.empty((M, I), device=x.device, dtype=x.dtype)

    BM = 16
    BI = 64
    BH = 64
    BK = 64
    BIW = 32
    BHW = 64

    _dhidden_to_dgate_dup_kernel[(triton.cdiv(M, BM), triton.cdiv(I, BI))](
        upstream,
        down_weight,
        gate,
        up,
        dgate,
        dup,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _ddown_weight_to_tensor_kernel[(triton.cdiv(H, 32), triton.cdiv(I, BI))](
        upstream,
        hidden,
        ddown_weight,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_H=32,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _dx_to_tensor_kernel[(triton.cdiv(M, BM), triton.cdiv(H, BH))](
        dgate,
        dup,
        gate_weight,
        up_weight,
        dx,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_H=BH,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _gate_up_weight_to_tensor_kernel[(triton.cdiv(I, BIW), triton.cdiv(H, BHW))](
        dgate,
        dup,
        x,
        dgate_weight,
        dup_weight,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_I=BIW,
        BLOCK_H=BHW,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    return dx, dgate_weight, dup_weight, ddown_weight


def kernel_function(x, gate_weight, up_weight, down_weight, upstream):
    H = x.shape[-1]
    I = gate_weight.shape[0]
    M = x.numel() // H

    is_bf16 = x.dtype == torch.bfloat16
    is_fp16 = x.dtype == torch.float16

    out_size = M * H
    dx_size = M * H
    gw_size = I * H
    uw_size = I * H
    dw_size = H * I

    off_out = 0
    off_dx = off_out + out_size
    off_gw = off_dx + dx_size
    off_uw = off_gw + gw_size
    off_dw = off_uw + uw_size
    total = off_dw + dw_size

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)

    gate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    up = torch.empty((M, I), device=x.device, dtype=x.dtype)
    hidden = torch.empty((M, I), device=x.device, dtype=x.dtype)
    dgate = torch.empty((M, I), device=x.device, dtype=x.dtype)
    dup = torch.empty((M, I), device=x.device, dtype=x.dtype)

    BM = 16
    BI = 64
    BH = 64
    BK = 64
    BIW = 32
    BHW = 64

    _gate_up_silu_kernel[(triton.cdiv(M, BM), triton.cdiv(I, BI))](
        x,
        gate_weight,
        up_weight,
        gate,
        up,
        hidden,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _down_to_pack_kernel[(triton.cdiv(M, BM), triton.cdiv(H, BH))](
        hidden,
        down_weight,
        packed,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_H=BH,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _dhidden_to_dgate_dup_kernel[(triton.cdiv(M, BM), triton.cdiv(I, BI))](
        upstream,
        down_weight,
        gate,
        up,
        dgate,
        dup,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _ddown_weight_to_pack_kernel[(triton.cdiv(H, 32), triton.cdiv(I, BI))](
        upstream,
        hidden,
        packed,
        off_dw,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_H=32,
        BLOCK_I=BI,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _dx_to_pack_kernel[(triton.cdiv(M, BM), triton.cdiv(H, BH))](
        dgate,
        dup,
        gate_weight,
        up_weight,
        packed,
        off_dx,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_M=BM,
        BLOCK_H=BH,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

    _gate_up_weight_to_pack_kernel[(triton.cdiv(I, BIW), triton.cdiv(H, BHW))](
        dgate,
        dup,
        x,
        packed,
        off_gw,
        off_uw,
        M,
        H,
        I,
        IS_BF16=is_bf16,
        IS_FP16=is_fp16,
        BLOCK_I=BIW,
        BLOCK_H=BHW,
        BLOCK_K=BK,
        num_warps=4,
        num_stages=4,
    )

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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen2_vl_2b_text_mlp', 'arguments': ['x', 'gate_weight', 'up_weight', 'down_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'gradient_return_order': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'scalars': [], 'deployment_entry': 'qwen2_vl_2b_text_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen2Vl2BTextMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen2_vl_2b_text_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        out, saved = qwen2_vl_2b_text_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
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
        dx, dgate_weight, dup_weight, ddown_weight = qwen2_vl_2b_text_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_weight, dup_weight, ddown_weight


def qwen2_vl_2b_text_mlp_deployment(x, gate_weight, up_weight, down_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = Qwen2Vl2BTextMlpFunction.apply(x, gate_weight, up_weight, down_weight)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Qwen2Vl2BTextMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, gate_weight, up_weight, down_weight):
        super().__init__()
        self.gate_weight = (gate_weight if isinstance(gate_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(gate_weight))
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return qwen2_vl_2b_text_mlp_deployment(x, self.gate_weight, self.up_weight, self.down_weight)


DEPLOYMENT_ENTRY = "qwen2_vl_2b_text_mlp_deployment"
