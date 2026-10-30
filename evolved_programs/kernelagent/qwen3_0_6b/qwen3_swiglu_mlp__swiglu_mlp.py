"""adbench-tools artifact: task qwen3_0_6b.qwen3_swiglu_mlp.swiglu_mlp, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation of Qwen3-style SwiGLU MLP forward + backward.
#
# Operator:
#   gate = x @ gate_weight.T
#   up   = x @ up_weight.T
#   act  = silu(gate) * up
#   y    = act @ down_weight.T
#
# Backward:
#   dact = dy @ down_weight
#   ddown = dy.T @ act
#   dgate = dact * up * silu'(gate)
#   dup   = dact * silu(gate)
#   dx = dgate @ gate_weight + dup @ up_weight
#   dgate_weight = dgate.T @ x
#   dup_weight   = dup.T @ x
#
# Fusion note:
#   Full forward+backward is split across several Triton kernels because the
#   operation contains multiple large matrix reductions with incompatible output
#   layouts. Compatible stages are fused: gate/up/activation, dact+dgate/dup,
#   combined dx paths, and combined gate/up weight-gradient paths.
#
# Fix:
#   Intermediate tensors gate/up/act/dgate/dup are stored as fp32 for stability,
#   while model tensors are bf16. tl.dot requires both operands to have the same
#   dtype, so fp32 intermediates are explicitly cast to bf16 at tensor-core
#   matmul boundaries.

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_act_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    act_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x = tl.load(
            x_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        gw = tl.load(
            gate_w_ptr + offs_n[None, :] * K + k_idxs[:, None],
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        )
        uw = tl.load(
            up_w_ptr + offs_n[None, :] * K + k_idxs[:, None],
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        )

        acc_gate += tl.dot(x, gw)
        acc_up += tl.dot(x, uw)

    sig = 1.0 / (1.0 + tl.exp(-acc_gate))
    silu = acc_gate * sig
    act = silu * acc_up

    out_ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(gate_ptr + out_ptrs, acc_gate, mask=mask)
    tl.store(up_ptr + out_ptrs, acc_up, mask=mask)
    tl.store(act_ptr + out_ptrs, act, mask=mask)


@triton.jit
def _act_down_kernel(
    act_ptr,
    down_w_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_o = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)

    for n0 in tl.range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n

        a = tl.load(
            act_ptr + offs_m[:, None] * N + n_idxs[None, :],
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        w = tl.load(
            down_w_ptr + offs_o[None, :] * N + n_idxs[:, None],
            mask=(offs_o[None, :] < K) & (n_idxs[:, None] < N),
            other=0.0,
        )

        acc += tl.dot(a, w)

    ptrs = y_ptr + offs_m[:, None] * K + offs_o[None, :]
    mask = (offs_m[:, None] < M) & (offs_o[None, :] < K)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _dgate_dup_kernel(
    dy_ptr,
    down_w_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    dact = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        dy = tl.load(
            dy_ptr + offs_m[:, None] * K + k_idxs[None, :],
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        dw = tl.load(
            down_w_ptr + k_idxs[:, None] * N + offs_n[None, :],
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        dact += tl.dot(dy, dw)

    ptrs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    g = tl.load(gate_ptr + ptrs, mask=mask, other=0.0)
    u = tl.load(up_ptr + ptrs, mask=mask, other=0.0)

    sig = 1.0 / (1.0 + tl.exp(-g))
    silu = g * sig
    dsilu = sig + g * sig * (1.0 - sig)

    dg = dact * u * dsilu
    du = dact * silu

    tl.store(dgate_ptr + ptrs, dg, mask=mask)
    tl.store(dup_ptr + ptrs, du, mask=mask)


@triton.jit
def _dx_kernel(
    dgate_ptr,
    dup_ptr,
    gate_w_ptr,
    up_w_ptr,
    dx_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n0 in tl.range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n

        dg = tl.load(
            dgate_ptr + offs_m[:, None] * N + n_idxs[None, :],
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        du = tl.load(
            dup_ptr + offs_m[:, None] * N + n_idxs[None, :],
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        gw = tl.load(
            gate_w_ptr + n_idxs[:, None] * K + offs_k[None, :],
            mask=(n_idxs[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )
        uw = tl.load(
            up_w_ptr + n_idxs[:, None] * K + offs_k[None, :],
            mask=(n_idxs[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc += tl.dot(dg, gw)
        acc += tl.dot(du, uw)

    ptrs = dx_ptr + offs_m[:, None] * K + offs_k[None, :]
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _dgatew_dupw_kernel(
    dgate_ptr,
    dup_ptr,
    x_ptr,
    dgate_w_ptr,
    dup_w_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc_gw = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc_uw = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        dg = tl.load(
            dgate_ptr + m_idxs[:, None] * N + offs_n[None, :],
            mask=(m_idxs[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        du = tl.load(
            dup_ptr + m_idxs[:, None] * N + offs_n[None, :],
            mask=(m_idxs[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)
        x = tl.load(
            x_ptr + m_idxs[:, None] * K + offs_k[None, :],
            mask=(m_idxs[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc_gw += tl.dot(tl.trans(dg), x)
        acc_uw += tl.dot(tl.trans(du), x)

    ptrs = offs_n[:, None] * K + offs_k[None, :]
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

    tl.store(dgate_w_ptr + ptrs, acc_gw, mask=mask)
    tl.store(dup_w_ptr + ptrs, acc_uw, mask=mask)


@triton.jit
def _ddown_kernel(
    dy_ptr,
    act_ptr,
    ddown_w_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        dy = tl.load(
            dy_ptr + m_idxs[:, None] * K + offs_k[None, :],
            mask=(m_idxs[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        act = tl.load(
            act_ptr + m_idxs[:, None] * N + offs_n[None, :],
            mask=(m_idxs[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.bfloat16)

        acc += tl.dot(tl.trans(dy), act)

    ptrs = offs_k[:, None] * N + offs_n[None, :]
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    tl.store(ddown_w_ptr + ptrs, acc, mask=mask)


@triton.jit
def _pack_kernel(
    y_ptr,
    dx_ptr,
    dgate_w_ptr,
    dup_w_ptr,
    ddown_w_ptr,
    out_ptr,
    total: tl.constexpr,
    n_y: tl.constexpr,
    n_dx: tl.constexpr,
    n_gw: tl.constexpr,
    n_uw: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    vals = tl.zeros((BLOCK,), dtype=tl.float32)

    base_y = 0
    base_dx = n_y
    base_gw = n_y + n_dx
    base_uw = n_y + n_dx + n_gw
    base_dw = n_y + n_dx + n_gw + n_uw

    m_y = offs < n_y
    v_y = tl.load(y_ptr + (offs - base_y), mask=m_y, other=0.0).to(tl.float32)
    vals = tl.where(m_y, v_y, vals)

    m_dx = (offs >= base_dx) & (offs < base_gw)
    v_dx = tl.load(dx_ptr + (offs - base_dx), mask=m_dx, other=0.0).to(tl.float32)
    vals = tl.where(m_dx, v_dx, vals)

    m_gw = (offs >= base_gw) & (offs < base_uw)
    v_gw = tl.load(dgate_w_ptr + (offs - base_gw), mask=m_gw, other=0.0).to(tl.float32)
    vals = tl.where(m_gw, v_gw, vals)

    m_uw = (offs >= base_uw) & (offs < base_dw)
    v_uw = tl.load(dup_w_ptr + (offs - base_uw), mask=m_uw, other=0.0).to(tl.float32)
    vals = tl.where(m_uw, v_uw, vals)

    m_dw = (offs >= base_dw) & (offs < total)
    v_dw = tl.load(ddown_w_ptr + (offs - base_dw), mask=m_dw, other=0.0).to(tl.float32)
    vals = tl.where(m_dw, v_dw, vals)

    tl.store(out_ptr + offs, vals, mask=offs < total)


def _shape_params(x, gate_weight, up_weight, down_weight):
    K = x.shape[-1]
    N = gate_weight.shape[0]
    M = x.numel() // K

    assert gate_weight.shape[1] == K
    assert up_weight.shape[0] == N and up_weight.shape[1] == K
    assert down_weight.shape[0] == K and down_weight.shape[1] == N
    assert x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda and down_weight.is_cuda

    return M, K, N


def qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    M, K, N = _shape_params(x, gate_weight, up_weight, down_weight)

    y = torch.empty_like(x)
    gate = torch.empty((M, N), device=x.device, dtype=torch.float32)
    up = torch.empty((M, N), device=x.device, dtype=torch.float32)
    act = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BM = 16
    BN = 32
    BK = 64

    _gate_up_act_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x,
        gate_weight,
        up_weight,
        gate,
        up,
        act,
        M,
        K,
        N,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
    )

    BO = 32
    RED_N = 64

    _act_down_kernel[(triton.cdiv(M, BM), triton.cdiv(K, BO))](
        act,
        down_weight,
        y,
        M,
        K,
        N,
        BLOCK_M=BM,
        BLOCK_O=BO,
        BLOCK_N=RED_N,
        num_warps=4,
    )

    saved = (x, gate_weight, up_weight, down_weight, gate, up, act)
    return y, saved


def qwen3_swiglu_mlp_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, gate_weight, up_weight, down_weight, gate, up, act = saved
    M, K, N = _shape_params(x, gate_weight, up_weight, down_weight)

    dx = torch.empty_like(x)
    dgate_weight = torch.empty_like(gate_weight)
    dup_weight = torch.empty_like(up_weight)
    ddown_weight = torch.empty_like(down_weight)

    dgate = torch.empty((M, N), device=x.device, dtype=torch.float32)
    dup = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BM = 16
    BN = 32
    BK = 64

    _dgate_dup_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        dy,
        down_weight,
        gate,
        up,
        dgate,
        dup,
        M,
        K,
        N,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        num_warps=4,
    )

    DX_BK = 32
    _dx_kernel[(triton.cdiv(M, BM), triton.cdiv(K, DX_BK))](
        dgate,
        dup,
        gate_weight,
        up_weight,
        dx,
        M,
        K,
        N,
        BLOCK_M=BM,
        BLOCK_K=DX_BK,
        BLOCK_N=BK,
        num_warps=4,
    )

    WG_BN = 32
    WG_BK = 32
    WG_BM = 32
    _dgatew_dupw_kernel[(triton.cdiv(N, WG_BN), triton.cdiv(K, WG_BK))](
        dgate,
        dup,
        x,
        dgate_weight,
        dup_weight,
        M,
        K,
        N,
        BLOCK_N=WG_BN,
        BLOCK_K=WG_BK,
        BLOCK_M=WG_BM,
        num_warps=4,
    )

    DW_BK = 32
    DW_BN = 32
    DW_BM = 32
    _ddown_kernel[(triton.cdiv(K, DW_BK), triton.cdiv(N, DW_BN))](
        dy,
        act,
        ddown_weight,
        M,
        K,
        N,
        BLOCK_K=DW_BK,
        BLOCK_N=DW_BN,
        BLOCK_M=DW_BM,
        num_warps=4,
    )

    return dx, dgate_weight, dup_weight, ddown_weight


def kernel_function(x, gate_weight, up_weight, down_weight, upstream):
    y, saved = qwen3_swiglu_mlp_forward_with_saved(
        x, gate_weight, up_weight, down_weight
    )
    dx, dgate_weight, dup_weight, ddown_weight = qwen3_swiglu_mlp_backward_from_saved(
        upstream, saved
    )

    n_y = y.numel()
    n_dx = dx.numel()
    n_gw = dgate_weight.numel()
    n_uw = dup_weight.numel()
    n_dw = ddown_weight.numel()
    total = n_y + n_dx + n_gw + n_uw + n_dw

    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    BLOCK = 1024
    _pack_kernel[(triton.cdiv(total, BLOCK),)](
        y,
        dx,
        dgate_weight,
        dup_weight,
        ddown_weight,
        out,
        total,
        n_y,
        n_dx,
        n_gw,
        n_uw,
        BLOCK=BLOCK,
        num_warps=4,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_swiglu_mlp', 'arguments': ['x', 'gate_weight', 'up_weight', 'down_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'gradient_return_order': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'scalars': [], 'deployment_entry': 'qwen3_swiglu_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen3SwigluMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_swiglu_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        out, saved = qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
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
        dx, dgate_weight, dup_weight, ddown_weight = qwen3_swiglu_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_weight, dup_weight, ddown_weight


def qwen3_swiglu_mlp_deployment(x, gate_weight, up_weight, down_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = Qwen3SwigluMlpFunction.apply(x, gate_weight, up_weight, down_weight)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Qwen3SwigluMlpModule(torch.nn.Module):
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
        return qwen3_swiglu_mlp_deployment(x, self.gate_weight, self.up_weight, self.down_weight)


DEPLOYMENT_ENTRY = "qwen3_swiglu_mlp_deployment"
