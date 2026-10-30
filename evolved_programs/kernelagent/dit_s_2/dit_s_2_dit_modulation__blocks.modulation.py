"""adbench-tools artifact: task dit_s_2.dit_s_2_dit_modulation.blocks.modulation, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for DiT modulation:
#   y = linear(silu(c), weight, bias)
#
# This file provides:
#   1. dit_s_2_dit_modulation_forward_with_saved
#   2. dit_s_2_dit_modulation_backward_from_saved
#   3. kernel_function
#
# Fusion note:
# - The forward kernel fuses SiLU, matrix multiplication, and bias addition.
# - The backward is split into two Triton kernels because it contains two
#   independent reductions / GEMMs:
#       grad_c      = (dout @ weight) * silu'(c)
#       grad_weight = dout.T @ silu(c)
#       grad_bias   = sum_m dout
# - kernel_function writes the forward output and all input gradients directly
#   into one flattened output tensor without using torch.cat or PyTorch math.

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    c_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    out_base: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute out = silu(c) @ weight.T + bias."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        c_vals = tl.load(
            c_ptr + offs_m[:, None] * stride_cm + k_idxs[None, :] * stride_ck,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-c_vals))
        silu = c_vals * sig

        w_vals = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k_idxs[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(silu, w_vals, acc, input_precision="ieee")

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_offsets = out_base + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _grad_c_kernel(
    c_ptr,
    w_ptr,
    dout_ptr,
    gc_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    gc_base: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute grad_c = (dout @ weight) * silu'(c)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n0 in tl.range(0, N, BLOCK_N):
        n_idxs = n0 + offs_n

        dout = tl.load(
            dout_ptr + offs_m[:, None] * stride_dm + n_idxs[None, :] * stride_dn,
            mask=(offs_m[:, None] < M) & (n_idxs[None, :] < N),
            other=0.0,
        )

        w = tl.load(
            w_ptr + n_idxs[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=(n_idxs[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        acc = tl.dot(dout, w, acc, input_precision="ieee")

    c_vals = tl.load(
        c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    ).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-c_vals))
    silu_grad = sig * (1.0 + c_vals * (1.0 - sig))
    grad = acc * silu_grad

    gc_offsets = gc_base + offs_m[:, None] * K + offs_k[None, :]
    tl.store(
        gc_ptr + gc_offsets,
        grad,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
    )


@triton.jit
def _grad_wb_kernel(
    c_ptr,
    dout_ptr,
    gw_ptr,
    gb_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_ck: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    gw_base: tl.constexpr,
    gb_base: tl.constexpr,
    SAME_OUTPUT_BUFFER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute grad_weight = dout.T @ silu(c), and grad_bias = sum(dout)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    acc_w = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m_idxs = m0 + offs_m

        dout = tl.load(
            dout_ptr + m_idxs[:, None] * stride_dm + offs_n[None, :] * stride_dn,
            mask=(m_idxs[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )

        c_vals = tl.load(
            c_ptr + m_idxs[:, None] * stride_cm + offs_k[None, :] * stride_ck,
            mask=(m_idxs[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-c_vals))
        silu = c_vals * sig

        acc_w = tl.dot(tl.trans(dout), silu, acc_w, input_precision="ieee")
        acc_b += tl.sum(dout.to(tl.float32), axis=0)

    gw_offsets = gw_base + offs_n[:, None] * K + offs_k[None, :]
    tl.store(
        gw_ptr + gw_offsets,
        acc_w,
        mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
    )

    # Only the first K-tile stores grad_bias.
    bias_mask = (pid_k == 0) & (offs_n < N)
    if SAME_OUTPUT_BUFFER:
        tl.store(gw_ptr + gb_base + offs_n, acc_b, mask=bias_mask)
    else:
        tl.store(gb_ptr + offs_n, acc_b, mask=bias_mask)


def dit_s_2_dit_modulation_forward_with_saved(c, weight, bias):
    """
    Forward pass for DiT modulation.

    Computes:
        output = linear(silu(c), weight, bias)

    All arithmetic is performed in Triton. The Python wrapper only allocates
    output storage and launches the Triton kernel.
    """
    assert c.is_cuda and weight.is_cuda and bias.is_cuda
    assert c.dim() == 2
    assert weight.dim() == 2
    assert bias.dim() == 1

    M = c.shape[0]
    K = c.shape[1]
    N = weight.shape[0]

    assert weight.shape[1] == K
    assert bias.shape[0] == N

    out = torch.empty((M, N), device=c.device, dtype=c.dtype)

    grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
    _forward_kernel[grid](
        c,
        weight,
        bias,
        out,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        weight.stride(0),
        weight.stride(1),
        0,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )

    saved = (c, weight, bias)
    return out, saved


def dit_s_2_dit_modulation_backward_from_saved(upstream, saved):
    """
    Backward pass for DiT modulation.

    Given dout, computes gradients with respect to:
        c, weight, bias

    All reductions and matrix products are implemented in Triton kernels.
    """
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    c, weight, bias = saved

    assert c.is_cuda and weight.is_cuda and bias.is_cuda and dout.is_cuda
    assert c.dim() == 2
    assert weight.dim() == 2
    assert dout.dim() == 2

    M = c.shape[0]
    K = c.shape[1]
    N = weight.shape[0]

    grad_c = torch.empty((M, K), device=c.device, dtype=c.dtype)
    grad_weight = torch.empty((N, K), device=weight.device, dtype=weight.dtype)
    grad_bias = torch.empty((N,), device=bias.device, dtype=bias.dtype)

    grid_gc = (triton.cdiv(M, 16), triton.cdiv(K, 64))
    _grad_c_kernel[grid_gc](
        c,
        weight,
        dout,
        grad_c,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        weight.stride(0),
        weight.stride(1),
        dout.stride(0),
        dout.stride(1),
        0,
        BLOCK_M=16,
        BLOCK_K=64,
        BLOCK_N=64,
        num_warps=4,
        num_stages=4,
    )

    grid_wb = (triton.cdiv(N, 32), triton.cdiv(K, 32))
    _grad_wb_kernel[grid_wb](
        c,
        dout,
        grad_weight,
        grad_bias,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        dout.stride(0),
        dout.stride(1),
        0,
        0,
        SAME_OUTPUT_BUFFER=False,
        BLOCK_N=32,
        BLOCK_K=32,
        BLOCK_M=32,
        num_warps=4,
        num_stages=4,
    )

    return grad_c, grad_weight, grad_bias


def kernel_function(c, weight, bias, upstream):
    """
    Fused forward+backward entry point required by the evaluator.

    Returns one flattened tensor containing, in order:
        forward output,
        grad_c,
        grad_weight,
        grad_bias

    The flattened result is written directly by Triton kernels. No torch.cat,
    PyTorch matmul, PyTorch activations, or PyTorch reductions are used.
    """
    if isinstance(upstream, (tuple, list)):
        dout = upstream[0]
    else:
        dout = upstream

    assert c.is_cuda and weight.is_cuda and bias.is_cuda and dout.is_cuda
    assert c.dim() == 2
    assert weight.dim() == 2
    assert dout.dim() == 2
    assert bias.dim() == 1

    M = c.shape[0]
    K = c.shape[1]
    N = weight.shape[0]

    assert weight.shape[1] == K
    assert bias.shape[0] == N
    assert dout.shape[0] == M
    assert dout.shape[1] == N

    out_numel = M * N
    gc_numel = M * K
    gw_numel = N * K
    gb_numel = N
    total = out_numel + gc_numel + gw_numel + gb_numel

    # Use c.dtype so actual.float() matches the dtype-rounding behavior of
    # the separately returned forward/backward tensors used by the test oracle.
    flat = torch.empty((total,), device=c.device, dtype=c.dtype)

    out_base = 0
    gc_base = out_base + out_numel
    gw_base = gc_base + gc_numel
    gb_base = gw_base + gw_numel

    grid_fwd = (triton.cdiv(M, 16), triton.cdiv(N, 64))
    _forward_kernel[grid_fwd](
        c,
        weight,
        bias,
        flat,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        weight.stride(0),
        weight.stride(1),
        out_base,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )

    grid_gc = (triton.cdiv(M, 16), triton.cdiv(K, 64))
    _grad_c_kernel[grid_gc](
        c,
        weight,
        dout,
        flat,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        weight.stride(0),
        weight.stride(1),
        dout.stride(0),
        dout.stride(1),
        gc_base,
        BLOCK_M=16,
        BLOCK_K=64,
        BLOCK_N=64,
        num_warps=4,
        num_stages=4,
    )

    grid_wb = (triton.cdiv(N, 32), triton.cdiv(K, 32))
    _grad_wb_kernel[grid_wb](
        c,
        dout,
        flat,
        flat,
        M,
        N,
        K,
        c.stride(0),
        c.stride(1),
        dout.stride(0),
        dout.stride(1),
        gw_base,
        gb_base,
        SAME_OUTPUT_BUFFER=True,
        BLOCK_N=32,
        BLOCK_K=32,
        BLOCK_M=32,
        num_warps=4,
        num_stages=4,
    )

    return flat
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'dit_s_2_dit_modulation', 'arguments': ['c', 'weight', 'bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dc', 'dweight', 'dbias'], 'gradient_return_order': ['dc', 'dweight', 'dbias'], 'scalars': [], 'deployment_entry': 'dit_s_2_dit_modulation_deployment', 'allowed_primitives': ['vendor_gemm']}


class DitS2DitModulationFunction(torch.autograd.Function):
    """Static autograd wiring for `dit_s_2_dit_modulation`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, c, weight, bias):
        out, saved = dit_s_2_dit_modulation_forward_with_saved(c, weight, bias)
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
        dc, dweight, dbias = dit_s_2_dit_modulation_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dc, dweight, dbias


def dit_s_2_dit_modulation_deployment(c, weight, bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if c.dim() > 2:
        _leading = c.shape[:-1]
        c = c.reshape(-1, c.shape[-1])
        weight = weight.reshape(-1, weight.shape[-1])
    out = DitS2DitModulationFunction.apply(c, weight, bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class DitS2DitModulationModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, c):
        return dit_s_2_dit_modulation_deployment(c, self.weight, self.bias)


DEPLOYMENT_ENTRY = "dit_s_2_dit_modulation_deployment"
