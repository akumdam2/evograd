"""adbench-tools artifact: task llama_3_2_1b.llama3_swiglu_mlp.swiglu_mlp, produced by kernelagent.

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
def _fused_gate_up_swiglu_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    hidden_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    gw_stride_n: tl.constexpr,
    gw_stride_k: tl.constexpr,
    uw_stride_n: tl.constexpr,
    uw_stride_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused first MLP stage:
      gate = x @ gate_weight.T
      up   = x @ up_weight.T
      hidden = silu(gate) * up

    Stores gate/up/hidden for the backward pass.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        x_vals = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + k_idxs[None, :] * x_stride_k,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        gate_vals = tl.load(
            gate_w_ptr + offs_n[None, :] * gw_stride_n + k_idxs[:, None] * gw_stride_k,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        ).to(tl.float32)

        up_vals = tl.load(
            up_w_ptr + offs_n[None, :] * uw_stride_n + k_idxs[:, None] * uw_stride_k,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        ).to(tl.float32)

        acc_gate += tl.dot(x_vals, gate_vals, input_precision="tf32")
        acc_up += tl.dot(x_vals, up_vals, input_precision="tf32")

    sigmoid_gate = 1.0 / (1.0 + tl.exp(-acc_gate))
    hidden = acc_gate * sigmoid_gate * acc_up

    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(gate_ptr + out_offsets, acc_gate, mask=mask)
    tl.store(up_ptr + out_offsets, acc_up, mask=mask)
    tl.store(hidden_ptr + out_offsets, hidden, mask=mask)


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    a_stride_m: tl.constexpr,
    a_stride_k: tl.constexpr,
    b_stride_k: tl.constexpr,
    b_stride_n: tl.constexpr,
    c_stride_m: tl.constexpr,
    c_stride_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Generic matmul:
      C[M, N] = A[M, K] @ B[K, N]

    Strides describe the logical A/B/C layouts.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a_vals = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + k_idxs[None, :] * a_stride_k,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        b_vals = tl.load(
            b_ptr + k_idxs[:, None] * b_stride_k + offs_n[None, :] * b_stride_n,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(a_vals, b_vals, input_precision="tf32")

    c_ptrs = c_ptr + offs_m[:, None] * c_stride_m + offs_n[None, :] * c_stride_n
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def _swiglu_backward_elementwise_kernel(
    grad_hidden_ptr,
    gate_ptr,
    up_ptr,
    dgate_ptr,
    dup_ptr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Elementwise SwiGLU backward:
      hidden = silu(gate) * up
      d_up = d_hidden * silu(gate)
      d_gate = d_hidden * up * silu'(gate)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    grad_hidden = tl.load(grad_hidden_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    silu_grad = sig + gate * sig * (1.0 - sig)

    dup = grad_hidden * silu
    dgate = grad_hidden * up * silu_grad

    tl.store(dup_ptr + offs, dup, mask=mask)
    tl.store(dgate_ptr + offs, dgate, mask=mask)


@triton.jit
def _dual_dx_kernel(
    dgate_ptr,
    dup_ptr,
    gate_w_ptr,
    up_w_ptr,
    dx_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    gw_stride_i: tl.constexpr,
    gw_stride_h: tl.constexpr,
    uw_stride_i: tl.constexpr,
    uw_stride_h: tl.constexpr,
    dx_stride_m: tl.constexpr,
    dx_stride_h: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """
    Fused input-gradient matmul:
      dx = dgate @ gate_weight + dup @ up_weight
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = tl.arange(0, BLOCK_I)

    acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for i0 in tl.range(0, I, BLOCK_I):
        i_idxs = i0 + offs_i

        dg = tl.load(
            dgate_ptr + offs_m[:, None] * I + i_idxs[None, :],
            mask=(offs_m[:, None] < M) & (i_idxs[None, :] < I),
            other=0.0,
        ).to(tl.float32)

        du = tl.load(
            dup_ptr + offs_m[:, None] * I + i_idxs[None, :],
            mask=(offs_m[:, None] < M) & (i_idxs[None, :] < I),
            other=0.0,
        ).to(tl.float32)

        gw = tl.load(
            gate_w_ptr + i_idxs[:, None] * gw_stride_i + offs_h[None, :] * gw_stride_h,
            mask=(i_idxs[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        ).to(tl.float32)

        uw = tl.load(
            up_w_ptr + i_idxs[:, None] * uw_stride_i + offs_h[None, :] * uw_stride_h,
            mask=(i_idxs[:, None] < I) & (offs_h[None, :] < H),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(dg, gw, input_precision="tf32")
        acc += tl.dot(du, uw, input_precision="tf32")

    ptrs = dx_ptr + offs_m[:, None] * dx_stride_m + offs_h[None, :] * dx_stride_h
    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _pack_kernel(
    y_ptr,
    dx_ptr,
    dgw_ptr,
    duw_ptr,
    ddw_ptr,
    out_ptr,
    y_num: tl.constexpr,
    dx_num: tl.constexpr,
    dgw_num: tl.constexpr,
    duw_num: tl.constexpr,
    ddw_num: tl.constexpr,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Packs:
      [forward_output.flatten(),
       grad_x.flatten(),
       grad_gate_weight.flatten(),
       grad_up_weight.flatten(),
       grad_down_weight.flatten()]
    into one float32 vector.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < total

    b0 = y_num
    b1 = b0 + dx_num
    b2 = b1 + dgw_num
    b3 = b2 + duw_num

    m0 = offs < b0
    idx0 = offs

    m1 = (offs >= b0) & (offs < b1)
    idx1 = offs - b0

    m2 = (offs >= b1) & (offs < b2)
    idx2 = offs - b1

    m3 = (offs >= b2) & (offs < b3)
    idx3 = offs - b2

    m4 = offs >= b3
    idx4 = offs - b3

    v0 = tl.load(y_ptr + tl.where(m0, idx0, 0), mask=valid & m0, other=0.0).to(tl.float32)
    v1 = tl.load(dx_ptr + tl.where(m1, idx1, 0), mask=valid & m1, other=0.0).to(tl.float32)
    v2 = tl.load(dgw_ptr + tl.where(m2, idx2, 0), mask=valid & m2, other=0.0).to(tl.float32)
    v3 = tl.load(duw_ptr + tl.where(m3, idx3, 0), mask=valid & m3, other=0.0).to(tl.float32)
    v4 = tl.load(ddw_ptr + tl.where(m4, idx4, 0), mask=valid & m4, other=0.0).to(tl.float32)

    tl.store(out_ptr + offs, v0 + v1 + v2 + v3 + v4, mask=valid)


def _dims(x, gate_weight):
    h = x.shape[-1]
    i = gate_weight.shape[0]
    m = x.numel() // h
    return m, h, i


def llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    """
    Forward for Llama-style SwiGLU MLP.

    Computes, in Triton:
      gate = x @ gate_weight.T
      up = x @ up_weight.T
      hidden = silu(gate) * up
      y = hidden @ down_weight.T

    The first three stages are fused in one Triton kernel.  Intermediates are
    saved because the backward pass needs gate/up/hidden.
    """
    assert x.is_cuda and gate_weight.is_cuda and up_weight.is_cuda and down_weight.is_cuda
    assert x.is_contiguous()
    assert gate_weight.is_contiguous()
    assert up_weight.is_contiguous()
    assert down_weight.is_contiguous()

    m, h, i = _dims(x, gate_weight)

    gate = torch.empty((m, i), device=x.device, dtype=torch.float32)
    up = torch.empty((m, i), device=x.device, dtype=torch.float32)
    hidden = torch.empty((m, i), device=x.device, dtype=torch.float32)
    y = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)

    bm, bn, bk = 16, 32, 64
    grid1 = (triton.cdiv(m, bm), triton.cdiv(i, bn))
    _fused_gate_up_swiglu_kernel[grid1](
        x,
        gate_weight,
        up_weight,
        gate,
        up,
        hidden,
        m,
        i,
        h,
        h,
        1,
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_weight.stride(0),
        up_weight.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    grid2 = (triton.cdiv(m, bm), triton.cdiv(h, bn))
    _matmul_kernel[grid2](
        hidden,
        down_weight,
        y,
        m,
        h,
        i,
        i,
        1,
        down_weight.stride(1),
        down_weight.stride(0),
        h,
        1,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    saved = (x, gate_weight, up_weight, down_weight, gate, up, hidden)
    return y, saved


def llama3_swiglu_mlp_backward_from_saved(upstream, saved):
    """
    Backward for the saved forward pass, implemented entirely with Triton
    kernels.

    Returns gradients for:
      x, gate_weight, up_weight, down_weight
    """
    if isinstance(upstream, (tuple, list)):
        grad_y = upstream[0]
    else:
        grad_y = upstream

    x, gate_weight, up_weight, down_weight, gate, up, hidden = saved
    m, h, i = _dims(x, gate_weight)

    grad_hidden = torch.empty((m, i), device=x.device, dtype=torch.float32)
    dgate = torch.empty((m, i), device=x.device, dtype=torch.float32)
    dup = torch.empty((m, i), device=x.device, dtype=torch.float32)

    grad_x = torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype)
    grad_gate_weight = torch.empty(tuple(gate_weight.shape), device=x.device, dtype=gate_weight.dtype)
    grad_up_weight = torch.empty(tuple(up_weight.shape), device=x.device, dtype=up_weight.dtype)
    grad_down_weight = torch.empty(tuple(down_weight.shape), device=x.device, dtype=down_weight.dtype)

    bm, bn, bk = 16, 32, 64

    # grad_hidden = grad_y @ down_weight
    grid_gh = (triton.cdiv(m, bm), triton.cdiv(i, bn))
    _matmul_kernel[grid_gh](
        grad_y,
        down_weight,
        grad_hidden,
        m,
        i,
        h,
        h,
        1,
        down_weight.stride(0),
        down_weight.stride(1),
        i,
        1,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    # Elementwise SwiGLU backward.
    total_hidden = m * i
    block = 256
    _swiglu_backward_elementwise_kernel[(triton.cdiv(total_hidden, block),)](
        grad_hidden,
        gate,
        up,
        dgate,
        dup,
        total_hidden,
        BLOCK_SIZE=block,
        num_warps=4,
    )

    # grad_x = dgate @ gate_weight + dup @ up_weight
    grid_dx = (triton.cdiv(m, bm), triton.cdiv(h, bn))
    _dual_dx_kernel[grid_dx](
        dgate,
        dup,
        gate_weight,
        up_weight,
        grad_x,
        m,
        h,
        i,
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_weight.stride(0),
        up_weight.stride(1),
        h,
        1,
        BLOCK_M=bm,
        BLOCK_H=bn,
        BLOCK_I=bk,
        num_warps=4,
        num_stages=4,
    )

    # grad_gate_weight = dgate.T @ x
    grid_gw = (triton.cdiv(i, bm), triton.cdiv(h, bn))
    _matmul_kernel[grid_gw](
        dgate,
        x,
        grad_gate_weight,
        i,
        h,
        m,
        1,
        i,
        h,
        1,
        gate_weight.stride(0),
        gate_weight.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    # grad_up_weight = dup.T @ x
    _matmul_kernel[grid_gw](
        dup,
        x,
        grad_up_weight,
        i,
        h,
        m,
        1,
        i,
        h,
        1,
        up_weight.stride(0),
        up_weight.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    # grad_down_weight = grad_y.T @ hidden
    grid_dw = (triton.cdiv(h, bm), triton.cdiv(i, bn))
    _matmul_kernel[grid_dw](
        grad_y,
        hidden,
        grad_down_weight,
        h,
        i,
        m,
        1,
        h,
        i,
        1,
        down_weight.stride(0),
        down_weight.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=4,
        num_stages=4,
    )

    return grad_x, grad_gate_weight, grad_up_weight, grad_down_weight


def kernel_function(x, gate_weight, up_weight, down_weight, upstream):
    """
    Required fused forward+backward step.

    This wrapper launches Triton kernels for:
      1. fused gate/up projection + SiLU gating,
      2. down projection,
      3. backward matmuls and SwiGLU derivative,
      4. packing all outputs/gradients into one float32 vector.

    The Python layer only validates metadata, allocates tensors, configures
    launches, and returns the packed result.
    """
    y, saved = llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
    grad_x, grad_gate_weight, grad_up_weight, grad_down_weight = llama3_swiglu_mlp_backward_from_saved(
        upstream, saved
    )

    y_num = y.numel()
    dx_num = grad_x.numel()
    dgw_num = grad_gate_weight.numel()
    duw_num = grad_up_weight.numel()
    ddw_num = grad_down_weight.numel()
    total = y_num + dx_num + dgw_num + duw_num + ddw_num

    packed = torch.empty((total,), device=x.device, dtype=torch.float32)

    block = 1024
    _pack_kernel[(triton.cdiv(total, block),)](
        y,
        grad_x,
        grad_gate_weight,
        grad_up_weight,
        grad_down_weight,
        packed,
        y_num,
        dx_num,
        dgw_num,
        duw_num,
        ddw_num,
        total,
        BLOCK_SIZE=block,
        num_warps=4,
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_swiglu_mlp', 'arguments': ['x', 'gate_weight', 'up_weight', 'down_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'gradient_return_order': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'scalars': [], 'deployment_entry': 'llama3_swiglu_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Llama3SwigluMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_swiglu_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        out, saved = llama3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
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
        dx, dgate_weight, dup_weight, ddown_weight = llama3_swiglu_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_weight, dup_weight, ddown_weight


def llama3_swiglu_mlp_deployment(x, gate_weight, up_weight, down_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = Llama3SwigluMlpFunction.apply(x, gate_weight, up_weight, down_weight)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Llama3SwigluMlpModule(torch.nn.Module):
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
        return llama3_swiglu_mlp_deployment(x, self.gate_weight, self.up_weight, self.down_weight)


DEPLOYMENT_ENTRY = "llama3_swiglu_mlp_deployment"
