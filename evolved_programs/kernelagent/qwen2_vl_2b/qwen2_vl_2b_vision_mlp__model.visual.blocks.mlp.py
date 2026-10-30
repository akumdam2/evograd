"""adbench-tools artifact: task qwen2_vl_2b.qwen2_vl_2b_vision_mlp.model.visual.blocks.mlp, produced by kernelagent.

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


# -----------------------------------------------------------------------------
# Qwen2-VL vision MLP:
#   z = x @ up_weight.T + up_bias
#   a = quick_gelu(z) = z * sigmoid(1.702 * z)
#   y = a @ down_weight.T + down_bias
#
# Backward computes gradients w.r.t.:
#   x, up_weight, up_bias, down_weight, down_bias
#
# The public kernel_function returns a single flattened float32 tensor containing:
#   cat([y.flatten(), grad_x.flatten(), grad_up_weight.flatten(),
#        grad_up_bias.flatten(), grad_down_weight.flatten(),
#        grad_down_bias.flatten()])
#
# Fusion notes:
#   * The first forward Triton kernel fuses the up projection, bias add, and
#     quick-GELU activation.
#   * The grad-gz Triton kernel fuses the down-backprop matmul with the
#     quick-GELU derivative.
#   * Full end-to-end fusion into one kernel is not practical because the
#     operation contains several large matrix products and reductions with
#     different output layouts. All math is nevertheless performed in Triton.
# -----------------------------------------------------------------------------


@triton.jit
def _up_act_kernel(
    x_ptr,
    up_w_ptr,
    up_b_ptr,
    z_ptr,
    a_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
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

    for k0 in tl.range(0, D, BLOCK_K):
        k = k0 + offs_k

        x_vals = tl.load(
            x_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < D),
            other=0.0,
        )

        w_vals = tl.load(
            up_w_ptr + offs_h[None, :] * D + k[:, None],
            mask=(offs_h[None, :] < H) & (k[:, None] < D),
            other=0.0,
        )

        acc += tl.dot(x_vals, w_vals)

    bias = tl.load(up_b_ptr + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
    z = acc + bias[None, :]

    # quick_gelu(x) = x * sigmoid(1.702*x)
    sig = 1.0 / (1.0 + tl.exp(-1.702 * z))
    act = z * sig

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(z_ptr + offs_m[:, None] * H + offs_h[None, :], z, mask=mask)
    tl.store(a_ptr + offs_m[:, None] * H + offs_h[None, :], act, mask=mask)


@triton.jit
def _down_kernel(
    a_ptr,
    down_w_ptr,
    down_b_ptr,
    y_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for k0 in tl.range(0, H, BLOCK_K):
        k = k0 + offs_k

        a_vals = tl.load(
            a_ptr + offs_m[:, None] * H + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < H),
            other=0.0,
        )

        w_vals = tl.load(
            down_w_ptr + offs_d[None, :] * H + k[:, None],
            mask=(offs_d[None, :] < D) & (k[:, None] < H),
            other=0.0,
        )

        acc += tl.dot(a_vals, w_vals)

    bias = tl.load(down_b_ptr + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
    out = acc + bias[None, :]

    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(y_ptr + offs_m[:, None] * D + offs_d[None, :], out, mask=mask)


@triton.jit
def _grad_down_weight_kernel(
    dy_ptr,
    a_ptr,
    grad_w_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_D, BLOCK_H), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m = m0 + offs_m

        dy_vals = tl.load(
            dy_ptr + m[None, :] * D + offs_d[:, None],
            mask=(m[None, :] < M) & (offs_d[:, None] < D),
            other=0.0,
        )

        a_vals = tl.load(
            a_ptr + m[:, None] * H + offs_h[None, :],
            mask=(m[:, None] < M) & (offs_h[None, :] < H),
            other=0.0,
        )

        acc += tl.dot(dy_vals, a_vals)

    mask = (offs_d[:, None] < D) & (offs_h[None, :] < H)
    tl.store(grad_w_ptr + offs_d[:, None] * H + offs_h[None, :], acc, mask=mask)


@triton.jit
def _grad_gz_kernel(
    dy_ptr,
    down_w_ptr,
    z_ptr,
    gz_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    grad_a = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k0 in tl.range(0, D, BLOCK_K):
        k = k0 + offs_k

        dy_vals = tl.load(
            dy_ptr + offs_m[:, None] * D + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < D),
            other=0.0,
        )

        w_vals = tl.load(
            down_w_ptr + k[:, None] * H + offs_h[None, :],
            mask=(k[:, None] < D) & (offs_h[None, :] < H),
            other=0.0,
        )

        grad_a += tl.dot(dy_vals, w_vals)

    z = tl.load(
        z_ptr + offs_m[:, None] * H + offs_h[None, :],
        mask=(offs_m[:, None] < M) & (offs_h[None, :] < H),
        other=0.0,
    ).to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-1.702 * z))
    # d/dx quick_gelu(x) = sigmoid(1.702x) + 1.702*x*sigmoid(1.702x)*(1-sigmoid(1.702x))
    deriv = sig + 1.702 * z * sig * (1.0 - sig)
    gz = grad_a * deriv

    mask = (offs_m[:, None] < M) & (offs_h[None, :] < H)
    tl.store(gz_ptr + offs_m[:, None] * H + offs_h[None, :], gz, mask=mask)


@triton.jit
def _grad_up_weight_kernel(
    gz_ptr,
    x_ptr,
    grad_w_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m = m0 + offs_m

        gz_vals = tl.load(
            gz_ptr + m[None, :] * H + offs_h[:, None],
            mask=(m[None, :] < M) & (offs_h[:, None] < H),
            other=0.0,
        )

        x_vals = tl.load(
            x_ptr + m[:, None] * D + offs_d[None, :],
            mask=(m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0,
        )

        acc += tl.dot(gz_vals, x_vals)

    mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)
    tl.store(grad_w_ptr + offs_h[:, None] * D + offs_d[None, :], acc, mask=mask)


@triton.jit
def _grad_x_kernel(
    gz_ptr,
    up_w_ptr,
    grad_x_ptr,
    M: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for k0 in tl.range(0, H, BLOCK_K):
        k = k0 + offs_k

        gz_vals = tl.load(
            gz_ptr + offs_m[:, None] * H + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < H),
            other=0.0,
        )

        w_vals = tl.load(
            up_w_ptr + k[:, None] * D + offs_d[None, :],
            mask=(k[:, None] < H) & (offs_d[None, :] < D),
            other=0.0,
        )

        acc += tl.dot(gz_vals, w_vals)

    mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    tl.store(grad_x_ptr + offs_m[:, None] * D + offs_d[None, :], acc, mask=mask)


@triton.jit
def _reduce_bias_kernel(
    src_ptr,
    bias_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_m = tl.arange(0, BLOCK_M)

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        m = m0 + offs_m
        vals = tl.load(
            src_ptr + m[:, None] * C + offs_c[None, :],
            mask=(m[:, None] < M) & (offs_c[None, :] < C),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    tl.store(bias_ptr + offs_c, acc, mask=offs_c < C)


@triton.jit
def _copy_to_flat_kernel(
    src_ptr,
    out_ptr,
    out_offset: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + out_offset + offs, vals, mask=mask)


def _forward_impl(x, up_weight, up_bias, down_weight, down_bias):
    assert x.is_cuda
    assert up_weight.is_cuda and up_bias.is_cuda
    assert down_weight.is_cuda and down_bias.is_cuda
    assert x.is_contiguous()
    assert up_weight.is_contiguous()
    assert up_bias.is_contiguous()
    assert down_weight.is_contiguous()
    assert down_bias.is_contiguous()

    D = x.shape[-1]
    M = x.numel() // D
    H = up_bias.numel()

    z = torch.empty((M, H), device=x.device, dtype=torch.float32)
    a = torch.empty((M, H), device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)

    bm = 16
    bh = 32
    bk = 32
    grid_up = (triton.cdiv(M, bm), triton.cdiv(H, bh))
    _up_act_kernel[grid_up](
        x,
        up_weight,
        up_bias,
        z,
        a,
        M,
        D,
        H,
        BLOCK_M=bm,
        BLOCK_H=bh,
        BLOCK_K=bk,
        num_warps=4,
    )

    bd = 32
    grid_down = (triton.cdiv(M, bm), triton.cdiv(D, bd))
    _down_kernel[grid_down](
        a,
        down_weight,
        down_bias,
        y,
        M,
        D,
        H,
        BLOCK_M=bm,
        BLOCK_D=bd,
        BLOCK_K=bk,
        num_warps=4,
    )

    return y, z, a


def qwen2_vl_2b_vision_mlp_forward_with_saved(
    x, up_weight, up_bias, down_weight, down_bias
):
    """
    Forward pass for Qwen2-VL vision MLP.

    Returns:
        output tensor and saved intermediates for the Triton backward pass.
    """
    y, z, a = _forward_impl(x, up_weight, up_bias, down_weight, down_bias)
    saved = (x, up_weight, up_bias, down_weight, down_bias, z, a)
    return y, saved


def qwen2_vl_2b_vision_mlp_backward_from_saved(upstream, saved):
    """
    Backward pass from saved forward intermediates.

    All gradient math is performed by Triton kernels.
    """
    if isinstance(upstream, (tuple, list)):
        dy = upstream[0]
    else:
        dy = upstream

    x, up_weight, up_bias, down_weight, down_bias, z, a = saved

    D = x.shape[-1]
    M = x.numel() // D
    H = up_bias.numel()

    grad_x = torch.empty_like(x)
    grad_up_weight = torch.empty_like(up_weight)
    grad_up_bias = torch.empty_like(up_bias)
    grad_down_weight = torch.empty_like(down_weight)
    grad_down_bias = torch.empty_like(down_bias)
    gz = torch.empty((M, H), device=x.device, dtype=x.dtype)

    # grad_down_weight = dy.T @ a
    _grad_down_weight_kernel[(triton.cdiv(D, 16), triton.cdiv(H, 16))](
        dy,
        a,
        grad_down_weight,
        M,
        D,
        H,
        BLOCK_D=16,
        BLOCK_H=16,
        BLOCK_M=32,
        num_warps=4,
    )

    # grad_down_bias = sum_m dy[m, :]
    _reduce_bias_kernel[(triton.cdiv(D, 32),)](
        dy,
        grad_down_bias,
        M,
        D,
        BLOCK_M=128,
        BLOCK_C=32,
        num_warps=4,
    )

    # gz = (dy @ down_weight) * quick_gelu'(z)
    _grad_gz_kernel[(triton.cdiv(M, 16), triton.cdiv(H, 32))](
        dy,
        down_weight,
        z,
        gz,
        M,
        D,
        H,
        BLOCK_M=16,
        BLOCK_H=32,
        BLOCK_K=32,
        num_warps=4,
    )

    # grad_up_weight = gz.T @ x
    _grad_up_weight_kernel[(triton.cdiv(H, 16), triton.cdiv(D, 16))](
        gz,
        x,
        grad_up_weight,
        M,
        D,
        H,
        BLOCK_H=16,
        BLOCK_D=16,
        BLOCK_M=32,
        num_warps=4,
    )

    # grad_up_bias = sum_m gz[m, :]
    _reduce_bias_kernel[(triton.cdiv(H, 32),)](
        gz,
        grad_up_bias,
        M,
        H,
        BLOCK_M=128,
        BLOCK_C=32,
        num_warps=4,
    )

    # grad_x = gz @ up_weight
    _grad_x_kernel[(triton.cdiv(M, 16), triton.cdiv(D, 32))](
        gz,
        up_weight,
        grad_x,
        M,
        D,
        H,
        BLOCK_M=16,
        BLOCK_D=32,
        BLOCK_K=32,
        num_warps=4,
    )

    return grad_x, grad_up_weight, grad_up_bias, grad_down_weight, grad_down_bias


def kernel_function(x, up_weight, up_bias, down_weight, down_bias, *upstream):
    """
    Complete forward+backward step for the Qwen2-VL vision MLP.

    This wrapper performs only validation/allocation/grid orchestration. The
    numerical work is executed in Triton kernels. The returned tensor matches the
    evaluator convention:

        torch.cat([output.float().reshape(-1),
                   grad_x.float().reshape(-1),
                   grad_up_weight.float().reshape(-1),
                   grad_up_bias.float().reshape(-1),
                   grad_down_weight.float().reshape(-1),
                   grad_down_bias.float().reshape(-1)])

    No PyTorch math or torch.cat is used; flattening/packing is done by Triton.
    """
    assert len(upstream) >= 1
    dy = upstream[0]

    y, saved = qwen2_vl_2b_vision_mlp_forward_with_saved(
        x, up_weight, up_bias, down_weight, down_bias
    )
    grad_x, grad_up_weight, grad_up_bias, grad_down_weight, grad_down_bias = (
        qwen2_vl_2b_vision_mlp_backward_from_saved(dy, saved)
    )

    n_y = y.numel()
    n_gx = grad_x.numel()
    n_guw = grad_up_weight.numel()
    n_gub = grad_up_bias.numel()
    n_gdw = grad_down_weight.numel()
    n_gdb = grad_down_bias.numel()

    total = n_y + n_gx + n_guw + n_gub + n_gdw + n_gdb
    out = torch.empty((total,), device=x.device, dtype=torch.float32)

    block = 1024

    off = 0
    _copy_to_flat_kernel[(triton.cdiv(n_y, block),)](
        y, out, off, n_y, BLOCK=block, num_warps=4
    )

    off += n_y
    _copy_to_flat_kernel[(triton.cdiv(n_gx, block),)](
        grad_x, out, off, n_gx, BLOCK=block, num_warps=4
    )

    off += n_gx
    _copy_to_flat_kernel[(triton.cdiv(n_guw, block),)](
        grad_up_weight, out, off, n_guw, BLOCK=block, num_warps=4
    )

    off += n_guw
    _copy_to_flat_kernel[(triton.cdiv(n_gub, block),)](
        grad_up_bias, out, off, n_gub, BLOCK=block, num_warps=4
    )

    off += n_gub
    _copy_to_flat_kernel[(triton.cdiv(n_gdw, block),)](
        grad_down_weight, out, off, n_gdw, BLOCK=block, num_warps=4
    )

    off += n_gdw
    _copy_to_flat_kernel[(triton.cdiv(n_gdb, block),)](
        grad_down_bias, out, off, n_gdb, BLOCK=block, num_warps=4
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen2_vl_2b_vision_mlp', 'arguments': ['x', 'up_weight', 'up_bias', 'down_weight', 'down_bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dup_weight', 'dup_bias', 'ddown_weight', 'ddown_bias'], 'gradient_return_order': ['dx', 'dup_weight', 'dup_bias', 'ddown_weight', 'ddown_bias'], 'scalars': [], 'deployment_entry': 'qwen2_vl_2b_vision_mlp_deployment', 'allowed_primitives': ['vendor_gemm']}


class Qwen2Vl2BVisionMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen2_vl_2b_vision_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, up_weight, up_bias, down_weight, down_bias):
        out, saved = qwen2_vl_2b_vision_mlp_forward_with_saved(x, up_weight, up_bias, down_weight, down_bias)
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
        dx, dup_weight, dup_bias, ddown_weight, ddown_bias = qwen2_vl_2b_vision_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dup_weight, dup_bias, ddown_weight, ddown_bias


def qwen2_vl_2b_vision_mlp_deployment(x, up_weight, up_bias, down_weight, down_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        up_weight = up_weight.reshape(-1, up_weight.shape[-1])
        down_weight = down_weight.reshape(-1, down_weight.shape[-1])
    out = Qwen2Vl2BVisionMlpFunction.apply(x, up_weight, up_bias, down_weight, down_bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Qwen2Vl2BVisionMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, up_weight, up_bias, down_weight, down_bias):
        super().__init__()
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.up_bias = (up_bias if isinstance(up_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(up_bias))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))
        self.down_bias = (down_bias if isinstance(down_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(down_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return qwen2_vl_2b_vision_mlp_deployment(x, self.up_weight, self.up_bias, self.down_weight, self.down_bias)


DEPLOYMENT_ENTRY = "qwen2_vl_2b_vision_mlp_deployment"
