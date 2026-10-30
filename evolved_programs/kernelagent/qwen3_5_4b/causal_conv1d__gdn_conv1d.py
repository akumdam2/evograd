"""adbench-tools artifact: task qwen3_5_4b.causal_conv1d.gdn_conv1d, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Triton implementation for:
#   forward:  depthwise causal Conv1d over x[B, D, L] with weight[D, W],
#             optionally followed by SiLU
#   backward: gradients w.r.t. x and weight from upstream dy
#
# Fused stages:
#   1. Forward kernel fuses causal depthwise convolution and optional SiLU.
#   2. Backward kernels fuse SiLU derivative with the convolution-gradient formulas.
#   3. kernel_function writes forward output, dx, and dweight directly into one
#      flat fp32 tensor, avoiding torch.cat / PyTorch compute in the wrapper.
#
# Note on fusion:
#   The pipeline is kept as three Triton launches because dx and dweight both
#   depend on the full saved pre-activation z from the forward pass, while
#   dweight is a cross-time reduction. Fully merging all stages into one program
#   would either serialize whole-channel work or require atomics/recomputation.

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    flat_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    w_s0: tl.constexpr,
    w_s1: tl.constexpr,
    y_s0: tl.constexpr,
    y_s1: tl.constexpr,
    y_s2: tl.constexpr,
    WIDTH: tl.constexpr,
    ACT_SILU: tl.constexpr,
    STORE_Y: tl.constexpr,
    STORE_FLAT: tl.constexpr,
    FLAT_BASE: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_bd = tl.program_id(1)

    b = pid_bd // D
    d = pid_bd - b * D

    offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_o = offs < L

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Causal depthwise convolution:
    # z[b,d,t] = sum_k weight[d,k] * x[b,d,t-(WIDTH-1)+k]
    for k in tl.static_range(0, WIDTH):
        xt = offs - (WIDTH - 1) + k
        mask_x = mask_o & (xt >= 0)
        xv = tl.load(
            x_ptr + b * x_s0 + d * x_s1 + xt * x_s2,
            mask=mask_x,
            other=0.0,
        ).to(tl.float32)
        wv = tl.load(w_ptr + d * w_s0 + k * w_s1).to(tl.float32)
        acc += xv * wv

    z_lin = (b * D + d) * L + offs
    tl.store(z_ptr + z_lin, acc, mask=mask_o)

    out = acc
    if ACT_SILU:
        sig = 1.0 / (1.0 + tl.exp(-acc))
        out = acc * sig

    if STORE_Y:
        tl.store(
            y_ptr + b * y_s0 + d * y_s1 + offs * y_s2,
            out,
            mask=mask_o,
        )

    if STORE_FLAT:
        # Match forward_with_saved output dtype before writing into fp32 flat buffer.
        flat_idx = FLAT_BASE + (b * D + d) * L + offs
        out_cast = out.to(y_ptr.dtype.element_ty).to(tl.float32)
        tl.store(flat_ptr + flat_idx, out_cast, mask=mask_o)


@triton.jit
def _dx_kernel(
    dy_ptr,
    w_ptr,
    z_ptr,
    dx_ptr,
    flat_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    dy_s0: tl.constexpr,
    dy_s1: tl.constexpr,
    dy_s2: tl.constexpr,
    w_s0: tl.constexpr,
    w_s1: tl.constexpr,
    dx_s0: tl.constexpr,
    dx_s1: tl.constexpr,
    dx_s2: tl.constexpr,
    WIDTH: tl.constexpr,
    ACT_SILU: tl.constexpr,
    STORE_DX: tl.constexpr,
    STORE_FLAT: tl.constexpr,
    FLAT_BASE: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_bd = tl.program_id(1)

    b = pid_bd // D
    d = pid_bd - b * D

    offs = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_xo = offs < L

    grad = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # x[b,d,s] contributes to output t = s + WIDTH - 1 - k.
    for k in tl.static_range(0, WIDTH):
        t = offs + (WIDTH - 1) - k
        mask_t = mask_xo & (t < L)

        dyv = tl.load(
            dy_ptr + b * dy_s0 + d * dy_s1 + t * dy_s2,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        z = tl.load(
            z_ptr + (b * D + d) * L + t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32)

        dz = dyv
        if ACT_SILU:
            sig = 1.0 / (1.0 + tl.exp(-z))
            dz = dyv * sig * (1.0 + z * (1.0 - sig))

        wv = tl.load(w_ptr + d * w_s0 + k * w_s1).to(tl.float32)
        grad += dz * wv

    if STORE_DX:
        tl.store(
            dx_ptr + b * dx_s0 + d * dx_s1 + offs * dx_s2,
            grad,
            mask=mask_xo,
        )

    if STORE_FLAT:
        # Match backward dx tensor dtype before writing into fp32 flat buffer.
        flat_idx = FLAT_BASE + (b * D + d) * L + offs
        grad_cast = grad.to(dx_ptr.dtype.element_ty).to(tl.float32)
        tl.store(flat_ptr + flat_idx, grad_cast, mask=mask_xo)


@triton.jit
def _dw_kernel(
    x_ptr,
    dy_ptr,
    z_ptr,
    dw_ptr,
    flat_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    dy_s0: tl.constexpr,
    dy_s1: tl.constexpr,
    dy_s2: tl.constexpr,
    dw_s0: tl.constexpr,
    dw_s1: tl.constexpr,
    WIDTH: tl.constexpr,
    ACT_SILU: tl.constexpr,
    STORE_DW: tl.constexpr,
    STORE_FLAT: tl.constexpr,
    FLAT_BASE: tl.constexpr,
    BLOCK_BL: tl.constexpr,
):
    d = tl.program_id(0)
    k = tl.program_id(1)

    offs = tl.arange(0, BLOCK_BL)
    total = B * L
    acc_vec = tl.zeros((BLOCK_BL,), dtype=tl.float32)

    # dweight[d,k] = sum_{b,t} dz[b,d,t] * x[b,d,t-(WIDTH-1)+k]
    for start in tl.range(0, total, BLOCK_BL):
        n = start + offs
        mask_n = n < total

        b = n // L
        t = n - b * L
        xt = t - (WIDTH - 1) + k
        mask = mask_n & (xt >= 0)

        xv = tl.load(
            x_ptr + b * x_s0 + d * x_s1 + xt * x_s2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        dyv = tl.load(
            dy_ptr + b * dy_s0 + d * dy_s1 + t * dy_s2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        z = tl.load(
            z_ptr + (b * D + d) * L + t,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        dz = dyv
        if ACT_SILU:
            sig = 1.0 / (1.0 + tl.exp(-z))
            dz = dyv * sig * (1.0 + z * (1.0 - sig))

        acc_vec += dz * xv

    acc = tl.sum(acc_vec, axis=0)

    if STORE_DW:
        tl.store(dw_ptr + d * dw_s0 + k * dw_s1, acc)

    if STORE_FLAT:
        # Match dweight tensor dtype before writing into fp32 flat buffer.
        acc_cast = acc.to(dw_ptr.dtype.element_ty).to(tl.float32)
        tl.store(flat_ptr + FLAT_BASE + d * WIDTH + k, acc_cast)


def _act_bool(silu):
    # Expected to be a Python bool in the task inputs.
    return bool(silu)


def causal_conv1d_forward_with_saved(x, weight, silu):
    """
    Forward causal depthwise conv1d with optional SiLU.

    Args:
        x:      [B, D, L]
        weight: [D, W]
        silu:   bool

    Returns:
        y, saved
    """
    assert x.is_cuda and weight.is_cuda
    assert x.dim() == 3
    assert weight.dim() == 2

    B, D, L = x.shape
    assert weight.shape[0] == D
    W = weight.shape[1]

    y = torch.empty_like(x)
    z = torch.empty((B, D, L), device=x.device, dtype=torch.float32)

    BLOCK_L = 128
    grid = (triton.cdiv(L, BLOCK_L), B * D)

    _forward_kernel[grid](
        x,
        weight,
        y,
        z,
        y,  # dummy flat pointer
        B,
        D,
        L,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        WIDTH=W,
        ACT_SILU=_act_bool(silu),
        STORE_Y=True,
        STORE_FLAT=False,
        FLAT_BASE=0,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )

    return y, (x, weight, z, _act_bool(silu))


def causal_conv1d_backward_from_saved(upstream, saved, *extra):
    """
    Backward pass for causal_conv1d_forward_with_saved.

    Some deployment harnesses call this as:
        backward_from_saved(upstream, saved)
    while others pass an extra output argument:
        backward_from_saved(upstream, output, saved)

    The variadic `extra` keeps the interface compatible without changing the
    actual Triton-computed backward path.
    """
    if extra:
        maybe_saved = extra[-1]
        if isinstance(maybe_saved, (tuple, list)) and len(maybe_saved) == 4:
            saved = maybe_saved

    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream
    x, weight, z, act_silu = saved

    assert dy.is_cuda and x.is_cuda and weight.is_cuda
    B, D, L = x.shape
    W = weight.shape[1]

    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)

    BLOCK_L = 128
    grid_dx = (triton.cdiv(L, BLOCK_L), B * D)

    _dx_kernel[grid_dx](
        dy,
        weight,
        z,
        dx,
        dx,  # dummy flat pointer
        B,
        D,
        L,
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        weight.stride(0),
        weight.stride(1),
        dx.stride(0),
        dx.stride(1),
        dx.stride(2),
        WIDTH=W,
        ACT_SILU=act_silu,
        STORE_DX=True,
        STORE_FLAT=False,
        FLAT_BASE=0,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )

    BLOCK_BL = 256
    grid_dw = (D, W)

    _dw_kernel[grid_dw](
        x,
        dy,
        z,
        dweight,
        dweight,  # dummy flat pointer
        B,
        D,
        L,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        dweight.stride(0),
        dweight.stride(1),
        WIDTH=W,
        ACT_SILU=act_silu,
        STORE_DW=True,
        STORE_FLAT=False,
        FLAT_BASE=0,
        BLOCK_BL=BLOCK_BL,
        num_warps=8,
    )

    return dx, dweight


def kernel_function(x, weight, silu, upstream):
    """
    Complete forward+backward step.

    Returns a single flat fp32 tensor equivalent to:

        torch.cat([
            y.float().reshape(-1),
            dx.float().reshape(-1),
            dweight.float().reshape(-1),
        ])

    but all computation and packing are performed in Triton kernels.
    """
    dy = upstream[0] if isinstance(upstream, (tuple, list)) else upstream

    assert x.is_cuda and weight.is_cuda and dy.is_cuda
    assert x.dim() == 3
    assert weight.dim() == 2

    B, D, L = x.shape
    W = weight.shape[1]
    assert weight.shape[0] == D

    n_y = B * D * L
    n_dx = n_y
    n_dw = D * W
    total = n_y + n_dx + n_dw

    out = torch.empty((total,), device=x.device, dtype=torch.float32)
    z = torch.empty((B, D, L), device=x.device, dtype=torch.float32)

    act_silu = _act_bool(silu)

    BLOCK_L = 128
    grid_fwd = (triton.cdiv(L, BLOCK_L), B * D)

    # Forward writes y directly into out[0:n_y] and saves pre-activation z.
    _forward_kernel[grid_fwd](
        x,
        weight,
        x,  # dummy y pointer; its dtype matches y dtype
        z,
        out,
        B,
        D,
        L,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        WIDTH=W,
        ACT_SILU=act_silu,
        STORE_Y=False,
        STORE_FLAT=True,
        FLAT_BASE=0,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )

    # Backward dx writes directly into out[n_y:n_y+n_dx].
    _dx_kernel[grid_fwd](
        dy,
        weight,
        z,
        x,  # dummy dx pointer; its dtype matches dx dtype
        out,
        B,
        D,
        L,
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        weight.stride(0),
        weight.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        WIDTH=W,
        ACT_SILU=act_silu,
        STORE_DX=False,
        STORE_FLAT=True,
        FLAT_BASE=n_y,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )

    # Backward dweight writes directly into out[n_y+n_dx:].
    BLOCK_BL = 256
    grid_dw = (D, W)

    _dw_kernel[grid_dw](
        x,
        dy,
        z,
        weight,  # dummy dweight pointer; its dtype matches dweight dtype
        out,
        B,
        D,
        L,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        dy.stride(0),
        dy.stride(1),
        dy.stride(2),
        weight.stride(0),
        weight.stride(1),
        WIDTH=W,
        ACT_SILU=act_silu,
        STORE_DW=False,
        STORE_FLAT=True,
        FLAT_BASE=n_y + n_dx,
        BLOCK_BL=BLOCK_BL,
        num_warps=8,
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
