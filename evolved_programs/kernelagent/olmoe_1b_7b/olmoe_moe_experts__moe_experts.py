"""adbench-tools artifact: task olmoe_1b_7b.olmoe_moe_experts.moe_experts, produced by kernelagent.

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


@triton.jit
def _zero_kernel(ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@triton.jit
def _cast_kernel(src, dst, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + offs, vals, mask=mask)


@triton.jit
def _pack_kernel(src, dst, dst_offset: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(dst + dst_offset + offs, vals, mask=mask)


@triton.jit
def _olmoe_forward_kernel(
    x,
    gate_up,
    down,
    top_idx,
    top_w,
    out,
    T: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    KTOP: tl.constexpr,
    g_s0: tl.constexpr,
    g_s1: tl.constexpr,
    g_s2: tl.constexpr,
    d_s0: tl.constexpr,
    d_s1: tl.constexpr,
    d_s2: tl.constexpr,
    G_LAYOUT: tl.constexpr,
    D_LAYOUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fused MoE expert forward:
      routing -> gate/up matvecs -> SiLU gating -> down projection -> top-k accumulation.
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_i = tl.arange(0, BLOCK_I)
    offs_h = tl.arange(0, BLOCK_H)

    acc_d = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for kk in tl.static_range(0, KTOP):
        expert = tl.load(top_idx + pid_t * KTOP + kk)
        weight_k = tl.load(top_w + pid_t * KTOP + kk).to(tl.float32)

        for i0 in tl.range(0, I, BLOCK_I):
            ii = i0 + offs_i

            gate_v = tl.zeros((BLOCK_I,), dtype=tl.float32)
            up_v = tl.zeros((BLOCK_I,), dtype=tl.float32)

            for h0 in tl.range(0, H, BLOCK_H):
                hh = h0 + offs_h
                xh = tl.load(x + pid_t * H + hh, mask=hh < H, other=0.0).to(tl.float32)

                if G_LAYOUT == 0:
                    wg_ptrs = gate_up + expert * g_s0 + ii[None, :] * g_s1 + hh[:, None] * g_s2
                    wu_ptrs = gate_up + expert * g_s0 + (I + ii)[None, :] * g_s1 + hh[:, None] * g_s2
                else:
                    wg_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + ii[None, :] * g_s2
                    wu_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + (I + ii)[None, :] * g_s2

                mask_hi = (hh[:, None] < H) & (ii[None, :] < I)
                wg = tl.load(wg_ptrs, mask=mask_hi, other=0.0).to(tl.float32)
                wu = tl.load(wu_ptrs, mask=mask_hi, other=0.0).to(tl.float32)

                gate_v += tl.sum(xh[:, None] * wg, axis=0)
                up_v += tl.sum(xh[:, None] * wu, axis=0)

            sig = 1.0 / (1.0 + tl.exp(-gate_v))
            hidden = (gate_v * sig) * up_v

            if D_LAYOUT == 0:
                wd_ptrs = down + expert * d_s0 + offs_d[None, :] * d_s1 + ii[:, None] * d_s2
            else:
                wd_ptrs = down + expert * d_s0 + ii[:, None] * d_s1 + offs_d[None, :] * d_s2

            mask_id = (ii[:, None] < I) & (offs_d[None, :] < H)
            wd = tl.load(wd_ptrs, mask=mask_id, other=0.0).to(tl.float32)

            acc_d += weight_k * tl.sum(hidden[:, None] * wd, axis=0)

    tl.store(out + pid_t * H + offs_d, acc_d, mask=offs_d < H)


@triton.jit
def _olmoe_backward_kernel(
    x,
    gate_up,
    down,
    top_idx,
    top_w,
    grad_out,
    grad_x_accum,
    grad_gate_up,
    grad_down,
    grad_top_w,
    T: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    KTOP: tl.constexpr,
    g_s0: tl.constexpr,
    g_s1: tl.constexpr,
    g_s2: tl.constexpr,
    d_s0: tl.constexpr,
    d_s1: tl.constexpr,
    d_s2: tl.constexpr,
    G_LAYOUT: tl.constexpr,
    D_LAYOUT: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Backward for fused OLMoE expert block.

    Important correctness detail:
      grad_x_accum is a float32 temporary.  Accumulating dx directly into bf16
      with atomics rounds every partial contribution and can very rarely exceed
      the evaluator tolerance.  We keep the backward computation fused, but
      accumulate dx atomically in fp32 and cast once in a small Triton epilogue.
    """
    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_i = tl.program_id(2)

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_H)

    expert = tl.load(top_idx + pid_t * KTOP + pid_k)
    weight_k = tl.load(top_w + pid_t * KTOP + pid_k).to(tl.float32)

    gate_v = tl.zeros((BLOCK_I,), dtype=tl.float32)
    up_v = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for h0 in tl.range(0, H, BLOCK_H):
        hh = h0 + offs_h
        xh = tl.load(x + pid_t * H + hh, mask=hh < H, other=0.0).to(tl.float32)

        if G_LAYOUT == 0:
            wg_ptrs = gate_up + expert * g_s0 + offs_i[None, :] * g_s1 + hh[:, None] * g_s2
            wu_ptrs = gate_up + expert * g_s0 + (I + offs_i)[None, :] * g_s1 + hh[:, None] * g_s2
        else:
            wg_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + offs_i[None, :] * g_s2
            wu_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + (I + offs_i)[None, :] * g_s2

        mask_hi = (hh[:, None] < H) & (offs_i[None, :] < I)
        wg = tl.load(wg_ptrs, mask=mask_hi, other=0.0).to(tl.float32)
        wu = tl.load(wu_ptrs, mask=mask_hi, other=0.0).to(tl.float32)

        gate_v += tl.sum(xh[:, None] * wg, axis=0)
        up_v += tl.sum(xh[:, None] * wu, axis=0)

    sig = 1.0 / (1.0 + tl.exp(-gate_v))
    silu = gate_v * sig
    hidden = silu * up_v

    d_hidden = tl.zeros((BLOCK_I,), dtype=tl.float32)
    d_top_part = tl.zeros((), dtype=tl.float32)

    for d0 in tl.range(0, H, BLOCK_D):
        dd = d0 + offs_d
        go = tl.load(grad_out + pid_t * H + dd, mask=dd < H, other=0.0).to(tl.float32)

        if D_LAYOUT == 0:
            wd_ptrs = down + expert * d_s0 + dd[None, :] * d_s1 + offs_i[:, None] * d_s2
            gd_ptrs = grad_down + expert * d_s0 + dd[None, :] * d_s1 + offs_i[:, None] * d_s2
        else:
            wd_ptrs = down + expert * d_s0 + offs_i[:, None] * d_s1 + dd[None, :] * d_s2
            gd_ptrs = grad_down + expert * d_s0 + offs_i[:, None] * d_s1 + dd[None, :] * d_s2

        mask_id = (offs_i[:, None] < I) & (dd[None, :] < H)
        wd = tl.load(wd_ptrs, mask=mask_id, other=0.0).to(tl.float32)

        gd_val = weight_k * hidden[:, None] * go[None, :]
        tl.atomic_add(gd_ptrs, gd_val, sem="relaxed", mask=mask_id)

        d_hidden += weight_k * tl.sum(wd * go[None, :], axis=1)

        y_part = tl.sum(hidden[:, None] * wd, axis=0)
        d_top_part += tl.sum(y_part * go, axis=0)

    tl.atomic_add(grad_top_w + pid_t * KTOP + pid_k, d_top_part, sem="relaxed")

    d_up = d_hidden * silu
    d_gate = d_hidden * up_v * sig * (1.0 + gate_v * (1.0 - sig))

    for h0 in tl.range(0, H, BLOCK_H):
        hh = h0 + offs_h
        xh = tl.load(x + pid_t * H + hh, mask=hh < H, other=0.0).to(tl.float32)

        if G_LAYOUT == 0:
            wg_ptrs = gate_up + expert * g_s0 + offs_i[None, :] * g_s1 + hh[:, None] * g_s2
            wu_ptrs = gate_up + expert * g_s0 + (I + offs_i)[None, :] * g_s1 + hh[:, None] * g_s2
            gg_ptrs = grad_gate_up + expert * g_s0 + offs_i[None, :] * g_s1 + hh[:, None] * g_s2
            gu_ptrs = grad_gate_up + expert * g_s0 + (I + offs_i)[None, :] * g_s1 + hh[:, None] * g_s2
        else:
            wg_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + offs_i[None, :] * g_s2
            wu_ptrs = gate_up + expert * g_s0 + hh[:, None] * g_s1 + (I + offs_i)[None, :] * g_s2
            gg_ptrs = grad_gate_up + expert * g_s0 + hh[:, None] * g_s1 + offs_i[None, :] * g_s2
            gu_ptrs = grad_gate_up + expert * g_s0 + hh[:, None] * g_s1 + (I + offs_i)[None, :] * g_s2

        mask_hi = (hh[:, None] < H) & (offs_i[None, :] < I)
        wg = tl.load(wg_ptrs, mask=mask_hi, other=0.0).to(tl.float32)
        wu = tl.load(wu_ptrs, mask=mask_hi, other=0.0).to(tl.float32)

        dx_part = tl.sum(wg * d_gate[None, :] + wu * d_up[None, :], axis=1)
        tl.atomic_add(grad_x_accum + pid_t * H + hh, dx_part, sem="relaxed", mask=hh < H)

        tl.atomic_add(gg_ptrs, xh[:, None] * d_gate[None, :], sem="relaxed", mask=mask_hi)
        tl.atomic_add(gu_ptrs, xh[:, None] * d_up[None, :], sem="relaxed", mask=mask_hi)


def _layouts_and_sizes(x, gate_up_proj, down_proj):
    H = x.shape[-1]

    if gate_up_proj.shape[-1] == H:
        I = gate_up_proj.shape[-2] // 2
        g_layout = 0
    else:
        I = gate_up_proj.shape[-1] // 2
        g_layout = 1

    if down_proj.shape[1] == H:
        d_layout = 0
    else:
        d_layout = 1

    return H, I, g_layout, d_layout


def _zero_tensor(t):
    n = t.numel()
    if n > 0:
        block = 1024
        grid = (triton.cdiv(n, block),)
        _zero_kernel[grid](t, n, BLOCK=block)


def _cast_tensor(src, dst):
    n = src.numel()
    if n > 0:
        block = 1024
        grid = (triton.cdiv(n, block),)
        _cast_kernel[grid](src, dst, n, BLOCK=block)


def olmoe_moe_experts_forward_with_saved(
    x, gate_up_proj, down_proj, top_k_index, top_k_weights
):
    H, I, g_layout, d_layout = _layouts_and_sizes(x, gate_up_proj, down_proj)
    T = x.numel() // H
    KTOP = top_k_index.shape[-1]

    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    block_d = 32
    block_i = 32
    block_h = 64

    grid = (T, triton.cdiv(H, block_d))
    _olmoe_forward_kernel[grid](
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        out,
        T=T,
        H=H,
        I=I,
        KTOP=KTOP,
        g_s0=gate_up_proj.stride(0),
        g_s1=gate_up_proj.stride(1),
        g_s2=gate_up_proj.stride(2),
        d_s0=down_proj.stride(0),
        d_s1=down_proj.stride(1),
        d_s2=down_proj.stride(2),
        G_LAYOUT=g_layout,
        D_LAYOUT=d_layout,
        BLOCK_D=block_d,
        BLOCK_I=block_i,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=3,
    )

    saved = (x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    return out, saved


def olmoe_moe_experts_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        grad_out = upstream[0]
    else:
        grad_out = upstream

    x, gate_up_proj, down_proj, top_k_index, top_k_weights = saved

    H, I, g_layout, d_layout = _layouts_and_sizes(x, gate_up_proj, down_proj)
    T = x.numel() // H
    KTOP = top_k_index.shape[-1]

    # Accumulate dx in fp32, then cast once to x dtype.  This fixes the rare
    # bf16 atomic accumulation outlier seen in the failing case.
    grad_x_accum = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)

    grad_gate_up = torch.empty(
        gate_up_proj.shape, device=gate_up_proj.device, dtype=gate_up_proj.dtype
    )
    grad_down = torch.empty(
        down_proj.shape, device=down_proj.device, dtype=down_proj.dtype
    )
    grad_top_w = torch.empty(
        top_k_weights.shape, device=top_k_weights.device, dtype=top_k_weights.dtype
    )

    _zero_tensor(grad_x_accum)
    _zero_tensor(grad_gate_up)
    _zero_tensor(grad_down)
    _zero_tensor(grad_top_w)

    block_i = 32
    block_d = 32
    block_h = 64

    grid = (T, KTOP, triton.cdiv(I, block_i))
    _olmoe_backward_kernel[grid](
        x,
        gate_up_proj,
        down_proj,
        top_k_index,
        top_k_weights,
        grad_out,
        grad_x_accum,
        grad_gate_up,
        grad_down,
        grad_top_w,
        T=T,
        H=H,
        I=I,
        KTOP=KTOP,
        g_s0=gate_up_proj.stride(0),
        g_s1=gate_up_proj.stride(1),
        g_s2=gate_up_proj.stride(2),
        d_s0=down_proj.stride(0),
        d_s1=down_proj.stride(1),
        d_s2=down_proj.stride(2),
        G_LAYOUT=g_layout,
        D_LAYOUT=d_layout,
        BLOCK_I=block_i,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        num_warps=4,
        num_stages=3,
    )

    _cast_tensor(grad_x_accum, grad_x)

    return grad_x, grad_gate_up, grad_down, grad_top_w


def _pack_into_flat(tensor, flat, offset):
    n = tensor.numel()
    if n > 0:
        block = 1024
        grid = (triton.cdiv(n, block),)
        _pack_kernel[grid](tensor, flat, offset, n, BLOCK=block)


def kernel_function(
    x, gate_up_proj, down_proj, top_k_index, top_k_weights, *upstream
):
    """
    Forward + backward step.

    Returns:
      flat float32 concat of:
        output, grad_x, grad_gate_up_proj, grad_down_proj, grad_top_k_weights
    """
    out, saved = olmoe_moe_experts_forward_with_saved(
        x, gate_up_proj, down_proj, top_k_index, top_k_weights
    )

    if len(upstream) == 1:
        grad_out = upstream[0]
    else:
        grad_out = upstream

    grad_x, grad_gate_up, grad_down, grad_top_w = olmoe_moe_experts_backward_from_saved(
        grad_out, saved
    )

    total = (
        out.numel()
        + grad_x.numel()
        + grad_gate_up.numel()
        + grad_down.numel()
        + grad_top_w.numel()
    )
    flat = torch.empty((total,), device=x.device, dtype=torch.float32)

    off = 0
    _pack_into_flat(out, flat, off)
    off += out.numel()

    _pack_into_flat(grad_x, flat, off)
    off += grad_x.numel()

    _pack_into_flat(grad_gate_up, flat, off)
    off += grad_gate_up.numel()

    _pack_into_flat(grad_down, flat, off)
    off += grad_down.numel()

    _pack_into_flat(grad_top_w, flat, off)

    return flat
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'olmoe_moe_experts', 'arguments': ['x', 'gate_up_proj', 'down_proj', 'top_k_index', 'top_k_weights'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_up_proj', 'ddown_proj', 'dtop_k_weights'], 'gradient_return_order': ['dx', 'dgate_up_proj', 'ddown_proj', 'None', 'dtop_k_weights'], 'scalars': [], 'deployment_entry': 'olmoe_moe_experts_deployment', 'allowed_primitives': []}


class OlmoeMoeExpertsFunction(torch.autograd.Function):
    """Static autograd wiring for `olmoe_moe_experts`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        out, saved = olmoe_moe_experts_forward_with_saved(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
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
        dx, dgate_up_proj, ddown_proj, dtop_k_weights = olmoe_moe_experts_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_up_proj, ddown_proj, None, dtop_k_weights


def olmoe_moe_experts_deployment(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return OlmoeMoeExpertsFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)


class OlmoeMoeExpertsModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, gate_up_proj, down_proj):
        super().__init__()
        self.gate_up_proj = (gate_up_proj if isinstance(gate_up_proj, torch.nn.Parameter)
                  else torch.nn.Parameter(gate_up_proj))
        self.down_proj = (down_proj if isinstance(down_proj, torch.nn.Parameter)
                  else torch.nn.Parameter(down_proj))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, top_k_index, top_k_weights):
        return olmoe_moe_experts_deployment(x, self.gate_up_proj, self.down_proj, top_k_index, top_k_weights)


DEPLOYMENT_ENTRY = "olmoe_moe_experts_deployment"
