import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _gate_pack_forward_kernel_fn(
    attn,
    gate,
    sigmoid_gate,
    x,
    total: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    QO: tl.constexpr,
    attn_sb: tl.constexpr,
    attn_sh: tl.constexpr,
    attn_st: tl.constexpr,
    attn_sd: tl.constexpr,
    gate_sb: tl.constexpr,
    gate_st: tl.constexpr,
    gate_sf: tl.constexpr,
    sig_sb: tl.constexpr,
    sig_st: tl.constexpr,
    sig_sf: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    feature = offs % QO
    bt = offs // QO
    t = bt % T
    b = bt // T
    h = feature // D
    d = feature - h * D

    a = tl.load(attn + b * attn_sb + h * attn_sh + t * attn_st + d * attn_sd, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(gate + b * gate_sb + t * gate_st + feature * gate_sf, mask=mask, other=0.0).to(tl.float32)

    s = 1.0 / (1.0 + tl.exp(0.0 - gv))
    s_b = s.to(tl.bfloat16).to(tl.float32)
    xv = a * s_b

    tl.store(sigmoid_gate + b * sig_sb + t * sig_st + feature * sig_sf, s_b, mask=mask)
    tl.store(x + offs, xv, mask=mask)


_gate_pack_forward_kernel = triton.jit(_gate_pack_forward_kernel_fn)


def _gate_backward_materialize_kernel_fn(
    dag,
    attn,
    sigmoid_gate,
    x,
    dattn,
    dgate,
    total: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    QO: tl.constexpr,
    attn_sb: tl.constexpr,
    attn_sh: tl.constexpr,
    attn_st: tl.constexpr,
    attn_sd: tl.constexpr,
    sig_sb: tl.constexpr,
    sig_st: tl.constexpr,
    sig_sf: tl.constexpr,
    dattn_sb: tl.constexpr,
    dattn_sh: tl.constexpr,
    dattn_st: tl.constexpr,
    dattn_sd: tl.constexpr,
    dgate_sb: tl.constexpr,
    dgate_st: tl.constexpr,
    dgate_sf: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    feature = offs % QO
    bt = offs // QO
    t = bt % T
    b = bt // T
    h = feature // D
    d = feature - h * D

    g = tl.load(dag + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(attn + b * attn_sb + h * attn_sh + t * attn_st + d * attn_sd, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(sigmoid_gate + b * sig_sb + t * sig_st + feature * sig_sf, mask=mask, other=0.0).to(tl.float32)

    xv = a * s
    dav = g * s
    dgv = g * a * s * (1.0 - s)

    tl.store(x + offs, xv, mask=mask)
    tl.store(dattn + b * dattn_sb + h * dattn_sh + t * dattn_st + d * dattn_sd, dav, mask=mask)
    tl.store(dgate + b * dgate_sb + t * dgate_st + feature * dgate_sf, dgv, mask=mask)


_gate_backward_materialize_kernel = triton.jit(_gate_backward_materialize_kernel_fn)


def _check_forward_inputs(q, k, v, gate, o_weight):
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k and v must be rank 4")
    if gate.dim() != 3 or o_weight.dim() != 2:
        raise ValueError("gate must be rank 3 and o_weight rank 2")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not gate.is_cuda or not o_weight.is_cuda:
        raise ValueError("all tensors must be CUDA tensors")
    if q.dtype != torch.bfloat16:
        raise ValueError("only bfloat16 q is supported")
    if k.dtype != q.dtype or v.dtype != q.dtype or gate.dtype != q.dtype or o_weight.dtype != q.dtype:
        raise ValueError("all tensors must have the same bfloat16 dtype")
    if q.device != k.device or q.device != v.device or q.device != gate.device or q.device != o_weight.device:
        raise ValueError("all tensors must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k and v must have contiguous last dimension")

    B = q.shape[0]
    HQ = q.shape[1]
    T = q.shape[2]
    D = q.shape[3]
    HK = k.shape[1]
    QO = HQ * D

    if k.shape[0] != B or v.shape[0] != B:
        raise ValueError("batch dimensions must match")
    if k.shape[2] != T or v.shape[2] != T:
        raise ValueError("token dimensions must match")
    if k.shape[3] != D or v.shape[3] != D:
        raise ValueError("head dimensions must match")
    if v.shape[1] != HK:
        raise ValueError("k and v head counts must match")
    if HK <= 0 or HQ % HK != 0:
        raise ValueError("query heads must be divisible by key value heads")
    if gate.shape[0] != B or gate.shape[1] != T or gate.shape[2] != QO:
        raise ValueError("gate shape mismatch")
    if o_weight.shape[1] != QO:
        raise ValueError("o_weight shape mismatch")
    if D <= 0 or T <= 0 or B <= 0 or HQ <= 0:
        raise ValueError("invalid empty dimension")
    return B, HQ, HK, T, D, o_weight.shape[0]


def qwen3_5_gated_attention_forward_with_saved(q, k, v, gate, o_weight):
    B, HQ, HK, T, D, HOUT = _check_forward_inputs(q, k, v, gate, o_weight)
    QO = HQ * D
    BT = B * T
    scale = float(D) ** -0.5

    prev_det = torch.are_deterministic_algorithms_enabled()
    if not prev_det:
        torch.use_deterministic_algorithms(True)
    try:
        attn_out, logsumexp = vendor_sdpa_forward(q, k, v, is_causal=bool(T > 1), scale=scale)
    finally:
        if not prev_det:
            torch.use_deterministic_algorithms(False)

    sigmoid_gate = torch.empty((B, T, QO), device=gate.device, dtype=gate.dtype)
    x = torch.empty((BT, QO), device=q.device, dtype=q.dtype)

    total = B * T * QO
    block = 256
    grid = (triton.cdiv(total, block),)
    _gate_pack_forward_kernel[grid](
        attn_out,
        gate,
        sigmoid_gate,
        x,
        total,
        T,
        D,
        QO,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        attn_out.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        sigmoid_gate.stride(0),
        sigmoid_gate.stride(1),
        sigmoid_gate.stride(2),
        BLOCK=block,
        num_warps=4,
    )

    y2 = vendor_gemm(x, o_weight.t())
    y = y2.view(B, T, HOUT)

    saved_tensors = (q, k, v, o_weight, attn_out, logsumexp, sigmoid_gate, B, HQ, HK, T, D, HOUT)
    return y, saved_tensors


def qwen3_5_gated_attention_backward_from_saved(dy, saved_tensors):
    q, k, v, o_weight, attn_out, logsumexp, sigmoid_gate, B, HQ, HK, T, D, HOUT = saved_tensors
    QO = HQ * D
    BT = B * T

    if dy.dim() != 3:
        raise ValueError("dy must be rank 3")
    if not dy.is_cuda:
        raise ValueError("dy must be a CUDA tensor")
    if dy.dtype != q.dtype:
        raise ValueError("dy dtype mismatch")
    if dy.device != q.device:
        raise ValueError("dy device mismatch")
    if dy.shape[0] != B or dy.shape[1] != T or dy.shape[2] != HOUT:
        raise ValueError("dy shape mismatch")
    if not dy.is_contiguous():
        raise ValueError("dy must be contiguous")

    dy2 = dy.view(BT, HOUT)
    dag2 = vendor_gemm(dy2, o_weight)

    x_recomputed = torch.empty((BT, QO), device=q.device, dtype=q.dtype)
    dattn = torch.empty_strided(attn_out.shape, attn_out.stride(), device=q.device, dtype=q.dtype)
    dgate = torch.empty((B, T, QO), device=q.device, dtype=q.dtype)

    total = B * T * QO
    block = 256
    grid = (triton.cdiv(total, block),)
    _gate_backward_materialize_kernel[grid](
        dag2,
        attn_out,
        sigmoid_gate,
        x_recomputed,
        dattn,
        dgate,
        total,
        T,
        D,
        QO,
        attn_out.stride(0),
        attn_out.stride(1),
        attn_out.stride(2),
        attn_out.stride(3),
        sigmoid_gate.stride(0),
        sigmoid_gate.stride(1),
        sigmoid_gate.stride(2),
        dattn.stride(0),
        dattn.stride(1),
        dattn.stride(2),
        dattn.stride(3),
        dgate.stride(0),
        dgate.stride(1),
        dgate.stride(2),
        BLOCK=block,
        num_warps=4,
    )

    do_weight = vendor_gemm(dy2.t(), x_recomputed)

    scale = float(D) ** -0.5
    prev_det = torch.are_deterministic_algorithms_enabled()
    if not prev_det:
        torch.use_deterministic_algorithms(True)
    try:
        dq, dk, dv = vendor_sdpa_backward(
            dattn,
            q,
            k,
            v,
            attn_out,
            logsumexp,
            is_causal=bool(T > 1),
            scale=scale,
        )
    finally:
        if not prev_det:
            torch.use_deterministic_algorithms(False)

    return dq, dk, dv, dgate, do_weight

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


# ── trusted primitive: vendor_sdpa ───────────────────────────────────────────
# Fixed, outside the evolve block, and checked byte-for-byte against
# evograd.pipelines.shared.primitives. Evolution may call it; it may not
# change it. The fused-attention kernel eager PyTorch itself runs, chosen the
# way its dispatcher chooses it: cuDNN in a training configuration, flash
# under torch.use_deterministic_algorithms (cuDNN's backward is not
# reproducible, so the deterministic dispatcher excludes it). Verification
# runs deterministic and compares flash with flash; timing runs as training
# does and compares cuDNN with cuDNN.

def _vendor_sdpa_deterministic():
    return torch.are_deterministic_algorithms_enabled()


def vendor_sdpa_forward(q, k, v, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, forward.

    ``q`` is ``[B, HQ, T, D]``; ``k`` and ``v`` are ``[B, HK, T, D]`` with
    ``HK`` dividing ``HQ`` (grouped-query attention is the kernel's own). No
    mask, no dropout. Returns ``(out, logsumexp)``; ``logsumexp`` is what the
    backward needs and is the caller's to save (its layout is the backend's;
    pass it back unchanged). ``no_grad`` because the pair differentiates itself.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("vendor_sdpa_forward takes [B, H, T, D] tensors")
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                q, k, v, 0.0, bool(is_causal), False, scale=scale)
        else:
            out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
                q, k, v, None, True, 0.0, bool(is_causal), False, scale=scale)
    return out, logsumexp


def vendor_sdpa_backward(dout, q, k, v, out, logsumexp, is_causal=True, scale=None):
    """Fused scaled-dot-product attention, backward.

    Takes the forward's inputs and both of its results, from the same mode
    (deterministic or not) as the forward; returns ``(dq, dk, dv)`` with the
    shapes of ``q``, ``k`` and ``v``.
    """
    if dout.shape != out.shape:
        raise ValueError(f"vendor_sdpa_backward: dout {tuple(dout.shape)} vs out {tuple(out.shape)}")
    # No dropout, no mask, dense [B, H, T, D] inputs: philox state, attn_bias
    # and the nested-layout cum_seq tensors stay undefined (None). A defined
    # cum_seq tensor would select the ragged layout.
    with torch.no_grad():
        if _vendor_sdpa_deterministic():
            dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, q.shape[2], k.shape[2], 0.0,
                bool(is_causal), None, None, scale=scale)
        else:
            dq, dk, dv = torch.ops.aten._scaled_dot_product_cudnn_attention_backward(
                dout, q, k, v, out, logsumexp, None, None, None, None, None,
                q.shape[2], k.shape[2], 0.0, bool(is_causal), scale=scale)
    return dq, dk, dv


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_gated_attention', 'arguments': ['q', 'k', 'v', 'gate', 'o_weight'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dq', 'dk', 'dv', 'dgate', 'do_weight'], 'gradient_return_order': ['dq', 'dk', 'dv', 'dgate', 'do_weight'], 'scalars': [], 'deployment_entry': 'qwen3_5_gated_attention_deployment', 'allowed_primitives': ['vendor_gemm', 'vendor_sdpa']}


class Qwen35GatedAttentionFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_gated_attention`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, gate, o_weight):
        y, saved = qwen3_5_gated_attention_forward_with_saved(q, k, v, gate, o_weight)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
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
        dq, dk, dv, dgate, do_weight = qwen3_5_gated_attention_backward_from_saved(
            dy, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, dgate, do_weight


def qwen3_5_gated_attention_deployment(q, k, v, gate, o_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen35GatedAttentionFunction.apply(q, k, v, gate, o_weight)


class Qwen35GatedAttentionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, o_weight):
        super().__init__()
        self.o_weight = (o_weight if isinstance(o_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(o_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v, gate):
        return qwen3_5_gated_attention_deployment(q, k, v, gate, self.o_weight)


DEPLOYMENT_ENTRY = "qwen3_5_gated_attention_deployment"
