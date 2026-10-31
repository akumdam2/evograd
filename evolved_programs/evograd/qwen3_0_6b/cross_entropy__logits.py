import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START

def _ceil_div_int(a, b):
    return (a + b - 1) // b


def _next_power_of_2_int(x):
    p = 1
    while p < x:
        p = p * 2
    return p


def _ce_fwd_partial_kernel_fn(logits, block_max, block_sum, C: tl.constexpr, stride_m: tl.constexpr, stride_c: tl.constexpr, NB: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    bid = tl.program_id(1)
    offs = bid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < C
    vals = tl.load(logits + row * stride_m + offs * stride_c, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(vals, axis=0)
    shifted = tl.exp(vals - m)
    s = tl.sum(tl.where(mask, shifted, 0.0), axis=0)
    tl.store(block_max + row * NB + bid, m)
    tl.store(block_sum + row * NB + bid, s)


_ce_fwd_partial_kernel = triton.jit(_ce_fwd_partial_kernel_fn)


def _ce_fwd_rows_kernel_fn(logits, target, block_max, block_sum, row_lse, row_loss, row_active, C: tl.constexpr, stride_m: tl.constexpr, stride_c: tl.constexpr, NB: tl.constexpr, BLOCK_NB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < NB
    bm = tl.load(block_max + row * NB + offs, mask=mask, other=-float("inf")).to(tl.float32)
    bs = tl.load(block_sum + row * NB + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.max(bm, axis=0)
    s = tl.sum(tl.where(mask, bs * tl.exp(bm - m), 0.0), axis=0)
    lse = m + tl.log(s)
    t = tl.load(target + row)
    active = t != -100
    safe_t = tl.where(active, t, 0)
    gold = tl.load(logits + row * stride_m + safe_t * stride_c).to(tl.float32)
    loss = lse - gold
    tl.store(row_lse + row, lse)
    tl.store(row_loss + row, tl.where(active, loss, 0.0))
    tl.store(row_active + row, tl.where(active, 1.0, 0.0))


_ce_fwd_rows_kernel = triton.jit(_ce_fwd_rows_kernel_fn)


def _ce_fwd_final_kernel_fn(row_loss, row_active, loss_out, active_count, M: tl.constexpr, BLOCK_M: tl.constexpr):
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M
    losses = tl.load(row_loss + offs, mask=mask, other=0.0).to(tl.float32)
    acts = tl.load(row_active + offs, mask=mask, other=0.0).to(tl.float32)
    total_loss = tl.sum(losses, axis=0)
    inv_count = 1.0 / tl.sum(acts, axis=0)
    tl.store(loss_out, total_loss * inv_count)
    tl.store(active_count, inv_count)


_ce_fwd_final_kernel = triton.jit(_ce_fwd_final_kernel_fn)


def _ce_bwd_kernel_fn(dloss, logits, target, row_lse, active_count, dlogits, C: tl.constexpr, stride_m: tl.constexpr, stride_c: tl.constexpr, out_stride_m: tl.constexpr, out_stride_c: tl.constexpr, BLOCK_C: tl.constexpr):
    row = tl.program_id(0)
    bid = tl.program_id(1)
    offs = bid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < C

    t = tl.load(target + row)
    active = t != -100
    x = tl.load(logits + row * stride_m + offs * stride_c, mask=mask, other=-float("inf")).to(tl.float32)
    lse = tl.load(row_lse + row).to(tl.float32)
    p = tl.exp2((x - lse) * 1.4426950408889634)
    one = offs == t
    scale = tl.load(dloss).to(tl.float32) * tl.load(active_count).to(tl.float32)
    g = scale * (p - one.to(tl.float32))
    g = tl.where(active, g, 0.0)
    tl.store(dlogits + row * out_stride_m + offs * out_stride_c, g, mask=mask)


_ce_bwd_kernel = triton.jit(_ce_bwd_kernel_fn)


def cross_entropy_forward_with_saved(logits, target):
    if not logits.is_cuda:
        raise RuntimeError("cross_entropy_forward_with_saved only supports CUDA logits")
    if not target.is_cuda:
        raise RuntimeError("cross_entropy_forward_with_saved only supports CUDA target")
    if logits.dim() != 2:
        raise RuntimeError("logits must be a 2D tensor")
    if target.dim() != 1:
        raise RuntimeError("target must be a 1D tensor")
    M = logits.shape[0]
    C = logits.shape[1]
    if target.shape[0] != M:
        raise RuntimeError("target length must match logits rows")
    if M <= 0 or C <= 0:
        raise RuntimeError("logits dimensions must be positive")
    if logits.dtype is not torch.float32:
        raise RuntimeError("only float32 logits are supported")
    if target.dtype is not torch.int64:
        raise RuntimeError("target must be int64")
    if logits.stride(1) <= 0 or logits.stride(0) <= 0:
        raise RuntimeError("logits must have positive strides")
    if target.stride(0) != 1:
        target = target.contiguous()

    BLOCK_C = 2048
    NB = _ceil_div_int(C, BLOCK_C)
    BLOCK_NB = _next_power_of_2_int(NB)
    BLOCK_M = _next_power_of_2_int(M)
    if BLOCK_NB > 2048:
        raise RuntimeError("class dimension is too large for this implementation")
    if BLOCK_M > 131072:
        raise RuntimeError("row dimension is too large for this implementation")

    block_max = torch.empty((M, NB), device=logits.device, dtype=torch.float32)
    block_sum = torch.empty((M, NB), device=logits.device, dtype=torch.float32)
    row_lse = torch.empty((M,), device=logits.device, dtype=torch.float32)
    row_loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    row_active = torch.empty((M,), device=logits.device, dtype=torch.float32)
    loss = torch.empty((), device=logits.device, dtype=logits.dtype)
    active_count = torch.empty((), device=logits.device, dtype=torch.float32)

    _ce_fwd_partial_kernel[(M, NB)](
        logits,
        block_max,
        block_sum,
        C,
        logits.stride(0),
        logits.stride(1),
        NB,
        BLOCK_C,
        num_warps=8,
    )

    _ce_fwd_rows_kernel[(M,)](
        logits,
        target,
        block_max,
        block_sum,
        row_lse,
        row_loss,
        row_active,
        C,
        logits.stride(0),
        logits.stride(1),
        NB,
        BLOCK_NB,
        num_warps=4,
    )

    _ce_fwd_final_kernel[(1,)](
        row_loss,
        row_active,
        loss,
        active_count,
        M,
        BLOCK_M,
        num_warps=8,
    )

    saved_tensors = (logits, target, row_lse, active_count, M, C)
    return loss, saved_tensors


def cross_entropy_backward_from_saved(dloss, saved_tensors):
    logits, target, row_lse, active_count, M, C = saved_tensors

    if not logits.is_cuda:
        raise RuntimeError("cross_entropy_backward_from_saved only supports CUDA logits")
    if not target.is_cuda:
        raise RuntimeError("cross_entropy_backward_from_saved only supports CUDA target")
    if not dloss.is_cuda:
        raise RuntimeError("cross_entropy_backward_from_saved only supports CUDA dloss")
    if logits.dim() != 2:
        raise RuntimeError("saved logits must be 2D")
    if target.dim() != 1:
        raise RuntimeError("saved target must be 1D")
    if row_lse.dim() != 1:
        raise RuntimeError("saved row_lse must be 1D")
    if logits.shape[0] != M or logits.shape[1] != C:
        raise RuntimeError("saved logits shape mismatch")
    if target.shape[0] != M or row_lse.shape[0] != M:
        raise RuntimeError("saved tensor shape mismatch")
    if logits.dtype is not torch.float32:
        raise RuntimeError("only float32 logits are supported")
    if dloss.dtype is not torch.float32:
        raise RuntimeError("only float32 dloss is supported")
    if target.dtype is not torch.int64:
        raise RuntimeError("target must be int64")
    if row_lse.dtype is not torch.float32 or active_count.dtype is not torch.float32:
        raise RuntimeError("saved reductions must be float32")
    if dloss.numel() != 1:
        raise RuntimeError("dloss must be scalar")
    if logits.stride(1) <= 0 or logits.stride(0) <= 0:
        raise RuntimeError("logits must have positive strides")
    if target.stride(0) != 1:
        target = target.contiguous()
    if row_lse.stride(0) != 1:
        row_lse = row_lse.contiguous()

    dlogits = torch.empty_strided(logits.shape, logits.stride(), device=logits.device, dtype=logits.dtype)

    BLOCK_C = 4096
    NB = _ceil_div_int(C, BLOCK_C)

    _ce_bwd_kernel[(M, NB)](
        dloss,
        logits,
        target,
        row_lse,
        active_count,
        dlogits,
        C,
        logits.stride(0),
        logits.stride(1),
        dlogits.stride(0),
        dlogits.stride(1),
        BLOCK_C,
        num_warps=8,
    )

    return (dlogits,)
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'cross_entropy', 'arguments': ['logits', 'target'], 'outputs': ['loss'], 'upstream_grads': ['dloss'], 'input_grads': ['dlogits'], 'gradient_return_order': ['dlogits', 'None'], 'scalars': [], 'deployment_entry': 'cross_entropy_deployment', 'allowed_primitives': []}


class CrossEntropyFunction(torch.autograd.Function):
    """Static autograd wiring for `cross_entropy`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, logits, target):
        loss, saved = cross_entropy_forward_with_saved(logits, target)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return loss

    @staticmethod
    def backward(ctx, dloss):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dloss = dloss.contiguous() if (dloss is not None and not dloss.is_contiguous()) else dloss
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dlogits, = cross_entropy_backward_from_saved(
            dloss, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dlogits, None


def cross_entropy_deployment(logits, target):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return CrossEntropyFunction.apply(logits, target)


class CrossEntropyModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, logits, target):
        return cross_entropy_deployment(logits, target)


DEPLOYMENT_ENTRY = "cross_entropy_deployment"
