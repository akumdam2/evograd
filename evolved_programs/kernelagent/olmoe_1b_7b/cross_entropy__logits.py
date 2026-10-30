"""adbench-tools artifact: task olmoe_1b_7b.cross_entropy.logits, produced by kernelagent.

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
def _ce_row_loss_kernel(
    logits_ptr,
    target_ptr,
    row_loss_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = row * C
    vals = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)

    tgt = tl.load(target_ptr + row)
    valid = tgt != -100
    safe_tgt = tl.where(valid, tgt, 0)

    tgt_logit = tl.load(logits_ptr + base + safe_tgt).to(tl.float32)
    loss = tl.log(sum_exp) + max_val - tgt_logit
    loss = tl.where(valid, loss, 0.0)

    tl.store(row_loss_ptr + row, loss)


@triton.jit
def _reduce_loss_count_kernel(
    row_loss_ptr,
    target_ptr,
    partial_loss_ptr,
    partial_count_ptr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < N

    losses = tl.load(row_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    targets = tl.load(target_ptr + offs, mask=mask, other=-100)
    valid = (targets != -100) & mask
    counts = valid.to(tl.float32)

    tl.store(partial_loss_ptr + pid, tl.sum(losses, axis=0))
    tl.store(partial_count_ptr + pid, tl.sum(counts, axis=0))


@triton.jit
def _final_loss_count_kernel(
    partial_loss_ptr,
    partial_count_ptr,
    loss_ptr,
    denom_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    LOSS_OFFSET: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_P)
    mask = offs < NUM_PARTIALS

    losses = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(partial_count_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(losses, axis=0)
    total_count = tl.sum(counts, axis=0)

    tl.store(denom_ptr, total_count)
    tl.store(loss_ptr + LOSS_OFFSET, total_loss / total_count)


@triton.jit
def _ce_bwd_kernel(
    logits_ptr,
    target_ptr,
    upstream_ptr,
    denom_ptr,
    grad_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GRAD_OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    base = row * C
    vals = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf")).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    probs = exp_vals / sum_exp

    tgt = tl.load(target_ptr + row)
    valid = tgt != -100
    safe_tgt = tl.where(valid, tgt, 0)

    denom = tl.load(denom_ptr).to(tl.float32)
    upstream = tl.load(upstream_ptr).to(tl.float32)
    scale = upstream / denom

    one_hot = offs == safe_tgt
    grad = (probs - one_hot.to(tl.float32)) * scale
    grad = tl.where(valid, grad, 0.0)

    tl.store(grad_ptr + GRAD_OFFSET + base + offs, grad, mask=mask)


def _ce_launch_meta(logits, target):
    C = logits.shape[-1]
    M = target.numel()
    block_c = triton.next_power_of_2(C)

    if block_c >= 32768:
        warps = 16
    elif block_c >= 2048:
        warps = 8
    elif block_c >= 512:
        warps = 4
    else:
        warps = 1

    return M, C, block_c, warps


def cross_entropy_forward_with_saved(logits, target):
    M, C, block_c, warps = _ce_launch_meta(logits, target)

    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    denom = torch.empty((), device=logits.device, dtype=torch.float32)
    row_loss = torch.empty((M,), device=logits.device, dtype=torch.float32)

    _ce_row_loss_kernel[(M,)](
        logits,
        target,
        row_loss,
        M,
        C,
        BLOCK_C=block_c,
        num_warps=warps,
    )

    reduce_block = 1024
    num_partials = triton.cdiv(M, reduce_block)
    partial_loss = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)

    _reduce_loss_count_kernel[(num_partials,)](
        row_loss,
        target,
        partial_loss,
        partial_count,
        M,
        BLOCK_M=reduce_block,
        num_warps=4,
    )

    block_p = triton.next_power_of_2(num_partials)

    _final_loss_count_kernel[(1,)](
        partial_loss,
        partial_count,
        loss,
        denom,
        NUM_PARTIALS=num_partials,
        BLOCK_P=block_p,
        LOSS_OFFSET=0,
        num_warps=1,
    )

    saved = (logits, target, denom)
    return loss, saved


def cross_entropy_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    logits, target, denom = saved
    M, C, block_c, warps = _ce_launch_meta(logits, target)

    grad_logits = torch.empty_like(logits)

    _ce_bwd_kernel[(M,)](
        logits,
        target,
        upstream,
        denom,
        grad_logits,
        M,
        C,
        BLOCK_C=block_c,
        GRAD_OFFSET=0,
        num_warps=warps,
    )

    return grad_logits


def kernel_function(logits, target, upstream):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    M, C, block_c, warps = _ce_launch_meta(logits, target)

    out = torch.empty((logits.numel() + 1,), device=logits.device, dtype=torch.float32)
    denom = torch.empty((), device=logits.device, dtype=torch.float32)
    row_loss = torch.empty((M,), device=logits.device, dtype=torch.float32)

    _ce_row_loss_kernel[(M,)](
        logits,
        target,
        row_loss,
        M,
        C,
        BLOCK_C=block_c,
        num_warps=warps,
    )

    reduce_block = 1024
    num_partials = triton.cdiv(M, reduce_block)
    partial_loss = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)

    _reduce_loss_count_kernel[(num_partials,)](
        row_loss,
        target,
        partial_loss,
        partial_count,
        M,
        BLOCK_M=reduce_block,
        num_warps=4,
    )

    block_p = triton.next_power_of_2(num_partials)

    _final_loss_count_kernel[(1,)](
        partial_loss,
        partial_count,
        out,
        denom,
        NUM_PARTIALS=num_partials,
        BLOCK_P=block_p,
        LOSS_OFFSET=0,
        num_warps=1,
    )

    # Backward cannot be fused into the row-loss pass when ignore_index=-100 is
    # supported because the mean-reduction denominator is a global count that
    # must be reduced before scaling gradients.
    _ce_bwd_kernel[(M,)](
        logits,
        target,
        upstream,
        denom,
        out,
        M,
        C,
        BLOCK_C=block_c,
        GRAD_OFFSET=1,
        num_warps=warps,
    )

    return out
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
        dlogits = cross_entropy_backward_from_saved(
            dloss, saved
        )
        if isinstance(dlogits, (tuple, list)):
            dlogits, = dlogits
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
