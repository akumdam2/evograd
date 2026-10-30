"""adbench-tools artifact: task qwen3_5_4b.cross_entropy.logits, produced by kernelagent.

Everything between the EVOLVE-BLOCK markers is the agent's module, verbatim.
Everything after them is generated from the declaration by evograd.
"""
import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
# kernel.py
#
# Complete Triton implementation for:
#   torch.nn.functional.cross_entropy(logits, target)
# differentiated with respect to logits.
#
# Fusion notes:
# - The forward row kernel fuses the numerically stable log-softmax statistics
#   computation and per-row NLL loss computation in one Triton program per row.
# - The backward kernel fuses softmax reconstruction and cross-entropy gradient
#   formation in one pass over logits.
# - A small reduction stage is kept separate because the mean-reduction
#   denominator/sum is global across rows.

import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_row_stats_kernel(
    logits_ptr,
    target_ptr,
    row_loss_ptr,
    row_valid_ptr,
    row_max_ptr,
    row_sumexp_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    """Compute per-row stable CE loss pieces.

    For each row:
      m = max_j logits[row, j]
      s = sum_j exp(logits[row, j] - m)
      loss_row = log(s) + m - logits[row, target[row]]
    Ignored rows contribute zero loss and zero count.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)

    # Stable row max.
    m = tl.full((), -float("inf"), dtype=tl.float32)
    for start in tl.range(0, C, BLOCK_C):
        cols = start + offs
        mask = cols < C
        vals = tl.load(logits_ptr + row * C + cols, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(vals, axis=0)
        m = tl.maximum(m, block_max)

    # Stable denominator sum exp(logit - max).
    s = tl.zeros((), dtype=tl.float32)
    for start in tl.range(0, C, BLOCK_C):
        cols = start + offs
        mask = cols < C
        vals = tl.load(logits_ptr + row * C + cols, mask=mask, other=-float("inf")).to(tl.float32)
        s += tl.sum(tl.exp(vals - m), axis=0)

    t = tl.load(target_ptr + row)
    valid = (t != IGNORE_INDEX) & (t >= 0) & (t < C)
    safe_t = tl.where(valid, t, 0)

    target_logit = tl.load(logits_ptr + row * C + safe_t).to(tl.float32)
    loss = tl.log(s) + m - target_logit
    loss = tl.where(valid, loss, 0.0)
    valid_f = tl.where(valid, 1.0, 0.0)

    tl.store(row_loss_ptr + row, loss)
    tl.store(row_valid_ptr + row, valid_f)
    tl.store(row_max_ptr + row, m)
    tl.store(row_sumexp_ptr + row, s)


@triton.jit
def _reduce_rows_kernel(
    row_loss_ptr,
    row_valid_ptr,
    partial_loss_ptr,
    partial_count_ptr,
    N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Reduce row losses/counts into block partials."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = offs < N

    losses = tl.load(row_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(row_valid_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(partial_loss_ptr + pid, tl.sum(losses, axis=0))
    tl.store(partial_count_ptr + pid, tl.sum(counts, axis=0))


@triton.jit
def _final_reduce_kernel(
    partial_loss_ptr,
    partial_count_ptr,
    loss_out_ptr,
    count_out_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    """Finalize mean loss and save denominator count."""
    offs = tl.arange(0, BLOCK_PARTIALS)
    mask = offs < NUM_PARTIALS

    partial_losses = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    partial_counts = tl.load(partial_count_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(partial_losses, axis=0)
    total_count = tl.sum(partial_counts, axis=0)

    mean_loss = total_loss / total_count
    tl.store(loss_out_ptr, mean_loss)
    tl.store(count_out_ptr, total_count)


@triton.jit
def _cross_entropy_backward_kernel(
    logits_ptr,
    target_ptr,
    row_max_ptr,
    row_sumexp_ptr,
    count_ptr,
    upstream_ptr,
    grad_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    """Compute d(mean cross entropy) / d(logits).

    grad[row, col] =
        upstream / count * (softmax[row, col] - 1[col == target[row]])
    for valid rows, otherwise zero.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)

    offs = block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < C

    vals = tl.load(logits_ptr + row * C + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.load(row_max_ptr + row).to(tl.float32)
    s = tl.load(row_sumexp_ptr + row).to(tl.float32)

    probs = tl.exp(vals - m) / s

    t = tl.load(target_ptr + row)
    valid = (t != IGNORE_INDEX) & (t >= 0) & (t < C)
    one_hot = offs == t

    count = tl.load(count_ptr).to(tl.float32)
    upstream = tl.load(upstream_ptr).to(tl.float32)
    scale = upstream / count

    grad = (probs - one_hot.to(tl.float32)) * scale
    grad = tl.where(valid, grad, 0.0)

    tl.store(grad_ptr + row * C + offs, grad, mask=mask)


def _shape_info(logits, target):
    """Metadata-only validation/helper."""
    assert logits.is_cuda and target.is_cuda, "logits and target must be CUDA tensors"
    assert logits.is_contiguous(), "this Triton implementation expects contiguous logits"
    assert target.is_contiguous(), "this Triton implementation expects contiguous target"
    assert logits.ndim >= 1, "logits must have at least one dimension"
    C = logits.shape[-1]
    N = target.numel()
    assert C > 0 and N > 0, "empty logits/target are not supported"
    assert logits.numel() == N * C, "target must match logits.shape[:-1]"
    return N, C


def _launch_forward(logits, target, loss_out):
    """Allocate temporaries and launch the Triton forward/reduction pipeline."""
    N, C = _shape_info(logits, target)

    row_loss = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_valid = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_sumexp = torch.empty((N,), device=logits.device, dtype=torch.float32)
    count = torch.empty((), device=logits.device, dtype=torch.float32)

    BLOCK_C_STATS = 1024
    _cross_entropy_row_stats_kernel[(N,)](
        logits,
        target,
        row_loss,
        row_valid,
        row_max,
        row_sumexp,
        N,
        C,
        BLOCK_C=BLOCK_C_STATS,
        IGNORE_INDEX=-100,
        num_warps=8,
    )

    BLOCK_ROWS = 1024
    num_partials = triton.cdiv(N, BLOCK_ROWS)
    partial_loss = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)

    _reduce_rows_kernel[(num_partials,)](
        row_loss,
        row_valid,
        partial_loss,
        partial_count,
        N,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=8,
    )

    block_partials = triton.next_power_of_2(num_partials)
    _final_reduce_kernel[(1,)](
        partial_loss,
        partial_count,
        loss_out,
        count,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=block_partials,
        num_warps=8,
    )

    return (logits, target, row_max, row_sumexp, count)


def cross_entropy_forward_with_saved(logits, target):
    """Forward pass for mean cross entropy with saved tensors for backward.

    Computes the scalar mean cross-entropy loss over all non-ignored targets
    using Triton only. The saved tuple contains logits/target and row-wise
    softmax statistics needed to reconstruct the gradient without materializing
    the full softmax.
    """
    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    saved = _launch_forward(logits, target, loss)
    return loss, saved


def cross_entropy_backward_from_saved(upstream, saved):
    """Backward pass w.r.t. logits for the saved cross-entropy forward pass."""
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    logits, target, row_max, row_sumexp, count = saved
    N, C = _shape_info(logits, target)

    grad = torch.empty_like(logits)

    BLOCK_C_BWD = 256
    grid = (N, triton.cdiv(C, BLOCK_C_BWD))
    _cross_entropy_backward_kernel[grid](
        logits,
        target,
        row_max,
        row_sumexp,
        count,
        upstream,
        grad,
        N,
        C,
        BLOCK_C=BLOCK_C_BWD,
        IGNORE_INDEX=-100,
        num_warps=4,
    )

    return grad


def kernel_function(logits, target, upstream):
    """Return flattened forward output followed by flattened logits gradient.

    This is the fused forward+backward step expected by the evaluator:
      actual = concat([loss.reshape(-1), grad_logits.reshape(-1)])

    The wrapper performs only validation, allocation, grid setup, and launches.
    All numerical work is done inside Triton kernels.
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    N, C = _shape_info(logits, target)

    # The evaluator compares against a float-flattened concatenation, so this
    # output buffer is float32: slot 0 is loss, remaining slots are grad logits.
    actual = torch.empty((logits.numel() + 1,), device=logits.device, dtype=torch.float32)
    count = torch.empty((), device=logits.device, dtype=torch.float32)

    row_loss = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_valid = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_sumexp = torch.empty((N,), device=logits.device, dtype=torch.float32)

    BLOCK_C_STATS = 1024
    _cross_entropy_row_stats_kernel[(N,)](
        logits,
        target,
        row_loss,
        row_valid,
        row_max,
        row_sumexp,
        N,
        C,
        BLOCK_C=BLOCK_C_STATS,
        IGNORE_INDEX=-100,
        num_warps=8,
    )

    BLOCK_ROWS = 1024
    num_partials = triton.cdiv(N, BLOCK_ROWS)
    partial_loss = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)

    _reduce_rows_kernel[(num_partials,)](
        row_loss,
        row_valid,
        partial_loss,
        partial_count,
        N,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=8,
    )

    block_partials = triton.next_power_of_2(num_partials)
    _final_reduce_kernel[(1,)](
        partial_loss,
        partial_count,
        actual,  # actual[0] receives scalar loss
        count,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=block_partials,
        num_warps=8,
    )

    BLOCK_C_BWD = 256
    grad_ptr_offset = 1
    grid = (N, triton.cdiv(C, BLOCK_C_BWD))
    _cross_entropy_backward_kernel[grid](
        logits,
        target,
        row_max,
        row_sumexp,
        count,
        upstream,
        actual[grad_ptr_offset:],
        N,
        C,
        BLOCK_C=BLOCK_C_BWD,
        IGNORE_INDEX=-100,
        num_warps=4,
    )

    return actual
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
