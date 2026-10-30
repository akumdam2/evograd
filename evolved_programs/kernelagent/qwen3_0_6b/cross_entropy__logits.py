"""adbench-tools artifact: task qwen3_0_6b.cross_entropy.logits, produced by kernelagent.

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
# Triton implementation of:
#   loss = mean(cross_entropy(logits, target, ignore_index=-100))
#   d_logits = upstream * grad(loss, logits)
#
# Fusion note:
# The expensive row-wise softmax work is fused with the backward computation:
# after forward-side row statistics are computed, the gradient kernel recomputes
# the softmax probabilities and writes dlogits directly.  The public
# kernel_function writes the scalar forward loss and flattened backward gradient
# into one contiguous output tensor without using torch.cat or PyTorch math.
# -----------------------------------------------------------------------------


@triton.jit
def _partial_max_kernel(
    logits_ptr,
    partial_max_ptr,
    n_cols: tl.constexpr,
    n_col_blocks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)

    offs = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < n_cols

    vals = tl.load(
        logits_ptr + row * n_cols + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    m = tl.max(vals, axis=0)
    tl.store(partial_max_ptr + row * n_col_blocks + cb, m)


@triton.jit
def _reduce_max_kernel(
    partial_max_ptr,
    row_max_ptr,
    n_col_blocks: tl.constexpr,
    BLOCK_NB: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < n_col_blocks

    vals = tl.load(
        partial_max_ptr + row * n_col_blocks + offs,
        mask=mask,
        other=-float("inf"),
    )
    m = tl.max(vals, axis=0)
    tl.store(row_max_ptr + row, m)


@triton.jit
def _partial_sum_kernel(
    logits_ptr,
    row_max_ptr,
    partial_sum_ptr,
    n_cols: tl.constexpr,
    n_col_blocks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)

    offs = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < n_cols

    m = tl.load(row_max_ptr + row)
    vals = tl.load(
        logits_ptr + row * n_cols + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    exps = tl.exp(vals - m)
    s = tl.sum(exps, axis=0)
    tl.store(partial_sum_ptr + row * n_col_blocks + cb, s)


@triton.jit
def _row_sum_loss_kernel(
    logits_ptr,
    target_ptr,
    partial_sum_ptr,
    row_max_ptr,
    row_sum_ptr,
    row_loss_ptr,
    n_cols: tl.constexpr,
    n_col_blocks: tl.constexpr,
    BLOCK_NB: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_NB)
    mask = offs < n_col_blocks

    parts = tl.load(
        partial_sum_ptr + row * n_col_blocks + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    denom = tl.sum(parts, axis=0)

    m = tl.load(row_max_ptr + row)
    tgt = tl.load(target_ptr + row)
    valid = (tgt >= 0) & (tgt < n_cols)

    safe_tgt = tl.where(valid, tgt, 0)
    target_logit = tl.load(
        logits_ptr + row * n_cols + safe_tgt,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    row_loss = tl.where(valid, tl.log(denom) + m - target_logit, 0.0)

    tl.store(row_sum_ptr + row, denom)
    tl.store(row_loss_ptr + row, row_loss)


@triton.jit
def _partial_loss_reduce_kernel(
    target_ptr,
    row_loss_ptr,
    partial_loss_ptr,
    partial_count_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = offs < n_rows

    losses = tl.load(row_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tgt = tl.load(target_ptr + offs, mask=mask, other=-1)

    valid = mask & (tgt >= 0) & (tgt < n_cols)
    counts = tl.where(valid, 1.0, 0.0)

    total_loss = tl.sum(losses, axis=0)
    total_count = tl.sum(counts, axis=0)

    tl.store(partial_loss_ptr + pid, total_loss)
    tl.store(partial_count_ptr + pid, total_count)


@triton.jit
def _final_loss_kernel(
    partial_loss_ptr,
    partial_count_ptr,
    loss_out_ptr,
    count_out_ptr,
    n_reduce_blocks: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_P)
    mask = offs < n_reduce_blocks

    losses = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(partial_count_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(losses, axis=0)
    total_count = tl.sum(counts, axis=0)

    mean_loss = total_loss / total_count

    tl.store(loss_out_ptr, mean_loss)
    tl.store(count_out_ptr, total_count)


@triton.jit
def _grad_kernel(
    logits_ptr,
    target_ptr,
    row_max_ptr,
    row_sum_ptr,
    count_ptr,
    upstream_ptr,
    grad_out_ptr,
    n_cols: tl.constexpr,
    GRAD_OFFSET: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cb = tl.program_id(1)

    offs = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < n_cols

    tgt = tl.load(target_ptr + row)
    valid = (tgt >= 0) & (tgt < n_cols)

    m = tl.load(row_max_ptr + row)
    denom = tl.load(row_sum_ptr + row).to(tl.float32)
    total_count = tl.load(count_ptr).to(tl.float32)
    upstream = tl.load(upstream_ptr).to(tl.float32)

    vals = tl.load(
        logits_ptr + row * n_cols + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    probs = tl.exp(vals - m) / denom
    one_hot = (offs == tgt) & mask

    scale = upstream / total_count
    grad = scale * (probs - one_hot.to(tl.float32))
    grad = tl.where(valid, grad, 0.0)

    tl.store(
        grad_out_ptr + GRAD_OFFSET + row * n_cols + offs,
        grad,
        mask=mask,
    )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _launch_forward(logits, target, loss_out):
    """
    Launches Triton forward cross-entropy reductions.

    Returns saved tensors needed for backward:
      (logits, target, row_max, row_sum, count_scalar)
    """
    assert logits.is_cuda and target.is_cuda
    assert logits.is_contiguous(), "This Triton implementation expects contiguous logits."
    assert target.is_contiguous(), "This Triton implementation expects contiguous target."
    assert target.numel() > 0
    assert logits.numel() % target.numel() == 0

    n_rows = target.numel()
    n_cols = logits.numel() // n_rows

    BLOCK_C = 1024
    BLOCK_R = 1024

    n_col_blocks = triton.cdiv(n_cols, BLOCK_C)
    block_nb = _next_power_of_2(n_col_blocks)

    partial_max = torch.empty((n_rows, n_col_blocks), device=logits.device, dtype=torch.float32)
    partial_sum = torch.empty((n_rows, n_col_blocks), device=logits.device, dtype=torch.float32)
    row_max = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_sum = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    row_loss = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)

    _partial_max_kernel[(n_rows, n_col_blocks)](
        logits,
        partial_max,
        n_cols,
        n_col_blocks,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _reduce_max_kernel[(n_rows,)](
        partial_max,
        row_max,
        n_col_blocks,
        BLOCK_NB=block_nb,
        num_warps=4,
    )

    _partial_sum_kernel[(n_rows, n_col_blocks)](
        logits,
        row_max,
        partial_sum,
        n_cols,
        n_col_blocks,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    _row_sum_loss_kernel[(n_rows,)](
        logits,
        target,
        partial_sum,
        row_max,
        row_sum,
        row_loss,
        n_cols,
        n_col_blocks,
        BLOCK_NB=block_nb,
        num_warps=4,
    )

    n_reduce_blocks = triton.cdiv(n_rows, BLOCK_R)
    block_p = _next_power_of_2(n_reduce_blocks)

    partial_loss = torch.empty((n_reduce_blocks,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((n_reduce_blocks,), device=logits.device, dtype=torch.float32)
    count_scalar = torch.empty((), device=logits.device, dtype=torch.float32)

    _partial_loss_reduce_kernel[(n_reduce_blocks,)](
        target,
        row_loss,
        partial_loss,
        partial_count,
        n_rows,
        n_cols,
        BLOCK_R=BLOCK_R,
        num_warps=4,
    )

    _final_loss_kernel[(1,)](
        partial_loss,
        partial_count,
        loss_out,
        count_scalar,
        n_reduce_blocks,
        BLOCK_P=block_p,
        num_warps=4,
    )

    return logits, target, row_max, row_sum, count_scalar


def cross_entropy_forward_with_saved(logits, target):
    """
    Forward half of cross entropy with saved row statistics.

    Computes mean cross entropy over flattened rows, treating the final logits
    dimension as the class dimension.  Targets outside [0, C) are ignored,
    matching the common ignore_index=-100 language-modeling case.
    """
    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    saved = _launch_forward(logits, target, loss)
    return loss, saved


def cross_entropy_backward_from_saved(upstream, saved):
    """
    Backward half for d(loss)/d(logits), using only Triton kernels.

    The upstream gradient is expected to be the scalar cotangent for the scalar
    mean loss.
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    logits, target, row_max, row_sum, count_scalar = saved

    n_rows = target.numel()
    n_cols = logits.numel() // n_rows
    BLOCK_C = 1024
    n_col_blocks = triton.cdiv(n_cols, BLOCK_C)

    grad = torch.empty_like(logits)

    _grad_kernel[(n_rows, n_col_blocks)](
        logits,
        target,
        row_max,
        row_sum,
        count_scalar,
        upstream,
        grad,
        n_cols,
        GRAD_OFFSET=0,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )

    return grad


def kernel_function(logits, target, upstream):
    """
    Fused forward+backward step required by the evaluator.

    Returns a single flat float32 tensor:
        [ mean_cross_entropy_loss, flattened_dlogits... ]

    All numerical work is performed in Triton kernels.  The wrapper only checks
    layout assumptions, allocates tensors, computes launch sizes, and dispatches
    kernels.  No torch.nn, torch.nn.functional, torch.cat, or tensor math is
    used in the execution path.
    """
    assert logits.is_cuda and target.is_cuda and upstream.is_cuda
    assert logits.is_contiguous(), "This Triton implementation expects contiguous logits."
    assert target.is_contiguous(), "This Triton implementation expects contiguous target."
    assert target.numel() > 0
    assert logits.numel() % target.numel() == 0

    out = torch.empty((logits.numel() + 1,), device=logits.device, dtype=torch.float32)

    saved = _launch_forward(logits, target, out)

    logits_s, target_s, row_max, row_sum, count_scalar = saved
    n_rows = target_s.numel()
    n_cols = logits_s.numel() // n_rows

    BLOCK_C = 1024
    n_col_blocks = triton.cdiv(n_cols, BLOCK_C)

    _grad_kernel[(n_rows, n_col_blocks)](
        logits_s,
        target_s,
        row_max,
        row_sum,
        count_scalar,
        upstream,
        out,
        n_cols,
        GRAD_OFFSET=1,
        BLOCK_C=BLOCK_C,
        num_warps=4,
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
