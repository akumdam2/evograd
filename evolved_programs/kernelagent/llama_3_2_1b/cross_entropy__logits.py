"""adbench-tools artifact: task llama_3_2_1b.cross_entropy.logits, produced by kernelagent.

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


# This implementation targets the common LLaMA-style cross entropy pattern:
#   logits: contiguous tensor with class/vocabulary dimension last
#   target: contiguous int tensor with target.numel() rows
#   loss = mean(-log_softmax(logits.reshape(-1, C))[target]), ignore_index=-100
#
# Fused stages:
#   1. Per-row logsumexp + NLL are fused in _ce_loss_rows_kernel.
#   2. Reduction of row losses and valid-token counts is done in Triton.
#   3. Backward recomputes per-row softmax and writes dlogits in one Triton pass.
#   4. kernel_function packs forward scalar loss and flattened dlogits directly
#      into one output tensor, avoiding torch.cat / PyTorch math.


@triton.jit
def _ce_loss_rows_kernel(
    logits_ptr,
    target_ptr,
    row_loss_ptr,
    row_count_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    t = tl.load(target_ptr + row)
    valid = t != IGNORE_INDEX

    vals = tl.load(logits_ptr + row * C + offs, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(vals, axis=0)
    exps = tl.exp(vals - m)
    s = tl.sum(exps, axis=0)
    logsum = tl.log(s) + m

    target_val = tl.load(
        logits_ptr + row * C + t,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    loss = tl.where(valid, logsum - target_val, 0.0)
    count = tl.where(valid, 1.0, 0.0)

    tl.store(row_loss_ptr + row, loss)
    tl.store(row_count_ptr + row, count)


@triton.jit
def _reduce_rows_kernel(
    row_loss_ptr,
    row_count_ptr,
    partial_loss_ptr,
    partial_count_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N

    losses = tl.load(row_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(row_count_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    tl.store(partial_loss_ptr + pid, tl.sum(losses, axis=0))
    tl.store(partial_count_ptr + pid, tl.sum(counts, axis=0))


@triton.jit
def _final_loss_kernel(
    partial_loss_ptr,
    partial_count_ptr,
    out_loss_ptr,
    count_scalar_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_P)
    mask = offs < NUM_PARTIALS

    losses = tl.load(partial_loss_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    counts = tl.load(partial_count_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    total_loss = tl.sum(losses, axis=0)
    total_count = tl.sum(counts, axis=0)

    mean_loss = total_loss / total_count

    tl.store(out_loss_ptr, mean_loss)
    tl.store(count_scalar_ptr, total_count)


@triton.jit
def _ce_backward_kernel(
    logits_ptr,
    target_ptr,
    upstream_ptr,
    count_scalar_ptr,
    grad_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    t = tl.load(target_ptr + row)
    valid = t != IGNORE_INDEX

    vals = tl.load(logits_ptr + row * C + offs, mask=mask, other=-float("inf")).to(tl.float32)

    m = tl.max(vals, axis=0)
    exps = tl.exp(vals - m)
    denom = tl.sum(exps, axis=0)
    probs = exps / denom

    is_target = offs == t
    grad = probs - tl.where(is_target, 1.0, 0.0)

    upstream = tl.load(upstream_ptr).to(tl.float32)
    count = tl.load(count_scalar_ptr).to(tl.float32)
    safe_count = tl.maximum(count, 1.0)

    scale = tl.where(valid, upstream / safe_count, 0.0)
    grad = grad * scale

    tl.store(grad_ptr + row * C + offs, grad, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _launch_forward(logits, target, out_loss):
    assert logits.is_cuda and target.is_cuda
    assert logits.is_contiguous(), "This Triton CE kernel expects contiguous logits"
    assert target.is_contiguous(), "This Triton CE kernel expects contiguous target"
    assert target.numel() > 0

    N = target.numel()
    assert logits.numel() % N == 0
    C = logits.numel() // N
    assert C > 0

    block_c = _next_power_of_2(C)

    row_loss = torch.empty((N,), device=logits.device, dtype=torch.float32)
    row_count = torch.empty((N,), device=logits.device, dtype=torch.float32)

    _ce_loss_rows_kernel[(N,)](
        logits,
        target,
        row_loss,
        row_count,
        N,
        C,
        IGNORE_INDEX=-100,
        BLOCK_C=block_c,
        num_warps=8,
        num_stages=4,
    )

    block_n = 1024
    num_partials = triton.cdiv(N, block_n)

    partial_loss = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)
    partial_count = torch.empty((num_partials,), device=logits.device, dtype=torch.float32)

    _reduce_rows_kernel[(num_partials,)](
        row_loss,
        row_count,
        partial_loss,
        partial_count,
        N,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=4,
    )

    count_scalar = torch.empty((1,), device=logits.device, dtype=torch.float32)
    block_p = _next_power_of_2(num_partials)

    _final_loss_kernel[(1,)](
        partial_loss,
        partial_count,
        out_loss,
        count_scalar,
        NUM_PARTIALS=num_partials,
        BLOCK_P=block_p,
        num_warps=4,
        num_stages=4,
    )

    return count_scalar


def cross_entropy_forward_with_saved(logits, target):
    """
    Forward cross entropy.

    Returns:
        loss: scalar float32 tensor
        saved: tensors needed by the Triton backward pass
    """
    loss = torch.empty((), device=logits.device, dtype=torch.float32)
    count_scalar = _launch_forward(logits, target, loss)
    return loss, (logits, target, count_scalar)


def cross_entropy_backward_from_saved(upstream, saved):
    """
    Backward for mean cross entropy w.r.t. logits only.

    No PyTorch numerical operations are used; the softmax gradient is computed
    directly in Triton.
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    logits, target, count_scalar = saved

    assert logits.is_cuda and target.is_cuda and upstream.is_cuda
    assert logits.is_contiguous()
    assert target.is_contiguous()

    N = target.numel()
    C = logits.numel() // N
    block_c = _next_power_of_2(C)

    grad = torch.empty_like(logits)

    _ce_backward_kernel[(N,)](
        logits,
        target,
        upstream,
        count_scalar,
        grad,
        N,
        C,
        IGNORE_INDEX=-100,
        BLOCK_C=block_c,
        num_warps=8,
        num_stages=4,
    )

    return grad


def kernel_function(logits, target, upstream):
    """
    Fused forward+backward API required by the evaluator.

    Computes:
        loss = cross_entropy(logits.reshape(-1, C), target.reshape(-1), mean, ignore_index=-100)
        grad_logits = d(loss * upstream) / dlogits

    Returns a single 1D float32 tensor equivalent to:
        torch.cat([loss.float().reshape(-1), grad_logits.float().reshape(-1)])

    The concatenation/packing is performed by writing the scalar loss directly
    into output[0] and the flattened gradient directly into output[1:].
    """
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    assert logits.is_cuda and target.is_cuda and upstream.is_cuda
    assert logits.is_contiguous(), "This Triton CE kernel expects contiguous logits"
    assert target.is_contiguous(), "This Triton CE kernel expects contiguous target"

    N = target.numel()
    assert N > 0
    assert logits.numel() % N == 0

    C = logits.numel() // N
    block_c = _next_power_of_2(C)

    # Packed evaluator output: [loss, flattened grad_logits...]
    packed = torch.empty((1 + logits.numel(),), device=logits.device, dtype=torch.float32)

    # Forward writes the scalar loss into packed[0].
    count_scalar = _launch_forward(logits, target, packed)

    # Backward writes flattened gradient directly into packed[1:].
    _ce_backward_kernel[(N,)](
        logits,
        target,
        upstream,
        count_scalar,
        packed[1:],
        N,
        C,
        IGNORE_INDEX=-100,
        BLOCK_C=block_c,
        num_warps=8,
        num_stages=4,
    )

    return packed
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
