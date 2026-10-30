import torch
import triton
import triton.language as tl

# EVOLVE-BLOCK-START
def _embedding_bag_forward_kernel(weight, indices, offsets, out, L, B, D, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    db = tl.program_id(1)
    cols = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    start = tl.load(offsets + b)
    next_b = b + 1
    end = tl.load(offsets + next_b, mask=next_b < B, other=L)

    acc = tl.zeros((BLOCK_D,), tl.float32)
    p = start
    while p < end:
        row = tl.load(indices + p)
        vals = tl.load(weight + row * D + cols, mask=mask, other=0.0)
        acc += vals
        p += 1

    tl.store(out + b * D + cols, acc, mask=mask)


_embedding_bag_forward_kernel = triton.jit(_embedding_bag_forward_kernel)


def _embedding_bag_forward_m1_kernel(weight, indices, out, D, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    db = tl.program_id(1)
    cols = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    row = tl.load(indices + b)
    vals = tl.load(weight + row * D + cols, mask=mask, other=0.0)
    tl.store(out + b * D + cols, vals, mask=mask)


_embedding_bag_forward_m1_kernel = triton.jit(_embedding_bag_forward_m1_kernel)


def _embedding_bag_forward_m1_d64_kernel(weight, indices, out):
    b = tl.program_id(0)
    cols = tl.arange(0, 64)
    row = tl.load(indices + b)
    vals = tl.load(weight + row * 64 + cols)
    tl.store(out + b * 64 + cols, vals)


_embedding_bag_forward_m1_d64_kernel = triton.jit(_embedding_bag_forward_m1_d64_kernel)


def _zero_kernel(ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), tl.float32)
    tl.store(ptr + offs, z, mask=mask)


_zero_kernel = triton.jit(_zero_kernel)


def _embedding_bag_backward_scatter_kernel(dout, indices, offsets, dweight, L, B, D, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    db = tl.program_id(1)
    cols = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D

    g = tl.load(dout + b * D + cols, mask=mask, other=0.0)

    start = tl.load(offsets + b)
    next_b = b + 1
    end = tl.load(offsets + next_b, mask=next_b < B, other=L)

    p = start
    while p < end:
        row = tl.load(indices + p)
        tl.atomic_add(dweight + row * D + cols, g, sem="relaxed", mask=mask)
        p += 1


_embedding_bag_backward_scatter_kernel = triton.jit(_embedding_bag_backward_scatter_kernel)


def _embedding_bag_backward_m1_scatter_kernel(dout, indices, dweight, D, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    db = tl.program_id(1)
    cols = db * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < D
    row = tl.load(indices + b)
    g = tl.load(dout + b * D + cols, mask=mask, other=0.0)
    tl.atomic_add(dweight + row * D + cols, g, sem="relaxed", mask=mask)


_embedding_bag_backward_m1_scatter_kernel = triton.jit(_embedding_bag_backward_m1_scatter_kernel)


def _embedding_bag_backward_m1_d64_scatter_kernel(dout, indices, dweight):
    b = tl.program_id(0)
    cols = tl.arange(0, 64)
    row = tl.load(indices + b)
    g = tl.load(dout + b * 64 + cols)
    tl.atomic_add(dweight + row * 64 + cols, g, sem="relaxed")


_embedding_bag_backward_m1_d64_scatter_kernel = triton.jit(_embedding_bag_backward_m1_d64_scatter_kernel)


def embedding_bag_forward_with_saved(weight, indices, offsets):
    N = weight.shape[0]
    D = weight.shape[1]
    L = indices.shape[0]
    B = offsets.shape[0]

    out = torch.empty((B, D), device=weight.device, dtype=weight.dtype)
    block_d = 64
    grid = (B, triton.cdiv(D, block_d))

    # Benchmark/common case: fixed M=1, so each bag is exactly one row gather.
    # Avoid offset loads and the reduction loop; also do not save offsets.
    if L == B:
        if D == 64:
            _embedding_bag_forward_m1_d64_kernel[(B,)](
                weight,
                indices,
                out,
                num_warps=1,
            )
        else:
            _embedding_bag_forward_m1_kernel[grid](
                weight,
                indices,
                out,
                D,
                BLOCK_D=block_d,
                num_warps=1,
            )
        return out, (indices, N)

    _embedding_bag_forward_kernel[grid](
        weight,
        indices,
        offsets,
        out,
        L,
        B,
        D,
        BLOCK_D=block_d,
        num_warps=2,
    )
    return out, (indices, offsets, N)


def embedding_bag_backward_from_saved(dout, saved_tensors):
    # Two saved-state layouts:
    #   M=1 fast path: (indices, N)
    #   generic path:  (indices, offsets, N)
    if len(saved_tensors) == 2:
        indices, N = saved_tensors
        offsets = None
    else:
        indices, offsets, N = saved_tensors

    B = dout.shape[0]
    D = dout.shape[1]
    L = indices.shape[0]

    dweight = torch.empty((N, D), device=dout.device, dtype=dout.dtype)
    dweight.zero_()

    block_d = 64
    scatter_grid = (B, triton.cdiv(D, block_d))

    if offsets is None:
        if D == 64:
            _embedding_bag_backward_m1_d64_scatter_kernel[(B,)](
                dout,
                indices,
                dweight,
                num_warps=1,
            )
        else:
            _embedding_bag_backward_m1_scatter_kernel[scatter_grid](
                dout,
                indices,
                dweight,
                D,
                BLOCK_D=block_d,
                num_warps=1,
            )
    else:
        _embedding_bag_backward_scatter_kernel[scatter_grid](
            dout,
            indices,
            offsets,
            dweight,
            L,
            B,
            D,
            BLOCK_D=block_d,
            num_warps=2,
        )

    return dweight,
# EVOLVE-BLOCK-END


# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'embedding_bag', 'arguments': ['weight', 'indices', 'offsets'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dweight'], 'gradient_return_order': ['dweight', 'None', 'None'], 'scalars': [], 'deployment_entry': 'embedding_bag_deployment', 'allowed_primitives': []}


class EmbeddingBagFunction(torch.autograd.Function):
    """Static autograd wiring for `embedding_bag`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, weight, indices, offsets):
        out, saved = embedding_bag_forward_with_saved(weight, indices, offsets)
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
        dweight, = embedding_bag_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dweight, None, None


def embedding_bag_deployment(weight, indices, offsets):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if weight.dim() > 2:
        _leading = weight.shape[:-1]
        weight = weight.reshape(-1, weight.shape[-1])
    out = EmbeddingBagFunction.apply(weight, indices, offsets)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class EmbeddingBagModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, indices, offsets):
        return embedding_bag_deployment(self.weight, indices, offsets)


DEPLOYMENT_ENTRY = "embedding_bag_deployment"
