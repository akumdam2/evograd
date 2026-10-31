"""adbench-tools artifact: task torchrec_dlrm.embedding_bag.tables, produced by kernelagent.

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
def _zero_segment_kernel(
    ptr,
    base: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(ptr + base + offs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


@triton.jit
def _zero_contiguous_kernel(
    ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(ptr + offs, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


@triton.jit
def _embedding_bag_forward_kernel(
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    out_ptr,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    emb_dim: tl.constexpr,
    stride_w0: tl.constexpr,
    stride_w1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bag = tl.program_id(0)
    d_block = tl.program_id(1)

    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < emb_dim

    start = tl.load(offsets_ptr + bag)
    end = tl.load(offsets_ptr + bag + 1, mask=bag + 1 < num_bags, other=num_indices)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    r = tl.arange(0, BLOCK_N)

    pos = start
    while pos < end:
        cur = pos + r
        row_mask = cur < end
        idx = tl.load(indices_ptr + cur, mask=row_mask, other=0)

        vals = tl.load(
            weight_ptr + idx[:, None] * stride_w0 + cols[None, :] * stride_w1,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(vals, axis=0)
        pos += BLOCK_N

    tl.store(out_ptr + bag * emb_dim + cols, acc, mask=col_mask)


@triton.jit
def _embedding_bag_backward_kernel(
    indices_ptr,
    offsets_ptr,
    upstream_ptr,
    grad_ptr,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    emb_dim: tl.constexpr,
    stride_u0: tl.constexpr,
    stride_u1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bag = tl.program_id(0)
    d_block = tl.program_id(1)

    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < emb_dim

    start = tl.load(offsets_ptr + bag)
    end = tl.load(offsets_ptr + bag + 1, mask=bag + 1 < num_bags, other=num_indices)

    upstream_vals = tl.load(
        upstream_ptr + bag * stride_u0 + cols * stride_u1,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    r = tl.arange(0, BLOCK_N)
    pos = start
    while pos < end:
        cur = pos + r
        row_mask = cur < end
        idx = tl.load(indices_ptr + cur, mask=row_mask, other=0)

        tl.atomic_add(
            grad_ptr + idx[:, None] * emb_dim + cols[None, :],
            upstream_vals[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            sem="relaxed",
        )

        pos += BLOCK_N


@triton.jit
def _embedding_bag_pair_flat_vec_kernel(
    weight_ptr,
    indices_ptr,
    offsets_ptr,
    upstream_ptr,
    flat_out_ptr,
    num_indices: tl.constexpr,
    num_bags: tl.constexpr,
    emb_dim: tl.constexpr,
    grad_offset: tl.constexpr,
    stride_w0: tl.constexpr,
    stride_w1: tl.constexpr,
    stride_u0: tl.constexpr,
    stride_u1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bag = tl.program_id(0)
    d_block = tl.program_id(1)

    cols = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < emb_dim

    start = tl.load(offsets_ptr + bag)
    end = tl.load(offsets_ptr + bag + 1, mask=bag + 1 < num_bags, other=num_indices)

    upstream_vals = tl.load(
        upstream_ptr + bag * stride_u0 + cols * stride_u1,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    r = tl.arange(0, BLOCK_N)

    pos = start
    while pos < end:
        cur = pos + r
        row_mask = cur < end
        idx = tl.load(indices_ptr + cur, mask=row_mask, other=0)

        vals = tl.load(
            weight_ptr + idx[:, None] * stride_w0 + cols[None, :] * stride_w1,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(vals, axis=0)

        tl.atomic_add(
            flat_out_ptr + grad_offset + idx[:, None] * emb_dim + cols[None, :],
            upstream_vals[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            sem="relaxed",
        )

        pos += BLOCK_N

    tl.store(flat_out_ptr + bag * emb_dim + cols, acc, mask=col_mask)


def _choose_block_d(emb_dim: int) -> int:
    if emb_dim <= 16:
        return 16
    if emb_dim <= 32:
        return 32
    if emb_dim <= 128:
        return 64
    return 128


def _choose_block_n(num_indices: int, num_bags: int, block_d: int) -> int:
    avg_len = triton.cdiv(num_indices, num_bags) if num_bags > 0 else 1
    if avg_len <= 1:
        return 1
    if avg_len <= 3:
        return 2
    if block_d <= 64:
        return 8
    return 4


def _choose_num_warps(block_d: int, block_n: int) -> int:
    elems = block_d * block_n
    if elems >= 1024:
        return 8
    if elems >= 256:
        return 4
    return 1


def embedding_bag_forward_with_saved(weight, indices, offsets):
    num_bags = offsets.numel()
    emb_dim = weight.shape[1]

    out = torch.empty((num_bags, emb_dim), device=weight.device, dtype=weight.dtype)

    if num_bags > 0 and emb_dim > 0:
        bd = _choose_block_d(emb_dim)
        bn = _choose_block_n(indices.numel(), num_bags, bd)
        nw = _choose_num_warps(bd, bn)

        _embedding_bag_forward_kernel[(num_bags, triton.cdiv(emb_dim, bd))](
            weight,
            indices,
            offsets,
            out,
            indices.numel(),
            num_bags,
            emb_dim,
            weight.stride(0),
            weight.stride(1),
            BLOCK_N=bn,
            BLOCK_D=bd,
            num_warps=nw,
            num_stages=3,
        )

    saved = (weight, indices, offsets)
    return out, saved


def embedding_bag_backward_from_saved(upstream, saved):
    if isinstance(upstream, (tuple, list)):
        upstream = upstream[0]

    weight, indices, offsets = saved
    num_bags = offsets.numel()
    emb_dim = weight.shape[1]
    grad_elements = weight.numel()

    grad_weight = torch.empty((weight.shape[0], emb_dim), device=weight.device, dtype=weight.dtype)

    if grad_elements > 0:
        _zero_contiguous_kernel[(triton.cdiv(grad_elements, 1024),)](
            grad_weight,
            grad_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

    if num_bags > 0 and emb_dim > 0:
        bd = _choose_block_d(emb_dim)
        bn = _choose_block_n(indices.numel(), num_bags, bd)
        nw = _choose_num_warps(bd, bn)

        _embedding_bag_backward_kernel[(num_bags, triton.cdiv(emb_dim, bd))](
            indices,
            offsets,
            upstream,
            grad_weight,
            indices.numel(),
            num_bags,
            emb_dim,
            upstream.stride(0),
            upstream.stride(1),
            BLOCK_N=bn,
            BLOCK_D=bd,
            num_warps=nw,
            num_stages=3,
        )

    return grad_weight


def kernel_function(weight, indices, offsets, upstream):
    num_bags = offsets.numel()
    emb_dim = weight.shape[1]
    out_elements = num_bags * emb_dim
    grad_elements = weight.numel()
    total_elements = out_elements + grad_elements

    flat = torch.empty((total_elements,), device=weight.device, dtype=torch.float32)

    # The forward sum and backward scatter-add are fused below.  The gradient
    # segment must still be zeroed in a separate launch because atomic_add has no
    # grid-wide ordering with stores from sibling programs in the same kernel.
    if grad_elements > 0:
        _zero_segment_kernel[(triton.cdiv(grad_elements, 1024),)](
            flat,
            out_elements,
            grad_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

    if num_bags > 0 and emb_dim > 0:
        bd = _choose_block_d(emb_dim)
        bn = _choose_block_n(indices.numel(), num_bags, bd)
        nw = _choose_num_warps(bd, bn)

        _embedding_bag_pair_flat_vec_kernel[(num_bags, triton.cdiv(emb_dim, bd))](
            weight,
            indices,
            offsets,
            upstream,
            flat,
            indices.numel(),
            num_bags,
            emb_dim,
            out_elements,
            weight.stride(0),
            weight.stride(1),
            upstream.stride(0),
            upstream.stride(1),
            BLOCK_N=bn,
            BLOCK_D=bd,
            num_warps=nw,
            num_stages=3,
        )

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
        dweight = embedding_bag_backward_from_saved(
            dout, saved
        )
        if isinstance(dweight, (tuple, list)):
            dweight, = dweight
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
