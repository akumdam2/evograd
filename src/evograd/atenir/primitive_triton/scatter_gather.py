"""Triton kernels for gather/scatter and dynamic tensor assembly.

Two kernels per op cover the two 2D cases (dim=0 and dim=1).
For non-2D tensors the functions fall back to PyTorch.

gather(input, dim, index)      -- indexed read
scatter_add(self, dim, index, src) -- indexed atomic accumulate into clone of self
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


if HAS_TRITON:

    @triton.jit
    def _gather_1d_kernel(input_ptr, index_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        idx = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + idx, mask=mask, other=0.0)

        tl.store(out_ptr + offs, val, mask=mask)

    @triton.jit
    def _scatter_add_1d_kernel(out_ptr, index_ptr, src_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        idx = tl.load(index_ptr + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + offs, mask=mask, other=0.0)

        tl.atomic_add(out_ptr + idx, val, mask=mask)

    @triton.jit
    def _gather_dim1_kernel(input_ptr, index_ptr, out_ptr, C_in, C_idx, BLOCK_C: tl.constexpr):
        """gather([R, C_in], dim=1, index=[R, C_idx]) → [R, C_idx].

        out[r, j] = input[r, index[r, j]]
        One program per row.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C_idx
        idx = tl.load(index_ptr + row * C_idx + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + row * C_in + idx, mask=mask, other=0.0)
        tl.store(out_ptr + row * C_idx + offs, val, mask=mask)

    @triton.jit
    def _gather_dim0_kernel(input_ptr, index_ptr, out_ptr, C, BLOCK_C: tl.constexpr):
        """gather([R_in, C], dim=0, index=[R_idx, C]) → [R_idx, C].

        out[r, c] = input[index[r, c], c]
        One program per row of index/out.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        idx = tl.load(index_ptr + row * C + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(input_ptr + idx * C + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row * C + offs, val, mask=mask)

    @triton.jit
    def _scatter_add_dim1_kernel(out_ptr, index_ptr, src_ptr, C_out, C_idx, BLOCK_C: tl.constexpr):
        """scatter_add into dim=1.

        out[r, index[r, k]] += src[r, k]
        One program per row.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C_idx
        idx = tl.load(index_ptr + row * C_idx + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + row * C_idx + offs, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + row * C_out + idx, val, mask=mask)

    @triton.jit
    def _scatter_add_dim0_kernel(out_ptr, index_ptr, src_ptr, C, BLOCK_C: tl.constexpr):
        """scatter_add into dim=0.

        out[index[r, c], c] += src[r, c]
        One program per row of src/index.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_C)
        mask = offs < C
        idx = tl.load(index_ptr + row * C + offs, mask=mask, other=0).to(tl.int64)
        val = tl.load(src_ptr + row * C + offs, mask=mask, other=0.0)
        tl.atomic_add(out_ptr + idx * C + offs, val, mask=mask)

    @triton.jit
    def _advanced_index_dim0_kernel(
        input_ptr,
        index_ptr,
        out_ptr,
        TAIL,
        N,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        index_offset = offs // TAIL
        tail_offset = offs % TAIL
        row = tl.load(index_ptr + index_offset, mask=mask, other=0).to(tl.int64)
        value = tl.load(
            input_ptr + row * TAIL + tail_offset,
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + offs, value, mask=mask)

    @triton.jit
    def _advanced_index_put_dim0_kernel(
        out_ptr,
        index_ptr,
        values_ptr,
        TAIL,
        N,
        ACCUMULATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        index_offset = offs // TAIL
        tail_offset = offs % TAIL
        row = tl.load(index_ptr + index_offset, mask=mask, other=0).to(tl.int64)
        value = tl.load(values_ptr + offs, mask=mask, other=0.0)
        target = out_ptr + row * TAIL + tail_offset
        if ACCUMULATE:
            tl.atomic_add(target, value, mask=mask)
        else:
            tl.store(target, value, mask=mask)

    @triton.jit
    def _fill_kernel(out_ptr, value, N, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(out_ptr + offs, value, mask=mask)

    @triton.jit
    def _cat2_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        A_AXIS,
        B_AXIS,
        INNER,
        N,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        out_axis = A_AXIS + B_AXIS
        inner_offset = offs % INNER
        axis_offset = (offs // INNER) % out_axis
        outer_offset = offs // (INNER * out_axis)
        from_a = axis_offset < A_AXIS
        a_offset = (
            outer_offset * A_AXIS * INNER
            + axis_offset * INNER
            + inner_offset
        )
        b_offset = (
            outer_offset * B_AXIS * INNER
            + (axis_offset - A_AXIS) * INNER
            + inner_offset
        )
        value = tl.where(
            from_a,
            tl.load(a_ptr + a_offset, mask=mask & from_a, other=0.0),
            tl.load(b_ptr + b_offset, mask=mask & ~from_a, other=0.0),
        )
        tl.store(out_ptr + offs, value, mask=mask)


def gather(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    actual_dim = int(dim) % input.ndim

    if HAS_TRITON and input.ndim == 1 and index.ndim == 1 and input.is_cuda:
        input = input.contiguous()
        index = index.contiguous()
        out = torch.empty(index.shape, device=input.device, dtype=input.dtype)

        BLOCK = min(_next_pow2(index.numel()), 1024)
        grid = (triton.cdiv(index.numel(), BLOCK),)
        _gather_1d_kernel[grid](input, index, out, index.numel(), BLOCK=BLOCK)
        return out

    if not HAS_TRITON or input.ndim != 2 or index.ndim != 2 or not input.is_cuda:
        return torch.gather(input, actual_dim, index)

    input = input.contiguous()
    index = index.contiguous()
    out = torch.empty(index.shape, device=input.device, dtype=input.dtype)

    if actual_dim == 1:
        R_idx, C_idx = index.shape
        _, C_in = input.shape
        BLOCK_C = min(_next_pow2(C_idx), 65536)
        _gather_dim1_kernel[(R_idx,)](input, index, out, C_in, C_idx, BLOCK_C=BLOCK_C)
    else:
        R_idx, C = index.shape
        BLOCK_C = min(_next_pow2(C), 65536)
        _gather_dim0_kernel[(R_idx,)](input, index, out, C, BLOCK_C=BLOCK_C)

    return out


def scatter_add(
    self_t: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    actual_dim = int(dim) % self_t.ndim

    if HAS_TRITON and self_t.ndim == 1 and index.ndim == 1 and src.ndim == 1 and self_t.is_cuda:
        out = self_t.clone().contiguous()
        index = index.contiguous()
        src = src.contiguous()

        BLOCK = min(_next_pow2(index.numel()), 1024)
        grid = (triton.cdiv(index.numel(), BLOCK),)
        _scatter_add_1d_kernel[grid](out, index, src, index.numel(), BLOCK=BLOCK)
        return out

    if not HAS_TRITON or self_t.ndim != 2 or index.ndim != 2 or not self_t.is_cuda:
        out = self_t.clone()
        out.scatter_add_(actual_dim, index, src)
        return out

    out = self_t.clone().contiguous()
    index = index.contiguous()
    src = src.contiguous()

    if actual_dim == 1:
        R, C_out = out.shape
        _, C_idx = index.shape
        BLOCK_C = min(_next_pow2(C_idx), 65536)
        _scatter_add_dim1_kernel[(R,)](out, index, src, C_out, C_idx, BLOCK_C=BLOCK_C)
    else:
        R_src, C = index.shape
        BLOCK_C = min(_next_pow2(C), 65536)
        _scatter_add_dim0_kernel[(R_src,)](out, index, src, C, BLOCK_C=BLOCK_C)

    return out


def advanced_index_dim0(input: torch.Tensor, indices) -> torch.Tensor:
    """Implement ``aten.index.Tensor(input, (index,))`` for dim-0 indexing."""
    if not isinstance(indices, (tuple, list)) or len(indices) != 1:
        raise ValueError("only one dim-0 advanced index tensor is supported")
    index = indices[0]
    if not torch.is_tensor(index):
        raise TypeError("advanced index must be a tensor")
    input, index = input.contiguous(), index.contiguous()
    tail_shape = tuple(input.shape[1:])
    tail = input.numel() // input.shape[0]
    output = torch.empty(
        (*index.shape, *tail_shape),
        device=input.device,
        dtype=input.dtype,
    )
    block = 256
    _advanced_index_dim0_kernel[(triton.cdiv(output.numel(), block),)](
        input,
        index,
        output,
        tail,
        output.numel(),
        BLOCK=block,
    )
    return output


def advanced_index_put_dim0(
    base: torch.Tensor,
    indices,
    values: torch.Tensor,
    *,
    accumulate: bool,
) -> torch.Tensor:
    """Implement dim-0 ``aten.index_put`` with optional atomic accumulation."""
    if not isinstance(indices, (tuple, list)) or len(indices) != 1:
        raise ValueError("only one dim-0 advanced index tensor is supported")
    index = indices[0]
    base, index, values = (
        base.contiguous(),
        index.contiguous(),
        values.contiguous(),
    )
    tail = base.numel() // base.shape[0]
    expected = index.numel() * tail
    if values.numel() != expected:
        raise ValueError("index_put values do not match indexed output shape")
    output = base.clone()
    block = 256
    _advanced_index_put_dim0_kernel[(triton.cdiv(expected, block),)](
        output,
        index,
        values,
        tail,
        expected,
        ACCUMULATE=bool(accumulate),
        BLOCK=block,
    )
    return output


def full(shape, fill_value, dtype):
    shape = tuple(int(size) for size in shape)
    output = torch.empty(shape, device="cuda", dtype=dtype)
    block = 256
    _fill_kernel[(triton.cdiv(output.numel(), block),)](
        output,
        float(fill_value),
        output.numel(),
        BLOCK=block,
    )
    return output


def split_with_sizes(a: torch.Tensor, sizes, dim: int):
    return tuple(torch.split(a, tuple(int(size) for size in sizes), dim=int(dim)))


def cat2(tensors, dim: int):
    if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
        raise ValueError("cat2 supports exactly two tensors")
    a, b = tensors[0].contiguous(), tensors[1].contiguous()
    actual_dim = int(dim) % a.ndim
    if a.ndim != b.ndim or any(
        a.shape[index] != b.shape[index]
        for index in range(a.ndim)
        if index != actual_dim
    ):
        raise ValueError("cat2 inputs must match outside the concatenated dim")
    output_shape = list(a.shape)
    output_shape[actual_dim] += b.shape[actual_dim]
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)
    inner = 1
    for size in a.shape[actual_dim + 1 :]:
        inner *= size
    block = 256
    _cat2_kernel[(triton.cdiv(output.numel(), block),)](
        a,
        b,
        output,
        a.shape[actual_dim],
        b.shape[actual_dim],
        inner,
        output.numel(),
        BLOCK=block,
    )
    return output
