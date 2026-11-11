"""Triton kernels for gather/scatter and dynamic tensor assembly.

Generic flattened-coordinate kernels cover arbitrary ranks and axes.

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
    def _indexed_kernel(INPUT, INDEX, SRC, OUT, N, INDEX_SHAPE: tl.constexpr,
                        INDEX_STRIDES: tl.constexpr, DATA_STRIDES: tl.constexpr,
                        SRC_STRIDES: tl.constexpr, AXIS: tl.constexpr,
                        AXIS_SIZE: tl.constexpr, OP: tl.constexpr, VALUE,
                        BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        index = tl.load(INDEX + offsets, mask, other=0).to(tl.int64)
        target = tl.full((BLOCK,), 0, tl.int64)
        source = tl.full((BLOCK,), 0, tl.int64)
        for dim in tl.static_range(len(INDEX_SHAPE)):
            coordinate = (offsets // INDEX_STRIDES[dim]) % INDEX_SHAPE[dim]
            target += (index if dim == AXIS else coordinate) * DATA_STRIDES[dim]
            source += coordinate * SRC_STRIDES[dim]
        tl.device_assert((~mask) | ((index >= 0) & (index < AXIS_SIZE)), 'index out of bounds')
        valid = mask & (index >= 0) & (index < AXIS_SIZE)
        if OP == 'gather':
            value = tl.load(INPUT + target, valid, other=0)
            tl.store(OUT + offsets, value, mask)
        elif OP == 'scatter_add':
            value = tl.load(SRC + source, mask, other=0)
            tl.atomic_add(OUT + target, value, valid)
        elif OP == 'scatter_src':
            value = tl.load(SRC + source, mask, other=0)
            tl.store(OUT + target, value, valid)
        else:
            tl.store(OUT + target, VALUE, valid)

    @triton.jit
    def _slice_scatter_kernel(SRC, OUT, N, SRC_SHAPE: tl.constexpr,
                              SRC_STRIDES: tl.constexpr, OUT_STRIDES: tl.constexpr,
                              AXIS: tl.constexpr, START: tl.constexpr, STEP: tl.constexpr,
                              BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        target = tl.full((BLOCK,), 0, tl.int64)
        for dim in tl.static_range(len(SRC_SHAPE)):
            coordinate = (offsets // SRC_STRIDES[dim]) % SRC_SHAPE[dim]
            if dim == AXIS:
                coordinate = START + coordinate * STEP
            target += coordinate * OUT_STRIDES[dim]
        value = tl.load(SRC + offsets, mask, other=0)
        tl.store(OUT + target, value, mask)

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
    def _embedding_bag_sum_kernel(WEIGHT, INDICES, OFFSETS, OUT, OFFSET2BAG, BAG_SIZE,
                                  L, B, D, stride_w, BLOCK_D: tl.constexpr):
        """One bag's sum over one block of columns: its rows added in index
        order to a float32 zero, as ATen's EmbeddingBag_updateOutputKernel_sum_mean."""
        bag = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D
        # The first bag starts at 0 whatever offsets[0] holds; the last ends at L.
        begin = tl.load(OFFSETS + bag, mask=bag > 0, other=0).to(tl.int64)
        end = tl.load(OFFSETS + bag + 1, mask=bag + 1 < B, other=L).to(tl.int64)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for k in range(begin, end):
            row = tl.load(INDICES + k).to(tl.int64)
            acc += tl.load(WEIGHT + row * stride_w + cols, mask=mask, other=0.0).to(tl.float32)
            if tl.program_id(1) == 0:
                tl.store(OFFSET2BAG + k, bag.to(tl.int64))
        if tl.program_id(1) == 0:
            tl.store(BAG_SIZE + bag, end - begin)
        tl.store(OUT + bag * D + cols, acc.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _embedding_bag_sum_backward_kernel(GRAD, INDICES, OFFSET2BAG, OUT, L, D, stride_g,
                                           ROWS: tl.constexpr, BLOCK_L: tl.constexpr,
                                           BLOCK_D: tl.constexpr):
        """One weight row's gradient, written by the first position holding its
        index, in the order of ATen's embedding_backward_cuda_kernel: the bag
        gradients of the positions holding the index, in position order, summed
        in runs of ROWS from a float32 zero, the runs then summed in order."""
        k = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D
        row = tl.load(INDICES + k).to(tl.int64)
        earlier = 0
        for start in range(0, k, BLOCK_L):
            positions = start + tl.arange(0, BLOCK_L)
            seen = tl.load(INDICES + positions, mask=positions < k, other=0).to(tl.int64)
            earlier += tl.sum(((positions < k) & (seen == row)).to(tl.int32))
        if earlier == 0:
            total = tl.zeros([BLOCK_D], dtype=tl.float32)
            run = tl.zeros([BLOCK_D], dtype=tl.float32)
            count = 0
            for j in range(k, L):
                if tl.load(INDICES + j).to(tl.int64) == row:
                    bag = tl.load(OFFSET2BAG + j).to(tl.int64)
                    run += tl.load(GRAD + bag * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
                    count += 1
                    if count == ROWS:
                        total += run
                        run = tl.zeros([BLOCK_D], dtype=tl.float32)
                        count -= ROWS
            if count > 0:
                total += run
            tl.store(OUT + row * D + cols, total.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _histc_kernel(INPUT, OUT, N, low, high, BINS: tl.constexpr, IS_FLOAT: tl.constexpr,
                      BLOCK: tl.constexpr):
        """Count each element into one of BINS equal bins over [low, high], the
        last bin closed, as ATen's kernelHistogram1D/getBin: float input binned in
        float32, integer input in int64."""
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        if IS_FLOAT:
            value = tl.load(INPUT + offsets, mask=mask, other=0).to(tl.float32)
            inside = mask & (value >= low) & (value <= high)
            span = tl.zeros_like(value) + (high - low)
            index = tl.math.div_rn((value - low) * BINS, span).to(tl.int64)
        else:
            value = tl.load(INPUT + offsets, mask=mask, other=0).to(tl.int64)
            inside = mask & (value >= low) & (value <= high)
            index = (value - low) * BINS // (high - low)
        index = tl.where(index == BINS, BINS - 1, index)
        one = tl.full((BLOCK,), 1, OUT.dtype.element_ty)
        tl.atomic_add(OUT + index, one, mask=inside)

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


def _check_indexed(base, dim, index):
    if not HAS_TRITON or not base.is_cuda:
        raise NotImplementedError('handwritten gather/scatter requires CUDA and Triton')
    if base.ndim == 0 or not -base.ndim <= int(dim) < base.ndim:
        raise IndexError('indexed dimension out of range')
    axis = int(dim) % base.ndim
    if index.ndim != base.ndim or index.dtype != torch.int64:
        raise ValueError('index must be int64 and have the same rank as the input')
    if index.device != base.device:
        raise ValueError('index and input must share a device')
    if any(index.shape[d] > base.shape[d] for d in range(base.ndim) if d != axis):
        raise ValueError('index exceeds input shape outside the indexed dimension')
    return axis


def gather(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    axis = _check_indexed(input, dim, index)
    input, index = input.contiguous(), index.contiguous()
    output = torch.empty(index.shape, device=input.device, dtype=input.dtype)
    if index.numel():
        _indexed_kernel[(triton.cdiv(index.numel(), 256),)](
            input, index, input, output, index.numel(),
            INDEX_SHAPE=tuple(index.shape), INDEX_STRIDES=tuple(index.stride()),
            DATA_STRIDES=tuple(input.stride()), SRC_STRIDES=tuple(index.stride()),
            AXIS=axis, AXIS_SIZE=input.shape[axis], OP='gather', VALUE=0, BLOCK=256,
        )
    return output


def _scatter_tensor(self_t, dim, index, src, operation):
    axis = _check_indexed(self_t, dim, index)
    if src.ndim != index.ndim or any(a > b for a, b in zip(index.shape, src.shape)):
        raise ValueError('scatter source must cover the index shape')
    if src.dtype != self_t.dtype or src.device != self_t.device:
        raise ValueError('scatter source must match the base dtype and device')
    output, index, src = self_t.contiguous().clone(), index.contiguous(), src.contiguous()
    if index.numel():
        _indexed_kernel[(triton.cdiv(index.numel(), 256),)](
            output, index, src, output, index.numel(),
            INDEX_SHAPE=tuple(index.shape), INDEX_STRIDES=tuple(index.stride()),
            DATA_STRIDES=tuple(output.stride()), SRC_STRIDES=tuple(src.stride()),
            AXIS=axis, AXIS_SIZE=output.shape[axis], OP=operation, VALUE=0, BLOCK=256,
        )
    return output


def scatter_add(self_t: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return _scatter_tensor(self_t, dim, index, src, 'scatter_add')


def scatter_src(self_t: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return _scatter_tensor(self_t, dim, index, src, 'scatter_src')


def scatter_value(self_t: torch.Tensor, dim: int, index: torch.Tensor, value) -> torch.Tensor:
    axis = _check_indexed(self_t, dim, index)
    output, index = self_t.contiguous().clone(), index.contiguous()
    if index.numel():
        _indexed_kernel[(triton.cdiv(index.numel(), 256),)](
            output, index, output, output, index.numel(),
            INDEX_SHAPE=tuple(index.shape), INDEX_STRIDES=tuple(index.stride()),
            DATA_STRIDES=tuple(output.stride()), SRC_STRIDES=tuple(index.stride()),
            AXIS=axis, AXIS_SIZE=output.shape[axis], OP='scatter_value', VALUE=value, BLOCK=256,
        )
    return output


def slice_scatter(base: torch.Tensor, src: torch.Tensor, dim=0, start=None, end=None, step=1):
    if not HAS_TRITON or not base.is_cuda:
        raise NotImplementedError('handwritten slice_scatter requires CUDA and Triton')
    if base.ndim == 0 or not -base.ndim <= int(dim) < base.ndim:
        raise IndexError('slice dimension out of range')
    if step <= 0:
        raise ValueError('slice_scatter step must be positive')
    axis = int(dim) % base.ndim
    begin, stop, stride = slice(start, end, step).indices(base.shape[axis])
    shape = list(base.shape)
    shape[axis] = len(range(begin, stop, stride))
    if tuple(src.shape) != tuple(shape) or src.dtype != base.dtype or src.device != base.device:
        raise ValueError('slice_scatter source must match the selected slice')
    output, src = base.contiguous().clone(), src.contiguous()
    if src.numel():
        _slice_scatter_kernel[(triton.cdiv(src.numel(), 256),)](
            src, output, src.numel(), SRC_SHAPE=tuple(src.shape),
            SRC_STRIDES=tuple(src.stride()), OUT_STRIDES=tuple(output.stride()),
            AXIS=axis, START=begin, STEP=stride, BLOCK=256,
        )
    return output


def select_scatter(base: torch.Tensor, src: torch.Tensor, dim: int, index: int):
    """Write one selected slice without introducing framework mathematics."""
    if base.ndim == 0 or not -base.ndim <= int(dim) < base.ndim:
        raise IndexError('select dimension out of range')
    axis = int(dim) % base.ndim
    selected = int(index)
    if selected < 0:
        selected += base.shape[axis]
    if not 0 <= selected < base.shape[axis]:
        raise IndexError('select index out of range')
    return slice_scatter(base, src.unsqueeze(axis), axis, selected, selected + 1, 1)


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


def full(shape, fill_value, dtype, device=None):
    shape = tuple(int(size) for size in shape)
    output = torch.empty(shape, device=device or "cuda", dtype=dtype)
    if not HAS_TRITON or not output.is_cuda:
        raise NotImplementedError('handwritten full requires CUDA and Triton')
    if not output.numel():
        return output
    block = 256
    _fill_kernel[(triton.cdiv(output.numel(), block),)](
        output,
        fill_value,
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


#: Rows per partial sum in ATen's CUDA embedding backward (NROWS_PER_THREAD).
_EMBEDDING_BACKWARD_ROWS = 10


def embedding_bag_sum(weight: torch.Tensor, indices: torch.Tensor, offsets: torch.Tensor):
    """``aten._embedding_bag`` in sum mode (no per-sample weights, no padding
    index, ``offsets`` holding each bag's start): ``(output, offset2bag,
    bag_size, max_indices)`` with the layouts ATen's CUDA kernel returns."""
    if weight.ndim != 2 or indices.ndim != 1 or offsets.ndim != 1:
        raise ValueError('embedding_bag takes a 2-D weight and 1-D indices and offsets')
    weight, indices, offsets = weight.contiguous(), indices.contiguous(), offsets.contiguous()
    bags, width = offsets.shape[0], weight.shape[1]
    output = torch.empty((bags, width), device=weight.device, dtype=weight.dtype)
    offset2bag = torch.empty(indices.shape, device=indices.device, dtype=indices.dtype)
    bag_size = torch.empty(offsets.shape, device=indices.device, dtype=indices.dtype)
    max_indices = torch.empty((0,), device=indices.device, dtype=indices.dtype)
    if bags and width:
        block = min(_next_pow2(width), 1024)
        _embedding_bag_sum_kernel[(bags, triton.cdiv(width, block))](
            weight, indices, offsets, output, offset2bag, bag_size,
            indices.shape[0], bags, width, weight.stride(0), BLOCK_D=block,
        )
    return output, offset2bag, bag_size, max_indices


def embedding_bag_sum_backward(grad: torch.Tensor, indices: torch.Tensor,
                               offset2bag: torch.Tensor, num_weights: int):
    """The dense weight gradient of :func:`embedding_bag_sum`: each bag's output
    gradient added to the rows its indices name, deterministically and in the
    order ATen's CUDA backward adds them."""
    grad, indices, offset2bag = grad.contiguous(), indices.contiguous(), offset2bag.contiguous()
    width = grad.shape[1]
    output = torch.empty((int(num_weights), width), device=grad.device, dtype=grad.dtype)
    if output.numel():
        _fill_kernel[(triton.cdiv(output.numel(), 1024),)](output, 0.0, output.numel(), BLOCK=1024)
    if indices.numel() and width:
        block = min(_next_pow2(width), 1024)
        _embedding_bag_sum_backward_kernel[(indices.numel(), triton.cdiv(width, block))](
            grad, indices, offset2bag, output, indices.numel(), width, grad.stride(0),
            ROWS=_EMBEDDING_BACKWARD_ROWS, BLOCK_L=256, BLOCK_D=block,
        )
    return output


def histc(input: torch.Tensor, bins: int, low, high) -> torch.Tensor:
    """``aten.histc`` with a fixed range ``low < high``: counts in ``input``'s dtype."""
    if not low < high:
        raise ValueError('handwritten histc needs a fixed range with min < max')
    floating = input.is_floating_point()
    input = input.contiguous()
    output = torch.empty((int(bins),), device=input.device, dtype=input.dtype)
    _fill_kernel[(triton.cdiv(int(bins), 1024),)](output, 0, int(bins), BLOCK=1024)
    if input.numel():
        _histc_kernel[(triton.cdiv(input.numel(), 1024),)](
            input, output, input.numel(),
            float(low) if floating else int(low), float(high) if floating else int(high),
            BINS=int(bins), IS_FLOAT=floating, BLOCK=1024,
        )
    return output
