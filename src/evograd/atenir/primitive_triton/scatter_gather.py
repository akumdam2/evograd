"""Triton kernels for gather/scatter and dynamic tensor assembly.

Generic flattened-coordinate kernels cover arbitrary ranks and axes.

gather(input, dim, index)      -- indexed read
scatter_add(self, dim, index, src) -- indexed atomic accumulate into clone of self
cat(tensors, dim)              -- aten.cat, any number of strided inputs
advanced_index(input, indices) -- aten.index.Tensor, any number of index tensors
advanced_index_put(...)        -- aten.index_put, deterministic, in ATen's summation order
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
    def _offset_of(linear, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
        """The offset, under STRIDES, of the element at row-major position
        ``linear`` of SHAPE (scalar or block)."""
        rest = linear.to(tl.int64)
        offset = rest * 0
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (rest % SHAPE[dim]) * STRIDES[dim]
            rest = rest // SHAPE[dim]
        return offset

    @triton.jit
    def _copy_into_kernel(SRC, DST, N, DST_OFFSET, SHAPE: tl.constexpr,
                          SRC_STRIDES: tl.constexpr, DST_STRIDES: tl.constexpr,
                          BLOCK: tl.constexpr):
        """DST[DST_OFFSET + ...] = SRC, both strided, element for element in
        DST's dtype (a cast where the dtypes differ)."""
        linear = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = linear < N
        value = tl.load(SRC + _offset_of(linear, SHAPE, SRC_STRIDES), mask)
        tl.store(DST + DST_OFFSET + _offset_of(linear, SHAPE, DST_STRIDES),
                 value.to(DST.dtype.element_ty), mask)

    @triton.jit
    def _index_offsets_kernel(INDEX, OFFSETS, N, SIZE, STRIDE, SHAPE: tl.constexpr,
                              INDEX_STRIDES: tl.constexpr, FIRST: tl.constexpr,
                              BLOCK: tl.constexpr):
        """OFFSETS[b] += the index at position b of the broadcast index shape
        SHAPE, wrapped once if negative, times STRIDE. An index out of range
        asserts (under debug) and marks the position -1 for good."""
        position = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = position < N
        index = tl.load(INDEX + _offset_of(position, SHAPE, INDEX_STRIDES), mask, other=0)
        index = index.to(tl.int64)
        valid = (index >= -SIZE) & (index < SIZE)
        tl.device_assert((~mask) | valid, 'index out of bounds')
        offset = tl.where(index < 0, index + SIZE, index) * STRIDE
        if not FIRST:
            earlier = tl.load(OFFSETS + position, mask, other=0)
            valid = valid & (earlier >= 0)
            offset += earlier
        tl.store(OFFSETS + position, tl.where(valid, offset, -1), mask)

    @triton.jit
    def _index_gather_kernel(SRC, OFFSETS, OUT, N, SHAPE: tl.constexpr,
                             SRC_STRIDES: tl.constexpr, POSITION_STRIDES: tl.constexpr,
                             BLOCK: tl.constexpr):
        """OUT (contiguous, of SHAPE) at o = SRC at o's plain coordinates (their
        strides in SRC_STRIDES, 0 on the index dims) plus the offset of o's index
        position (its row-major strides over the index dims in POSITION_STRIDES)."""
        linear = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = linear < N
        offset = tl.load(OFFSETS + _offset_of(linear, SHAPE, POSITION_STRIDES), mask, other=-1)
        source = _offset_of(linear, SHAPE, SRC_STRIDES) + offset
        tl.store(OUT + linear, tl.load(SRC + source, mask & (offset >= 0), other=0), mask)

    @triton.jit
    def _index_put_hits(KEYS, start, N, key, BLOCK_N: tl.constexpr):
        """The index positions start + [0, BLOCK_N) and which of them name ``key``."""
        peers = start + tl.arange(0, BLOCK_N)
        return peers, (peers < N) & (tl.load(KEYS + peers, peers < N, other=-1) == key)

    @triton.jit
    def _halve_lanes(lanes, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        """[2 * WIDTH, BLOCK] -> [WIDTH, BLOCK]: lane l plus lane l + WIDTH, a
        warp's shuffle-down step."""
        low, high = tl.split(tl.permute(tl.reshape(lanes, (2, WIDTH, BLOCK)), (1, 2, 0)))
        return low + high

    @triton.jit
    def _index_put_value(VALUES, outer, j, mask, SHAPE: tl.constexpr,
                         POSITION_STRIDES: tl.constexpr, OPMATH: tl.constexpr):
        """Index position j's values over one block of plain coordinates."""
        position = _offset_of(j, SHAPE, POSITION_STRIDES)
        return tl.load(VALUES + outer + position, mask, other=0).to(OPMATH)

    @triton.jit
    def _index_put_kernel(OUT, VALUES, KEYS, N, M, OUTER_SHAPE: tl.constexpr,
                          OUT_STRIDES: tl.constexpr, VALUE_STRIDES: tl.constexpr,
                          SHAPE: tl.constexpr, POSITION_STRIDES: tl.constexpr,
                          ACCUMULATE: tl.constexpr, ORDER: tl.constexpr,
                          OPMATH: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK: tl.constexpr):
        """One target's update over one block of its M plain coordinates, made
        by the first of the N index positions naming it (KEYS: each position's
        target offset in OUT, -1 if out of range), its values in position order.

        Without ACCUMULATE the last position's value is written. With it, ORDER
        is the summation order of ATen's sort-based CUDA index_put (a stable
        sort: position order within a target) or of its serial CPU kernel:
        0  each value added to the target in turn, rounded to the target's
           dtype each time (CUDA with more than 32 plain elements per index
           position, and CPU);
        1  the values summed in OPMATH from a zero, the sum then added to the
           target (CUDA with 2 to 32 plain elements per index position);
        2  as 1, except that a target named n >= 32 times splits its first
           32 * (n // 32) values across 32 lanes (value i to lane i % 32), each
           lane summed from a zero, the lanes reduced by halving (lane l plus
           lane l + 16, then l + 8, ...), the remaining values added after
           (CUDA with one plain element per index position)."""
        b = tl.program_id(0)
        outer = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = outer < M
        key = tl.load(KEYS + b)
        earlier = 0
        for start in range(0, b, BLOCK_N):
            peers, hit = _index_put_hits(KEYS, start, b, key, BLOCK_N)
            earlier += tl.sum(hit.to(tl.int32))
        if (key >= 0) & (earlier == 0):
            target = OUT + key + _offset_of(outer, OUTER_SHAPE, OUT_STRIDES)
            values = _offset_of(outer, OUTER_SHAPE, VALUE_STRIDES)
            if ACCUMULATE:
                base = tl.load(target, mask, other=0).to(OPMATH)
                total = tl.zeros((BLOCK,), OPMATH)
                laned = 0
                if ORDER == 2:
                    count = 0
                    for start in range(b, N, BLOCK_N):
                        peers, hit = _index_put_hits(KEYS, start, N, key, BLOCK_N)
                        count += tl.sum(hit.to(tl.int32))
                    laned = (count // 32) * 32
                    lane = tl.arange(0, 32)
                    lanes = tl.zeros((32, BLOCK), OPMATH)
                    seen = 0
                    for start in range(b, N, BLOCK_N):
                        peers, hit = _index_put_hits(KEYS, start, N, key, BLOCK_N)
                        hit = hit & (seen + tl.cumsum(hit.to(tl.int32), 0) <= laned)
                        for _ in range(tl.sum(hit.to(tl.int32))):
                            j = tl.min(tl.where(hit, peers, N))
                            hit = hit & (peers != j)
                            value = _index_put_value(VALUES, values, j, mask, SHAPE,
                                                     POSITION_STRIDES, OPMATH)
                            lanes = tl.where(lane[:, None] == seen % 32,
                                             lanes + value[None, :], lanes)
                            seen += 1
                    lanes = _halve_lanes(lanes, 16, BLOCK)
                    lanes = _halve_lanes(lanes, 8, BLOCK)
                    lanes = _halve_lanes(lanes, 4, BLOCK)
                    lanes = _halve_lanes(lanes, 2, BLOCK)
                    total = tl.reshape(_halve_lanes(lanes, 1, BLOCK), (BLOCK,))
                running = base
                seen = 0
                for start in range(b, N, BLOCK_N):
                    peers, found = _index_put_hits(KEYS, start, N, key, BLOCK_N)
                    hit = found & (seen + tl.cumsum(found.to(tl.int32), 0) > laned)
                    seen += tl.sum(found.to(tl.int32))
                    for _ in range(tl.sum(hit.to(tl.int32))):
                        j = tl.min(tl.where(hit, peers, N))
                        hit = hit & (peers != j)
                        value = _index_put_value(VALUES, values, j, mask, SHAPE,
                                                 POSITION_STRIDES, OPMATH)
                        if ORDER == 0:
                            running = (running + value).to(OUT.dtype.element_ty).to(OPMATH)
                        else:
                            total += value
                if ORDER == 0:
                    tl.store(target, running.to(OUT.dtype.element_ty), mask)
                else:
                    tl.store(target, (base + total).to(OUT.dtype.element_ty), mask)
            else:
                last = b
                for start in range(b, N, BLOCK_N):
                    peers, hit = _index_put_hits(KEYS, start, N, key, BLOCK_N)
                    last = tl.maximum(last, tl.max(tl.where(hit, peers, -1)))
                position = _offset_of(last, SHAPE, POSITION_STRIDES)
                tl.store(target, tl.load(VALUES + values + position, mask), mask)

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


def _row_major_strides(shape) -> list[int]:
    strides, step = [], 1
    for size in reversed(list(shape)):
        strides.append(step)
        step *= int(size)
    return strides[::-1]


def _broadcast_strides(t: torch.Tensor, shape) -> tuple[int, ...]:
    """``t``'s element strides read as broadcast to ``shape`` (0 on new or
    expanded dims), as ``t.expand(shape)`` would have them."""
    lead = len(shape) - t.ndim
    if lead < 0:
        raise ValueError(f'shape mismatch: {tuple(t.shape)} cannot be broadcast to {tuple(shape)}')
    strides = []
    for dim, size in enumerate(shape):
        if dim < lead or (t.shape[dim - lead] == 1 and size != 1):
            strides.append(0)
        elif t.shape[dim - lead] == size:
            strides.append(t.stride(dim - lead))
        else:
            raise ValueError(f'shape mismatch: {tuple(t.shape)} cannot be broadcast to {tuple(shape)}')
    return tuple(strides)


def _index_plan(self_t: torch.Tensor, indices):
    """ATen's advanced-indexing layout: the indexed dims, the index tensors'
    broadcast shape, and the plain dims placed before and after it in the
    result (all after it when the indexed dims are not adjacent)."""
    if not isinstance(indices, (tuple, list)):
        raise TypeError('advanced indices must be a list of tensors or None')
    if len(indices) > self_t.ndim:
        raise IndexError(f'too many indices for tensor of dimension {self_t.ndim} (got {len(indices)})')
    dims = [dim for dim, index in enumerate(indices) if index is not None]
    if not dims:
        raise NotImplementedError('advanced indexing without an index tensor')
    for dim in dims:
        index = indices[dim]
        if not torch.is_tensor(index):
            raise TypeError('advanced index must be a tensor')
        if index.dtype not in (torch.int64, torch.int32):
            # A bool or uint8 mask selects a data-dependent number of elements.
            raise NotImplementedError(f'handwritten advanced indexing takes int64 or int32 '
                                      f'indices, not {index.dtype}')
        if index.device != self_t.device:
            raise ValueError('index and input must share a device')
    shape = tuple(torch.broadcast_shapes(*(indices[dim].shape for dim in dims)))
    plain = [dim for dim in range(self_t.ndim) if dim not in dims]
    if dims == list(range(dims[0], dims[-1] + 1)):
        before = [dim for dim in plain if dim < dims[0]]
        after = [dim for dim in plain if dim > dims[-1]]
    else:
        before, after = [], plain
    return dims, shape, before, after


def _index_offsets(indices, dims, shape, sizes, strides, device):
    """Each broadcast index position's element offset under ``strides`` of the
    indexed dims (-1 where an index is out of range), and how many positions."""
    count = 1
    for size in shape:
        count *= size
    offsets = torch.empty(shape, device=device, dtype=torch.int64)
    if count:
        for first, dim in enumerate(dims):
            index = indices[dim]
            _index_offsets_kernel[(triton.cdiv(count, 1024),)](
                index, offsets, count, sizes[dim], strides[dim], SHAPE=shape,
                INDEX_STRIDES=_broadcast_strides(index, shape), FIRST=first == 0, BLOCK=1024,
            )
    return offsets, count


def advanced_index(input: torch.Tensor, indices) -> torch.Tensor:
    """``aten.index.Tensor(input, indices)``: any number of int64/int32 index
    tensors (broadcast together, negative indices wrapped) on any dims, None for
    a dim taken whole, any input strides; a contiguous result laid out as ATen's
    (the index shape in place of adjacent indexed dims, else in front)."""
    dims, shape, before, after = _index_plan(input, indices)
    offsets, count = _index_offsets(indices, dims, shape, input.shape, input.stride(), input.device)
    rank = len(shape)
    out_shape = (*(input.shape[d] for d in before), *shape, *(input.shape[d] for d in after))
    source = (*(input.stride(d) for d in before), *([0] * rank), *(input.stride(d) for d in after))
    position = (*([0] * len(before)), *_row_major_strides(shape), *([0] * len(after)))
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    if output.numel():
        _index_gather_kernel[(triton.cdiv(output.numel(), 1024),)](
            input, offsets, output, output.numel(), SHAPE=out_shape, SRC_STRIDES=source,
            POSITION_STRIDES=position, BLOCK=1024,
        )
    return output


#: index_put's summation orders (see _index_put_kernel).
_INDEX_PUT_IN_TURN, _INDEX_PUT_FROM_ZERO, _INDEX_PUT_LANES = 0, 1, 2


def _index_put_order(device: torch.device, slice_size: int) -> int:
    """How ATen's ``index_put(accumulate=True)`` on ``device`` sums a target's
    values, given the plain elements per index position: CUDA's sort-based
    kernels by slice size (indexing_backward_kernel, _small_stride, _stride_1),
    CPU's serial kernel in turn."""
    if device.type != 'cuda' or slice_size > 32:
        return _INDEX_PUT_IN_TURN
    return _INDEX_PUT_FROM_ZERO if slice_size > 1 else _INDEX_PUT_LANES


def advanced_index_put(base: torch.Tensor, indices, values: torch.Tensor,
                       accumulate: bool = False) -> torch.Tensor:
    """``aten.index_put(base, indices, values, accumulate)``, deterministically
    and without atomics: :func:`advanced_index`'s indexing, ``values`` broadcast
    to its result. A target named more than once takes the last value, or with
    ``accumulate`` the sum of its values in the order ATen's kernel on that
    device adds them: the sort-based CUDA kernel (whose rounding depends on the
    plain elements per index position) or the serial CPU one."""
    dims, shape, before, after = _index_plan(base, indices)
    if values.dtype != base.dtype or values.device != base.device:
        raise ValueError('index_put values must match the destination dtype and device')
    if accumulate and not base.is_floating_point():
        raise NotImplementedError(f'handwritten index_put accumulates floating types, not {base.dtype}')
    output = torch.empty_like(base)
    if base.numel():
        _copy_into_kernel[(triton.cdiv(base.numel(), 1024),)](
            base, output, base.numel(), 0, SHAPE=tuple(base.shape), SRC_STRIDES=tuple(base.stride()),
            DST_STRIDES=tuple(output.stride()), BLOCK=1024,
        )
    keys, count = _index_offsets(indices, dims, shape, base.shape, output.stride(), base.device)
    outer = before + after
    rank = len(before)
    result = (*(base.shape[d] for d in before), *shape, *(base.shape[d] for d in after))
    value_strides = _broadcast_strides(values, result)
    plain = 1
    for dim in outer:
        plain *= base.shape[dim]
    if not count or not plain:
        return output
    slice_size = 1
    for dim in after:
        slice_size *= base.shape[dim]
    order = _index_put_order(output.device, slice_size)
    # The 32 lanes of the last order are kept per plain element.
    block = min(_next_pow2(plain), 128 if order == _INDEX_PUT_LANES else 1024)
    opmath = {torch.float16: tl.float32, torch.bfloat16: tl.float32,
              torch.float32: tl.float32, torch.float64: tl.float64}.get(base.dtype, tl.float32)
    _index_put_kernel[(count, triton.cdiv(plain, block))](
        output, values, keys, count, plain, OUTER_SHAPE=tuple(base.shape[d] for d in outer),
        OUT_STRIDES=tuple(output.stride(d) for d in outer),
        VALUE_STRIDES=value_strides[:rank] + value_strides[rank + len(shape):],
        SHAPE=shape, POSITION_STRIDES=value_strides[rank:rank + len(shape)],
        ACCUMULATE=bool(accumulate), ORDER=order, OPMATH=opmath, BLOCK_N=256, BLOCK=block,
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


def cat(tensors, dim: int = 0):
    """``aten.cat(tensors, dim)``: any number of tensors of any strides along any
    dim, their dtypes promoted, 1-D empty tensors skipped as ATen skips them; a
    contiguous result, one strided copy per input."""
    tensors = list(tensors)
    if not tensors:
        raise ValueError('torch.cat(): expected a non-empty list of Tensors')
    if any(t.ndim == 0 for t in tensors):
        raise ValueError('zero-dimensional tensor cannot be concatenated')
    kept = [t for t in tensors if not (t.ndim == 1 and t.numel() == 0)]
    first = kept[0] if kept else tensors[0]
    if not -first.ndim <= int(dim) < first.ndim:
        raise IndexError(f'torch.cat(): dimension {dim} out of range')
    axis = int(dim) % first.ndim
    for t in kept:
        if t.ndim != first.ndim or any(t.shape[d] != first.shape[d]
                                       for d in range(first.ndim) if d != axis):
            raise ValueError('torch.cat(): tensors must match outside the concatenated dim')
        if t.device != first.device:
            raise ValueError('torch.cat(): tensors must share a device')
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    shape = list(first.shape) if kept else [0]
    if kept:
        shape[axis] = sum(t.shape[axis] for t in kept)
    output = torch.empty(shape, device=first.device, dtype=dtype)
    start = 0
    for t in kept:
        if t.numel():
            _copy_into_kernel[(triton.cdiv(t.numel(), 1024),)](
                t, output, t.numel(), start * output.stride(axis), SHAPE=tuple(t.shape),
                SRC_STRIDES=tuple(t.stride()), DST_STRIDES=tuple(output.stride()), BLOCK=1024,
            )
        start += t.shape[axis]
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
