"""Shape-generic handwritten Triton reductions.

Axes are flattened with metadata views/copies; numerical work stays in Triton.
Long rows are processed in chunks rather than silently truncating at 65536.
"""
from __future__ import annotations

from typing import Callable
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
    def _nan_maximum(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _nan_minimum(a, b):
        return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def _reduce_rows_kernel(X, Y, C, OP: tl.constexpr, CORRECTION: tl.constexpr,
                            BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        if X.dtype.element_ty == tl.float64:
            acc = tl.full((BLOCK,), 0, tl.float64)
        elif X.dtype.element_ty.is_floating():
            acc = tl.full((BLOCK,), 0, tl.float32)
        else:
            acc = tl.full((BLOCK,), 0, tl.int64)
        if OP == 'amax':
            acc = tl.full((BLOCK,), float('-inf'), acc.dtype)
        elif OP == 'amin':
            acc = tl.full((BLOCK,), float('inf'), acc.dtype)
        elif OP == 'prod':
            acc = tl.full((BLOCK,), 1, acc.dtype)
        for start in range(0, C, BLOCK):
            cols = start + lane
            if OP == 'amax':
                value = tl.load(X + row * C + cols, cols < C, other=float('-inf')).to(acc.dtype)
                acc = _nan_maximum(acc, value)
            elif OP == 'amin':
                value = tl.load(X + row * C + cols, cols < C, other=float('inf')).to(acc.dtype)
                acc = _nan_minimum(acc, value)
            elif OP == 'prod':
                value = tl.load(X + row * C + cols, cols < C, other=1).to(acc.dtype)
                acc *= value
            else:
                value = tl.load(X + row * C + cols, cols < C, other=0).to(acc.dtype)
                if OP == 'any':
                    acc = tl.maximum(acc, (value != 0).to(acc.dtype))
                else:
                    acc += value
        if OP == 'any':
            result = tl.max(acc, 0)
        elif OP == 'amax':
            result = tl.reduce(acc, 0, _nan_maximum)
        elif OP == 'amin':
            result = tl.reduce(acc, 0, _nan_minimum)
        elif OP == 'prod':
            result = tl.reduce(acc, 0, _product_combine)
        else:
            result = tl.sum(acc, 0)
        if OP == 'mean' or OP == 'var':
            mean = result / C
            if OP == 'mean':
                result = mean
            else:
                squared = tl.full((BLOCK,), 0, acc.dtype)
                for start in range(0, C, BLOCK):
                    cols = start + lane
                    value = tl.load(X + row * C + cols, cols < C, other=0).to(acc.dtype)
                    delta = value - mean
                    squared += tl.where(cols < C, delta * delta, 0)
                result = tl.sum(squared, 0) / tl.maximum(C - CORRECTION, 0)
        tl.store(Y + row, result)

    @triton.jit
    def _product_combine(a, b):
        return a * b

    @triton.jit
    def _softmax_rows_kernel(X, Y, C, LOG: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        maximum = tl.full((BLOCK,), float('-inf'), tl.float32)
        for start in range(0, C, BLOCK):
            cols = start + lane
            values = tl.load(X + row * C + cols, cols < C, other=float('-inf')).to(tl.float32)
            maximum = _nan_maximum(maximum, values)
        row_max = tl.reduce(maximum, 0, _nan_maximum)
        total = tl.full((BLOCK,), 0, tl.float32)
        for start in range(0, C, BLOCK):
            cols = start + lane
            values = tl.load(X + row * C + cols, cols < C, other=float('-inf')).to(tl.float32)
            total += tl.where(cols < C, tl.exp(values - row_max), 0)
        denominator = tl.sum(total, 0)
        for start in range(0, C, BLOCK):
            cols = start + lane
            values = tl.load(X + row * C + cols, cols < C, other=0).to(tl.float32)
            if LOG:
                output = values - row_max - tl.log(denominator)
            else:
                output = tl.exp(values - row_max) / denominator
            tl.store(Y + row * C + cols, output, cols < C)

    @triton.jit
    def _extreme_rows_kernel(X, Y, I, C, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        best = tl.full((), float('-inf') if MAXIMUM else float('inf'), tl.float32)
        best_index = tl.full((), 0x7fffffffffffffff, tl.int64)
        for start in range(0, C, BLOCK):
            cols = start + lane
            values = tl.load(X + row * C + cols, cols < C,
                             other=float('-inf') if MAXIMUM else float('inf')).to(tl.float32)
            value = tl.reduce(values, 0, _nan_maximum) if MAXIMUM else tl.reduce(values, 0, _nan_minimum)
            equal = (values == value) | ((values != values) & (value != value))
            index = tl.min(tl.where((cols < C) & equal, cols.to(tl.int64), 0x7fffffffffffffff), 0)
            improves = value > best if MAXIMUM else value < best
            improves = improves | ((value != value) & (best == best))
            equal_best = (value == best) | ((value != value) & (best != best))
            choose = improves | (equal_best & (index < best_index))
            best, best_index = tl.where(choose, value, best), tl.where(choose, index, best_index)
        tl.store(Y + row, best)
        tl.store(I + row, best_index)

    @triton.jit
    def _cumsum_rows_kernel(X, Y, C, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        if X.dtype.element_ty.is_floating():
            running = tl.full((), 0, tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32)
        else:
            running = tl.full((), 0, tl.int64)
        for start in range(0, C, BLOCK):
            cols = start + lane
            values = tl.load(X + row * C + cols, cols < C, other=0).to(running.dtype)
            result = tl.cumsum(values, 0) + running
            tl.store(Y + row * C + cols, result, cols < C)
            running += tl.sum(values, 0)

    @triton.jit
    def _sort_rows_kernel(X, Y, I, C, DESCENDING: tl.constexpr, LOG_BLOCK: tl.constexpr,
                          BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        values = tl.load(X + row * C + lane, lane < C,
                         other=float('-inf') if DESCENDING else float('inf')).to(tl.float32)
        indices = lane
        for stage in tl.static_range(1, LOG_BLOCK + 1):
            for bit in tl.static_range(stage - 1, -1, -1):
                peer = lane ^ (1 << bit)
                other = tl.gather(values, peer, 0)
                other_index = tl.gather(indices, peer, 0)
                less = other > values if DESCENDING else other < values
                other_nan, this_nan = other != other, values != values
                if DESCENDING:
                    less = less | (other_nan & ~this_nan)
                else:
                    less = less | (~other_nan & this_nan)
                equal = (other == values) | (other_nan & this_nan)
                less = less | (equal & (other_index < indices))
                # Padding must sort after every real value, including NaNs
                # and infinities; a numeric +/-inf sentinel alone cannot do it.
                less = tl.where((other_index < C) != (indices < C), other_index < C, less)
                ascending = (lane & (1 << stage)) == 0
                take_less = ((lane & (1 << bit)) == 0) == ascending
                swap = less == take_less
                values = tl.where(swap, other, values)
                indices = tl.where(swap, other_index, indices)
        tl.store(Y + row * C + lane, values, lane < C)
        tl.store(I + row * C + lane, indices, lane < C)

    @triton.jit
    def _int_extremes_kernel(X, OUT, N, BLOCK: tl.constexpr):
        """OUT[0] and OUT[1]: the smallest and the largest of N integer keys, as int64."""
        lane = tl.arange(0, BLOCK)
        first = tl.load(X).to(tl.int64)
        low = tl.full((BLOCK,), 0, tl.int64) + first
        high = low
        for start in range(0, N, BLOCK):
            offsets = start + lane
            keys = tl.load(X + offsets, offsets < N, other=0).to(tl.int64)
            keys = tl.where(offsets < N, keys, first)
            low = tl.minimum(low, keys)
            high = tl.maximum(high, keys)
        tl.store(OUT, tl.min(low, 0))
        tl.store(OUT + 1, tl.max(high, 0))

    @triton.jit
    def _key_bins(ROW, cols, C, base, DESCENDING: tl.constexpr):
        """The counting-sort bin of each key in ROW[cols]: its distance from the
        smallest key (ascending) or the largest (descending); -1 past the row."""
        keys = tl.load(ROW + cols, cols < C, other=0).to(tl.int64)
        bins = base - keys if DESCENDING else keys - base
        return tl.where(cols < C, bins, -1).to(tl.int32)

    @triton.jit
    def _key_counts_kernel(X, COUNTS, C, BLOCKS, base, BINS, DESCENDING: tl.constexpr,
                           BLOCK: tl.constexpr, CHUNK: tl.constexpr):
        """COUNTS[row, bin * BLOCKS + block]: how many keys of one block of a row
        fall in each bin, every bin written."""
        row, block = tl.program_id(0) // BLOCKS, tl.program_id(0) % BLOCKS
        bins = _key_bins(X + row * C, block * BLOCK + tl.arange(0, BLOCK), C, base, DESCENDING)
        for start in range(0, BINS, CHUNK):
            ids = start + tl.arange(0, CHUNK)
            count = tl.sum((bins[:, None] == ids[None, :]).to(tl.int32), 0)
            tl.store(COUNTS + row * BINS * BLOCKS + ids * BLOCKS + block, count, ids < BINS)

    @triton.jit
    def _counting_sort_scatter_kernel(X, COUNTS, ENDS, VALUES, INDICES, C, BLOCKS, base, BINS,
                                      DESCENDING: tl.constexpr, BLOCK: tl.constexpr,
                                      CHUNK: tl.constexpr):
        """Place each key of one block of a row: where its (bin, block) starts in
        the row (ENDS, the running sum of COUNTS, minus COUNTS), plus how many
        earlier keys of its block share its bin. Equal keys keep their order."""
        row, block = tl.program_id(0) // BLOCKS, tl.program_id(0) % BLOCKS
        lane = tl.arange(0, BLOCK)
        cols = block * BLOCK + lane
        mask = cols < C
        bins = _key_bins(X + row * C, cols, C, base, DESCENDING)
        rank = tl.zeros((BLOCK,), tl.int32)
        for start in range(0, BLOCK, CHUNK):
            peers = start + tl.arange(0, CHUNK)
            peer_bins = _key_bins(X + row * C, block * BLOCK + peers, C, base, DESCENDING)
            earlier = (peer_bins[None, :] == bins[:, None]) & (peers[None, :] < lane[:, None])
            rank += tl.sum(earlier.to(tl.int32), 1)
        slot = tl.where(mask, row * BINS * BLOCKS + bins * BLOCKS + block, 0)
        first = tl.load(ENDS + slot, mask, other=0) - tl.load(COUNTS + slot, mask, other=0)
        position = row * C + first + rank
        tl.store(VALUES + position, tl.load(X + row * C + cols, mask), mask)
        tl.store(INDICES + position, cols.to(tl.int64), mask)


def _require_cuda(a):
    if not HAS_TRITON or not a.is_cuda:
        raise NotImplementedError('handwritten reductions require CUDA and Triton')


def _reduction_plan(a, dims, keepdim):
    rank = a.ndim
    if rank == 0:
        if dims and (len(dims) != 1 or int(dims[0]) not in (-1, 0)):
            raise IndexError('scalar reduction dimension out of range')
        return a.reshape(1, 1), ()
    axes = list(range(rank)) if not dims else [int(d) % max(rank, 1) for d in dims]
    if any(int(d) < -max(rank, 1) or int(d) >= max(rank, 1) for d in (dims or [])):
        raise IndexError('reduction dimension out of range')
    if len(set(axes)) != len(axes):
        raise ValueError('duplicate reduction dimension')
    axes = sorted(axes)
    other = [d for d in range(rank) if d not in axes]
    order = other + axes
    rows, width = 1, 1
    for d in other:
        rows *= a.shape[d]
    for d in axes:
        width *= a.shape[d]
    matrix = a.permute(order).contiguous().reshape(rows, width) if rank else a.reshape(1, 1)
    shape = tuple(1 if d in axes else a.shape[d] for d in range(rank)) if keepdim else tuple(a.shape[d] for d in other)
    return matrix, shape


def _reduce(a, dims, keepdim, operation, dtype=None, correction=1):
    _require_cuda(a)
    if dtype is not None:
        a = a.to(dtype)
    matrix, shape = _reduction_plan(a, dims, keepdim)
    rows, width = matrix.shape
    if operation in ('amax', 'amin') and width == 0:
        raise ValueError('cannot reduce an empty dimension without an identity')
    if operation in ('amax', 'amin') and not a.is_floating_point():
        raise NotImplementedError('handwritten extrema currently require floating point input')
    if operation in ('mean', 'var') and not a.is_floating_point():
        raise TypeError('mean and variance require floating point input')
    output_dtype = a.dtype
    if operation in ('sum', 'prod') and dtype is None and not a.is_floating_point():
        output_dtype = torch.int64
    if operation == 'any':
        output_dtype = torch.uint8 if a.dtype == torch.uint8 else torch.bool
    output = torch.empty((rows,), device=a.device, dtype=output_dtype)
    if rows:
        block = min(_next_pow2(width), 1024)
        _reduce_rows_kernel[(rows,)](matrix, output, width, OP=operation,
                                     CORRECTION=correction, BLOCK=block)
    return output.reshape(shape)


def _row_sum(a, keepdim):
    return _reduce(a, [1], keepdim, 'sum')


def _col_sum(a, keepdim):
    return _reduce(a, [0], keepdim, 'sum')


def make_sum_kernel(reduction_dims, keepdim, dtype=None) -> Callable:
    _dims, _keepdim, _dtype = reduction_dims, bool(keepdim), dtype
    def triton_sum(a):
        return _reduce(a, _dims, _keepdim, 'sum', _dtype)
    return triton_sum


def make_mean_kernel(reduction_dims, keepdim, dtype=None) -> Callable:
    _dims, _keepdim, _dtype = reduction_dims, bool(keepdim), dtype
    def triton_mean(a):
        return _reduce(a, _dims, _keepdim, 'mean', _dtype)
    return triton_mean


def _axis_matrix(a, dim):
    rank = a.ndim
    if not -max(rank, 1) <= int(dim) < max(rank, 1):
        raise IndexError('dimension out of range')
    axis = int(dim) % max(rank, 1)
    order = [d for d in range(rank) if d != axis] + ([axis] if rank else [])
    shape = tuple(a.shape[d] for d in order)
    width = a.shape[axis] if rank else 1
    rows = 1
    for size in shape[:-1]:
        rows *= size
    matrix = a.permute(order).contiguous().reshape(rows, width) if rank else a.reshape(1, 1)
    inverse = [order.index(d) for d in range(rank)]
    return matrix, shape, inverse


def _softmax(a, dim, logarithmic=False, dtype=None):
    _require_cuda(a)
    if dtype is not None:
        a = a.to(dtype)
    if not a.is_floating_point():
        raise TypeError('softmax requires floating point input')
    if a.dtype == torch.float64:
        raise NotImplementedError('handwritten softmax supports float16, bfloat16 and float32')
    matrix, shape, inverse = _axis_matrix(a, dim)
    rows, width = matrix.shape
    out = torch.empty(matrix.shape, device=a.device, dtype=a.dtype)
    if rows and width:
        _softmax_rows_kernel[(rows,)](matrix, out, width, LOG=logarithmic,
                                      BLOCK=min(_next_pow2(width), 1024))
    return out.reshape(shape).permute(inverse)


def _softmax_2d(a):
    return _softmax(a, -1, dtype=torch.float32)


def make_softmax_kernel(dim=None, dtype=None) -> Callable:
    _dim, _dtype = dim, dtype
    def triton_softmax(a, dim_arg=None, *rest):
        axis = int(dim_arg) if dim_arg is not None else (_dim if _dim is not None else -1)
        return _softmax(a, axis, dtype=_dtype)
    return triton_softmax


def make_log_softmax_kernel(dim=None, dtype=None) -> Callable:
    _dim, _dtype = dim, dtype
    def triton_log_softmax(a, dim_arg=None, *rest):
        axis = int(dim_arg) if dim_arg is not None else (_dim if _dim is not None else -1)
        return _softmax(a, axis, logarithmic=True, dtype=_dtype)
    return triton_log_softmax


def make_amax_kernel(reduction_dims, keepdim) -> Callable:
    _dims, _keepdim = reduction_dims, bool(keepdim)
    def triton_amax(a):
        return _reduce(a, _dims, _keepdim, 'amax')
    return triton_amax


def make_amin_kernel(reduction_dims, keepdim) -> Callable:
    _dims, _keepdim = reduction_dims, bool(keepdim)
    def triton_amin(a):
        return _reduce(a, _dims, _keepdim, 'amin')
    return triton_amin


def _extreme(a, dim, keepdim, maximum):
    _require_cuda(a)
    matrix, shape = _reduction_plan(a, None if dim is None else [dim], keepdim)
    rows, width = matrix.shape
    if not width:
        raise ValueError('cannot reduce an empty dimension')
    if not a.is_floating_point() or a.dtype == torch.float64:
        raise NotImplementedError('indexed extrema currently support float16, bfloat16 and float32')
    values = torch.empty((rows,), dtype=a.dtype, device=a.device)
    indices = torch.empty((rows,), dtype=torch.int64, device=a.device)
    if rows:
        _extreme_rows_kernel[(rows,)](matrix, values, indices, width, MAXIMUM=maximum,
                                      BLOCK=min(_next_pow2(width), 1024))
    return values.reshape(shape), indices.reshape(shape)


def make_max_dim_kernel(dim, keepdim) -> Callable:
    _dim, _keepdim = int(dim), bool(keepdim)
    def triton_max_dim(a):
        return _extreme(a, _dim, _keepdim, True)
    return triton_max_dim


def make_min_dim_kernel(dim, keepdim) -> Callable:
    _dim, _keepdim = int(dim), bool(keepdim)
    def triton_min_dim(a):
        return _extreme(a, _dim, _keepdim, False)
    return triton_min_dim


def make_argmax_kernel(dim, keepdim) -> Callable:
    _dim, _keepdim = dim, bool(keepdim)
    def triton_argmax(a):
        return _extreme(a, _dim, _keepdim, True)[1]
    return triton_argmax


def make_argmin_kernel(dim, keepdim) -> Callable:
    _dim, _keepdim = dim, bool(keepdim)
    def triton_argmin(a):
        return _extreme(a, _dim, _keepdim, False)[1]
    return triton_argmin


def make_prod_kernel(reduction_dims, keepdim) -> Callable:
    _dims, _keepdim = reduction_dims, bool(keepdim)
    def triton_prod(a):
        return _reduce(a, _dims, _keepdim, 'prod')
    return triton_prod


def make_any_kernel(reduction_dims, keepdim) -> Callable:
    _dims, _keepdim = reduction_dims, bool(keepdim)
    def triton_any(a):
        return _reduce(a, _dims, _keepdim, 'any')
    return triton_any


def make_var_kernel(reduction_dims, keepdim, correction=1) -> Callable:
    _dims, _keepdim, _correction = reduction_dims, bool(keepdim), correction
    def triton_var(a):
        return _reduce(a, _dims, _keepdim, 'var', correction=_correction)
    return triton_var


def make_cumsum_kernel(dim, dtype=None) -> Callable:
    _dim, _dtype = int(dim), dtype
    def triton_cumsum(a):
        _require_cuda(a)
        if _dtype is not None:
            a = a.to(_dtype)
        matrix, shape, inverse = _axis_matrix(a, _dim)
        rows, width = matrix.shape
        output_dtype = a.dtype if a.is_floating_point() or _dtype is not None else torch.int64
        out = torch.empty(matrix.shape, device=a.device, dtype=output_dtype)
        if rows and width:
            _cumsum_rows_kernel[(rows,)](matrix, out, width, BLOCK=min(_next_pow2(width), 1024))
        return out.reshape(shape).permute(inverse)
    return triton_cumsum


def make_sort_kernel(dim=-1, descending=False, stable=False) -> Callable:
    _dim, _descending = int(dim), bool(descending)
    def triton_sort(a):
        _require_cuda(a)
        matrix, shape, inverse = _axis_matrix(a, _dim)
        rows, width = matrix.shape
        if width > 16384 or not a.is_floating_point() or a.dtype == torch.float64:
            raise NotImplementedError('handwritten sort supports float16/bfloat16/float32 axes up to16384')
        values = torch.empty(matrix.shape, device=a.device, dtype=a.dtype)
        indices = torch.empty(matrix.shape, device=a.device, dtype=torch.int64)
        if rows and width:
            block = _next_pow2(width)
            _sort_rows_kernel[(rows,)](matrix, values, indices, width, DESCENDING=_descending,
                                       LOG_BLOCK=block.bit_length()-1, BLOCK=block)
        return values.reshape(shape).permute(inverse), indices.reshape(shape).permute(inverse)
    return triton_sort


#: The most distinct key values integer_sort takes: its counting sort's bins.
_SORT_KEY_RANGE = 4096


def integer_sort(a, dim=-1, descending=False):
    """``aten.sort(stable=True)`` of integer or bool keys: ``(values, int64
    indices)``, equal keys in input order. A counting sort: the keys' range is
    read back to size the bins (at most ``_SORT_KEY_RANGE``), each block of a
    row counts its keys per bin, a running sum over (bin, block) gives where
    each bin's keys from each block start, and each key lands there after the
    earlier keys of its block in the same bin."""
    if a.is_floating_point() or a.is_complex():
        raise NotImplementedError('integer_sort takes integer or bool keys')
    matrix, shape, inverse = _axis_matrix(a, dim)
    rows, width = matrix.shape
    values = torch.empty(matrix.shape, device=a.device, dtype=a.dtype)
    indices = torch.empty(matrix.shape, device=a.device, dtype=torch.int64)
    if rows and width:
        extremes = torch.empty((2,), device=a.device, dtype=torch.int64)
        _int_extremes_kernel[(1,)](matrix, extremes, rows * width, BLOCK=1024)
        low, high = extremes.tolist()
        if high - low >= _SORT_KEY_RANGE:
            raise NotImplementedError(f'handwritten integer sort takes keys spanning at most '
                                      f'{_SORT_KEY_RANGE} values, not {high - low + 1}')
        # Blocks of at most 256 keys: their pairwise rank stays in registers.
        bins, block = high - low + 1, min(max(_next_pow2(width), 16), 256)
        blocks = triton.cdiv(width, block)
        counts = torch.empty((rows, bins * blocks), device=a.device, dtype=torch.int32)
        ends = torch.empty((rows, bins * blocks), device=a.device, dtype=torch.int64)
        base = high if descending else low
        _key_counts_kernel[(rows * blocks,)](matrix, counts, width, blocks, base, bins,
                                             DESCENDING=bool(descending), BLOCK=block, CHUNK=16)
        _cumsum_rows_kernel[(rows,)](counts, ends, bins * blocks,
                                     BLOCK=min(_next_pow2(bins * blocks), 1024))
        _counting_sort_scatter_kernel[(rows * blocks,)](
            matrix, counts, ends, values, indices, width, blocks, base, bins,
            DESCENDING=bool(descending), BLOCK=block, CHUNK=16)
    return values.reshape(shape).permute(inverse), indices.reshape(shape).permute(inverse)
