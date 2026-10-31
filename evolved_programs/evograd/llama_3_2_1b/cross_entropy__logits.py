"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'cols': 128256, 'rows': 4096}, 'dtype': 'float32', 'device': 'cuda:0', 'family': None, 'dynamic': False}

# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

def _reduce(a, dims, keepdim, operation, dtype=None, correction=1):
    _require_cuda(a)
    if dtype is not None:
        a = a.to(dtype)
    matrix, shape = _reduction_plan(a, dims, keepdim)
    rows, width = matrix.shape
    if operation in ('amax', 'amin') and width == 0:
        raise ValueError('cannot reduce an empty dimension without an identity')
    if operation in ('amax', 'amin') and (not a.is_floating_point()):
        raise NotImplementedError('handwritten extrema currently require floating point input')
    if operation in ('mean', 'var') and (not a.is_floating_point()):
        raise TypeError('mean and variance require floating point input')
    output_dtype = a.dtype
    if operation in ('sum', 'prod') and dtype is None and (not a.is_floating_point()):
        output_dtype = torch.int64
    if operation == 'any':
        output_dtype = torch.uint8 if a.dtype == torch.uint8 else torch.bool
    output = torch.empty((rows,), device=a.device, dtype=output_dtype)
    if rows:
        block = min(_next_pow2(width), 1024)
        _reduce_rows_kernel[rows,](matrix, output, width, OP=operation, CORRECTION=correction, BLOCK=block)
    return output.reshape(shape)

def _require_cuda(a):
    if not HAS_TRITON or not a.is_cuda:
        raise NotImplementedError('handwritten reductions require CUDA and Triton')

def _reduction_plan(a, dims, keepdim):
    rank = a.ndim
    if rank == 0:
        if dims and (len(dims) != 1 or int(dims[0]) not in (-1, 0)):
            raise IndexError('scalar reduction dimension out of range')
        return (a.reshape(1, 1), ())
    axes = list(range(rank)) if not dims else [int(d) % max(rank, 1) for d in dims]
    if any((int(d) < -max(rank, 1) or int(d) >= max(rank, 1) for d in dims or [])):
        raise IndexError('reduction dimension out of range')
    if len(set(axes)) != len(axes):
        raise ValueError('duplicate reduction dimension')
    axes = sorted(axes)
    other = [d for d in range(rank) if d not in axes]
    order = other + axes
    rows, width = (1, 1)
    for d in other:
        rows *= a.shape[d]
    for d in axes:
        width *= a.shape[d]
    matrix = a.permute(order).contiguous().reshape(rows, width) if rank else a.reshape(1, 1)
    shape = tuple((1 if d in axes else a.shape[d] for d in range(rank))) if keepdim else tuple((a.shape[d] for d in other))
    return (matrix, shape)

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

@triton.jit
def _reduce_rows_kernel(X, Y, C, OP: tl.constexpr, CORRECTION: tl.constexpr, BLOCK: tl.constexpr):
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
def _nan_maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

@triton.jit
def _nan_minimum(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)

@triton.jit
def _product_combine(a, b):
    return a * b

def sub_tt(a: torch.Tensor, b: torch.Tensor, alpha=1) -> torch.Tensor:
    if a.dtype == torch.bool or b.dtype == torch.bool:
        raise RuntimeError('Subtraction with Boolean tensors is not supported')
    a, b = _binary_inputs(a, b)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _sub_kernel[_grid(a.numel())](a, b, out, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

def _binary_inputs(a, b, *, true_divide=False):
    """ATen promotion must precede broadcasting away scalar-tensor rank."""
    dtype = torch.result_type(a, b)
    if true_divide and (not dtype.is_floating_point) and (not dtype.is_complex):
        dtype = torch.get_default_dtype()
    a, b = torch.broadcast_tensors(a, b)
    return (a.to(dtype).contiguous(), b.to(dtype).contiguous())

def _check_alpha(alpha, dtype):
    if isinstance(alpha, bool) and dtype != torch.bool:
        raise RuntimeError('Boolean alpha only supported for Boolean results')
    if isinstance(alpha, float) and (not (dtype.is_floating_point or dtype.is_complex)):
        raise RuntimeError('For integral input tensors, alpha must not be floating point')
    if isinstance(alpha, complex) and (not dtype.is_complex):
        raise RuntimeError('For non-complex input tensors, alpha must not be complex')

@triton.jit
def _sub_kernel(a_ptr, b_ptr, out_ptr, alpha, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    tl.store(out_ptr + offs, a - alpha * b, mask=mask)

def _grid(n: int):
    import math
    return (math.ceil(n / _BLOCK),)

def exp(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_exp_kernel, a)

def _unary_f32(kernel, a: torch.Tensor) -> torch.Tensor:
    """Run a unary kernel that computes in float32; cast result back to a.dtype."""
    orig = a.dtype
    a = a.contiguous()
    out = _f32_out(a)
    kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out.to(orig)

def _f32_out(a: torch.Tensor) -> torch.Tensor:
    """Allocate a float32 output with the same shape/device as a."""
    return torch.empty_like(a, dtype=torch.float32)

@triton.jit
def _exp_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(a), mask=mask)

def log_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log_kernel, a)

@triton.jit
def _log_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.log(a), mask=mask)

def ne_scalar(a, s):
    return _cmp_scalar(_ne_scalar_kernel, a, s)

def _cmp_scalar(kernel, a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out

def _bool_out(a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a, dtype=torch.bool)

@triton.jit
def _ne_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (a != scalar).to(tl.int8), mask=mask)

def fill_tensor(shape, value, *, dtype, device='cuda'):
    """Allocate a constant tensor without a PyTorch fill kernel."""
    out = torch.empty(tuple(shape), dtype=dtype, device=device)
    if out.numel():
        _fill_scalar_kernel[_grid(out.numel())](out, value, N=out.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _fill_scalar_kernel(out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, scalar, mask=mask)

def where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dtype = torch.result_type(x, y)
    condition, x, y = torch.broadcast_tensors(condition, x, y)
    condition = condition.contiguous()
    x = x.contiguous().to(out_dtype)
    y = y.contiguous().to(out_dtype)
    out = torch.empty_like(x)
    N = x.numel()
    _where_kernel[_grid(N)](condition, x, y, out, N, BLOCK=_BLOCK)
    return out

@triton.jit
def _where_kernel(cond_ptr, x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    cond = tl.load(cond_ptr + offs, mask=mask, other=0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.where(cond != 0, x, y), mask=mask)

def gather(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    axis = _check_indexed(input, dim, index)
    input, index = (input.contiguous(), index.contiguous())
    output = torch.empty(index.shape, device=input.device, dtype=input.dtype)
    if index.numel():
        _indexed_kernel[triton.cdiv(index.numel(), 256),](input, index, input, output, index.numel(), INDEX_SHAPE=tuple(index.shape), INDEX_STRIDES=tuple(index.stride()), DATA_STRIDES=tuple(input.stride()), SRC_STRIDES=tuple(index.stride()), AXIS=axis, AXIS_SIZE=input.shape[axis], OP='gather', VALUE=0, BLOCK=256)
    return output

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
    if any((index.shape[d] > base.shape[d] for d in range(base.ndim) if d != axis)):
        raise ValueError('index exceeds input shape outside the indexed dimension')
    return axis

@triton.jit
def _indexed_kernel(INPUT, INDEX, SRC, OUT, N, INDEX_SHAPE: tl.constexpr, INDEX_STRIDES: tl.constexpr, DATA_STRIDES: tl.constexpr, SRC_STRIDES: tl.constexpr, AXIS: tl.constexpr, AXIS_SIZE: tl.constexpr, OP: tl.constexpr, VALUE, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    index = tl.load(INDEX + offsets, mask, other=0).to(tl.int64)
    target = tl.full((BLOCK,), 0, tl.int64)
    source = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(INDEX_SHAPE)):
        coordinate = offsets // INDEX_STRIDES[dim] % INDEX_SHAPE[dim]
        target += (index if dim == AXIS else coordinate) * DATA_STRIDES[dim]
        source += coordinate * SRC_STRIDES[dim]
    tl.device_assert(~mask | (index >= 0) & (index < AXIS_SIZE), 'index out of bounds')
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

def neg(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _neg_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _neg_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, -a, mask=mask)

def div_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _binary_inputs(a, b, true_divide=True)
    out = torch.empty_like(a)
    _div_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _div_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, a / b, mask=mask)

def scatter_value(self_t: torch.Tensor, dim: int, index: torch.Tensor, value) -> torch.Tensor:
    axis = _check_indexed(self_t, dim, index)
    output, index = (self_t.contiguous().clone(), index.contiguous())
    if index.numel():
        _indexed_kernel[triton.cdiv(index.numel(), 256),](output, index, output, output, index.numel(), INDEX_SHAPE=tuple(index.shape), INDEX_STRIDES=tuple(index.stride()), DATA_STRIDES=tuple(output.stride()), SRC_STRIDES=tuple(index.stride()), AXIS=axis, AXIS_SIZE=output.shape[axis], OP='scatter_value', VALUE=value, BLOCK=256)
    return output

def mul_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _binary_inputs(a, b)
    out = torch.empty_like(a)
    _mul_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _mul_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * b, mask=mask)

def _k0(a, _dims=[1], _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'amax')

def _k1(a, b, alpha=1):
    return sub_tt(a, b, alpha)

def _k2(a, _dims=[1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k3(a, fn_s=ne_scalar, s=-100.0):
    return fn_s(a, s)

def _k4(out_dtype=torch.int64, val=0):
    return fill_tensor((), val, dtype=out_dtype, device='cuda')

def _k5(self_t, index, dim=1):
    return gather(self_t, dim, index)

def _k6(out_dtype=torch.float32, val=0):
    return fill_tensor((), val, dtype=out_dtype, device='cuda')

def _k7(a, _dims=None, _dtype=None, _keepdim=False):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k8(dtype=torch.float32, fill_val=0, shape=[4096, 128256]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

def _k9(base, index, dim=1, value=-1.0):
    return scatter_value(base, dim, index, value)

@triton.jit
def _strided_copy_kernel(src_ptr, out_ptr, N, s0, s1, s2, s3, s4, s5, s6, s7, t0, t1, t2, t3, t4, t5, t6, t7, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    rem = offs
    src_off = tl.zeros_like(offs)
    idx = rem % s7
    rem = rem // s7
    src_off += idx * t7
    idx = rem % s6
    rem = rem // s6
    src_off += idx * t6
    idx = rem % s5
    rem = rem // s5
    src_off += idx * t5
    idx = rem % s4
    rem = rem // s4
    src_off += idx * t4
    idx = rem % s3
    rem = rem // s3
    src_off += idx * t3
    idx = rem % s2
    rem = rem // s2
    src_off += idx * t2
    idx = rem % s1
    rem = rem // s1
    src_off += idx * t1
    idx = rem % s0
    src_off += idx * t0
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)

def strided_copy(a, out_shape, src_strides):
    """Materialize a contiguous tensor of ``out_shape`` gathered from ``a``.

    ``src_strides[i]`` is the element stride of ``a`` along output dim ``i``
    (0 for broadcast dims). Rank up to 8.
    """
    out_shape = [int(s) for s in out_shape]
    src_strides = [int(s) for s in src_strides]
    out = torch.empty(out_shape, device=a.device, dtype=a.dtype)
    n = out.numel()
    if n == 0:
        return out
    pad = 8 - len(out_shape)
    sizes = [1] * pad + out_shape
    strides = [0] * pad + src_strides
    grid = ((n + 1023) // 1024,)
    _strided_copy_kernel[grid](a, out, n, *sizes, *strides, BLOCK=1024)
    return out

def permute_copy(a, dims):
    """Contiguous permute: equivalent to a.permute(dims).contiguous()."""
    dims = [int(d) % a.ndim for d in dims]
    return strided_copy(a, [a.shape[d] for d in dims], [a.stride(d) for d in dims])

def expand_copy(a, shape):
    """Contiguous expand: equivalent to a.expand(shape).contiguous().

    Follows aten.expand semantics: ``shape`` may prepend new leading dims, and
    -1 keeps the source size. Size-1 source dims broadcast with stride 0.
    """
    shape = [int(s) for s in shape]
    lead = len(shape) - a.ndim
    out_shape, strides = ([], [])
    for i, s in enumerate(shape):
        if i < lead:
            out_shape.append(s)
            strides.append(0)
            continue
        src_dim = i - lead
        src_size = a.shape[src_dim]
        size = src_size if s == -1 else s
        if size != src_size and src_size != 1:
            raise ValueError(f'expand_copy: cannot expand dim {src_dim} of size {src_size} to {size}')
        out_shape.append(size)
        strides.append(0 if src_size == 1 and size != 1 else a.stride(src_dim))
    return strided_copy(a, out_shape, strides)

def clone_copy(a):
    """Contiguous clone via the strided-copy kernel."""
    return strided_copy(a, list(a.shape), list(a.stride()))

def _unified_forward_graph(t_primals_1, t_primals_2):
    t_amax = _k0(t_primals_1)  # aten.amax.default
    t_sub = _k1(t_primals_1, t_amax)  # aten.sub.Tensor
    t_exp = exp(t_sub)  # aten.exp.default
    t_sum_1 = _k2(t_exp)  # aten.sum.dim_IntList
    t_log = log_(t_sum_1)  # aten.log.default
    t_sub_1 = _k1(t_sub, t_log)  # aten.sub.Tensor
    t_ne = _k3(t_primals_2)  # aten.ne.Scalar
    t_scalar_tensor = _k4()  # aten.scalar_tensor.default
    t_where = where(t_ne, t_primals_2, t_scalar_tensor)  # aten.where.self
    t_unsqueeze = t_where.unsqueeze(1)  # aten.unsqueeze.default
    t_gather = _k5(t_sub_1, t_unsqueeze)  # aten.gather.default
    t_squeeze = t_gather.squeeze(1)  # aten.squeeze.dim
    t_neg = neg(t_squeeze)  # aten.neg.default
    t_scalar_tensor_1 = _k6()  # aten.scalar_tensor.default
    t_where_1 = where(t_ne, t_neg, t_scalar_tensor_1)  # aten.where.self
    t_sum_2 = _k7(t_ne)  # aten.sum.default
    t_convert_element_type = t_sum_2.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_sum_3 = _k7(t_where_1)  # aten.sum.default
    t_div = div_tt(t_sum_3, t_convert_element_type)  # aten.div.Tensor
    return (t_div, t_primals_1, t_primals_2, t_amax, t_log, t_convert_element_type)

def _unified_backward_graph(t_primals_1, t_primals_2, t_amax, t_log, t_convert_element_type, t_tangents_1):
    t_div_1 = div_tt(t_tangents_1, t_convert_element_type)  # aten.div.Tensor
    t_unsqueeze_1 = t_primals_2.unsqueeze(1)  # aten.unsqueeze.default
    t_ne_3 = _k3(t_unsqueeze_1)  # aten.ne.Scalar
    t_scalar_tensor = _k4()  # aten.scalar_tensor.default
    t_where_2 = where(t_ne_3, t_unsqueeze_1, t_scalar_tensor)  # aten.where.self
    t_full = _k8()  # aten.full.default
    t_scatter = _k9(t_full, t_where_2)  # aten.scatter.value
    t_scalar_tensor_1 = _k6()  # aten.scalar_tensor.default
    t_where_3 = where(t_ne_3, t_div_1, t_scalar_tensor_1)  # aten.where.self
    t_mul = mul_tt(t_scatter, t_where_3)  # aten.mul.Tensor
    t_sub = _k1(t_primals_1, t_amax)  # aten.sub.Tensor
    t_sub_1 = _k1(t_sub, t_log)  # aten.sub.Tensor
    t_alias = t_sub_1  # aten.alias.default
    t_alias_1 = t_alias  # aten.alias.default
    t_exp_1 = exp(t_alias_1)  # aten.exp.default
    t_sum_4 = _k2(t_mul)  # aten.sum.dim_IntList
    t_mul_1 = mul_tt(t_exp_1, t_sum_4)  # aten.mul.Tensor
    t_sub_2 = _k1(t_mul, t_mul_1)  # aten.sub.Tensor
    return (t_sub_2, None)

@triton.jit
def _ce_fwd_rows(X, T, ROW_LOSS, LSE, C: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    tgt = tl.load(T + r)
    ok = tgt != -100

    mvec = tl.full((BLOCK,), float("-inf"), tl.float32)
    for s in range(0, C, BLOCK):
        c = s + offs
        x = tl.load(X + r * C + c, mask=c < C, other=float("-inf")).to(tl.float32)
        mvec = tl.maximum(mvec, x)
    m = tl.max(mvec, axis=0)

    avec = tl.zeros((BLOCK,), tl.float32)
    for s in range(0, C, BLOCK):
        c = s + offs
        x = tl.load(X + r * C + c, mask=c < C, other=float("-inf")).to(tl.float32)
        avec += tl.exp(x - m)
    lse = tl.log(tl.sum(avec, axis=0)) + m
    xt = tl.load(X + r * C + tgt, mask=ok, other=0.0).to(tl.float32)
    tl.store(LSE + r, lse)
    tl.store(ROW_LOSS + r, tl.where(ok, lse - xt, 0.0))

@triton.jit
def _ce_finish_fwd(ROW_LOSS, T, LOSS, CNT, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < 4096
    t = tl.load(T + offs, mask=mask, other=-100)
    active = mask & (t != -100)
    tot = tl.sum(tl.load(ROW_LOSS + offs, mask=mask, other=0.0).to(tl.float32), axis=0)
    cnt = tl.sum(tl.where(active, 1.0, 0.0), axis=0)
    tl.store(LOSS, tot / cnt)
    tl.store(CNT, cnt)

@triton.jit
def _ce_bwd(X, T, LSE, CNT, DLOSS, DX, C: tl.constexpr, BLOCK: tl.constexpr):
    cb = tl.program_id(0)
    r = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < C

    tgt = tl.load(T + r)
    active = tgt != -100
    lse = tl.load(LSE + r).to(tl.float32)
    scale = tl.load(DLOSS).to(tl.float32) / tl.load(CNT).to(tl.float32)
    scale = tl.where(active, scale, 0.0)

    x = tl.load(X + r * C + offs, mask=mask, other=float("-inf")).to(tl.float32)
    g = tl.exp(x - lse) * scale
    g -= tl.where(offs == tgt, scale, 0.0)
    tl.store(DX + r * C + offs, g, mask=mask)

def cross_entropy_forward_with_saved(logits, target):
    if tuple(logits.shape) != (4096, 128256) or logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError('bad logits')
    if tuple(target.shape) != (4096,) or target.dtype != torch.int64 or not target.is_cuda:
        raise ValueError('bad target')
    with torch.cuda.device(logits.device):
        row_loss = torch.empty((4096,), device=logits.device, dtype=torch.float32)
        lse = torch.empty((4096,), device=logits.device, dtype=torch.float32)
        loss = torch.empty((), device=logits.device, dtype=torch.float32)
        cnt = torch.empty((), device=logits.device, dtype=torch.float32)
        _ce_fwd_rows[(4096,)](logits, target, row_loss, lse, C=128256, BLOCK=1024, num_warps=8)
        _ce_finish_fwd[(1,)](row_loss, target, loss, cnt, BLOCK=4096, num_warps=8)
    return loss, (logits, target, lse, cnt)

def cross_entropy_backward_from_saved(dloss, saved_tensors):
    if len(saved_tensors) != 4:
        raise ValueError('bad saved state')
    logits, target, lse, cnt = saved_tensors
    if tuple(logits.shape) != (4096, 128256) or logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError('bad logits')
    if tuple(target.shape) != (4096,) or target.dtype != torch.int64 or not target.is_cuda:
        raise ValueError('bad target')
    if tuple(lse.shape) != (4096,) or lse.dtype != torch.float32 or not lse.is_cuda:
        raise ValueError('bad lse')
    if tuple(cnt.shape) != () or cnt.dtype != torch.float32 or not cnt.is_cuda:
        raise ValueError('bad count')
    with torch.cuda.device(logits.device):
        out = torch.empty_like(logits)
        _ce_bwd[(triton.cdiv(128256, 1024), 4096)](logits, target, lse, cnt, dloss, out, C=128256, BLOCK=1024, num_warps=8)
    return (out,)


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
        dlogits, = cross_entropy_backward_from_saved(
            dloss, saved
        )
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

