"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm'}, {'side': 'forward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_1'}, {'side': 'forward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_2'}, {'side': 'backward', 'node_name': 'mm_3', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_3'}, {'side': 'backward', 'node_name': 'mm_4', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_4'}, {'side': 'backward', 'node_name': 'mm_5', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_5'}, {'side': 'backward', 'node_name': 'mm_6', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_6'}, {'side': 'backward', 'node_name': 'mm_7', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_7'}, {'side': 'backward', 'node_name': 'mm_8', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_8'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 2, 'T': 2048, 'H': 1024, 'HQ': 16, 'HK': 8, 'D': 128, 'QO': 2048, 'KVO': 1024}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_3(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((1024, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_4(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_5(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((1024, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_6(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_7(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((2048, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_8(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

def pow_scalar(a: torch.Tensor, exp) -> torch.Tensor:
    a = a.contiguous()
    exp_f = float(exp)
    if exp_f == 0.0:
        return fill_scalar(a, 1)
    if exp_f == 1.0:
        return a.clone()
    if exp_f == 2.0:
        return mul_tt(a, a)
    out = torch.empty_like(a, dtype=torch.float32)
    _pow_scalar_kernel[_grid(a.numel())](a, out, exp_f, N=a.numel(), BLOCK=_BLOCK)
    return out.to(a.dtype)

def fill_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    """Return a new tensor with the same shape/dtype as a, filled with scalar."""
    out = torch.empty_like(a)
    _fill_scalar_kernel[_grid(a.numel())](out, scalar, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _fill_scalar_kernel(out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + offs, scalar, mask=mask)

def _grid(n: int):
    import math
    return (math.ceil(n / _BLOCK),)

def mul_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _binary_inputs(a, b)
    out = torch.empty_like(a)
    _mul_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out

def _binary_inputs(a, b, *, true_divide=False):
    """ATen promotion must precede broadcasting away scalar-tensor rank."""
    dtype = torch.result_type(a, b)
    if true_divide and (not dtype.is_floating_point) and (not dtype.is_complex):
        dtype = torch.get_default_dtype()
    a, b = torch.broadcast_tensors(a, b)
    return (a.to(dtype).contiguous(), b.to(dtype).contiguous())

@triton.jit
def _mul_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * b, mask=mask)

@triton.jit
def _pow_scalar_kernel(a_ptr, out_ptr, exp, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offs, libdevice.pow(a, exp), mask=mask)

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

def add_scalar(a: torch.Tensor, scalar, alpha=1) -> torch.Tensor:
    a = _scalar_input(a, scalar)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_scalar_kernel[_grid(a.numel())](a, out, scalar, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

def _scalar_input(a, scalar, *, true_divide=False):
    dtype = torch.result_type(a, scalar)
    if true_divide and (not dtype.is_floating_point) and (not dtype.is_complex):
        dtype = torch.get_default_dtype()
    return a.to(dtype).contiguous()

def _check_alpha(alpha, dtype):
    if isinstance(alpha, bool) and dtype != torch.bool:
        raise RuntimeError('Boolean alpha only supported for Boolean results')
    if isinstance(alpha, float) and (not (dtype.is_floating_point or dtype.is_complex)):
        raise RuntimeError('For integral input tensors, alpha must not be floating point')
    if isinstance(alpha, complex) and (not dtype.is_complex):
        raise RuntimeError('For non-complex input tensors, alpha must not be complex')

@triton.jit
def _add_scalar_kernel(a_ptr, out_ptr, scalar, alpha, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
    tl.store(out_ptr + offs, a + alpha * scalar, mask=mask)

def rsqrt(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _rsqrt_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _rsqrt_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offs, 1.0 / tl.sqrt(a), mask=mask)

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

def cat2(tensors, dim: int):
    if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
        raise ValueError('cat2 supports exactly two tensors')
    a, b = (tensors[0].contiguous(), tensors[1].contiguous())
    actual_dim = int(dim) % a.ndim
    if a.ndim != b.ndim or any((a.shape[index] != b.shape[index] for index in range(a.ndim) if index != actual_dim)):
        raise ValueError('cat2 inputs must match outside the concatenated dim')
    output_shape = list(a.shape)
    output_shape[actual_dim] += b.shape[actual_dim]
    output = torch.empty(output_shape, device=a.device, dtype=a.dtype)
    inner = 1
    for size in a.shape[actual_dim + 1:]:
        inner *= size
    block = 256
    _cat2_kernel[triton.cdiv(output.numel(), block),](a, b, output, a.shape[actual_dim], b.shape[actual_dim], inner, output.numel(), BLOCK=block)
    return output

@triton.jit
def _cat2_kernel(a_ptr, b_ptr, out_ptr, A_AXIS, B_AXIS, INNER, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out_axis = A_AXIS + B_AXIS
    inner_offset = offs % INNER
    axis_offset = offs // INNER % out_axis
    outer_offset = offs // (INNER * out_axis)
    from_a = axis_offset < A_AXIS
    a_offset = outer_offset * A_AXIS * INNER + axis_offset * INNER + inner_offset
    b_offset = outer_offset * B_AXIS * INNER + (axis_offset - A_AXIS) * INNER + inner_offset
    value = tl.where(from_a, tl.load(a_ptr + a_offset, mask=mask & from_a, other=0.0), tl.load(b_ptr + b_offset, mask=mask & ~from_a, other=0.0))
    tl.store(out_ptr + offs, value, mask=mask)

def add_tt(a: torch.Tensor, b: torch.Tensor, alpha=1) -> torch.Tensor:
    a, b = _binary_inputs(a, b)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_kernel[_grid(a.numel())](a, b, out, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, alpha, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
    tl.store(out_ptr + offs, a + alpha * b, mask=mask)

def fill_tensor(shape, value, *, dtype, device='cuda'):
    """Allocate a constant tensor without a PyTorch fill kernel."""
    out = torch.empty(tuple(shape), dtype=dtype, device=device)
    if out.numel():
        _fill_scalar_kernel[_grid(out.numel())](out, value, N=out.numel(), BLOCK=_BLOCK)
    return out

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
    output, src = (base.contiguous().clone(), src.contiguous())
    if src.numel():
        _slice_scatter_kernel[triton.cdiv(src.numel(), 256),](src, output, src.numel(), SRC_SHAPE=tuple(src.shape), SRC_STRIDES=tuple(src.stride()), OUT_STRIDES=tuple(output.stride()), AXIS=axis, START=begin, STEP=stride, BLOCK=256)
    return output

@triton.jit
def _slice_scatter_kernel(SRC, OUT, N, SRC_SHAPE: tl.constexpr, SRC_STRIDES: tl.constexpr, OUT_STRIDES: tl.constexpr, AXIS: tl.constexpr, START: tl.constexpr, STEP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    target = tl.full((BLOCK,), 0, tl.int64)
    for dim in tl.static_range(len(SRC_SHAPE)):
        coordinate = offsets // SRC_STRIDES[dim] % SRC_SHAPE[dim]
        if dim == AXIS:
            coordinate = START + coordinate * STEP
        target += coordinate * OUT_STRIDES[dim]
    value = tl.load(SRC + offsets, mask, other=0)
    tl.store(OUT + target, value, mask)

def mul_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = _scalar_input(a, scalar)
    out = torch.empty_like(a)
    _mul_scalar_kernel[_grid(a.numel())](a, out, scalar, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _mul_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * scalar, mask=mask)

def div_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = _scalar_input(a, scalar, true_divide=True)
    out = torch.empty_like(a)
    _div_scalar_kernel[_grid(a.numel())](a, out, scalar, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _div_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a / scalar, mask=mask)

def _k0(a, exp_val=2):
    return pow_scalar(a, exp_val)

def _k1(a, _dims=[-1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'mean', _dtype)

def _k2(a, alpha=1, s=1e-06):
    return add_scalar(a, s, alpha)

def _k3(*runtime, plan=[(True, None), (False, 3), (False, 64), (False, 9223372036854775807)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    tensor = args[0]
    dim = args[1] if len(args) > 1 else 0
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    step = args[4] if len(args) > 4 else 1
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end, step)
    return tensor[tuple(slices)].contiguous()

def _k4(*runtime, plan=[(True, None), (False, 3), (False, 0), (False, 64)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    tensor = args[0]
    dim = args[1] if len(args) > 1 else 0
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    step = args[4] if len(args) > 4 else 1
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end, step)
    return tensor[tuple(slices)].contiguous()

def _k5(tensors, dim=-1):
    return cat2(tensors, dim)

def _k6(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k7(*runtime, plan=[(True, None), (False, 3), (False, 64), (False, 128)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    tensor = args[0]
    dim = args[1] if len(args) > 1 else 0
    start = args[2] if len(args) > 2 else None
    end = args[3] if len(args) > 3 else None
    step = args[4] if len(args) > 4 else 1
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end, step)
    return tensor[tuple(slices)].contiguous()

def _k8(dtype=torch.bfloat16, fill_val=0, shape=[2, 8, 2048, 128]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

def _k9(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 0), (False, 64)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

def _k10(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 64), (False, 9223372036854775807)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

def _k11(dtype=torch.bfloat16, fill_val=0, shape=[2, 16, 2048, 128]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

def _k12(a, _dims=[0, 1, 2], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k13(a, _dims=[3], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k14(a, exp_val=3):
    return pow_scalar(a, exp_val)

def _k15(a, s=-0.5):
    return mul_scalar(a, s)

def _k16(a, s=128):
    return div_scalar(a, s)

def _k17(a, exp_val=1.0):
    return pow_scalar(a, exp_val)

def _k18(a, s=2.0):
    return mul_scalar(a, s)

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

@triton.jit
def _rope128_kernel(x, cos, sin, out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    d = offs & 127
    t = (offs >> 7) % 2048
    other = tl.where(d < 64, offs + 64, offs - 64)
    sgn = tl.where(d < 64, -1.0, 1.0)
    x0 = tl.load(x + offs, mask, other=0.).to(tl.float32)
    xr = tl.load(x + other, mask, other=0.).to(tl.float32) * sgn
    c = tl.load(cos + t * 128 + d, mask, other=0.).to(tl.float32)
    s = tl.load(sin + t * 128 + d, mask, other=0.).to(tl.float32)
    tl.store(out + offs, x0 * c + xr * s, mask)

@triton.jit
def _unrope128_kernel(g, cos, sin, out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    d = offs & 127
    t = (offs >> 7) % 2048
    od = tl.where(d < 64, d + 64, d - 64)
    oo = tl.where(d < 64, offs + 64, offs - 64)
    sgn = tl.where(d < 64, 1.0, -1.0)
    g0 = tl.load(g + offs, mask, other=0.).to(tl.float32)
    gr = tl.load(g + oo, mask, other=0.).to(tl.float32)
    c = tl.load(cos + t * 128 + d, mask, other=0.).to(tl.float32)
    s = tl.load(sin + t * 128 + od, mask, other=0.).to(tl.float32)
    tl.store(out + offs, g0 * c + sgn * gr * s, mask)

def rope128(x, cos, sin):
    out = torch.empty_like(x)
    _rope128_kernel[_grid(x.numel())](x, cos, sin, out, x.numel(), BLOCK=_BLOCK)
    return out

def unrope128(g, cos, sin):
    out = torch.empty_like(g)
    _unrope128_kernel[_grid(g.numel())](g, cos, sin, out, g.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _rms_rope128_fwd_kernel(mm, nw, cos, sin, out, rs, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    bt = row // HEADS
    h = row - bt * HEADS
    t = bt & 2047
    b = bt >> 11
    base = bt * (HEADS * 128) + h * 128
    x = tl.load(mm + base + d).to(tl.float32)
    r = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / 128.0 + 1.0e-6)
    tl.store(rs + row, r)
    od = tl.where(d < 64, d + 64, d - 64)
    sgn = tl.where(d < 64, -1.0, 1.0)
    y0 = x * r * tl.load(nw + d).to(tl.float32)
    y1 = tl.load(mm + base + od).to(tl.float32) * r * tl.load(nw + od).to(tl.float32) * sgn
    c = tl.load(cos + t * 128 + d).to(tl.float32)
    s = tl.load(sin + t * 128 + d).to(tl.float32)
    obase = ((b * HEADS + h) * 2048 + t) * 128
    tl.store(out + obase + d, y0 * c + y1 * s)

def rms_rope128(mm, nw, cos, sin, heads: int):
    out = torch.empty((2, heads, 2048, 128), device=mm.device, dtype=mm.dtype)
    rs = torch.empty((2, 2048, heads, 1), device=mm.device, dtype=torch.float32)
    _rms_rope128_fwd_kernel[(4096 * heads,)](mm, nw, cos, sin, out, rs, HEADS=heads, BLOCK=128)
    return out, rs

@triton.jit
def _rms_unrope_dh_kernel(g, mm, rptr, nw, cos, sin, dh, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    bt = row // HEADS
    h = row - bt * HEADS
    t = bt & 2047
    b = bt >> 11
    od = tl.where(d < 64, d + 64, d - 64)
    sgn = tl.where(d < 64, 1.0, -1.0)
    gbase = ((b * HEADS + h) * 2048 + t) * 128
    g0 = tl.load(g + gbase + d).to(tl.float32)
    gp = tl.load(g + gbase + od).to(tl.float32)
    c = tl.load(cos + t * 128 + d).to(tl.float32)
    s = tl.load(sin + t * 128 + od).to(tl.float32)
    dnorm = g0 * c + sgn * gp * s
    mbase = row * 128
    x = tl.load(mm + mbase + d).to(tl.float32)
    r = tl.load(rptr + row).to(tl.float32)
    w = tl.load(nw + d).to(tl.float32)
    xhat = x * r
    gw = dnorm * w
    mean = tl.sum(gw * xhat, axis=0) / 128.0
    tl.store(dh + mbase + d, r * (gw - xhat * mean))

@triton.jit
def _rms_unrope_dw_kernel(g, mm, rptr, cos, sin, dnw, ROWS: tl.constexpr, HEADS: tl.constexpr, BLOCK: tl.constexpr):
    d = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    od = tl.where(d < 64, d + 64, d - 64)
    sgn = tl.where(d < 64, 1.0, -1.0)
    for start in range(0, ROWS, BLOCK):
        row = start + lane
        bt = row // HEADS
        h = row - bt * HEADS
        t = bt & 2047
        b = bt >> 11
        mask = row < ROWS
        gbase = ((b * HEADS + h) * 2048 + t) * 128
        g0 = tl.load(g + gbase + d, mask=mask, other=0.0).to(tl.float32)
        gp = tl.load(g + gbase + od, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos + t * 128 + d, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin + t * 128 + od, mask=mask, other=0.0).to(tl.float32)
        dnorm = g0 * c + sgn * gp * s
        x = tl.load(mm + row * 128 + d, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(rptr + row, mask=mask, other=0.0).to(tl.float32)
        acc += dnorm * x * r
    tl.store(dnw + d, tl.sum(acc, axis=0))

def rms_unrope_bwd128(g, mm, r, nw, cos, sin, heads: int):
    dh = torch.empty_like(mm)
    dnw = torch.empty((128,), device=mm.device, dtype=nw.dtype)
    rows = 4096 * heads
    _rms_unrope_dh_kernel[(rows,)](g, mm, r, nw, cos, sin, dh, HEADS=heads, BLOCK=128)
    _rms_unrope_dw_kernel[(128,)](g, mm, r, cos, sin, dnw, ROWS=rows, HEADS=heads, BLOCK=1024)
    return dh, dnw

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4, t_primals_5, t_primals_6, t_primals_7, t_primals_8):
    t_permute = t_primals_2.permute(1, 0)  # metadata transpose for extern mm
    t_view = t_primals_1.reshape([4096, 1024])  # aten.view.default
    t_mm = _unified_external_forward_t_mm(t_view, t_permute)  # aten.mm.default
    t_add_2, t_rsqrt = rms_rope128(t_mm, t_primals_5, t_primals_7, t_primals_8, 16)
    t_permute_2 = t_primals_3.permute(1, 0)  # metadata transpose for extern mm
    t_mm_1 = _unified_external_forward_t_mm_1(t_view, t_permute_2)  # aten.mm.default
    t_add_3, t_rsqrt_1 = rms_rope128(t_mm_1, t_primals_6, t_primals_7, t_primals_8, 8)
    t_permute_4 = t_primals_4.permute(1, 0)  # metadata transpose for extern mm
    t_mm_2 = _unified_external_forward_t_mm_2(t_view, t_permute_4)  # aten.mm.default
    t_view_7 = t_mm_2.reshape([2, 2048, 1024])  # aten.view.default
    t_view_8 = t_view_7.reshape([2, 2048, -1, 128])  # aten.view.default
    t_permute_5 = permute_copy(t_view_8, [0, 2, 1, 3])  # aten.permute.default
    # q/k RMSNorm and RoPE are fused above
    t_permute_9 = t_primals_4
    t_permute_14 = t_primals_3
    t_permute_19 = t_primals_2
    return (t_add_2, t_add_3, t_permute_5, t_primals_5, t_primals_6, t_primals_7, t_primals_8, t_view, t_mm, t_rsqrt, t_mm_1, t_rsqrt_1, t_permute_9, t_permute_14, t_permute_19)

def _unified_backward_graph(t_primals_5, t_primals_6, t_primals_7, t_primals_8, t_view, t_mm, t_rsqrt, t_mm_1, t_rsqrt_1, t_permute_9, t_permute_14, t_permute_19, t_tangents_1, t_tangents_2, t_tangents_3):
    # q/k unRoPE is fused into RMSNorm backward below
    t_permute_6 = permute_copy(t_tangents_3, [0, 2, 1, 3])  # aten.permute.default
    t_view_9 = t_permute_6.reshape([2, 2048, 1024])  # aten.view.default
    t_view_10 = t_view_9.reshape([4096, 1024])  # aten.view.default
    t_permute_7 = t_view_10.permute(1, 0)  # metadata transpose for extern mm
    t_mm_3 = _unified_external_backward_t_mm_3(t_permute_7, t_view)  # aten.mm.default
    t_permute_8 = t_mm_3
    t_mm_4 = _unified_external_backward_t_mm_4(t_view_10, t_permute_9)  # aten.mm.default
    t_view_11 = t_mm_4.reshape([2, 2048, 1024])  # aten.view.default
    t_permute_10 = t_permute_8
    t_view_14, t_view_12 = rms_unrope_bwd128(t_tangents_2, t_mm_1, t_rsqrt_1, t_primals_6, t_primals_7, t_primals_8, 8)
    t_permute_12 = t_view_14.permute(1, 0)  # metadata transpose for extern mm
    t_mm_5 = _unified_external_backward_t_mm_5(t_permute_12, t_view)  # aten.mm.default
    t_permute_13 = t_mm_5
    t_mm_6 = _unified_external_backward_t_mm_6(t_view_14, t_permute_14)  # aten.mm.default
    t_view_15 = t_mm_6.reshape([2, 2048, 1024])  # aten.view.default
    t_add_9 = _k6(t_view_11, t_view_15)  # aten.add.Tensor
    t_permute_15 = t_permute_13
    t_view_18, t_view_16 = rms_unrope_bwd128(t_tangents_1, t_mm, t_rsqrt, t_primals_5, t_primals_7, t_primals_8, 16)
    t_permute_17 = t_view_18.permute(1, 0)  # metadata transpose for extern mm
    t_mm_7 = _unified_external_backward_t_mm_7(t_permute_17, t_view)  # aten.mm.default
    t_permute_18 = t_mm_7
    t_mm_8 = _unified_external_backward_t_mm_8(t_view_18, t_permute_19)  # aten.mm.default
    t_view_19 = t_mm_8.reshape([2, 2048, 1024])  # aten.view.default
    t_add_11 = _k6(t_add_9, t_view_19)  # aten.add.Tensor
    t_permute_20 = t_permute_18
    return (t_add_11, t_permute_20, t_permute_15, t_permute_10, t_view_16, t_view_12, None, None)

def qwen3_qkv_norm_rope_forward_with_saved(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps=1e-6):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (2, 2048, 1024) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(q_weight, torch.Tensor) or not q_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor q_weight')
    if tuple(q_weight.shape) != (2048, 1024) or q_weight.dtype != torch.bfloat16 or q_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for q_weight')
    if not isinstance(k_weight, torch.Tensor) or not k_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor k_weight')
    if tuple(k_weight.shape) != (1024, 1024) or k_weight.dtype != torch.bfloat16 or k_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for k_weight')
    if not isinstance(v_weight, torch.Tensor) or not v_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor v_weight')
    if tuple(v_weight.shape) != (1024, 1024) or v_weight.dtype != torch.bfloat16 or v_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for v_weight')
    if not isinstance(q_norm_weight, torch.Tensor) or not q_norm_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor q_norm_weight')
    if tuple(q_norm_weight.shape) != (128,) or q_norm_weight.dtype != torch.bfloat16 or q_norm_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for q_norm_weight')
    if not isinstance(k_norm_weight, torch.Tensor) or not k_norm_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor k_norm_weight')
    if tuple(k_norm_weight.shape) != (128,) or k_norm_weight.dtype != torch.bfloat16 or k_norm_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for k_norm_weight')
    if not isinstance(cos, torch.Tensor) or not cos.is_cuda:
        raise ValueError('unified seed requires CUDA tensor cos')
    if tuple(cos.shape) != (1, 2048, 128) or cos.dtype != torch.bfloat16 or cos.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for cos')
    if not isinstance(sin, torch.Tensor) or not sin.is_cuda:
        raise ValueError('unified seed requires CUDA tensor sin')
    if tuple(sin.shape) != (1, 2048, 128) or sin.dtype != torch.bfloat16 or sin.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for sin')
    if any(t.device != x.device for t in (q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    if eps != 1e-06:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin)
    return _outs[:3], tuple(_outs[3:])

def qwen3_qkv_norm_rope_backward_from_saved(output_grads, saved_tensors, eps=1e-6):
    if eps != 1e-06:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    dq, dk, dv = output_grads
    if len(saved_tensors) != 12:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (128,) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (128,) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (1, 2048, 128) or saved_tensors[2].dtype != torch.bfloat16 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[3], torch.Tensor) or not saved_tensors[3].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[3].shape) != (1, 2048, 128) or saved_tensors[3].dtype != torch.bfloat16 or saved_tensors[3].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[4], torch.Tensor) or not saved_tensors[4].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[4].shape) != (4096, 1024) or saved_tensors[4].dtype != torch.bfloat16 or saved_tensors[4].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[5], torch.Tensor) or not saved_tensors[5].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[5].shape) != (4096, 2048) or saved_tensors[5].dtype != torch.bfloat16 or saved_tensors[5].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[6], torch.Tensor) or not saved_tensors[6].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[6].shape) != (2, 2048, 16, 1) or saved_tensors[6].dtype != torch.float32 or saved_tensors[6].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[7], torch.Tensor) or not saved_tensors[7].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[7].shape) != (4096, 1024) or saved_tensors[7].dtype != torch.bfloat16 or saved_tensors[7].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[8], torch.Tensor) or not saved_tensors[8].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[8].shape) != (2, 2048, 8, 1) or saved_tensors[8].dtype != torch.float32 or saved_tensors[8].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[9], torch.Tensor) or not saved_tensors[9].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[9].shape) != (1024, 1024) or saved_tensors[9].dtype != torch.bfloat16 or saved_tensors[9].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[10], torch.Tensor) or not saved_tensors[10].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[10].shape) != (1024, 1024) or saved_tensors[10].dtype != torch.bfloat16 or saved_tensors[10].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[11], torch.Tensor) or not saved_tensors[11].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[11].shape) != (2048, 1024) or saved_tensors[11].dtype != torch.bfloat16 or saved_tensors[11].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dq, torch.Tensor) or not dq.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dq.shape) != (2, 16, 2048, 128) or dq.dtype != torch.bfloat16 or dq.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dk, torch.Tensor) or not dk.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dk.shape) != (2, 8, 2048, 128) or dk.dtype != torch.bfloat16 or dk.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dv, torch.Tensor) or not dv.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dv.shape) != (2, 8, 2048, 128) or dv.dtype != torch.bfloat16 or dv.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], saved_tensors[4], saved_tensors[5], saved_tensors[6], saved_tensors[7], saved_tensors[8], saved_tensors[9], saved_tensors[10], saved_tensors[11], dq.contiguous(), dk.contiguous(), dv.contiguous())
    return (_grads[0], _grads[1], _grads[2], _grads[3], _grads[4], _grads[5])


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_qkv_norm_rope', 'arguments': ['x', 'q_weight', 'k_weight', 'v_weight', 'q_norm_weight', 'k_norm_weight', 'cos', 'sin', 'eps'], 'outputs': ['q', 'k', 'v'], 'upstream_grads': ['dq', 'dk', 'dv'], 'input_grads': ['dx', 'dq_weight', 'dk_weight', 'dv_weight', 'dq_norm_weight', 'dk_norm_weight'], 'gradient_return_order': ['dx', 'dq_weight', 'dk_weight', 'dv_weight', 'dq_norm_weight', 'dk_norm_weight', 'None', 'None', 'None'], 'scalars': ['eps'], 'deployment_entry': 'qwen3_qkv_norm_rope_deployment', 'allowed_primitives': []}


class Qwen3QkvNormRopeFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_qkv_norm_rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps):
        (q, k, v), saved = qwen3_qkv_norm_rope_forward_with_saved(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return q, k, v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dq = dq.contiguous() if (dq is not None and not dq.is_contiguous()) else dq
        dk = dk.contiguous() if (dk is not None and not dk.is_contiguous()) else dk
        dv = dv.contiguous() if (dv is not None and not dv.is_contiguous()) else dv
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dq_weight, dk_weight, dv_weight, dq_norm_weight, dk_norm_weight = qwen3_qkv_norm_rope_backward_from_saved(
            (dq, dk, dv), saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dq_weight, dk_weight, dv_weight, dq_norm_weight, dk_norm_weight, None, None, None


def qwen3_qkv_norm_rope_deployment(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Qwen3QkvNormRopeFunction.apply(x, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, cos, sin, eps)


class Qwen3QkvNormRopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, eps=1e-06):
        super().__init__()
        self.q_weight = (q_weight if isinstance(q_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(q_weight))
        self.k_weight = (k_weight if isinstance(k_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(k_weight))
        self.v_weight = (v_weight if isinstance(v_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(v_weight))
        self.q_norm_weight = (q_norm_weight if isinstance(q_norm_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(q_norm_weight))
        self.k_norm_weight = (k_norm_weight if isinstance(k_norm_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(k_norm_weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return qwen3_qkv_norm_rope_deployment(x, self.q_weight, self.k_weight, self.v_weight, self.q_norm_weight, self.k_norm_weight, cos, sin, self.eps)


DEPLOYMENT_ENTRY = "qwen3_qkv_norm_rope_deployment"

