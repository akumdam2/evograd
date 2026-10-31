"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'addmm', 'target': 'aten.addmm.default', 'kernel': 'extern_kernels.addmm', 'wrapper': '_unified_external_forward_t_addmm'}, {'side': 'forward', 'node_name': 'addmm_1', 'target': 'aten.addmm.default', 'kernel': 'extern_kernels.addmm', 'wrapper': '_unified_external_forward_t_addmm_1'}, {'side': 'backward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm'}, {'side': 'backward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_1'}, {'side': 'backward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_2'}, {'side': 'backward', 'node_name': 'mm_3', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_3'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 32, 'Y': 28, 'X': 28, 'H': 192, 'I': 768}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_addmm(arg0, arg1, arg2):
    with torch.no_grad():
        out = torch.empty((25088, 768), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.addmm(arg0, arg1, arg2, out=out)
        return out


def _unified_external_forward_t_addmm_1(arg0, arg1, arg2):
    with torch.no_grad():
        out = torch.empty((25088, 192), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.addmm(arg0, arg1, arg2, out=out)
        return out


def _unified_external_backward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((25088, 768), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((192, 768), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((25088, 192), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_3(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((768, 192), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

def mul_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    return torch.mul(a, scalar)

def _scalar_input(a, scalar, *, true_divide=False):
    dtype = torch.result_type(a, scalar)
    if true_divide and (not dtype.is_floating_point) and (not dtype.is_complex):
        dtype = torch.get_default_dtype()
    return a.to(dtype).contiguous()

@triton.jit
def _mul_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a * scalar, mask=mask)

def _grid(n: int):
    import math
    return (math.ceil(n / _BLOCK),)

def erf_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_erf_kernel, a)

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
def _erf_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, libdevice.erf(a), mask=mask)

def add_scalar(a: torch.Tensor, scalar, alpha=1) -> torch.Tensor:
    return torch.add(a, scalar, alpha=alpha)

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

def mul_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mul(a, b)

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

def div_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.div(a, b)

@triton.jit
def _div_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, a / b, mask=mask)

def add_tt(a: torch.Tensor, b: torch.Tensor, alpha=1) -> torch.Tensor:
    return torch.add(a, b, alpha=alpha)

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

def neg(a: torch.Tensor) -> torch.Tensor:
    return torch.neg(a)

@triton.jit
def _neg_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, -a, mask=mask)

def div_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    return torch.div(a, scalar)

@triton.jit
def _div_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a / scalar, mask=mask)

def eq_scalar(a, s):
    return _cmp_scalar(_eq_scalar_kernel, a, s)

def _cmp_scalar(kernel, a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out

def _bool_out(a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a, dtype=torch.bool)

@triton.jit
def _eq_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (a == scalar).to(tl.int8), mask=mask)

def fill_tensor(shape, value, *, dtype, device='cuda'):
    """Allocate a constant tensor without a PyTorch fill kernel."""
    out = torch.empty(tuple(shape), dtype=dtype, device=device)
    if out.numel():
        _fill_scalar_kernel[_grid(out.numel())](out, value, N=out.numel(), BLOCK=_BLOCK)
    return out

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

def exp(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_exp_kernel, a)

@triton.jit
def _exp_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(a), mask=mask)

def gelu_bf16(a: torch.Tensor) -> torch.Tensor:
    a = a.contiguous()
    out = torch.empty_like(a)
    _gelu_kernel[_grid(a.numel())](a, out, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _gelu_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, y, mask=mask)

def gelu_val_grad_bf16(a: torch.Tensor):
    a = a.contiguous()
    y = torch.empty_like(a)
    g = torch.empty_like(a, dtype=torch.float32)
    _gelu_val_grad_kernel[_grid(a.numel())](a, y, g, N=a.numel(), BLOCK=_BLOCK)
    return y, g

@triton.jit
def _gelu_val_grad_kernel(a_ptr, y_ptr, g_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = libdevice.erf(x * 0.7071067811865476)
    y = 0.5 * x * (1.0 + e)
    g = 0.5 * (1.0 + e) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(g_ptr + offs, g, mask=mask)

def grn_finish_bf16(dy, gw, gelu_y, gx, den, ssum):
    ch = torch.empty_like(den)
    _grn_chsum_kernel[(32,)](ssum, gx, den, ch, BLOCK=1024)
    out = torch.empty_like(gelu_y)
    _grn_finish_kernel[_grid(gelu_y.numel())](dy, gw, gelu_y, gx, den, ssum, ch, out, N=gelu_y.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _grn_chsum_kernel(ssum, gx, den, out, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    mask = c < 768
    d = tl.load(den + b).to(tl.float32)
    s = tl.load(ssum + b * 768 + c, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gx + b * 768 + c, mask=mask, other=0.0).to(tl.float32)
    v = -s * g / (d * d)
    tl.store(out + b, tl.sum(tl.where(mask, v, 0.0), axis=0))

@triton.jit
def _grn_finish_kernel(dy, gw, gelu_y, gx, den, ssum, ch, out, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    c = offs % 768
    b = offs // 602112
    y = tl.load(gelu_y + offs, mask=mask, other=0.0).to(tl.float32)
    dyo = tl.load(dy + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(gw + c, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gx + b * 768 + c, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(den + b, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(ssum + b * 768 + c, mask=mask, other=0.0).to(tl.float32)
    cs = tl.load(ch + b, mask=mask, other=0.0).to(tl.float32)
    q = dyo * w
    direct = dyo + q * (g / d)
    dldg = s / d + cs * 0.0013020833333333333
    ratio = tl.where(g == 0.0, 0.0, y / g)
    tl.store(out + offs, direct + dldg * ratio, mask=mask)

def _k0(a, s=0.5):
    return mul_scalar(a, s)

def _k1(a, s=0.7071067811865476):
    return mul_scalar(a, s)

def _k2(a, alpha=1, s=1):
    return add_scalar(a, s, alpha)

def _k3(a, exp_val=2):
    return pow_scalar(a, exp_val)

def _sum_keep(a, dims):
    r = torch.sum(a, dim=tuple(dims), keepdim=True, dtype=torch.float32)
    return r if a.dtype == torch.float32 else r.to(a.dtype)

def _mean_keep(a, dims):
    r = torch.mean(a, dim=tuple(dims), keepdim=True, dtype=torch.float32)
    return r if a.dtype == torch.float32 else r.to(a.dtype)

def _k4(a, _dims=[1, 2], _dtype=None, _keepdim=True):
    return _sum_keep(a, _dims)

def _k5(a, exp_val=0.5):
    return pow_scalar(a, exp_val)

def _k6(a, _dims=[-1], _dtype=None, _keepdim=True):
    return _mean_keep(a, _dims)

def _k7(a, alpha=1, s=1e-06):
    return add_scalar(a, s, alpha)

def _k8(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k9(a, _dims=[0], _dtype=None, _keepdim=True):
    return _sum_keep(a, _dims)

def _k10(a, _dims=[0, 1, 2], _dtype=None, _keepdim=True):
    return _sum_keep(a, _dims)

def _k11(a, _dims=[3], _dtype=None, _keepdim=True):
    return _sum_keep(a, _dims)

def _k12(a, s=768):
    return div_scalar(a, s)

def _k13(a, fn_s=eq_scalar, s=0.0):
    return fn_s(a, s)

def _k14(out_dtype=torch.bfloat16, val=0.0):
    return fill_tensor((), val, dtype=out_dtype, device='cuda')

def _k15(a, s=-0.5):
    return mul_scalar(a, s)

def _k16(a, s=0.3989422804014327):
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
    """Fast path 2D transpose as a view; fall back to materialized copy otherwise."""
    dims = [int(d) % a.ndim for d in dims]
    if a.ndim == 2 and dims == [1, 0]:
        return a.t()
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

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4, t_primals_5, t_primals_6, t_primals_7):
    t_view = t_primals_1.reshape([25088, 192])  # aten.view.default
    t_permute = permute_copy(t_primals_2, [1, 0])  # aten.permute.default
    t_addmm = _unified_external_forward_t_addmm(t_primals_3, t_view, t_permute)  # aten.addmm.default
    t_view_1 = t_addmm.reshape([32, 28, 28, 768])  # aten.view.default
    t_convert_element_type_4 = gelu_bf16(t_view_1)  # fused exact GELU, bf16 output
    t_convert_element_type_5 = t_convert_element_type_4.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_pow_1 = _k3(t_convert_element_type_5)  # aten.pow.Tensor_Scalar
    t_sum_1 = _k4(t_pow_1)  # aten.sum.dim_IntList
    t_pow_2 = _k5(t_sum_1)  # aten.pow.Tensor_Scalar
    t_convert_element_type_6 = t_pow_2.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_mean = _k6(t_convert_element_type_6)  # aten.mean.dim
    t_add_1 = _k7(t_mean)  # aten.add.Tensor
    t_div = div_tt(t_convert_element_type_6, t_add_1)  # aten.div.Tensor
    t_mul_3 = mul_tt(t_convert_element_type_4, t_div)  # aten.mul.Tensor
    t_mul_4 = mul_tt(t_primals_4, t_mul_3)  # aten.mul.Tensor
    t_add_2 = _k8(t_mul_4, t_primals_5)  # aten.add.Tensor
    t_add_3 = _k8(t_add_2, t_convert_element_type_4)  # aten.add.Tensor
    t_view_2 = t_add_3.reshape([25088, 768])  # aten.view.default
    t_permute_1 = permute_copy(t_primals_6, [1, 0])  # aten.permute.default
    t_addmm_1 = _unified_external_forward_t_addmm_1(t_primals_7, t_view_2, t_permute_1)  # aten.addmm.default
    t_view_3 = t_addmm_1.reshape([32, 28, 28, 192])  # aten.view.default
    return (t_view_3, t_primals_4, t_view, t_addmm, t_convert_element_type_6, t_add_1, t_view_2, t_primals_6, t_primals_2)

def _unified_backward_graph(t_primals_4, t_view, t_addmm, t_convert_element_type_6, t_add_1, t_view_2, t_permute_2, t_permute_6, t_tangents_1):
    t_view_4 = t_tangents_1.reshape([25088, 192])  # aten.view.default
    t_mm = _unified_external_backward_t_mm(t_view_4, t_permute_2)  # aten.mm.default
    t_permute_3 = permute_copy(t_view_4, [1, 0])  # aten.permute.default
    t_mm_1 = _unified_external_backward_t_mm_1(t_permute_3, t_view_2)  # aten.mm.default
    t_sum_2 = _k9(t_view_4)  # aten.sum.dim_IntList
    t_view_5 = t_sum_2.reshape([192])  # aten.view.default
    t_view_6 = t_mm.reshape([32, 28, 28, 768])  # aten.view.default
    t_sum_3 = _k10(t_view_6)  # aten.sum.dim_IntList
    t_mul_5 = mul_tt(t_view_6, t_primals_4)  # aten.mul.Tensor
    t_view_1 = t_addmm.reshape([32, 28, 28, 768])  # aten.view.default
    t_convert_element_type_4, t_add_8 = gelu_val_grad_bf16(t_view_1)  # fused GELU value + derivative
    t_div = div_tt(t_convert_element_type_6, t_add_1)  # aten.div.Tensor
    t_mul_3 = mul_tt(t_convert_element_type_4, t_div)  # aten.mul.Tensor
    t_mul_6 = mul_tt(t_view_6, t_mul_3)  # aten.mul.Tensor
    t_sum_4 = _k10(t_mul_6)  # aten.sum.dim_IntList
    t_mul_7 = mul_tt(t_mul_5, t_convert_element_type_4)  # aten.mul.Tensor
    t_sum_5 = _k4(t_mul_7)  # aten.sum.dim_IntList
    t_add_6 = grn_finish_bf16(t_view_6, t_primals_4, t_convert_element_type_4, t_convert_element_type_6, t_add_1, t_sum_5)
    t_convert_element_type_14 = t_add_6.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_mul_17 = mul_tt(t_convert_element_type_14, t_add_8)  # aten.mul.Tensor
    t_convert_element_type_16 = t_mul_17.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_view_7 = t_convert_element_type_16.reshape([25088, 768])  # aten.view.default
    t_mm_2 = _unified_external_backward_t_mm_2(t_view_7, t_permute_6)  # aten.mm.default
    t_permute_7 = permute_copy(t_view_7, [1, 0])  # aten.permute.default
    t_mm_3 = _unified_external_backward_t_mm_3(t_permute_7, t_view)  # aten.mm.default
    t_sum_7 = _k9(t_view_7)  # aten.sum.dim_IntList
    t_view_8 = t_sum_7.reshape([768])  # aten.view.default
    t_view_9 = t_mm_2.reshape([32, 28, 28, 192])  # aten.view.default
    return (t_view_9, t_mm_3, t_view_8, t_sum_4, t_sum_3, t_mm_1, t_view_5)

def convnextv2_tiny_convnext_mlp_forward_with_saved(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (32, 28, 28, 192) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(up_weight, torch.Tensor) or not up_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor up_weight')
    if tuple(up_weight.shape) != (768, 192) or up_weight.dtype != torch.bfloat16 or up_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for up_weight')
    if not isinstance(up_bias, torch.Tensor) or not up_bias.is_cuda:
        raise ValueError('unified seed requires CUDA tensor up_bias')
    if tuple(up_bias.shape) != (768,) or up_bias.dtype != torch.bfloat16 or up_bias.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for up_bias')
    if not isinstance(grn_weight, torch.Tensor) or not grn_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor grn_weight')
    if tuple(grn_weight.shape) != (1, 1, 1, 768) or grn_weight.dtype != torch.bfloat16 or grn_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for grn_weight')
    if not isinstance(grn_bias, torch.Tensor) or not grn_bias.is_cuda:
        raise ValueError('unified seed requires CUDA tensor grn_bias')
    if tuple(grn_bias.shape) != (1, 1, 1, 768) or grn_bias.dtype != torch.bfloat16 or grn_bias.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for grn_bias')
    if not isinstance(down_weight, torch.Tensor) or not down_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor down_weight')
    if tuple(down_weight.shape) != (192, 768) or down_weight.dtype != torch.bfloat16 or down_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for down_weight')
    if not isinstance(down_bias, torch.Tensor) or not down_bias.is_cuda:
        raise ValueError('unified seed requires CUDA tensor down_bias')
    if tuple(down_bias.shape) != (192,) or down_bias.dtype != torch.bfloat16 or down_bias.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for down_bias')
    if any(t.device != x.device for t in (up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
    return _outs[0], tuple(_outs[1:])

def convnextv2_tiny_convnext_mlp_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 8:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (1, 1, 1, 768) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (25088, 192) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (25088, 768) or saved_tensors[2].dtype != torch.bfloat16 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[3], torch.Tensor) or not saved_tensors[3].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[3].shape) != (32, 1, 1, 768) or saved_tensors[3].dtype != torch.bfloat16 or saved_tensors[3].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[4], torch.Tensor) or not saved_tensors[4].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[4].shape) != (32, 1, 1, 1) or saved_tensors[4].dtype != torch.bfloat16 or saved_tensors[4].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[5], torch.Tensor) or not saved_tensors[5].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[5].shape) != (25088, 768) or saved_tensors[5].dtype != torch.bfloat16 or saved_tensors[5].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[6], torch.Tensor) or not saved_tensors[6].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[6].shape) != (192, 768) or saved_tensors[6].dtype != torch.bfloat16 or saved_tensors[6].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[7], torch.Tensor) or not saved_tensors[7].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[7].shape) != (768, 192) or saved_tensors[7].dtype != torch.bfloat16 or saved_tensors[7].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (32, 28, 28, 192) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], saved_tensors[4], saved_tensors[5], saved_tensors[6], saved_tensors[7], dout.contiguous())
    return (_grads[0], _grads[1], _grads[2], _grads[3], _grads[4], _grads[5], _grads[6])


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'convnextv2_tiny_convnext_mlp', 'arguments': ['x', 'up_weight', 'up_bias', 'grn_weight', 'grn_bias', 'down_weight', 'down_bias'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'gradient_return_order': ['dx', 'dup_weight', 'dup_bias', 'dgrn_weight', 'dgrn_bias', 'ddown_weight', 'ddown_bias'], 'scalars': [], 'deployment_entry': 'convnextv2_tiny_convnext_mlp_deployment', 'allowed_primitives': []}


class Convnextv2TinyConvnextMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `convnextv2_tiny_convnext_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        out, saved = convnextv2_tiny_convnext_mlp_forward_with_saved(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
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
        dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias = convnextv2_tiny_convnext_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dup_weight, dup_bias, dgrn_weight, dgrn_bias, ddown_weight, ddown_bias


def convnextv2_tiny_convnext_mlp_deployment(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        grn_weight = grn_weight.reshape(-1, grn_weight.shape[-1])
        grn_bias = grn_bias.reshape(-1, grn_bias.shape[-1])
    out = Convnextv2TinyConvnextMlpFunction.apply(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Convnextv2TinyConvnextMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
        super().__init__()
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.up_bias = (up_bias if isinstance(up_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(up_bias))
        self.grn_weight = (grn_weight if isinstance(grn_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_weight))
        self.grn_bias = (grn_bias if isinstance(grn_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(grn_bias))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))
        self.down_bias = (down_bias if isinstance(down_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(down_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return convnextv2_tiny_convnext_mlp_deployment(x, self.up_weight, self.up_bias, self.grn_weight, self.grn_bias, self.down_weight, self.down_bias)


DEPLOYMENT_ENTRY = "convnextv2_tiny_convnext_mlp_deployment"

