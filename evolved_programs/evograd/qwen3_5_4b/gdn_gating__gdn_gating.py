"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 2048, 'heads': 32}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

def sigmoid_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_sigmoid_kernel, a)

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

def _grid(n: int):
    import math
    return (math.ceil(n / _BLOCK),)

@triton.jit
def _sigmoid_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.sigmoid(a), mask=mask)

def exp(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_exp_kernel, a)

@triton.jit
def _exp_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(a), mask=mask)

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

def add_tt(a: torch.Tensor, b: torch.Tensor, alpha=1) -> torch.Tensor:
    a, b = _binary_inputs(a, b)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_kernel[_grid(a.numel())](a, b, out, alpha, N=a.numel(), BLOCK=_BLOCK)
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

def log1p_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_log1p_kernel, a)

@triton.jit
def _log1p_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.log(1.0 + a), mask=mask)

def gt_scalar(a, s):
    return _cmp_scalar(_gt_scalar_kernel, a, s)

def _cmp_scalar(kernel, a: torch.Tensor, scalar) -> torch.Tensor:
    a = a.contiguous()
    out = _bool_out(a)
    kernel[_grid(a.numel())](a, out, float(scalar), N=a.numel(), BLOCK=_BLOCK)
    return out

def _bool_out(a: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a, dtype=torch.bool)

@triton.jit
def _gt_scalar_kernel(a_ptr, out_ptr, scalar, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (a > scalar).to(tl.int8), mask=mask)

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

def mul_scalar(a: torch.Tensor, scalar) -> torch.Tensor:
    a = _scalar_input(a, scalar)
    out = torch.empty_like(a)
    _mul_scalar_kernel[_grid(a.numel())](a, out, scalar, N=a.numel(), BLOCK=_BLOCK)
    return out

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

def add_scalar(a: torch.Tensor, scalar, alpha=1) -> torch.Tensor:
    a = _scalar_input(a, scalar)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_scalar_kernel[_grid(a.numel())](a, out, scalar, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _add_scalar_kernel(a_ptr, out_ptr, scalar, alpha, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
    tl.store(out_ptr + offs, a + alpha * scalar, mask=mask)

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

def rsub_scalar(a: torch.Tensor, scalar, alpha=1) -> torch.Tensor:
    if a.dtype == torch.bool or isinstance(scalar, bool):
        raise RuntimeError('Subtraction with Boolean tensors is not supported')
    a = _scalar_input(a, scalar)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _rsub_scalar_kernel[_grid(a.numel())](a, out, scalar, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _rsub_scalar_kernel(a_ptr, out_ptr, scalar, alpha, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
    tl.store(out_ptr + offs, scalar - alpha * a, mask=mask)

def _k0(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k1(a, fn_s=gt_scalar, s=20.0):
    return fn_s(a, s)

def _k2(a, _dims=[0], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k3(a, s=1):
    return mul_scalar(a, s)

def _k4(a, alpha=1, s=1.0):
    return add_scalar(a, s, alpha)

def _k5(a, alpha=1, s=1):
    return rsub_scalar(a, s, alpha)

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

_BM_GDN = 16
_BH_GDN = 32

@triton.jit
def _gdn_fwd_kernel(a, b, A_log, dt_bias, g, beta,
                    ROWS: tl.constexpr, H: tl.constexpr,
                    BM: tl.constexpr, BH: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.arange(0, BH)
    m = (r[:, None] < ROWS) & (h[None, :] < H)
    off = r[:, None] * H + h[None, :]

    av = tl.load(a + off, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(b + off, mask=m, other=0.0).to(tl.float32)
    A = tl.exp(tl.load(A_log + h, mask=h < H, other=0.0).to(tl.float32))
    dt = tl.load(dt_bias + h, mask=h < H, other=0.0).to(tl.float32)

    u = av + dt[None, :]
    sp = tl.where(u > 20.0, u, tl.log(1.0 + tl.exp(u)))
    tl.store(g + off, -A[None, :] * sp, mask=m)
    tl.store(beta + off, tl.sigmoid(bv), mask=m)

@triton.jit
def _gdn_bwd_tile_kernel(a, b, A_log, dt_bias, dg, dbeta, da, db, part,
                         ROWS: tl.constexpr, H: tl.constexpr, NBLK: tl.constexpr,
                         BM: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BM + tl.arange(0, BM)
    h = tl.arange(0, BH)
    hm = h < H
    m = (r[:, None] < ROWS) & (h[None, :] < H)
    off = r[:, None] * H + h[None, :]

    av = tl.load(a + off, mask=m, other=0.0).to(tl.float32)
    bv = tl.load(b + off, mask=m, other=0.0).to(tl.float32)
    dgv = tl.load(dg + off, mask=m, other=0.0).to(tl.float32)
    dbetav = tl.load(dbeta + off, mask=m, other=0.0).to(tl.float32)

    A = tl.exp(tl.load(A_log + h, mask=hm, other=0.0).to(tl.float32))
    dt = tl.load(dt_bias + h, mask=hm, other=0.0).to(tl.float32)

    u = av + dt[None, :]
    sp = tl.where(u > 20.0, u, tl.log(1.0 + tl.exp(u)))
    gv = -A[None, :] * sp

    dsoft = dgv * (-A[None, :]) * tl.sigmoid(u)
    tl.store(da + off, dsoft, mask=m)

    betav = tl.sigmoid(bv)
    tl.store(db + off, dbetav * betav * (1.0 - betav), mask=m)

    tl.store(part + pid * H + h, tl.sum(dgv * gv, axis=0), mask=hm)
    tl.store(part + (NBLK * H) + pid * H + h, tl.sum(dsoft, axis=0), mask=hm)

@triton.jit
def _gdn_bwd_reduce_kernel(part, dA_log, ddt_bias,
                           NBLK: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < NBLK
    sA = tl.load(part + o * H + h, mask=m, other=0.0)
    sD = tl.load(part + (NBLK * H) + o * H + h, mask=m, other=0.0)
    tl.store(dA_log + h, tl.sum(sA, axis=0))
    tl.store(ddt_bias + h, tl.sum(sD, axis=0))

def gdn_gating_forward_with_saved(a, b, A_log, dt_bias):
    a = a.contiguous()
    b = b.contiguous()
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b)
    rows = a.shape[0]
    heads = a.shape[1]
    nblk = (rows + _BM_GDN - 1) // _BM_GDN
    _gdn_fwd_kernel[(nblk,)](
        a, b, A_log, dt_bias, g, beta,
        ROWS=rows, H=heads, BM=_BM_GDN, BH=_BH_GDN
    )
    return (g, beta), (a, b, A_log, dt_bias)

def gdn_gating_backward_from_saved(output_grads, saved_tensors):
    dg, dbeta = output_grads
    a, b, A_log, dt_bias = saved_tensors
    dg = dg.contiguous()
    dbeta = dbeta.contiguous()

    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dA_log = torch.empty_like(A_log)
    ddt_bias = torch.empty_like(dt_bias)

    rows = a.shape[0]
    heads = a.shape[1]
    nblk = (rows + _BM_GDN - 1) // _BM_GDN
    part = torch.empty((2, nblk, heads), device=a.device, dtype=torch.float32)

    _gdn_bwd_tile_kernel[(nblk,)](
        a, b, A_log, dt_bias, dg, dbeta, da, db, part,
        ROWS=rows, H=heads, NBLK=nblk, BM=_BM_GDN, BH=_BH_GDN
    )
    _gdn_bwd_reduce_kernel[(heads,)](
        part, dA_log, ddt_bias,
        NBLK=nblk, H=heads, BLOCK=_next_pow2(nblk)
    )
    return da, db, dA_log, ddt_bias


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gdn_gating', 'arguments': ['a', 'b', 'A_log', 'dt_bias'], 'outputs': ['g', 'beta'], 'upstream_grads': ['dg', 'dbeta'], 'input_grads': ['da', 'db', 'dA_log', 'ddt_bias'], 'gradient_return_order': ['da', 'db', 'dA_log', 'ddt_bias'], 'scalars': [], 'deployment_entry': 'gdn_gating_deployment', 'allowed_primitives': []}


class GdnGatingFunction(torch.autograd.Function):
    """Static autograd wiring for `gdn_gating`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, a, b, A_log, dt_bias):
        (g, beta), saved = gdn_gating_forward_with_saved(a, b, A_log, dt_bias)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
        return g, beta

    @staticmethod
    def backward(ctx, dg, dbeta):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dg = dg.contiguous() if (dg is not None and not dg.is_contiguous()) else dg
        dbeta = dbeta.contiguous() if (dbeta is not None and not dbeta.is_contiguous()) else dbeta
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        da, db, dA_log, ddt_bias = gdn_gating_backward_from_saved(
            (dg, dbeta), saved
        )
        # One value per forward argument, `None` for every inactive one.
        return da, db, dA_log, ddt_bias


def gdn_gating_deployment(a, b, A_log, dt_bias):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if a.dim() > 2:
        _leading = a.shape[:-1]
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    g, beta = GdnGatingFunction.apply(a, b, A_log, dt_bias)
    if _leading is not None:
        g = g.view(*_leading, g.shape[-1])
        beta = beta.view(*_leading, beta.shape[-1])
    return g, beta


class GdnGatingModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, A_log, dt_bias):
        super().__init__()
        self.A_log = (A_log if isinstance(A_log, torch.nn.Parameter)
                  else torch.nn.Parameter(A_log))
        self.dt_bias = (dt_bias if isinstance(dt_bias, torch.nn.Parameter)
                  else torch.nn.Parameter(dt_bias))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, a, b):
        return gdn_gating_deployment(a, b, self.A_log, self.dt_bias)


DEPLOYMENT_ENTRY = "gdn_gating_deployment"

