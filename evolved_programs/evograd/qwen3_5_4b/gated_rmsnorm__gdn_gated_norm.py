"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 65536, 'hidden': 128}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

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

def reciprocal(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_reciprocal_kernel, a)

@triton.jit
def _reciprocal_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(out_ptr + offs, 1.0 / a, mask=mask)

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

def _k0(a, exp_val=2):
    return pow_scalar(a, exp_val)

def _k1(a, _dims=[-1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'mean', _dtype)

def _k2(a, alpha=1, s=1e-06):
    return add_scalar(a, s, alpha)

def _k3(a, alpha=1, s=1):
    return add_scalar(a, s, alpha)

def _k4(a, s=1):
    return mul_scalar(a, s)

def _k5(a, alpha=1, s=1):
    return rsub_scalar(a, s, alpha)

def _k6(a, _dims=[0], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k7(a, _dims=[1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k8(a, exp_val=3):
    return pow_scalar(a, exp_val)

def _k9(a, s=-0.5):
    return mul_scalar(a, s)

def _k10(a, s=128):
    return div_scalar(a, s)

def _k11(a, exp_val=1.0):
    return pow_scalar(a, exp_val)

def _k12(a, s=2.0):
    return mul_scalar(a, s)

def _k13(a, b, alpha=1):
    return add_tt(a, b, alpha)

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
def _grn_fwd_kernel(X, Z, W, Y, EPS: tl.constexpr, C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    base = rows[:, None] * C + cols[None, :]

    x = tl.load(X + base).to(tl.float32)
    rr = 1.0 / tl.sqrt(tl.sum(tl.trans(x * x), axis=0) / C + EPS)

    xh_b = (x * rr[:, None]).to(tl.bfloat16)
    w = tl.load(W + cols).to(tl.bfloat16)
    n_b = (w[None, :] * xh_b).to(tl.bfloat16)

    z = tl.load(Z + base).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-z))
    tl.store(Y + base, n_b.to(tl.float32) * (z * sig))


@triton.jit
def _grn_bwd_fused_kernel(X, Z, W, DY, DX, DZ, P, EPS: tl.constexpr, C: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    bid = tl.program_id(0)
    rows = bid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    base = rows[:, None] * C + cols[None, :]

    x = tl.load(X + base).to(tl.float32)
    z = tl.load(Z + base).to(tl.float32)
    dy = tl.load(DY + base).to(tl.float32)
    w = tl.load(W + cols).to(tl.float32)

    rr = 1.0 / tl.sqrt(tl.sum(tl.trans(x * x), axis=0) / C + EPS)
    xh = x * rr[:, None]

    sig = 1.0 / (1.0 + tl.exp(-z))
    s = z * sig
    sp = sig * (1.0 + z * (1.0 - sig))

    g = dy * s * w[None, :]
    dot = tl.sum(tl.trans(g * xh), axis=0) / C

    tl.store(DX + base, rr[:, None] * (g - xh * dot[:, None]))
    tl.store(DZ + base, dy * (w[None, :] * xh) * sp)
    tl.store(P + bid * C + cols, tl.sum(dy * s * xh, axis=0))


@triton.jit
def _grn_dw_reduce_kernel(P, DW, NB: tl.constexpr, C: tl.constexpr, BLOCK_B: tl.constexpr):
    col = tl.program_id(0)
    offs = tl.arange(0, BLOCK_B)
    vals = tl.load(P + offs * C + col).to(tl.float32)
    tl.store(DW + col, tl.sum(vals, axis=0))


def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3):
    t_convert_element_type = t_primals_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_pow_1 = _k0(t_convert_element_type)  # aten.pow.Tensor_Scalar
    t_mean = _k1(t_pow_1)  # aten.mean.dim
    t_add = _k2(t_mean)  # aten.add.Tensor
    t_rsqrt = rsqrt(t_add)  # aten.rsqrt.default
    t_mul = mul_tt(t_convert_element_type, t_rsqrt)  # aten.mul.Tensor
    t_convert_element_type_1 = t_mul.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_mul_1 = mul_tt(t_primals_3, t_convert_element_type_1)  # aten.mul.Tensor
    t_convert_element_type_2 = t_primals_2.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_neg = neg(t_convert_element_type_2)  # aten.neg.default
    t_exp = exp(t_neg)  # aten.exp.default
    t_add_1 = _k3(t_exp)  # aten.add.Tensor
    t_div = div_tt(t_convert_element_type_2, t_add_1)  # aten.div.Tensor
    t_mul_2 = mul_tt(t_mul_1, t_div)  # aten.mul.Tensor
    t_convert_element_type_3 = t_mul_2.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    return (t_convert_element_type_3, t_primals_1, t_primals_2, t_primals_3, t_rsqrt)

def _unified_backward_graph(t_primals_1, t_primals_2, t_primals_3, t_rsqrt, t_tangents_1):
    t_convert_element_type_4 = t_tangents_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_convert_element_type = t_primals_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_mul = mul_tt(t_convert_element_type, t_rsqrt)  # aten.mul.Tensor
    t_convert_element_type_1 = t_mul.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_mul_1 = mul_tt(t_primals_3, t_convert_element_type_1)  # aten.mul.Tensor
    t_mul_3 = mul_tt(t_convert_element_type_4, t_mul_1)  # aten.mul.Tensor
    t_convert_element_type_2 = t_primals_2.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_neg = neg(t_convert_element_type_2)  # aten.neg.default
    t_exp = exp(t_neg)  # aten.exp.default
    t_add_1 = _k3(t_exp)  # aten.add.Tensor
    t_div = div_tt(t_convert_element_type_2, t_add_1)  # aten.div.Tensor
    t_mul_4 = mul_tt(t_convert_element_type_4, t_div)  # aten.mul.Tensor
    t_convert_element_type_5 = t_mul_4.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_reciprocal = reciprocal(t_add_1)  # aten.reciprocal.default
    t_mul_5 = _k4(t_reciprocal)  # aten.mul.Tensor
    t_mul_6 = mul_tt(t_mul_3, t_mul_5)  # aten.mul.Tensor
    t_sub = _k5(t_mul_5)  # aten.sub.Tensor
    t_mul_7 = mul_tt(t_convert_element_type_2, t_sub)  # aten.mul.Tensor
    t_add_3 = _k3(t_mul_7)  # aten.add.Tensor
    t_mul_8 = mul_tt(t_mul_6, t_add_3)  # aten.mul.Tensor
    t_convert_element_type_6 = t_mul_8.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    t_mul_9 = mul_tt(t_convert_element_type_5, t_primals_3)  # aten.mul.Tensor
    t_mul_10 = mul_tt(t_convert_element_type_5, t_convert_element_type_1)  # aten.mul.Tensor
    t_sum_1 = _k6(t_mul_10)  # aten.sum.dim_IntList
    t_view = t_sum_1.reshape([128])  # aten.view.default
    t_convert_element_type_7 = t_mul_9.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_mul_11 = mul_tt(t_convert_element_type_7, t_convert_element_type)  # aten.mul.Tensor
    t_mul_12 = mul_tt(t_convert_element_type_7, t_rsqrt)  # aten.mul.Tensor
    t_sum_2 = _k7(t_mul_11)  # aten.sum.dim_IntList
    t_alias = t_rsqrt  # aten.alias.default
    t_alias_1 = t_alias  # aten.alias.default
    t_pow_2 = _k8(t_alias_1)  # aten.pow.Tensor_Scalar
    t_mul_13 = _k9(t_sum_2)  # aten.mul.Scalar
    t_mul_14 = mul_tt(t_mul_13, t_pow_2)  # aten.mul.Tensor
    t_expand = expand_copy(t_mul_14, [65536, 128])  # aten.expand.default
    t_div_1 = _k10(t_expand)  # aten.div.Scalar
    t_pow_3 = _k11(t_convert_element_type)  # aten.pow.Tensor_Scalar
    t_mul_15 = _k12(t_pow_3)  # aten.mul.Scalar
    t_mul_16 = mul_tt(t_div_1, t_mul_15)  # aten.mul.Tensor
    t_add_4 = _k13(t_mul_12, t_mul_16)  # aten.add.Tensor
    t_convert_element_type_8 = t_add_4.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    return (t_convert_element_type_8, t_convert_element_type_6, t_view)

def gated_rmsnorm_forward_with_saved(x, z, weight, eps=1e-6):
    if tuple(x.shape) != (65536, 128) or tuple(z.shape) != (65536, 128) or tuple(weight.shape) != (128,):
        raise ValueError('configuration mismatch')
    if eps != 1e-06:
        raise ValueError('eps mismatch')
    y = torch.empty_like(x)
    _grn_fwd_kernel[(16384,)](x, z, weight, y, eps, C=128, BLOCK_M=4, BLOCK_N=128, num_warps=4)
    return y, (x, z, weight)

def gated_rmsnorm_backward_from_saved(dy, saved_tensors, eps=1e-6):
    if eps != 1e-06 or len(saved_tensors) != 3:
        raise ValueError('configuration mismatch')
    x, z, weight = saved_tensors
    dx = torch.empty_like(x)
    dz = torch.empty_like(z)
    dweight = torch.empty_like(weight)

    BM = 16
    NB = 4096
    partials = torch.empty((NB, 128), device=x.device, dtype=torch.float32)
    _grn_bwd_fused_kernel[(NB,)](x, z, weight, dy, dx, dz, partials, eps, C=128, BLOCK_M=BM, BLOCK_N=128, num_warps=8)
    _grn_dw_reduce_kernel[(128,)](partials, dweight, NB=NB, C=128, BLOCK_B=4096, num_warps=8)

    return dx, dz, dweight


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gated_rmsnorm', 'arguments': ['x', 'z', 'weight', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dz', 'dweight'], 'gradient_return_order': ['dx', 'dz', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'gated_rmsnorm_deployment', 'allowed_primitives': []}


class GatedRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `gated_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, z, weight, eps):
        y, saved = gated_rmsnorm_forward_with_saved(x, z, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dy = dy.contiguous() if (dy is not None and not dy.is_contiguous()) else dy
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dz, dweight = gated_rmsnorm_backward_from_saved(
            dy, saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dz, dweight, None


def gated_rmsnorm_deployment(x, z, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        z = z.reshape(-1, z.shape[-1])
    y = GatedRmsnormFunction.apply(x, z, weight, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class GatedRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, z):
        return gated_rmsnorm_deployment(x, z, self.weight, self.eps)


DEPLOYMENT_ENTRY = "gated_rmsnorm_deployment"

