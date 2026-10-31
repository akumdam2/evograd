"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 4096, 'cols': 2048}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

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

def _grid(n: int):
    import math
    return (math.ceil(n / _BLOCK),)

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

def _k0(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k1(a, exp_val=2):
    return pow_scalar(a, exp_val)

def _k2(a, _dims=[-1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'mean', _dtype)

def _k3(a, alpha=1, s=1e-05):
    return add_scalar(a, s, alpha)

def _k4(a, _dims=[0], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k5(a, _dims=[1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k6(a, exp_val=3):
    return pow_scalar(a, exp_val)

def _k7(a, s=-0.5):
    return mul_scalar(a, s)

def _k8(a, s=2048):
    return div_scalar(a, s)

def _k9(a, exp_val=1.0):
    return pow_scalar(a, exp_val)

def _k10(a, s=2.0):
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
def _rrn_fwd_kernel(X, R, W, OUT, SUMMED, RSTD, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptrs = row * 2048 + offs
    x = tl.load(X + ptrs).to(tl.float32)
    r = tl.load(R + ptrs).to(tl.float32)
    # Materialize summed in bf16, then use that rounded value for the norm,
    # matching the saved/output residual stream.
    sb = (x + r).to(tl.bfloat16)
    s = sb.to(tl.float32)
    ss = tl.sum(s * s, axis=0)
    inv = tl.rsqrt(ss * 0.00048828125 + EPS)
    w = tl.load(W + offs).to(tl.float32)
    tl.store(SUMMED + ptrs, sb)
    tl.store(RSTD + row, inv)
    tl.store(OUT + ptrs, s * inv * w)

@triton.jit
def _rrn_bwd_dx_kernel(SUMMED, W, RSTD, DOUT, DSUMMED, DTOTAL, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptrs = row * 2048 + offs
    # s and dout are read again by the dweight reduction right after this
    # kernel, so bias them toward L2 retention.  dsummed is single-use.
    s = tl.load(SUMMED + ptrs, eviction_policy='evict_last').to(tl.float32)
    dout = tl.load(DOUT + ptrs, eviction_policy='evict_last').to(tl.float32)
    ds = tl.load(DSUMMED + ptrs, eviction_policy='evict_first').to(tl.float32)
    w = tl.load(W + offs, eviction_policy='evict_last').to(tl.float32)
    inv = tl.load(RSTD + row, eviction_policy='evict_last').to(tl.float32)
    dnorm = dout * w
    dot = tl.sum(dnorm * s, axis=0)
    corr = dot * 0.00048828125 * inv * inv
    dt = ds + inv * (dnorm - s * corr)
    tl.store(DTOTAL + ptrs, dt)

@triton.jit
def _rrn_dw_part_kernel(SUMMED, RSTD, DOUT, PART, BM: tl.constexpr, BN: tl.constexpr):
    gm = tl.program_id(0)
    gn = tl.program_id(1)
    rows = gm * BM + tl.arange(0, BM)
    cols = gn * BN + tl.arange(0, BN)
    ptrs = rows[:, None] * 2048 + cols[None, :]
    # Final use of these tensors in backward; don't retain them after the
    # partial dweight accumulation.
    s = tl.load(SUMMED + ptrs, eviction_policy='evict_first').to(tl.float32)
    dout = tl.load(DOUT + ptrs, eviction_policy='evict_first').to(tl.float32)
    inv = tl.load(RSTD + rows, eviction_policy='evict_first').to(tl.float32)[:, None]
    acc = tl.sum(dout * s * inv, axis=0)
    tl.store(PART + gm * 2048 + cols, acc)

@triton.jit
def _rrn_dw_final_kernel(PART, DW, GROUPS: tl.constexpr, BN: tl.constexpr, BG: tl.constexpr):
    gn = tl.program_id(0)
    cols = gn * BN + tl.arange(0, BN)
    g = tl.arange(0, BG)
    vals = tl.load(PART + g[:, None] * 2048 + cols[None, :], mask=g[:, None] < GROUPS, other=0.0)
    acc = tl.sum(vals, axis=0)
    tl.store(DW + cols, acc)

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3):
    out = torch.empty_like(t_primals_1)
    summed = torch.empty_like(t_primals_1)
    rstd = torch.empty((4096, 1), device=t_primals_1.device, dtype=torch.float32)
    _rrn_fwd_kernel[(4096,)](t_primals_1, t_primals_2, t_primals_3, out, summed, rstd, EPS=1e-5, BLOCK=2048, num_warps=8)
    return (out, summed, t_primals_3, summed, rstd)

def _unified_backward_graph(t_primals_3, t_add, t_rsqrt, t_tangents_1, t_tangents_2):
    dtotal = torch.empty_like(t_add)
    # Use 32-row partial reductions: same global read volume, but half the
    # scratch writes and half the final reduction work versus BM=16.
    part = torch.empty((128, 2048), device=t_add.device, dtype=torch.float32)
    dw = torch.empty_like(t_primals_3)
    _rrn_bwd_dx_kernel[(4096,)](t_add, t_primals_3, t_rsqrt, t_tangents_1, t_tangents_2, dtotal, BLOCK=2048, num_warps=8)
    _rrn_dw_part_kernel[(128, 16)](t_add, t_rsqrt, t_tangents_1, part, BM=32, BN=128, num_warps=8)
    _rrn_dw_final_kernel[(16,)](part, dw, GROUPS=128, BN=128, BG=128, num_warps=4)
    return (dtotal, dtotal, dw)

def llama3_residual_rmsnorm_forward_with_saved(x, r, weight, eps=1e-5):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (4096, 2048) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(r, torch.Tensor) or not r.is_cuda:
        raise ValueError('unified seed requires CUDA tensor r')
    if tuple(r.shape) != (4096, 2048) or r.dtype != torch.bfloat16 or r.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for r')
    if not isinstance(weight, torch.Tensor) or not weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor weight')
    if tuple(weight.shape) != (2048,) or weight.dtype != torch.bfloat16 or weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for weight')
    if any(t.device != x.device for t in (r, weight)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    if eps != 1e-05:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, r, weight)
    return _outs[:2], tuple(_outs[2:])

def llama3_residual_rmsnorm_backward_from_saved(output_grads, saved_tensors, eps=1e-5):
    if eps != 1e-05:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    dout, dsummed = output_grads
    if len(saved_tensors) != 3:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (2048,) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (4096, 2048) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (4096, 1) or saved_tensors[2].dtype != torch.float32 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (4096, 2048) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dsummed, torch.Tensor) or not dsummed.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dsummed.shape) != (4096, 2048) or dsummed.dtype != torch.bfloat16 or dsummed.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], dout.contiguous(), dsummed.contiguous())
    return (_grads[0], _grads[1], _grads[2])


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_residual_rmsnorm', 'arguments': ['x', 'r', 'weight', 'eps'], 'outputs': ['out', 'summed'], 'upstream_grads': ['dout', 'dsummed'], 'input_grads': ['dx', 'dr', 'dweight'], 'gradient_return_order': ['dx', 'dr', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'llama3_residual_rmsnorm_deployment', 'allowed_primitives': []}


class Llama3ResidualRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_residual_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, r, weight, eps):
        (out, summed), saved = llama3_residual_rmsnorm_forward_with_saved(x, r, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return out, summed

    @staticmethod
    def backward(ctx, dout, dsummed):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dout = dout.contiguous() if (dout is not None and not dout.is_contiguous()) else dout
        dsummed = dsummed.contiguous() if (dsummed is not None and not dsummed.is_contiguous()) else dsummed
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dx, dr, dweight = llama3_residual_rmsnorm_backward_from_saved(
            (dout, dsummed), saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dr, dweight, None


def llama3_residual_rmsnorm_deployment(x, r, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        r = r.reshape(-1, r.shape[-1])
    out, summed = Llama3ResidualRmsnormFunction.apply(x, r, weight, eps)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return out, summed


class Llama3ResidualRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-05):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, r):
        return llama3_residual_rmsnorm_deployment(x, r, self.weight, self.eps)


DEPLOYMENT_ENTRY = "llama3_residual_rmsnorm_deployment"

