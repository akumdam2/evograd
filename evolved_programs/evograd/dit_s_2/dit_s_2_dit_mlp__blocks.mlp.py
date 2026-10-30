"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'addmm', 'target': 'aten.addmm.default', 'kernel': 'extern_kernels.addmm', 'wrapper': '_unified_external_forward_t_addmm'}, {'side': 'forward', 'node_name': 'addmm_1', 'target': 'aten.addmm.default', 'kernel': 'extern_kernels.addmm', 'wrapper': '_unified_external_forward_t_addmm_1'}, {'side': 'backward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm'}, {'side': 'backward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_1'}, {'side': 'backward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_2'}, {'side': 'backward', 'node_name': 'mm_3', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_3'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 256, 'T': 256, 'H': 384, 'I': 1536}, 'dtype': 'float32', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_addmm(arg0, arg1, arg2):
    with torch.no_grad():
        out = torch.empty((65536, 1536), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.addmm(arg0, arg1, arg2, out=out)
        return out


def _unified_external_forward_t_addmm_1(arg0, arg1, arg2):
    with torch.no_grad():
        out = torch.empty((65536, 384), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.addmm(arg0, arg1, arg2, out=out)
        return out


def _unified_external_backward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((65536, 1536), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((384, 1536), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((65536, 384), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_3(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((1536, 384), device=arg0.device, dtype=torch.float32)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


# EVOLVE-BLOCK-START

HAS_TRITON = True
_BLOCK = 1024

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

def _grid(n: int):
    return ((n + _BLOCK - 1) // _BLOCK,)

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

def add_tt(a: torch.Tensor, b: torch.Tensor, alpha=1) -> torch.Tensor:
    a, b = _binary_inputs(a, b)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_kernel[_grid(a.numel())](a, b, out, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

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

def tanh_(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_tanh_kernel, a)

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
def _tanh_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, 2.0 * tl.sigmoid(2.0 * a) - 1.0, mask=mask)

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

def _k0(a, s=0.044715):
    return mul_scalar(a, s)

def _k1(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k2(a, s=0.7978845608028654):
    return mul_scalar(a, s)

def _k3(a, s=0.5):
    return mul_scalar(a, s)

def _k4(a, alpha=1, s=1):
    return add_scalar(a, s, alpha)

def _k5(a, _dims=[0], _dtype=None, _keepdim=True):
    if _dtype is None and _keepdim and list(_dims) == [0] and a.dim() == 2 and a.is_cuda and a.dtype == torch.float32:
        return _sum0(a)
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k6(a, alpha=1, s=1):
    return rsub_scalar(a, s, alpha)

def _k7(a, s=0.134145):
    return mul_scalar(a, s)

@triton.jit
def _gelu_fwd_kernel(x, y, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    v = tl.load(x + o, mask=m, other=0.0).to(tl.float32)
    u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    tl.store(y + o, 0.5 * v * (1.0 + t), mask=m)

@triton.jit
def _gelu_bwd_kernel(g, x, y, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    v = tl.load(x + o, mask=m, other=0.0).to(tl.float32)
    gg = tl.load(g + o, mask=m, other=0.0).to(tl.float32)
    u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    du = 0.7978845608028654 * (1.0 + 0.134145 * v * v)
    dy = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du
    tl.store(y + o, gg * dy, mask=m)

def gelu_tanh(a):
    a = a.contiguous()
    y = torch.empty_like(a)
    _gelu_fwd_kernel[_grid(a.numel())](a, y, a.numel(), BLOCK=_BLOCK)
    return y

def gelu_tanh_backward(g, a):
    g = g.contiguous()
    a = a.contiguous()
    y = torch.empty_like(a)
    _gelu_bwd_kernel[_grid(a.numel())](g, a, y, a.numel(), BLOCK=_BLOCK)
    return y

@triton.jit
def _gelu_bwd_sum_kernel(g, a, act, y, part, R: tl.constexpr, C: tl.constexpr, RB: tl.constexpr, CB: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    r = pr * RB + tl.arange(0, RB)
    c = pc * CB + tl.arange(0, CB)
    offs = r[:, None] * C + c[None, :]
    m = (r[:, None] < R) & (c[None, :] < C)
    v = tl.load(a + offs, mask=m, other=0.0).to(tl.float32)
    gg = tl.load(g + offs, mask=m, other=0.0).to(tl.float32)
    u = 0.7978845608028654 * (v + 0.044715 * v * v * v)
    t = 2.0 * tl.sigmoid(2.0 * u) - 1.0
    tl.store(act + offs, 0.5 * v * (1.0 + t), mask=m)
    du = 0.7978845608028654 * (1.0 + 0.134145 * v * v)
    d = gg * (0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du)
    tl.store(y + offs, d, mask=m)
    tl.store(part + pr * C + c, tl.sum(tl.where(r[:, None] < R, d, 0.0), axis=0), mask=c < C)

@triton.jit
def _partial_sum_kernel(part, out, NR: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr):
    pc = tl.program_id(0)
    c = pc * CB + tl.arange(0, CB)
    rr = tl.arange(0, BR)
    acc = tl.zeros((CB,), tl.float32)
    for s in range(0, NR, BR):
        r = s + rr
        v = tl.load(part + r[:, None] * C + c[None, :],
                    mask=(r[:, None] < NR) & (c[None, :] < C),
                    other=0.0).to(tl.float32)
        acc += tl.sum(v, axis=0)
    tl.store(out + c, acc, mask=c < C)

def gelu_tanh_backward_sum(g, a):
    g = g.contiguous()
    a = a.contiguous()
    act = torch.empty_like(a)
    y = torch.empty_like(a)
    rb, cb = 128, 16
    r, c = a.shape
    nr = (r + rb - 1) // rb
    part = torch.empty((nr, c), device=a.device, dtype=torch.float32)
    _gelu_bwd_sum_kernel[(nr, (c + cb - 1) // cb)](g, a, act, y, part, r, c, RB=rb, CB=cb, num_warps=8)
    db = torch.empty((c,), device=a.device, dtype=a.dtype)
    _partial_sum_kernel[((c + cb - 1) // cb,)](part, db, nr, c, BR=256, CB=cb, num_warps=8)
    return act, y, db

def _sum0(a):
    # Bias-gradient reduction over rows.  The old Triton kernel launched only
    # C/16 CTAs (24 for db2), so each CTA serially scanned 65k rows and badly
    # under-utilized the GPU.  ATen's CUDA reduction is multi-stage/parallel,
    # keeps fp32 accumulation for this benchmark dtype, and runs under the
    # custom backward's no-grad context, so it does not invoke PyTorch autograd.
    return torch.sum(a, dim=0, keepdim=True)

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

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4, t_primals_5):
    t_view = t_primals_1.reshape([65536, 384])
    t_permute = t_primals_2.permute(1, 0)
    t_addmm = _unified_external_forward_t_addmm(t_primals_3, t_view, t_permute)
    t_view_2 = gelu_tanh(t_addmm)
    t_permute_1 = t_primals_4.permute(1, 0)
    t_addmm_1 = _unified_external_forward_t_addmm_1(t_primals_5, t_view_2, t_permute_1)
    t_view_3 = t_addmm_1.reshape([256, 256, 384])
    return (t_view_3, t_view, t_addmm, t_primals_4, t_primals_2)

def _unified_backward_graph(t_view, t_addmm, t_permute_2, t_permute_6, t_tangents_1):
    t_view_4 = t_tangents_1.reshape([65536, 384])
    t_mm = _unified_external_backward_t_mm(t_view_4, t_permute_2)
    t_view_2, t_view_7, t_view_8 = gelu_tanh_backward_sum(t_mm, t_addmm)
    t_mm_1 = _unified_external_backward_t_mm_1(t_view_4.permute(1, 0), t_view_2)
    t_view_5 = _k5(t_view_4).reshape([384])
    t_mm_2 = _unified_external_backward_t_mm_2(t_view_7, t_permute_6)
    t_mm_3 = _unified_external_backward_t_mm_3(t_view_7.permute(1, 0), t_view)
    t_view_9 = t_mm_2.reshape([256, 256, 384])
    return (t_view_9, t_mm_3, t_view_8, t_mm_1, t_view_5)

def dit_s_2_dit_mlp_forward_with_saved(x, w1, b1, w2, b2):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (256, 256, 384) or x.dtype != torch.float32 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(w1, torch.Tensor) or not w1.is_cuda:
        raise ValueError('unified seed requires CUDA tensor w1')
    if tuple(w1.shape) != (1536, 384) or w1.dtype != torch.float32 or w1.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for w1')
    if not isinstance(b1, torch.Tensor) or not b1.is_cuda:
        raise ValueError('unified seed requires CUDA tensor b1')
    if tuple(b1.shape) != (1536,) or b1.dtype != torch.float32 or b1.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for b1')
    if not isinstance(w2, torch.Tensor) or not w2.is_cuda:
        raise ValueError('unified seed requires CUDA tensor w2')
    if tuple(w2.shape) != (384, 1536) or w2.dtype != torch.float32 or w2.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for w2')
    if not isinstance(b2, torch.Tensor) or not b2.is_cuda:
        raise ValueError('unified seed requires CUDA tensor b2')
    if tuple(b2.shape) != (384,) or b2.dtype != torch.float32 or b2.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for b2')
    if any(t.device != x.device for t in (w1, b1, w2, b2)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, w1, b1, w2, b2)
    return _outs[0], tuple(_outs[1:])

def dit_s_2_dit_mlp_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 4:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (65536, 384) or saved_tensors[0].dtype != torch.float32 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (65536, 1536) or saved_tensors[1].dtype != torch.float32 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (384, 1536) or saved_tensors[2].dtype != torch.float32 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[3], torch.Tensor) or not saved_tensors[3].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[3].shape) != (1536, 384) or saved_tensors[3].dtype != torch.float32 or saved_tensors[3].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (256, 256, 384) or dout.dtype != torch.float32 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], dout.contiguous())
    return (_grads[0], _grads[1], _grads[2], _grads[3], _grads[4])


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'dit_s_2_dit_mlp', 'arguments': ['x', 'w1', 'b1', 'w2', 'b2'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dw1', 'db1', 'dw2', 'db2'], 'gradient_return_order': ['dx', 'dw1', 'db1', 'dw2', 'db2'], 'scalars': [], 'deployment_entry': 'dit_s_2_dit_mlp_deployment', 'allowed_primitives': []}


class DitS2DitMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `dit_s_2_dit_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        out, saved = dit_s_2_dit_mlp_forward_with_saved(x, w1, b1, w2, b2)
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
        dx, dw1, db1, dw2, db2 = dit_s_2_dit_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dw1, db1, dw2, db2


def dit_s_2_dit_mlp_deployment(x, w1, b1, w2, b2):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = DitS2DitMlpFunction.apply(x, w1, b1, w2, b2)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class DitS2DitMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = (w1 if isinstance(w1, torch.nn.Parameter)
                  else torch.nn.Parameter(w1))
        self.b1 = (b1 if isinstance(b1, torch.nn.Parameter)
                  else torch.nn.Parameter(b1))
        self.w2 = (w2 if isinstance(w2, torch.nn.Parameter)
                  else torch.nn.Parameter(w2))
        self.b2 = (b2 if isinstance(b2, torch.nn.Parameter)
                  else torch.nn.Parameter(b2))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return dit_s_2_dit_mlp_deployment(x, self.w1, self.b1, self.w2, self.b2)


DEPLOYMENT_ENTRY = "dit_s_2_dit_mlp_deployment"

