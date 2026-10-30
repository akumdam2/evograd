"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 100352, 'hidden': 96}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

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

def _k0(a, _dims=[1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'mean', _dtype)

def _k1(a, b, alpha=1):
    return sub_tt(a, b, alpha)

def _k2(a, alpha=1, s=1e-06):
    return add_scalar(a, s, alpha)

def _k3(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k4(a, s=96):
    return mul_scalar(a, s)

def _k5(a, _dims=[1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k6(a, s=96):
    return div_scalar(a, s)

def _k7(a, _dims=[0], _dtype=None, _keepdim=False):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

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
def _ln_fwd_kernel(X, W, B, Y, EPS: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < H
    x = tl.load(X + r * H + c, mask=m, other=0.).to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    xc = tl.where(m, x - mean, 0.)
    var = tl.sum(xc * xc, axis=0) / H
    rs = tl.rsqrt(var + EPS)
    w = tl.load(W + c, mask=m, other=0.).to(tl.float32)
    b = tl.load(B + c, mask=m, other=0.).to(tl.float32)
    y = xc * rs * w + b
    tl.store(Y + r * H + c, y, mask=m)

@triton.jit
def _ln_bwd_kernel(X, W, DY, DX, PW, PB,
                   R: tl.constexpr, H: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(0)
    rr = pid * BM + tl.arange(0, BM)
    cc = tl.arange(0, BN)
    mask = (rr[:, None] < R) & (cc[None, :] < H)
    offs = rr[:, None] * H + cc[None, :]

    x = tl.load(X + offs, mask=mask, other=0.).to(tl.float32)
    dy = tl.load(DY + offs, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cc, mask=cc < H, other=0.).to(tl.float32)

    mean = tl.sum(tl.where(cc[None, :] < H, x, 0.), axis=1) / H
    xc = tl.where(mask, x - mean[:, None], 0.)
    var = tl.sum(xc * xc, axis=1) / H
    rs = tl.rsqrt(var + EPS)

    xhat = xc * rs[:, None]
    dyw = dy * w[None, :]

    s1 = tl.sum(tl.where(cc[None, :] < H, dyw, 0.), axis=1)
    s2 = tl.sum(tl.where(cc[None, :] < H, dyw * xhat, 0.), axis=1)
    dx = (dyw * H - s1[:, None] - xhat * s2[:, None]) * (rs[:, None] / H)
    tl.store(DX + offs, dx, mask=mask)

    pdw = tl.sum(tl.where(mask, dy * xhat, 0.), axis=0)
    pdb = tl.sum(tl.where(mask, dy, 0.), axis=0)
    tl.store(PW + pid * H + cc, pdw, mask=cc < H)
    tl.store(PB + pid * H + cc, pdb, mask=cc < H)

@triton.jit
def _ln_reduce_kernel(PW, PB, DW, DB,
                      NB: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < NB
    dw = tl.sum(tl.load(PW + o * H + c, mask=m, other=0.).to(tl.float32), axis=0)
    db = tl.sum(tl.load(PB + o * H + c, mask=m, other=0.).to(tl.float32), axis=0)
    tl.store(DW + c, dw)
    tl.store(DB + c, db)

def _unified_forward_graph(x, weight, bias):
    y = torch.empty_like(x)
    _ln_fwd_kernel[(x.shape[0],)](
        x, weight, bias, y,
        EPS=1e-6, H=x.shape[1], BLOCK=128,
        num_warps=4,
    )
    return (y, x, weight)

def _unified_backward_graph(x, weight, dy):
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(weight)
    BM = 16
    nb = triton.cdiv(x.shape[0], BM)
    pw = torch.empty((nb, x.shape[1]), device=x.device, dtype=torch.float32)
    pb = torch.empty((nb, x.shape[1]), device=x.device, dtype=torch.float32)

    _ln_bwd_kernel[(nb,)](
        x, weight, dy, dx, pw, pb,
        R=x.shape[0], H=x.shape[1], BM=BM, BN=128, EPS=1e-6,
        num_warps=8,
    )
    _ln_reduce_kernel[(x.shape[1],)](
        pw, pb, dw, db,
        NB=nb, H=x.shape[1], BLOCK=_next_pow2(nb),
        num_warps=8,
    )
    return (dx, dw, db)

def layernorm_forward_with_saved(x, weight, bias, eps=1e-6):
    if eps != 1e-6:
        raise ValueError('eps mismatch')
    if tuple(x.shape) != (100352, 96) or x.dtype != torch.bfloat16:
        raise ValueError('x mismatch')
    with torch.cuda.device(x.device):
        y, sx, sw = _unified_forward_graph(x, weight, bias)
    return y, (sx, sw)

def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-6):
    if eps != 1e-6:
        raise ValueError('eps mismatch')
    if len(saved_tensors) != 2:
        raise ValueError('saved-state arity mismatch')
    x, weight = saved_tensors
    with torch.cuda.device(x.device):
        return _unified_backward_graph(x, weight, dy.contiguous())


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'layernorm', 'arguments': ['x', 'weight', 'bias', 'eps'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias', 'None'], 'scalars': ['eps'], 'deployment_entry': 'layernorm_deployment', 'allowed_primitives': []}


class LayernormFunction(torch.autograd.Function):
    """Static autograd wiring for `layernorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, saved = layernorm_forward_with_saved(x, weight, bias, eps)
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
        dx, dweight, dbias = layernorm_backward_from_saved(
            dy, saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias, None


def layernorm_deployment(x, weight, bias, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = LayernormFunction.apply(x, weight, bias, eps)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class LayernormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return layernorm_deployment(x, self.weight, self.bias, self.eps)


DEPLOYMENT_ENTRY = "layernorm_deployment"

