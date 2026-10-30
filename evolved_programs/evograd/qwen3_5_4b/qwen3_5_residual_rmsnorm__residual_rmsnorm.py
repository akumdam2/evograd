"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 1, 'T': 2048, 'H': 2560}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

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

def _k3(a, alpha=1, s=1e-06):
    return add_scalar(a, s, alpha)

def _k4(a, alpha=1, s=1.0):
    return add_scalar(a, s, alpha)

def _k5(a, _dims=[0, 1], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k6(a, _dims=[2], _dtype=None, _keepdim=True):
    return _reduce(a, _dims, _keepdim, 'sum', _dtype)

def _k7(a, exp_val=3):
    return pow_scalar(a, exp_val)

def _k8(a, s=-0.5):
    return mul_scalar(a, s)

def _k9(a, s=2560):
    return div_scalar(a, s)

def _k10(a, exp_val=1.0):
    return pow_scalar(a, exp_val)

def _k11(a, s=2.0):
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
def _rrn_fwd_kernel(branch, residual, weight, normed, summed, rptr,
                    eps: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    base = row * H + offs

    b = tl.load(branch + base, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.load(residual + base, mask=mask, other=0.0).to(tl.float32)
    s = b + r0
    # Masked loads already use 0.0 for inactive lanes, so avoid an extra select.
    # Use rsqrt directly and multiply by reciprocal H to reduce scalar ops.
    ss = tl.sum(s * s, axis=0)
    rr = tl.rsqrt(ss * (1.0 / H) + eps)

    w = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
    y = s * rr * (1.0 + w)

    tl.store(summed + base, s, mask=mask)
    tl.store(normed + base, y, mask=mask)
    tl.store(rptr + row, rr)


def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3):
    normed = torch.empty_like(t_primals_1)
    summed = torch.empty_like(t_primals_1)
    r = torch.empty((2048,), device=t_primals_1.device, dtype=torch.float32)
    _rrn_fwd_kernel[(2048,)](
        t_primals_1, t_primals_2, t_primals_3, normed, summed, r,
        eps=1e-6, H=2560, BLOCK=4096, num_warps=8
    )
    return (normed, summed, t_primals_3, summed, r)

@triton.jit
def _rrn_bwd_ds_kernel(weight, summed, rptr, dnormed, dsummed, dsout,
                       H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    base = row * H + offs

    s = tl.load(summed + base, mask=mask, other=0.0).to(tl.float32)
    dn = tl.load(dnormed + base, mask=mask, other=0.0).to(tl.float32)
    dsum = tl.load(dsummed + base, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
    rr = tl.load(rptr + row).to(tl.float32)

    xhat = s * rr
    g = dn * (1.0 + w)
    # Inactive lanes have dn/s == 0 from masked loads, so g*xhat is already 0.
    dot = tl.sum(g * xhat, axis=0) * (1.0 / H)
    ds = dsum + rr * (g - xhat * dot)
    tl.store(dsout + base, ds, mask=mask)








@triton.jit
def _rrn_dw_part_kernel(summed, rptr, dnormed, part,
                        H: tl.constexpr, T: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    ns = pid_n * BN + tl.arange(0, BN)
    ms = pid_m * BM + tl.arange(0, BM)

    offs = ms[:, None] * H + ns[None, :]
    mask = (ms[:, None] < T) & (ns[None, :] < H)

    s = tl.load(summed + offs, mask=mask, other=0.0).to(tl.float32)
    dn = tl.load(dnormed + offs, mask=mask, other=0.0).to(tl.float32)
    rr = tl.load(rptr + ms, mask=ms < T, other=0.0).to(tl.float32)
    acc = tl.sum(dn * (s * rr[:, None]), axis=0)
    tl.store(part + pid_m * H + ns, acc, mask=ns < H)


@triton.jit
def _rrn_dw_reduce_kernel(part, dw, H: tl.constexpr, NM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)
    ns = pid_n * BN + tl.arange(0, BN)
    ms = tl.arange(0, NM)
    vals = tl.load(part + ms[:, None] * H + ns[None, :],
                   mask=ns[None, :] < H, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(dw + ns, acc, mask=ns < H)





def _unified_backward_graph(t_primals_3, t_add, t_rsqrt, t_tangents_1, t_tangents_2):
    ds = torch.empty_like(t_add)
    _rrn_bwd_ds_kernel[(2048,)](
        t_primals_3, t_add, t_rsqrt, t_tangents_1, t_tangents_2, ds,
        H=2560, BLOCK=4096, num_warps=8
    )

    # Same 4096 elements/program as 64x64, but halves the number of T-partials.
    # This reduces the intermediate from 32xH to 16xH and makes the final
    # reduction cheaper, while keeping contiguous column groups reasonably wide.
    part = torch.empty((16, 2560), device=t_add.device, dtype=torch.float32)
    _rrn_dw_part_kernel[(80, 16)](
        t_add, t_rsqrt, t_tangents_1, part,
        H=2560, T=2048, BM=128, BN=32, num_warps=4
    )

    dw = torch.empty_like(t_primals_3)
    _rrn_dw_reduce_kernel[(80,)](
        part, dw, H=2560, NM=16, BN=32, num_warps=4
    )
    return (ds, ds, dw)

def qwen3_5_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps=1e-6):
    if eps != 1e-06:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    _outs = _unified_forward_graph(branch, residual, weight)
    return _outs[:2], tuple(_outs[2:])

def qwen3_5_residual_rmsnorm_backward_from_saved(output_grads, saved_tensors, eps=1e-6):
    if eps != 1e-06:
        raise ValueError('unified seed scalar configuration mismatch for eps')
    dnormed, dsummed = output_grads
    _grads = _unified_backward_graph(
        saved_tensors[0], saved_tensors[1], saved_tensors[2],
        dnormed.contiguous(), dsummed.contiguous()
    )
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_5_residual_rmsnorm', 'arguments': ['branch', 'residual', 'weight', 'eps'], 'outputs': ['normed', 'summed'], 'upstream_grads': ['dnormed', 'dsummed'], 'input_grads': ['dbranch', 'dresidual', 'dweight'], 'gradient_return_order': ['dbranch', 'dresidual', 'dweight', 'None'], 'scalars': ['eps'], 'deployment_entry': 'qwen3_5_residual_rmsnorm_deployment', 'allowed_primitives': []}


class Qwen35ResidualRmsnormFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_5_residual_rmsnorm`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, branch, residual, weight, eps):
        (normed, summed), saved = qwen3_5_residual_rmsnorm_forward_with_saved(branch, residual, weight, eps)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.eps = eps
        return normed, summed

    @staticmethod
    def backward(ctx, dnormed, dsummed):
        # The engine hands back whatever view the consumer's backward produced
        # (a permuted or sliced output arrives as a strided gradient). The pair
        # was declared, verified and evolved on contiguous gradients and may
        # refuse anything else, so the copy is made here, once, inside this
        # candidate's own backward; a contiguous gradient passes through as is.
        dnormed = dnormed.contiguous() if (dnormed is not None and not dnormed.is_contiguous()) else dnormed
        dsummed = dsummed.contiguous() if (dsummed is not None and not dsummed.is_contiguous()) else dsummed
        saved = list(ctx.saved_tensors)
        for index, value in ctx.saved_values:
            saved.insert(index, value)
        saved = tuple(saved)
        dbranch, dresidual, dweight = qwen3_5_residual_rmsnorm_backward_from_saved(
            (dnormed, dsummed), saved, ctx.eps
        )
        # One value per forward argument, `None` for every inactive one.
        return dbranch, dresidual, dweight, None


def qwen3_5_residual_rmsnorm_deployment(branch, residual, weight, eps):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if branch.dim() > 3:
        _leading = branch.shape[:-1]
        branch = branch.reshape(-1, branch.shape[-1])
        residual = residual.reshape(-1, residual.shape[-1])
    normed, summed = Qwen35ResidualRmsnormFunction.apply(branch, residual, weight, eps)
    if _leading is not None:
        normed = normed.view(*_leading, normed.shape[-1])
        summed = summed.view(*_leading, summed.shape[-1])
    return normed, summed


class Qwen35ResidualRmsnormModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, eps=1e-06):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.eps = eps
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, branch, residual):
        return qwen3_5_residual_rmsnorm_deployment(branch, residual, self.weight, self.eps)


DEPLOYMENT_ENTRY = "qwen3_5_residual_rmsnorm_deployment"

