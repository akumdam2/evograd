"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'convolution', 'target': 'aten.convolution.default', 'kernel': 'extern_kernels.convolution', 'wrapper': '_unified_external_forward_t_convolution'}, {'side': 'backward', 'node_name': 'convolution_backward', 'target': 'aten.convolution_backward.default', 'kernel': 'torch.ops.aten.convolution_backward.default', 'wrapper': '_unified_external_backward_t_convolution_backward'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 32, 'C': 3, 'Y': 224, 'X': 224, 'O': 96, 'CG': 3, 'K': 4, 'OY': 56, 'OX': 56, 'S': 4, 'P': 0, 'G': 1}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_convolution(arg0, arg1, arg2):
    with torch.no_grad():
        return _unified_extern_kernels.convolution(arg0, arg1, arg2, [4, 4], [0, 0], [1, 1], False, [0, 0], 1)


def _unified_external_backward_t_convolution_backward(arg0, arg1, arg2):
    with torch.no_grad():
        return torch.ops.aten.convolution_backward.default(arg0, arg1, arg2, [96], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])


# EVOLVE-BLOCK-START

HAS_TRITON = True

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

@triton.jit
def _dbias_stage1_kernel(D, TMP, NT: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < 100352  # 32*56*56

    hw = offs % 3136      # 56*56
    n = offs // 3136
    vals = tl.load(D + n * 301056 + o * 3136 + hw, mask=mask, other=0.).to(tl.float32)
    tl.store(TMP + o * NT + tile, tl.sum(vals, axis=0))


@triton.jit
def _dbias_stage2_kernel(TMP, OUT, NT: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(TMP + o * NT + offs, mask=offs < NT, other=0.).to(tl.float32)
    tl.store(OUT + o, tl.sum(vals, axis=0))


def _k0(a, _dims=[0, 2, 3], _dtype=None, _keepdim=False):
    # Specialize the only reduction used by this graph:
    # dbias = dout.sum(dim=(0,2,3)) for contiguous NCHW [32,96,56,56].
    #
    # The generic reducer first materializes dout.permute(1,0,2,3).contiguous(),
    # copying ~19MB before reducing.  This two-stage reduction reads dout in-place
    # and only allocates a tiny fp32 partial buffer.
    if (
        _dtype is None
        and not _keepdim
        and tuple(_dims) == (0, 2, 3)
        and a.ndim == 4
        and tuple(a.shape) == (32, 96, 56, 56)
        and a.is_contiguous()
        and a.is_cuda
    ):
        NT = 98  # ceil(32*56*56 / 1024)
        tmp = torch.empty((96, NT), device=a.device, dtype=torch.float32)
        out = torch.empty((96,), device=a.device, dtype=a.dtype)
        _dbias_stage1_kernel[(96, NT)](a, tmp, NT, BLOCK=1024, num_warps=4)
        _dbias_stage2_kernel[(96,)](tmp, out, NT, BLOCK=128, num_warps=4)
        return out
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

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3):
    t_convolution = _unified_external_forward_t_convolution(t_primals_1, t_primals_2, t_primals_3)  # aten.convolution.default
    return (t_convolution, t_primals_1, t_primals_2)

def _unified_backward_graph(t_primals_1, t_primals_2, t_tangents_1):
    t_sum_1 = _k0(t_tangents_1)  # aten.sum.dim_IntList
    t_convolution_backward = _unified_external_backward_t_convolution_backward(t_tangents_1, t_primals_1, t_primals_2)  # aten.convolution_backward.default
    t_getitem = t_convolution_backward[0]  # <built-in function getitem>
    t_getitem_1 = t_convolution_backward[1]  # <built-in function getitem>
    return (t_getitem, t_getitem_1, t_sum_1)

def spatial_conv2d_forward_with_saved(x, weight, bias, stride=4, padding=0, groups=1):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (32, 3, 224, 224) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(weight, torch.Tensor) or not weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor weight')
    if tuple(weight.shape) != (96, 3, 4, 4) or weight.dtype != torch.bfloat16 or weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for weight')
    if not isinstance(bias, torch.Tensor) or not bias.is_cuda:
        raise ValueError('unified seed requires CUDA tensor bias')
    if tuple(bias.shape) != (96,) or bias.dtype != torch.bfloat16 or bias.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for bias')
    if any(t.device != x.device for t in (weight, bias)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    if stride != 4:
        raise ValueError('unified seed scalar configuration mismatch for stride')
    if padding != 0:
        raise ValueError('unified seed scalar configuration mismatch for padding')
    if groups != 1:
        raise ValueError('unified seed scalar configuration mismatch for groups')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, weight, bias)
    return _outs[0], tuple(_outs[1:])

def spatial_conv2d_backward_from_saved(dout, saved_tensors, stride=4, padding=0, groups=1):
    if stride != 4:
        raise ValueError('unified seed scalar configuration mismatch for stride')
    if padding != 0:
        raise ValueError('unified seed scalar configuration mismatch for padding')
    if groups != 1:
        raise ValueError('unified seed scalar configuration mismatch for groups')
    if len(saved_tensors) != 2:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (32, 3, 224, 224) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (96, 3, 4, 4) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (32, 96, 56, 56) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], dout.contiguous())
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'spatial_conv2d', 'arguments': ['x', 'weight', 'bias', 'stride', 'padding', 'groups'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dweight', 'dbias'], 'gradient_return_order': ['dx', 'dweight', 'dbias', 'None', 'None', 'None'], 'scalars': ['stride', 'padding', 'groups'], 'deployment_entry': 'spatial_conv2d_deployment', 'allowed_primitives': []}


class SpatialConv2DFunction(torch.autograd.Function):
    """Static autograd wiring for `spatial_conv2d`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, groups):
        out, saved = spatial_conv2d_forward_with_saved(x, weight, bias, stride, padding, groups)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
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
        dx, dweight, dbias = spatial_conv2d_backward_from_saved(
            dout, saved, ctx.stride, ctx.padding, ctx.groups
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dweight, dbias, None, None, None


def spatial_conv2d_deployment(x, weight, bias, stride, padding, groups):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        weight = weight.reshape(-1, weight.shape[-1])
    out = SpatialConv2DFunction.apply(x, weight, bias, stride, padding, groups)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class SpatialConv2DModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, weight, bias, stride=4, padding=0, groups=1):
        super().__init__()
        self.weight = (weight if isinstance(weight, torch.nn.Parameter)
                  else torch.nn.Parameter(weight))
        self.bias = (bias if isinstance(bias, torch.nn.Parameter)
                  else torch.nn.Parameter(bias))
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return spatial_conv2d_deployment(x, self.weight, self.bias, self.stride, self.padding, self.groups)


DEPLOYMENT_ENTRY = "spatial_conv2d_deployment"

