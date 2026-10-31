"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm'}, {'side': 'forward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_1'}, {'side': 'forward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_2'}, {'side': 'backward', 'node_name': 'mm_3', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_3'}, {'side': 'backward', 'node_name': 'mm_4', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_4'}, {'side': 'backward', 'node_name': 'mm_5', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_5'}, {'side': 'backward', 'node_name': 'mm_6', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_6'}, {'side': 'backward', 'node_name': 'mm_7', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_7'}, {'side': 'backward', 'node_name': 'mm_8', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_8'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 2, 'T': 2048, 'H': 2048, 'HQ': 32, 'HK': 8, 'D': 64, 'QO': 2048, 'KVO': 512}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 512), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 512), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_3(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((512, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_4(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_5(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((512, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_6(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_7(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((2048, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_8(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
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
    import math
    return (math.ceil(n / _BLOCK),)

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

def add3_tt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Accumulate into the first temporary.  dx_v is freshly allocated by the
    # preceding matmul and is not reused, so this avoids one large allocation.
    _add3_kernel[_grid(a.numel())](a, b, c, N=a.numel(), BLOCK=_BLOCK)
    return a

@triton.jit
def _add3_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        c = c.to(tl.float32)
    tl.store(a_ptr + offs, a + b + c, mask=mask)

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

def _k0(*runtime, plan=[(True, None), (False, 3), (False, 32), (False, 9223372036854775807)]):
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

def _k1(*runtime, plan=[(True, None), (False, 3), (False, 0), (False, 32)]):
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

def _k2(tensors, dim=-1):
    return cat2(tensors, dim)

def _k3(a, b, alpha=1):
    return add_tt(a, b, alpha)

def _k4(*runtime, plan=[(True, None), (False, 3), (False, 32), (False, 64)]):
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

def _k5(dtype=torch.bfloat16, fill_val=0, shape=[2, 8, 2048, 64]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

def _k6(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 0), (False, 32)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

def _k7(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 32), (False, 9223372036854775807)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

def _k8(dtype=torch.bfloat16, fill_val=0, shape=[2, 32, 2048, 64]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

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
def _rope_from_flat_kernel(inp, cos, sin, out, N, HEADS: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    # out is flat [B,T,H,D] storage; Python returns out.view(B,T,H,D).transpose(1,2).
    # This keeps input/output accesses mostly contiguous and produces the semantic
    # non-contiguous head-major view without an extra layout-copy kernel.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    d = offs % D
    row_h = offs // D
    h = row_h % HEADS
    row = row_h // HEADS
    t = row % T
    half = D // 2
    od = tl.where(d < half, d + half, d - half)
    base = offs - d

    x = tl.load(inp + offs, mask=m, other=0.).to(tl.float32)
    xo = tl.load(inp + base + od, mask=m, other=0.).to(tl.float32)
    c = tl.load(cos + t * D + d, mask=m, other=0.).to(tl.float32)
    s = tl.load(sin + t * D + d, mask=m, other=0.).to(tl.float32)
    r = tl.where(d < half, -xo, xo)
    tl.store(out + offs, x * c + r * s, mask=m)

@triton.jit
def _flat_to_heads_kernel(inp, out, N, HEADS: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    d = offs % D
    tmp = offs // D
    t = tmp % T
    h = (tmp // T) % HEADS
    b = tmp // (T * HEADS)
    row = b * T + t
    v = tl.load(inp + (row * HEADS + h) * D + d, mask=m, other=0.)
    tl.store(out + offs, v, mask=m)

@triton.jit
def _unrope_to_flat2_kernel(g, cos, sin, flat, N, HEADS: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    d = offs % D
    h = (offs // D) % HEADS
    t = (offs // (D * HEADS)) % T
    b = offs // (D * HEADS * T)
    half = D // 2
    gin = ((b * HEADS + h) * T + t) * D
    x = tl.load(g + gin + d, mask=m, other=0.).to(tl.float32)
    od = tl.where(d < half, d + half, d - half)
    xo = tl.load(g + gin + od, mask=m, other=0.).to(tl.float32)
    c = tl.load(cos + t * D + d, mask=m, other=0.).to(tl.float32)
    so = tl.load(sin + t * D + od, mask=m, other=0.).to(tl.float32)
    v = tl.where(d < half, x * c + xo * so, x * c - xo * so)
    row = b * T + t
    fo = (row * HEADS + h) * D + d
    tl.store(flat + fo, v, mask=m)

@triton.jit
def _heads_to_flat2_kernel(g, flat, N, HEADS: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    d = offs % D
    h = (offs // D) % HEADS
    t = (offs // (D * HEADS)) % T
    b = offs // (D * HEADS * T)
    row = b * T + t
    v = tl.load(g + ((b * HEADS + h) * T + t) * D + d, mask=m, other=0.)
    fo = (row * HEADS + h) * D + d
    tl.store(flat + fo, v, mask=m)

def rope_from_flat(inp, cos, sin, heads):
    out = torch.empty((2, 2048, heads, 64), device=inp.device, dtype=inp.dtype)
    _rope_from_flat_kernel[(triton.cdiv(out.numel(), 256),)](inp, cos, sin, out, out.numel(), HEADS=heads, T=2048, D=64, BLOCK=256)
    return out.transpose(1, 2)

def flat_to_heads(inp, heads):
    return inp.reshape(2, 2048, heads, 64).transpose(1, 2)

def unrope_to_flat2(g, cos, sin, heads):
    flat = torch.empty((4096, heads * 64), device=g.device, dtype=g.dtype)
    _unrope_to_flat2_kernel[(triton.cdiv(g.numel(), 256),)](g, cos, sin, flat, g.numel(), HEADS=heads, T=2048, D=64, BLOCK=256)
    return flat, flat.permute(1, 0)

def heads_to_flat2(g, heads):
    flat = torch.empty((4096, heads * 64), device=g.device, dtype=g.dtype)
    _heads_to_flat2_kernel[(triton.cdiv(g.numel(), 256),)](g, flat, g.numel(), HEADS=heads, T=2048, D=64, BLOCK=256)
    return flat, flat.permute(1, 0)

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4, t_primals_5, t_primals_6):
    t_view = t_primals_1.reshape([4096, 2048])
    q_flat = _unified_external_forward_t_mm(t_view, t_primals_2.permute(1, 0))
    k_flat = _unified_external_forward_t_mm_1(t_view, t_primals_3.permute(1, 0))
    v_flat = _unified_external_forward_t_mm_2(t_view, t_primals_4.permute(1, 0))
    q = rope_from_flat(q_flat, t_primals_5, t_primals_6, 32)
    k = rope_from_flat(k_flat, t_primals_5, t_primals_6, 8)
    v = flat_to_heads(v_flat, 8)
    return (q, k, v, t_primals_5, t_primals_6, t_view, t_primals_4, t_primals_3, t_primals_2)

def _unified_backward_graph(t_primals_5, t_primals_6, t_view, t_permute_9, t_permute_14, t_permute_19, t_tangents_1, t_tangents_2, t_tangents_3):
    dq_flat, dq_t = unrope_to_flat2(t_tangents_1, t_primals_5, t_primals_6, 32)
    dk_flat, dk_t = unrope_to_flat2(t_tangents_2, t_primals_5, t_primals_6, 8)
    dv_flat, dv_t = heads_to_flat2(t_tangents_3, 8)

    dv_weight = _unified_external_backward_t_mm_3(dv_t, t_view)
    dx_v = _unified_external_backward_t_mm_4(dv_flat, t_permute_9).reshape([2, 2048, 2048])

    dk_weight = _unified_external_backward_t_mm_5(dk_t, t_view)
    dx_k = _unified_external_backward_t_mm_6(dk_flat, t_permute_14).reshape([2, 2048, 2048])

    dq_weight = _unified_external_backward_t_mm_7(dq_t, t_view)
    dx_q = _unified_external_backward_t_mm_8(dq_flat, t_permute_19).reshape([2, 2048, 2048])

    dx = add3_tt(dx_v, dx_k, dx_q)
    return (dx, dq_weight, dk_weight, dv_weight, None, None)

def llama3_qkv_rope_forward_with_saved(x, q_weight, k_weight, v_weight, cos, sin):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (2, 2048, 2048) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(q_weight, torch.Tensor) or not q_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor q_weight')
    if tuple(q_weight.shape) != (2048, 2048) or q_weight.dtype != torch.bfloat16 or q_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for q_weight')
    if not isinstance(k_weight, torch.Tensor) or not k_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor k_weight')
    if tuple(k_weight.shape) != (512, 2048) or k_weight.dtype != torch.bfloat16 or k_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for k_weight')
    if not isinstance(v_weight, torch.Tensor) or not v_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor v_weight')
    if tuple(v_weight.shape) != (512, 2048) or v_weight.dtype != torch.bfloat16 or v_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for v_weight')
    if not isinstance(cos, torch.Tensor) or not cos.is_cuda:
        raise ValueError('unified seed requires CUDA tensor cos')
    if tuple(cos.shape) != (1, 2048, 64) or cos.dtype != torch.bfloat16 or cos.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for cos')
    if not isinstance(sin, torch.Tensor) or not sin.is_cuda:
        raise ValueError('unified seed requires CUDA tensor sin')
    if tuple(sin.shape) != (1, 2048, 64) or sin.dtype != torch.bfloat16 or sin.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for sin')
    if any(t.device != x.device for t in (q_weight, k_weight, v_weight, cos, sin)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, q_weight, k_weight, v_weight, cos, sin)
    return _outs[:3], tuple(_outs[3:])

def llama3_qkv_rope_backward_from_saved(output_grads, saved_tensors):
    dq, dk, dv = output_grads
    if len(saved_tensors) != 6:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (1, 2048, 64) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (1, 2048, 64) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (4096, 2048) or saved_tensors[2].dtype != torch.bfloat16 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[3], torch.Tensor) or not saved_tensors[3].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[3].shape) != (512, 2048) or saved_tensors[3].dtype != torch.bfloat16 or saved_tensors[3].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[4], torch.Tensor) or not saved_tensors[4].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[4].shape) != (512, 2048) or saved_tensors[4].dtype != torch.bfloat16 or saved_tensors[4].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[5], torch.Tensor) or not saved_tensors[5].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[5].shape) != (2048, 2048) or saved_tensors[5].dtype != torch.bfloat16 or saved_tensors[5].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dq, torch.Tensor) or not dq.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dq.shape) != (2, 32, 2048, 64) or dq.dtype != torch.bfloat16 or dq.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dk, torch.Tensor) or not dk.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dk.shape) != (2, 8, 2048, 64) or dk.dtype != torch.bfloat16 or dk.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dv, torch.Tensor) or not dv.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dv.shape) != (2, 8, 2048, 64) or dv.dtype != torch.bfloat16 or dv.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], saved_tensors[4], saved_tensors[5], dq.contiguous(), dk.contiguous(), dv.contiguous())
    return (_grads[0], _grads[1], _grads[2], _grads[3])


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_qkv_rope', 'arguments': ['x', 'q_weight', 'k_weight', 'v_weight', 'cos', 'sin'], 'outputs': ['q', 'k', 'v'], 'upstream_grads': ['dq', 'dk', 'dv'], 'input_grads': ['dx', 'dq_weight', 'dk_weight', 'dv_weight'], 'gradient_return_order': ['dx', 'dq_weight', 'dk_weight', 'dv_weight', 'None', 'None'], 'scalars': [], 'deployment_entry': 'llama3_qkv_rope_deployment', 'allowed_primitives': []}


class Llama3QkvRopeFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_qkv_rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, q_weight, k_weight, v_weight, cos, sin):
        (q, k, v), saved = llama3_qkv_rope_forward_with_saved(x, q_weight, k_weight, v_weight, cos, sin)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
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
        dx, dq_weight, dk_weight, dv_weight = llama3_qkv_rope_backward_from_saved(
            (dq, dk, dv), saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dq_weight, dk_weight, dv_weight, None, None


def llama3_qkv_rope_deployment(x, q_weight, k_weight, v_weight, cos, sin):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Llama3QkvRopeFunction.apply(x, q_weight, k_weight, v_weight, cos, sin)


class Llama3QkvRopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, q_weight, k_weight, v_weight):
        super().__init__()
        self.q_weight = (q_weight if isinstance(q_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(q_weight))
        self.k_weight = (k_weight if isinstance(k_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(k_weight))
        self.v_weight = (v_weight if isinstance(v_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(v_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return llama3_qkv_rope_deployment(x, self.q_weight, self.k_weight, self.v_weight, cos, sin)


DEPLOYMENT_ENTRY = "llama3_qkv_rope_deployment"

