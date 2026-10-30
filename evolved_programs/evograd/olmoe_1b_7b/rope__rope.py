"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 2, 'n_heads': 16, 'T': 2048, 'head_dim': 128}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

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

def _k0(*runtime, plan=[(True, None), (False, 3), (False, 0), (False, 64)]):
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

def _k1(*runtime, plan=[(True, None), (False, 3), (False, 64), (False, 9223372036854775807)]):
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

def _k4(*runtime, plan=[(True, None), (False, 3), (False, 64), (False, 128)]):
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

def _k5(dtype=torch.bfloat16, fill_val=0, shape=[2, 16, 2048, 128]):
    return fill_tensor(shape, fill_val, dtype=dtype, device='cuda')

def _k6(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 64), (False, 9223372036854775807)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

def _k7(*runtime, plan=[(True, None), (True, None), (False, 3), (False, 0), (False, 64)]):
    values = iter(runtime)
    args = [next(values) if dynamic else constant for dynamic, constant in plan]
    return slice_scatter(*args)

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
def _rope_fwd_kernel(x, c, s, y, NBH: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                     HD: tl.constexpr, BB: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    bh = g * BB + tl.arange(0, BB)
    col = tl.arange(0, HD)
    m = bh[:, None] < NBH
    off = ((bh[:, None] * T + t) * D + col[None, :])
    cc = tl.load(c + t * D + col).to(tl.float32)[None, :]
    ss = tl.load(s + t * D + col).to(tl.float32)[None, :]
    x1 = tl.load(x + off, mask=m, other=0.).to(tl.float32)
    x2 = tl.load(x + off + HD, mask=m, other=0.).to(tl.float32)
    tl.store(y + off, x1 * cc - x2 * ss, mask=m)
    tl.store(y + off + HD, x2 * cc + x1 * ss, mask=m)

@triton.jit
def _rope_bwd_kernel(dy, c, s, dx, CS: tl.constexpr, SS: tl.constexpr,
                     NBH: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                     HD: tl.constexpr, BB: tl.constexpr):
    t = tl.program_id(0)
    g = tl.program_id(1)
    bh = g * BB + tl.arange(0, BB)
    col = tl.arange(0, HD)
    m = bh[:, None] < NBH
    off = ((bh[:, None] * T + t) * D + col[None, :])
    cc = tl.load(c + t * CS + col).to(tl.float32)[None, :]
    ss = tl.load(s + t * SS + col).to(tl.float32)[None, :]
    d1 = tl.load(dy + off, mask=m, other=0.).to(tl.float32)
    d2 = tl.load(dy + off + HD, mask=m, other=0.).to(tl.float32)
    tl.store(dx + off, d1 * cc + d2 * ss, mask=m)
    tl.store(dx + off + HD, d2 * cc - d1 * ss, mask=m)

def rope_forward_with_saved(x, cos, sin):
    if not (isinstance(x, torch.Tensor) and x.is_cuda):
        raise ValueError('rope expects CUDA x')
    if not x.is_contiguous():
        x = x.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    B, H, T, D = x.shape
    HD = D // 2
    y = torch.empty_like(x)
    BB = 8
    with torch.cuda.device(x.device):
        _rope_fwd_kernel[(T, triton.cdiv(B * H, BB))](
            x, cos, sin, y, B * H, T, D, HD, BB, num_warps=4
        )

    # Save the original rotary tables: avoids two slice-contiguous copy kernels
    # in the measured forward path.  Backward only loads the first half using
    # row stride D.
    return y, (cos, sin)

def rope_backward_from_saved(dy, saved_tensors):
    if len(saved_tensors) != 2:
        raise ValueError('rope saved-state arity mismatch')
    c0, s0 = saved_tensors
    if not (isinstance(dy, torch.Tensor) and dy.is_cuda):
        raise ValueError('rope backward expects CUDA dy')
    if not dy.is_contiguous():
        dy = dy.contiguous()

    B, H, T, D = dy.shape
    HD = D // 2
    dx = torch.empty_like(dy)
    BB = 8
    with torch.cuda.device(dy.device):
        _rope_bwd_kernel[(T, triton.cdiv(B * H, BB))](
            dy, c0, s0, dx, D, D, B * H, T, D, HD, BB, num_warps=4
        )
    return (dx,)


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'rope', 'arguments': ['x', 'cos', 'sin'], 'outputs': ['y'], 'upstream_grads': ['dy'], 'input_grads': ['dx'], 'gradient_return_order': ['dx', 'None', 'None'], 'scalars': [], 'deployment_entry': 'rope_deployment', 'allowed_primitives': []}


class RopeFunction(torch.autograd.Function):
    """Static autograd wiring for `rope`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        y, saved = rope_forward_with_saved(x, cos, sin)
        # save_for_backward takes tensors (or None) only; the pair's saved state may
        # also hold plain Python values (shapes, block sizes), kept on ctx in order.
        saved = tuple(saved) if isinstance(saved, (tuple, list)) else (saved,)
        ctx.saved_values = tuple((i, v) for i, v in enumerate(saved)
                                 if v is not None and not torch.is_tensor(v))
        ctx.save_for_backward(*(v for v in saved if v is None or torch.is_tensor(v)))
        pass
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
        dx, = rope_backward_from_saved(
            dy, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, None, None


def rope_deployment(x, cos, sin):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 4:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    y = RopeFunction.apply(x, cos, sin)
    if _leading is not None:
        y = y.view(*_leading, y.shape[-1])
    return y


class RopeModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x, cos, sin):
        return rope_deployment(x, cos, sin)


DEPLOYMENT_ENTRY = "rope_deployment"

