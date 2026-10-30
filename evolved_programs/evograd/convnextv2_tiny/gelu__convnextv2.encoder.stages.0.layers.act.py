"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 100352, 'hidden': 384}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

# EVOLVE-BLOCK-START

_BLOCK = 1024

@triton.jit
def _gelu_fwd_kernel(xp, op, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(xp + offs, mask=m, other=0.0).to(tl.float32)
    y = 0.5 * x * (1.0 + libdevice.erf(x * 0.7071067811865476))
    tl.store(op + offs, y, mask=m)

@triton.jit
def _gelu_bwd_kernel(dp, xp, gp, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(xp + offs, mask=m, other=0.0).to(tl.float32)
    d = tl.load(dp + offs, mask=m, other=0.0).to(tl.float32)
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    pdfx = x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    tl.store(gp + offs, d * (cdf + pdfx), mask=m)

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
    a = _scalar_input(a, scalar)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _add_scalar_kernel[_grid(a.numel())](a, out, scalar, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

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

def exp(a: torch.Tensor) -> torch.Tensor:
    return _unary_f32(_exp_kernel, a)

@triton.jit
def _exp_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(a), mask=mask)

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

def _k0(a, s=0.5):
    return mul_scalar(a, s)

def _k1(a, s=0.7071067811865476):
    return mul_scalar(a, s)

def _k2(a, alpha=1, s=1):
    return add_scalar(a, s, alpha)

def _k3(a, s=-0.5):
    return mul_scalar(a, s)

def _k4(a, s=0.3989422804014327):
    return mul_scalar(a, s)

def _k5(a, b, alpha=1):
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

def _unified_forward_graph(t_primals_1):
    t_convert_element_type = t_primals_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_mul = _k0(t_convert_element_type)  # aten.mul.Tensor
    t_mul_1 = _k1(t_convert_element_type)  # aten.mul.Tensor
    t_erf = erf_(t_mul_1)  # aten.erf.default
    t_add = _k2(t_erf)  # aten.add.Tensor
    t_mul_2 = mul_tt(t_mul, t_add)  # aten.mul.Tensor
    t_convert_element_type_1 = t_mul_2.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    return (t_convert_element_type_1, t_primals_1)

def _unified_backward_graph(t_primals_1, t_tangents_1):
    t_convert_element_type_2 = t_tangents_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_convert_element_type = t_primals_1.to(dtype=torch.float32)  # prims.convert_element_type.default
    t_mul_1 = _k1(t_convert_element_type)  # aten.mul.Tensor
    t_erf = erf_(t_mul_1)  # aten.erf.default
    t_add = _k2(t_erf)  # aten.add.Tensor
    t_mul_4 = _k0(t_add)  # aten.mul.Tensor
    t_mul_5 = mul_tt(t_convert_element_type, t_convert_element_type)  # aten.mul.Tensor
    t_mul_6 = _k3(t_mul_5)  # aten.mul.Tensor
    t_exp = exp(t_mul_6)  # aten.exp.default
    t_mul_7 = _k4(t_exp)  # aten.mul.Tensor
    t_mul_8 = mul_tt(t_convert_element_type, t_mul_7)  # aten.mul.Tensor
    t_add_2 = _k5(t_mul_4, t_mul_8)  # aten.add.Tensor
    t_mul_9 = mul_tt(t_convert_element_type_2, t_add_2)  # aten.mul.Tensor
    t_convert_element_type_4 = t_mul_9.to(dtype=torch.bfloat16)  # prims.convert_element_type.default
    return (t_convert_element_type_4,)

def gelu_forward_with_saved(x):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('gelu_forward requires CUDA tensor x')
    x = x.contiguous() if not x.is_contiguous() else x
    out = torch.empty_like(x)
    n = x.numel()
    if n:
        _gelu_fwd_kernel[((n + _BLOCK - 1) // _BLOCK,)](x, out, n, BLOCK=_BLOCK)
    return out, (x,)

def gelu_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 1:
        raise ValueError('gelu_backward saved-state arity mismatch')
    x = saved_tensors[0]
    if not (isinstance(x, torch.Tensor) and isinstance(dout, torch.Tensor) and x.is_cuda and dout.is_cuda):
        raise ValueError('gelu_backward requires CUDA tensors')
    dout = dout.contiguous() if not dout.is_contiguous() else dout
    dx = torch.empty_like(dout)
    n = dout.numel()
    if n:
        _gelu_bwd_kernel[((n + _BLOCK - 1) // _BLOCK,)](dout, x, dx, n, BLOCK=_BLOCK)
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'gelu', 'arguments': ['x'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx'], 'gradient_return_order': ['dx'], 'scalars': [], 'deployment_entry': 'gelu_deployment', 'allowed_primitives': []}


class GeluFunction(torch.autograd.Function):
    """Static autograd wiring for `gelu`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x):
        out, saved = gelu_forward_with_saved(x)
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
        dx, = gelu_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx,


def gelu_deployment(x):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = GeluFunction.apply(x)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class GeluModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return gelu_deployment(x)


DEPLOYMENT_ENTRY = "gelu_deployment"

