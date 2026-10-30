"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = []

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'rows': 256, 'hidden': 5120}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

# EVOLVE-BLOCK-START

_BLOCK = 1024
_GRID = (1280,)  # 256 * 5120 // 1024

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
    return ((n + _BLOCK - 1) // _BLOCK,)

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

@triton.jit
def _sigmoid_kernel(a_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, tl.sigmoid(a), mask=mask)

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

def rsub_scalar(a: torch.Tensor, scalar, alpha=1) -> torch.Tensor:
    if a.dtype == torch.bool or isinstance(scalar, bool):
        raise RuntimeError('Subtraction with Boolean tensors is not supported')
    a = _scalar_input(a, scalar)
    _check_alpha(alpha, a.dtype)
    out = torch.empty_like(a)
    _rsub_scalar_kernel[_grid(a.numel())](a, out, scalar, alpha, N=a.numel(), BLOCK=_BLOCK)
    return out

def _check_alpha(alpha, dtype):
    if isinstance(alpha, bool) and dtype != torch.bool:
        raise RuntimeError('Boolean alpha only supported for Boolean results')
    if isinstance(alpha, float) and (not (dtype.is_floating_point or dtype.is_complex)):
        raise RuntimeError('For integral input tensors, alpha must not be floating point')
    if isinstance(alpha, complex) and (not dtype.is_complex):
        raise RuntimeError('For non-complex input tensors, alpha must not be complex')

@triton.jit
def _rsub_scalar_kernel(a_ptr, out_ptr, scalar, alpha, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
        a = a.to(tl.float32)
    tl.store(out_ptr + offs, scalar - alpha * a, mask=mask)

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

def _k0(a, s=1.702):
    return mul_scalar(a, s)

def _k1(a, alpha=1, s=1):
    return rsub_scalar(a, s, alpha)

def _k2(a, b, alpha=1):
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

# Dead generic copy helpers removed; QuickGELU only uses the specialized kernels below.

@triton.jit
def _qgelu_fwd_kernel(xp, yp, dp, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(xp + offs).to(tl.float32)

    u = x * 1.702
    s = tl.sigmoid(u.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    tl.store(yp + offs, x * s)
    tl.store(dp + offs, s + u * s * (1.0 - s))

@triton.jit
def _qgelu_bwd_kernel(dp, gp, dxp, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.load(dp + offs).to(tl.float32)
    g = tl.load(gp + offs).to(tl.float32)
    tl.store(dxp + offs, g * d)

def _unified_forward_graph(t_primals_1):
    out = torch.empty_like(t_primals_1)
    deriv = torch.empty_like(t_primals_1)
    _qgelu_fwd_kernel[_GRID](
        t_primals_1, out, deriv, BLOCK=_BLOCK, num_warps=4
    )
    return (out, deriv)

def _unified_backward_graph(t_primals_1, t_tangents_1):
    dx = torch.empty_like(t_tangents_1)
    _qgelu_bwd_kernel[_GRID](
        t_primals_1, t_tangents_1, dx, BLOCK=_BLOCK, num_warps=4
    )
    return (dx,)

def quick_gelu_forward_with_saved(x):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (256, 5120) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    _outs = _unified_forward_graph(x)
    return _outs[0], tuple(_outs[1:])

def quick_gelu_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 1:
        raise ValueError('unified seed saved-state arity mismatch')
    deriv = saved_tensors[0]
    if not isinstance(deriv, torch.Tensor) or not deriv.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(deriv.shape) != (256, 5120) or deriv.dtype != torch.bfloat16 or deriv.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (256, 5120) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    _grads = _unified_backward_graph(deriv, dout)
    return (_grads[0],)


# EVOLVE-BLOCK-END



# ══════════════════════════════════════════════════════════════════════════════
# Deployment layer -- generated from the declaration, NOT evolvable.
#
# The public pair above is the implementation. This layer only wires it into
# autograd and exposes a stable entry point. Argument order, output arity,
# upstream-gradient order and input-gradient order are fixed by EvoGrad; the
# kernels, launch strategy and saved state above may change freely.
# ══════════════════════════════════════════════════════════════════════════════

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'quick_gelu', 'arguments': ['x'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx'], 'gradient_return_order': ['dx'], 'scalars': [], 'deployment_entry': 'quick_gelu_deployment', 'allowed_primitives': []}


class QuickGeluFunction(torch.autograd.Function):
    """Static autograd wiring for `quick_gelu`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x):
        out, saved = quick_gelu_forward_with_saved(x)
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
        dx, = quick_gelu_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx,


def quick_gelu_deployment(x):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 2:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = QuickGeluFunction.apply(x)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class QuickGeluModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, ):
        super().__init__()
        pass

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return quick_gelu_deployment(x)


DEPLOYMENT_ENTRY = "quick_gelu_deployment"

