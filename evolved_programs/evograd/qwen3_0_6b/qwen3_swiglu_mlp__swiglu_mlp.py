"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm'}, {'side': 'forward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_1'}, {'side': 'forward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm_2'}, {'side': 'backward', 'node_name': 'mm_3', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_3'}, {'side': 'backward', 'node_name': 'mm_4', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_4'}, {'side': 'backward', 'node_name': 'mm_5', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_5'}, {'side': 'backward', 'node_name': 'mm_6', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_6'}, {'side': 'backward', 'node_name': 'mm_7', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_7'}, {'side': 'backward', 'node_name': 'mm_8', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_8'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 2, 'T': 2048, 'H': 1024, 'I': 3072}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_external_forward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 3072), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 3072), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_forward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_3(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((1024, 3072), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_4(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 3072), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_5(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((3072, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_6(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_7(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((3072, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_8(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 1024), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


# EVOLVE-BLOCK-START

_BLOCK = 1024

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

def _grid(n: int):
    return ((n + _BLOCK - 1) // _BLOCK,)

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

def div_tt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _binary_inputs(a, b, true_divide=True)
    out = torch.empty_like(a)
    _div_kernel[_grid(a.numel())](a, b, out, N=a.numel(), BLOCK=_BLOCK)
    return out

def _binary_inputs(a, b, *, true_divide=False):
    """ATen promotion must precede broadcasting away scalar-tensor rank."""
    dtype = torch.result_type(a, b)
    if true_divide and (not dtype.is_floating_point) and (not dtype.is_complex):
        dtype = torch.get_default_dtype()
    a, b = torch.broadcast_tensors(a, b)
    return (a.to(dtype).contiguous(), b.to(dtype).contiguous())

@triton.jit
def _div_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, a / b, mask=mask)

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

def _k0(a, alpha=1, s=1):
    return add_scalar(a, s, alpha)

def _k1(a, s=1):
    return mul_scalar(a, s)

def _k2(a, alpha=1, s=1):
    return rsub_scalar(a, s, alpha)

def _k3(a, b, alpha=1):
    # Only used for dx accumulation; shapes/dtypes already match and `a` is dead after this.
    _add_kernel[_grid(a.numel())](a, b, a, alpha, N=a.numel(), BLOCK=_BLOCK)
    return a

def swiglu_forward(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(g)
    _swiglu_fwd_kernel[_grid(g.numel())](g, u, out, N=g.numel(), BLOCK=_BLOCK)
    return out

@triton.jit
def _swiglu_fwd_kernel(gp, up, outp, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    tl.store(outp + offs, (g * s) * u, mask=mask)

def swiglu_backward(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor):
    dup = torch.empty_like(g)
    dg = torch.empty_like(g)
    hidden = torch.empty_like(g)
    _swiglu_bwd_kernel[_grid(g.numel())](dh, g, u, dup, dg, hidden, N=g.numel(), BLOCK=_BLOCK)
    return dup, dg, hidden

@triton.jit
def _swiglu_bwd_kernel(dhp, gp, up, dupp, dgp, hp, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    dh = tl.load(dhp + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gp + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sigmoid(g)
    silu = g * s
    h = silu * u
    dsilu = s * (1.0 + g * (1.0 - s))
    tl.store(dupp + offs, dh * silu, mask=mask)
    tl.store(dgp + offs, dh * u * dsilu, mask=mask)
    tl.store(hp + offs, h, mask=mask)

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

def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4):
    t_view = t_primals_1.reshape([4096, 1024])
    t_mm = _unified_external_forward_t_mm(t_view, t_primals_2.t())
    t_mm_1 = _unified_external_forward_t_mm_1(t_view, t_primals_3.t())
    t_mul = swiglu_forward(t_mm.reshape([2, 2048, 3072]), t_mm_1.reshape([2, 2048, 3072]))
    t_mm_2 = _unified_external_forward_t_mm_2(t_mul.reshape([4096, 3072]), t_primals_4.t())
    return (t_mm_2.reshape([2, 2048, 1024]), t_view, t_mm, t_mm_1, t_primals_4, t_primals_3, t_primals_2)

def _unified_backward_graph(t_view, t_mm, t_mm_1, t_permute_5, t_permute_9, t_permute_13, t_tangents_1):
    t_view_6 = t_tangents_1.reshape([4096, 1024])
    t_mm_4 = _unified_external_backward_t_mm_4(t_view_6, t_permute_5)
    t_mul_1, t_convert_element_type_14, t_hidden = swiglu_backward(
        t_mm_4.reshape([2, 2048, 3072]),
        t_mm.reshape([2, 2048, 3072]),
        t_mm_1.reshape([2, 2048, 3072]),
    )
    t_mm_3 = _unified_external_backward_t_mm_3(t_view_6.t(), t_hidden.reshape([4096, 3072]))
    t_view_8 = t_mul_1.reshape([4096, 3072])
    t_mm_5 = _unified_external_backward_t_mm_5(t_view_8.t(), t_view)
    t_mm_6 = _unified_external_backward_t_mm_6(t_view_8, t_permute_9)
    t_view_10 = t_convert_element_type_14.reshape([4096, 3072])
    t_mm_7 = _unified_external_backward_t_mm_7(t_view_10.t(), t_view)
    t_mm_8 = _unified_external_backward_t_mm_8(t_view_10, t_permute_13)
    return (_k3(t_mm_6.reshape([2, 2048, 1024]), t_mm_8.reshape([2, 2048, 1024])), t_mm_7, t_mm_5, t_mm_3)

def qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight):
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError('unified seed requires CUDA tensor x')
    if tuple(x.shape) != (2, 2048, 1024) or x.dtype != torch.bfloat16 or x.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for x')
    if not isinstance(gate_weight, torch.Tensor) or not gate_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor gate_weight')
    if tuple(gate_weight.shape) != (3072, 1024) or gate_weight.dtype != torch.bfloat16 or gate_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for gate_weight')
    if not isinstance(up_weight, torch.Tensor) or not up_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor up_weight')
    if tuple(up_weight.shape) != (3072, 1024) or up_weight.dtype != torch.bfloat16 or up_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for up_weight')
    if not isinstance(down_weight, torch.Tensor) or not down_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor down_weight')
    if tuple(down_weight.shape) != (1024, 3072) or down_weight.dtype != torch.bfloat16 or down_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for down_weight')
    if any(t.device != x.device for t in (gate_weight, up_weight, down_weight)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(x, gate_weight, up_weight, down_weight)
    return _outs[0], tuple(_outs[1:])

def qwen3_swiglu_mlp_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 6:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (2, 2048, 1024) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    shapes = ((4096, 1024), (4096, 3072), (4096, 3072), (1024, 3072), (3072, 1024), (3072, 1024))
    for t, s in zip(saved_tensors, shapes):
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError('unified backward requires CUDA tensors')
        if tuple(t.shape) != s or t.dtype != torch.bfloat16 or t.device != torch.device('cuda:0'):
            raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], saved_tensors[4], saved_tensors[5], dout.contiguous())
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'qwen3_swiglu_mlp', 'arguments': ['x', 'gate_weight', 'up_weight', 'down_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'gradient_return_order': ['dx', 'dgate_weight', 'dup_weight', 'ddown_weight'], 'scalars': [], 'deployment_entry': 'qwen3_swiglu_mlp_deployment', 'allowed_primitives': []}


class Qwen3SwigluMlpFunction(torch.autograd.Function):
    """Static autograd wiring for `qwen3_swiglu_mlp`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, x, gate_weight, up_weight, down_weight):
        out, saved = qwen3_swiglu_mlp_forward_with_saved(x, gate_weight, up_weight, down_weight)
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
        dx, dgate_weight, dup_weight, ddown_weight = qwen3_swiglu_mlp_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dx, dgate_weight, dup_weight, ddown_weight


def qwen3_swiglu_mlp_deployment(x, gate_weight, up_weight, down_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    _leading = None
    if x.dim() > 3:
        _leading = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    out = Qwen3SwigluMlpFunction.apply(x, gate_weight, up_weight, down_weight)
    if _leading is not None:
        out = out.view(*_leading, out.shape[-1])
    return out


class Qwen3SwigluMlpModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, gate_weight, up_weight, down_weight):
        super().__init__()
        self.gate_weight = (gate_weight if isinstance(gate_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(gate_weight))
        self.up_weight = (up_weight if isinstance(up_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(up_weight))
        self.down_weight = (down_weight if isinstance(down_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(down_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, x):
        return qwen3_swiglu_mlp_deployment(x, self.gate_weight, self.up_weight, self.down_weight)


DEPLOYMENT_ENTRY = "qwen3_swiglu_mlp_deployment"

