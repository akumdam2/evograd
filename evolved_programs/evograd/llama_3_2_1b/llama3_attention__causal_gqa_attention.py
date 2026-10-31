"""Compiler-aware semantic seed: AOTAutograd + handwritten Triton + preserved external calls."""

from __future__ import annotations
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice
from torch._inductor.select_algorithm import extern_kernels as _unified_extern_kernels

UNIFIED_EXTERNAL_REGIONS = [{'side': 'forward', 'node_name': '_scaled_dot_product_flash_attention', 'target': 'aten._scaled_dot_product_flash_attention.default', 'kernel': 'torch.ops.aten._scaled_dot_product_flash_attention.default', 'wrapper': '_unified_external_forward_t__scaled_dot_product_flash_attention'}, {'side': 'forward', 'node_name': 'mm', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_forward_t_mm'}, {'side': 'backward', 'node_name': 'mm_1', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_1'}, {'side': 'backward', 'node_name': 'mm_2', 'target': 'aten.mm.default', 'kernel': 'extern_kernels.mm', 'wrapper': '_unified_external_backward_t_mm_2'}, {'side': 'backward', 'node_name': '_scaled_dot_product_flash_attention_backward', 'target': 'aten._scaled_dot_product_flash_attention_backward.default', 'kernel': 'torch.ops.aten._scaled_dot_product_flash_attention_backward.default', 'wrapper': '_unified_external_backward_t__scaled_dot_product_flash_attention_backward'}]

UNIFIED_DEPLOYMENT_CONFIG = {'dims': {'B': 2, 'T': 2048, 'HQ': 32, 'HK': 8, 'D': 64, 'QO': 2048, 'H': 2048}, 'dtype': 'bfloat16', 'device': 'cuda:0', 'family': None, 'dynamic': False}

def _unified_sdpa_forward(q, k, v, attn_bias=None, compute_log_sumexp=True, dropout_p=0.0,
                          is_causal=False, return_debug_mask=False, scale=None):
    """Fused attention forward, cuDNN argument order (a captured call omits trailing
    defaults); flash under deterministic algorithms."""
    if torch.are_deterministic_algorithms_enabled():
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            q, k, v, dropout_p, is_causal, return_debug_mask, scale=scale)
    return torch.ops.aten._scaled_dot_product_cudnn_attention.default(
        q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, scale=scale)


def _unified_sdpa_backward(grad_out, q, k, v, out, logsumexp, philox_seed, philox_offset, attn_bias,
                           cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, scale=None):
    """Fused attention backward, cuDNN argument order; flash under deterministic algorithms."""
    if torch.are_deterministic_algorithms_enabled():
        return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
            grad_out, q, k, v, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p,
            is_causal, philox_seed, philox_offset, scale=scale)
    return torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(
        grad_out, q, k, v, out, logsumexp, philox_seed, philox_offset, attn_bias, cum_seq_q,
        cum_seq_k, max_q, max_k, dropout_p, is_causal, scale=scale)


def _unified_external_forward_t__scaled_dot_product_flash_attention(arg0, arg1, arg2):
    with torch.no_grad():
        return _unified_sdpa_forward(arg0, arg1, arg2, None, True, 0.0, True, False, scale=0.125)


def _unified_external_forward_t_mm(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_1(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((2048, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t_mm_2(arg0, arg1):
    with torch.no_grad():
        out = torch.empty((4096, 2048), device=arg0.device, dtype=torch.bfloat16)
        _unified_extern_kernels.mm(arg0, arg1, out=out)
        return out


def _unified_external_backward_t__scaled_dot_product_flash_attention_backward(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
    with torch.no_grad():
        return _unified_sdpa_backward(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, None, None, None, 2048, 2048, 0.0, True, scale=0.125)


# EVOLVE-BLOCK-START

@triton.jit
def _attn_to_merged_kernel(x, y, s0, s1, s2, s3, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # destination is contiguous [B*T, HQ*D]
    col = offs & 2047
    d = col & 63
    h = col >> 6
    bt = offs >> 11
    t = bt & 2047
    b = bt >> 11

    # source is possibly non-contiguous [B, HQ, T, D]
    src = b * s0 + h * s1 + t * s2 + d * s3
    tl.store(y + offs, tl.load(x + src))


@triton.jit
def _merged_to_attn_kernel(x, y, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # destination is contiguous [B, HQ, T, D]
    d = offs & 63
    r = offs >> 6
    t = r & 2047
    r = r >> 11
    h = r & 31
    b = r >> 5

    # source is contiguous [B*T, HQ*D]
    src = (((b << 11) + t) << 11) + (h << 6) + d
    tl.store(y + offs, tl.load(x + src))


def attn_to_merged(a):
    out = torch.empty((4096, 2048), device=a.device, dtype=a.dtype)
    _attn_to_merged_kernel[(8192,)](
        a, out, a.stride(0), a.stride(1), a.stride(2), a.stride(3), BLOCK=1024
    )
    return out


def merged_to_attn(a):
    out = torch.empty((2, 32, 2048, 64), device=a.device, dtype=a.dtype)
    _merged_to_attn_kernel[(8192,)](a, out, BLOCK=1024)
    return out


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
    out_shape = [int(s) for s in out_shape]
    src_strides = [int(s) for s in src_strides]
    out = torch.empty(out_shape, device=a.device, dtype=a.dtype)
    n = out.numel()
    pad = 8 - len(out_shape)
    sizes = [1] * pad + out_shape
    strides = [0] * pad + src_strides
    _strided_copy_kernel[((n + 1023) // 1024,)](a, out, n, *sizes, *strides, BLOCK=1024)
    return out


def permute_copy(a, dims):
    dims = [int(d) % a.ndim for d in dims]
    return strided_copy(a, [a.shape[d] for d in dims], [a.stride(d) for d in dims])



def _unified_forward_graph(t_primals_1, t_primals_2, t_primals_3, t_primals_4):
    t__scaled_dot_product_flash_attention = _unified_external_forward_t__scaled_dot_product_flash_attention(t_primals_1, t_primals_2, t_primals_3)  # aten._scaled_dot_product_flash_attention.default
    t_getitem = t__scaled_dot_product_flash_attention[0]  # attention output, [2,32,2048,64]
    t_getitem_1 = t__scaled_dot_product_flash_attention[1]
    t_getitem_6 = t__scaled_dot_product_flash_attention[6]
    t_getitem_7 = t__scaled_dot_product_flash_attention[7]

    # Materialize [B,HQ,T,D] -> [B*T,HQ*D] for the o_proj GEMM.
    # The attention output may be non-contiguous, so the specialized kernel uses strides.
    t_view_2 = attn_to_merged(t_getitem)

    # cuBLAS/extern mm can consume a transposed 2D view; avoid an 8MB copy.
    t_permute_1 = t_primals_4.transpose(0, 1)
    t_mm = _unified_external_forward_t_mm(t_view_2, t_permute_1)
    t_view_3 = t_mm.reshape([2, 2048, 2048])

    # Save original o_weight, not a copied transpose-back.
    return (t_view_3, t_primals_1, t_primals_2, t_primals_3, t_getitem_1, t_getitem_6, t_getitem_7, t_view_2, t_primals_4)

def _unified_backward_graph(t_primals_1, t_primals_2, t_primals_3, t_getitem_1, t_getitem_6, t_getitem_7, t_view_2, t_permute_4, t_tangents_1):
    t_view_4 = t_tangents_1.reshape([4096, 2048])

    # do_weight = dout^T @ merged.  Use a transpose view instead of materializing
    # dout^T, and return the GEMM result directly; the old code transposed it
    # twice, producing the same layout after two extra full-matrix copies.
    t_permute_2 = t_view_4.transpose(0, 1)
    t_mm_1 = _unified_external_backward_t_mm_1(t_permute_2, t_view_2)

    # dmerged = dout @ o_weight.
    t_mm_2 = _unified_external_backward_t_mm_2(t_view_4, t_permute_4)
    t_view_5 = t_mm_2.reshape([2, 2048, 2048])
    t_view_6 = t_view_5.reshape([2, 2048, 32, 64])
    t_permute_6 = permute_copy(t_view_6, [0, 2, 1, 3])

    # Reconstruct attention output as a view from saved merged.
    # This removes one saved [2,32,2048,64] bf16 tensor.
    t_alias_1 = t_view_2.reshape([2, 2048, 32, 64]).transpose(1, 2)
    t__scaled_dot_product_flash_attention_backward = _unified_external_backward_t__scaled_dot_product_flash_attention_backward(t_permute_6, t_primals_1, t_primals_2, t_primals_3, t_alias_1, t_getitem_1, t_getitem_6, t_getitem_7)
    t_getitem_9 = t__scaled_dot_product_flash_attention_backward[0]
    t_getitem_10 = t__scaled_dot_product_flash_attention_backward[1]
    t_getitem_11 = t__scaled_dot_product_flash_attention_backward[2]
    return (t_getitem_9, t_getitem_10, t_getitem_11, t_mm_1)

def llama3_attention_forward_with_saved(q, k, v, o_weight):
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError('unified seed requires CUDA tensor q')
    if tuple(q.shape) != (2, 32, 2048, 64) or q.dtype != torch.bfloat16 or q.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for q')
    if not isinstance(k, torch.Tensor) or not k.is_cuda:
        raise ValueError('unified seed requires CUDA tensor k')
    if tuple(k.shape) != (2, 8, 2048, 64) or k.dtype != torch.bfloat16 or k.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for k')
    if not isinstance(v, torch.Tensor) or not v.is_cuda:
        raise ValueError('unified seed requires CUDA tensor v')
    if tuple(v.shape) != (2, 8, 2048, 64) or v.dtype != torch.bfloat16 or v.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for v')
    if not isinstance(o_weight, torch.Tensor) or not o_weight.is_cuda:
        raise ValueError('unified seed requires CUDA tensor o_weight')
    if tuple(o_weight.shape) != (2048, 2048) or o_weight.dtype != torch.bfloat16 or o_weight.device != torch.device('cuda:0'):
        raise ValueError('unified seed configuration mismatch for o_weight')
    if any(t.device != q.device for t in (k, v, o_weight)):
        raise ValueError('all unified seed inputs must use the same CUDA device')
    with torch.cuda.device('cuda:0'):
        _outs = _unified_forward_graph(q, k, v, o_weight)
    return _outs[0], tuple(_outs[1:])

def llama3_attention_backward_from_saved(dout, saved_tensors):
    if len(saved_tensors) != 8:
        raise ValueError('unified seed saved-state arity mismatch')
    if not isinstance(saved_tensors[0], torch.Tensor) or not saved_tensors[0].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[0].shape) != (2, 32, 2048, 64) or saved_tensors[0].dtype != torch.bfloat16 or saved_tensors[0].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[1], torch.Tensor) or not saved_tensors[1].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[1].shape) != (2, 8, 2048, 64) or saved_tensors[1].dtype != torch.bfloat16 or saved_tensors[1].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[2], torch.Tensor) or not saved_tensors[2].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[2].shape) != (2, 8, 2048, 64) or saved_tensors[2].dtype != torch.bfloat16 or saved_tensors[2].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[3], torch.Tensor) or not saved_tensors[3].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if saved_tensors[3].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[4], torch.Tensor) or not saved_tensors[4].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if saved_tensors[4].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[5], torch.Tensor) or not saved_tensors[5].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if saved_tensors[5].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[6], torch.Tensor) or not saved_tensors[6].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[6].shape) != (4096, 2048) or saved_tensors[6].dtype != torch.bfloat16 or saved_tensors[6].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    if not isinstance(saved_tensors[7], torch.Tensor) or not saved_tensors[7].is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(saved_tensors[7].shape) != (2048, 2048) or saved_tensors[7].dtype != torch.bfloat16 or saved_tensors[7].device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    # saved_tensors[8] used to be the attention output. It is reconstructed
    # from saved_tensors[6] to reduce saved activation memory.
    if not isinstance(dout, torch.Tensor) or not dout.is_cuda:
        raise ValueError('unified backward requires CUDA tensors')
    if tuple(dout.shape) != (2, 2048, 2048) or dout.dtype != torch.bfloat16 or dout.device != torch.device('cuda:0'):
        raise ValueError('unified backward configuration mismatch')
    with torch.cuda.device('cuda:0'):
        _grads = _unified_backward_graph(saved_tensors[0], saved_tensors[1], saved_tensors[2], saved_tensors[3], saved_tensors[4], saved_tensors[5], saved_tensors[6], saved_tensors[7], dout.contiguous())
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

ARTIFACT_CONTRACT = {'contract': 'evograd-artifact/1', 'op': 'llama3_attention', 'arguments': ['q', 'k', 'v', 'o_weight'], 'outputs': ['out'], 'upstream_grads': ['dout'], 'input_grads': ['dq', 'dk', 'dv', 'do_weight'], 'gradient_return_order': ['dq', 'dk', 'dv', 'do_weight'], 'scalars': [], 'deployment_entry': 'llama3_attention_deployment', 'allowed_primitives': []}


class Llama3AttentionFunction(torch.autograd.Function):
    """Static autograd wiring for `llama3_attention`. No declaration is read at runtime."""

    @staticmethod
    def forward(ctx, q, k, v, o_weight):
        out, saved = llama3_attention_forward_with_saved(q, k, v, o_weight)
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
        dq, dk, dv, do_weight = llama3_attention_backward_from_saved(
            dout, saved
        )
        # One value per forward argument, `None` for every inactive one.
        return dq, dk, dv, do_weight


def llama3_attention_deployment(q, k, v, o_weight):
    """The deployment callable: tier 2 benchmarks it, tier 3 patches it in.

    Leading dimensions are flattened to the declared rank and restored
    afterwards. `reshape` on a contiguous tensor is metadata only -- no copy --
    and the kernel is entitled to assume the rank the declaration states.
    """
    return Llama3AttentionFunction.apply(q, k, v, o_weight)


class Llama3AttentionModule(torch.nn.Module):
    """Thin module holding the declared parameters; no logic of its own."""

    def __init__(self, o_weight):
        super().__init__()
        self.o_weight = (o_weight if isinstance(o_weight, torch.nn.Parameter)
                  else torch.nn.Parameter(o_weight))

        self.adapter_kind = "evolved_direct_autograd_module"

    def forward(self, q, k, v):
        return llama3_attention_deployment(q, k, v, self.o_weight)


DEPLOYMENT_ENTRY = "llama3_attention_deployment"

