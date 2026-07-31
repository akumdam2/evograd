"""Pipeline D seed for `rmsnorm` (float16) -- Inductor kernels, unmodified.

Produced by AOTAutograd + min_cut_rematerialization_partition + Inductor. The
save-set below is the partitioner's choice, not a policy of this pipeline;
Pipeline B's inputs-only contract is the same decision at memory budget 0.

Partitioner save-set -- 3 entries,
66,816 bytes at the capture shape:
    primals_1: ['s26', 's49'] torch.float16 <- forward input
    primals_2: ['s49'] torch.float16 <- forward input
    rsqrt: ['s26', 1] torch.float32 <- aten.rsqrt.default

This is a float16 specialist, because Inductor specializes on dtype. Shapes are
generic: captured with dynamic shapes, so sizes arrive as runtime arguments.

The saved-tensor contract is a private agreement between the two functions in
this file, so the search may change it -- the forward's returns and the
backward's unpacking need only stay consistent. Correctness is checked against
`torch.autograd.grad` on the declared forward reference.

Captured on cuda.
"""

from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties


# EVOLVE-BLOCK-START

# ==== forward kernels ====================================================
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %buf0 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=buf0]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_2 : Tensor "f16[s49][1]cuda:0" = PlaceHolder[target=primals_2]
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %mul : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %convert_element_type), kwargs = {})
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul, [-1], True), kwargs = {})
#   %add : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul_2 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %convert_element_type_3), kwargs = {})
#   %convert_element_type_4 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.float16), kwargs = {})
#   return %buf0,%rsqrt,%convert_element_type_4

triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'fwd_triton_red_fused__to_copy_add_mean_mul_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'CA95D8F30F1EC9BA1CBBE069AF120E14BCC3CD8A2C22B202F79770958779697A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 512, 'r0_': 197632}}
)
@triton.jit
def fwd_triton_red_fused__to_copy_add_mean_mul_rsqrt_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = ks0
    tmp7 = tmp6.to(tl.float32)
    tmp8 = (tmp4 / tmp7)
    tmp9 = tl.full([1, 1], 1e-05, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 * tmp11
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tl.store(out_ptr0 + (r0_1 + ks0*x0), tmp18, r0_mask & xmask)




# ==== backward kernels ===================================================
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Topologically Sorted Source Nodes: [convert_element_type_5, mul_3, sum_1, view, convert_element_type_6], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
# Source node to ATen node mapping:
#   convert_element_type_5 => convert_element_type_5
#   convert_element_type_6 => convert_element_type_6
#   mul_3 => mul_3
#   sum_1 => sum_1
#   view => view
# Graph fragment:
#   %tangents_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_1 : Tensor "f32[1, s49][s49, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %convert_element_type_5 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %mul_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %rsqrt), kwargs = {})
#   %mul_3 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %mul_1), kwargs = {})
#   %sum_1 : Tensor "f32[1, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0], True), kwargs = {})
#   %view : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_1, [%sym_size_int]), kwargs = {})
#   %convert_element_type_6 : Tensor "f16[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float16), kwargs = {})
#   return %sum_1,%convert_element_type_6

triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'bwd_triton_red_fused__to_copy_mul_sum_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'CA95D8F30F1EC9BA1CBBE069AF120E14BCC3CD8A2C22B202F79770958779697A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 133120, 'r0_': 256}}
)
@triton.jit
def bwd_triton_red_fused__to_copy_mul_sum_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tmp8.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp10, xmask)



# Topologically Sorted Source Nodes: [convert_element_type_5, mul_4, mul_5, mul_6, sum_2, mul_7, pow_1, mul_8, convert_element_type_7, expand, div, mul_9, convert_element_type_8, add_1, add_2], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_2 => add_2
#   convert_element_type_5 => convert_element_type_5
#   convert_element_type_7 => convert_element_type_7
#   convert_element_type_8 => convert_element_type_8
#   div => div
#   expand => expand
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   pow_1 => pow_1
#   sum_2 => sum_2
# Graph fragment:
#   %tangents_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_2 : Tensor "f16[s49][1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_2 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=sum_2]
#   %convert_element_type_5 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul_4 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %convert_element_type_3), kwargs = {})
#   %mul_5 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %convert_element_type), kwargs = {})
#   %mul_6 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %rsqrt), kwargs = {})
#   %sum_2 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [1], True), kwargs = {})
#   %mul_7 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_2, -0.5), kwargs = {})
#   %pow_1 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt, 3), kwargs = {})
#   %mul_8 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %pow_1), kwargs = {})
#   %convert_element_type_7 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.float16), kwargs = {})
#   %expand : Tensor "f32[s26, s49][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_8, [%sym_size_int_1, %sym_size_int_2]), kwargs = {})
#   %div : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, %sym_size_int_2), kwargs = {})
#   %mul_9 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %convert_element_type), kwargs = {})
#   %convert_element_type_8 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.float16), kwargs = {})
#   %add_1 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_7, %convert_element_type_8), kwargs = {})
#   %add_2 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %convert_element_type_8), kwargs = {})
#   return %sum_2,%add_2

triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr1': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'bwd_triton_red_fused__to_copy_add_div_expand_mul_pow_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'CA95D8F30F1EC9BA1CBBE069AF120E14BCC3CD8A2C22B202F79770958779697A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 256, 'r0_': 263168}}
)
@triton.jit
def bwd_triton_red_fused__to_copy_add_div_expand_mul_pow_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.full([1, 1], -0.5, tl.float32)
        tmp20 = tmp9 * tmp19
        tmp21 = tmp16 * tmp16
        tmp22 = tmp21 * tmp16
        tmp23 = tmp20 * tmp22
        tmp24 = ks0
        tmp25 = tmp24.to(tl.float32)
        tmp26 = (tmp23 / tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp18 + tmp30
        tmp32 = tmp31 + tmp30
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp32, r0_mask & xmask)





def _fwd_call(primals_1, primals_2):
    primals_1_size = primals_1.size()
    s26 = primals_1_size[0]
    s49 = primals_1_size[1]
    assert_size_stride(primals_1, (s26, s49), (s49, 1))
    assert_size_stride(primals_2, (s49, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s26, 1), (1, s26), torch.float32)
        buf1 = reinterpret_tensor(buf0, (s26, 1), (1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((s26, s49), (s49, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        fwd_triton_red_fused__to_copy_add_mean_mul_rsqrt_0.run(buf1, primals_1, primals_2, buf2, s49, s26, s49, stream=stream0)
    return (buf2, primals_1, primals_2, buf1, )


def _bwd_call(primals_1, primals_2, rsqrt, tangents_1):
    primals_1_size = primals_1.size()
    s26 = primals_1_size[0]
    s49 = primals_1_size[1]
    assert_size_stride(primals_1, (s26, s49), (s49, 1))
    assert_size_stride(primals_2, (s49, ), (1, ))
    assert_size_stride(rsqrt, (s26, 1), (1, 1))
    assert_size_stride(tangents_1, (s26, s49), (s49, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((s49, ), (1, ), torch.float16)
        # Topologically Sorted Source Nodes: [convert_element_type_5, mul_3, sum_1, view, convert_element_type_6], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.view]
        stream0 = get_raw_stream(0)
        bwd_triton_red_fused__to_copy_mul_sum_view_0.run(tangents_1, primals_1, rsqrt, buf1, s49, s49, s26, stream=stream0)
        buf3 = empty_strided_cuda((s26, s49), (s49, 1), torch.float16)
        # Topologically Sorted Source Nodes: [convert_element_type_5, mul_4, mul_5, mul_6, sum_2, mul_7, pow_1, mul_8, convert_element_type_7, expand, div, mul_9, convert_element_type_8, add_1, add_2], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
        stream0 = get_raw_stream(0)
        bwd_triton_red_fused__to_copy_add_div_expand_mul_pow_sum_1.run(tangents_1, primals_2, primals_1, rsqrt, buf3, s49, s26, s49, stream=stream0)
        del primals_1
        del primals_2
        del rsqrt
        del tangents_1
    return (buf3, buf1, )



_SEED_DTYPE = torch.float16

# How to build the backward's argument list: ("s", i) takes saved[i], ("t", i)
# takes tangent i. AOTAutograd orders the backward's placeholders with symbolic
# sizes first while the forward returns them in graph order, so the save-set is
# a permutation apart rather than a pass-through.
_BWD_ARG_SPEC = (('s', 0), ('s', 1), ('s', 2), ('t', 0))


def _check_dtype(tensor):
    if tensor.dtype is not _SEED_DTYPE:
        raise NotImplementedError(
            f"rmsnorm: this seed is a {_SEED_DTYPE} specialist, got {tensor.dtype}"
        )


def _build_backward_args(spec, saved, tangents):
    """Assemble the backward's argument list.

    Saved values pass through untouched: they came from this file's forward, so
    they already carry the exact strides the generated code asserts. Forcing
    them contiguous would break saved transposed views. Tangents arrive from the
    caller, so those are normalized.
    """
    return [saved[i] if kind == "s" else tangents[i].contiguous() for kind, i in spec]


def _forward_with_saved_impl(x, weight, eps=1e-5):
    # Baked into the kernels at trace time (eps=1e-05); the parameter is
    # accepted for API compatibility. Changing it requires re-capturing,
    # or rewriting the kernels to take it as a runtime argument.
    _ = eps
    _check_dtype(x)
    _out = _fwd_call(x.contiguous(), weight.contiguous())
    return _out[0], tuple(_out[1:])


def _backward_from_saved_impl(dy, saved_tensors, eps=1e-5):
    _ = eps
    _grads = _bwd_call(
        *_build_backward_args(_BWD_ARG_SPEC, saved_tensors, (dy,))
    )
    return (_grads[0], _grads[1],)


def rmsnorm_forward_with_saved(x, weight, eps=1e-5):
    return _forward_with_saved_impl(x, weight, eps)


def rmsnorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    return _backward_from_saved_impl(dy, saved_tensors, eps)

# EVOLVE-BLOCK-END

