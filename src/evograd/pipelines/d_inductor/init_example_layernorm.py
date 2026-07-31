"""Pipeline D seed for `layernorm` (float16) -- Inductor kernels, unmodified.

Produced by AOTAutograd + min_cut_rematerialization_partition + Inductor. The
save-set below is the partitioner's choice, not a policy of this pipeline;
Pipeline B's inputs-only contract is the same decision at memory budget 0.

Partitioner save-set -- 5 entries,
67,072 bytes at the capture shape:
    primals_1: ['s26', 's49'] torch.float16 <- forward input
    primals_2: ['s49'] torch.float16 <- forward input
    mean: ['s26', 1] torch.float32 <- aten.mean.dim
    rsqrt: ['s26', 1] torch.float32 <- aten.rsqrt.default
    sym_size_int: symint <- aten.sym_size.int

This is a float16 specialist, because Inductor specializes on dtype. Shapes are
generic: captured with dynamic shapes, so sizes arrive as runtime arguments.

Launch configs are pinned to the config autotuning chose at capture.

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
from torch._C._dynamo.guards import copy_if_misaligned
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


# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mean, aten.sub, aten.pow, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %buf0 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=buf0]
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=mean]
#   %buf2 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=buf2]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %primals_2 : Tensor "f16[s49][1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "f16[s49][1]cuda:0" = PlaceHolder[target=primals_3]
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convert_element_type, [-1], True), kwargs = {})
#   %sub : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %mean), kwargs = {})
#   %pow_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_1 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %convert_element_type_3), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %add_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %convert_element_type_4), kwargs = {})
#   %convert_element_type_5 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float16), kwargs = {})
#   return %buf0,%mean,%buf2,%rsqrt,%convert_element_type_5

triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 2, 'R0_BLOCK': 512, 'num_warps': 4, 'num_stages': 1},
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'fwd_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sub_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 1024, 'r0_': 198656}, 'backend_hash': '240F77412A6363F4D266566D63B201FE42440549AB6618DECFFBECD4EE8A38E2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
)
@triton.jit
def fwd_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sub_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = (ks0).to(tl.float32)
    tmp6 = (tmp5).to(tl.float32)
    tmp7 = (tmp3 / tmp6)
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 - tmp7
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = (ks0).to(tl.float32)
    tmp16 = (tmp15).to(tl.float32)
    tmp17 = (tmp13 / tmp16)
    tmp18 = tl.full([1, 1], 1e-05, tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(in_out_ptr1 + (x0), tmp20, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp21 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp7
        tmp24 = tmp23 * tmp20
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 * tmp26
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 + tmp29
        tmp31 = tmp30.to(tl.float32)
        tl.store(out_ptr0 + (r0_1 + ks0*x0), tmp31, r0_mask & xmask)




# ==== backward kernels ===================================================
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# Topologically Sorted Source Nodes: [convert_element_type_6, sum_1, view, convert_element_type_7, mul_2, sum_2, view_1, convert_element_type_8], Original ATen: [aten._to_copy, aten.sum, aten.view, aten.sub, aten.mul]
# Source node to ATen node mapping:
#   convert_element_type_6 => convert_element_type_6
#   convert_element_type_7 => convert_element_type_7
#   convert_element_type_8 => convert_element_type_8
#   mul_2 => mul_2
#   sum_1 => sum_1
#   sum_2 => sum_2
#   view => view
#   view_1 => view_1
# Graph fragment:
#   %tangents_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=mean]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_1 : Tensor "f32[1, s49][s49, 1]cuda:0" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "f32[1, s49][s49, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %convert_element_type_6 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %sum_1 : Tensor "f32[1, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_6, [0], True), kwargs = {})
#   %view : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_1, [%sym_size_int]), kwargs = {})
#   %convert_element_type_7 : Tensor "f16[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.float16), kwargs = {})
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %mean), kwargs = {})
#   %mul : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %mul), kwargs = {})
#   %sum_2 : Tensor "f32[1, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [0], True), kwargs = {})
#   %view_1 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sum_2, [%sym_size_int_1]), kwargs = {})
#   %convert_element_type_8 : Tensor "f16[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float16), kwargs = {})
#   return %sum_1,%sum_2,%convert_element_type_7,%convert_element_type_8

triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 4, 'R0_BLOCK': 64, 'num_warps': 2, 'num_stages': 1},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp16', 'out_ptr3': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'bwd_triton_red_fused__to_copy_mul_sub_sum_view_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 135168, 'r0_': 512}, 'backend_hash': '240F77412A6363F4D266566D63B201FE42440549AB6618DECFFBECD4EE8A38E2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
)
@triton.jit
def bwd_triton_red_fused__to_copy_mul_sub_sum_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr1 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 - tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tmp1 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tmp3.to(tl.float32)
    tmp16 = tmp13.to(tl.float32)
    tl.store(out_ptr2 + (x0), tmp15, xmask)
    tl.store(out_ptr3 + (x0), tmp16, xmask)



# Topologically Sorted Source Nodes: [convert_element_type_6, mul_3, mul_4, mul_5, sum_3, mul_6, pow_2, mul_7, neg, sum_4, convert_element_type_9, expand, div, pow_3, mul_8, mul_9, neg_1, sum_5, add_2, convert_element_type_10, add_3, expand_1, div_1, convert_element_type_11, add_4], Original ATen: [aten._to_copy, aten.sub, aten.mul, aten.sum, aten.pow, aten.neg, aten.expand, aten.div, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   convert_element_type_10 => convert_element_type_10
#   convert_element_type_11 => convert_element_type_11
#   convert_element_type_6 => convert_element_type_6
#   convert_element_type_9 => convert_element_type_9
#   div => div
#   div_1 => div_1
#   expand => expand
#   expand_1 => expand_1
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   neg => neg
#   neg_1 => neg_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   sum_3 => sum_3
#   sum_4 => sum_4
#   sum_5 => sum_5
# Graph fragment:
#   %tangents_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_2 : Tensor "f16[s49][1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_1 : Tensor "f16[s26, s49][s49, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %mean : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=mean]
#   %rsqrt : Tensor "f32[s26, 1][1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_3 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=sum_3]
#   %sum_4 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=sum_4]
#   %sum_5 : Tensor "f32[s26, 1][1, s26]cuda:0" = PlaceHolder[target=sum_5]
#   %convert_element_type_6 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %mean), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[s49][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %mul_3 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %convert_element_type_3), kwargs = {})
#   %mul_4 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sub), kwargs = {})
#   %mul_5 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %rsqrt), kwargs = {})
#   %sum_3 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1], True), kwargs = {})
#   %mul_6 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_3, -0.5), kwargs = {})
#   %pow_2 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt, 3), kwargs = {})
#   %mul_7 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %pow_2), kwargs = {})
#   %neg : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_5,), kwargs = {})
#   %sum_4 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg, [1], True), kwargs = {})
#   %convert_element_type_9 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.float16), kwargs = {})
#   %expand : Tensor "f32[s26, s49][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_7, [%sym_size_int_2, %sym_size_int_3]), kwargs = {})
#   %div : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, %sym_size_int_3), kwargs = {})
#   %pow_3 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 1.0), kwargs = {})
#   %mul_8 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_3, 2.0), kwargs = {})
#   %mul_9 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_8), kwargs = {})
#   %neg_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_9,), kwargs = {})
#   %sum_5 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg_1, [1], True), kwargs = {})
#   %add_2 : Tensor "f32[s26, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_4, %sum_5), kwargs = {})
#   %convert_element_type_10 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.float16), kwargs = {})
#   %add_3 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, %convert_element_type_10), kwargs = {})
#   %expand_1 : Tensor "f32[s26, s49][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%add_2, [%sym_size_int_2, %sym_size_int_3]), kwargs = {})
#   %div_1 : Tensor "f32[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, %sym_size_int_3), kwargs = {})
#   %convert_element_type_11 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.float16), kwargs = {})
#   %add_4 : Tensor "f16[s26, s49][s49, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %convert_element_type_11), kwargs = {})
#   return %sum_3,%sum_4,%sum_5,%add_4

triton_helpers.set_driver_to_gpu()

@triton_heuristics.fixed_config(
    config={'XBLOCK': 2, 'R0_BLOCK': 512, 'num_warps': 4, 'num_stages': 1},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr3': '*fp16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'bwd_triton_red_fused__to_copy_add_div_expand_mul_neg_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 3, 'autotune_hints': set(), 'tiling_scores': {'x': 512, 'r0_': 263168}, 'backend_hash': '240F77412A6363F4D266566D63B201FE42440549AB6618DECFFBECD4EE8A38E2', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
)
@triton.jit
def bwd_triton_red_fused__to_copy_add_div_expand_mul_neg_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
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
        tmp8 = tmp6 - tmp7
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask & xmask, tmp12, _tmp11)
        tmp14 = tmp4 * tmp13
        tmp15 = -tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    _tmp35 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.full([1, 1], -0.5, tl.float32)
        tmp20 = tmp11 * tmp19
        tmp21 = tmp13 * tmp13
        tmp22 = tmp21 * tmp13
        tmp23 = tmp20 * tmp22
        tmp24 = (ks0).to(tl.float32)
        tmp25 = (tmp24).to(tl.float32)
        tmp26 = (tmp23 / tmp25)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp28 - tmp7
        tmp30 = tl.full([1, 1], 2.0, tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp26 * tmp31
        tmp33 = -tmp32
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(r0_mask & xmask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp37 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp38 = tmp37.to(tl.float32)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp38 * tmp40
        tmp42 = tmp41 * tmp13
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tl.full([1, 1], -0.5, tl.float32)
        tmp45 = tmp11 * tmp44
        tmp46 = tmp13 * tmp13
        tmp47 = tmp46 * tmp13
        tmp48 = tmp45 * tmp47
        tmp49 = (ks0).to(tl.float32)
        tmp50 = (tmp49).to(tl.float32)
        tmp51 = (tmp48 / tmp50)
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp53 - tmp7
        tmp55 = tl.full([1, 1], 2.0, tl.float32)
        tmp56 = tmp54 * tmp55
        tmp57 = tmp51 * tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp59 = tmp43 + tmp58
        tmp60 = tmp17 + tmp35
        tmp61 = (tmp60 / tmp50)
        tmp62 = tmp61.to(tl.float32)
        tmp63 = tmp59 + tmp62
        tl.store(out_ptr3 + (r0_1 + ks0*x0), tmp63, r0_mask & xmask)





def _fwd_call(primals_1, primals_2, primals_3):
    primals_1_size = primals_1.size()
    s26 = primals_1_size[0]
    s49 = primals_1_size[1]
    assert_size_stride(primals_1, (s26, s49), (s49, 1), 'input')
    assert_size_stride(primals_2, (s49, ), (1, ), 'input')
    assert_size_stride(primals_3, (s49, ), (1, ), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        primals_1 = copy_if_misaligned(primals_1)
        primals_2 = copy_if_misaligned(primals_2)
        primals_3 = copy_if_misaligned(primals_3)
        buf0 = empty_strided_cuda((s26, 1), (1, s26), torch.float32)
        buf1 = reinterpret_tensor(buf0, (s26, 1), (1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((s26, 1), (1, s26), torch.float32)
        buf3 = reinterpret_tensor(buf2, (s26, 1), (1, 1), 0); del buf2  # reuse
        buf4 = empty_strided_cuda((s26, s49), (s49, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mean, aten.sub, aten.pow, aten.add, aten.rsqrt, aten.mul]
        raw_stream0 = get_raw_stream(0)
        fwd_triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sub_0.run(buf1, buf3, primals_1, primals_2, primals_3, buf4, s49, s26, s49, stream=raw_stream0)
        del primals_3
    return (buf4, primals_1, primals_2, buf1, buf3, s49, )


def _bwd_call(sym_size_int, primals_1, primals_2, mean, rsqrt, tangents_1):
    s49 = sym_size_int
    primals_1_size = primals_1.size()
    s26 = primals_1_size[0]
    assert_size_stride(tangents_1, (s26, s49), (s49, 1), 'input')
    assert_size_stride(primals_1, (s26, s49), (s49, 1), 'input')
    assert_size_stride(mean, (s26, 1), (1, 1), 'input')
    assert_size_stride(rsqrt, (s26, 1), (1, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        tangents_1 = copy_if_misaligned(tangents_1)
        primals_1 = copy_if_misaligned(primals_1)
        mean = copy_if_misaligned(mean)
        rsqrt = copy_if_misaligned(rsqrt)
        buf1 = empty_strided_cuda((s49, ), (1, ), torch.float16)
        buf3 = empty_strided_cuda((s49, ), (1, ), torch.float16)
        # Topologically Sorted Source Nodes: [convert_element_type_6, sum_1, view, convert_element_type_7, mul_2, sum_2, view_1, convert_element_type_8], Original ATen: [aten._to_copy, aten.sum, aten.view, aten.sub, aten.mul]
        raw_stream0 = get_raw_stream(0)
        bwd_triton_red_fused__to_copy_mul_sub_sum_view_0.run(tangents_1, primals_1, mean, rsqrt, buf1, buf3, s49, s49, s26, stream=raw_stream0)
        assert_size_stride(primals_2, (s49, ), (1, ), 'input')
        primals_2 = copy_if_misaligned(primals_2)
        buf7 = empty_strided_cuda((s26, s49), (s49, 1), torch.float16)
        # Topologically Sorted Source Nodes: [convert_element_type_6, mul_3, mul_4, mul_5, sum_3, mul_6, pow_2, mul_7, neg, sum_4, convert_element_type_9, expand, div, pow_3, mul_8, mul_9, neg_1, sum_5, add_2, convert_element_type_10, add_3, expand_1, div_1, convert_element_type_11, add_4], Original ATen: [aten._to_copy, aten.sub, aten.mul, aten.sum, aten.pow, aten.neg, aten.expand, aten.div, aten.add]
        raw_stream0 = get_raw_stream(0)
        bwd_triton_red_fused__to_copy_add_div_expand_mul_neg_pow_sub_sum_1.run(tangents_1, primals_2, primals_1, mean, rsqrt, buf7, s49, s26, s49, stream=raw_stream0)
        del mean
        del primals_1
        del primals_2
        del rsqrt
        del tangents_1
    return (buf7, buf3, buf1, )



_SEED_DTYPE = torch.float16

# How to build the backward's argument list: ("s", i) takes saved[i], ("t", i)
# takes tangent i. AOTAutograd orders the backward's placeholders with symbolic
# sizes first while the forward returns them in graph order, so the save-set is
# a permutation apart rather than a pass-through.
_BWD_ARG_SPEC = (('s', 4), ('s', 0), ('s', 1), ('s', 2), ('s', 3), ('t', 0))


def _check_dtype(tensor):
    if tensor.dtype is not _SEED_DTYPE:
        raise NotImplementedError(
            f"layernorm: this seed is a {_SEED_DTYPE} specialist, got {tensor.dtype}"
        )


def _build_backward_args(spec, saved, tangents):
    """Assemble the backward's argument list.

    Saved values pass through untouched: they came from this file's forward, so
    they already carry the exact strides the generated code asserts. Forcing
    them contiguous would break saved transposed views. Tangents arrive from the
    caller, so those are normalized.
    """
    return [saved[i] if kind == "s" else tangents[i].contiguous() for kind, i in spec]


def _forward_with_saved_impl(x, weight, bias, eps=1e-5):
    # Baked into the kernels at trace time (eps=1e-05); the parameter is
    # accepted for API compatibility. Changing it requires re-capturing,
    # or rewriting the kernels to take it as a runtime argument.
    _ = eps
    _check_dtype(x)
    _out = _fwd_call(x.contiguous(), weight.contiguous(), bias.contiguous())
    return _out[0], tuple(_out[1:])


def _backward_from_saved_impl(dy, saved_tensors, eps=1e-5):
    _ = eps
    _grads = _bwd_call(
        *_build_backward_args(_BWD_ARG_SPEC, saved_tensors, (dy,))
    )
    return (_grads[0], _grads[1], _grads[2],)


def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    return _forward_with_saved_impl(x, weight, bias, eps)


def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    return _backward_from_saved_impl(dy, saved_tensors, eps)

# EVOLVE-BLOCK-END

