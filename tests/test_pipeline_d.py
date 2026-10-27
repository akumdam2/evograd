"""Pipeline D seed assembly — pure string/dataclass work, no torch, no GPU.

The capture step needs a real device and is exercised by running the pipeline;
what is unit-testable is everything downstream of it: splitting Inductor's
generated modules, lifting Triton kernels out of the strings they ship in,
keeping the two halves from colliding, and rendering a wrapper that honours the
declared gradient contract.
"""

import ast
import unittest
from dataclasses import replace

from evograd.benchmark import get_task
from evograd.opdecl.activity import Workload
from evograd.pipelines.d_inductor.capture import CapturedPair, SavedTensor
from evograd.pipelines.d_inductor.seed_codegen import (
    _legalize_launch_code,
    _named_parameters,
    _split_top_level,
    _split_module,
    dtype_tag,
    generate_inductor_seed,
)
from evograd.pipelines.d_inductor.synthesize import (
    _dims_are_distinct,
    _with_distinct_dims,
    select_capture_workloads,
)
from evograd.pipelines.shared.artifact import (
    evolvable_region_covers_pair,
    find_forwarding_aliases,
)
from evograd.pipelines.shared.primitives import PrimitiveViolation, check_source

Q = "'" * 3

# A CPU capture: the kernel body is C++ and stays inside async_compile.
_CPP_TEMPLATE = """import torch
from torch._inductor.async_compile import AsyncCompile
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
async_compile = AsyncCompile()


@@KERNEL@@ = async_compile.cpp_pybinding(['float*', 'const float*'], r@@Q@@
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C" void kernel(float* out, const float* in) { }
@@Q@@)


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def call(self, args):
@@BODY@@

runner = Runner(partitions=[])
call = runner.call
"""

# A CUDA capture. Mirrors WrapperCodegen._format_kernel_definition plus
# TritonKernel.gen_common_triton_imports: the kernel is source text handed to
# async_compile.triton, carrying its own imports and both decorators.
_TRITON_TEMPLATE = """import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
aten = torch.ops.aten
inductor_ops = torch.ops.inductor
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


@@KERNEL@@ = async_compile.triton('@@KERNEL@@', @@Q@@
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, DeviceProperties

@triton_heuristics.pointwise(
    size_hints={'x': 1024},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32'}},
    inductor_meta={'kernel_name': '@@KERNEL@@'},
    min_elem_per_thread=0
)
@triton.jit
def @@KERNEL@@(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + xindex, xmask)
    tl.store(out_ptr0 + xindex, tmp0 * tmp0, xmask)
@@Q@@, device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def call(self, args):
@@BODY@@

runner = Runner(partitions=[])
call = runner.call
"""


def fake_module(template: str, kernel: str, body: str) -> str:
    return (
        template.replace("@@KERNEL@@", kernel)
        .replace("@@BODY@@", body)
        .replace("@@Q@@", Q)
    )


FWD_BODY = (
    "        primals_1, primals_2 = args\n"
    "        args.clear()\n"
    "        buf0 = empty_strided_cpu((4,), (1,), torch.float32)\n"
    "        triton_poi_fused_mul_0.run(primals_1, buf0, 4)\n"
    "        return (buf0, primals_1, primals_2, )"
)

BWD_BODY = (
    "        primals_1, primals_2, tangents_1 = args\n"
    "        args.clear()\n"
    "        buf0 = empty_strided_cpu((4,), (1,), torch.float32)\n"
    "        triton_poi_fused_mul_0.run(tangents_1, buf0, 4)\n"
    "        return (buf0, buf0, )"
)

# Both halves deliberately reuse the same kernel name: Inductor numbers kernels
# per graph, so a collision across the two modules is the normal case.
CPP_FWD = fake_module(_CPP_TEMPLATE, "triton_poi_fused_mul_0", FWD_BODY)
CPP_BWD = fake_module(_CPP_TEMPLATE, "triton_poi_fused_mul_0", BWD_BODY)
TRITON_FWD = fake_module(_TRITON_TEMPLATE, "triton_poi_fused_mul_0", FWD_BODY)
TRITON_BWD = fake_module(_TRITON_TEMPLATE, "triton_poi_fused_mul_0", BWD_BODY)


_CONFIGS = {
    "forward": {"triton_poi_fused_mul_0": {"XBLOCK": 1024, "num_warps": 4, "num_stages": 1}},
    "backward": {"triton_poi_fused_mul_0": {"XBLOCK": 512, "num_warps": 8, "num_stages": 2}},
}


def capture_for(op, spec=None, grad_indices=None, triton=False) -> CapturedPair:
    return CapturedPair(
        forward_source=TRITON_FWD if triton else CPP_FWD,
        backward_source=TRITON_BWD if triton else CPP_BWD,
        saved=(
            SavedTensor("primals_1", "placeholder", (4,), "torch.float32", True, 16),
            SavedTensor("primals_2", "placeholder", (4,), "torch.float32", True, 16),
        ),
        grad_indices=grad_indices if grad_indices is not None else (0, 1, 2),
        backward_arg_spec=spec or (("s", 0), ("s", 1), ("t", 0)),
        kernel_configs={k: dict(v) for k, v in _CONFIGS.items()},
        device="cuda" if triton else "cpu",
        dynamic=True,
        baked_scalars={},
    )


class TestSplitModule(unittest.TestCase):
    def test_splits_into_header_kernels_and_body(self):
        split = _split_module(CPP_FWD, "forward")
        self.assertEqual(split.header[-1], "async_compile = AsyncCompile()")
        self.assertEqual(split.symbols, ("triton_poi_fused_mul_0",))
        self.assertIn("primals_1, primals_2 = args", "\n".join(split.call_body))
        self.assertNotIn("runner = Runner", "\n".join(split.call_body))

    def test_rejects_module_without_call(self):
        with self.assertRaisesRegex(RuntimeError, "no 'def call"):
            _split_module(CPP_FWD.replace("def call(self, args):", "def x(self):"), "f")


class TestSeedAssembly(unittest.TestCase):
    def setUp(self):
        self.op = get_task("layernorm")

    def test_colliding_kernel_names_are_prefixed_apart(self):
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        self.assertIn("fwd_triton_poi_fused_mul_0", seed)
        self.assertIn("bwd_triton_poi_fused_mul_0", seed)
        self.assertNotIn("\ntriton_poi_fused_mul_0 = async_compile", seed)

    def test_emits_pair_contract_and_evolve_block(self):
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        self.assertIn("# EVOLVE-BLOCK-START", seed)
        self.assertIn("# EVOLVE-BLOCK-END", seed)
        self.assertIn("def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):", seed)
        self.assertIn("def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):", seed)
        # The wrapper and the launch sequence must be evolvable, so the block
        # closes after them, not before.
        self.assertLess(seed.index("def _fwd_call("), seed.index("# EVOLVE-BLOCK-END"))
        self.assertLess(
            seed.index("def layernorm_backward_from_saved"),
            seed.index("# EVOLVE-BLOCK-END"),
        )

    def test_one_kernel_set_and_a_dtype_guard(self):
        seed = generate_inductor_seed(self.op, "bfloat16", capture_for(self.op))
        self.assertEqual(seed.count("= async_compile"), 2)
        self.assertEqual(seed.count("def _fwd_call("), 1)
        self.assertEqual(seed.count("def _bwd_call("), 1)
        # A specialist must say so rather than silently mis-compute.
        self.assertIn("_SEED_DTYPE = torch.bfloat16", seed)
        self.assertIn("_check_dtype(x)", seed)

    def test_backward_arg_spec_is_carried_into_the_seed(self):
        # Symbolic sizes come first in the backward but last out of the
        # forward, so the permutation has to reach the generated file.
        spec = (("s", 1), ("s", 0), ("t", 0))
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op, spec=spec))
        self.assertIn(str(spec), seed)
        self.assertIn("_build_backward_args(_BWD_ARG_SPEC, saved_tensors,", seed)

    def test_saved_values_are_not_forced_contiguous(self):
        # Saved transposed views carry the strides the generated code asserts.
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        body = seed[seed.index("def _build_backward_args") : seed.index("def layernorm_forward")]
        self.assertNotIn("saved[i].contiguous()", body)

    def test_deployment_layer_follows_the_block(self):
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        end = seed.index("# EVOLVE-BLOCK-END")
        self.assertGreater(seed.index("ARTIFACT_CONTRACT = "), end)
        self.assertGreater(seed.index("class LayernormFunction(torch.autograd.Function)"), end)
        self.assertIn("'allowed_primitives': []", seed)

    def test_call_wrappers_take_named_parameters(self):
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        self.assertIn("def _fwd_call(primals_1, primals_2):", seed)
        self.assertIn("def _bwd_call(primals_1, primals_2, tangents_1):", seed)
        self.assertNotIn("def _fwd_call(args)", seed)
        self.assertNotIn("= args", seed)
        self.assertNotIn("args.clear()", seed)
        # Call sites pass positionally rather than packing a list, and inputs
        # keep their strides: the generated code asserts the captured ones.
        self.assertIn("_fwd_call(x, weight, bias)", seed)
        self.assertNotIn(".contiguous(), weight", seed)
        self.assertIn("*_build_backward_args(", seed)

    def test_gradient_selection_follows_the_declared_contract(self):
        op = get_task("evoattention")
        # res_mask is tensor arg 3 and gets a None slot; the contract skips it.
        seed = generate_inductor_seed(
            op, "float16", capture_for(op, grad_indices=(0, 1, 2, 4))
        )
        self.assertIn("return (_grads[0], _grads[1], _grads[2], _grads[4],)", seed)


class TestTritonInlining(unittest.TestCase):
    """On CUDA the kernels must land as real code, not quoted blobs."""

    def setUp(self):
        self.op = get_task("layernorm")
        self.seed = generate_inductor_seed(
            self.op, "float32", capture_for(self.op, triton=True)
        )

    def test_kernels_are_module_level_code(self):
        self.assertIn("@triton.jit", self.seed)
        self.assertIn("def fwd_triton_poi_fused_mul_0(in_ptr0, out_ptr0", self.seed)
        self.assertIn("def bwd_triton_poi_fused_mul_0(in_ptr0, out_ptr0", self.seed)
        self.assertIn("tl.store(out_ptr0 + xindex, tmp0 * tmp0, xmask)", self.seed)

    def test_no_quoted_kernel_bodies_remain(self):
        self.assertNotIn("async_compile", self.seed)
        self.assertNotIn("AsyncCompile", self.seed)

    def test_launch_sites_still_resolve(self):
        # The decorator binds the prefixed name, so .run() must match the def.
        self.assertIn("fwd_triton_poi_fused_mul_0.run(primals_1, buf0", self.seed)
        self.assertIn("bwd_triton_poi_fused_mul_0.run(tangents_1, buf0", self.seed)

    def test_decorator_metadata_stays_consistent_with_the_new_name(self):
        self.assertIn("'kernel_name': 'fwd_triton_poi_fused_mul_0'", self.seed)

    def test_seed_parses_as_python(self):
        ast.parse(self.seed)

    def test_no_autotune_pins_one_config_per_kernel(self):
        seed = generate_inductor_seed(
            self.op, "float32", capture_for(self.op, triton=True), autotune=False
        )
        ast.parse(seed)
        # size_hints exists to generate candidates; pinning removes the sweep.
        self.assertNotIn("size_hints", seed)
        self.assertNotIn("@triton_heuristics.pointwise", seed)
        self.assertEqual(seed.count("@triton_heuristics.fixed_config("), 2)
        # Forward and backward keep their own winning config.
        self.assertIn("config={'XBLOCK': 1024, 'num_warps': 4, 'num_stages': 1}", seed)
        self.assertIn("config={'XBLOCK': 512, 'num_warps': 8, 'num_stages': 2}", seed)

    def test_no_autotune_keeps_compile_and_launch_metadata(self):
        # triton_meta is what Triton needs to compile; inductor_meta carries
        # grid_type, without which there is no launch grid. Neither is tuning.
        seed = generate_inductor_seed(
            self.op, "float32", capture_for(self.op, triton=True), autotune=False
        )
        self.assertEqual(seed.count("triton_meta="), 2)
        self.assertEqual(seed.count("inductor_meta="), 2)
        self.assertEqual(seed.count("filename=__file__"), 2)

    def test_pinning_without_a_recorded_config_is_an_error(self):
        blank = replace(capture_for(self.op, triton=True), kernel_configs={})
        with self.assertRaisesRegex(RuntimeError, "no autotuner config recorded"):
            generate_inductor_seed(self.op, "float32", blank, autotune=False)

    def test_cpp_capture_keeps_the_async_compile_scaffold(self):
        # C++ bodies (CPU capture) cannot be inlined as Python.
        seed = generate_inductor_seed(self.op, "float32", capture_for(self.op))
        self.assertIn("async_compile.wait(globals())", seed)
        ast.parse(seed)


# A scoped case as Inductor emits it on CUDA: a cuBLAS extern call, the private
# device guard, two outputs and two upstream gradients.
GEMM_FWD_BODY = (
    "        primals_1, primals_2, primals_3 = args\n"
    "        args.clear()\n"
    "        assert_size_stride(primals_1, (4, 4), (1, 4))\n"
    "        with torch.cuda._DeviceGuard(0):\n"
    "            torch.cuda.set_device(0)\n"
    "            buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)\n"
    "            extern_kernels.mm(primals_1, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf0)\n"
    "            buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)\n"
    "            triton_poi_fused_mul_0.run(buf0, buf1, 16)\n"
    "        return (buf1, buf0, primals_1, primals_3, )"
)
GEMM_BWD_BODY = (
    "        primals_1, primals_3, tangents_1, tangents_2 = args\n"
    "        args.clear()\n"
    "        with torch.cuda._DeviceGuard(0):\n"
    "            torch.cuda.set_device(0)\n"
    "            buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)\n"
    "            triton_poi_fused_mul_0.run(tangents_1, tangents_2, buf0, 16)\n"
    "            buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)\n"
    "            extern_kernels.mm(buf0, primals_3, out=buf1)\n"
    "        return (buf1, buf0, buf1, )"
)


class TestScopedCaseSeed(unittest.TestCase):
    """A multi-output case with a library GEMM, granted vendor_gemm."""

    def setUp(self):
        self.op = get_task("fused_add_rms_norm")
        self.captured = replace(
            capture_for(self.op, triton=True),
            forward_source=fake_module(_TRITON_TEMPLATE, "triton_poi_fused_mul_0", GEMM_FWD_BODY),
            backward_source=fake_module(_TRITON_TEMPLATE, "triton_poi_fused_mul_0", GEMM_BWD_BODY),
            backward_arg_spec=(("s", 0), ("s", 1), ("t", 0), ("t", 1)),
            dynamic=False,
        )
        self.seed = generate_inductor_seed(
            self.op, "float32", self.captured, allowed_primitives=("vendor_gemm",)
        )

    def test_forward_returns_every_output_and_backward_takes_one_gradient_each(self):
        self.assertIn("    return _out[:2], tuple(_out[2:])", self.seed)
        self.assertIn(
            "def fused_add_rms_norm_backward_from_saved(output_grads, saved_tensors, eps=1e-6):",
            self.seed,
        )
        self.assertIn("_build_backward_args(_BWD_ARG_SPEC, saved_tensors, output_grads)", self.seed)

    def test_extern_gemm_runs_through_the_granted_primitive(self):
        self.assertIn("buf0 = vendor_gemm(primals_1, reinterpret_tensor(primals_3", self.seed)
        self.assertIn("buf1 = vendor_gemm(buf0, primals_3)", self.seed)
        # The buffers planned for the out= argument are gone with it; the
        # Triton kernels' output buffers stay.
        self.assertEqual(self.seed.count("= empty_strided_cuda((4, 4), (4, 1), torch.float32)"), 2)
        self.assertEqual(check_source(self.seed, allowed=("vendor_gemm",))["primitive_call_sites"],
                         {"vendor_gemm": 2})
        with self.assertRaises(PrimitiveViolation):
            check_source(self.seed, allowed=())

    def test_inductor_runtime_boilerplate_and_library_handles_are_gone(self):
        for text in ("_DeviceGuard", "empty_strided_p2p", "_distributed_c10d", "extern_kernels",
                     "aten = ", "inductor_ops", "alloc_from_pool"):
            self.assertNotIn(text, self.seed)
        self.assertIn("with torch.cuda.device(0):", self.seed)

    def test_deployment_layer_records_the_grant_after_the_block(self):
        end = self.seed.index("# EVOLVE-BLOCK-END")
        self.assertGreater(self.seed.index("def vendor_gemm(a, b):"), end)
        self.assertGreater(self.seed.index("ARTIFACT_CONTRACT = "), end)
        self.assertIn("'allowed_primitives': ['vendor_gemm']", self.seed)
        self.assertEqual(find_forwarding_aliases(self.seed), [])
        self.assertTrue(evolvable_region_covers_pair(self.op, self.seed))

    def test_a_library_call_without_its_grant_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "vendor_gemm grant"):
            generate_inductor_seed(self.op, "float32", self.captured)


class TestLaunchCodeLegalization(unittest.TestCase):
    def test_reused_buffer_is_released_instead_of_written(self):
        body = [
            "        buf2 = reinterpret_tensor(buf0, (8, 4), (4, 1), 0); del buf0  # reuse",
            "        # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]",
            "        extern_kernels.mm(a, b, out=buf2)",
        ]
        self.assertEqual(
            _legalize_launch_code(body, ("vendor_gemm",), "backward"),
            [
                "        del buf0",
                "        # Topologically Sorted Source Nodes: [mm], Original ATen: [aten.mm]",
                "        buf2 = vendor_gemm(a, b)",
            ],
        )

    def test_gemm_into_a_live_buffer_is_refused(self):
        body = ["        triton_poi_0.run(x, buf0, 4)", "        extern_kernels.mm(a, b, out=buf0)"]
        with self.assertRaisesRegex(RuntimeError, "aliases a live buffer"):
            _legalize_launch_code(body, ("vendor_gemm",), "forward")

    def test_cudnn_attention_runs_through_vendor_sdpa_with_the_save_set_kept(self):
        tag = "'torch.ops.aten._scaled_dot_product_cudnn_attention.default'"
        forward = [
            "        buf0 = torch.ops.aten._scaled_dot_product_cudnn_attention.default("
            "q, k, v, None, True, 0.0, True, scale=0.125)",
            "        buf1 = buf0[0]",
            f"        assert_size_stride(buf1, (1, 2, 4, 8), (64, 8, 16, 1), {tag})",
            "        buf2 = buf0[1]",
            f"        assert_size_stride(buf2, (1, 2, 4, 1), (8, 4, 1, 1), {tag})",
            f"        assert_alignment(buf2, 16, {tag})",
            "        buf3 = buf0[6]",
            f"        assert_size_stride(buf3, (), (), {tag})",
            "        del buf0",
        ]
        # The output keeps its asserted layout. The logsumexp is the backend's
        # (flash under deterministic algorithms lays it out differently from
        # cuDNN), and the rng state means nothing without dropout.
        self.assertEqual(
            _legalize_launch_code(forward, ("vendor_sdpa",), "forward"),
            [
                "        buf0 = vendor_sdpa_forward(q, k, v, is_causal=True, scale=0.125)",
                "        buf1 = buf0[0]",
                f"        assert_size_stride(buf1, (1, 2, 4, 8), (64, 8, 16, 1), {tag})",
                "        buf2 = buf0[1]",
                "        buf3 = None  # backend rng/sequence state; unused without dropout",
                "        del buf0",
            ],
        )
        backward = [
            "        assert_size_stride(getitem_1, (1, 2, 4, 1), (8, 4, 1, 1))",
            "        assert_size_stride(getitem_6, (), ())",
            "        assert_size_stride(dout, (1, 2, 4, 8), (64, 32, 8, 1))",
            "        buf3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default("
            "dout, q, k, v, out, getitem_1, getitem_6, getitem_7, None, None, None, 4, 4, 0.0, "
            "True, scale=0.125)",
        ]
        self.assertEqual(
            _legalize_launch_code(backward, ("vendor_sdpa",), "backward"),
            [
                "        assert_size_stride(dout, (1, 2, 4, 8), (64, 32, 8, 1))",
                "        buf3 = vendor_sdpa_backward(dout, q, k, v, out, getitem_1, "
                "is_causal=True, scale=0.125)",
            ],
        )

    def test_attention_needs_its_grant_and_no_dropout(self):
        call = ("        buf0 = torch.ops.aten._scaled_dot_product_flash_attention.default("
                "q, k, v, 0.1, True, scale=0.125)")
        with self.assertRaisesRegex(RuntimeError, "vendor_sdpa grant"):
            _legalize_launch_code([call], ("vendor_gemm",), "forward")
        with self.assertRaisesRegex(RuntimeError, "dropout"):
            _legalize_launch_code([call], ("vendor_sdpa",), "forward")

    def test_any_other_library_call_is_an_error(self):
        for line in (
            "        extern_kernels.bmm(a, b, out=buf0)",
            "        buf0 = torch.ops.aten._scaled_dot_product_efficient_attention.default(q, k, v, None, True)",
            "        buf0 = aten.mm.default(a, b)",
        ):
            with self.subTest(line=line), self.assertRaisesRegex(RuntimeError, "no granted primitive"):
                _legalize_launch_code([line], ("vendor_gemm", "vendor_sdpa"), "forward")


class TestSplitTopLevel(unittest.TestCase):
    def test_ignores_commas_inside_nesting_and_quotes(self):
        parts = _split_top_level("a={'x': 1, 'y': 2}, b=[3, 4], c='s, t', d=5")
        self.assertEqual([p.split("=")[0].strip() for p in parts], ["a", "b", "c", "d"])


class TestNamedParameters(unittest.TestCase):
    def test_single_argument_unpack(self):
        names, rest = _named_parameters(
            ["        primals_1, = args", "        args.clear()", "        return ()"],
            "forward",
        )
        self.assertEqual(names, ["primals_1"])
        self.assertEqual(rest, ["        return ()"])

    def test_rejects_a_body_that_never_unpacks_args(self):
        with self.assertRaisesRegex(RuntimeError, "does not unpack"):
            _named_parameters(["        return ()"], "forward")


class TestCaptureWorkloadSelection(unittest.TestCase):
    def test_detects_duplicate_dims(self):
        self.assertTrue(_dims_are_distinct(Workload(dims=dict(M=4, N=8), dtype="float32")))
        self.assertFalse(_dims_are_distinct(Workload(dims=dict(M=8, N=8), dtype="float32")))
        # Size-1 dims specialize under symbolic tracing.
        self.assertFalse(_dims_are_distinct(Workload(dims=dict(M=1, N=8), dtype="float32")))

    def test_perturbs_duplicates_minimally(self):
        out = _with_distinct_dims(Workload(dims=dict(M=8, N=8, K=8), dtype="float32"))
        self.assertEqual(list(out.dims.values()), [8, 9, 10])

    def test_prefers_a_declared_workload_with_distinct_dims(self):
        op = get_task("matmul")
        selected = select_capture_workloads(op, ("float32", "float16"))
        for dtype, workload in selected.items():
            values = list(workload.dims.values())
            self.assertEqual(len(set(values)), len(values), f"{dtype}: {workload.dims}")

    def test_static_capture_takes_the_declared_case_exactly(self):
        # Equal dims are part of a static case (attention's QO = HQ * D).
        case = Workload(dims=dict(M=128, K=128, N=256), dtype="float16")
        op = replace(get_task("matmul"), correctness=(case,))
        static = select_capture_workloads(op, ("float16",), dynamic=False)["float16"]
        self.assertEqual(static, case)
        dynamic = select_capture_workloads(op, ("float16",))["float16"]
        self.assertEqual(dict(dynamic.dims), {"M": 128, "K": 129, "N": 256})

    def test_rejects_undeclared_dtype(self):
        with self.assertRaisesRegex(ValueError, "no declared workload"):
            select_capture_workloads(get_task("rmsnorm"), ("float64",))


class TestDtypeTag(unittest.TestCase):
    def test_known_and_unknown_dtypes(self):
        self.assertEqual(dtype_tag("bfloat16"), "bf16")
        self.assertEqual(dtype_tag("float8_e4m3fn"), "float8_e4m3fn")


if __name__ == "__main__":
    unittest.main()
