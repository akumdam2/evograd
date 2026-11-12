"""Capture ABI regressions independent of GPU compilation."""

import shutil
import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from evograd.benchmark import get_task
from evograd.opdecl import Active, Workload, declare_op
from evograd.pipelines.d_inductor.capture import _backward_arg_spec, capture_inductor_pair
from evograd.pipelines.unified.capture import (_capture_device, _encode_arg, _external_regions,
                                               _same_call, _serialize_graph)
from tests._small_cases import SMALL


def saved_sum_ref(x):
    """Saves a [R, C] float32 reduction whose rows Inductor pads when C > 1024."""
    return x.sum(-1).square().sum(-1)


class TestUnifiedCaptureContract(unittest.TestCase):
    def test_cuda_capture_pins_an_unspecified_device_index(self):
        with patch("torch.cuda.current_device", return_value=3) as current:
            self.assertEqual(_capture_device("cuda"), "cuda:3")
            current.assert_called_once_with()
        with patch("torch.cuda.current_device") as current:
            self.assertEqual(_capture_device("cuda:1"), "cuda:1")
            self.assertEqual(_capture_device("cpu"), "cpu")
            current.assert_not_called()

    def test_unified_constant_encoding_matches_the_shared_atenir_serializer(self):
        from evograd.atenir.extract import _serialize_arg

        for value in (torch.strided, torch.contiguous_format, torch.float32, torch.device("cuda:0")):
            with self.subTest(value=value):
                self.assertEqual(_encode_arg(value), _serialize_arg(value))

    def test_multi_output_tangents_follow_declared_output_order(self):
        op = get_task("fused_add_rms_norm")
        self.assertEqual(
            _backward_arg_spec(op, ["saved_x", "saved_weight"],
                               ["saved_weight", "saved_x", "tangents_1", "tangents_2"],
                               num_tangents=2),
            (("s", 1), ("s", 0), ("t", 0), ("t", 1)),
        )
        with self.assertRaisesRegex(RuntimeError, "expected 2"):
            _backward_arg_spec(op, [], ["tangents_1"], num_tangents=2)

    def test_legacy_pipeline_d_still_refuses_multi_output_without_opt_in(self):
        op = get_task("fused_add_rms_norm")
        with self.assertRaisesRegex(NotImplementedError, "Pipeline D"):
            capture_inductor_pair(op, SMALL["fused_add_rms_norm"][0], device="cpu")


class TestCompilerRewrites(unittest.TestCase):
    """A preserved call still maps back after post-grad passes rewrote its inputs."""

    def _call(self, fn, *inputs):
        gm = make_fx(fn)(*inputs)
        graph = _serialize_graph(gm)
        by_name = {n["name"]: n for n in graph["nodes"] if "name" in n}
        mm = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.mm.default)
        return gm, by_name, mm

    def test_a_dropped_alias_names_the_same_value(self):
        # Qwen3.5's attention backward: remove_noop_ops drops alias(alias(out)).
        gm, by_name, mm = self._call(lambda x, w: torch.mm(torch.ops.aten.alias(x), w),
                                     torch.randn(2, 3), torch.randn(3, 4))
        alias = next(n for n in gm.graph.nodes if n.target is torch.ops.aten.alias.default)
        alias.replace_all_uses_with(alias.args[0])
        gm.graph.erase_node(alias)
        self.assertTrue(_same_call(by_name[mm.name], mm, by_name))

    def test_a_recreated_node_is_the_node_it_repeats_and_nothing_else(self):
        # A causal conv1d's backward: the compiler re-creates the slice_scatter
        # its convolution_backward reads, same target and inputs, new name.
        gm, by_name, mm = self._call(lambda x, w: torch.mm(x * 2, w),
                                     torch.randn(2, 3), torch.randn(3, 4))
        mul, x = mm.args[0], mm.args[0].args[0]
        with gm.graph.inserting_before(mm):
            repeat = gm.graph.call_function(mul.target, mul.args, mul.kwargs)
            other = gm.graph.call_function(mul.target, (x, 3), {})
        mm.replace_input_with(mul, repeat)
        self.assertTrue(_same_call(by_name[mm.name], mm, by_name))
        mm.replace_input_with(repeat, other)
        self.assertFalse(_same_call(by_name[mm.name], mm, by_name))

    def test_a_recreated_random_call_is_a_new_value(self):
        # Two draws of the same random op on the same inputs differ.
        gm, by_name, mm = self._call(lambda x, w: torch.mm(torch.rand_like(x), w),
                                     torch.randn(2, 3), torch.randn(3, 4))
        rand = mm.args[0]
        with gm.graph.inserting_before(mm):
            redraw = gm.graph.call_function(rand.target, rand.args, rand.kwargs)
        mm.replace_input_with(rand, redraw)
        self.assertFalse(_same_call(by_name[mm.name], mm, by_name))

    def test_memory_efficient_attention_is_preserved_for_float32_only(self):
        # vendor_sdpa runs it for float32; half precision there is cuDNN's or flash's.
        from types import SimpleNamespace

        from torch._inductor import ir

        kernel = "torch.ops.aten._scaled_dot_product_efficient_attention.default"
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(1, 2, 8, 16, dtype=dtype)
            gm = make_fx(lambda q, k, v: torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, None, True), tracing_mode="fake")(q, q, q)
            buffer = object.__new__(ir.ExternKernel)  # the compiler's call, as lowering leaves it
            for name, value in (("python_kernel_name", kernel), ("name", "buf0"),
                                ("fx_node", next(n for n in gm.graph.nodes if str(n.target) in kernel))):
                object.__setattr__(buffer, name, value)
            lowering = SimpleNamespace(buffers=[buffer], operations=[])
            with self.subTest(dtype=dtype):
                if dtype == torch.float32:
                    (region,) = _external_regions("forward", lowering, _serialize_graph(gm))
                    self.assertEqual(region.kernel, kernel)
                else:
                    with self.assertRaisesRegex(NotImplementedError, "float32 operands only"):
                        _external_regions("forward", lowering, _serialize_graph(gm))


class TestSavedActivationLayout(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "Inductor's CPU backend needs a C++ compiler")
    def test_the_backward_takes_a_saved_activation_the_forward_padded(self):
        # Qwen3.5's gated delta rule: Inductor padded a saved float32 [.., 33, 33]
        # buffer to a 1120-element stride, and the backward, compiled for 1089,
        # rejected it at run time unless the capture hands it the real strides.
        import torch._inductor.config as inductor_config

        op = declare_op(
            name="padded_saved_sum", forward="tests.test_unified_capture:saved_sum_ref",
            dims=("R", "C", "P"), args=(Active("x", "[R, C, P]"),), output=Active("y", "[R]"),
            forward_semantics="Sum, square, sum.", backward_semantics="Its gradient.",
            benchmark=(Workload(dims={"R": 4, "C": 1089, "P": 2}, dtype="float32"),),
        )
        with inductor_config.patch({"disable_padding_cpu": False, "cpp.cxx": (None, "g++")}):
            pair = capture_inductor_pair(op, op.benchmark[0], device="cpu", dynamic=False)
        self.assertIn("1120", pair.forward_source)
        self.assertEqual([s.shape for s in pair.saved if not s.is_input], [(4, 1089)])


if __name__ == "__main__":
    unittest.main()
