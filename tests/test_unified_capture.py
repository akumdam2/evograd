"""Capture ABI regressions independent of GPU compilation."""

import unittest
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from evograd.benchmark import get_task
from evograd.pipelines.d_inductor.capture import _backward_arg_spec, capture_inductor_pair
from evograd.pipelines.unified.capture import (_capture_device, _encode_arg, _same_call,
                                               _serialize_graph)
from tests._small_cases import SMALL


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


if __name__ == "__main__":
    unittest.main()
