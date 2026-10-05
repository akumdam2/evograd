"""Capture ABI regressions independent of GPU compilation."""

import unittest
from unittest.mock import patch

import torch

from evograd.benchmark import get_task
from evograd.pipelines.d_inductor.capture import _backward_arg_spec, capture_inductor_pair
from evograd.pipelines.unified.capture import _capture_device, _encode_arg


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
            capture_inductor_pair(op, op.correctness[0], device="cpu")


if __name__ == "__main__":
    unittest.main()
