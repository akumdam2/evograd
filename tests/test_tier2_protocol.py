"""The Tier-2 timing protocol: fixed samples, isolation, recorded identity."""

from __future__ import annotations

import unittest

import torch

from evograd.bench.tier2 import (
    REPETITIONS,
    WARMUP_ITERS,
    _summarize_samples,
    tensor_checksum,
)


class TestFixedSampleCount(unittest.TestCase):
    def test_the_protocol_declares_a_count_not_a_duration(self):
        # A wall-clock budget samples a fast provider far more often than a
        # slow one, so their quantiles rest on different amounts of evidence.
        self.assertEqual(REPETITIONS, 500)
        self.assertEqual(WARMUP_ITERS, 10)

    def test_quantiles_are_the_median_and_the_twenty_eighty_pair(self):
        summary = _summarize_samples([float(i) for i in range(101)])
        self.assertAlmostEqual(summary["median_ms"], 50.0)
        self.assertAlmostEqual(summary["q20_ms"], 20.0)
        self.assertAlmostEqual(summary["q80_ms"], 80.0)
        self.assertEqual(summary["samples"], 101)

    def test_the_sample_count_is_reported_so_it_can_be_checked(self):
        self.assertIn("samples", _summarize_samples([1.0, 2.0]))


class TestInputIdentity(unittest.TestCase):
    def test_the_same_tensor_hashes_the_same_and_a_changed_one_does_not(self):
        a = torch.zeros(4, 8)
        self.assertEqual(tensor_checksum(a), tensor_checksum(a.clone()))
        b = a.clone()
        b[0, 0] = 1.0
        self.assertNotEqual(tensor_checksum(a), tensor_checksum(b))

    def test_the_checksum_pins_shape_dtype_and_stride_not_only_content(self):
        base = torch.zeros(4, 8)
        checksum = tensor_checksum(base)
        self.assertIn("(4, 8)", checksum)
        self.assertIn("torch.float32", checksum)
        # Same bytes, different layout must not read as the same input.
        self.assertNotEqual(checksum, tensor_checksum(base.t().contiguous()))

    def test_a_scalar_is_recorded_rather_than_skipped(self):
        self.assertEqual(tensor_checksum(1e-6), "scalar:1e-06")


class TestProviderOrderAndSelection(unittest.TestCase):
    def test_run_case_accepts_a_seed_and_a_single_provider(self):
        import inspect

        from evograd.bench.tier2 import run_case

        params = inspect.signature(run_case).parameters
        # `only` is what lets the CLI put one provider in one process.
        self.assertIn("only", params)
        self.assertIn("order_seed", params)

    def test_the_order_is_shuffled_deterministically_from_the_seed(self):
        import random

        names = ["eager", "compile", "liger", "candidate"]
        first = list(names)
        random.Random(0).shuffle(first)
        again = list(names)
        random.Random(0).shuffle(again)
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()


class TestDeclaredArgumentOrder(unittest.TestCase):
    """The direct-deployment wrapper must call by declaration, not by category.

    ``qwen3_qkv_norm_rope`` declares ``cos`` and ``sin`` as inactive tensors
    sitting between the parameters and ``eps``. Concatenating activations,
    parameters and scalars silently omits them, and the resulting TypeError only
    appears once a candidate is actually run at that operator.
    """

    def _spy_module(self, op_name: str):
        from evograd.bench.tier2 import native_module
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.ops import get_op

        op = get_op(op_name)
        values = make_case_inputs(op, op.correctness[0], device="cpu")
        seen: dict[str, tuple] = {}

        def spy(*args):
            seen["args"] = args
            return tuple(torch.zeros(1) for _ in op.output_names)

        module = native_module(op, spy, values=values,
                               adapter_kind="spy_direct_module")
        module(*[values[n] for n in module._activation_names])
        return op, values, seen["args"]

    def test_every_declared_argument_reaches_the_deployment_entry(self):
        op, values, args = self._spy_module("qwen3_qkv_norm_rope")
        self.assertEqual(len(args), len(op.args))
        for arg, got in zip(op.args, args):
            if torch.is_tensor(values[arg.name]):
                self.assertTrue(
                    torch.equal(got, values[arg.name]),
                    f"{arg.name} did not arrive in its declared slot",
                )
            else:
                self.assertEqual(got, values[arg.name])

    def test_the_inactive_tensors_are_passed_not_dropped(self):
        _, values, args = self._spy_module("qwen3_qkv_norm_rope")
        self.assertTrue(any(torch.is_tensor(a) and torch.equal(a, values["cos"])
                            for a in args))
        self.assertTrue(any(torch.is_tensor(a) and torch.equal(a, values["sin"])
                            for a in args))

    def test_parameters_arrive_from_the_module_not_the_input_dict(self):
        from evograd.bench.tier2 import native_module
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.ops import get_op

        op = get_op("qwen3_qkv_norm_rope")
        values = make_case_inputs(op, op.correctness[0], device="cpu")
        seen: dict[str, tuple] = {}

        def spy(*args):
            seen["args"] = args
            return tuple(torch.zeros(1) for _ in op.output_names)

        module = native_module(op, spy, values=values,
                               adapter_kind="spy_direct_module")
        slot = [a.name for a in op.args].index("q_weight")
        module(*[values[n] for n in module._activation_names])
        self.assertIs(seen["args"][slot], module.q_weight)

    def test_an_operator_without_inactive_tensors_is_unchanged(self):
        op, values, args = self._spy_module("fused_add_rms_norm")
        self.assertEqual(len(args), len(op.args))
        for arg, got in zip(op.args, args):
            if torch.is_tensor(values[arg.name]):
                self.assertTrue(torch.equal(got, values[arg.name]))
