"""Tier-2 timing and declared correctness: actual driver, isolation, input identity."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from evograd.evaluation.tier2.runner import tensor_checksum


class TestTimingDriver(unittest.TestCase):
    def test_measurement_uses_the_duration_budget_for_both_regions(self):
        from evograd.evaluation.tier2.runner import measure_module

        class Operator(torch.nn.Module):
            adapter_kind = "test"

            def activations(self, values):
                return [values["x"]]

            def forward(self, x):
                return x.square()

        op = SimpleNamespace(is_multi_output=False, output_names=("out",),
                             upstream_grad_names=("dout",))
        values = {"x": torch.arange(4, dtype=torch.float32), "dout": torch.ones(4)}

        def measure(fn, **kwargs):
            fn()  # Exercise the real forward/backward callable on CPU.
            return (1.0, 0.8, 1.2)

        bench = mock.Mock(side_effect=measure)
        with mock.patch.dict("sys.modules", {"triton.testing": SimpleNamespace(do_bench=bench)}), \
                mock.patch("torch.cuda.synchronize"), \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=1234):
            result = measure_module(op, Operator(), values, warmup_ms=13, rep_ms=17)

        self.assertEqual(bench.call_count, 2)
        for call in bench.call_args_list:
            self.assertEqual(call.kwargs["warmup"], 13)
            self.assertEqual(call.kwargs["rep"], 17)
            self.assertEqual(call.kwargs["quantiles"], [0.5, 0.2, 0.8])
        self.assertEqual(result["full_step"],
                         {"median_ms": 1.0, "q20_ms": 0.8, "q80_ms": 1.2})
        self.assertEqual(result["peak_memory_bytes"], 1234)


class TestDeclaredNumerics(unittest.TestCase):
    def test_retired_profile_environment_cannot_change_tier1_or_tier2(self):
        from evograd.opdecl.verify import _check
        from evograd.evaluation.tier2.runner import _compare

        want = torch.full((16,), 32.0, dtype=torch.bfloat16)
        actual = want.clone()
        actual[0] = 32.25
        for mode in ("strict", "report_first"):
            with mock.patch.dict("os.environ", {
                "EVOGRAD_NUMERICAL_PROFILE": "dtype_default",
                "EVOGRAD_TIER3_ENFORCEMENT": mode,
            }):
                self.assertFalse(_check("out", actual, want, 1e-2, 1e-3).ok)
                result = _compare(actual, want, 1e-2, 1e-3)
                self.assertFalse(result["ok"])
                self.assertEqual((result["atol"], result["rtol"]), (1e-2, 1e-3))

    def test_legacy_declared_flag_cannot_select_the_retired_strategy(self):
        import contextlib
        import io
        from evograd.evaluation.tier2.cli import _parser

        args = _parser().parse_args(["--op", "rmsnorm", "--numerical-profile", "declared"])
        self.assertEqual(args.numerical_profile, "declared")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parser().parse_args(["--op", "rmsnorm", "--numerical-profile", "dtype_default"])


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

        from evograd.evaluation.tier2.runner import run_case

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
        from evograd.evaluation.tier2.runner import native_module
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.benchmark import get_task

        op = get_task(op_name)
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
        from evograd.evaluation.tier2.runner import native_module
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.benchmark import get_task

        op = get_task("qwen3_qkv_norm_rope")
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
