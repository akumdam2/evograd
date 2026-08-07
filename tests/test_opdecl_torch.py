"""CPU tests for the torch-facing opdecl stack (oracle, bind, verify).

Self-skips when torch is unavailable (dev boxes); runs on any node with torch
installed — no GPU or triton needed, everything here is pure torch on CPU.
Kernel-level validation still requires scripts/gpu_smoke.py on a CUDA node.
"""

import types
import unittest

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from evograd.opdecl import Active, Inactive, Workload, declare_op


def toy_op():
    """y = (x * scale_vec) @ w, with an inactive additive mask and scalar eps."""
    return declare_op(
        name="toy",
        forward="tests.test_opdecl_torch:toy_forward_ref",
        dims=("M", "N"),
        args=(
            Active("x", "[M, N]"),
            Active("w", "[N, N]"),
            Inactive("mask", "[M, N]"),
            Inactive("eps", default=1e-5),
        ),
        output=Active("y", "[M, N]"),
        forward_semantics="toy",
        backward_semantics="toy",
        correctness=(Workload(dims={"M": 4, "N": 8}, dtype="float32"),),
        tolerances={"float32": (1e-5, 1e-5)},
    )


def toy_forward_ref(x, w, mask, eps=1e-5):
    return (x + mask) @ w + eps


def _toy_seed_module(correct=True, forward_offset=0.0, save_scalar=False):
    """A pure-torch seed implementing the autograd-pair contract by hand."""

    def forward_with_saved(x, w, mask, eps=1e-5):
        y = (x + mask) @ w + eps + forward_offset
        saved = (x, w, mask, x.shape[0]) if save_scalar else (x, w, mask)
        return y, saved

    def backward_from_saved(dy, saved_tensors, eps=1e-5):
        x, w, mask = saved_tensors[:3]
        dx = dy @ w.T
        dw = (x + mask).T @ dy
        if not correct:
            dw = dw * 2.0
        return dx, dw

    module = types.ModuleType("toy_seed")
    module.forward_with_saved = forward_with_saved
    module.backward_from_saved = backward_from_saved
    return module


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestLazyExports(unittest.TestCase):
    def test_submodule_import_does_not_shadow_lazy_exports(self):
        # oracle/bind/verify name both a submodule and its main function; the
        # import system pins the submodule onto the package after these
        # imports, which must not shadow the function exports.
        import evograd.opdecl.bind  # noqa: F401
        import evograd.opdecl.oracle  # noqa: F401
        import evograd.opdecl.verify  # noqa: F401

        from evograd.opdecl import bind, oracle, verify

        self.assertTrue(callable(oracle))
        self.assertTrue(callable(bind))
        self.assertTrue(callable(verify))


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestAtenIRExtraction(unittest.TestCase):
    def test_string_kwargs_survive_classification(self):
        # aten.gelu's approximate="tanh" arrives as a str kwarg; dropping it
        # silently swapped the tanh formulation for erf (~1e-3 apart).
        from evograd.atenir.extract import _classify_args

        node = types.SimpleNamespace(
            target="aten.gelu.default", args=(), kwargs={"approximate": "tanh"}
        )
        _, _, _, args_ordered = _classify_args(node, {})
        self.assertIn({"kind": "scalar", "value": "tanh"}, args_ordered)

    def test_gelu_dispatch_honors_tanh_approximate(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton not installed")
        from evograd.atenir.primitive_triton import dispatch, elementwise

        base = {"target": "aten.gelu.default", "name": "gelu"}
        tanh_node = {
            **base,
            "args_ordered": [
                {"kind": "node", "name": "x"},
                {"kind": "scalar", "value": "tanh"},
            ],
        }
        erf_node = {**base, "args_ordered": [{"kind": "node", "name": "x"}]}
        self.assertIs(dispatch.make_kernel(tanh_node), elementwise.gelu_tanh)
        self.assertIs(dispatch.make_kernel(erf_node), elementwise.gelu)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestOracle(unittest.TestCase):
    def test_oracle_matches_closed_form(self):
        from evograd.opdecl import make_case_inputs, oracle

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        y, grads = oracle(op, inputs)

        x, w, mask, dy = inputs["x"], inputs["w"], inputs["mask"], inputs["dy"]
        torch.testing.assert_close(y, (x + mask) @ w + 1e-5)
        self.assertEqual(list(grads), ["dx", "dw"])  # Active only, mask excluded
        torch.testing.assert_close(grads["dx"], dy @ w.T)
        torch.testing.assert_close(grads["dw"], (x + mask).T @ dy)

    def test_oracle_never_grads_inactive(self):
        from evograd.opdecl import make_case_inputs, oracle

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        _, grads = oracle(op, inputs)
        self.assertNotIn("dmask", grads)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestBind(unittest.TestCase):
    def test_bound_function_matches_plain_autograd(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module())

        x = inputs["x"].detach().clone().requires_grad_(True)
        w = inputs["w"].detach().clone().requires_grad_(True)
        y = fn(x, w, inputs["mask"])
        y.backward(inputs["dy"])

        x2 = inputs["x"].detach().clone().requires_grad_(True)
        w2 = inputs["w"].detach().clone().requires_grad_(True)
        toy_forward_ref(x2, w2, inputs["mask"]).backward(inputs["dy"])

        torch.testing.assert_close(x.grad, x2.grad)
        torch.testing.assert_close(w.grad, w2.grad)

    def test_inactive_slot_gets_none(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module())

        mask = inputs["mask"].detach().clone().requires_grad_(True)
        y = fn(inputs["x"], inputs["w"], mask)
        y.backward(inputs["dy"])
        self.assertIsNone(mask.grad)  # Inactive: harness emitted None for its slot

    def test_plain_scalar_saved_state_round_trips(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module(save_scalar=True))
        x = inputs["x"].detach().clone().requires_grad_(True)
        w = inputs["w"].detach().clone().requires_grad_(True)
        fn(x, w, inputs["mask"]).backward(inputs["dy"])
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestVerify(unittest.TestCase):
    def test_correct_seed_passes(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(), device="cpu")
        self.assertTrue(report.ok, report.to_dict())

    def test_wrong_gradient_fails_named_check(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(correct=False), device="cpu")
        self.assertFalse(report.ok)
        failing = [c.name for case in report.cases for c in case.checks if not c.ok]
        self.assertEqual(failing, ["dw"])  # exactly the sabotaged gradient

    def test_wrong_forward_fails_even_with_correct_backward(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(forward_offset=1.0), device="cpu")
        self.assertFalse(report.ok)
        failing = [c.name for case in report.cases for c in case.checks if not c.ok]
        self.assertEqual(failing, ["y"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestBaselineSchema(unittest.TestCase):
    def test_liger_does_not_claim_pytorch_metrics(self):
        from evograd.bench.harness import _add_pytorch_autograd_aliases

        report = {
            "baseline_backward_ms": 2.0,
            "baseline_full_step_ms": 3.0,
            "speedup_vs_baseline_backward": 1.5,
            "speedup_vs_baseline_full_step": 1.25,
        }
        result = _add_pytorch_autograd_aliases(report, "liger")
        self.assertNotIn("pytorch_autograd_backward_ms", result)
        self.assertNotIn("speedup_vs_pytorch_autograd_backward", result)

    def test_pytorch_baseline_keeps_legacy_aliases(self):
        from evograd.bench.harness import _add_pytorch_autograd_aliases

        report = {
            "baseline_backward_ms": 2.0,
            "baseline_full_step_ms": 3.0,
            "speedup_vs_baseline_backward": 1.5,
            "speedup_vs_baseline_full_step": 1.25,
        }
        result = _add_pytorch_autograd_aliases(report, "pytorch_autograd")
        self.assertEqual(result["pytorch_autograd_backward_ms"], 2.0)
        self.assertEqual(result["pytorch_autograd_full_step_ms"], 3.0)
        self.assertEqual(result["speedup_vs_pytorch_autograd_backward"], 1.5)
        self.assertEqual(result["speedup_vs_pytorch_autograd_full_step"], 1.25)

    def test_evaluator_metrics_use_selected_baseline_fields(self):
        from evograd.evolve.evaluator import _baseline_metrics

        aggregate = {
            "baseline_backward_ms": 2.0,
            "baseline_full_step_ms": 3.0,
            "speedup_vs_baseline_backward": 1.5,
            "speedup_vs_baseline_full_step": 1.25,
            # Deliberately inconsistent legacy values: these must be ignored.
            "pytorch_autograd_backward_ms": 20.0,
            "pytorch_autograd_full_step_ms": 30.0,
            "speedup_vs_pytorch_autograd_backward": 15.0,
            "speedup_vs_pytorch_autograd_full_step": 12.5,
        }
        self.assertEqual(
            _baseline_metrics(aggregate),
            {
                "speedup": 1.5,
                "full_step_speedup": 1.25,
                "baseline_latency_ms": 2.0,
                "baseline_full_step_ms": 3.0,
            },
        )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestFairBenchmarkPrimitives(unittest.TestCase):
    def test_final_protocol_has_stable_generic_version(self):
        from evograd.bench.fair import PROTOCOL_VERSION

        self.assertEqual(PROTOCOL_VERSION, "evograd-final-runtime-v1")

    def test_pytorch_autograd_provider_matches_oracle(self):
        from evograd.bench.fair import pytorch_autograd_provider
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.opdecl.oracle import oracle
        from evograd.ops import get_op

        op = get_op("matmul")
        workload = op.correctness[0]
        values = make_case_inputs(op, workload, device="cpu")
        expected_output, expected_grads = oracle(
            op,
            make_case_inputs(op, workload, device="cpu"),
        )
        provider = pytorch_autograd_provider(op)
        output, saved = provider.forward(values)
        actual_grads = provider.backward(
            values[op.upstream_grad_name],
            saved,
            values,
        )
        self.assertTrue(torch.allclose(output, expected_output))
        for name, actual in zip(op.grad_names(), actual_grads):
            self.assertTrue(torch.allclose(actual, expected_grads[name]))
        self.assertEqual(provider.adapter_kind, "pytorch_eager_autograd")

    def test_mutation_guard_accepts_unchanged_and_rejects_in_place_write(self):
        from evograd.bench.fair import assert_tensors_unchanged, snapshot_tensors

        values = {"x": torch.arange(4.0), "eps": 1e-5}
        snapshots = snapshot_tensors(values)
        assert_tensors_unchanged(values, snapshots, provider="candidate")
        values["x"].add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "mutated benchmark input 'x'"):
            assert_tensors_unchanged(values, snapshots, provider="candidate")

    def test_identity_provider_reuses_exact_callables(self):
        from evograd.bench.fair import PairProvider, renamed_provider

        def forward(values):
            return values["x"], (values["x"],)

        def backward(dout, saved, values):
            return dout

        baseline = PairProvider(
            "liger", forward, backward, "hash", "stock_liger_pair"
        )
        identity = renamed_provider(baseline, "candidate")
        self.assertIs(identity.forward, baseline.forward)
        self.assertIs(identity.backward, baseline.backward)
        self.assertEqual(identity.source_hash, baseline.source_hash)
        self.assertEqual(identity.adapter_kind, baseline.adapter_kind)

    def test_saved_state_report_distinguishes_alias_and_new_storage(self):
        from evograd.bench.fair import _saved_state_report

        x = torch.arange(8.0)
        report = _saved_state_report((x[:4], x.clone()), {"x": x})
        self.assertEqual(report["logical_saved_bytes"], 12 * x.element_size())
        self.assertEqual(report["unique_saved_storage_bytes"], 16 * x.element_size())
        self.assertEqual(report["retained_input_bytes"], 8 * x.element_size())


if __name__ == "__main__":
    unittest.main()
