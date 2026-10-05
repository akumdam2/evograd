"""Domain-preserving variant admission, independent of CUDA seed synthesis."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import torch

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.importing import resolve_callable
from evograd.opdecl.inputs import as_output_tuple
from evograd.opdecl.verify import verify
from evograd.pipelines.unified.validation import _variants, validate_numerical_variants


def mean_ref(x):
    return x.mean(-1, keepdim=True)


def exponential_ref(x):
    return torch.exp(x)


def invalid_ref(x):
    return x * float("nan")


def two_outputs_ref(x):
    return 2 * x, x.mean(-1, keepdim=True)


def _mean_op(dtype="float16"):
    return declare_op(
        name="validation_mean", forward="tests.test_unified_validation:mean_ref",
        dims=("rows", "cols"), args=(Active("x", "[rows, cols]"),),
        output=Active("y", "[rows, 1]"), parameter_args=(),
        forward_semantics="Row mean", backward_semantics="Broadcast dy / cols",
        correctness=(Workload({"rows": 2, "cols": 100}, dtype),),
        tolerances={dtype: (1e-3, 1e-3)},
    )


def _reference_pair(op):
    forward = resolve_callable(op.forward)
    active_names = {arg.name for arg in op.active_args()}

    def fwd(*args):
        return forward(*args), args

    def bwd(upstream, saved, **kwargs):
        with torch.enable_grad():
            args, leaves = [], []
            for declaration, value in zip(op.args, saved):
                if declaration.name in active_names:
                    value = value.detach().clone().requires_grad_(True)
                    leaves.append(value)
                args.append(value)
            outputs = as_output_tuple(op, forward(*args))
            grads = torch.autograd.grad(outputs, leaves,
                                        upstream if isinstance(upstream, tuple) else (upstream,))
            by_name = {arg.grad_name: grad for arg, grad in zip(op.active_args(), grads)}
            return tuple(by_name[name] for name in op.grad_names())

    return SimpleNamespace(forward_with_saved=fwd, backward_from_saved=bwd)


class TestUnifiedNumericalVariants(unittest.TestCase):
    def test_mean_overflow_candidate_passes_default_but_fails_stress(self):
        op = _mean_op()

        def forward(x):
            return x.sum(-1, keepdim=True) / x.shape[-1], (x,)

        def backward(dy, saved):
            x, = saved
            return (dy.expand_as(x) / x.shape[-1],)

        candidate = SimpleNamespace(forward_with_saved=forward, backward_from_saved=backward)
        self.assertTrue(verify(op, candidate, device="cpu").ok)
        report = validate_numerical_variants(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        large = next(case for case in report["cases"] if case["variant"] == "stress_large_constant")
        self.assertEqual(large["status"], "failed_candidate")
        self.assertFalse(large["pair"]["ok"])
        self.assertFalse(large["deployment"]["ok"])
        json.dumps(report, allow_nan=False)

    def test_correct_mean_passes_all_seven_variants(self):
        op = _mean_op()
        report = validate_numerical_variants(op, _reference_pair(op), device="cpu")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["coverage"]["passed"], 7)
        self.assertEqual(report["coverage"]["skipped_invalid_reference"], 0)

    def test_variants_are_reproducible_and_restore_rng_state(self):
        op = _mean_op("float32")
        module = _reference_pair(op)
        before = torch.random.get_rng_state().clone()
        first = list(_variants(op, op.correctness[0], module, "cpu"))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = list(_variants(op, op.correctness[0], module, "cpu"))
        for (name, a, _, _), (other_name, b, _, _) in zip(first, second):
            self.assertEqual(name, other_name)
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))
        self.assertFalse(torch.equal(first[0][1]["x"], first[1][1]["x"]))
        self.assertFalse(torch.equal(first[1][1]["x"], first[2][1]["x"]))

    def test_kl_factory_reseeding_preserves_probability_domain(self):
        from evograd.ops.level1.kl_div import op as declared

        op = replace(declared, correctness=(Workload({"rows": 2, "cols": 8}, "float32"),))
        variants = list(_variants(op, op.correctness[0], SimpleNamespace(), "cpu"))
        self.assertEqual(len(variants), 5)
        self.assertFalse(torch.equal(variants[0][1]["y_pred"], variants[1][1]["y_pred"]))
        for name, values, stress, _ in variants:
            with self.subTest(variant=name):
                self.assertFalse(stress)
                self.assertTrue(bool((values["y_true"] > 0).all()))
                torch.testing.assert_close(values["y_true"].sum(-1), torch.ones(2))
                torch.testing.assert_close(values["y_pred"].exp().sum(-1), torch.ones(2))
        report = validate_numerical_variants(op, _reference_pair(op), device="cpu")
        self.assertTrue(report["ok"], report)

    def test_norm_stress_preserves_parameters_inactive_metadata_and_scalars(self):
        def inputs(t, op, workload, device):
            t.manual_seed(41)
            x = t.randn(2, 8, device=device)
            return {"x": x, "weight": t.randn(8, device=device),
                    "bias": t.randn(8, device=device), "mask": t.rand(2, 8, device=device) > .5,
                    "labels": t.randint(0, 8, (2,), device=device),
                    "cos": t.arange(8, device=device).float().cos(),
                    "sin": t.arange(8, device=device).float().sin(),
                    "eps": 1e-5, "dy": t.randn(2, 1, device=device)}

        base = _mean_op("float32")
        op = replace(base, args=(Active("x", "[rows, cols]"),
                                Active("weight", "[cols]"), Active("bias", "[cols]"),
                                Inactive("mask", "[rows, cols]", dtype="bool"),
                                Inactive("labels", "[rows]", dtype="int64"),
                                Inactive("cos", "[cols]"), Inactive("sin", "[cols]"),
                                Inactive("eps", default=1e-5)),
                     make_inputs=inputs, parameter_args=("weight", "bias"), family=None,
                     correctness=(Workload({"rows": 2, "cols": 8}, "float32"),))
        module = SimpleNamespace(UNIFIED_DEPLOYMENT_CONFIG={"family": "norm"})
        variants = list(_variants(op, op.correctness[0], module, "cpu"))
        default = variants[0][1]
        stress_cases = [(values, metadata) for _, values, stress, metadata in variants if stress]
        self.assertEqual(len(stress_cases), 2)
        for values, metadata in stress_cases:
            self.assertEqual(metadata["changed_activations"], ["x"])
            for name in ("weight", "bias", "mask", "labels", "cos", "sin"):
                self.assertTrue(torch.equal(values[name], default[name]), name)
            self.assertEqual(values["eps"], default["eps"])

    def test_invalid_stress_references_are_reported_not_counted_as_passes(self):
        op = replace(_mean_op("float32"), forward="tests.test_unified_validation:exponential_ref",
                     output=Active("y", "[rows, cols]"))
        report = validate_numerical_variants(op, _reference_pair(op), device="cpu")
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["coverage"]["passed"], 5)
        self.assertEqual(report["coverage"]["skipped_invalid_reference"], 2)
        skipped = [case for case in report["cases"] if case["ok"] is None]
        self.assertTrue(all(case["status"] == "skipped_invalid_reference" for case in skipped))

    def test_invalid_default_reference_is_a_failure(self):
        op = replace(_mean_op("float32"), forward="tests.test_unified_validation:invalid_ref",
                     output=Active("y", "[rows, cols]"))
        report = validate_numerical_variants(op, _reference_pair(op), device="cpu")
        self.assertFalse(report["ok"])
        self.assertEqual(report["cases"][0]["status"], "failed_reference")

    def test_backward_must_return_every_declared_gradient(self):
        op = _mean_op("float32")
        candidate = _reference_pair(op)
        candidate.backward_from_saved = lambda upstream, saved: ()
        report = validate_numerical_variants(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        self.assertIn("expected 1 gradients, got 0", report["cases"][0]["error"])

    def test_structured_outputs_each_receive_independent_upstream_gradients(self):
        op = replace(_mean_op("float32"), forward="tests.test_unified_validation:two_outputs_ref",
                     output=(Active("y", "[rows, cols]"), Active("z", "[rows, 1]")))
        candidate = _reference_pair(op)
        report = validate_numerical_variants(op, candidate, device="cpu")
        self.assertTrue(report["ok"], report)
        self.assertEqual({check["name"] for check in report["cases"][0]["pair"]["checks"]},
                         {"y", "z", "dx"})
        candidate.backward_from_saved = lambda upstream, saved: (2 * upstream[0],)
        report = validate_numerical_variants(op, candidate, device="cpu")
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
