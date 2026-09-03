"""Declarations with more than one output.

An operator that returns ``(q, k, v)`` cannot be described by a contract that
knows about one tensor, and the failure mode of pretending otherwise is quiet:
a backward that consumes only the first upstream gradient produces gradients
that are wrong for reasons no shape check can see. These tests exercise the
multi-output path through every framework surface, and re-assert that the
single-output path is unchanged.
"""

from __future__ import annotations

import unittest

import torch

from evograd.opdecl import Active, Inactive, Workload, declare_op
from evograd.opdecl.bind import bind
from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
from evograd.opdecl.oracle import oracle
from evograd.opdecl.verify import verify
from evograd.ops import get_op

from tests import _structured_fixture as fixture

OP = fixture.op
CASE = OP.correctness[0]


class TestDeclaration(unittest.TestCase):
    def test_a_tuple_output_is_reported_as_several(self):
        self.assertTrue(OP.is_multi_output)
        self.assertEqual(OP.output_names, ("hi", "lo"))
        self.assertEqual(OP.upstream_grad_names, ("dhi", "dlo"))
        self.assertEqual(len(OP.outputs), 2)

    def test_outputs_may_have_different_shapes(self):
        self.assertEqual([o.shape for o in OP.outputs], ["[R, C]", "[R]"])

    def test_the_single_upstream_accessor_refuses_a_multi_output_declaration(self):
        """Returning the first would silently produce a wrong backward."""
        with self.assertRaises(ValueError) as ctx:
            OP.upstream_grad_name
        self.assertIn("upstream_grad_names", str(ctx.exception))

    def test_the_candidate_contract_names_the_tuple(self):
        self.assertEqual(OP.forward_returns(), "(hi, lo), saved_tensors")
        self.assertTrue(OP.backward_parameters().startswith("output_grads, saved_tensors"))
        self.assertEqual(OP.backward_returns(), "dx, dw")

    def test_an_empty_output_tuple_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._declare(output=())
        self.assertIn("non-empty", str(ctx.exception))

    def test_a_non_active_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self._declare(output=(Active("a", "[R]"), Inactive("b", "[R]")))

    def test_duplicate_output_names_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._declare(output=(Active("a", "[R]"), Active("a", "[R]")))
        self.assertIn("duplicate output names", str(ctx.exception))

    def test_an_output_name_colliding_with_an_argument_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._declare(output=(Active("x", "[R]"), Active("b", "[R]")))
        self.assertIn("collide with argument names", str(ctx.exception))

    def test_output_shapes_are_bound_against_the_declared_dims(self):
        with self.assertRaises(ValueError):
            self._declare(output=(Active("a", "[R]"), Active("b", "[NOPE]")))

    def _declare(self, *, output):
        return declare_op(
            name="probe",
            forward="tests._structured_fixture:split_scale_forward_ref",
            dims=("R", "C"),
            args=(Active("x", "[R, C]"),),
            output=output,
            forward_semantics="probe",
            backward_semantics="probe",
        )


class TestInputsAndOracle(unittest.TestCase):
    def test_one_upstream_gradient_is_generated_per_output(self):
        values = make_case_inputs(OP, CASE, device="cpu")
        self.assertEqual(list(values["dhi"].shape), [4, 8])
        self.assertEqual(list(values["dlo"].shape), [4])
        grads = upstream_grad_values(OP, values)
        self.assertIsInstance(grads, tuple)
        self.assertEqual(len(grads), 2)

    def test_the_oracle_returns_outputs_in_declaration_order(self):
        values = make_case_inputs(OP, CASE, device="cpu")
        outputs, grads = oracle(OP, values)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual([list(o.shape) for o in outputs], [[4, 8], [4]])
        self.assertEqual(sorted(grads), ["dw", "dx"])

    def test_the_oracle_uses_every_upstream_gradient(self):
        """Zeroing the second output's gradient must change the result."""
        values = make_case_inputs(OP, CASE, device="cpu")
        _out, full = oracle(OP, values)
        values["dlo"] = torch.zeros_like(values["dlo"])
        _out, partial = oracle(OP, values)
        self.assertFalse(torch.allclose(full["dx"], partial["dx"]))

    def test_a_mismatched_gradient_count_is_refused(self):
        values = make_case_inputs(OP, CASE, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            oracle(OP, values, dout=(values["dhi"],))
        self.assertIn("upstream gradients", str(ctx.exception))

    def test_as_output_tuple_enforces_the_arity(self):
        with self.assertRaises(ValueError):
            as_output_tuple(OP, torch.zeros(4, 8))
        with self.assertRaises(ValueError):
            as_output_tuple(OP, (torch.zeros(4, 8),))
        single = get_op("layernorm")
        with self.assertRaises(ValueError):
            as_output_tuple(single, (torch.zeros(2),))


class TestCorrectnessGate(unittest.TestCase):
    def _verify(self, module):
        return verify(OP, module, device="cpu")

    def test_a_correct_pair_passes(self):
        report = self._verify(fixture.good_module())
        for case in report.cases:
            self.assertIsNone(case.error)
            self.assertEqual([c.name for c in case.checks], ["hi", "lo", "dx", "dw"])
            self.assertTrue(all(c.ok for c in case.checks))

    def _failed_checks(self, module):
        report = self._verify(module)
        names = []
        for case in report.cases:
            if case.error:
                names.append("error")
            else:
                names.extend(c.name for c in case.checks if not c.ok)
        return set(names)

    def test_a_single_tensor_where_a_tuple_is_declared_fails(self):
        self.assertEqual(self._failed_checks(fixture.single_output_module()), {"error"})

    def test_a_wrong_output_count_fails(self):
        self.assertEqual(self._failed_checks(fixture.wrong_arity_module()), {"error"})

    def test_swapped_outputs_fail(self):
        """Ordering is part of the contract, not a detail."""
        self.assertTrue(self._failed_checks(fixture.swapped_outputs_module()))

    def test_a_wrong_shape_on_either_output_fails(self):
        self.assertIn("hi", self._failed_checks(fixture.wrong_shape_module(0)))
        self.assertIn("lo", self._failed_checks(fixture.wrong_shape_module(1)))

    def test_a_wrong_dtype_on_either_output_fails(self):
        self.assertIn("hi", self._failed_checks(fixture.wrong_dtype_module(0)))
        self.assertIn("lo", self._failed_checks(fixture.wrong_dtype_module(1)))

    def test_only_the_wrong_output_is_reported(self):
        """The point of per-output checks: a right ``hi`` stays right."""
        failed = self._failed_checks(fixture.wrong_second_output_module())
        self.assertEqual(failed, {"lo"})

    def test_a_wrong_gradient_fails(self):
        self.assertEqual(self._failed_checks(fixture.wrong_gradient_module()), {"dw"})

    def test_a_backward_that_ignores_the_second_upstream_gradient_fails(self):
        failed = self._failed_checks(fixture.ignores_second_grad_module())
        self.assertEqual(failed, {"dx", "dw"})


class TestLayoutAndAutograd(unittest.TestCase):
    def test_a_non_contiguous_output_is_visible_in_the_report(self):
        report = verify(OP, fixture.non_contiguous_output_module(), device="cpu")
        # Values still agree, so the numeric gate passes; the layout difference
        # is what a stride-sensitive consumer would see.
        self.assertTrue(all(c.ok for case in report.cases for c in case.checks))
        values = make_case_inputs(OP, CASE, device="cpu")
        module = fixture.non_contiguous_output_module()
        (hi, _lo), _saved = getattr(module, OP.forward_fn_name)(values["x"], values["w"])
        self.assertFalse(hi.is_contiguous())

    def test_bind_wires_autograd_through_every_output(self):
        values = make_case_inputs(OP, CASE, device="cpu")
        fn = bind(OP, fixture.good_module())
        x = values["x"].detach().clone().requires_grad_(True)
        w = values["w"].detach().clone().requires_grad_(True)
        hi, lo = fn(x, w)
        torch.autograd.backward((hi, lo), (values["dhi"], values["dlo"]))
        _out, expected = oracle(OP, values)
        self.assertTrue(torch.allclose(x.grad, expected["dx"], atol=1e-5))
        self.assertTrue(torch.allclose(w.grad, expected["dw"], atol=1e-5))


class TestRuntimeForward(unittest.TestCase):
    def test_a_matching_production_spelling_passes(self):
        from evograd.opdecl import baselines

        baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)
        baselines.verify_runtime_forward(OP, device="cpu")

    def test_a_production_spelling_that_differs_on_one_output_is_rejected(self):
        from evograd.opdecl import baselines

        def wrong(x, w, alpha=2.0):
            scaled = x * w
            return scaled * alpha, scaled.sum(-1) + 1.0

        original = fixture.split_scale_forward_production
        fixture.split_scale_forward_production = wrong
        try:
            baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)
            with self.assertRaises(RuntimeError) as ctx:
                baselines.verify_runtime_forward(OP, device="cpu")
            self.assertIn("output 'lo'", str(ctx.exception))
        finally:
            fixture.split_scale_forward_production = original
            baselines._RUNTIME_FORWARD_VERIFIED.discard(OP.name)


class TestSingleOutputRegression(unittest.TestCase):
    """Every existing declaration keeps the old surface exactly."""

    #: The registered operators with structured outputs. Listed explicitly so
    #: adding another cannot silently weaken the regression coverage below.
    MULTI_OUTPUT = {"qwen3_qkv_norm_rope", "fused_add_rms_norm"}

    def test_the_multi_output_registry_is_exactly_what_is_expected(self):
        from evograd.ops import OPS

        self.assertEqual(
            {name for name, op in OPS.items() if op.is_multi_output}, self.MULTI_OUTPUT
        )

    def test_every_registered_operator_is_single_output_and_unchanged(self):
        from evograd.ops import OPS

        for name, op in OPS.items():
            if name in self.MULTI_OUTPUT:
                continue
            with self.subTest(op=name):
                self.assertFalse(op.is_multi_output)
                self.assertIsInstance(op.output, Active)
                self.assertEqual(op.outputs, (op.output,))
                self.assertEqual(op.output_names, (op.output.name,))
                self.assertEqual(op.upstream_grad_name, op.output.grad_name)
                self.assertEqual(op.upstream_grad_parameter, op.output.grad_name)
                self.assertEqual(op.forward_returns(), f"{op.output.name}, saved_tensors")

    def test_single_output_inputs_and_oracle_are_unchanged(self):
        op = get_op("layernorm")
        case = op.correctness[0]
        values = make_case_inputs(op, case, device="cpu")
        self.assertIn(op.upstream_grad_name, values)
        grads = upstream_grad_values(op, values)
        self.assertTrue(torch.is_tensor(grads))
        y, ref = oracle(op, values)
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(sorted(ref), sorted(op.grad_names()))

    def test_single_output_backward_parameters_keep_the_gradient_name(self):
        op = get_op("layernorm")
        self.assertTrue(op.backward_parameters().startswith("dy, saved_tensors"))
        self.assertNotIn("output_grads", op.backward_parameters())


if __name__ == "__main__":
    unittest.main()
