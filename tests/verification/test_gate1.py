"""Gate 1: the local pair against the model reference, on supplied cases, on CPU.

Gate 1 judges only a model-reference declaration, and only on the cases it is
given (recorded payloads in production). The fixtures here are put under a
model-reference contract (``F.model_reference``) and their declared inputs are
supplied as the case values (``F.supplied_cases``).
"""

from __future__ import annotations

import ast
import json
import pathlib
import unittest
from unittest import mock

import torch

from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle
from evograd.verification import gate1
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from tests.verification import _fixtures as F

CPU = "cpu"
PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "src" / "evograd" / "verification"

SCALE_SHIFT = F.model_reference(F.SCALE_SHIFT)
SPLIT = F.model_reference(F.SPLIT)


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def verify(op, candidate, **options):
    """Gate 1 on ``op`` (already a model reference) over its supplied cases."""
    return gate1.verify_pair(op, candidate, cases=F.supplied_cases(op), device=CPU, **options)


class TestPairContract(unittest.TestCase):
    def test_correct_pair_passes_every_declared_result(self):
        report = verify(SCALE_SHIFT, F.scale_shift_pair())
        self.assertEqual(report.status, PASSED, codes(report))
        names = [r.name for r in report.cases[0].results]
        self.assertEqual(names, ["out:y", "grad_in:dx", "grad_in:dw", "grad_in:db"])
        self.assertEqual(len(report.cases), 2)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_multi_output_pair(self):
        report = verify(SPLIT, F.split_pair())
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual([r.name for r in report.cases[0].results],
                         ["out:hi", "out:lo", "grad_in:dx", "grad_in:dw"])

    def test_wrong_forward_and_wrong_backward_are_rejected_by_name(self):
        fwd = verify(SCALE_SHIFT, F.scale_shift_pair(fwd_bias=1e-3))
        self.assertEqual(fwd.status, FAILED)
        failing = {r.name for r in fwd.cases[0].results if r.status == FAILED}
        self.assertEqual(failing, {"out:y"})
        bwd = verify(SCALE_SHIFT, F.scale_shift_pair(dw_scale=1.001))
        self.assertEqual({r.name for r in bwd.cases[0].results if r.status == FAILED},
                         {"grad_in:dw"})
        multi = verify(SPLIT, F.split_pair(lo_bias=1.0))
        self.assertEqual({r.name for r in multi.cases[0].results if r.status == FAILED}, {"out:lo"})

    def test_missing_extra_and_none_gradients(self):
        for kwargs, code in (({"drop_grad": True}, "gradient_arity"),
                             ({"extra_grad": True}, "gradient_arity"),
                             ({"none_grad": True}, "missing_result")):
            report = verify(SCALE_SHIFT, F.scale_shift_pair(**kwargs))
            self.assertEqual(report.status, FAILED, kwargs)
            self.assertIn(code, codes(report), kwargs)
        arity = verify(SCALE_SHIFT, F.scale_shift_pair(drop_grad=True))
        # Arity failure is not a silent truncation: every gradient is reported missing.
        missing = {r.name for r in arity.cases[0].results
                   if any(f.code == "missing_result" for f in r.findings)}
        self.assertEqual(missing, {"grad_in:dx", "grad_in:dw", "grad_in:db"})

    def test_shape_and_dtype(self):
        dtype = verify(SCALE_SHIFT, F.scale_shift_pair(wrong_dtype=True))
        self.assertIn("dtype_mismatch", codes(dtype))
        shape = verify(SCALE_SHIFT, F.scale_shift_pair(wrong_shape=True))
        self.assertIn("shape_mismatch", codes(shape))

    def test_non_finite_candidate_fails_and_json_stays_finite(self):
        report = verify(SCALE_SHIFT, F.scale_shift_pair(nan_output=True))
        self.assertEqual(report.status, FAILED)
        self.assertIn("non_finite_candidate", codes(report))
        json.dumps(report.to_dict(), allow_nan=False)

    def test_non_finite_reference_is_incomplete_where_allclose_said_ok(self):
        """Finite hardening: ``allclose`` treats inf == inf as close, so an
        allclose comparator accepts this output. The core calls the reference
        invalid and does not pass."""
        wl = SCALE_SHIFT.benchmark[0]
        values = make_case_inputs(SCALE_SHIFT, wl, device=CPU)
        values["b"] = values["b"].clone()
        values["b"][0] = float("inf")
        case = gate1.Gate1Case("inf_input", wl, values)
        report = gate1.verify_pair(SCALE_SHIFT, F.scale_shift_pair(), device=CPU, cases=[case])
        self.assertEqual(report.status, INCOMPLETE, codes(report))
        self.assertIn("invalid_reference", codes(report))
        self.assertNotIn("non_finite_candidate", codes(report))
        y_ref, _ = oracle(SCALE_SHIFT, values)
        y_cand, _ = F.scale_shift_pair().forward_with_saved(values["x"], values["w"], values["b"])
        self.assertTrue(torch.allclose(y_cand, y_ref, atol=1e-5, rtol=1e-5))

    def test_empty_case_set_never_passes(self):
        report = gate1.verify_pair(SCALE_SHIFT, F.scale_shift_pair(), device=CPU, cases=[])
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("no_cases", codes(report))

    def test_there_is_no_default_case_set(self):
        with self.assertRaises(TypeError):
            gate1.verify_pair(SCALE_SHIFT, F.scale_shift_pair(), device=CPU)
        with self.assertRaises(TypeError):
            gate1.Gate1Case("no_values", SCALE_SHIFT.benchmark[0])

    def test_a_declaration_without_a_model_reference_is_refused(self):
        cases = F.supplied_cases(F.SCALE_SHIFT)
        with self.assertRaisesRegex(ValueError, "model-reference"):
            gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), cases=cases, device=CPU)

    def test_input_mutation_is_a_contract(self):
        report = verify(SCALE_SHIFT, F.scale_shift_pair(mutate_input=True))
        self.assertIn("input_mutation", codes(report))
        upstream = verify(SCALE_SHIFT, F.scale_shift_pair(mutate_upstream=True))
        self.assertIn("input_mutation", codes(upstream))
        recorded = verify(SCALE_SHIFT, F.scale_shift_pair(mutate_upstream=True),
                          input_mutation=gate1.RECORD)
        self.assertEqual(recorded.status, PASSED)
        self.assertEqual(recorded.cases[0].evidence["input_mutation"]["changed"], ["dy"])

    def test_verification_leaves_the_global_rng_alone(self):
        cases = F.supplied_cases(SCALE_SHIFT)
        torch.manual_seed(123)
        state = torch.get_rng_state()
        gate1.verify_pair(SCALE_SHIFT, F.scale_shift_pair(), cases=cases, device=CPU)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))


class TestReferenceAndTolerance(unittest.TestCase):
    def test_the_oracle_promotes_to_the_reference_dtype_and_casts_back(self):
        # A candidate that computes in float32 and rounds once equals the
        # float32-then-cast oracle; the same-dtype oracle rounds x + 1000 to
        # bfloat16 and loses x.
        values = make_case_inputs(F.CANCEL_F32_REF, F.CANCEL_F32_REF.benchmark[0], device=CPU)
        promoted, _ = oracle(F.CANCEL_F32_REF, values)
        same, _ = oracle(F.CANCEL_SAME_DTYPE, values)
        candidate, _ = F.cancel_f32_pair().forward_with_saved(values["x"])
        self.assertEqual(promoted.dtype, torch.bfloat16)               # cast back to the contract
        self.assertTrue(torch.equal(promoted, candidate))
        self.assertFalse(torch.equal(same, candidate))

    def test_an_error_inside_the_dtype_default_passes(self):
        wl = SPLIT.benchmark[0]
        values = make_case_inputs(SPLIT, wl, device=CPU)
        _, grads = oracle(SPLIT, values)
        atol, rtol = 1e-5, 1.3e-6                                       # float32 defaults
        bound = float((atol + rtol * grads["dw"].abs()).min())

        def shifted(delta):
            pair = F.split_pair()
            backward = pair.backward_from_saved

            def bwd(output_grads, saved, alpha=2.0):
                dx, dw = backward(output_grads, saved, alpha)
                return dx, dw + delta
            pair.backward_from_saved = bwd
            return pair

        self.assertEqual(verify(SPLIT, shifted(0.5 * bound)).status, PASSED)
        self.assertEqual(verify(SPLIT, shifted(2.0 * atol + 1e-3)).status, FAILED)

    def test_a_result_whose_tolerance_is_unavailable_is_incomplete(self):
        with mock.patch("evograd.verification.gate1.runner.result_tolerances",
                        side_effect=ValueError("unsupported result dtype")):
            report = verify(SCALE_SHIFT, F.scale_shift_pair())
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("tolerance_unavailable", codes(report))


class TestEntries(unittest.TestCase):
    def test_module_and_provider_reach_the_same_core(self):
        provider = F.CountingProvider(F.scale_shift_pair())
        via_provider = verify(SCALE_SHIFT, provider)
        via_module = verify(SCALE_SHIFT, F.scale_shift_pair())
        self.assertEqual(via_provider.status, PASSED, codes(via_provider))
        self.assertEqual(len(provider.prepared), 2)                 # prepare really ran
        self.assertEqual(via_provider.subject["candidate_identity"]["adapter_kind"], "test_provider")
        self.assertEqual([r.metrics for r in via_provider.cases[0].results],
                         [r.metrics for r in via_module.cases[0].results])

    def test_declared_baseline_hook(self):
        import types

        pair = F.scale_shift_pair()
        hook = types.SimpleNamespace(
            pair_factory=lambda: (pair.forward_with_saved, pair.backward_from_saved),
            forward_args=("x", "w", "b"), backward_extras=())
        report = verify(SCALE_SHIFT, gate1.pair_from_baseline(SCALE_SHIFT, "toy", hook))
        self.assertEqual(report.status, PASSED, codes(report))
        bad = F.scale_shift_pair(dw_scale=1.1)
        hook.pair_factory = lambda: (bad.forward_with_saved, bad.backward_from_saved)
        self.assertEqual(verify(SCALE_SHIFT, gate1.pair_from_baseline(
            SCALE_SHIFT, "toy", hook)).status, FAILED)

    def test_a_probe_with_an_invalid_reference_is_skipped_not_passed(self):
        op = F.model_reference(F.SUMSQ_SAME_DTYPE)
        wl = op.benchmark[0]
        values = make_case_inputs(op, wl, device=CPU)
        values["x"] = torch.full_like(values["x"], 1e30)            # sum of squares overflows
        probe = gate1.Gate1Case("stress", wl, values, gate1.PROBE_CASE)
        report = gate1.verify_pair(op, F.sumsq_f32_pair(), device=CPU, cases=[probe])
        self.assertEqual({c.status for c in report.cases}, {"skipped"})
        self.assertEqual(report.status, INCOMPLETE)                # nothing decisive ran

    def test_supplied_inputs_must_match_the_declaration(self):
        wl = SCALE_SHIFT.benchmark[0]
        values = make_case_inputs(SCALE_SHIFT, wl, device=CPU)
        values["x"] = values["x"][:, :-1]
        report = gate1.verify_pair(SCALE_SHIFT, F.scale_shift_pair(), device=CPU,
                                   cases=[gate1.Gate1Case("bad", wl, values)])
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("invalid_case_inputs", codes(report))

    def test_smoke_mode_is_never_a_pass(self):
        report = verify(SCALE_SHIFT, F.scale_shift_pair(), mode=gate1.SMOKE)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("smoke_only", codes(report))
        broken = verify(SCALE_SHIFT, F.scale_shift_pair(wrong_shape=True), mode=gate1.SMOKE)
        self.assertEqual(broken.status, FAILED)

    def test_evolution_summary_and_saved_state_observer(self):
        report = verify(SCALE_SHIFT, F.scale_shift_pair(dw_scale=1.01),
                        observers={"saved_state": gate1.saved_state_observer})
        summary = gate1.evolution_correctness(report, SCALE_SHIFT)
        self.assertEqual((summary["passed"], summary["total"]), (0, 2))
        self.assertEqual(summary["forward_correctness"], 1.0)
        self.assertEqual(summary["dw_correctness"], 0.0)
        self.assertEqual(summary["dx_correctness"], 1.0)
        saved = report.cases[0].evidence["saved_state"]
        self.assertGreater(saved["logical_saved_bytes"], 0)
        self.assertGreater(saved["retained_input_bytes"], 0)       # it saves x and w as-is


class TestIndependence(unittest.TestCase):
    """The new cores do not call back into the old verifiers or gates."""

    FORBIDDEN = ("evograd.opdecl.verify", "evograd.evolve", "evograd.pipelines.unified",
                 "evograd.evaluation.common.provider", "evograd.evaluation.tier3.gate",
                 "evograd.evaluation.tier3.runner", "evograd.evaluation.tier3.block",
                 "evograd.opdecl.baselines")
    FORBIDDEN_NAMES = {"check_module", "check_candidate", "model_correctness",
                       "check_model_correctness", "verify_pair_provider", "_run_correctness",
                       "validate_numerical_variants", "run_training", "training_distances"}

    def test_no_import_of_old_verifiers(self):
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for bad in self.FORBIDDEN:
                        self.assertFalse(node.module == bad or node.module.startswith(bad + "."),
                                         f"{path.name} imports {node.module}")
                    names = {a.name for a in node.names}
                    self.assertFalse(names & self.FORBIDDEN_NAMES, f"{path.name}: {names}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in self.FORBIDDEN:
                            self.assertFalse(alias.name.startswith(bad), path.name)


if __name__ == "__main__":
    unittest.main()
