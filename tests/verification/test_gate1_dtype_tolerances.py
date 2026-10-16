"""Model Gate 1 uses dtype defaults, independently of legacy declaration scales."""

from dataclasses import replace
import unittest
from unittest import mock

import torch

from evograd.opdecl import Workload
from evograd.verification import gate1
from evograd.verification.common import contracts as C
from evograd.verification.gate1.runner import compare_results, PROTOCOL
from evograd.verification.gate1.tolerances import (
    MODEL_POLICY, model_policy_identity, result_tolerances,
)
from tests.verification import _fixtures as F


class DtypeToleranceTests(unittest.TestCase):
    def setUp(self):
        self.op = replace(F.SPLIT, reference_contract="evograd-model-reference/test/1")
        self.workload = Workload({"R": 4, "C": 8}, "bfloat16", atol=99, rtol=99)

    def compare(self, reference, candidate, *, name="out:y", op=None):
        kind = C.OUTPUT if name.startswith("out:") else C.INPUT_GRAD
        contract = C.ResultContract((C.ResultSpec(name, kind),))
        return compare_results(op or self.op, self.workload, contract,
                               {name: candidate}, {name: reference})[0]

    def test_defaults_match_installed_assert_close_policy(self):
        # Private helper is used only to audit the frozen public defaults in tests.
        from torch.testing._comparison import default_tolerances

        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                atol, rtol = result_tolerances(self.op, self.workload, "y", torch.ones(1, dtype=dtype))
                self.assertEqual((rtol, atol), default_tolerances(dtype))

    def test_model_results_ignore_every_legacy_tolerance_override(self):
        op = replace(self.op, tolerances={"bfloat16": (99, 99)},
                     tolerance_multipliers={"dw": (1000, 1000)},
                     tolerance_hook=mock.Mock(side_effect=AssertionError("legacy hook called")))
        ref = torch.tensor([1.0], dtype=torch.bfloat16)
        bad = torch.tensor([1.03125], dtype=torch.bfloat16)
        for name in ("out:hi", "out:lo", "grad_in:dw"):
            with self.subTest(name=name):
                check = self.compare(ref, bad, name=name, op=op)
                self.assertEqual(check.limits, {"atol": 1e-5, "rtol": 0.016})
                self.assertEqual(check.status, "failed")

    def test_bfloat16_boundary_matches_assert_close(self):
        ref = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        for values in ([1.0078125, 0.0], [1.03125, 0.0], [1.0, 2e-5]):
            candidate = torch.tensor(values, dtype=torch.bfloat16)
            try:
                torch.testing.assert_close(candidate, ref)
                expected = "passed"
            except AssertionError:
                expected = "failed"
            self.assertEqual(self.compare(ref, candidate).status, expected)

    def test_result_dtype_not_bfloat16_workload_selects_float32_loss_budget(self):
        ref = torch.tensor([1.0], dtype=torch.float32)
        check = self.compare(ref, ref + 1e-3, name="out:loss")
        self.assertEqual(check.limits, {"atol": 1e-5, "rtol": 1.3e-6})
        self.assertEqual(check.status, "failed")

    def test_zero_and_negated_gradients_above_absolute_floor_fail(self):
        ref = torch.tensor([1e-4, -1e-4], dtype=torch.bfloat16)
        for bad in (torch.zeros_like(ref), -ref):
            self.assertEqual(self.compare(ref, bad, name="grad_in:dx").status, "failed")

    def test_fixed_absolute_floor_still_accepts_tiny_zeroed_gradient(self):
        # Document the selected policy's real limitation; do not promise that
        # changing the defaults detects every zeroed tiny training gradient.
        ref = torch.tensor([1e-6], dtype=torch.bfloat16)
        self.assertEqual(self.compare(ref, torch.zeros_like(ref), name="grad_in:dx").status, "passed")

    def test_wrong_candidate_dtype_does_not_select_a_looser_budget(self):
        ref = torch.ones(1, dtype=torch.float32)
        check = self.compare(ref, ref.to(torch.bfloat16))
        self.assertEqual(check.status, "failed")
        self.assertIn("dtype_mismatch", [f.code for f in check.findings])

    def test_unknown_model_result_dtype_is_not_assigned_a_guessed_tolerance(self):
        with self.assertRaisesRegex(ValueError, "unsupported result dtype"):
            result_tolerances(self.op, self.workload, "y", torch.ones(1, dtype=torch.complex64))

    def test_policy_identity_includes_effective_table_and_report_version(self):
        report = gate1.verify_pair(self.op, F.split_pair(), device="cpu")
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.protocol, PROTOCOL)
        self.assertEqual(report.subject["tolerance_policy"], model_policy_identity())
        self.assertEqual(report.subject["tolerance_policy"]["schema"], MODEL_POLICY)
        for result in report.cases[0].results:
            self.assertEqual(result.limits, {"atol": 1e-5, "rtol": 1.3e-6})

    def test_generic_declaration_multipliers_and_hooks_are_preserved(self):
        workload = F.SPLIT.correctness[0]
        for name in ("lo", "dw"):
            self.assertEqual(result_tolerances(F.SPLIT, workload, name, torch.ones(1)),
                             F.SPLIT.tolerance_for(workload, name))


if __name__ == "__main__":
    unittest.main()
