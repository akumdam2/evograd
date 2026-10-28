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
    ENVELOPE_GAMMA, MODEL_POLICY, model_policy_identity, result_tolerances, rounding_floor,
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
        bad = torch.tensor([1.0625], dtype=torch.bfloat16)   # 8 ulps: beyond the rounding floor
        for name in ("out:hi", "out:lo", "grad_in:dw"):
            with self.subTest(name=name):
                check = self.compare(ref, bad, name=name, op=op)
                self.assertEqual((check.limits["atol"], check.limits["rtol"]), (1e-5, 0.016))
                self.assertEqual(check.status, "failed")

    def test_bfloat16_boundary_matches_assert_close(self):
        # The dtype defaults alone, without the rounding floor.
        ref = torch.tensor([1.0, 0.0], dtype=torch.bfloat16)
        for values in ([1.0078125, 0.0], [1.03125, 0.0], [1.0, 2e-5]):
            candidate = torch.tensor(values, dtype=torch.bfloat16)
            try:
                torch.testing.assert_close(candidate, ref)
                expected = "passed"
            except AssertionError:
                expected = "failed"
            with mock.patch("evograd.verification.gate1.runner.rounding_floor", return_value=0.0):
                self.assertEqual(self.compare(ref, candidate).status, expected)

    def test_the_compile_envelope_covers_a_cancelled_element_but_not_a_scale_fault(self):
        # Two terms of size 0.5 cancel to 0.03 in the reference; rounding the operands
        # moves the result by 0.02, past the rounding floor (4 ulps of 0.75 = 0.0156).
        # The trusted compile moved it by up to 0.0075 on the calibration payloads:
        # envelope = 4 * 0.0075.
        ref = torch.tensor([0.75, -0.5, 0.03125, 0.25], dtype=torch.bfloat16)
        rounded = ref.clone(); rounded[2] = 0.05078125
        contract = C.ResultContract((C.ResultSpec("out:y", C.OUTPUT),))
        judge = lambda cand, env: compare_results(self.op, self.workload, contract, {"out:y": cand},
                                                  {"out:y": ref}, envelope=env)[0]
        self.assertEqual(judge(rounded, None).status, "failed")          # no calibration
        check = judge(rounded, {"out:y": 0.03})
        self.assertEqual(check.status, "passed")
        self.assertEqual(check.limits["envelope"], 0.03)
        self.assertEqual(judge((ref.float() * 1.05).to(ref.dtype), {"out:y": 0.03}).status, "failed")
        self.assertEqual(judge(rounded, {"out:other": 0.03}).status, "failed")   # per result

    def test_a_result_compile_reproduces_bit_for_bit_still_accepts_another_rounding(self):
        # Envelope 0 (a vendor kernel under compile too): a cancelled element one rounding
        # of a 0.5-sized operand away (0.0039) passes; a 5% scale fault does not.
        ref = torch.tensor([0.75, -0.5, 0.03125, 0.25], dtype=torch.bfloat16)
        rounded = ref.clone(); rounded[2] = 0.03515625
        floor = ENVELOPE_GAMMA * 2.0 ** -7 * 2.0 ** -1      # ulp of 0.75 in bf16 is 2**-8
        self.assertEqual(rounding_floor(self.op, ref), floor)
        check = self.compare(ref, rounded)
        self.assertEqual(check.status, "passed")
        self.assertEqual((check.limits["envelope"], check.limits["rounding_floor"]), (floor, floor))
        self.assertEqual(self.compare(ref, (ref.float() * 1.05).to(ref.dtype)).status, "failed")
        self.assertEqual(rounding_floor(self.op, torch.zeros(3, dtype=torch.bfloat16)), 0.0)
        self.assertEqual(rounding_floor(F.SPLIT, ref), 0.0)          # generic task: declared rule

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
            self.assertEqual((result.limits["atol"], result.limits["rtol"]), (1e-5, 1.3e-6))
            self.assertEqual(result.limits["envelope"], result.limits["rounding_floor"])

    def test_generic_declaration_multipliers_and_hooks_are_preserved(self):
        workload = F.SPLIT.correctness[0]
        for name in ("lo", "dw"):
            self.assertEqual(result_tolerances(F.SPLIT, workload, name, torch.ones(1)),
                             F.SPLIT.tolerance_for(workload, name))


if __name__ == "__main__":
    unittest.main()
