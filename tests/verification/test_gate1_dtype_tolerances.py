"""Model Gate 1: max|c-t| <= bound*max|t| per result against a float32 truth; bound = the larger of
the budget (by the call's least precise dtype) and twice native's own error, for every task."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from evograd.opdecl import Active, Inactive, Workload
from evograd.verification import gate1
from evograd.verification.common import contracts as C
from evograd.verification.gate1 import runner, tolerances
from evograd.verification.gate1.runner import (
    PROTOCOL, compare_results, native_errors, truth_results, widened,
)
from evograd.verification.gate1.tolerances import (
    BUDGETS, MODEL_POLICY, NATIVE_FACTOR, allowance, model_policy_identity, policy_identity,
    result_bound, result_budget, rule_dtype,
)
from tests.verification import _fixtures as F

BF16 = torch.bfloat16


class DtypeRuleTests(unittest.TestCase):
    def setUp(self):
        self.op = replace(F.SPLIT, reference_contract="evograd-model-reference/test/1")
        self.workload = Workload({"R": 4, "C": 8}, "bfloat16", atol=99, rtol=99)

    def compare(self, reference, candidate, *, name="out:y", truth=None, native_error=None):
        kind = C.OUTPUT if name.startswith("out:") else C.INPUT_GRAD
        contract = C.ResultContract((C.ResultSpec(name, kind),))
        return compare_results(self.op, self.workload, contract, {name: candidate}, {name: reference},
                               truth=None if truth is None else {name: truth},
                               native_error=native_error)[0]

    def test_the_budget_table(self):
        self.assertEqual(NATIVE_FACTOR, 2.0)
        self.assertEqual(MODEL_POLICY, "evograd-gate1-model-dtype-budget/10")
        self.assertNotIn("atol", model_policy_identity())
        self.assertEqual(BUDGETS, {torch.float16: 5e-3, BF16: 4e-2, torch.float32: 1e-5,
                                   torch.float64: 1e-7})

    def test_every_element_gets_the_same_allowance_from_the_largest_magnitude(self):
        ref = torch.tensor([1.0, 0.0], dtype=BF16)
        check = self.compare(ref, torch.tensor([1.0, 0.0390625], dtype=BF16))      # 0.039 < 0.04
        self.assertEqual(check.status, "passed")
        self.assertEqual(check.limits, {"dtype": "bfloat16", "budget": 4e-2,
                                        "native_error": 0.0, "native_bound": False, "bound": 4e-2,
                                        "allowance": 4e-2})
        self.assertEqual(self.compare(ref, torch.tensor([1.0, 0.04296875], dtype=BF16)).status,
                         "failed")
        # The large element has the same allowance, not one relative to itself.
        self.assertEqual(self.compare(ref, torch.tensor([1.0390625, 0.0], dtype=BF16)).status,
                         "passed")
        self.assertEqual(self.compare(ref, torch.tensor([1.046875, 0.0], dtype=BF16)).status,
                         "failed")

    def test_numbers_are_judged_against_the_truth_not_native(self):
        # Native is 3% off the truth on its largest element; a candidate equal to the truth
        # passes, one 3% off on the other side (6% from native, 3% from the truth) passes,
        # and native scaled by 1.05 (8% from the truth) fails.
        truth = torch.tensor([1.0, -0.5, 0.25])
        native = torch.tensor([1.03125, -0.5, 0.25], dtype=BF16)
        self.assertEqual(self.compare(native, truth.to(BF16), truth=truth).status, "passed")
        other = torch.tensor([0.96875, -0.5, 0.25], dtype=BF16)
        self.assertEqual(self.compare(native, other, truth=truth).status, "passed")
        self.assertEqual(self.compare(native, (native.float() * 1.05).to(BF16), truth=truth).status,
                         "failed")

    def test_the_bound_is_never_below_twice_natives_own_error(self):
        self.assertEqual(result_bound(4e-2, 2e-2), 4e-2)
        self.assertAlmostEqual(result_bound(4e-2, 3e-2), 6e-2)       # continuous: no step at the budget
        self.assertAlmostEqual(result_bound(4e-2, 8e-2), 0.16)
        truth = torch.tensor([1.0, -0.5, 0.25])
        native = torch.tensor([1.078125, -0.5, 0.25], dtype=BF16)                 # 7.8% off
        cand = torch.tensor([0.90625, -0.5, 0.25], dtype=BF16)                    # 9.4% off
        plain = self.compare(native, cand, truth=truth)
        self.assertEqual(plain.status, "failed")
        widened_check = self.compare(native, cand, truth=truth, native_error={"out:y": 0.078125})
        self.assertEqual(widened_check.status, "passed")
        self.assertTrue(widened_check.limits["native_bound"])
        self.assertAlmostEqual(widened_check.limits["bound"], 2 * 0.078125)
        far = torch.tensor([0.8125, -0.5, 0.25], dtype=BF16)                      # 18.8% off
        self.assertEqual(self.compare(native, far, truth=truth,
                                      native_error={"out:y": 0.078125}).status, "failed")

    def test_a_tiny_result_is_judged_at_its_own_scale(self):
        # No absolute term: a gradient peaking at 5e-5 scaled by 1.05, or returned as zeros,
        # fails although every difference is below torch's default atol of 1e-5.
        ref = torch.tensor([5e-5, -2e-5, 1e-6, 3e-5], dtype=BF16)
        self.assertEqual(self.compare(ref, (ref.float() * 1.05).to(BF16), name="grad_in:dx").status,
                         "failed")
        self.assertEqual(self.compare(ref, torch.zeros_like(ref), name="grad_in:dx").status,
                         "failed")
        self.assertEqual(self.compare(ref, ref.clone(), name="grad_in:dx").status, "passed")

    def test_an_exactly_zero_truth_passes_only_an_exactly_zero_candidate(self):
        zeros = torch.zeros(3, dtype=torch.float32)
        self.assertEqual(allowance(zeros, 1e-5), 0.0)
        self.assertEqual(self.compare(zeros, zeros.clone()).status, "passed")
        self.assertEqual(self.compare(zeros, torch.tensor([0.0, 1e-30, 0.0])).status, "failed")
        # A residue near zero is judged at its own scale like any other result.
        residue = torch.tensor([3e-16, -1e-14, 0.0])
        self.assertEqual(self.compare(residue, torch.tensor([3e-16, -1.03e-14, 0.0])).status, "passed")
        self.assertEqual(self.compare(residue, torch.tensor([0.0, 0.0, 2e-15])).status, "failed")

    def test_the_budget_follows_the_least_precise_dtype_of_the_call(self):
        op = replace(self.op, args=(Active("x", "[R, C]"), Active("w", "[C]", dtype="float32"),
                                    Inactive("mask", "[R]", dtype="bool"),
                                    Inactive("alpha", None, default=2.0)))
        bf16_call, fp32_call = Workload({"R": 4, "C": 8}, "bfloat16"), Workload({"R": 4, "C": 8}, "float32")
        # A float32 parameter gradient of a bfloat16 call carries bfloat16 rounding.
        self.assertEqual(rule_dtype(op, bf16_call, torch.float32), BF16)
        self.assertEqual(result_budget(op, bf16_call, torch.float32), 4e-2)
        self.assertEqual(rule_dtype(op, fp32_call, torch.float32), torch.float32)
        self.assertEqual(result_budget(op, fp32_call, torch.float32), 1e-5)
        # A bfloat16 result of a float32 call is judged at bfloat16.
        self.assertEqual(rule_dtype(op, fp32_call, BF16), BF16)

    def test_the_truth_widens_low_precision_inputs_and_native_error_measures_native(self):
        values = {"x": torch.randn(4, 8, dtype=BF16), "w": torch.randn(8), "k": 3}
        wide, changed = widened(values)
        self.assertTrue(changed)
        self.assertEqual((wide["x"].dtype, wide["w"].dtype, wide["k"]), (torch.float32, torch.float32, 3))
        self.assertFalse(widened({"w": torch.randn(8)})[1])
        # A float32 call has nothing to widen: the truth is native and its own error is 0.
        cases = F.supplied_cases(self.op)
        native = {"out:hi": torch.ones(2)}
        self.assertIs(truth_results(self.op, dict(cases[0].values), "cpu", native), native)
        self.assertTrue(all(v == 0.0 for v in native_errors(self.op, cases, "cpu").values()))

    def test_the_truth_is_computed_after_the_candidate(self):
        # The truth never coexists with the candidate's working state; given native's
        # error (a cache), Gate 1 skips its own pre-pass.
        events = []
        pair = F.split_pair()
        forward = pair.forward_with_saved

        def forward_with_saved(x, w, alpha=2.0):
            events.append("candidate")
            return forward(x, w, alpha)

        pair.forward_with_saved = forward_with_saved
        real_truth = runner.truth_results

        def truth(*args, **kwargs):
            events.append("truth")
            return real_truth(*args, **kwargs)

        cases = F.supplied_cases(self.op, [Workload({"R": 4, "C": 8}, "bfloat16")])
        with mock.patch.object(runner, "truth_results", truth):
            report = gate1.verify_pair(self.op, pair, cases=cases, device="cpu")
            self.assertEqual(report.status, "passed")
            self.assertEqual(events, ["truth", "candidate", "truth"])      # pre-pass, then the case
            events.clear()
            cached = gate1.verify_pair(self.op, pair, cases=cases, device="cpu",
                                       native_error=report.subject["native_error"])
        self.assertEqual(events, ["candidate", "truth"])
        self.assertEqual(cached.status, "passed")
        self.assertEqual(cached.subject["native_error"], report.subject["native_error"])

    def test_native_error_is_cached_for_the_same_cases_and_recomputed_otherwise(self):
        from evograd.verification.integrations.gates import gate1_native_errors

        bf16 = F.supplied_cases(self.op, [Workload({"R": 4, "C": 8}, "bfloat16")])
        self.assertIsNone(gate1_native_errors(self.op, bf16, "cpu", None))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "native_error.json"
            first = gate1_native_errors(self.op, bf16, "cpu", path)
            self.assertEqual(first, native_errors(self.op, bf16, "cpu"))
            self.assertGreater(max(first.values()), 0.0)
            with mock.patch.object(runner, "native_errors", side_effect=AssertionError("recomputed")):
                self.assertEqual(gate1_native_errors(self.op, bf16, "cpu", path), first)
            # Other cases do not reuse it.
            other = [replace(bf16[0], case_id="vt_split/other")]
            with mock.patch.object(runner, "native_errors", return_value={"out:hi": 0.5}) as compute:
                self.assertEqual(gate1_native_errors(self.op, other, "cpu", path), {"out:hi": 0.5})
            compute.assert_called_once()
            self.assertEqual(json.loads(path.read_text())["binding"]["cases"][0]["case_id"],
                             "vt_split/other")
            # A case whose reference fails is never cached: Gate 1 computes it and reports why.
            path.unlink()
            with mock.patch.object(runner, "reference_results", side_effect=RuntimeError("no")):
                self.assertIsNone(gate1_native_errors(self.op, bf16, "cpu", path))
                with self.assertRaises(RuntimeError):
                    native_errors(self.op, bf16, "cpu", strict=True)
                self.assertEqual(native_errors(self.op, bf16, "cpu"), {})
            self.assertFalse(path.exists())

    def test_unknown_model_result_dtype_is_not_assigned_a_guessed_rule(self):
        with self.assertRaisesRegex(ValueError, "unsupported result dtype"):
            result_budget(self.op, self.workload, torch.complex64)

    def test_policy_identity_includes_the_rule_table_and_report_version(self):
        report = gate1.verify_pair(self.op, F.split_pair(), cases=F.supplied_cases(self.op),
                                   device="cpu")
        self.assertEqual(report.status, "passed")
        self.assertEqual(report.protocol, PROTOCOL)
        self.assertEqual(report.subject["tolerance_policy"], model_policy_identity())
        self.assertEqual(report.subject["tolerance_policy"]["schema"], MODEL_POLICY)
        self.assertIn("native_error", report.subject)
        self.assertNotIn("envelope", report.subject)
        for result in report.cases[0].results:
            self.assertEqual((result.limits["dtype"], result.limits["budget"]), ("float32", 1e-5))

    def test_a_declaration_without_a_model_reference_has_no_gate1_rule(self):
        # Every entry refuses it.
        workload = F.SPLIT.benchmark[0]
        for check in (lambda: result_budget(F.SPLIT, workload, torch.float32),
                      lambda: policy_identity(F.SPLIT)):
            with self.assertRaisesRegex(ValueError, "model-reference"):
                check()


if __name__ == "__main__":
    unittest.main()
