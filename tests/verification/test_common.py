"""common/: contracts, name matching, isolation, the two comparators, reports."""

from __future__ import annotations

import json
import math
import unittest

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common import reports as R
from evograd.verification.common import tensors as T


class TestMatching(unittest.TestCase):
    def contract(self):
        return C.ResultContract((
            C.ResultSpec("out:y", C.OUTPUT),
            C.ResultSpec("grad:w", C.PARAM_GRAD),
            C.ResultSpec("grad:unused", C.PARAM_GRAD, presence=C.ABSENT),
            C.ResultSpec("out:opt", C.OUTPUT, presence=C.OPTIONAL),
        ))

    def test_missing_candidate_result_fails_and_is_never_truncated(self):
        t = torch.ones(2)
        pairs, findings = C.match_results(self.contract(), {"out:y": t},
                                          {"out:y": t, "grad:w": t})
        self.assertEqual([p.spec.name for p in pairs], ["out:y"])
        self.assertEqual([(f.code, f.outcome, f.subject) for f in findings],
                         [("missing_result", R.FAILED, "grad:w")])

    def test_reference_without_required_result_is_incomplete(self):
        t = torch.ones(2)
        _, findings = C.match_results(self.contract(), {"out:y": t, "grad:w": t}, {"out:y": t})
        self.assertEqual([(f.code, f.outcome) for f in findings],
                         [("reference_missing_result", R.INCOMPLETE)])

    def test_absent_and_unknown_results(self):
        t = torch.ones(2)
        _, findings = C.match_results(self.contract(),
                                      {"out:y": t, "grad:w": t, "grad:unused": t, "out:extra": t},
                                      {"out:y": t, "grad:w": t})
        codes = sorted((f.code, f.subject) for f in findings)
        self.assertEqual(codes, [("unexpected_result", "grad:unused"),
                                 ("unexpected_result", "out:extra")])

    def test_optional_result(self):
        t = torch.ones(2)
        base = {"out:y": t, "grad:w": t}
        self.assertEqual(C.match_results(self.contract(), base, base)[1], [])
        _, findings = C.match_results(self.contract(), base, {**base, "out:opt": t})
        self.assertEqual([f.code for f in findings], ["missing_result"])

    def test_contract_rejects_duplicate_names(self):
        with self.assertRaises(ValueError):
            C.ResultContract((C.ResultSpec("a", C.OUTPUT), C.ResultSpec("a", C.OUTPUT)))


class TestStructure(unittest.TestCase):
    def check(self, cand, ref, **spec):
        return C.check_structure(C.Pairing(C.ResultSpec("r", C.OUTPUT, **spec), cand, ref))

    def test_shape_dtype_type(self):
        ref = torch.zeros(2, 3)
        self.assertEqual([f.code for f in self.check(torch.zeros(3, 2), ref).findings],
                         ["shape_mismatch"])
        self.assertEqual([f.code for f in self.check(ref.double(), ref).findings],
                         ["dtype_mismatch"])
        self.assertEqual([f.code for f in self.check(3.0, ref).findings], ["type_mismatch"])

    def test_non_finite_reference_and_candidate_both_recorded(self):
        ref = torch.tensor([1.0, float("nan")])
        cand = torch.tensor([float("inf"), float("nan")])
        findings = self.check(cand, ref).findings
        self.assertEqual(sorted((f.code, f.outcome) for f in findings),
                         [("invalid_reference", R.INCOMPLETE), ("non_finite_candidate", R.FAILED)])

    def test_candidate_not_blamed_where_the_reference_is_non_finite(self):
        ref = torch.tensor([1.0, float("inf")])
        findings = self.check(torch.tensor([1.0, float("inf")]), ref).findings
        self.assertEqual([(f.code, f.outcome) for f in findings],
                         [("invalid_reference", R.INCOMPLETE)])

    def test_non_floating_results_compare_exactly(self):
        ref = torch.tensor([1, 2, 3])
        self.assertEqual(self.check(ref.clone(), ref).mode, "exact")
        self.assertEqual([f.code for f in self.check(torch.tensor([1, 2, 4]), ref).findings],
                         ["exact_mismatch"])
        self.assertEqual([f.code for f in self.check(7, 8).findings], ["exact_mismatch"])
        self.assertEqual(self.check(7, 7).findings, [])

    def test_bitwise_equal_treats_nan_as_equal_bits(self):
        a = torch.tensor([float("nan"), 1.0])
        self.assertTrue(C.bitwise_equal(a, a.clone()))
        self.assertFalse(torch.equal(a, a.clone()))


class TestIsolation(unittest.TestCase):
    def test_isolate_preserves_strides_offsets_and_aliasing(self):
        base = torch.arange(24.0).reshape(4, 6)
        view, row = base[:, 1:4], base[1]
        out = T.isolate({"view": view, "row": row, "scalar": 3, "same": view})
        self.assertEqual(out["view"].stride(), view.stride())
        self.assertEqual(out["view"].storage_offset(), view.storage_offset())
        self.assertTrue(torch.equal(out["view"], view))
        self.assertIs(out["same"], out["view"])
        self.assertEqual(out["view"].untyped_storage().data_ptr(),
                         out["row"].untyped_storage().data_ptr())
        self.assertNotEqual(out["view"].untyped_storage().data_ptr(),
                            view.untyped_storage().data_ptr())
        out["row"].fill_(-1.0)                       # aliasing kept inside the copy...
        self.assertEqual(float(out["view"][1, 0]), -1.0)
        self.assertEqual(float(view[1, 0]), 7.0)     # ...and nothing leaks to the source

    def test_mutation_detection_and_allowed_overwrite(self):
        values = {"x": torch.ones(3), "dy": torch.ones(3)}
        before = T.snapshot(values, metadata_only=("dy",))
        values["dy"].mul_(2)
        self.assertEqual(T.mutations(values, before), [])
        values["x"].add_(1)
        self.assertEqual(T.mutations(values, before), ["x"])

    def test_isolated_rng_restores_and_reproduces(self):
        torch.manual_seed(0)
        state = torch.get_rng_state()
        with T.isolated_rng(5):
            a = torch.rand(3)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        with T.isolated_rng(5):
            b = torch.rand(3)
        self.assertTrue(torch.equal(a, b))


class TestComparators(unittest.TestCase):
    def test_relative_error_definition_in_float64(self):
        r = torch.tensor([3.0, 4.0, 0.0, 0.0], dtype=torch.float64)
        c = r + torch.tensor([0.0, 0.0, 0.1, 0.0], dtype=torch.float64)
        err = T.relative_error(c, r)
        self.assertAlmostEqual(err.E2, 0.1 / 5.0, places=12)
        self.assertAlmostEqual(err.Emax, 0.1 / (5.0 / 2.0), places=12)   # RMS = 5/sqrt(4)
        self.assertFalse(err.bitwise)

    def test_bfloat16_inputs_are_compared_exactly_widened(self):
        r = torch.full((1000,), 1.0, dtype=torch.bfloat16)
        c = r.clone()
        c[0] = 1.0078125                                      # one bf16 ulp above 1
        err = T.relative_error(c, r)
        self.assertAlmostEqual(err.Emax, 0.0078125, places=12)

    def test_zero_and_empty_reference(self):
        zero = torch.zeros(4)
        self.assertEqual(T.relative_error(zero, zero).E2, 0.0)
        tiny = T.relative_error(torch.full((4,), 1e-20), zero)
        self.assertTrue(tiny.reference_zero)
        self.assertGreater(tiny.E2, 1e9)                       # divided by EPSILON, by definition
        empty = T.relative_error(torch.empty(0), torch.empty(0))
        self.assertTrue(empty.empty)
        self.assertEqual((empty.E2, empty.Emax), (0.0, 0.0))

    def test_chunking_matches_one_pass(self):
        g = torch.Generator().manual_seed(1)
        r = torch.randn(T.CHUNK + 1234, generator=g)
        c = r + 1e-3 * torch.randn(r.shape, generator=g)
        err = T.relative_error(c, r)
        d = (c.double() - r.double())
        self.assertAlmostEqual(err.E2, float(d.norm() / r.double().norm()), places=12)
        self.assertAlmostEqual(err.Emax, float(d.abs().max() / r.double().pow(2).mean().sqrt()),
                               places=12)

    def test_elementwise_is_the_allclose_formula(self):
        r = torch.tensor([1.0, 100.0, 0.0])
        ok = T.elementwise_error(r + torch.tensor([0.0, 0.9, 0.0]), r, atol=0.0, rtol=0.01)
        self.assertTrue(ok.ok)
        bad = T.elementwise_error(r + torch.tensor([0.0, 0.0, 2.0 ** -10]), r, atol=2.0 ** -13,
                                  rtol=0.01)
        self.assertEqual((bad.violations, bad.worst_index), (1, [2]))
        self.assertEqual(bad.headroom, 8.0)
        with self.assertRaises(ValueError):
            T.elementwise_error(r, r, atol=-1.0, rtol=0.0)

    def test_comparators_refuse_non_finite(self):
        bad = torch.tensor([float("inf")])
        with self.assertRaises(ValueError):
            T.relative_error(bad, bad)


class TestReports(unittest.TestCase):
    def test_status_keeps_failure_and_missing_evidence(self):
        status, complete = R.summarize([R.failure("a", "x"), R.missing("b", "y")])
        self.assertEqual((status, complete), (R.FAILED, False))
        self.assertEqual(R.summarize([]), (R.PASSED, True))
        self.assertEqual(R.summarize([R.missing("b", "y")]), (R.INCOMPLETE, False))

    def test_empty_report_is_incomplete(self):
        report = R.VerificationReport("gate1", "p", {})
        self.assertEqual(report.status, R.INCOMPLETE)
        self.assertIn("no_cases", report.to_dict()["incomplete_because"])

    def test_result_never_passes_without_comparison(self):
        check = R.ResultCheck("r", C.OUTPUT)
        self.assertEqual(check.status, R.INCOMPLETE)

    def test_json_never_carries_nan(self):
        report = R.VerificationReport("gate2", "p", {"x": float("nan")})
        case = R.CaseReport("c")
        case.results.append(R.ResultCheck("r", C.OUTPUT, metrics={"E2": float("inf")}, compared=True))
        report.cases.append(case)
        payload = report.to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload["subject"]["x"])
        self.assertIn("$.subject.x", payload["non_finite_values_removed"])

    def test_report_first_does_not_change_the_verdict(self):
        report = R.VerificationReport("gate2", "p", {})
        case = R.CaseReport("c")
        case.findings.append(R.failure("threshold_exceeded", "too big"))
        report.cases.append(case)
        self.assertFalse(R.continue_after(report, R.STRICT))
        self.assertTrue(R.continue_after(report, R.REPORT_FIRST))
        self.assertEqual(report.status, R.FAILED)
        case.findings.append(R.failure("candidate_runtime_error", "boom"))
        self.assertFalse(R.continue_after(report, R.REPORT_FIRST))


class TestEnvironment(unittest.TestCase):
    def test_fingerprint_is_jsonable_and_differences_are_named(self):
        env = C.environment_fingerprint("cpu")
        json.dumps(env, allow_nan=False)
        changed = dict(env, torch="0.0")
        self.assertTrue(any(d.startswith("torch:") for d in C.environment_differences(env, changed)))
        self.assertTrue(math.isfinite(len(C.stable_hash(env))))


if __name__ == "__main__":
    unittest.main()
