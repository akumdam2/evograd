"""The canonical report boundary. Pure dict-to-dataclass work, no GPU.

These pin the rule that exists because the suite once published a saved-memory
aggregate of 1.000 for every operator: a missing measurement must raise rather
than become a zero, and ratios are derived from times, never read.
"""

from __future__ import annotations

import unittest

from evograd.evaluation.common.report import (
    PROTOCOL_FAIR,
    TIER_OPERATOR,
    BenchReport,
    CaseMetrics,
    ReportFieldError,
    from_tier2_report,
)


def _tier2_provider(*, forward, full, peak=1024.0, ok=True, error=None):
    if not ok:
        return {"ok": False, "error": error or "boom", "kind": "candidate_pair"}
    return {
        "ok": True,
        "kind": "candidate_pair",
        "forward": {"median_ms": forward, "q20_ms": forward, "q80_ms": forward},
        "full_step": {"median_ms": full, "q20_ms": full, "q80_ms": full},
        "peak_memory_bytes": peak,
        "adapter_kind": "candidate_pair_module",
    }


def _tier2_report(*, candidate_full=1.0, eager_full=2.0, candidate_ok=True):
    return {
        "environment": {"gpu_name": "GH200"},
        "cases": [
            {
                "dims": {"rows": 1024},
                "dtype": "bfloat16",
                "providers": {
                    "candidate": _tier2_provider(
                        forward=0.5, full=candidate_full, ok=candidate_ok,
                        error="died at this shape",
                    ),
                    "eager": _tier2_provider(forward=1.0, full=eager_full, peak=2048.0),
                },
            }
        ],
    }


class TestMissingFieldsAreLoud(unittest.TestCase):
    def test_a_missing_nested_measurement_names_the_full_path(self):
        report = _tier2_report()
        del report["cases"][0]["providers"]["candidate"]["full_step"]
        with self.assertRaises(ReportFieldError) as caught:
            from_tier2_report("layernorm", report, baseline="eager")
        self.assertIn("providers.candidate.full_step", str(caught.exception))

    def test_naming_the_wrong_baseline_provider_raises(self):
        # The historical hazard: guessing the provider key would have reported
        # another provider's measurement, plausibly.
        with self.assertRaises(ReportFieldError):
            from_tier2_report("layernorm", _tier2_report(), baseline="cublas_pair")


class TestRatiosAreDerived(unittest.TestCase):
    def test_speedup_comes_from_times_not_from_a_key(self):
        metrics = CaseMetrics(
            dims={"rows": 8}, dtype="bfloat16", ok=True,
            candidate_full_ms=1.0, baseline_full_ms=2.0,
            candidate_backward_ms=1.0, baseline_backward_ms=10.0,
        )
        self.assertAlmostEqual(metrics.speedup_full, 2.0)
        self.assertAlmostEqual(metrics.speedup_backward, 10.0)

    def test_a_case_that_did_not_run_has_no_speedup(self):
        metrics = CaseMetrics(dims={"rows": 8}, dtype="bfloat16", ok=False)
        self.assertIsNone(metrics.speedup_full)

    def test_a_zero_candidate_time_does_not_divide_by_zero(self):
        metrics = CaseMetrics(
            dims={"rows": 8}, dtype="bfloat16", ok=True,
            candidate_full_ms=0.0, baseline_full_ms=1.0,
        )
        self.assertIsNone(metrics.speedup_full)


class TestTierTwo(unittest.TestCase):
    def test_a_tier2_case_is_read_against_the_named_baseline(self):
        report = from_tier2_report(
            "layernorm", _tier2_report(candidate_full=1.0, eager_full=2.0), baseline="eager")
        self.assertAlmostEqual(report.cases[0].speedup_full, 2.0)
        self.assertEqual((report.tier, report.protocol), (TIER_OPERATOR, PROTOCOL_FAIR))
        self.assertEqual(report.baseline, "eager")

    def test_one_report_can_be_read_against_several_baselines(self):
        # Three providers measured once; each pairing is a different reading
        # at no extra GPU cost.
        raw = _tier2_report()
        raw["cases"][0]["providers"]["cublas_pair"] = _tier2_provider(forward=0.8, full=1.5)
        against_eager = from_tier2_report("layernorm", raw, baseline="eager")
        against_pair = from_tier2_report("layernorm", raw, baseline="cublas_pair")
        self.assertAlmostEqual(against_eager.cases[0].speedup_full, 2.0)
        self.assertAlmostEqual(against_pair.cases[0].speedup_full, 1.5)

    def test_a_provider_that_failed_has_no_speedup(self):
        report = from_tier2_report("layernorm", _tier2_report(candidate_ok=False),
                                   baseline="eager")
        case = report.cases[0]
        self.assertFalse(case.ok)
        self.assertIsNone(case.speedup_full)
        self.assertIn("candidate", case.error)
        self.assertEqual(report.ok_cases, ())

    def test_tier2_reports_no_backward_only_metric(self):
        # The autograd engine exposes no backward-only region; `.backward()` is
        # the whole step. A reader that invented one would be reporting a
        # number nothing measured.
        report = from_tier2_report("layernorm", _tier2_report(), baseline="eager")
        self.assertIsNone(report.cases[0].speedup_backward)

    def test_peak_memory_travels(self):
        report = from_tier2_report("layernorm", _tier2_report(), baseline="eager")
        self.assertAlmostEqual(report.cases[0].candidate_peak_memory_bytes, 1024.0)
        self.assertAlmostEqual(report.cases[0].baseline_peak_memory_bytes, 2048.0)


class TestTier(unittest.TestCase):
    def test_an_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            BenchReport(op="swiglu", tier="level2", protocol=PROTOCOL_FAIR, baseline="cublas_pair")


if __name__ == "__main__":
    unittest.main()
