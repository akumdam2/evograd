"""The canonical report boundary. Pure dict-to-dataclass work, no GPU.

These pin the two rules that exist because the suite once published a
saved-memory aggregate of 1.000 for every operator: a missing measurement must
raise rather than become a zero, and two protocols reporting the same
measurement must produce the same suite row.
"""

from __future__ import annotations

import unittest

from evograd.bench.report import (
    PROTOCOL_FAIR,
    PROTOCOL_FAST,
    TIER_OPERATOR,
    TIER_PAIR,
    BenchReport,
    CaseMetrics,
    ReportFieldError,
    from_fair_report,
    from_harness_report,
    from_tier2_report,
)
from evograd.bench.suite import task_from_report


def _harness_case(*, full=2.0, backward=10.0, saved=100.0, inputs=200.0):
    """Candidate full step of 1 ms, so the baseline time *is* the speedup."""
    return {
        "dims": {"rows": 1024},
        "dtype": "bfloat16",
        "ok": True,
        "forward_ms": 0.5,
        "backward_from_saved_ms": 1.0,
        "raw_forward_backward_full_step_ms": 1.0,
        "baseline_backward_ms": backward,
        "baseline_raw_full_step_ms": full,
        "saved_bytes": saved,
        "input_bytes": inputs,
    }


def _harness_report(cases, baseline="liger"):
    return {"performance_baseline": baseline, "cases": cases, "error": None}


def _provider(*, forward, backward, full, saved):
    return {
        "forward": {"median_ms": forward},
        "backward": {"median_ms": backward},
        "pair_full": {"median_ms": full},
        "saved_state": {"logical_saved_bytes": saved},
    }


def _fair_report(*, candidate_saved=100.0, baseline_saved=999.0, baseline="liger"):
    return {
        "environment": {"gpu_name": "GH200"},
        "cases": [
            {
                "dims": {"rows": 1024},
                "dtype": "bfloat16",
                "providers": {
                    "candidate": _provider(
                        forward=0.5, backward=1.0, full=1.0, saved=candidate_saved
                    ),
                    baseline: _provider(
                        forward=1.0, backward=10.0, full=2.0, saved=baseline_saved
                    ),
                },
            }
        ],
    }


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
                        error="failed correctness at this shape",
                    ),
                    "eager": _tier2_provider(forward=1.0, full=eager_full, peak=2048.0),
                },
            }
        ],
    }


class _FakeOp:
    """Just enough declaration for the saved-memory denominator."""

    class _Arg:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    def __init__(self):
        self.args = (self._Arg("x", "[rows]"),)

    def memory_input_names(self):
        return ("x",)


class TestMissingFieldsAreLoud(unittest.TestCase):
    def test_a_missing_measurement_raises_and_names_the_path(self):
        case = _harness_case()
        del case["baseline_raw_full_step_ms"]
        with self.assertRaises(ReportFieldError) as caught:
            from_harness_report("swiglu", _harness_report([case]))
        self.assertIn("baseline_raw_full_step_ms", str(caught.exception))

    def test_a_missing_nested_measurement_names_the_full_path(self):
        report = _fair_report()
        del report["cases"][0]["providers"]["candidate"]["saved_state"]
        with self.assertRaises(ReportFieldError) as caught:
            from_fair_report("swiglu", report, baseline="liger")
        self.assertIn("providers.candidate.saved_state", str(caught.exception))

    def test_the_original_bug_now_raises_instead_of_publishing_1_000(self):
        """The exact drift that shipped: fair reports carry `logical_saved_bytes`,
        the adapter asked for the harness's `saved_bytes`, the miss became 0.0,
        and a geometric mean over an all-zero set published 1.000 for every
        operator. Renaming the key back to the harness spelling reproduces the
        drift; the reader must now refuse it rather than average it.
        """
        report = _fair_report()
        saved = report["cases"][0]["providers"]["candidate"]["saved_state"]
        saved["saved_bytes"] = saved.pop("logical_saved_bytes")
        with self.assertRaises(ReportFieldError) as caught:
            from_fair_report("swiglu", report, baseline="liger")
        self.assertIn("logical_saved_bytes", str(caught.exception))

    def test_naming_the_wrong_baseline_provider_raises(self):
        # The historical hazard: guessing the provider key would have reported
        # the baseline's retained memory as the candidate's, plausibly.
        with self.assertRaises(ReportFieldError):
            from_fair_report("swiglu", _fair_report(), baseline="pytorch_autograd")


class TestRatiosAreDerived(unittest.TestCase):
    def test_speedup_comes_from_times_not_from_a_key(self):
        case = _harness_case(full=2.0, backward=10.0)
        case["speedup_vs_baseline_raw_full_step"] = 99.0  # never read
        report = from_harness_report("swiglu", _harness_report([case]))
        self.assertAlmostEqual(report.cases[0].speedup_full, 2.0)
        self.assertAlmostEqual(report.cases[0].speedup_backward, 10.0)

    def test_a_case_that_did_not_run_has_no_speedup(self):
        metrics = CaseMetrics(dims={"rows": 8}, dtype="bfloat16", ok=False)
        self.assertIsNone(metrics.speedup_full)

    def test_a_zero_candidate_time_does_not_divide_by_zero(self):
        metrics = CaseMetrics(
            dims={"rows": 8}, dtype="bfloat16", ok=True,
            candidate_full_ms=0.0, baseline_full_ms=1.0,
        )
        self.assertIsNone(metrics.speedup_full)


class TestTheTwoProtocolsAgree(unittest.TestCase):
    """The regression test for the bug this module exists to prevent."""

    def test_same_measurement_same_suite_row(self):
        fair = task_from_report(
            from_fair_report("swiglu", _fair_report(), baseline="liger", op=_FakeOp()),
            level=1, family="activation",
        )
        harness = task_from_report(
            from_harness_report(
                "swiglu", _harness_report([_harness_case(saved=100.0, inputs=1024 * 2)])
            ),
            level=1, family="activation",
        )
        self.assertAlmostEqual(fair.speedup, harness.speedup)
        self.assertAlmostEqual(fair.saved_bytes, harness.saved_bytes)
        self.assertAlmostEqual(fair.input_bytes, harness.input_bytes)
        self.assertAlmostEqual(fair.saved_memory_ratio, harness.saved_memory_ratio)

    def test_the_fair_reader_takes_the_candidates_memory_not_the_baselines(self):
        report = from_fair_report(
            "swiglu", _fair_report(candidate_saved=100.0, baseline_saved=999.0),
            baseline="liger",
        )
        self.assertAlmostEqual(report.cases[0].saved_bytes, 100.0)


class TestTierTwo(unittest.TestCase):
    def test_a_tier2_case_becomes_an_ordinary_suite_row(self):
        task = task_from_report(
            from_tier2_report(
                "layernorm", _tier2_report(candidate_full=1.0, eager_full=2.0),
                baseline="eager",
            ),
            level=1, family="norm",
        )
        self.assertAlmostEqual(task.speedup, 2.0)
        self.assertEqual(task.tier, TIER_OPERATOR)
        self.assertEqual(task.baseline, "eager")

    def test_one_report_can_be_read_against_several_baselines(self):
        # Three providers measured once; each pairing is a different suite row
        # at no extra GPU cost.
        raw = _tier2_report()
        raw["cases"][0]["providers"]["liger"] = _tier2_provider(forward=0.8, full=1.5)
        against_eager = from_tier2_report("layernorm", raw, baseline="eager")
        against_liger = from_tier2_report("layernorm", raw, baseline="liger")
        self.assertAlmostEqual(against_eager.cases[0].speedup_full, 2.0)
        self.assertAlmostEqual(against_liger.cases[0].speedup_full, 1.5)

    def test_a_provider_that_failed_costs_coverage_not_speedup(self):
        task = task_from_report(
            from_tier2_report("layernorm", _tier2_report(candidate_ok=False), baseline="eager"),
            level=1, family="norm",
        )
        self.assertEqual(task.speedups, ())
        self.assertEqual((task.cases_ok, task.cases_total), (0, 1))
        self.assertIn("candidate", task.case_errors[0])

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


class TestTierAndProtocolTravel(unittest.TestCase):
    def test_a_row_says_how_it_was_measured(self):
        fair = task_from_report(
            from_fair_report("swiglu", _fair_report(), baseline="liger"),
            level=1, family="activation",
        )
        fast = task_from_report(
            from_harness_report("swiglu", _harness_report([_harness_case()])),
            level=1, family="activation",
        )
        self.assertEqual((fair.tier, fair.protocol), (TIER_PAIR, PROTOCOL_FAIR))
        self.assertEqual((fast.tier, fast.protocol), (TIER_PAIR, PROTOCOL_FAST))
        self.assertEqual(fair.level, 1)  # task hierarchy, independent of tier
        self.assertIn("tier", fair.to_dict())
        self.assertIn("protocol", fair.to_dict())

    def test_an_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            BenchReport(op="swiglu", tier="level2", protocol=PROTOCOL_FAIR, baseline="liger")


if __name__ == "__main__":
    unittest.main()
