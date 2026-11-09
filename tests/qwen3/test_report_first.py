"""The Tier-3 enforcement mode: what a numerical mismatch does, and what it must never do.

The property under test is a separation, not a tolerance: a finite numerical
disagreement is *recorded* and (in report-first mode) admissible, while a
missing gradient, a non-finite value or a runtime error is not, in either mode.
The numerical verdict itself is identical under both modes -- nothing here may
turn a numerical failure into a numerical pass, and a comparison that could not
be made is reported unavailable rather than passed. Every Tier-3 report records
the active mode.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from evograd.evaluation.tier3.gate import enforcement as enf


class _Mode(unittest.TestCase):
    """Each test picks its own mode; the process default is restored after."""

    def setUp(self):
        self._saved = os.environ.get(enf.ENV)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(enf.ENV, None)
        else:
            os.environ[enf.ENV] = self._saved


class TestEnforcementModule(_Mode):
    def test_the_default_is_the_historical_behaviour(self):
        os.environ.pop(enf.ENV, None)
        self.assertEqual(enf.active_mode(), enf.STRICT)
        for kind in enf.KINDS:
            self.assertTrue(enf.blocks(kind, enf.STRICT), kind)

    def test_report_first_blocks_only_structural_and_execution(self):
        self.assertFalse(enf.blocks(enf.NUMERICAL, enf.REPORT_FIRST))
        self.assertFalse(enf.blocks(enf.UNAVAILABLE, enf.REPORT_FIRST))
        self.assertTrue(enf.blocks(enf.STRUCTURAL, enf.REPORT_FIRST))
        self.assertTrue(enf.blocks(enf.EXECUTION, enf.REPORT_FIRST))

    def test_a_numerical_mismatch_stays_a_failure_in_both_modes(self):
        findings = [enf.finding("protocol4:kl_mean", enf.NUMERICAL, "kl too large")]
        for mode, admissible in ((enf.STRICT, False), (enf.REPORT_FIRST, True)):
            verdict = enf.summarize(findings, gate="g", active=mode)
            self.assertIs(verdict["ok"], admissible, mode)
            self.assertIs(verdict["numerical_ok"], False, mode)          # never a pass
            self.assertEqual(verdict["numerical_status"], "mismatches_recorded")
            self.assertEqual(verdict["numerical_failed_at"], "protocol4:kl_mean")
            self.assertTrue(verdict["execution_ok"], mode)

    def test_an_unavailable_comparison_is_never_reported_as_a_pass(self):
        verdict = enf.summarize([enf.finding("policy_binding", enf.UNAVAILABLE, "does not bind")],
                                gate="g", active=enf.REPORT_FIRST)
        self.assertTrue(verdict["ok"])                      # execution may continue
        self.assertIsNone(verdict["numerical_ok"])          # but it is not a pass
        self.assertEqual(verdict["numerical_status"], "unavailable")
        self.assertEqual(verdict["numerical_unavailable_at"], ["policy_binding"])
        self.assertFalse(enf.summarize([enf.finding("policy_binding", enf.UNAVAILABLE, "x")],
                                       gate="g", active=enf.STRICT)["ok"])

    def test_structural_and_execution_failures_are_not_numerical(self):
        for kind in (enf.STRUCTURAL, enf.EXECUTION):
            verdict = enf.summarize([enf.finding("live_boundary", kind, "missing gradient")],
                                    gate="g", active=enf.REPORT_FIRST)
            self.assertFalse(verdict["ok"], kind)
            self.assertFalse(verdict["execution_ok"], kind)
            self.assertEqual(verdict["execution_failed_at"], "live_boundary")
            self.assertIs(verdict["numerical_ok"], True)   # nothing numerical was found
            self.assertEqual(verdict["numerical_status"], "within_limits")

    def test_clean_run_and_unknown_kinds(self):
        verdict = enf.summarize([], gate="g", active=enf.REPORT_FIRST)
        self.assertTrue(verdict["ok"] and verdict["execution_ok"] and verdict["numerical_ok"])
        self.assertEqual(verdict["numerical_status"], "within_limits")
        with self.assertRaises(ValueError):
            enf.finding("s", "sort_of_wrong", "r")
        with self.assertRaises(ValueError):
            enf.mode("lenient")

    def test_the_mode_is_inherited_through_the_environment(self):
        enf.set_active_mode("report-first")
        self.assertEqual(os.environ[enf.ENV], enf.REPORT_FIRST)
        self.assertEqual(enf.active_mode(), enf.REPORT_FIRST)


class TestClassifiers(_Mode):
    def test_a_boundary_failure_is_numerical_only_when_nothing_else_went_wrong(self):
        numerical = {"numerical_only": True, "failure_count": 3}
        self.assertEqual(enf.boundary_finding_kind(numerical), enf.NUMERICAL)
        self.assertEqual(enf.boundary_finding_kind({**numerical, "non_finite_results": ["a"]}), enf.EXECUTION)
        self.assertEqual(enf.boundary_finding_kind({**numerical, "errors": ["shadow raised"]}), enf.EXECUTION)
        self.assertEqual(enf.boundary_finding_kind({**numerical, "missing_or_extra": {"attention": 1},
                                                 "numerical_only": False}), enf.STRUCTURAL)
        self.assertEqual(enf.boundary_finding_kind({"numerical_only": False}), enf.STRUCTURAL)

    def test_a_preflight_failure_is_numerical_only_when_no_case_raised(self):
        tolerance = {"ok": False, "detail": {"failures": [{"error": None, "failed": ["dq"]}]}}
        raised = {"ok": False, "detail": {"failures": [{"error": "Traceback...", "failed": []}]}}
        self.assertTrue(enf.preflight_numerical_only(tolerance))
        self.assertFalse(enf.preflight_numerical_only(raised))
        self.assertFalse(enf.preflight_numerical_only({"ok": False}))          # unknown -> not numerical


class TestReportStatesTheMode(_Mode):
    def test_the_report_states_which_mode_produced_the_verdicts(self):
        from evograd.evaluation.tier3.runner import verification_policy

        enf.set_active_mode(enf.REPORT_FIRST)
        policy = verification_policy(mock.Mock(loss_delta_threshold=None))
        self.assertEqual(policy["numerical_enforcement"]["mode"], enf.REPORT_FIRST)
        self.assertIn("recorded", policy["numerical_enforcement"]["rule"])


class TestTier2IsUntouchedByTheMode(_Mode):
    """The Tier-3 enforcement mode is not a Tier-2 or Tier-1 concept."""

    def test_the_tier2_runner_never_reads_the_enforcement_variable(self):
        import inspect

        from evograd.evaluation.tier2 import runner

        self.assertNotIn(enf.ENV, inspect.getsource(runner))
        self.assertNotIn("enforcement", inspect.getsource(runner))


if __name__ == "__main__":
    unittest.main()
