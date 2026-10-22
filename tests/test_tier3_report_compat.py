"""Reading tier-3 reports: the current protocol only, nothing numeric by default.

``tests/fixtures/tier3/model_report_v3.json`` is a whole-model report
(benchmark level 3) as ``assemble_report`` writes it: identity fields, one
failed provider, and no saved-state numbers (the model runner cannot see them).
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evograd.evaluation.common.report import ReportFieldError
from evograd.evaluation.tier3.report import (
    MODEL_LEVEL,
    SCOPE_MODEL,
    TIER3_MODEL_PROTOCOL_VERSION,
    UnknownTier3Report,
    group_reports,
    identify_report,
    provider_rows,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tier3"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestModelReport(unittest.TestCase):
    def setUp(self):
        self.report = _load("model_report_v3.json")

    def test_it_is_model_scope_level_three(self):
        identity = identify_report(self.report)
        self.assertEqual(identity.protocol, TIER3_MODEL_PROTOCOL_VERSION)
        self.assertEqual(identity.execution_scope, SCOPE_MODEL)
        self.assertEqual((identity.benchmark_level, MODEL_LEVEL), (3, 3))
        self.assertEqual(identity.evaluation_tier, 3)
        self.assertEqual(identity.workload, "sample_decoder")
        self.assertEqual(identity.timing_boundary, "loss.backward() + optimizer step")

    def test_undeclared_candidate_levels_are_empty_not_invented(self):
        self.assertEqual(identify_report(self.report).candidate_task_levels, ())

    def test_rows_read_the_training_step_latency_and_nothing_unmeasured(self):
        rows = {row.provider: row for row in provider_rows(self.report)}
        self.assertEqual(rows["eager"].latency_ms, 100.0)
        self.assertEqual(rows["candidate"].latency_ms, 80.0)
        self.assertAlmostEqual(rows["candidate"].speedup_vs_reference, 1.25)
        self.assertEqual(rows["candidate"].execution_peak_bytes, 17_000_000_000)
        # The model runner cannot observe saved state or a validation peak.
        # Those are None, never 0.
        self.assertIsNone(rows["candidate"].saved_state_bytes)
        self.assertIsNone(rows["candidate"].validation_peak_bytes)
        self.assertEqual(rows["candidate"].patched, ("rms_norm", "swiglu"))

    def test_a_failed_provider_keeps_its_reason_and_no_numbers(self):
        rows = {row.provider: row for row in provider_rows(self.report)}
        failing = rows["failing"]
        self.assertFalse(failing.ok)
        self.assertEqual(failing.failed_at, "preflight")
        self.assertIn("PreflightFailure", failing.error)
        self.assertIsNone(failing.latency_ms)
        self.assertIsNone(failing.speedup_vs_reference)
        self.assertIsNone(failing.execution_peak_bytes)

    def test_a_missing_peak_memory_field_is_an_error_not_a_zero(self):
        broken = copy.deepcopy(self.report)
        del broken["providers"]["eager"]["peak_memory_bytes"]
        with self.assertRaises(ReportFieldError) as caught:
            provider_rows(broken)
        self.assertIn("peak_memory_bytes", str(caught.exception))

    def test_a_null_peak_memory_reads_as_not_measured(self):
        """A CPU run of the model runner records None; that is a real value."""
        cpu = copy.deepcopy(self.report)
        cpu["providers"]["eager"]["peak_memory_bytes"] = None
        rows = {row.provider: row for row in provider_rows(cpu)}
        self.assertIsNone(rows["eager"].execution_peak_bytes)

    def test_reports_of_one_workload_group_together(self):
        groups = group_reports([self.report, copy.deepcopy(self.report)])
        self.assertEqual(list(groups), ["tier3/model/sample_decoder"])
        self.assertEqual(len(groups["tier3/model/sample_decoder"]), 2)

    def test_identity_fields_are_required(self):
        for field in ("evaluation_tier", "execution_scope", "benchmark_level"):
            broken = dict(self.report)
            del broken[field]
            with self.assertRaises(ReportFieldError, msg=field):
                identify_report(broken)

    def test_a_report_claiming_another_level_or_scope_is_refused(self):
        for override in ({"benchmark_level": 4}, {"execution_scope": "block"}):
            with self.assertRaises(UnknownTier3Report, msg=override):
                identify_report(dict(self.report, **override))


class TestCurrentRunner(unittest.TestCase):
    def test_assemble_report_states_model_scope_level_three(self):
        from evograd.evaluation.tier3.runner import assemble_report
        from tests._registry_fixture import SAMPLE_SITES

        class Workload:
            name = "sample"
            unit_name = "tokens"
            site_registry = SAMPLE_SITES
            loss_delta_threshold = None

            def units_per_step(self):
                return 16

            def describe(self):
                return {"workload": "sample_decoder", "name": self.name}

        report = assemble_report(
            Workload(), {}, [], warmup=1, steps=1, blocks=1, loss_steps=1,
            learning_rate=1e-4, seed=0, verify=True, isolation="test",
        )
        self.assertEqual(report["protocol"], TIER3_MODEL_PROTOCOL_VERSION)
        self.assertEqual(report["execution_scope"], SCOPE_MODEL)
        self.assertEqual(report["benchmark_level"], 3)
        self.assertEqual(report["evaluation_tier"], 3)
        self.assertEqual(report["timing_protocol"]["boundary"],
                         "loss.backward() + optimizer step")
        self.assertEqual(identify_report(report).benchmark_level, 3)


class TestOtherProtocols(unittest.TestCase):
    def test_earlier_and_unknown_protocols_are_refused_by_name(self):
        for protocol in ("evograd-tier3-model-v2", "evograd-tier3-block-v1",
                         "evograd-tier3-something-v9"):
            with self.assertRaises(UnknownTier3Report) as caught:
                identify_report({"protocol": protocol})
            self.assertIn(protocol, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
