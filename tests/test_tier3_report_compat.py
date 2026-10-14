"""Tier-3 reports across the 2026-10-13 renumbering (three benchmark levels).

Fixtures pin the contract:

* ``tests/fixtures/tier3/legacy_model_report_v2.json`` was written by the real
  ``assemble_report`` *before* the explicit scope fields existed, under the v2
  protocol (whole model = level 4). It must keep reading -- as today's level 3,
  with ``renumbered_from=4`` -- and must not acquire numbers it never measured.
* ``tests/fixtures/tier3/block_report_example.json`` is a report of the removed
  block scope. It must be refused, never read as a measurement.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evograd.evaluation.common.report import ReportFieldError
from evograd.evaluation.tier3.report import (
    MODEL_LEVEL,
    REMOVED_BLOCK_PROTOCOL,
    SCOPE_MODEL,
    TIER3_MODEL_PROTOCOL_V2,
    TIER3_MODEL_PROTOCOL_VERSION,
    UnknownTier3Report,
    group_reports,
    identify_report,
    provider_rows,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tier3"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestLegacyModelReportStillReads(unittest.TestCase):
    """A v2 report from before the scope fields existed."""

    def setUp(self):
        self.report = _load("legacy_model_report_v2.json")

    def test_the_fixture_really_predates_the_fields(self):
        for field in ("evaluation_tier", "execution_scope", "benchmark_level"):
            self.assertNotIn(field, self.report)
        self.assertEqual(self.report["protocol"], TIER3_MODEL_PROTOCOL_V2)

    def test_it_is_model_scope_level_three_renumbered_from_four(self):
        identity = identify_report(self.report)
        self.assertEqual(identity.execution_scope, SCOPE_MODEL)
        self.assertEqual(identity.benchmark_level, MODEL_LEVEL)
        self.assertEqual(MODEL_LEVEL, 3)
        self.assertEqual(identity.renumbered_from, 4)
        self.assertEqual(identity.evaluation_tier, 3)
        self.assertTrue(identity.legacy)
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
        liger = rows["liger"]
        self.assertFalse(liger.ok)
        self.assertEqual(liger.failed_at, "preflight")
        self.assertIn("PreflightFailure", liger.error)
        self.assertIsNone(liger.latency_ms)
        self.assertIsNone(liger.speedup_vs_reference)
        self.assertIsNone(liger.execution_peak_bytes)

    def test_a_v2_report_must_declare_its_own_level(self):
        """A v2 report that claims level 3 is a mislabel, not a renumbering."""
        relabelled = dict(self.report, evaluation_tier=3, execution_scope=SCOPE_MODEL,
                          benchmark_level=3)
        with self.assertRaises(UnknownTier3Report):
            identify_report(relabelled)

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


class TestCurrentModelReports(unittest.TestCase):
    """The runner writes v3: the whole model is benchmark level 3."""

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
        identity = identify_report(report)
        self.assertFalse(identity.legacy)
        self.assertIsNone(identity.renumbered_from)

    def test_a_v3_report_claiming_level_four_is_refused(self):
        legacy = _load("legacy_model_report_v2.json")
        wrong = dict(legacy, protocol=TIER3_MODEL_PROTOCOL_VERSION, evaluation_tier=3,
                     execution_scope=SCOPE_MODEL, benchmark_level=4)
        with self.assertRaises(UnknownTier3Report):
            identify_report(wrong)

    def test_v2_and_v3_reports_of_one_workload_group_together(self):
        """The renumbering must not split old and new reports of the same workload."""
        legacy = _load("legacy_model_report_v2.json")
        declared = dict(legacy, protocol=TIER3_MODEL_PROTOCOL_VERSION, evaluation_tier=3,
                        execution_scope=SCOPE_MODEL, benchmark_level=3)
        groups = group_reports([legacy, declared])
        self.assertEqual(list(groups), ["tier3/model/sample_decoder"])
        self.assertEqual([i.renumbered_from for i in groups["tier3/model/sample_decoder"]],
                         [4, None])


class TestRemovedBlockScope(unittest.TestCase):
    def test_a_block_report_is_refused_by_name(self):
        block = _load("block_report_example.json")
        self.assertEqual(block["protocol"], REMOVED_BLOCK_PROTOCOL)
        with self.assertRaises(UnknownTier3Report) as caught:
            identify_report(block)
        self.assertIn("removed", str(caught.exception))
        with self.assertRaises(UnknownTier3Report):
            provider_rows(block)

    def test_an_unknown_protocol_is_refused_by_name(self):
        with self.assertRaises(UnknownTier3Report) as caught:
            identify_report({"protocol": "evograd-tier3-something-v9"})
        self.assertIn("evograd-tier3-something-v9", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
