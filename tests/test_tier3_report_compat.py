"""Tier-3 reports: the model scope that exists and the block scope that is designed.

Two fixtures pin the contract from both sides:

* ``tests/fixtures/tier3/legacy_model_report_v2.json`` was written by the real
  ``assemble_report`` *before* the explicit scope fields existed. It must keep
  reading as a whole-model (level 4) report, must never be reinterpreted as a
  block measurement, and must not acquire numbers it never measured.
* ``tests/fixtures/tier3/block_report_example.json`` is the block-scope report
  ``evograd.evaluation.tier3.report`` describes. The reader is fixed before a
  runner emits it, so the runner is written to a format that is already read.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from evograd.evaluation.common.report import ReportFieldError
from evograd.evaluation.tier3.report import (
    SCOPE_BLOCK,
    SCOPE_MODEL,
    TIER3_BLOCK_PROTOCOL_VERSION,
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
    """A report from before the scope fields existed."""

    def setUp(self):
        self.report = _load("legacy_model_report_v2.json")

    def test_the_fixture_really_predates_the_fields(self):
        for field in ("evaluation_tier", "execution_scope", "benchmark_level"):
            self.assertNotIn(field, self.report)
        self.assertEqual(self.report["protocol"], TIER3_MODEL_PROTOCOL_VERSION)

    def test_it_is_model_scope_level_four_and_marked_inferred(self):
        identity = identify_report(self.report)
        self.assertEqual(identity.execution_scope, SCOPE_MODEL)
        self.assertEqual(identity.benchmark_level, 4)
        self.assertEqual(identity.evaluation_tier, 3)
        self.assertTrue(identity.legacy)
        self.assertIsNone(identity.case)
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

    def test_a_legacy_report_cannot_be_relabelled_as_a_block(self):
        relabelled = dict(self.report, execution_scope=SCOPE_BLOCK, benchmark_level=3)
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


class TestCurrentModelReportsDeclareTheirScope(unittest.TestCase):
    """The runner now writes the fields the legacy reader infers."""

    def test_assemble_report_states_model_scope(self):
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
        self.assertEqual(report["execution_scope"], SCOPE_MODEL)
        self.assertEqual(report["benchmark_level"], 4)
        self.assertEqual(report["evaluation_tier"], 3)
        self.assertEqual(report["timing_protocol"]["boundary"],
                         "loss.backward() + optimizer step")
        identity = identify_report(report)
        self.assertFalse(identity.legacy)
        self.assertEqual(identity.execution_scope, SCOPE_MODEL)

    def test_a_declared_model_report_still_groups_with_the_legacy_one(self):
        """Declaring the field must not split old and new model reports of the
        same workload into different aggregation groups."""
        legacy = _load("legacy_model_report_v2.json")
        declared = dict(legacy, evaluation_tier=3, execution_scope=SCOPE_MODEL,
                        benchmark_level=4)
        groups = group_reports([legacy, declared])
        self.assertEqual(list(groups), ["tier3/model/sample_decoder"])
        self.assertEqual([i.legacy for i in groups["tier3/model/sample_decoder"]],
                         [True, False])


class TestBlockReportExample(unittest.TestCase):
    """The designed block-scope format, read by the same reader."""

    def setUp(self):
        self.report = _load("block_report_example.json")

    def test_it_is_block_scope_level_three_of_level_two_candidates(self):
        identity = identify_report(self.report)
        self.assertEqual(identity.protocol, TIER3_BLOCK_PROTOCOL_VERSION)
        self.assertEqual(identity.execution_scope, SCOPE_BLOCK)
        self.assertEqual(identity.benchmark_level, 3)
        self.assertEqual(identity.candidate_task_levels, (2,))
        self.assertEqual(identity.workload, "qwen3_0_6b")
        self.assertEqual(identity.case, "qwen3_0_6b/decoder_layer@14/captured/bs2.seq2048.bf16")
        self.assertEqual(identity.timing_boundary, "forward + vjp")
        self.assertFalse(identity.legacy)

    def test_the_example_carries_every_field_the_design_requires(self):
        case = self.report["case"]
        for field in ("architecture", "architecture_revision", "block_kind", "block_index",
                      "case_id", "case_hash", "source_mode", "source", "weights_hash",
                      "inputs_hash", "cotangents_hash", "dims", "dtype", "boundary",
                      "sites", "expected_invocations", "excluded", "state_contract",
                      "gradient_presence"):
            self.assertIn(field, case, field)
        self.assertIn("policy_hash", self.report["policy"])
        for name in ("boundary", "cotangent_source", "reset", "backend"):
            self.assertIn(name, self.report["timing_protocol"], name)
        best = self.report["providers"]["candidate:best3"]
        for field in ("patch_coverage", "correctness", "latency", "memory", "kernel_sources"):
            self.assertIn(field, best, field)
        coverage = best["patch_coverage"]
        self.assertEqual(coverage["requested"], coverage["actual"])
        self.assertEqual(coverage["supporting_native"], ["attention"])
        self.assertEqual(sum(coverage["invocations"].values()), 4)
        correctness = best["correctness"]
        for section in ("outputs", "input_gradients", "parameter_gradients",
                        "gradient_presence", "purity", "input_mutation"):
            self.assertIn(section, correctness, section)
        self.assertEqual(correctness["gradient_presence"]["missing"], [])
        for source in best["kernel_sources"]:
            self.assertEqual(source["task_level"], 2)
            self.assertIn("source_hash", source)

    def test_rows_span_forward_plus_vjp_against_the_native_block(self):
        rows = {row.provider: row for row in provider_rows(self.report)}
        self.assertEqual(rows["native"].latency_ms, 11.42)
        self.assertAlmostEqual(rows["candidate:best3"].speedup_vs_reference, 11.42 / 9.88)
        self.assertEqual(rows["candidate:best3"].saved_state_bytes, 872415232)
        self.assertEqual(rows["candidate:best3"].execution_peak_bytes, 3087007744)
        # Not measured for this provider: null in the report, None in the row.
        self.assertIsNone(rows["candidate:best3"].validation_peak_bytes)
        self.assertEqual(rows["native"].validation_peak_bytes, 6442450944)

    def test_a_backward_only_fault_is_a_failed_row_with_its_reason(self):
        rows = {row.provider: row for row in provider_rows(self.report)}
        failed = rows["candidate:attention"]
        self.assertFalse(failed.ok)
        self.assertEqual(failed.failed_at, "block_correctness")
        self.assertIn("backward-only", failed.error)
        self.assertIsNone(failed.latency_ms)
        self.assertIsNone(failed.speedup_vs_reference)
        # The report still shows the forward agreed: a block gate that only
        # compared outputs would have timed this provider.
        correctness = self.report["providers"]["candidate:attention"]["correctness"]
        self.assertTrue(correctness["outputs"]["hidden_states"]["ok"])
        self.assertFalse(correctness["input_gradients"]["hidden_states"]["ok"])

    def test_a_missing_memory_field_is_refused_rather_than_zeroed(self):
        broken = copy.deepcopy(self.report)
        del broken["providers"]["native"]["memory"]["saved_state_bytes"]
        with self.assertRaises(ReportFieldError) as caught:
            provider_rows(broken)
        self.assertIn("saved_state_bytes", str(caught.exception))

    def test_a_block_report_cannot_claim_model_scope_or_level_four(self):
        for override in ({"execution_scope": SCOPE_MODEL}, {"benchmark_level": 4}):
            with self.subTest(override=override):
                with self.assertRaises(UnknownTier3Report):
                    identify_report(dict(self.report, **override))

    def test_the_example_states_its_exclusions_rather_than_claiming_full_coverage(self):
        excluded = self.report["case"]["excluded"]
        self.assertTrue(any("cross-layer" in e for e in excluded))
        self.assertEqual(self.report["case"]["expected_invocations"]["residual_rmsnorm"], 1)


class TestScopesNeverPool(unittest.TestCase):
    def test_block_and_model_reports_of_one_architecture_group_apart(self):
        legacy = _load("legacy_model_report_v2.json")
        block = _load("block_report_example.json")
        # Same architecture label on both sides, deliberately.
        legacy = dict(legacy, workload="qwen3_0_6b")
        groups = group_reports([legacy, block])
        self.assertEqual(sorted(groups), [
            "tier3/block/qwen3_0_6b/qwen3_0_6b/decoder_layer@14/captured/bs2.seq2048.bf16",
            "tier3/model/qwen3_0_6b",
        ])

    def test_an_unknown_protocol_is_refused_by_name(self):
        with self.assertRaises(UnknownTier3Report) as caught:
            identify_report({"protocol": "evograd-tier3-something-v9"})
        self.assertIn("evograd-tier3-something-v9", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
