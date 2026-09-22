"""Report-first Tier-3: what a numerical mismatch does, and what it must never do.

The property under test is a separation, not a tolerance: a finite numerical
disagreement is *recorded* and (in report-first mode) execution continues to
training and timing, while a missing gradient, a non-finite value, an impure
provider, a coverage mismatch or a runtime error stops the provider in either
mode. The numerical verdict itself is identical under both modes -- nothing here
may turn a numerical failure into a numerical pass, and a comparison that could
not be made is reported unavailable rather than passed.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
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
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import _boundary_finding_kind

        numerical = {"numerical_only": True, "failure_count": 3}
        self.assertEqual(_boundary_finding_kind(numerical), enf.NUMERICAL)
        self.assertEqual(_boundary_finding_kind({**numerical, "non_finite_results": ["a"]}), enf.EXECUTION)
        self.assertEqual(_boundary_finding_kind({**numerical, "errors": ["shadow raised"]}), enf.EXECUTION)
        self.assertEqual(_boundary_finding_kind({**numerical, "missing_or_extra": {"attention": 1},
                                                 "numerical_only": False}), enf.STRUCTURAL)
        self.assertEqual(_boundary_finding_kind({"numerical_only": False}), enf.STRUCTURAL)

    def test_a_preflight_failure_is_numerical_only_when_no_case_raised(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import _preflight_numerical_only

        tolerance = {"ok": False, "detail": {"failures": [{"error": None, "failed": ["dq"]}]}}
        raised = {"ok": False, "detail": {"failures": [{"error": "Traceback...", "failed": []}]}}
        self.assertTrue(_preflight_numerical_only(tolerance))
        self.assertFalse(_preflight_numerical_only(raised))
        self.assertFalse(_preflight_numerical_only({"ok": False}))          # unknown -> not numerical

    def test_block_verdict_stages_map_to_kinds_and_unknown_is_structural(self):
        from evograd.evaluation.tier3.gate.block import finding_kind

        self.assertEqual(finding_kind({"failed_at": "envelope"}), "numerical")
        self.assertEqual(finding_kind({"failed_at": "finiteness"}), "execution")
        self.assertEqual(finding_kind({"failed_at": "gradient_presence"}), "structural")
        self.assertEqual(finding_kind({"failed_at": "no_policy"}), "unavailable")
        self.assertEqual(finding_kind({"failed_at": "something_new"}), "structural")


class TestBoundaryRecords(_Mode):
    """Part A keeps every failed invocation, and says which elements disagree."""

    def _entry(self, actual, want, atol, rtol):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import boundary

        return boundary._judge(actual, want, atol, rtol, None, name="out", role="forward_output",
                               reference="declared runtime spelling")

    def test_a_failed_comparison_carries_representative_violating_elements(self):
        want = torch.full((4, 8), 32.0)
        actual = want.clone()
        actual[2, 3] = 40.0
        entry = self._entry(actual, want, 0.02, 0.01)
        self.assertFalse(entry["ok"])
        record = entry["discrepancy"]
        self.assertEqual(record["violations"], 1)
        self.assertEqual(record["shape"], [4, 8])
        self.assertEqual(record["role"], "forward_output")
        worst = record["violating_examples"][0]
        self.assertEqual(worst["coordinate"], [2, 3])
        self.assertAlmostEqual(worst["actual"], 40.0)
        self.assertAlmostEqual(worst["reference"], 32.0)
        self.assertAlmostEqual(worst["allowed"], 0.02 + 0.01 * 32.0)
        self.assertGreater(worst["error_over_allowance"], 1.0)
        self.assertIn("e_rms", record["scale_normalized"])

    def test_a_passing_comparison_carries_no_discrepancy(self):
        want = torch.full((4, 8), 32.0)
        self.assertNotIn("discrepancy", self._entry(want.clone(), want, 0.02, 0.01))

    def test_every_failed_invocation_is_kept_and_classified(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.boundary import BoundaryReport, invocation_id

        report = BoundaryReport()
        for ordinal in range(1, 41):          # more than any preview limit
            identity = invocation_id("attention", ordinal - 1, ordinal)
            report.ids.add(identity)
            report.counts["attention"] = report.counts.get("attention", 0) + 1
            report.invocations.append({
                "id": identity, "site": "attention", "layer": ordinal - 1, "category": None,
                "ordinal": ordinal, "op": "qwen3_attention", "gradients": [],
                "outputs": [{"name": "out", "max_abs_err": 0.5, "rel_l2": 1e-3, "atol": 0.02,
                             "rtol": 0.01, "declared_ok": False, "ok": False,
                             "finite": True}]})
        summary = report.to_dict(expected={"attention": 40})
        self.assertEqual(summary["failure_count"], 40)
        self.assertEqual(len(summary["failures"]), 40)          # nothing silently dropped
        self.assertTrue(summary["numerical_only"])
        self.assertEqual(summary["failures"][0]["role"], "output")
        self.assertEqual(summary["non_finite_results"], [])

    def test_a_non_finite_result_is_not_numerical_only(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.boundary import BoundaryReport

        report = BoundaryReport()
        report.ids.add("attention:layer0:#1")
        report.counts["attention"] = 1
        report.invocations.append({
            "id": "attention:layer0:#1", "site": "attention", "layer": 0, "category": None,
            "ordinal": 1, "op": "qwen3_attention", "gradients": [],
            "outputs": [{"name": "out", "max_abs_err": float("inf"), "rel_l2": float("inf"),
                         "atol": 0.02, "rtol": 0.01, "declared_ok": False, "ok": False,
                         "finite": False}]})
        summary = report.to_dict(expected={"attention": 1})
        self.assertFalse(summary["numerical_only"])
        self.assertEqual(summary["non_finite_results"], ["attention:layer0:#1"])


# ── the hook's control flow, with the heavy stages stubbed ───────────────────

class _Policy:
    schema = "evograd-qwen3-t3-protocol/4"
    training_plan = {"mode": "screening_abc", "steps": 0}
    thresholds = {"kl_mean": 2e-6, "global_grad_rel_l2": 8.4e-3}

    def require_binding(self, **_):
        return None


def _measured(kl=1.1e-3, grad=3.0e-2, **over):
    m = {"kl_mean": kl, "global_grad_rel_l2": grad, "missing_grads": [],
         "grad_presence": {"missing": [], "shape_mismatch": [], "extra": [], "parameters": 310,
                           "compared": 310, "worst_by_squared_error": [], "worst_by_rel_l2": []},
         "finite": {"ok": True}, "non_finite_training_steps": [], "kl_max_position": [0, 1],
         "kl_std": 1e-4, "valid_positions": 4094, "logits_rel_l2_valid": 1.6e-2}
    m.update(over)
    return m


class TestHookControlFlow(_Mode):
    """The whole model-scope hook, with preflight/purity/boundary/captures stubbed."""

    def _workload(self, calibration: Path, **extra):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.protocol4_cli import PRETRAINED, REAL_TEXT
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

        return Qwen3Workload.from_config({
            "dtype": "bfloat16", "device": "cpu", "seed": 0, "data_seed": 0,
            "weights": PRETRAINED, "data": REAL_TEXT,
            "protocol4_calibration_path": str(calibration),
            "protocol4_verdict_path": str(calibration.parent / "holdout.json"), **extra})

    def _run(self, mode, *, local, measured=None, policy=_Policy(), screening=None,
             preflight=None, purity_ok=True, **workload_extra):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import boundary, protocol4, purity
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import simple as S
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W

        enf.set_active_mode(mode)
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "policy.json"
            calibration.write_text(json.dumps({"policy": {}, "local_envelope": None}))
            (Path(tmp) / "holdout.json").write_text(json.dumps(
                {"results": screening if screening is not None else [
                    {"seed": 11, "measured": {"kernel_origin": ["candidate:direct_deployment"]},
                     "verdict": {"ok": True, "ratios": {}}}]}))
            workload = self._workload(calibration, **workload_extra)
            kernels = mock.Mock()
            kernels.patched = ("attention",)
            kernels.sources = [mock.Mock(origin="candidate:direct_deployment")]
            patches = {
                "load_policy": mock.Mock(return_value=policy),
                "capture_first_step": mock.Mock(return_value={"loss": 1.0, "grads": {}, "logits": None}),
                "step_distances": mock.Mock(return_value=dict(measured or _measured())),
            }
            with mock.patch.object(S, "PatchSet", mock.Mock(of=mock.Mock(return_value=mock.Mock()))), \
                 mock.patch.object(W, "_provider_identity", lambda k: [{"site": "attention"}]), \
                 mock.patch.object(protocol4, "load_policy", patches["load_policy"]), \
                 mock.patch.object(protocol4, "capture_first_step", patches["capture_first_step"]), \
                 mock.patch.object(protocol4, "step_distances", patches["step_distances"]), \
                 mock.patch.object(purity, "run_for", mock.Mock(return_value={"ok": purity_ok, "sites": []})), \
                 mock.patch.object(boundary, "validate_all_invocations", mock.Mock(return_value=local)), \
                 mock.patch.object(type(workload), "site_preflight",
                                   lambda self, k, device="cpu": preflight or {"ok": True, "sites": ["attention"]}), \
                 mock.patch.object(type(workload), "batch_for", lambda self, seed=0: (None, None)), \
                 mock.patch.object(type(workload), "data_identity_digest", lambda self: "d"):
                return workload._protocol4_hook(kernels, "cpu")

    NUMERICAL_LOCAL = {"ok": False, "numerical_only": True, "failure_count": 7,
                       "checked_invocations": 56, "failures": [{"id": "attention:layer0:#2", "result": "out", "role": "output",
                                     "max_abs_err": 0.5, "atol": 0.0195, "rtol": 0.01}],
                       "local_check_mode": "declared_or_reference_envelope"}
    CLEAN_LOCAL = {"ok": True, "numerical_only": False, "failure_count": 0,
                   "checked_invocations": 56, "local_check_mode": "declared_or_reference_envelope"}

    def test_report_first_records_the_mismatch_and_proceeds(self):
        verdict = self._run(enf.REPORT_FIRST, local=self.NUMERICAL_LOCAL)
        self.assertTrue(verdict["ok"])                      # training and timing may run
        self.assertTrue(verdict["execution_ok"])
        self.assertIs(verdict["numerical_ok"], False)       # and the failure stands
        self.assertEqual(verdict["numerical_status"], "mismatches_recorded")
        stages = {f["stage"] for f in verdict["findings"]}
        self.assertIn("live_boundary", stages)                       # part A recorded
        self.assertIn("protocol4:kl_mean", stages)                   # part B recorded
        self.assertIn("protocol4:global_grad_rel_l2", stages)        # part C recorded
        self.assertTrue(all(f["kind"] == enf.NUMERICAL for f in verdict["findings"]))
        self.assertEqual(verdict["enforcement"]["mode"], enf.REPORT_FIRST)
        # every comparison ran: parts B and C are measured, not skipped
        self.assertAlmostEqual(verdict["step"]["kl_mean"], 1.1e-3)
        self.assertEqual(len(verdict["numerical_records"]), 2)
        self.assertEqual(verdict["numerical_records"][0]["metric"], "kl_mean")
        self.assertFalse(verdict["numerical_records"][0]["ok"])

    def test_strict_still_stops_at_the_same_mismatch(self):
        verdict = self._run(enf.STRICT, local=self.NUMERICAL_LOCAL)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "live_boundary")
        self.assertIs(verdict["numerical_ok"], False)
        self.assertTrue(verdict["execution_ok"])            # nothing structural was wrong
        self.assertNotIn("step", verdict)                   # stopped before parts B and C

    def test_a_missing_gradient_stops_both_modes(self):
        measured = _measured(missing_grads=["model.layers.0.self_attn.k_proj.weight"],
                             grad_presence={"missing": ["model.layers.0.self_attn.k_proj.weight"],
                                            "shape_mismatch": [], "extra": []})
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, measured=measured)
            self.assertFalse(verdict["ok"], mode)
            self.assertFalse(verdict["execution_ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "protocol4:presence")
            kinds = {f["stage"]: f["kind"] for f in verdict["findings"]}
            self.assertEqual(kinds["protocol4:presence"], enf.STRUCTURAL)

    def test_a_non_finite_result_stops_both_modes(self):
        measured = _measured(finite={"ok": False})
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, measured=measured)
            self.assertFalse(verdict["ok"], mode)
            self.assertFalse(verdict["execution_ok"], mode)
            self.assertEqual({f["kind"] for f in verdict["findings"] if f["stage"] == "protocol4:finite"},
                             {enf.EXECUTION})

    def test_a_non_finite_local_result_stops_both_modes(self):
        local = {**self.NUMERICAL_LOCAL, "numerical_only": False,
                 "non_finite_results": ["attention:layer3:#7"]}
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=local)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "live_boundary")

    def test_an_impure_provider_stops_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, purity_ok=False)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "provider_purity")

    def test_a_kernel_that_raises_at_preflight_stops_both_modes(self):
        raised = {"ok": False, "reason": "PreflightFailure: T <= 128 only",
                  "detail": {"failures": [{"error": "Traceback", "failed": []}]}}
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, preflight=raised)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "site_preflight")

    def test_a_policy_that_does_not_bind_is_unavailable_never_a_pass(self):
        policy = _Policy()
        policy.require_binding = mock.Mock(side_effect=RuntimeError("workload hash differs"))
        strict = self._run(enf.STRICT, local=self.CLEAN_LOCAL, policy=policy)
        self.assertFalse(strict["ok"])
        self.assertEqual(strict["failed_at"], "policy_binding")
        report_first = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL, policy=policy)
        self.assertTrue(report_first["ok"])                 # execution continues
        self.assertIsNone(report_first["numerical_ok"])     # but nothing is claimed
        self.assertEqual(report_first["numerical_status"], "unavailable")
        self.assertFalse(report_first["protocol4"]["available"])
        self.assertNotIn("thresholds", report_first["protocol4"])

    def test_a_failed_frozen_screening_is_numerical_and_still_refused_in_strict(self):
        rows = [{"seed": 11, "measured": {"kernel_origin": ["candidate:direct_deployment"]},
                 "verdict": {"ok": False, "failed_at": "local_A", "reason": "part A failed", "ratios": {}}}]
        strict = self._run(enf.STRICT, local=self.CLEAN_LOCAL, measured=_measured(kl=1e-9, grad=1e-9),
                           screening=rows)
        self.assertFalse(strict["ok"])
        self.assertEqual(strict["failed_at"], "screening_holdout")
        loose = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL,
                          measured=_measured(kl=1e-9, grad=1e-9), screening=rows)
        self.assertTrue(loose["ok"])
        self.assertIs(loose["numerical_ok"], False)

    def test_a_structurally_failed_screening_row_is_not_numerical(self):
        rows = [{"seed": 11, "measured": {"kernel_origin": ["candidate:direct_deployment"]},
                 "verdict": {"ok": False, "failed_at": "invocation_counts", "ratios": {}}}]
        verdict = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL,
                            measured=_measured(kl=1e-9, grad=1e-9), screening=rows)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["execution_failed_at"], "screening_holdout")

    def test_a_missing_screening_verdict_is_unavailable(self):
        verdict = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL,
                            measured=_measured(kl=1e-9, grad=1e-9), screening=[])
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["numerical_status"], "unavailable")
        self.assertIn("screening_holdout", verdict["numerical_unavailable_at"])

    def test_the_historical_diagnostic_timing_escape_still_works(self):
        rows = [{"seed": 11, "measured": {"kernel_origin": ["candidate:direct_deployment"]},
                 "verdict": {"ok": False, "failed_at": "local_A", "ratios": {}}}]
        verdict = self._run(enf.STRICT, local=self.CLEAN_LOCAL, measured=_measured(kl=1e-9, grad=1e-9),
                            screening=rows, protocol4_diagnostic_timing=True)
        self.assertTrue(verdict["ok"])
        self.assertIs(verdict["numerical_ok"], False)
        self.assertIn("diagnostic_only", verdict)

    def test_a_clean_provider_passes_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, measured=_measured(kl=1e-9, grad=1e-9))
            self.assertTrue(verdict["ok"], mode)
            self.assertIs(verdict["numerical_ok"], True, mode)
            self.assertEqual(verdict["numerical_status"], "within_limits")


class TestRunnerStopsOnlyOnAdmissibility(_Mode):
    def _workload(self, verdict):
        class _W:
            def model_correctness(self, kernels, *, device="cpu"):
                return verdict
        return _W()

    def _kernels(self):
        kernels = mock.Mock()
        kernels.patched = ("attention",)
        return kernels

    def test_a_recorded_numerical_mismatch_reaches_training_and_timing(self):
        from evograd.evaluation.tier3.runner import model_correctness_check

        verdict = enf.summarize([enf.finding("live_boundary", enf.NUMERICAL, "7 of 56")],
                                gate="qwen3_protocol4", active=enf.REPORT_FIRST)
        returned = model_correctness_check(self._workload(verdict), self._kernels(),
                                           verify=True, device="cpu")
        self.assertIs(returned["numerical_ok"], False)      # recorded, not converted
        self.assertTrue(returned["ok"])

    def test_the_same_mismatch_stops_the_provider_in_strict_mode(self):
        from evograd.evaluation.tier3.runner import ModelCorrectnessFailure, model_correctness_check

        verdict = enf.summarize([enf.finding("live_boundary", enf.NUMERICAL, "7 of 56")],
                                gate="qwen3_protocol4", active=enf.STRICT)
        with self.assertRaises(ModelCorrectnessFailure) as caught:
            model_correctness_check(self._workload(verdict), self._kernels(), verify=True, device="cpu")
        self.assertIn("live_boundary", str(caught.exception))

    def test_a_structural_failure_stops_the_provider_under_report_first(self):
        from evograd.evaluation.tier3.runner import ModelCorrectnessFailure, model_correctness_check

        verdict = enf.summarize([enf.finding("protocol4:presence", enf.STRUCTURAL, "missing gradients")],
                                gate="qwen3_protocol4", active=enf.REPORT_FIRST)
        with self.assertRaises(ModelCorrectnessFailure):
            model_correctness_check(self._workload(verdict), self._kernels(), verify=True, device="cpu")

    def test_the_report_states_which_mode_produced_the_verdicts(self):
        from evograd.evaluation.tier3.runner import verification_policy

        enf.set_active_mode(enf.REPORT_FIRST)
        policy = verification_policy(mock.Mock(loss_delta_threshold=None), verify=True)
        self.assertEqual(policy["numerical_enforcement"]["mode"], enf.REPORT_FIRST)
        self.assertIn("recorded", policy["numerical_enforcement"]["rule"])


class TestTier2IsUntouchedByTheMode(_Mode):
    """The Tier-3 enforcement mode is not a Tier-2 or Tier-1 concept."""

    def test_the_declared_tier2_comparison_is_unchanged_under_report_first(self):
        from evograd.evaluation.tier2.runner import _compare

        want = torch.full((16,), 32.0, dtype=torch.bfloat16)
        flipped = want.clone()
        flipped[1] = 32.25
        enf.set_active_mode(enf.STRICT)
        strict = _compare(flipped, want, 1e-2, 1e-3)
        enf.set_active_mode(enf.REPORT_FIRST)
        loose = _compare(flipped, want, 1e-2, 1e-3)
        self.assertEqual(strict["ok"], loose["ok"])
        self.assertFalse(loose["ok"])                      # still a Tier-2 failure
        self.assertEqual(strict["headroom"], loose["headroom"])
        self.assertEqual(strict["atol"], loose["atol"])

    def test_the_tier2_cli_never_reads_the_enforcement_variable(self):
        import inspect

        from evograd.evaluation.tier2 import cli, runner

        for module in (cli, runner):
            self.assertNotIn(enf.ENV, inspect.getsource(module))
            self.assertNotIn("enforcement", inspect.getsource(module))


if __name__ == "__main__":
    unittest.main()
