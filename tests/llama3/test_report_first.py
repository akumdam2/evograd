"""Llama's model-scope gate under report-first, and its new providers.

The same separation Qwen's gate makes, in Llama's own gate: a finite numerical
disagreement is recorded and execution continues; a missing gradient, a
non-finite value, an impure provider, wrong invocation coverage and any runtime
failure still stop the provider, in either mode. Plus the two things Llama did
not have: an unpatched whole-model ``torch.compile`` provider, and a training
loop -- the shared one -- fed by deterministic synthetic batches.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from evograd.evaluation.tier3.gate import enforcement as enf

SMALL = {
    "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
    "arch_overrides": {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
                       "num_attention_heads": 4, "num_key_value_heads": 1,
                       "head_dim": 16, "vocab_size": 128},
}


def _workload():
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    return Llama3Workload.from_config(SMALL)


class _Mode(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(enf.ENV)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(enf.ENV, None)
        else:
            os.environ[enf.ENV] = self._saved


class _Policy:
    """The shape ``check_model_correctness`` reads, with no thresholds of its own."""

    workload_id = "llama-test"
    environment_hash = "env"
    notes = {"margin": 2.0, "gated_metrics": ["rel_l2"]}
    envelopes: dict = {}
    bound_pair_envelopes: dict = {}
    bound_pair_trajectory = None

    class trajectory:
        horizon = 2
        learning_rate = 1e-4


class TestLlamaGateControlFlow(_Mode):
    CLEAN_LOCAL = {"ok": True, "failure_count": 0, "checked_invocations": 32,
                   "numerical_only": False}
    NUMERICAL_LOCAL = {"ok": False, "failure_count": 5, "checked_invocations": 32,
                       "numerical_only": True,
                       "failures": [{"id": "attention:layer0:#1", "result": "out", "role": "output",
                                     "max_abs_err": 0.5, "atol": 0.02, "rtol": 0.01}]}

    def _run(self, mode, *, local, envelope_ok=True, finite=True, counts_ok=True,
             trajectory_ok=True, purity_ok=True, preflight=None):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import gate as G
        from evograd.evaluation.tier3.workloads.llama3_2_1b import boundary, purity

        enf.set_active_mode(mode)
        step = {"provenance": {"method": "module_surgery"}, "counts": {"attention": 2},
                "expected_counts": {"attention": 2},
                "count_problems": ([] if counts_ok else ["attention: expected 2, saw 1"]),
                "missing_grads": [], "stateless_parameters": []}
        samples = [{"name": "grad:x", "finite": finite, "rel_l2": 1e-3}]
        checked = {"ok": envelope_ok, "checked": 1,
                   "exceeded": ([] if envelope_ok else
                                [{"name": "grad:x", "group": "gradient|q_proj", "metric": "rel_l2",
                                  "value": 0.5, "threshold": 0.01, "ratio": 50.0}])}
        with mock.patch.object(purity, "run_for",
                               mock.Mock(return_value={"ok": purity_ok, "sites": []})), \
             mock.patch.object(boundary, "validate_all_invocations", mock.Mock(return_value=local)), \
             mock.patch.object(G, "_step", mock.Mock(return_value=step)), \
             mock.patch.object(G, "build_references",
                               mock.Mock(return_value={"eager": {}, "bound": {}})), \
             mock.patch.object(G, "_compare", mock.Mock(return_value=samples)), \
             mock.patch.object(G, "combined_envelope", mock.Mock(return_value={})), \
             mock.patch.object(G, "check_against", mock.Mock(return_value=dict(checked))), \
             mock.patch.object(G, "_worst", mock.Mock(return_value=None)), \
             mock.patch.object(G, "_trajectory", mock.Mock(return_value=[1.0, 0.9])), \
             mock.patch.object(G.numerics, "combined_trajectory",
                               mock.Mock(return_value=mock.Mock(
                                   check=mock.Mock(return_value={"ok": trajectory_ok})))):
            kernels = mock.Mock()
            kernels.patched = ("attention",)
            return G.check_model_correctness(
                _workload(), kernels, policy=_Policy(), data_seed=0,
                preflight=preflight or {"ok": True})

    def test_a_numerical_envelope_failure_is_recorded_and_execution_continues(self):
        verdict = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL, envelope_ok=False)
        self.assertTrue(verdict["ok"])                   # training and timing may run
        self.assertTrue(verdict["execution_ok"])
        self.assertIs(verdict["numerical_ok"], False)    # and the failure stands
        self.assertEqual(verdict["numerical_status"], "mismatches_recorded")
        self.assertEqual(verdict["numerical_failed_at"], "numerical_envelopes")
        self.assertTrue(verdict["evaluation_complete"])  # every comparison was made
        # the whole gate ran: the trajectory stage is reached after the envelopes
        self.assertIn("trajectory", verdict)

    def test_strict_stops_at_the_same_envelope_failure(self):
        verdict = self._run(enf.STRICT, local=self.CLEAN_LOCAL, envelope_ok=False)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "numerical_envelopes")
        self.assertIs(verdict["numerical_ok"], False)
        self.assertNotIn("trajectory", verdict)          # stopped before it

    def test_a_local_numerical_mismatch_is_recorded_and_continues(self):
        verdict = self._run(enf.REPORT_FIRST, local=self.NUMERICAL_LOCAL)
        self.assertTrue(verdict["ok"])
        self.assertIs(verdict["numerical_ok"], False)
        self.assertEqual({f["stage"] for f in verdict["findings"]}, {"live_boundary"})
        self.assertFalse(self._run(enf.STRICT, local=self.NUMERICAL_LOCAL)["ok"])

    def test_wrong_invocation_counts_stop_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, counts_ok=False)
            self.assertFalse(verdict["ok"], mode)
            self.assertFalse(verdict["execution_ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "counts_and_provenance")

    def test_non_finite_results_stop_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, finite=False)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "numerical_envelopes")
            self.assertEqual([f["kind"] for f in verdict["findings"]], [enf.EXECUTION])

    def test_an_impure_provider_stops_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, purity_ok=False)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "provider_purity")

    def test_a_kernel_that_raises_at_preflight_stops_both_modes(self):
        raised = {"ok": False, "reason": "PreflightFailure: boom",
                  "detail": {"failures": [{"error": "Traceback", "failed": []}]}}
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL, preflight=raised)
            self.assertFalse(verdict["ok"], mode)
            self.assertEqual(verdict["execution_failed_at"], "site_preflight")

    def test_a_tolerance_only_preflight_failure_is_numerical(self):
        tolerance = {"ok": False, "reason": "PreflightFailure: dq",
                     "detail": {"failures": [{"error": None, "failed": ["dq"]}]}}
        verdict = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL, preflight=tolerance)
        self.assertTrue(verdict["ok"])
        self.assertIs(verdict["numerical_ok"], False)
        self.assertEqual(verdict["findings"][0]["kind"], enf.NUMERICAL)

    def test_a_trajectory_failure_is_numerical(self):
        verdict = self._run(enf.REPORT_FIRST, local=self.CLEAN_LOCAL, trajectory_ok=False)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["numerical_failed_at"], "loss_trajectory")
        self.assertFalse(self._run(enf.STRICT, local=self.CLEAN_LOCAL, trajectory_ok=False)["ok"])

    def test_a_clean_provider_passes_both_modes(self):
        for mode in (enf.STRICT, enf.REPORT_FIRST):
            verdict = self._run(mode, local=self.CLEAN_LOCAL)
            self.assertTrue(verdict["ok"], mode)
            self.assertIs(verdict["numerical_ok"], True, mode)
            self.assertTrue(verdict["evaluation_complete"], mode)


class TestUnavailableCalibration(_Mode):
    """No bound calibration: strict refuses; report-first runs everything that
    needs no threshold and reports the rest unavailable -- never a pass, and
    never a *complete* evaluation."""

    def _workload_without_calibration(self):
        workload = _workload()
        object.__setattr__(workload, "calibration_path", "/nonexistent/calibration.json")
        return workload

    def test_strict_refuses_an_uncalibrated_gate(self):
        enf.set_active_mode(enf.STRICT)
        verdict = self._workload_without_calibration().model_correctness(
            mock.Mock(patched=("attention",)), device="cpu")
        self.assertFalse(verdict["ok"])
        self.assertIsNone(verdict["numerical_ok"])
        self.assertEqual(verdict["numerical_status"], "unavailable")
        self.assertFalse(verdict["evaluation_complete"])
        self.assertEqual(verdict["evaluation_incomplete_because"], ["policy_binding"])

    def test_report_first_runs_the_threshold_free_checks_and_says_what_is_missing(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import gate as G

        enf.set_active_mode(enf.REPORT_FIRST)
        workload = self._workload_without_calibration()
        seen = {}

        def fake_check(wl, kernels, *, policy, data_seed, preflight=None, **rest):
            seen["policy"] = policy
            seen["preflight_ran"] = preflight is not None
            return enf.summarize(
                [enf.finding("numerical_envelopes", enf.UNAVAILABLE, "no thresholds")],
                gate="llama3_model_correctness", measured="ran")

        with mock.patch.object(G, "check_model_correctness", fake_check):
            verdict = workload.model_correctness(mock.Mock(patched=("attention",)), device="cpu")
        self.assertIsNone(seen["policy"])            # nothing foreign was reused
        self.assertTrue(seen["preflight_ran"])       # the threshold-free checks still ran
        self.assertTrue(verdict["ok"])               # execution may continue
        self.assertEqual(verdict["numerical_status"], "unavailable")
        self.assertIsNone(verdict["numerical_ok"])
        self.assertFalse(verdict["evaluation_complete"])
        self.assertFalse(verdict["policy_binding"]["ok"])
        self.assertIn("no numerics calibration", verdict["policy_binding"]["reason"])

    def test_a_foreign_calibration_is_never_reused(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import gate as G

        enf.set_active_mode(enf.REPORT_FIRST)
        seen = {}

        def fake_check(wl, kernels, *, policy, data_seed, preflight=None, **rest):
            seen["policy"] = policy
            return enf.summarize([], gate="llama3_model_correctness")

        with mock.patch.object(G, "load_policy",
                               mock.Mock(return_value=mock.Mock(workload_id="somebody-elses"))), \
             mock.patch.object(G, "check_model_correctness", fake_check):
            verdict = _workload().model_correctness(mock.Mock(patched=("attention",)), device="cpu")
        self.assertIsNone(seen["policy"])            # the foreign policy is dropped
        self.assertIn("somebody-elses", verdict["policy_binding"]["reason"])
        self.assertFalse(verdict["policy_binding"]["ok"])

    def test_the_policy_free_gate_still_judges_every_live_invocation(self):
        """The declared tolerances need no calibration, so part A is real even
        when the envelope stage is unavailable."""
        from evograd.evaluation.tier3.workloads.llama3_2_1b import gate as G

        enf.set_active_mode(enf.REPORT_FIRST)
        local = {"ok": False, "failure_count": 3, "checked_invocations": 32,
                 "numerical_only": True,
                 "failures": [{"id": "attention:layer0:#1", "result": "out", "role": "output",
                               "max_abs_err": 0.5, "atol": 0.02, "rtol": 0.01}]}
        with mock.patch.object(G, "_step", mock.Mock(return_value={
                "provenance": {}, "counts": {}, "expected_counts": {}, "count_problems": [],
                "missing_grads": [], "stateless_parameters": []})), \
             mock.patch.object(G, "build_references", mock.Mock(return_value={"eager": {}, "bound": {}})), \
             mock.patch.object(G, "_compare", mock.Mock(return_value=[{"name": "g", "finite": True, "rel_l2": 1e-3}])), \
             mock.patch.object(G, "_worst", mock.Mock(return_value={"name": "g", "rel_l2": 1e-3})), \
             mock.patch("evograd.evaluation.tier3.workloads.llama3_2_1b.purity.run_for",
                        mock.Mock(return_value={"ok": True, "sites": []})), \
             mock.patch("evograd.evaluation.tier3.workloads.llama3_2_1b.boundary.validate_all_invocations",
                        mock.Mock(return_value=local)):
            verdict = G.check_model_correctness(_workload(), mock.Mock(patched=("attention",)),
                                                policy=None, data_seed=0, preflight={"ok": True})
        kinds = {f["stage"]: f["kind"] for f in verdict["findings"]}
        self.assertEqual(kinds["live_boundary"], enf.NUMERICAL)          # judged, and failed
        self.assertEqual(kinds["numerical_envelopes"], enf.UNAVAILABLE)  # not judged, and says so
        self.assertIs(verdict["numerical_ok"], False)
        self.assertFalse(verdict["evaluation_complete"])
        self.assertIsNone(verdict["vs_eager"]["ok"])                     # measured, not judged
        self.assertTrue(verdict["vs_eager"]["measured_only"])
        self.assertEqual(verdict["vs_eager"]["worst"]["rel_l2"], 1e-3)


class TestWholeModelCompileProvider(unittest.TestCase):
    def test_it_patches_nothing_and_is_recognised(self):
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import llama3_sites
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import (
            whole_model_compile_kernels, whole_model_compile_requested,
        )

        registry = llama3_sites()
        kernels = whole_model_compile_kernels(registry)
        self.assertEqual(kernels.patched, ())
        self.assertTrue(whole_model_compile_requested(kernels))
        self.assertFalse(whole_model_compile_requested(KernelSet(registry=registry)))

    def test_it_cannot_be_combined_with_a_site_patch(self):
        from evograd.evaluation.tier3.patch import KernelSource, patch
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import whole_model_compile_kernels

        workload = _workload()
        both = patch(whole_model_compile_kernels(workload.site_registry), "swiglu_mlp",
                     lambda *a: None,
                     source=KernelSource("swiglu_mlp", "llama3_swiglu_mlp", None, "candidate"))
        with self.assertRaises(ValueError) as caught:
            workload.build_patched(both)
        self.assertIn("cannot be combined", str(caught.exception))

    def test_the_adapter_offers_it_by_flag(self):
        import evograd.evaluation.tier3.cli as cli

        args = cli._parser().parse_args(
            ["--model", "llama_3_2_1b", "--device", "cpu", "--whole-model-compile"])
        cli.check_options(args)                      # the workload accepts the flag
        workload = cli.build_workload(args)
        providers = cli.build_providers(args, quiet=True, registry=workload.site_registry)
        self.assertIn("torch_compile_model", providers)
        self.assertEqual(providers["torch_compile_model"].patched, ())


class TestSyntheticBatches(unittest.TestCase):
    def test_training_and_validation_streams_are_disjoint_and_indexed(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import training as T

        workload = _workload()
        train = T.train_batches(workload, 4)
        validation = T.validation_batches(workload, 2)
        self.assertNotEqual(train.offset, validation.offset)
        self.assertGreater(abs(train.offset - validation.offset), 1000)
        self.assertFalse(torch.equal(train.batch(0)[0], validation.batch(0)[0]))
        # positional, so re-reading gives the same tokens and validation cannot
        # advance the training stream
        self.assertTrue(torch.equal(validation.batch(1)[0], validation.batch(1)[0]))
        self.assertTrue(torch.equal(train.batch(2)[0], train.batch(2)[0]))
        self.assertIn("synthetic", validation.describe()["source"])
        self.assertIn("not", validation.describe()["note"])   # the label travels

    def test_validation_does_not_move_the_training_stream(self):
        from evograd.evaluation.tier3.gate.training import evaluate_validation
        from evograd.evaluation.tier3.workloads.llama3_2_1b import training as T

        workload = _workload()
        train = T.train_batches(workload, 3)
        before = train.batch(1)[0].clone()
        evaluate_validation(workload.eager_evaluator(), T.validation_batches(workload, 1), steps=1)
        self.assertTrue(torch.equal(train.batch(1)[0], before))


class TestLlamaBoundaryRecords(unittest.TestCase):
    def test_a_failed_comparison_carries_violating_elements_and_nothing_is_dropped(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.boundary import BoundaryReport, _judge

        want = torch.full((4, 8), 8.0)
        actual = want.clone()
        actual[1, 1] = 12.0
        entry = _judge(actual, want, 0.02, 0.01, name="out", role="forward_output",
                       reference="declared runtime spelling")
        self.assertFalse(entry["ok"])
        record = entry["discrepancy"]
        self.assertEqual(record["violations"], 1)
        self.assertEqual(record["violating_examples"][0]["coordinate"], [1, 1])
        self.assertAlmostEqual(record["violating_examples"][0]["reference"], 8.0)
        self.assertGreater(record["violating_examples"][0]["error_over_allowance"], 1.0)

        report = BoundaryReport()
        for ordinal in range(1, 41):
            identity = f"attention:layer{ordinal - 1}:#{ordinal}"
            report.ids.add(identity)
            report.counts["attention"] = report.counts.get("attention", 0) + 1
            report.invocations.append({
                "id": identity, "site": "attention", "layer": ordinal - 1, "category": None,
                "ordinal": ordinal, "op": "llama3_attention", "gradients": [],
                "outputs": [{"name": "out", "max_abs_err": 0.5, "atol": 0.02, "rtol": 0.01,
                             "ok": False, "finite": True}]})
        summary = report.to_dict(expected={"attention": 40})
        self.assertEqual(summary["failure_count"], 40)
        self.assertEqual(len(summary["failures"]), 40)      # untruncated
        self.assertTrue(summary["numerical_only"])
        self.assertEqual(enf.boundary_finding_kind(summary), enf.NUMERICAL)


if __name__ == "__main__":
    unittest.main()
