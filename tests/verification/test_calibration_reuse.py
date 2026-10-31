"""Optional reuse removes real executions without relaxing identity checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evograd.verification import gate2, gate3
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.gate2 import calibration, runner
from evograd.verification.gate3 import global_loss
from evograd.verification.reuse import FLAGS, calibration_reuse, native_execution, settings
from tests.verification import _fixtures as F, _toy_integration as T
from tests.verification.test_gate3_stepwise import record
from tests.verification.test_gate3_window_policy import CONTEXT


class Gate2ReuseTests(unittest.TestCase):
    def test_two_patch_sets_share_only_native_calibration_and_keep_repeats_distinct(self):
        sample = [Sample("fixed", 101)]
        with calibration_reuse({"share_gate2_calibration_native": True}, log=lambda *a: None), \
                patch.object(calibration, "execute", wraps=runner.execute) as execute:
            first = calibration.calibrate(F.toy_module_scope(), patch_set=("norm",),
                                          samples=sample, compile_backend="aot_eager")
            second = calibration.calibrate(F.toy_module_scope(), patch_set=("mix",),
                                           samples=sample, compile_backend="aot_eager")
            self.assertIsNotNone(first.policy, first.to_dict())
            self.assertIsNotNone(second.policy, second.to_dict())
            roles = [c.args[1].role for c in execute.call_args_list]
            self.assertEqual(roles.count("native"), 2)
            self.assertEqual(roles.count("control"), 4)
            a = first.policy.sample_identities["fixed"]["executions"]
            b = second.policy.sample_identities["fixed"]["executions"]
            self.assertEqual(a["native"], b["native"])
            self.assertEqual(a["native_repeats"], b["native_repeats"])
            self.assertNotEqual(a["native"], a["native_repeats"][0])
            self.assertNotEqual(a["compile"], b["compile"])

    def test_verification_reference_is_separate_from_calibration_and_fault_still_fails(self):
        sample = [Sample("fixed", 101)]
        with calibration_reuse({name: True for name in FLAGS}, log=lambda *a: None):
            scope = F.toy_module_scope()
            policy = calibration.calibrate(scope, patch_set=("norm",), samples=sample,
                                           compile_backend="aot_eager").policy
            self.assertIsNotNone(policy)
            with patch.object(runner, "execute", wraps=runner.execute) as execute:
                good = gate2.verify_integration(scope, candidate("good", {"norm": F.norm_site}),
                                               policy=policy, samples=sample)
                bad = gate2.verify_integration(scope, candidate("bad", {
                    "norm": lambda x, w: F.norm_site(x, w) * 1.2}), policy=policy, samples=sample)
            roles = [c.args[1].role for c in execute.call_args_list]
            self.assertEqual(roles.count("native"), 1)
            self.assertEqual(roles.count("candidate"), 2)
            self.assertEqual(good.status, "passed", good.to_dict())
            self.assertEqual(bad.status, "failed", bad.to_dict())
            ref = good.cases[0].evidence["reference"]["execution_id"]
            self.assertNotIn(ref, policy.calibration_executions)
            self.assertEqual(ref, bad.cases[0].evidence["reference"]["execution_id"])
            self.assertIn("reused_from", bad.cases[0].evidence["reference"])

    def test_defaults_disable_reuse_and_identity_changes_miss_the_cache(self):
        scope, sample = F.toy_module_scope(), Sample("fixed", 101)
        with patch.object(runner, "execute", wraps=runner.execute) as execute:
            for _ in range(2):
                native_execution(scope, sample, slot="verification", execute=execute)
            self.assertEqual(execute.call_count, 2)
            execute.reset_mock()
            with calibration_reuse({"share_gate2_verification_reference": True}, log=lambda *a: None):
                first = native_execution(scope, sample, slot="verification", execute=execute)
                repeated = native_execution(scope, sample, slot="verification", execute=execute)
                # Retrieval is independent storage: callers cannot mutate the saved reference.
                next(value for value in repeated.results.values() if hasattr(value, "zero_")).zero_()
                clean = native_execution(scope, sample, slot="verification", execute=execute)
                self.assertEqual(clean.results_digest, first.results_digest)
                native_execution(scope, Sample("fixed", 202), slot="verification", execute=execute)
                scope.config = {**scope.config, "changed": True}
                native_execution(scope, sample, slot="verification", execute=execute)
            self.assertEqual(execute.call_count, 3)
        self.assertFalse(any(settings()[name] for name in FLAGS))

    def test_changed_input_content_cannot_pass_with_a_cached_reference(self):
        from tests.verification.test_gate2_matched import _STATE, lm_scope
        _STATE.clear()
        self.addCleanup(_STATE.clear)
        sample = [Sample("fixed", 101)]
        with calibration_reuse({"share_gate2_verification_reference": True}, log=lambda *a: None):
            scope = lm_scope()
            policy = calibration.calibrate(scope, patch_set=("norm",), samples=sample,
                                           compile_backend="aot_eager").policy
            good = candidate("good", {"norm": F.norm_site})
            self.assertEqual(gate2.verify_integration(scope, good, policy=policy, samples=sample).status, "passed")
            _STATE["token_shift"] = 1
            changed = gate2.verify_integration(scope, good, policy=policy, samples=sample)
            self.assertEqual(changed.status, "incomplete", changed.to_dict())
            codes = {finding.code for finding in changed.all_findings()}
            self.assertTrue(codes & {"candidate_sample_mismatch", "sample_identity_mismatch",
                                     "policy_reference_mismatch"})


class Gate3ReuseTests(unittest.TestCase):
    def group(self, site):
        context = {**copy.deepcopy(CONTEXT), "patch_set": [site],
                   "native_reuse": global_loss.NATIVE_REUSE}
        native_context = {**context, "patch_set": []}
        a = record([2.] * 4, run="a", context=native_context)
        b = record([2.] * 4, run="b", context=native_context)
        c = record([2., 2.01, 2., 2.], role="control", run=f"compile-{site}", context=context)
        c["deployment"]["patch_set"] = [site]
        return a, b, c, context

    def test_shared_native_policy_preserves_limits_and_patch_checks(self):
        a, b, c, context = self.group("norm")
        shared = gate3.calibrate_global_loss([gate3.TrainingControl(a, b, c)],
                                             floor_rms=1e-3, floor_max=1e-3, share_native=True)
        self.assertIsNotNone(shared.policy, shared.to_dict())
        old_a, old_b = copy.deepcopy(a), copy.deepcopy(b)
        for run in (old_a, old_b):
            run["identity"]["protocol_context"]["patch_set"] = ["norm"]
        original = gate3.calibrate_global_loss([gate3.TrainingControl(old_a, old_b, c)],
                                               floor_rms=1e-3, floor_max=1e-3)
        self.assertEqual(original.policy.thresholds, shared.policy.thresholds)
        self.assertIsNone(gate3.calibrate_global_loss([gate3.TrainingControl(a, b, c)],
                                                    floor_rms=1e-3, floor_max=1e-3).policy)
        good = record([2.] * 4, role="candidate", run="good", context=context)
        self.assertEqual(gate3.verify_global_loss(good, a, policy=shared.policy).status, "passed")
        self.assertEqual(gate3.verify_training(good, a, policy=shared.policy.to_dict()).status, "passed")
        for field in ("data", "seed", "optimizer", "initial_state", "environment"):
            other = copy.deepcopy(good)
            other["identity"][field] = 42
            self.assertNotEqual(gate3.verify_global_loss(other, a, policy=shared.policy).status, "passed")
        wrong = copy.deepcopy(good)
        wrong["deployment"]["patch_set"] = ["mix"]
        self.assertNotEqual(gate3.verify_global_loss(wrong, a, policy=shared.policy).status, "passed")
        bad = record([5.] * 4, role="candidate", run="bad", context=context)
        self.assertEqual(gate3.verify_global_loss(bad, a, policy=shared.policy).status, "failed")
        c["deployment"]["patch_set"] = ["mix"]
        self.assertIsNone(gate3.calibrate_global_loss([gate3.TrainingControl(a, b, c)],
                                                     floor_rms=1e-3, floor_max=1e-3, share_native=True).policy)

    def test_real_native_training_reused_across_patch_labels_but_not_training_settings(self):
        from evograd.verification.integrations import gates
        with tempfile.TemporaryDirectory() as directory, T.toy_environment(), \
                calibration_reuse({"share_gate3_native": True}, log=lambda *a: None):
            model = T.model(gate3_steps=2)
            gates_path = Path(directory) / "gate3" / "native_a.json"
            from evograd.verification.integrations.base import configure
            configure()
            with patch.object(gate3, "measure_training", wraps=gate3.measure_training) as measure:
                first = gates.train(model, gates_path, Deployment.native(), model.load_workload(),
                                    ("act",), reuse=True, log=lambda *a: None)
                second = gates.train(model, gates_path, Deployment.native(), model.load_workload(),
                                     ("another-site",), reuse=True, log=lambda *a: None)
                self.assertEqual(measure.call_count, 1)
                self.assertEqual(first["run_id"], second["run_id"])
                self.assertEqual(first["identity"]["protocol_context"]["patch_set"], [])
                model = T.model(gate3_steps=3)
                gates.train(model, gates_path, Deployment.native(), model.load_workload(),
                            ("act",), reuse=True, log=lambda *a: None)
                self.assertEqual(measure.call_count, 2)
