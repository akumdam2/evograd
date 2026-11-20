"""Optional reuse of Gate 3's native training controls, without relaxing identity checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evograd.verification import gate3
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.gate3 import global_loss
from evograd.verification.reuse import calibration_reuse
from tests.verification import _fixtures as F, _toy_integration as T
from tests.verification.test_gate3_stepwise import record
from tests.verification.test_gate3_window_policy import CONTEXT


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
