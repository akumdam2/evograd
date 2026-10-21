"""Global-loss Gate 3: whole-horizon RMS and max of the same-step loss difference."""

from __future__ import annotations

import copy
import json
import math
import unittest

import torch

from evograd.verification import gate3
from evograd.verification.gate3 import global_loss
from tests.verification.test_gate3_stepwise import record
from tests.verification.test_gate3_window_policy import CONTEXT

N = 4
BASE = [2.0] * N


def calibrate(native_b=None, compiled=None, *, floor_rms=1e-3, floor_max=1e-3, gamma=2.0,
              groups=None, **kw):
    native_b = native_b or BASE
    compiled = compiled or [2.0, 2.02, 2.0, 2.0]
    groups = groups or [gate3.TrainingControl(
        record(BASE, run="cal_a", **kw), record(list(native_b), run="cal_b", **kw),
        record(list(compiled), role="control", run="cal_compile", **kw))]
    return gate3.calibrate_global_loss(groups, floor_rms=floor_rms, floor_max=floor_max, gamma=gamma)


class TestMetrics(unittest.TestCase):
    def test_rms_and_max_of_same_step_differences(self):
        m = gate3.loss_metrics([1.0, 3.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0])
        self.assertEqual((m["max_abs"], m["max_abs_step"]), (2.0, 2))
        self.assertAlmostEqual(m["rms"], math.sqrt(4.0 / 4))

    def test_overflow_safe_and_unbounded(self):
        m = gate3.loss_metrics([1e300, -1e300], [0.0, 0.0])
        self.assertTrue(math.isfinite(m["rms"]))
        self.assertAlmostEqual(m["rms"] / 1e300, 1.0)
        huge = gate3.loss_metrics([1.7e308], [-1.7e308])        # the difference itself overflows
        self.assertEqual(huge["rms"], math.inf)
        self.assertTrue(huge["overflow"])


class TestGlobalPolicy(unittest.TestCase):
    def setUp(self):
        cal = calibrate()
        self.assertIsNotNone(cal.policy, cal.to_dict())
        self.policy = cal.policy
        self.native = record(BASE, run="eval_native")
        # compile: d = (0, .02, 0, 0) -> R = .01, M = .02; repeat 0; floors 1e-3
        self.T = {"rms": 2 * 0.01, "max_abs": 2 * 0.02}

    def verify(self, losses, *, role="candidate", native=None, policy=None, **kw):
        return gate3.verify_training(record(losses, role=role, run="test", **kw),
                                     native or self.native, policy=policy or self.policy)

    def test_thresholds_from_whole_trajectories(self):
        for m in global_loss.METRICS:
            self.assertAlmostEqual(self.policy.thresholds[m], self.T[m])
        data = self.policy.to_dict()
        self.assertEqual(data["controls"][0]["trusted_compile"]["max_abs_step"], 2)
        self.assertEqual(data["controls"][0]["native_repeat"]["rms"], 0.0)
        self.assertTrue(data["controls"][0]["native_repeat_bitwise"])
        noisy = calibrate(native_b=[2.0, 2.0, 2.001, 2.0]).policy.to_dict()
        self.assertFalse(noisy["controls"][0]["native_repeat_bitwise"])

    def test_sign_cancellation_does_not_hide_error(self):
        # Mean loss difference is zero, but R = .03 > T_R = .02.
        result = self.verify([2.03, 1.97, 2.03, 1.97])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.summary["passed"], {"rms": False, "max_abs": True})

    def test_single_spike_fails_max(self):
        result = self.verify([2.0, 2.0, 2.05, 2.0])        # R = .025 > .02; M = .05 > .04
        self.assertEqual(result.status, "failed")
        spike = self.verify([2.0, 2.0, 2.041, 2.0])         # R = .0205; M = .041
        self.assertFalse(spike.summary["passed"]["max_abs"])

    def test_max_only_and_rms_only_failures(self):
        # M fails alone: one spike of .045 with R = .0225? -> keep R under: N = 4 needs care.
        only_max = gate3.verify_training(
            record([2.0] * 3 + [2.041], role="candidate", run="t"), self.native, policy=self.policy)
        self.assertFalse(only_max.summary["passed"]["max_abs"])
        self.assertFalse(only_max.summary["passed"]["rms"])  # .0205 > .02 here as well
        persistent = self.verify([2.025, 2.025, 2.025, 2.025])  # R = M = .025: R fails, M passes
        self.assertEqual(persistent.summary["passed"], {"rms": False, "max_abs": True})
        self.assertEqual(persistent.status, "failed")
        spike_only = self.verify([2.039, 2.0, 2.0, 2.0])     # R = .0195 passes, M = .039 passes
        self.assertEqual(spike_only.status, "passed")

    def test_max_fails_while_rms_passes(self):
        cal = calibrate(compiled=[2.0, 2.0, 2.0, 2.0], floor_rms=0.05, floor_max=1e-3)
        result = gate3.verify_training(record([2.0, 2.01, 2.0, 2.0], role="candidate", run="t"),
                                       self.native, policy=cal.policy)
        self.assertEqual(result.summary["passed"], {"rms": True, "max_abs": False})
        self.assertEqual(result.status, "failed")

    def test_exact_equality_passes(self):
        cal = calibrate(compiled=BASE, floor_rms=0.0125, floor_max=0.0125)   # T = .025 both
        exact = gate3.verify_training(record([2.025] * N, role="candidate", run="t"),
                                      self.native, policy=cal.policy)
        error = gate3.loss_metrics([2.025] * N, BASE)
        expected = "passed" if error["rms"] <= .025 and error["max_abs"] <= .025 else "failed"
        self.assertEqual(exact.status, expected)
        same = self.verify(BASE)
        self.assertEqual(same.status, "passed")
        self.assertEqual(same.summary["rms"], 0.0)

    def test_control_metrics_per_trajectory_then_max(self):
        g1 = gate3.TrainingControl(record(BASE, run="a1"), record(BASE, run="b1"),
                                   record([2.0, 2.0, 2.0, 2.04], role="control", run="c1"))
        g2 = gate3.TrainingControl(record(BASE, run="a2"), record([2.01] * N, run="b2"),
                                   record(BASE, role="control", run="c2"))
        cal = gate3.calibrate_global_loss([g1, g2], floor_rms=1e-4, floor_max=1e-4, gamma=2.0)
        # R: max(.02 [c1], .01 [b2]) -> .04; M: max(.04 [c1], .01 [b2]) -> .08
        self.assertAlmostEqual(cal.policy.thresholds["rms"], 0.04)
        self.assertAlmostEqual(cal.policy.thresholds["max_abs"], 0.08)

    def test_floor_and_gamma(self):
        cal = calibrate(compiled=BASE, floor_rms=1e-4, floor_max=1e-4, gamma=2.0)
        self.assertEqual(cal.policy.thresholds, {"rms": 2e-4, "max_abs": 2e-4})
        three = calibrate(gamma=3.0)
        self.assertAlmostEqual(three.policy.thresholds["max_abs"], 0.06)
        for bad in ({"floor_rms": 0.0}, {"floor_max": math.nan}, {"gamma": 0.5}):
            self.assertIsNone(calibrate(**bad).policy, bad)

    def test_missing_reordered_truncated_and_nonfinite(self):
        trunc = record(BASE, role="candidate", run="t")
        trunc["train_steps"] = trunc["train_steps"][:3]
        self.assertEqual(gate3.verify_training(trunc, self.native, policy=self.policy).status,
                         "incomplete")
        reordered = record(BASE, role="candidate", run="t")
        reordered["train_steps"][0], reordered["train_steps"][1] = \
            reordered["train_steps"][1], reordered["train_steps"][0]
        self.assertIn("step_misaligned", [f.code for f in gate3.verify_training(
            reordered, self.native, policy=self.policy).findings])
        old = record(BASE, role="candidate", run="t")
        del old["train_steps"][2]["objective"]
        self.assertIn("objective_not_recorded", [f.code for f in gate3.verify_training(
            old, self.native, policy=self.policy).findings])
        nan = self.verify([2.0, math.nan, 2.0, 2.0])
        self.assertEqual(nan.status, "failed")
        self.assertFalse(nan.summary["judged"])
        inf_ref = record(BASE, run="ref")
        inf_ref["train_steps"][3]["objective"] = math.inf
        self.assertEqual(self.verify(BASE, native=inf_ref).status, "incomplete")
        json.dumps(nan.to_dict(), allow_nan=False)

    def test_policy_tampering(self):
        data = self.policy.to_dict()
        edited = copy.deepcopy(data)
        edited["thresholds"]["rms"] = 1.0
        with self.assertRaises(ValueError):
            gate3.GlobalLossPolicy.from_dict(edited)                      # hash
        edited["policy_hash"] = gate3.policy._hash({k: v for k, v in edited.items()
                                                    if k != "policy_hash"})
        with self.assertRaises(ValueError):
            gate3.GlobalLossPolicy.from_dict(edited)                      # formula
        for change in ({"formula": "mean"}, {"metrics": ["rms"]}, {"gamma": 0.9},
                       {"floors": {"rms": 0.0, "max_abs": 1e-3}}, {"steps": 3}):
            other = copy.deepcopy(data) | change
            other["policy_hash"] = gate3.policy._hash({k: v for k, v in other.items()
                                                       if k != "policy_hash"})
            with self.assertRaises(ValueError, msg=str(change)):
                gate3.GlobalLossPolicy.from_dict(other)
        self.assertEqual(self.verify(BASE, policy=data).status, "passed")  # mapping accepted

    def test_candidate_or_fault_cannot_calibrate_and_executions_are_distinct(self):
        disguised = gate3.TrainingControl(record(BASE, run="a"), record(BASE, run="b"),
                                          record(BASE, role="candidate", run="fault"))
        self.assertIsNone(gate3.calibrate_global_loss([disguised], floor_rms=1e-4,
                                                      floor_max=1e-4).policy)
        twice = record(BASE, run="a")
        self.assertIsNone(gate3.calibrate_global_loss(
            [gate3.TrainingControl(twice, twice, record(BASE, role="control", run="c"))],
            floor_rms=1e-4, floor_max=1e-4).policy)
        fresh = record(BASE, run="a")
        fresh["run_id"] = "another-execution"           # bitwise-equal deterministic repeat
        self.assertIsNotNone(gate3.calibrate_global_loss(
            [gate3.TrainingControl(twice, fresh, record(BASE, role="control", run="c"))],
            floor_rms=1e-4, floor_max=1e-4).policy)

    def test_reusing_a_calibration_execution_is_rejected(self):
        cal_compile = record([2.0, 2.02, 2.0, 2.0], role="control", run="cal_compile")
        result = gate3.verify_training(cal_compile, record(BASE, run="cal_a"), policy=self.policy)
        self.assertEqual(result.status, "incomplete")
        self.assertIn("tested_run_used_in_calibration", [f.code for f in result.findings])

    def test_same_data_fresh_run_and_binding_mismatch(self):
        fresh = gate3.verify_training(record([2.0, 2.02, 2.0, 2.0], role="control", run="eval_c"),
                                      self.native, policy=self.policy)
        self.assertEqual(fresh.status, "passed", fresh.to_dict())
        for change in ({"order": 808}, {"initial": "w1"},
                       {"context": {**CONTEXT, "reference": "other"}}):
            r = gate3.verify_training(record([9.0] * N, role="candidate", run="t", **change),
                                      record(BASE, run="n", **change), policy=self.policy)
            self.assertEqual(r.status, "incomplete", change)
            self.assertIn("policy_binding_mismatch", [f.code for f in r.findings])
            self.assertFalse(r.summary["judged"])
        mixed = gate3.verify_training(record(BASE, role="candidate", run="t", order=808),
                                      self.native, policy=self.policy)
        self.assertIn("run_binding_mismatch", [f.code for f in mixed.findings])

    def test_schema_dispatch_and_old_schemas(self):
        from tests.verification.test_gate3_window_policy import controls

        self.assertIsInstance(self.verify(BASE), gate3.GlobalVerdict)
        step = gate3.calibrate_step_loss([gate3.TrainingControl(
            record(BASE, run="a"), record(BASE, run="b"), record(BASE, role="control", run="c"))],
            floor=1e-3).policy
        self.assertIsInstance(self.verify(BASE, policy=step), gate3.StepVerdict)
        window = gate3.calibrate_window_loss([controls()], floor=.001).policy
        self.assertIsInstance(gate3.verify_training(record(BASE, role="candidate", run="t"),
                                                    self.native, policy=window),
                              gate3.TrainingVerdict)
        for foreign in (step.to_dict(), window.to_dict()):
            r = gate3.verify_global_loss(record(BASE, role="candidate", run="t"), self.native,
                                         policy=foreign)
            self.assertIn("wrong_policy_schema", [f.code for f in r.findings])
            self.assertEqual(r.status, "incomplete")
        with self.assertRaises(ValueError):
            gate3.GlobalLossPolicy.from_dict(step.to_dict())
        # The reverse: a global policy is never read as step limits.
        own = calibrate().policy
        r = gate3.verify_step_loss(record(BASE, role="candidate", run="t"), self.native,
                                   policy=own.to_dict())
        self.assertIn("wrong_policy_schema", [f.code for f in r.findings])
        self.assertEqual(r.status, "incomplete")

    def test_training_only_plans(self):
        self.assertIsNone(calibrate(validation=True).policy)


class TestGlobalOnRealTraining(unittest.TestCase):
    def test_tiny_lm_fixed_sequence(self):
        from evograd.verification.common.execution import Deployment
        from tests.verification import _fixtures as F
        from tests.verification.test_gate3 import adapter

        plan = gate3.TrainingPlan(steps=4, window=1, checkpoints=(), validation=False)
        compiled = Deployment("compile", "control",
                              {"norm": torch.compile(F.norm_site, backend="aot_eager")},
                              {"kind": "torch_compile", "backend": "aot_eager"})

        def measure(deployment):
            a = adapter()
            a.validation_data = None
            a.train_data = [F.lm_batch(100 + i) for i in range(4)]
            return gate3.measure_training(a, deployment, plan, seed=0, protocol_context=CONTEXT)

        cal = gate3.calibrate_global_loss([gate3.TrainingControl(
            measure(Deployment.native()), measure(Deployment.native()), measure(compiled))],
            floor_rms=1e-4, floor_max=1e-4)
        self.assertIsNotNone(cal.policy, cal.to_dict())
        native = measure(Deployment.native())
        exact = gate3.verify_training(measure(gate3.candidate("exact", {"norm": F.norm_site})),
                                      native, policy=cal.policy)
        self.assertEqual(exact.status, "passed", exact.to_dict())
        wrong = gate3.verify_training(
            measure(gate3.candidate("off", {"norm": lambda *a, **k: F.norm_site(*a, **k) * 1.5})),
            native, policy=cal.policy)
        self.assertEqual(wrong.status, "failed")


if __name__ == "__main__":
    unittest.main()
