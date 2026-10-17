"""Frozen window-loss acceptance; synthetic records test the rule, not GPU validity."""

import copy
import json
import math
import unittest

from evograd.verification import gate3


CONTEXT = {"reference": "test-reference-v1", "data_recipe": "test-dataset-v1",
           "patch_set": ["norm"], "numerics": {"dtype": "float32"}}


def record(losses=(1., 1., 1., 1.), *, weights=None, role="native", seed=0):
    weights = weights or [1.] * len(losses)
    plan = gate3.TrainingPlan(steps=len(losses), window=2, checkpoints=(), validation=False)
    acc = gate3.metrics.WindowAccumulator(plan)
    raw = []
    for step, (loss, weight) in enumerate(zip(losses, weights), 1):
        acc.add(step, loss * weight, weight)
        raw.append({"step": step, "loss_sum": loss * weight, "weight": weight, "mean_loss": loss})
    return {"completed": True, "findings": [], "deployment": {
        "role": role, "patch_set": [] if role == "native" else ["norm"],
        "identity": {"kind": "torch_compile" if role == "control" else role}},
        "identity": {"plan": plan.to_dict(), "seed": seed,
                     "data": {"order_seed": seed, "dataset": "test-dataset-v1"},
                     "initial_state": "same-initial-weights", "optimizer": {"name": "SGD", "lr": .1},
                     "evaluator": None, "adapter": {"objective": "test-loss"},
                     "environment": {"device": "test-cpu"}, "protocol_context": copy.deepcopy(CONTEXT)},
        "train_steps": raw, "train_windows": acc.windows, "validation": {}}


def controls(seed=0):
    return gate3.TrainingControl(record(seed=seed),
                                record((1.01,) * 4, seed=seed),
                                record((1.02,) * 4, seed=seed, role="control"))


class TestWindowPolicy(unittest.TestCase):
    def setUp(self):
        cal = gate3.calibrate_window_loss([controls()], floor=.001)
        self.assertIsNotNone(cal.policy, cal.to_dict())
        self.policy = cal.policy

    def verify(self, candidate, native=None, policy=None):
        return gate3.verify_training(candidate, native or record(seed=11),
                                     policy=self.policy if policy is None else policy)

    def test_reference_only_calibration_formula(self):
        self.assertAlmostEqual(self.policy.threshold, .04)
        floor = gate3.calibrate_window_loss([controls()], floor=.1)
        self.assertAlmostEqual(floor.policy.threshold, .2)

    def test_every_window_has_a_verdict(self):
        result = self.verify(record((1.01, 1.01, 1.06, 1.06), role="candidate", seed=11))
        self.assertEqual(result.status, "failed")
        self.assertEqual([w["status"] for w in result.windows], ["passed", "failed"])
        self.assertEqual(result.findings[0].subject, "window:2")
        self.assertTrue(result.complete)

    def test_window_mean_not_per_step_max(self):
        # A short oscillation cancels in the chosen mean statistic. This is an
        # intentional limitation, not a hidden second per-step difference gate.
        result = self.verify(record((1.2, .8, 1.2, .8), role="candidate", seed=11))
        self.assertEqual(result.status, "passed")
        self.assertAlmostEqual(result.distances["max_window_abs_delta"], 0.)

    def test_weights_are_tokens_not_step_means(self):
        n = record(weights=[1., 9., 1., 9.], seed=11)
        c = record((1.2, .8, 1.2, .8), weights=[1., 9., 1., 9.], role="candidate", seed=11)
        result = self.verify(c, n)
        self.assertAlmostEqual(result.distances["max_window_abs_delta"], .16)
        self.assertEqual(result.status, "failed")

    def test_missing_policy_never_passes(self):
        r = gate3.verify_training(record(role="candidate", seed=11), record(seed=11), policy=None)
        self.assertEqual(r.status, "incomplete")

    def test_policy_roundtrip_and_tampering(self):
        policy = gate3.WindowLossPolicy.from_dict(json.loads(json.dumps(self.policy.to_dict())))
        self.assertEqual(self.verify(record(role="candidate", seed=11), policy=policy).status, "passed")
        broken = policy.to_dict()
        broken["threshold"] = 100.
        self.assertEqual(self.verify(record(role="candidate", seed=11), policy=broken).status, "incomplete")

    def test_binding_changes_withhold_thresholds(self):
        for key, replacement in (("optimizer", {"lr": 100}), ("environment", {"device": "other"}),
                                 ("initial_state", "changed"), ("adapter", {"objective": "other"}),
                                 ("protocol_context", {**CONTEXT, "reference": "v2"})):
            with self.subTest(key=key):
                n = record(seed=11)
                c = record((2.,) * 4, role="candidate", seed=11)
                n["identity"][key] = c["identity"][key] = replacement
                result = self.verify(c, n)
                self.assertEqual(result.status, "incomplete")
                self.assertEqual(result.windows, [])
                self.assertNotIn("window_loss_exceeded", [f.code for f in result.findings])

    def test_horizon_or_window_changes_need_recalibration(self):
        for n, c in ((record((1.,) * 6, seed=11), record((1.,) * 6, role="candidate", seed=11)),):
            self.assertEqual(self.verify(c, n).status, "incomplete")

    def test_calibration_overlap_even_if_rng_seed_changed(self):
        c = record(role="candidate", seed=0)
        n = record(seed=0)
        c["identity"]["seed"] = n["identity"]["seed"] = 99
        result = self.verify(c, n)
        self.assertEqual(result.status, "incomplete")
        self.assertIn("calibration_overlap", [f.code for f in result.findings])

    def test_raw_step_nonfinite_cannot_hide_in_window(self):
        for side in ("candidate", "native"):
            c, n = record(role="candidate", seed=11), record(seed=11)
            (c if side == "candidate" else n)["train_steps"][0]["loss_sum"] = float("nan")
            result = self.verify(c, n)
            self.assertEqual(result.status, "failed" if side == "candidate" else "incomplete")
            json.dumps(result.to_dict(), allow_nan=False)

    def test_missing_raw_steps_or_edited_summaries_never_pass(self):
        for field in ("train_steps", "train_windows"):
            c = record(role="candidate", seed=11)
            c[field] = []
            self.assertEqual(self.verify(c).status, "incomplete")
        c = record(role="candidate", seed=11)
        c["train_windows"][0]["mean_loss"] = .5
        self.assertEqual(self.verify(c).status, "incomplete")

    def test_patch_set_mismatch(self):
        c = record(role="candidate", seed=11)
        c["deployment"]["patch_set"] = ["other"]
        self.assertEqual(self.verify(c).status, "incomplete")

    def test_incomplete_and_recorded_failures_are_not_hidden(self):
        c = record(role="candidate", seed=11)
        c["completed"] = False
        self.assertEqual(self.verify(c).status, "failed")
        c["completed"] = True
        c["findings"] = [{"outcome": "failed", "code": "coverage", "message": "missing site"}]
        self.assertEqual(self.verify(c).status, "failed")

    def test_empty_calibration_bad_values_and_candidate_control(self):
        self.assertIsNone(gate3.calibrate_window_loss([], floor=.001).policy)
        for gamma, floor in ((0, .01), (math.inf, .01), (2, -.1), (2, math.nan)):
            self.assertIsNone(gate3.calibrate_window_loss([controls()], gamma=gamma, floor=floor).policy)
        group = controls()
        group.compiled["deployment"]["role"] = "candidate"
        self.assertIsNone(gate3.calibrate_window_loss([group], floor=.001).policy)

    def test_bad_control_or_missing_context_cannot_calibrate(self):
        group = controls()
        group.repeated_native["completed"] = False
        self.assertIsNone(gate3.calibrate_window_loss([group], floor=.001).policy)
        group = controls()
        group.native["identity"]["protocol_context"] = {}
        self.assertIsNone(gate3.calibrate_window_loss([group], floor=.001).policy)

    def test_multiple_calibration_orders_and_repeat_rejected(self):
        self.assertIsNotNone(gate3.calibrate_window_loss([controls(0), controls(1)], floor=.001).policy)
        self.assertIsNone(gate3.calibrate_window_loss([controls(0), controls(0)], floor=.001).policy)

    def test_native_and_compile_holdout_can_use_same_rule(self):
        self.assertEqual(self.verify(record(seed=11)).status, "passed")
        self.assertEqual(self.verify(record(role="control", seed=11)).status, "passed")


class TestRealTrainingWindowPolicy(unittest.TestCase):
    def test_tiny_lm_measure_calibrate_verify_without_validation(self):
        import torch
        from evograd.verification.common.execution import Deployment
        from tests.verification.test_gate3 import adapter
        from tests.verification import _fixtures as F

        plan = gate3.TrainingPlan(steps=4, window=2, checkpoints=(), validation=False)
        compiled = Deployment("compile", "control", {"norm": torch.compile(F.norm_site, backend="aot_eager")},
                              {"kind": "torch_compile", "backend": "aot_eager"})

        def measure(deployment, offset):
            a = adapter()
            a.validation_data = None
            a.train_data = [F.lm_batch(offset + i) for i in range(4)]
            return gate3.measure_training(a, deployment, plan, seed=0, protocol_context=CONTEXT)

        native = measure(Deployment.native(), 100)
        repeat = measure(Deployment.native(), 100)
        control = measure(compiled, 100)
        cal = gate3.calibrate_window_loss([gate3.TrainingControl(native, repeat, control)], floor=1e-4)
        self.assertIsNotNone(cal.policy, cal.to_dict())
        holdout = measure(Deployment.native(), 200)
        candidate = measure(gate3.candidate("exact", {"norm": F.norm_site}), 200)
        result = gate3.verify_training(candidate, holdout, policy=cal.policy)
        self.assertEqual(result.status, "passed", result.to_dict())
        self.assertEqual(len(result.windows), 2)
        self.assertEqual(candidate.validation, {})
