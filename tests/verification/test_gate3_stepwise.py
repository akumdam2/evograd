"""Stepwise Gate 3: one frozen limit per step on a fixed workload."""

from __future__ import annotations

import copy
import json
import math
import unittest

import torch
import unittest.mock

from evograd.verification import gate3
from evograd.verification.gate3 import stepwise
from tests.verification.test_gate3_window_policy import CONTEXT


def record(losses, *, role="native", order=707, initial="w0", run="r", context=None,
           validation=False, window=1):
    plan = gate3.TrainingPlan(steps=len(losses), window=window, checkpoints=(),
                              validation=validation) if not validation else None
    plan_dict = plan.to_dict() if plan else {"steps": len(losses), "window": window,
                                             "checkpoints": [0], "validation_enabled": True}
    raw = [{"step": t, "loss_sum": 4.0 * v, "weight": 4.0, "mean_loss": v, "objective": v}
           for t, v in enumerate(losses, 1)]
    return {"completed": True, "findings": [], "run_id": f"id-{run}", "deployment": {
        "role": role, "patch_set": [] if role == "native" else ["norm"], "name": run,
        "identity": {"kind": "torch_compile" if role == "control" else role, "run": run}},
        "identity": {"plan": plan_dict, "seed": 0,
                     "data": {"order_seed": order, "order_sha256": f"order-{order}"},
                     "initial_state": initial, "optimizer": {"name": "AdamW", "lr": 1e-4},
                     "evaluator": None, "adapter": {"objective": "test-loss"},
                     "environment": {"device": "test-cpu", "deterministic": True},
                     "protocol_context": copy.deepcopy(context or CONTEXT)},
        "train_steps": raw, "train_windows": [], "validation": {}}


BASE = [2.0, 2.0, 2.0, 2.0]


def calibrate(native_b=(2.0, 2.0, 2.0, 2.0), compiled=(2.0, 2.05, 2.0, 2.01), floor=1e-3, **kw):
    group = gate3.TrainingControl(record(BASE, run="cal_a", **kw), record(list(native_b), run="cal_b", **kw),
                                  record(list(compiled), role="control", run="cal_compile", **kw))
    return gate3.calibrate_step_loss([group], floor=floor, gamma=2.0)


class TestStepPolicy(unittest.TestCase):
    def setUp(self):
        cal = calibrate()
        self.assertIsNotNone(cal.policy, cal.to_dict())
        self.policy = cal.policy
        self.native = record(BASE, run="eval_native")

    def verify(self, losses, *, role="candidate", native=None, policy=None, **kw):
        return gate3.verify_training(record(losses, role=role, run="test", **kw),
                                     native or self.native, policy=policy or self.policy)

    def test_thresholds_differ_per_step(self):
        expected = [2 * max(0.0, abs(c - 2.0), 1e-3) for c in (2.0, 2.05, 2.0, 2.01)]
        self.assertEqual(self.policy.thresholds, expected)
        self.assertAlmostEqual(self.policy.thresholds[1], 0.1)
        self.assertNotEqual(len(set(self.policy.thresholds)), 1)
        # 0.05 is inside step 2's limit and outside step 1's.
        self.assertEqual(self.verify([2.0, 2.05, 2.0, 2.0]).status, "passed")
        result = self.verify([2.05, 2.0, 2.0, 2.0])
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.summary["first_failing_step"], 1)
        self.assertEqual(result.summary["violations"], 1)

    def test_single_spike_and_opposite_signs_are_not_averaged(self):
        spike = self.verify([2.0, 2.0, 2.003, 2.0])
        self.assertEqual(spike.status, "failed")
        self.assertEqual(spike.summary["failing_steps"], [3])
        # +/- 0.003 cancels in any window mean; every step is judged on its own.
        swing = self.verify([2.003, 2.0, 1.997, 2.0])
        self.assertEqual(swing.summary["failing_steps"], [1, 3])

    def test_exact_equality_passes(self):
        limits = self.policy.thresholds
        exact = self.verify([2.0 + limits[0], 2.0, 2.0, 2.0])
        # 2.0 + 0.002 is not exactly representable; compare the computed error.
        error = abs((2.0 + limits[0]) - 2.0)
        self.assertEqual(exact.status, "passed" if error <= limits[0] else "failed")
        at = self.verify([2.0, 2.0, 2.0, 2.0])
        self.assertEqual(at.status, "passed")
        self.assertEqual([s["status"] for s in at.steps], ["passed"] * 4)

    def test_nonfinite_and_incomplete_records(self):
        bad = self.verify([2.0, math.nan, 2.0, 2.0])
        self.assertEqual(bad.status, "failed")
        self.assertIn("non_finite_step_record", [f.code for f in bad.findings])
        self.assertFalse(bad.summary["judged"])
        ref = record(BASE, run="ref")
        ref["train_steps"][1]["objective"] = math.inf
        self.assertEqual(self.verify(BASE, native=ref).status, "incomplete")
        truncated = record(BASE, role="candidate", run="t")
        truncated["train_steps"] = truncated["train_steps"][:3]
        self.assertEqual(gate3.verify_training(truncated, self.native, policy=self.policy).status,
                         "incomplete")
        reordered = record(BASE, role="candidate", run="t")
        reordered["train_steps"][1], reordered["train_steps"][2] = \
            reordered["train_steps"][2], reordered["train_steps"][1]
        result = gate3.verify_training(reordered, self.native, policy=self.policy)
        self.assertIn("step_misaligned", [f.code for f in result.findings])
        old = record(BASE, role="candidate", run="t")
        for row in old["train_steps"]:
            del row["objective"]
        self.assertIn("objective_not_recorded",
                      [f.code for f in gate3.verify_training(old, self.native,
                                                             policy=self.policy).findings])
        edited = record(BASE, role="candidate", run="t")
        edited["train_steps"][0]["objective"] = 2.5        # disagrees with loss_sum/weight
        self.assertIn("inconsistent_step_record",
                      [f.code for f in gate3.verify_training(edited, self.native,
                                                             policy=self.policy).findings])
        unfinished = record(BASE, role="candidate", run="t")
        unfinished["completed"] = False
        self.assertEqual(gate3.verify_training(unfinished, self.native, policy=self.policy).status,
                         "failed")

    def test_policy_tampering_and_formula(self):
        data = self.policy.to_dict()
        edited = copy.deepcopy(data)
        edited["thresholds"][2] = 1.0
        with self.assertRaises(ValueError):
            gate3.StepLossPolicy.from_dict(edited)                      # hash
        edited["policy_hash"] = gate3.policy._hash({k: v for k, v in edited.items()
                                                    if k != "policy_hash"})
        with self.assertRaises(ValueError):
            gate3.StepLossPolicy.from_dict(edited)                      # formula recomputed
        for key, value in (("gamma", 0.5), ("floor", 0.0), ("steps", 3)):
            changed = copy.deepcopy(data)
            changed[key] = value
            changed["policy_hash"] = gate3.policy._hash({k: v for k, v in changed.items()
                                                         if k != "policy_hash"})
            with self.assertRaises(ValueError):
                gate3.StepLossPolicy.from_dict(changed)
        short = copy.deepcopy(data)
        short["controls"][0]["native_repeat"] = short["controls"][0]["native_repeat"][:3]
        short["policy_hash"] = gate3.policy._hash({k: v for k, v in short.items() if k != "policy_hash"})
        with self.assertRaises(ValueError):
            gate3.StepLossPolicy.from_dict(short)
        roundtrip = gate3.StepLossPolicy(json.dumps(data))
        self.assertEqual(roundtrip.thresholds, self.policy.thresholds)
        self.assertEqual(self.verify(BASE, policy=data).status, "passed")   # mapping accepted

    def test_window_policy_is_not_reinterpreted(self):
        from tests.verification.test_gate3_window_policy import controls

        window = gate3.calibrate_window_loss([controls()], floor=.001).policy
        result = gate3.verify_step_loss(record(BASE, role="candidate", run="t"), self.native,
                                        policy=window.to_dict())
        self.assertEqual(result.status, "incomplete")
        self.assertIn("wrong_policy_schema", [f.code for f in result.findings])
        # The shared entry point keeps the window path for window policies.
        self.assertIsInstance(gate3.verify_training(
            record(BASE, role="candidate", run="t"), self.native, policy=window),
            gate3.TrainingVerdict)

    def test_candidate_cannot_calibrate_and_runs_must_be_distinct(self):
        disguised = gate3.TrainingControl(record(BASE, run="a"), record(BASE, run="b"),
                                          record(BASE, role="candidate", run="c"))
        self.assertIsNone(gate3.calibrate_step_loss([disguised], floor=1e-3).policy)
        not_native = gate3.TrainingControl(record(BASE, run="a"), record(BASE, role="control", run="b"),
                                           record(BASE, role="control", run="c"))
        self.assertIsNone(gate3.calibrate_step_loss([not_native], floor=1e-3).policy)
        twice = record(BASE, run="a")
        same = gate3.TrainingControl(twice, twice, record(BASE, role="control", run="c"))
        self.assertIsNone(gate3.calibrate_step_loss([same], floor=1e-3).policy)
        self.assertIsNone(calibrate(floor=0.0).policy)
        self.assertIsNone(calibrate(compiled=(2.0, math.nan, 2.0, 2.0)).policy)
        mixed = gate3.TrainingControl(record(BASE, run="a"), record(BASE, run="b", order=808),
                                      record(BASE, role="control", run="c"))
        self.assertIsNone(gate3.calibrate_step_loss([mixed], floor=1e-3).policy)

    def test_same_sequence_fresh_runs_are_accepted(self):
        # Unlike the window rule, reusing the calibration data order is the design.
        fresh = self.verify([2.0, 2.04, 2.0, 2.0], role="control")
        self.assertEqual(fresh.status, "passed", fresh.to_dict())
        native = gate3.verify_training(record(BASE, run="eval_native"), record(BASE, run="cal_a"),
                                       policy=self.policy)
        self.assertEqual(native.status, "passed")

    def test_bitwise_equal_fresh_runs_are_distinct_executions(self):
        a, b = record(BASE, run="same"), record(BASE, run="same")
        b["run_id"] = "another-execution"            # deterministic repeat: equal records
        group = gate3.TrainingControl(a, b, record(BASE, role="control", run="c"))
        self.assertIsNotNone(gate3.calibrate_step_loss([group], floor=1e-3).policy)
        self.assertIsNone(gate3.calibrate_step_loss(
            [gate3.TrainingControl(a, copy.deepcopy(a), record(BASE, role="control", run="c"))],
            floor=1e-3).policy)

    def test_tested_calibration_run_is_not_independent(self):
        cal_compile = record([2.0, 2.05, 2.0, 2.01], role="control", run="cal_compile")
        result = gate3.verify_training(cal_compile, record(BASE, run="cal_a"), policy=self.policy)
        self.assertEqual(result.status, "incomplete")
        self.assertIn("tested_run_used_in_calibration", [f.code for f in result.findings])

    def test_binding_mismatch_withholds_judgement(self):
        for change in ({"order": 808}, {"initial": "w1"},
                       {"context": {**CONTEXT, "reference": "other-reference"}}):
            result = gate3.verify_training(record([9.0] * 4, role="candidate", run="t", **change),
                                           record(BASE, run="n", **change), policy=self.policy)
            self.assertEqual(result.status, "incomplete", change)
            self.assertIn("policy_binding_mismatch", [f.code for f in result.findings])
            self.assertFalse(result.summary["judged"])
            self.assertEqual(result.steps, [])
        unmatched = gate3.verify_training(record(BASE, role="candidate", run="t", order=808),
                                          self.native, policy=self.policy)
        self.assertIn("run_binding_mismatch", [f.code for f in unmatched.findings])

    def test_training_only_plans(self):
        with self.assertRaises(ValueError):
            stepwise.step_binding(record(BASE, validation=True))
        self.assertIsNone(calibrate(validation=True).policy)


class TestObjectiveRecording(unittest.TestCase):
    def test_records_the_differentiated_float32_scalar(self):
        from evograd.verification.common.execution import Deployment

        class One(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(()))

        seen = {}

        class Adapter:
            device = "cpu"

            def identity(self):
                return {"objective": "p * 1 / 3"}

            def data_identity(self):
                return {"steps": "fixed"}

            def optimizer_config(self):
                return {"name": "SGD", "lr": 0.0}

            def build_model(self, deployment, hooks):
                return One()

            def build_optimizer(self, model):
                return torch.optim.SGD(model.parameters(), lr=0.0)

            def train_batch(self, index):
                return index

            def train_loss(self, model, batch):
                return model.p * torch.ones((), dtype=torch.float32), 3

        def on_step(record, model, optimizer, calls):
            seen[record["step"]] = float(model.p.grad)

        plan = gate3.TrainingPlan(steps=2, window=1, checkpoints=(), validation=False)
        run = gate3.measure_training(Adapter(), Deployment.native(), plan, seed=0, on_step=on_step)
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        row = run.train_steps[0]
        expected = float(torch.ones((), dtype=torch.float32) / 3)     # the float32 quotient
        self.assertEqual(row["objective"], expected)
        self.assertNotEqual(row["objective"], row["mean_loss"])        # float64 1/3 differs
        self.assertEqual(row["mean_loss"], 1.0 / 3.0)                  # historical field kept
        self.assertEqual(seen[1], expected)                            # d objective / d p
        self.assertIn("evograd-gate3-step-objective/1", run.to_dict()["objective"])
        self.assertEqual(stepwise.objectives(run.to_dict(), label="x", reference=True)[1], [])


class TestStepwiseOnRealTraining(unittest.TestCase):
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

        cal = gate3.calibrate_step_loss([gate3.TrainingControl(
            measure(Deployment.native()), measure(Deployment.native()), measure(compiled))],
            floor=1e-4)
        self.assertIsNotNone(cal.policy, cal.to_dict())
        self.assertEqual(len(cal.policy.thresholds), 4)
        native = measure(Deployment.native())
        exact = gate3.verify_training(measure(gate3.candidate("exact", {"norm": F.norm_site})),
                                      native, policy=cal.policy)
        self.assertEqual(exact.status, "passed", exact.to_dict())
        wrong = gate3.verify_training(
            measure(gate3.candidate("off", {"norm": lambda *a, **k: F.norm_site(*a, **k) * 1.5})),
            native, policy=cal.policy)
        self.assertEqual(wrong.status, "failed")
        self.assertEqual(wrong.summary["first_failing_step"], 1)


if __name__ == "__main__":
    unittest.main()
