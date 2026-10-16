"""Gate 3 training-only plans and per-step callbacks, without experiment drivers."""

from __future__ import annotations

import json
import unittest

import torch

from evograd.verification import gate3
from evograd.verification.common.execution import Deployment, candidate
from tests.verification import _fixtures as F
from tests.verification.test_gate3 import PLAN, adapter

TRAIN_ONLY = gate3.TrainingPlan(steps=6, window=1, checkpoints=(), validation=False)


def train_only_adapter():
    """No validation data, and any attempt to evaluate or read held-out data fails."""
    base = adapter()

    class Guarded(type(base)):
        def build_evaluator(self):
            raise AssertionError("a training-only run built an evaluator")

        def validation_batches(self):
            raise AssertionError("a training-only run read validation data")

        def validation_loss(self, evaluator, batch):
            raise AssertionError("a training-only run computed a validation loss")

    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["validation_data"] = None
    return Guarded(**fields)


class TestTrainingOnlyPlan(unittest.TestCase):
    def test_plan_rules(self):
        with self.assertRaises(ValueError):
            gate3.TrainingPlan(steps=4, window=2, checkpoints=(2,), validation=False)
        with self.assertRaises(ValueError):
            gate3.TrainingPlan(steps=4, window=2, checkpoints=(), validation="no")
        self.assertTrue(PLAN.validation)
        self.assertTrue(PLAN.to_dict()["validation_enabled"])
        self.assertFalse(TRAIN_ONLY.to_dict()["validation_enabled"])

    def test_no_evaluator_and_no_validation_data(self):
        run = gate3.measure_training(train_only_adapter(), Deployment.native(), TRAIN_ONLY)
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        self.assertEqual(len(run.train_steps), TRAIN_ONLY.steps)
        self.assertEqual(run.validation, {})
        self.assertIsNone(run.identity["evaluator"])
        self.assertIsNone(run.identity["data"]["validation"])
        self.assertFalse(run.identity["plan"]["validation_enabled"])
        payload = run.to_dict()
        self.assertEqual(payload["status"], "incomplete")          # never passed
        json.dumps(payload, allow_nan=False)

    def test_training_curve_is_the_one_a_validating_run_sees(self):
        with_validation = gate3.measure_training(adapter(), Deployment.native(), PLAN)
        alone = gate3.measure_training(train_only_adapter(), Deployment.native(), TRAIN_ONLY)
        self.assertEqual(alone.train_steps, with_validation.train_steps)
        self.assertEqual(alone.identity["initial_state"], with_validation.identity["initial_state"])

    def test_default_plan_without_validation_data_is_incomplete_not_a_crash(self):
        fields = {f: getattr(adapter(), f) for f in adapter().__dataclass_fields__}
        fields["validation_data"] = None
        run = gate3.measure_training(type(adapter())(**fields), Deployment.native(), PLAN)
        self.assertIn("no_validation_data", {f.code for f in run.findings})
        self.assertFalse(run.completed)


class TestStepCallback(unittest.TestCase):
    def test_sees_every_step_in_order_after_the_update(self):
        seen = []

        def on_step(record, model, optimizer, site_calls):
            first = next(iter(model.parameters()))
            seen.append((record["step"], int(optimizer.state[first]["step"]), dict(site_calls)))
            record["mean_loss"] = -1.0                       # a copy: the run keeps its own

        run = gate3.measure_training(train_only_adapter(), candidate("n", {"norm": F.norm_site}),
                                     TRAIN_ONLY, on_step=on_step)
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        self.assertEqual([s for s, _, _ in seen], list(range(1, 7)))
        self.assertEqual([u for _, u, _ in seen], list(range(1, 7)))
        self.assertEqual([c["norm"] for _, _, c in seen], [3 * s for s in range(1, 7)])
        self.assertTrue(all(s["mean_loss"] > 0 for s in run.train_steps))

    def test_exception_stops_the_run_and_keeps_earlier_steps(self):
        def on_step(record, *_):
            if record["step"] == 3:
                raise RuntimeError("stop here")

        run = gate3.measure_training(train_only_adapter(), candidate("n", {"norm": F.norm_site}),
                                     TRAIN_ONLY, on_step=on_step)
        self.assertFalse(run.completed)
        self.assertEqual(len(run.train_steps), 3)
        finding = run.findings[0].to_dict()
        self.assertEqual(finding["code"], "training_runtime_error")
        self.assertIn("on_step 3", finding["message"])
        self.assertEqual(run.verdict()[0], "failed")

    def test_without_callback_the_run_is_unchanged(self):
        a = gate3.measure_training(adapter(), Deployment.native(), PLAN)
        b = gate3.measure_training(adapter(), Deployment.native(), PLAN, on_step=lambda *_: None)
        self.assertEqual(a.train_steps, b.train_steps)
        self.assertEqual(a.validation, b.validation)


if __name__ == "__main__":
    unittest.main()
