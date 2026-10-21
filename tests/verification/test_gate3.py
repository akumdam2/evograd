"""Gate 3 (whole model): continuous training measurement on a tiny CPU LM."""

from __future__ import annotations

import json
import unittest

import torch

from evograd.verification import gate3
from evograd.verification.common.execution import Deployment, candidate
from tests.verification import _fixtures as F

PLAN = gate3.TrainingPlan(steps=6, window=2, checkpoints=(0, 2, 6))


def adapter(**options):
    train = [F.lm_batch(100 + i, masked=i % 3) for i in range(12)]       # unequal token counts
    validation = [F.lm_batch(900 + i) for i in range(3)]
    return gate3.site_model_training(
        lambda sites: F.TinyLM(sites), F.NATIVE_SITES, train_data=train,
        validation_data=validation, optimizer=gate3.adamw(lr=1e-2),
        logits_of=lambda model, ids: model(ids)["logits"], **options)


def strip_time(payload):
    """Without what differs between two executions by design: time and run id."""
    payload = dict(payload)
    payload.pop("seconds")
    payload.pop("run_id")
    return payload


class TestContinuousTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = gate3.measure_training(adapter(), Deployment.native(), PLAN, record_state=True)

    def test_state_evolves_continuously(self):
        run = self.native
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        steps = {k: v["optimizer"]["step_counts"] for k, v in run.state.items()}
        self.assertEqual(steps, {"0": [], "2": [2], "6": [6]})          # never reset to 1
        digests = [run.state[k]["parameters"] for k in ("0", "2", "6")]
        self.assertEqual(len(set(digests)), 3)
        self.assertNotEqual(run.identity["initial_state"], None)
        self.assertEqual(len(run.train_steps), PLAN.steps)

    def test_deterministic_repeat(self):
        again = gate3.measure_training(adapter(), Deployment.native(), PLAN, record_state=True)
        self.assertEqual(strip_time(again.to_dict()), strip_time(self.native.to_dict()))

    def test_windows_are_weighted_by_tokens(self):
        steps = self.native.train_steps
        self.assertNotEqual(steps[0]["weight"], steps[1]["weight"])
        for window in self.native.train_windows:
            first, last = window["steps"]
            chosen = steps[first - 1:last]
            expected = sum(s["loss_sum"] for s in chosen) / sum(s["weight"] for s in chosen)
            self.assertAlmostEqual(window["mean_loss"], expected, places=12)
            naive = sum(s["mean_loss"] for s in chosen) / len(chosen)
            self.assertNotAlmostEqual(window["mean_loss"], naive, places=9)

    def test_validation_is_native_on_trained_weights_and_fixed_data(self):
        cand = gate3.measure_training(adapter(), candidate("scaled", {"norm": F.norm_scaled(1.05)}),
                                      PLAN)
        self.assertTrue(cand.completed, [f.to_dict() for f in cand.findings])
        # Same initial weights, native evaluator: identical at checkpoint 0 even
        # though the candidate's own forward is off by 5%.
        self.assertEqual(cand.validation["0"], self.native.validation["0"])
        self.assertNotEqual(cand.train_steps[0]["loss_sum"], self.native.train_steps[0]["loss_sum"])
        self.assertNotEqual(cand.validation["6"]["mean_loss"], self.native.validation["6"]["mean_loss"])
        # The candidate site ran 3x per training forward and never during validation.
        self.assertEqual(cand.site_calls["total"], {"norm": 3 * PLAN.steps})
        self.assertEqual(cand.validation["2"]["weight"], self.native.validation["2"]["weight"])

    def test_evaluator_that_runs_candidate_code_is_caught(self):
        class Leaky(type(adapter())):
            def build_model(self, deployment, hooks):
                model = super().build_model(deployment, hooks)
                self._sites = dict(model.sites)
                return model

            def build_evaluator(self):
                return F.TinyLM(self._sites)

        base = adapter()
        leaky = Leaky(**{f: getattr(base, f) for f in base.__dataclass_fields__})
        run = gate3.measure_training(leaky, candidate("c", {"norm": F.norm_site}), PLAN)
        self.assertIn("evaluator_used_candidate", {f.code for f in run.findings})
        self.assertFalse(run.completed)

    def test_site_that_does_not_run_fails(self):
        # The model drops the installed site and keeps the native one: the loss
        # stays native, so only the call count can tell.
        fallback = gate3.site_model_training(
            lambda sites: F.TinyLM({**sites, "norm": F.norm_site}), F.NATIVE_SITES,
            train_data=[F.lm_batch(100 + i) for i in range(12)], validation_data=[F.lm_batch(900)],
            optimizer=gate3.adamw(lr=1e-2), logits_of=lambda model, ids: model(ids)["logits"])
        run = gate3.measure_training(fallback, candidate("c", {"norm": F.norm_site}),
                                     gate3.TrainingPlan(steps=3, window=3, checkpoints=()))
        self.assertEqual([f.code for f in run.findings], ["site_not_called"])
        self.assertFalse(run.completed)
        self.assertEqual(run.verdict()[0], "failed")

    def test_non_finite_candidate_loss(self):
        run = gate3.measure_training(adapter(), candidate("nan", {"norm": F.norm_nan}), PLAN)
        self.assertEqual([f.code for f in run.findings], ["non_finite_training_loss"])
        self.assertEqual(run.verdict()[0], "failed")
        json.dumps(run.to_dict(), allow_nan=False)

    def test_short_and_longer_horizons_share_the_core(self):
        short = gate3.measure_training(adapter(), Deployment.native(),
                                       gate3.TrainingPlan(steps=5, window=5, checkpoints=(5,)))
        self.assertTrue(short.completed)
        self.assertEqual(short.train_steps, self.native.train_steps[:5])


class TestCurvesAndAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = gate3.measure_training(adapter(), Deployment.native(), PLAN)
        cls.cand = gate3.measure_training(adapter(), candidate("s", {"norm": F.norm_scaled(1.05)}),
                                          PLAN)

    def test_aligned_curves_give_distances_but_no_verdict(self):
        comparison = gate3.compare_curves(self.cand, self.native)
        payload = comparison.to_dict()
        self.assertEqual(payload["status"], "incomplete")
        self.assertFalse(payload["acceptance"]["configured"])
        d = payload["distances"]
        self.assertEqual(d["windows"], 3)
        self.assertEqual(d["checkpoints"], ["0", "2", "6"])
        self.assertEqual(d["checkpoint_abs_deltas"]["0"], 0.0)
        expected = max(abs(a["mean_loss"] - b["mean_loss"])
                       for a, b in zip(self.cand.train_windows, self.native.train_windows))
        self.assertEqual(d["max_window_abs_delta"], expected)
        json.dumps(payload, allow_nan=False)

    def test_misalignment_is_rejected_without_distances(self):
        other = gate3.measure_training(adapter(), Deployment.native(),
                                       gate3.TrainingPlan(steps=6, window=3, checkpoints=(0, 6)))
        plan = gate3.compare_curves(self.cand, other)
        self.assertIsNone(plan.distances)
        self.assertIn("misaligned_plan", {f.code for f in plan.findings})
        short = self.cand.to_dict()
        short["train_windows"] = short["train_windows"][:-1]
        cut = gate3.compare_curves(short, self.native)
        self.assertIsNone(cut.distances)
        self.assertIn("window_count_mismatch", {f.code for f in cut.findings})
        fewer = self.cand.to_dict()
        fewer["validation"].pop("2")
        self.assertIn("checkpoint_mismatch",
                      {f.code for f in gate3.compare_curves(fewer, self.native).findings})
        self.assertIn("not_native_reference",
                      {f.code for f in gate3.compare_curves(self.native, self.cand).findings})

    def test_no_acceptance_policy_means_no_pass(self):
        with self.assertRaises(NotImplementedError):
            gate3.measure_training(adapter(), Deployment.native(), PLAN, acceptance={"max": 0.1})
        self.assertEqual(self.native.to_dict()["status"], "incomplete")
        self.assertTrue(self.native.completed)

    def test_plan_validation(self):
        for bad in ({"steps": 5, "window": 2, "checkpoints": ()},
                    {"steps": 4, "window": 2, "checkpoints": (5,)},
                    {"steps": 4, "window": 2, "checkpoints": (2, 0)},
                    {"steps": 0, "window": 1, "checkpoints": ()}):
            with self.assertRaises(ValueError, msg=bad):
                gate3.TrainingPlan(**bad)


if __name__ == "__main__":
    unittest.main()
