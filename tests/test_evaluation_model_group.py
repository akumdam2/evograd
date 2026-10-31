"""All agents stay separate while sharing controls and one total deadline."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evograd.evaluation import case_group, model_group
from evograd.evaluation.model_cache import ENV


class ModelGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.plan = {"model": "qwen3_0_6b", "state": str(root / "state"), "jobs": [
            {"out": str(root / agent), "selection": {"agent": agent, "corpus": "manifest.json",
             "candidates": {"task/case": {"path": f"/{agent}/submission.py"}}}}
            for agent in ("gepa", "codex")]}

    def test_agents_use_separate_candidates_and_outputs_and_same_state(self):
        with patch.object(case_group, "evaluate_group", return_value=0) as evaluate:
            self.assertEqual(model_group.run(self.plan, deadline_min=45, weight_cache=False), 0)
        evaluate.assert_called_once()
        self.assertEqual(evaluate.call_args.args[0], self.plan)
        self.assertEqual(evaluate.call_args.kwargs["tier3_options"], {"blocks": 10})

    def test_suspension_stops_group_and_restores_environment(self):
        with patch.dict(os.environ, {ENV: "previous"}), patch.object(case_group, "evaluate_group", return_value=3) as evaluate:
            self.assertEqual(model_group.run(self.plan, deadline_min=45, weight_cache=False), 3)
            self.assertEqual(os.environ[ENV], "previous")
        self.assertEqual(evaluate.call_count, 1)

    def test_elapsed_deadline_does_not_start_another_agent(self):
        with patch.object(case_group, "evaluate_group") as evaluate:
            self.assertEqual(model_group.run(self.plan, deadline_min=0, weight_cache=False), 3)
            evaluate.assert_not_called()


def test_group_applies_frozen_reuse_options_and_restores_defaults(tmp_path):
    from evograd.verification.reuse import FLAGS, settings
    config = {name: True for name in FLAGS}
    plan = {"model": "toy", "state": str(tmp_path / "state"), "jobs": [],
            "calibration_reuse": config}
    def evaluate(*args, **kwargs):
        assert all(settings()[name] for name in FLAGS)
        return 0
    with patch.object(case_group, "evaluate_group", side_effect=evaluate):
        assert model_group.run(plan, deadline_min=45, weight_cache=False) == 0
    assert not any(settings()[name] for name in FLAGS)
