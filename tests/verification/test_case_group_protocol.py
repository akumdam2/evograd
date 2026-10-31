"""Real operator/model gates across agents, with fixed CPU performance clocks."""
import json
from pathlib import Path
import tempfile
import time
import unittest

from evograd.evaluation.case_group import evaluate_group
from evograd.verification.integrations import base
from tests.verification import _toy_integration as T


class CaseGroupProtocolTests(unittest.TestCase):
    def test_accepted_and_faulty_agents_stay_separate_with_shared_eager_and_controls(self):
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                self.check_group(reuse)

    def check_group(self, reuse):
        from evograd.verification.reuse import FLAGS, calibration_reuse
        with tempfile.TemporaryDirectory() as directory, T.toy_environment(), \
                calibration_reuse({name: reuse for name in FLAGS}, log=lambda *a: None):
            root = Path(directory)
            model = T.model()
            manifest = base.capture(model, root / "corpus", device="cpu", log=lambda *a: None)
            plan = {"model": "toy", "state": str(root / "state"), "jobs": []}
            for agent, scale in (("good", 1.0), ("fault", 1.05)):
                program = root / f"{agent}.py"
                program.write_text(T.CANDIDATE.format(scale=scale))
                plan["jobs"].append({"out": str(root / agent), "selection": {
                    "agent": agent, "corpus": str(manifest),
                    "candidates": {T.CASE_ID: {"path": str(program)}}}})
            timed = []

            def tier2(op, modules, inputs, **_kwargs):
                self.assertEqual(set(modules), {"good", "fault"})
                self.assertEqual(len(inputs), 1)
                return {"rows": {agent: {"eager": {"speedup": 2.0, "candidate_ms": 1.0, "baseline_ms": 2.0}}
                                 for agent in modules}}

            def provider(run, name, spec, options):
                timed.append(name)
                ms = 100 if name == "eager" else 80
                return {"ok": True, "step_ms": ms, "per_block_ms": [ms] * 3,
                        "peak_memory_bytes": 100, "cpu_bound_fraction": .1}

            arguments = dict(model=model, deadline=time.time() + 60, time_tier2=tier2,
                             time_provider=provider, rules=False, check_imports=False,
                             device="cpu", compile_backend="aot_eager", log=lambda *a: None)
            self.assertEqual(evaluate_group(plan, **arguments), 0)
            good = json.loads((root / "good/evaluation.json").read_text())
            fault = json.loads((root / "fault/evaluation.json").read_text())
            self.assertTrue(good["cases"][T.CASE_ID]["accepted"])
            self.assertFalse(fault["cases"][T.CASE_ID]["accepted"])
            self.assertTrue(good["tier3"]["combination"]["accepted"])
            self.assertEqual(good["tier3"]["combination"]["cases"], [T.CASE_ID])
            self.assertEqual(fault["tier3"]["combination"]["cases"], [])
            self.assertEqual(timed.count("eager"), 1)
            self.assertIn(f"fault::{T.CASE_ID}", timed)  # diagnostic only
            self.assertNotIn("fault::combination", timed)
            self.assertEqual(good["cases"][T.CASE_ID]["shared_tier2_measurement"],
                             fault["cases"][T.CASE_ID]["shared_tier2_measurement"])
            timed.clear()
            self.assertEqual(evaluate_group(plan, **arguments), 0)
            self.assertEqual(timed, [])
