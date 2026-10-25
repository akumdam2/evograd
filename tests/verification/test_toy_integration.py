"""The shared capture and gates, end to end, on a model that is not a language model.

On the toy classifier of :mod:`tests.verification._toy_integration` (CPU):

* capture records every site call with its cotangent, instrumentation changes
  nothing, and the corpus names its batches by content;
* ``verify`` accepts the native-equivalent candidate through Gate 1, operator
  Gate 2 and its single-site model Gate 2 (loss, logits, parameter gradients),
  passes its combination through model Gate 2 and Gate 3, and rejects a 5%
  gradient fault (which then leaves no combination to judge);
* a verify stopped by its deadline inside Gate 3 continues on rerun to the
  verdict of a run that never stopped; finished stages are reused, Gate 2
  calibrations come back from ``--state``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

warnings.filterwarnings("ignore")

FAST = "aot_eager"


class ToyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.verification.integrations import base
        from tests.verification import _toy_integration as T

        cls.T = T
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model = T.model()
        with T.toy_environment():
            cls.manifest = base.capture(cls.model, root / "corpus", device="cpu", log=lambda *a: None)
        cls.programs = {}
        for scale in (1.0, 1.05):
            path = root / f"candidate_{scale}.py"
            path.write_text(T.CANDIDATE.format(scale=scale))
            cls.programs[scale] = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def verify(self, scale, out, **options):
        from evograd.verification.integrations.gates import verify

        with self.T.toy_environment():
            return verify(self.model, {self.T.CASE_ID: self.programs[scale]}, self.manifest, out,
                          device="cpu", compile_backend=FAST, log=lambda *a: None, **options)

    def test_capture_is_complete_and_faithful(self):
        manifest = json.loads(Path(self.manifest).read_text())
        self.assertTrue(manifest["coverage"]["complete"], manifest["coverage"])
        self.assertTrue(all(f["identical"] for f in manifest["capture_fidelity"].values()))
        self.assertIn("logits_bitwise", manifest["capture_fidelity"]["101"])
        self.assertEqual(manifest["identity"]["loss"], self.T.ToyWorkload.objective_definition)
        self.assertEqual(len(manifest["payloads"]), 4)
        self.assertEqual(sorted(manifest["identity"]["parameter_state"]), ["parameters", "state_sha256",
                                                                           "tied"])

    def test_native_equivalent_is_accepted_and_a_fault_rejected(self):
        root = Path(self.tmp.name)
        good = self.verify(1.0, root / "good")
        entry = good["candidates"][self.T.CASE_ID]
        self.assertTrue(entry["accepted"], (entry, good["combination"]))
        self.assertEqual(entry["gate2_model"], "passed")
        self.assertEqual(good["combination"]["cases"], [self.T.CASE_ID])
        self.assertEqual(good["combination"]["gate2_model"], "passed")
        self.assertEqual(good["combination"]["gate3"], "passed")
        self.assertTrue(good["combination"]["accepted"])
        bad = self.verify(1.05, root / "bad")
        entry = bad["candidates"][self.T.CASE_ID]
        self.assertFalse(entry["accepted"])
        self.assertEqual(entry["gate2_operator"], "failed")

    def test_deadline_suspends_and_rerun_continues_to_the_same_verdict(self):
        from evograd.verification.gate3 import checkpoint

        root = Path(self.tmp.name)
        unbroken = self.verify(1.0, root / "unbroken", state=root / "state_a")
        calls = []

        def expire_after_two(self_):
            calls.append(1)
            return len(calls) > 2

        out = root / "resumed"
        with mock.patch.object(checkpoint.RunCheckpoint, "expired", expire_after_two):
            first = self.verify(1.0, out, state=root / "state_b", deadline_min=60)
        self.assertIn("suspended", first)
        self.assertTrue((out / "cases" / "swiglu__act.json").exists())
        second = self.verify(1.0, out, state=root / "state_b")
        self.assertNotIn("suspended", second)
        self.assertEqual(second["candidates"], unbroken["candidates"])
        self.assertEqual(second["combination"], unbroken["combination"])
        resumed = json.loads((out / "combination" / "gate3_candidate.json").read_text())
        straight = json.loads((root / "unbroken" / "combination" / "gate3_candidate.json").read_text())
        self.assertEqual([s["objective"] for s in resumed["train_steps"]],
                         [s["objective"] for s in straight["train_steps"]])
        # The calibrations were saved under --state and are reused by a new candidate.
        operator = list((root / "state_b" / "gate2" / "operator").rglob("*.json"))
        self.assertEqual(len(operator), 1)
        again = self.verify(1.05, root / "again", state=root / "state_b")
        report = json.loads((root / "again" / "cases" / "swiglu__act.json").read_text())
        self.assertIn("reused_policy", report["gate2_operator_calibration"]["evidence"])
        self.assertFalse(again["candidates"][self.T.CASE_ID]["accepted"])

    def test_gate3_controls_are_kept_per_optimizer(self):
        import dataclasses

        from evograd.verification.integrations.gates import verify

        root = Path(self.tmp.name)
        state = root / "state_opt"
        self.verify(1.0, root / "constant", state=state)
        control = next((state / "gate3").rglob("native_a.json"))
        constant = json.loads(control.read_text())
        self.assertEqual(constant["identity"]["optimizer"]["schedule"], "constant")
        # Another optimizer trains its own controls beside the protocol's ones.
        model = dataclasses.replace(self.model, gate3_optimizer={"warmup_steps": 2,
                                                                 "clip_grad_norm": 1.0})
        with self.T.toy_environment():
            warm = verify(model, {self.T.CASE_ID: self.programs[1.0]}, self.manifest,
                          root / "warmup", state=state, device="cpu", compile_backend=FAST,
                          log=lambda *a: None)
            beside = control.parent / "clip_grad_norm=1.0_warmup_steps=2" / "native_a.json"
            trained = json.loads(beside.read_text())
            self.assertEqual(trained["identity"]["optimizer"]["warmup_steps"], 2)
            self.assertIn("lr", trained["train_steps"][0])
            self.assertEqual(warm["combination"]["gate3"], "passed")
            self.assertEqual(json.loads(control.read_text())["run_id"], constant["run_id"])
            # Under the same optimizer again, both are read back.
            verify(model, {self.T.CASE_ID: self.programs[1.0]}, self.manifest, root / "warmup2",
                   state=state, device="cpu", compile_backend=FAST, log=lambda *a: None)
        self.assertEqual(json.loads(beside.read_text())["run_id"], trained["run_id"])
        # Gradient accumulation is a training setting too: its own controls, judged as usual.
        model = dataclasses.replace(self.model, gate3_accumulation=2)
        with self.T.toy_environment():
            accumulated = verify(model, {self.T.CASE_ID: self.programs[1.0]}, self.manifest,
                                 root / "accumulated", state=state, device="cpu",
                                 compile_backend=FAST, log=lambda *a: None)
        run = json.loads((control.parent / "accumulation=2" / "native_a.json").read_text())
        self.assertEqual(run["identity"]["plan"]["accumulation"], 2)
        self.assertEqual(accumulated["combination"]["gate3"], "passed")

    def test_a_cached_control_of_another_optimizer_is_not_reused(self):
        from evograd.verification.integrations import gates

        root = Path(self.tmp.name)
        state = root / "state_stale"
        self.verify(1.0, root / "stale", state=state)
        control = next((state / "gate3").rglob("native_a.json"))
        cached = json.loads(control.read_text())
        # A control file whose recorded optimizer differs (as if written by
        # other code) is trained again rather than trusted.
        cached["identity"]["optimizer"]["lr"] = 123.0
        control.write_text(json.dumps(cached))
        self.verify(1.0, root / "stale2", state=state)
        again = json.loads(control.read_text())
        self.assertNotEqual(again["run_id"], cached["run_id"])
        self.assertEqual(again["identity"]["optimizer"]["lr"], gates.LR)

    def test_a_cached_control_of_another_workload_description_is_not_reused(self):
        root = Path(self.tmp.name)
        state = root / "state_rebound"
        self.verify(1.0, root / "rebound", state=state)
        control = next((state / "gate3").rglob("native_a.json"))
        cached = json.loads(control.read_text())
        # Written by an older checkout whose workload described itself differently.
        cached["identity"]["adapter"]["config"] = {"workload": {"older": True}}
        control.write_text(json.dumps(cached))
        verdict = self.verify(1.0, root / "rebound2", state=state)
        again = json.loads(control.read_text())
        self.assertNotEqual(again["run_id"], cached["run_id"])
        self.assertEqual(verdict["combination"]["gate3"], "passed")

    def test_without_model_gates_level2_is_judged_by_gate1_and_operator_gate2(self):
        import dataclasses

        from evograd.verification.integrations.gates import verify

        root = Path(self.tmp.name)
        model = dataclasses.replace(self.model, model_gates=False)
        with self.T.toy_environment():
            verdict = verify(model, {self.T.CASE_ID: self.programs[1.0]}, self.manifest,
                             root / "no_model_gates", device="cpu", compile_backend=FAST,
                             log=lambda *a: None)
        self.assertIn("not_applicable", verdict["combination"])
        self.assertTrue(verdict["candidates"][self.T.CASE_ID]["accepted"])
        self.assertNotIn("gate2_model", verdict["candidates"][self.T.CASE_ID])
        self.assertFalse((root / "no_model_gates" / "combination").exists())


if __name__ == "__main__":
    unittest.main()


class TestGate1Envelope(unittest.TestCase):
    def test_gate1_envelope_names_gradients_as_gate1_does(self):
        from types import SimpleNamespace

        from evograd.benchmark import get_task
        from evograd.verification.integrations.gates import gate1_envelope

        op = get_task("qwen3_qkv_norm_rope")
        policy = SimpleNamespace(thresholds={
            "out:q": {"envelope": 0.75}, "grad:q_weight": {"envelope": 0.0039},
            "grad_in:x": {"envelope": 0.0001}, "grad:v_weight": {"E2": 1.0}})   # no envelope
        self.assertEqual(gate1_envelope(op, policy),
                         {"out:q": 0.75, "grad_in:dq_weight": 0.0039, "grad_in:dx": 0.0001})
        self.assertEqual(gate1_envelope(op, None), {})

