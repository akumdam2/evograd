"""One run identity for every harvest and capture, and the rule that publishes it (CPU).

``base.real_run`` is the harvest's first step and ``base.prepare_run`` the setup
it shares with ``base.capture``; ``base.run_identity`` is what both record and
compare; ``base.require_real_run`` is the one rule every snapshot writer applies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

warnings.filterwarnings("ignore")


def with_warm_start(path: Path):
    """DiT's integration with its warm start at ``path`` (the config names a path)."""
    from evograd.verification.integrations import load

    dit = load("dit_s_2")
    return dataclasses.replace(dit, workload_config={**dit.workload_config,
                                                     "init_checkpoint": str(path)})


class OneRunIdentity(unittest.TestCase):
    def test_the_harvest_step_is_the_first_captured_step(self):
        from evograd.verification.integrations import base
        from evograd.verification.integrations.objectives import batch_digest
        from tests.verification import _toy_integration as T

        model = T.model()
        with T.toy_environment(), tempfile.TemporaryDirectory() as tmp:
            _, _, batch, run = base.real_run(model, device="cpu")
            corpus = base.capture(model, Path(tmp) / "corpus", selection="all", device="cpu",
                                  log=lambda *a: None)
            identity = json.loads(Path(corpus).read_text())["identity"]
        self.assertEqual(base.HARVEST_ORDER, base.ORDERS[0])
        self.assertEqual(set(run), set(base.RUN_KEYS) | {"execution"})
        self.assertEqual(run["order"], 101)
        self.assertEqual(run["batch"], identity["data"]["batches"]["101"])
        self.assertEqual(run["parameter_state"], identity["parameter_state"])
        self.assertEqual(batch_digest(batch), run["batch"]["batch_sha256"])
        self.assertEqual(run["execution"], identity["execution"])

    def test_a_capture_without_the_harvest_order_is_refused_first(self):
        from evograd.verification.integrations import base
        from tests.verification import _toy_integration as T

        for orders in ((202, 303), ()):
            with self.subTest(orders=orders), tempfile.TemporaryDirectory() as tmp, \
                    self.assertRaisesRegex(ValueError, "omit 101"):
                base.capture(T.model(), Path(tmp) / "corpus", orders=orders, device="cpu")

    def test_no_cuda_fails_before_the_workload_is_loaded(self):
        from evograd.verification.integrations import base
        from tests.verification import _toy_integration as T

        def load_workload():
            raise AssertionError("loaded before the device check")

        model = dataclasses.replace(T.model(), load_workload=load_workload)
        with mock.patch("torch.cuda.is_available", return_value=False), \
                self.assertRaisesRegex(RuntimeError, "no CUDA device"):
            base.real_run(model, device="cuda")

    def test_a_warm_start_is_named_by_its_content_not_its_path(self):
        from evograd.verification.integrations import base

        with tempfile.TemporaryDirectory() as tmp:
            here, there = Path(tmp) / "a" / "warm_start.pt", Path(tmp) / "b" / "warm_start.pt"
            for path in (here, there):
                path.parent.mkdir()
                path.write_bytes(b"state")
            pins = base.workload_pins(with_warm_start(here))
            self.assertNotIn("init_checkpoint", pins)
            self.assertEqual(pins["init_checkpoint_sha256"], hashlib.sha256(b"state").hexdigest())
            self.assertEqual(pins, base.workload_pins(with_warm_start(there)))
            there.write_bytes(b"regenerated")
            self.assertNotEqual(pins, base.workload_pins(with_warm_start(there)))

    def test_the_rerun_hint_names_each_models_own_commands(self):
        import importlib.util

        from evograd.benchmark.topdown import TOPDOWN_WORKLOADS, harvest_commands

        for name, package in TOPDOWN_WORKLOADS.items():
            with self.subTest(name):
                text = harvest_commands(name)
                self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", text)
                self.assertIn(f"evograd.benchmark.topdown.{package}.harvest.harvest --out", text)
                if name == "megafold":
                    self.assertTrue(text.endswith("--write-snapshot"))
                    continue
                module = f"evograd.benchmark.topdown.{package}.harvest.snapshot"
                self.assertIn(f"{module} --harvest", text)
                self.assertIsNotNone(importlib.util.find_spec(module))


class PublishRule(unittest.TestCase):
    """``require_real_run``: only a run identity of the verification run publishes."""

    @staticmethod
    def manifest(model):
        from evograd.verification.integrations import base

        run = {"model": model.name, "workload": base.workload_pins(model), "data": {},
               "order": base.HARVEST_ORDER, "batch": {}, "parameter_state": {}, "execution": {}}
        return json.loads(json.dumps({"workload_id": model.load_workload().spec.workload_id,
                                      "run": run}))

    def test_only_a_harvest_of_the_verification_run_publishes(self):
        from evograd.verification.integrations import base, load

        with tempfile.TemporaryDirectory() as tmp:
            warm = Path(tmp) / "warm_start.pt"
            warm.write_bytes(b"state")
            for model in (load("convnextv2_tiny"), with_warm_start(warm)):
                good = self.manifest(model)
                base.require_real_run(model, good)
                run = good["run"]
                refused = {
                    "records no run": {**good, "run": None},
                    "lacks": {**good, "run": {k: v for k, v in run.items() if k != "batch"}},
                    "order 202": {**good, "run": {**run, "order": 202}},
                    "of 'other'": {**good, "run": {**run, "model": "other"}},
                    "workload pins": {**good, "run": {**run, "workload": {**run["workload"],
                                                                         "stream": False}}},
                    "not the verification workload": {**good, "workload_id": "reduced"},
                }
                for reason, manifest in refused.items():
                    with self.subTest(model.name, reason=reason), \
                            self.assertRaisesRegex(ValueError, reason):
                        base.require_real_run(model, manifest)
            # A warm start regenerated after the harvest: its digest no longer matches.
            stale = self.manifest(with_warm_start(warm))
            warm.write_bytes(b"regenerated")
            with self.assertRaisesRegex(ValueError, "workload pins"):
                base.require_real_run(with_warm_start(warm), stale)


if __name__ == "__main__":
    unittest.main()
