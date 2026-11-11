"""Every LM harvest observes its own integration's real run (wiring only, no model built).

``tests/verification/test_integration_qwen3.py`` runs the whole chain on a tiny
Qwen3; here each of the four harvests is checked to hand ``base.real_run`` its
own integration (the one whose snapshot a capture checks) and to observe what
comes back under the verification workload's spec.
"""

from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest import mock

PACKAGES = ("qwen3_0_6b", "llama3_2_1b", "olmoe_1b_7b", "qwen3_5_4b")


class EveryHarvestObservesItsIntegration(unittest.TestCase):
    def test_the_harvest_passes_its_integrations_run_to_the_observer(self):
        for package in PACKAGES:
            with self.subTest(package):
                harvest = importlib.import_module(
                    f"evograd.benchmark.topdown.{package}.harvest.harvest")
                declaration = importlib.import_module(
                    f"evograd.benchmark.topdown.{package}.declaration")
                integration = importlib.import_module(f"evograd.verification.integrations.{package}")
                workload = SimpleNamespace(spec="the verification spec")
                ran = []

                def real_run(model, *, device):
                    ran.append((model, device))
                    return workload, "net", "batch", {"order": 101}

                with mock.patch("evograd.verification.integrations.base.real_run", real_run), \
                        mock.patch("evograd.benchmark.topdown.common.harvest.run_harvest",
                                   return_value={"run": "observed"}) as observe:
                    self.assertEqual(harvest.harvest_real_run(device="cuda"), {"run": "observed"})
                self.assertEqual(ran, [(integration.MODEL, "cuda")])
                self.assertEqual(integration.MODEL.name, declaration.WORKLOAD.name)
                observe.assert_called_once_with(declaration.WORKLOAD, "the verification spec",
                                                model="net", batch="batch",
                                                identity={"order": 101})


if __name__ == "__main__":
    unittest.main()
