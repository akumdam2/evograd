"""A harvest runs the step a capture records first (toy integration, on CPU).

:func:`~evograd.verification.integrations.harvest_run.first_step` names the
batch, the initial parameter state and the settings exactly as the capture's
corpus identity does for order 101.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


class FirstStep(unittest.TestCase):
    def test_the_harvest_step_is_the_first_captured_step(self):
        from evograd.verification.integrations import base
        from evograd.verification.integrations.harvest_run import HARVEST_ORDER, first_step
        from evograd.verification.integrations.objectives import batch_digest
        from tests.verification import _toy_integration as T

        model = T.model()
        with T.toy_environment(), tempfile.TemporaryDirectory() as tmp:
            _, _, batch, run, settings = first_step(model, device="cpu")
            corpus = base.capture(model, Path(tmp) / "corpus", device="cpu", log=lambda *a: None)
            identity = json.loads(Path(corpus).read_text())["identity"]
        self.assertEqual(HARVEST_ORDER, base.ORDERS[0])
        self.assertEqual(run, {"data": {}, "order": 101,
                               "batch_digest": identity["data"]["batches"]["101"]["batch_sha256"],
                               "parameter_state_sha256": identity["parameter_state"]["state_sha256"]})
        self.assertEqual(batch_digest(batch), run["batch_digest"])
        self.assertEqual(settings, identity["execution"])


if __name__ == "__main__":
    unittest.main()
