"""The language-model verification paths are unchanged, bit for bit.

Compares :mod:`tests.verification._lm_goldens` against ``_golden_lm.json``,
recorded before the gates were generalized beyond language models. Skipped
where the environment (torch, transformers, Python, machine) differs from the
recording's, since tiny-model numbers are not portable across those.
"""

from __future__ import annotations

import json
import unittest

from tests.verification import _lm_goldens as G

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


def _golden() -> dict | None:
    return json.loads(G.GOLDEN.read_text()) if G.GOLDEN.exists() else None


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class LanguageModelPathsUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        golden = _golden()
        if golden is None:
            raise unittest.SkipTest("no recorded goldens")
        if golden["environment"] != G.environment():
            raise unittest.SkipTest(f"goldens recorded under {golden['environment']}")
        cls.golden = golden
        cls.now = json.loads(json.dumps(G.compute(), sort_keys=True, default=str))

    def test_qwen3_capture(self):
        self.assertEqual(self.now["qwen3_capture"], self.golden["qwen3_capture"])

    def test_qwen3_5_capture(self):
        self.assertEqual(self.now["qwen3_5_capture"], self.golden["qwen3_5_capture"])

    def test_workload_model_scope(self):
        self.assertEqual(self.now["model_scope"], self.golden["model_scope"])

    def test_gate3_llama(self):
        self.assertEqual(self.now["gate3_llama"], self.golden["gate3_llama"])


if __name__ == "__main__":
    unittest.main()
