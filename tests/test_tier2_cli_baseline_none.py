"""``tier2_cli --baseline none`` must mean no pair baseline, as in tier 3."""
import unittest


class TestBaselineNone(unittest.TestCase):
    def _names(self, argv):
        from evograd.bench import tier2_cli
        args = tier2_cli._parser().parse_args(["--op", "qwen3_attention", *argv])
        return [spec.name for spec in tier2_cli._specs(args, None)]

    def test_none_adds_no_baseline_provider(self):
        self.assertEqual(self._names(["--baseline", "none"]), ["eager", "torch_compile"])

    def test_a_real_baseline_is_still_offered(self):
        self.assertEqual(self._names(["--baseline", "liger"]), ["eager", "torch_compile", "liger"])

    def test_no_compile_still_honoured(self):
        self.assertEqual(self._names(["--baseline", "none", "--no-compile"]), ["eager"])


if __name__ == "__main__":
    unittest.main()
