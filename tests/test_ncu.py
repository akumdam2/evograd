"""Pure tests for the NCU refinement layer."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from evograd.ncu.profile import _parse_csv, _script
from evograd.ncu.roofline import analyze
from evograd.benchmark import get_task


class TestEvaluateProgram(unittest.TestCase):
    def test_the_child_evaluator_is_configured_for_the_op_and_restored_after(self):
        import os

        from evograd.evolve.run import _EVALUATOR_ENTRY
        from evograd.ncu.refine import _evaluate_program

        op = get_task("rmsnorm")
        seen = {}

        def fake_isolated(evaluator_file, program_path, *, timeout):
            seen.update(file=evaluator_file, program=program_path, timeout=timeout,
                        env={k: os.environ.get(k) for k in (
                            "EVOGRAD_OP", "EVOGRAD_SCORING", "EVOGRAD_BENCHMARK_WARMUP",
                            "EVOGRAD_BENCHMARK_REPS", "EVOGRAD_BASELINE_TIMING_CACHE_PATH")})
            return SimpleNamespace(metrics={"correct": 1.0, "combined_score": 1.0},
                                   artifacts={"benchmark": '{"cases": []}'})

        before = os.environ.get("EVOGRAD_OP")
        with mock.patch("evograd.evolve.evaluator.evaluate_isolated", fake_isolated):
            payload = _evaluate_program(op, Path("/tmp/candidate.py"), baseline="auto",
                                        warmup=3, reps=7, timeout=11,
                                        cache_path=Path("/tmp/cache.json"))
        self.assertEqual(seen["file"], str(_EVALUATOR_ENTRY))
        self.assertEqual(seen["program"], "/tmp/candidate.py")
        self.assertEqual(seen["timeout"], 11)
        self.assertEqual(seen["env"], {
            "EVOGRAD_OP": "rmsnorm", "EVOGRAD_SCORING": "speed_memory_min",
            "EVOGRAD_BENCHMARK_WARMUP": "3", "EVOGRAD_BENCHMARK_REPS": "7",
            "EVOGRAD_BASELINE_TIMING_CACHE_PATH": "/tmp/cache.json"})
        self.assertEqual(payload["benchmark"], {"cases": []})
        self.assertEqual(os.environ.get("EVOGRAD_OP"), before)


class TestNCUHelpers(unittest.TestCase):
    def test_profile_scripts_compile(self):
        op = get_task("rmsnorm")
        for warmup in (5, None):
            source = _script(
                op,
                Path("/tmp/candidate.py"),
                op.correctness[0],
                Path("/tmp/inputs.pt"),
                warmup=warmup,
            )
            compile(source, "<ncu-script>", "exec")

    def test_csv_metrics_are_normalized(self):
        csv_text = "\n".join(
            (
                '"Kernel Name","Metric Name","Metric Unit","Metric Value"',
                '"kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","70.0"',
                '"kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","40.0"',
            )
        )
        metrics, rows = _parse_csv(csv_text)
        self.assertEqual(metrics["sm_throughput_pct"], 70.0)
        self.assertEqual(metrics["dram_throughput_pct"], 40.0)
        self.assertEqual(len(rows), 2)

    def test_roofline_classification_and_skip(self):
        result = analyze(
            {"sm_throughput_pct": 96.0, "dram_throughput_pct": 35.0}
        )
        self.assertEqual(result.bottleneck, "compute")
        self.assertTrue(result.at_roofline)


if __name__ == "__main__":
    unittest.main()
