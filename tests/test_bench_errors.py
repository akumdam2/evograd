"""Failure reporting in the benchmark harness.

``evograd bench`` used to let every exception escape as a bare traceback, so a
run wrapped in ``> /dev/null 2>&1`` left no evidence of what went wrong and no
report file. ``on_error="record"`` captures failures into the report instead;
the default ``"raise"`` keeps the OpenEvolve evaluator's control flow, which
relies on a raised exception to score a candidate as dead.
"""

import json
import unittest
from types import SimpleNamespace

from evograd.bench.harness import run_benchmarks


def _op():
    # Only the attributes run_benchmarks touches before dispatching a case;
    # benchmark_case itself will fail on this stub, which is the point.
    return SimpleNamespace(name="fake", benchmark=(), performance_baselines={})


def _workload():
    return SimpleNamespace(dims={"rows": 8, "cols": 64}, dtype="float16")


class TestRunBenchmarksErrors(unittest.TestCase):
    def test_setup_failure_recorded(self):
        report = run_benchmarks(
            _op(), None, workloads=(), performance_baseline="auto", on_error="record"
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"]["phase"], "setup")
        self.assertEqual(report["error"]["error_type"], "ValueError")
        self.assertIn("no benchmark workloads selected", report["error"]["error_message"])
        self.assertEqual(report["cases"], [])
        json.dumps(report)  # the report has to survive --out

    def test_setup_failure_raises_by_default(self):
        with self.assertRaises(ValueError):
            run_benchmarks(_op(), None, workloads=())

    def test_case_failure_recorded_per_workload(self):
        workload = _workload()
        report = run_benchmarks(_op(), None, workloads=(workload, workload), on_error="record")
        self.assertFalse(report["ok"])
        self.assertIsNone(report["error"])  # setup was fine; the cases were not
        self.assertEqual(len(report["cases"]), 2)
        for case in report["cases"]:
            self.assertFalse(case["ok"])
            self.assertEqual(case["dims"], {"rows": 8, "cols": 64})
            self.assertEqual(case["dtype"], "float16")
            self.assertEqual(case["error"]["phase"], "benchmark_case")
            self.assertTrue(case["error"]["traceback"])
        json.dumps(report)

    def test_case_failure_raises_by_default(self):
        with self.assertRaises(Exception):
            run_benchmarks(_op(), None, workloads=(_workload(),))

    def test_empty_aggregate_is_fully_keyed_and_zeroed(self):
        report = run_benchmarks(_op(), None, workloads=(), on_error="record")
        aggregate = report["aggregate"]
        for key in (
            "speedup_vs_baseline_backward",
            "speedup_vs_pytorch_autograd_backward",
            "geomean_speedup_vs_baseline_backward",
            "geomean_min_speedup_per_case",
            "weighted_geomean_min_speedup_per_case",
            "worst_case_min_speedup",
            "saved_memory_ratio",
        ):
            self.assertEqual(aggregate[key], 0.0, key)

    def test_rejects_unknown_on_error_mode(self):
        with self.assertRaises(ValueError):
            run_benchmarks(_op(), None, workloads=(), on_error="ignore")


if __name__ == "__main__":
    unittest.main()
