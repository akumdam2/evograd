"""Cross-operator suite aggregation. Pure arithmetic on synthetic reports.

No GPU and no torch tensors: the point is to pin the aggregation rules, which
are where a benchmark most easily flatters itself — by averaging away failures,
by letting one crowded family decide the headline, or by quietly reading the
asymmetric backward-only speedup.
"""

from __future__ import annotations

import math
import unittest

from evograd.bench.suite import (
    FULL_STEP_SPEEDUP_KEY,
    SuiteReport,
    TaskResult,
    task_from_benchmark_report,
)


def _case(speedup, ok=True, dims=None):
    """A harness case whose *times* encode the speedup.

    Ratios are derived from measured times now, not read from a key, so every
    precomputed speedup below is deliberately wrong. Two guards ride along:
    the backward times are off by the same 10x the old fixture used, so the
    defence against reading the asymmetric backward-only metric survives; and
    the precomputed full-step key is off by 100x, so a regression to reading a
    key instead of dividing fails loudly rather than agreeing by accident.
    """
    dims = dims or {"rows": 1024}
    if not ok:
        return {
            "dims": dims,
            "dtype": "bfloat16",
            "ok": False,
            "error": {"error_type": "CompileError", "error_message": "no such shape"},
        }
    return {
        "dims": dims,
        "dtype": "bfloat16",
        "ok": True,
        "forward_ms": 0.5,
        "backward_from_saved_ms": 1.0,
        "raw_forward_backward_full_step_ms": 1.0,
        "baseline_backward_ms": speedup * 10.0,
        "baseline_raw_full_step_ms": speedup * 1.0,
        "saved_bytes": 100.0,
        "input_bytes": 200.0,
        FULL_STEP_SPEEDUP_KEY: speedup * 100.0,
        "speedup_vs_baseline_backward": speedup * 10.0,
    }


def _report(cases, saved=100.0, inputs=200.0, baseline="liger", error=None):
    cases = [
        {**case, "saved_bytes": saved, "input_bytes": inputs} if case["ok"] else case
        for case in cases
    ]
    return {
        "performance_baseline": baseline,
        "cases": cases,
        # Deliberately wrong: memory is summed from the cases that ran. Reading
        # a precomputed aggregate is how the fair adapter once published 1.000
        # for every operator.
        "aggregate": {"saved_bytes": -1.0, "input_bytes": -1.0},
        "error": error,
    }


def _task(op, level, family, speedups, cases_total=None, **kw):
    speedups = tuple(speedups)
    return TaskResult(
        op=op,
        level=level,
        family=family,
        baseline="liger",
        speedups=speedups,
        cases_ok=len(speedups),
        cases_total=cases_total if cases_total is not None else len(speedups),
        **kw,
    )


class TestTaskFromReport(unittest.TestCase):
    def test_reads_the_full_step_metric_not_the_backward_one(self):
        task = task_from_benchmark_report("swiglu", 1, "activation", _report([_case(2.0)]))
        # 20.0 would mean it read speedup_vs_baseline_backward.
        self.assertAlmostEqual(task.speedup, 2.0)

    def test_failed_cases_lower_coverage_and_contribute_no_speedup(self):
        report = _report([_case(4.0), _case(0.0, ok=False), _case(4.0)])
        task = task_from_benchmark_report("swiglu", 1, "activation", report)
        self.assertAlmostEqual(task.speedup, 4.0)
        self.assertEqual((task.cases_ok, task.cases_total), (2, 3))
        self.assertAlmostEqual(task.coverage, 2 / 3)
        self.assertFalse(task.ok)

    def test_non_finite_and_non_positive_speedups_are_dropped(self):
        report = _report([_case(2.0), _case(float("inf")), _case(0.0), _case(-1.0)])
        task = task_from_benchmark_report("swiglu", 1, "activation", report)
        self.assertEqual(task.speedups, (2.0,))

    def test_setup_failure_is_recorded_and_leaves_the_task_uncovered(self):
        report = _report([], error={"error_type": "ImportError", "error_message": "boom"})
        task = task_from_benchmark_report("swiglu", 1, "activation", report)
        self.assertIn("ImportError", task.error)
        self.assertFalse(task.ok)
        self.assertEqual(task.speedup, 0.0)

    def test_saved_memory_ratio(self):
        task = task_from_benchmark_report(
            "swiglu", 1, "activation", _report([_case(1.0)], saved=50.0, inputs=200.0)
        )
        self.assertAlmostEqual(task.saved_memory_ratio, 0.25)


class TestAggregation(unittest.TestCase):
    def test_one_crowded_family_cannot_decide_the_headline(self):
        """Pooling by family first is what stops declaration count from voting.

        Ten fast norms and one slow gemm: a flat mean over operators would land
        near the norms, because there are ten of them. Pooling within a family
        first gives each family one vote, so the answer is the geometric mean of
        4.0 and 1.0.
        """
        tasks = [_task(f"norm{i}", 1, "norm", [4.0]) for i in range(10)]
        tasks.append(_task("matmul", 1, "gemm", [1.0]))
        report = SuiteReport(tasks=tasks)
        macro = report.to_dict()["overall"]["speedup_full_step_macro"]
        self.assertAlmostEqual(macro, math.sqrt(4.0 * 1.0))

        flat = math.exp(sum(math.log(t.speedup) for t in tasks) / len(tasks))
        self.assertGreater(flat, 3.0)  # what the naive pooling would have said

    def test_a_wholly_failed_family_lowers_coverage_not_the_speedup(self):
        tasks = [
            _task("swiglu", 1, "activation", [2.0]),
            _task("conv2d", 1, "conv", [], cases_total=4),
        ]
        overall = SuiteReport(tasks=tasks).to_dict()["overall"]
        # A zero folded into a geometric mean would zero the whole report.
        self.assertAlmostEqual(overall["speedup_full_step_macro"], 2.0)
        self.assertAlmostEqual(overall["coverage_operators"], 0.5)
        self.assertAlmostEqual(overall["coverage_cases"], 1 / 5)

    def test_levels_are_reported_separately(self):
        tasks = [
            _task("swiglu", 1, "activation", [2.0]),
            _task("flce", 2, "loss", [3.0]),
            _task("llama3_decoder_layer", 3, "llm_block", [1.5]),
        ]
        levels = SuiteReport(tasks=tasks).to_dict()["levels"]
        self.assertEqual(set(levels), {"1", "2", "3"})
        self.assertEqual(levels["1"]["name"], "primitive")
        self.assertEqual(levels["3"]["name"], "block")
        self.assertAlmostEqual(levels["2"]["speedup_full_step_macro"], 3.0)

    def test_within_operator_pooling_is_geometric(self):
        # 2x on one shape and 8x on another is 4x, not 5x: a benchmark that
        # averaged arithmetically would let one lucky shape carry the result.
        task = _task("swiglu", 1, "activation", [2.0, 8.0])
        self.assertAlmostEqual(task.speedup, 4.0)

    def test_empty_report_does_not_divide_by_zero(self):
        overall = SuiteReport(tasks=[]).to_dict()["overall"]
        self.assertEqual(overall["speedup_full_step_macro"], 0.0)
        self.assertEqual(overall["coverage_cases"], 0.0)
        self.assertEqual(overall["operators"], 0)


class TestMarkdown(unittest.TestCase):
    def test_renders_levels_operators_and_failures(self):
        tasks = [
            _task("swiglu", 1, "activation", [2.0]),
            _task("conv2d", 1, "conv", [], cases_total=4),
            _task("llama3_decoder_layer", 3, "llm_block", [1.5]),
        ]
        text = SuiteReport(tasks=tasks).to_markdown()
        self.assertIn("full-step", text)
        self.assertIn("`swiglu`", text)
        self.assertIn("Not fully covered", text)
        self.assertIn("`conv2d`", text)
        # An operator with no measurement must not print a fabricated speedup.
        conv_row = next(line for line in text.splitlines() if "`conv2d`" in line and "|" in line)
        self.assertIn("—", conv_row)


class TestAgainstTheRealRegistry(unittest.TestCase):
    def test_every_declared_operator_can_be_placed_in_the_report(self):
        from evograd.ops import OPS

        tasks = [
            _task(name, op.level, op.family, [1.0])
            for name, op in OPS.items()
            if op.level is not None
        ]
        data = SuiteReport(tasks=tasks).to_dict()
        self.assertEqual(data["overall"]["operators"], len(tasks))
        self.assertEqual(set(data["levels"]), {"1", "2", "3"})
        self.assertAlmostEqual(data["overall"]["speedup_full_step_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
