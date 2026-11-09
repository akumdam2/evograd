"""The benchmark's pooling across tasks (``evograd summarize``). Pure arithmetic.

No GPU and no torch tensors: the point is to pin the pooling rules, which are
where a benchmark most easily flatters itself -- by averaging away failures or
by letting one crowded family decide the headline.
"""

from __future__ import annotations

import math
import unittest

from evograd.benchmark.core.report import TaskResult, _pool_by_family


def _task(op, level, family, speedups, cases_total=None, **kw):
    speedups = tuple(speedups)
    return TaskResult(
        op=op,
        level=level,
        family=family,
        baseline="cublas_pair",
        speedups=speedups,
        cases_ok=len(speedups),
        cases_total=cases_total if cases_total is not None else len(speedups),
        **kw,
    )


class TestAggregation(unittest.TestCase):
    def test_one_crowded_family_cannot_decide_the_headline(self):
        """Pooling by family first is what stops declaration count from voting.

        Ten fast norms and one slow gemm: a flat mean over operators would land
        near the norms, because there are ten of them. Pooling within a family
        first gives each family one vote, so the answer is the geometric mean of
        4.0 and 1.0.
        """
        tasks = [_task(f"norm{i}", 1, "norm", [4.0]) for i in range(10)]
        tasks.append(_task("linear", 1, "gemm", [1.0]))
        self.assertAlmostEqual(_pool_by_family(tasks), math.sqrt(4.0 * 1.0))

        flat = math.exp(sum(math.log(t.speedup) for t in tasks) / len(tasks))
        self.assertGreater(flat, 3.0)  # what the naive pooling would have said

    def test_a_wholly_failed_family_lowers_coverage_not_the_speedup(self):
        tasks = [
            _task("swiglu", 1, "activation", [2.0]),
            _task("spatial_conv2d", 1, "conv", [], cases_total=4),
        ]
        # A zero folded into a geometric mean would zero the whole report.
        self.assertAlmostEqual(_pool_by_family(tasks), 2.0)
        self.assertEqual([t.coverage for t in tasks], [1.0, 0.0])
        self.assertFalse(tasks[1].ok)

    def test_within_operator_pooling_is_geometric(self):
        # 2x on one shape and 8x on another is 4x, not 5x: a benchmark that
        # averaged arithmetically would let one lucky shape carry the result.
        task = _task("swiglu", 1, "activation", [2.0, 8.0])
        self.assertAlmostEqual(task.speedup, 4.0)

    def test_nothing_measured_pools_to_zero(self):
        self.assertEqual(_pool_by_family([]), 0.0)
        self.assertEqual(_pool_by_family([_task("swiglu", 1, "activation", [], cases_total=2)]),
                         0.0)


if __name__ == "__main__":
    unittest.main()
