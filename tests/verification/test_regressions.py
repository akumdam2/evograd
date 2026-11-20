"""Regressions for defects found in review of the first implementation."""

from __future__ import annotations

import types
import unittest
import warnings
from unittest import mock

import torch

from evograd.opdecl import Workload
from evograd.verification import gate1, gate2, gate3
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Invocation, Sample, SiteHooks, candidate
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from evograd.verification.common.tensors import elementwise_error, relative_error
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.adapters.operator import OperatorScope
from tests.verification import _fixtures as F

warnings.filterwarnings("ignore")
CPU = "cpu"
FAST = "aot_eager"


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def samples(prefix, seeds):
    return [Sample(f"{prefix}/seed{s}", s) for s in seeds]


class TestOverwriteAllowance(unittest.TestCase):
    def test_declared_overwrite_is_honoured_and_an_undeclared_one_is_gate1s_to_refuse(self):
        from evograd.verification.gate2.operator import verify_operator

        op = F.SCALE_SHIFT_OVERWRITE
        pair = F.scale_shift_pair(overwrite_x=True)
        self.assertEqual(F.gate1_report(op, pair).status, PASSED)
        scope = OperatorScope(op, op.benchmark[0], device=CPU)
        report = verify_operator(scope, scope.candidate(pair), samples=scope.samples([1, 2]),
                                 compile_backend=FAST)
        self.assertEqual(report.status, PASSED, codes(report))
        # Without the allowance the same pair is an input mutation: Gate 1 refuses it
        # (operator Gate 2 runs the same pair and judges only its numbers).
        refused = F.gate1_report(F.SCALE_SHIFT, pair)
        self.assertIn("input_mutation", codes(refused))

    def test_outputs_are_judged_before_the_backward_reuses_them(self):
        from evograd.verification.gate2.operator import verify_operator

        pair = F.scale_shift_pair(clobber_output=True)
        self.assertEqual(F.gate1_report(F.SCALE_SHIFT, pair).status, PASSED)
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device=CPU)
        report = verify_operator(scope, scope.candidate(pair), samples=scope.samples([1, 2]),
                                 compile_backend=FAST)
        self.assertEqual(report.status, PASSED, codes(report))


class TestComparatorEdges(unittest.TestCase):
    def test_large_finite_float64_does_not_overflow(self):
        r = torch.full((8,), 1e160, dtype=torch.float64)
        c = r * (1 + 1e-9)
        err = relative_error(c, r)
        self.assertAlmostEqual(err.E2, 1e-9, delta=1e-12)
        from evograd.verification.gate2.model import e2
        self.assertAlmostEqual(e2(c, r), 1e-9, delta=1e-12)
        huge = elementwise_error(torch.tensor([1e308], dtype=torch.float64),
                                 torch.tensor([-1e308], dtype=torch.float64), atol=0.0, rtol=0.0)
        self.assertEqual(huge.violations, 1)

    def test_complex_keeps_the_imaginary_part(self):
        r = torch.tensor([1 + 0j, 2 + 0j])
        c = torch.tensor([1 + 1j, 2 + 2j])
        self.assertGreater(relative_error(c, r).E2, 0.5)
        self.assertEqual(elementwise_error(c, r, atol=1e-3, rtol=1e-3).violations, 2)


class TestExactValues(unittest.TestCase):
    def scope(self, attn):
        def outputs(result, invocation, sample):
            return {"y": result[0], "aux": result[1], "tag": result[2], "attn": attn(result)}

        return ModuleScope(
            scope="model", name="toy_model_attn", factory=lambda sites: F.ToyModel(sites),
            native_sites=dict(F.NATIVE_SITES), make_invocation=F.toy_invocation,
            select_outputs=outputs, output_names=("y", "aux", "tag", "attn"),
            metadata_outputs=("tag", "attn"), calls_per_execution={"norm": 2, "mix": 2},
            config={"width": 16}, device=CPU)

    def test_none_metadata_is_a_value(self):
        scope = self.scope(lambda result: None)
        from evograd.verification.gate2.model import verify_model
        from tests.verification.test_gate2 import SCALE

        report = verify_model(scope, candidate("c", {"norm": F.norm_site}), samples=samples("none", [1, 2]),
                              gamma=4.0, compile_backend=SCALE)
        self.assertEqual(report.status, PASSED, codes(report))

    def test_container_metadata_compares_exactly(self):
        self.assertTrue(C.exact_equal((torch.ones(3), {"k": [1, 2]}), (torch.ones(3), {"k": [1, 2]})))
        self.assertFalse(C.exact_equal((torch.ones(3),), (torch.zeros(3),)))
        self.assertFalse(C.exact_equal(None, torch.ones(1)))
        self.assertFalse(C.exact_equal(1, 1.0))


class TestAutocast(unittest.TestCase):
    def test_correct_candidate_passes_under_autocast(self):
        from evograd.verification.gate2.model import verify_model
        from tests.verification.test_gate2 import SCALE

        scope = F.toy_module_scope()
        scope.autocast_dtype = "bfloat16"
        report = verify_model(scope, candidate("c", {"norm": F.norm_site}), samples=samples("ac", [1, 2]),
                              gamma=4.0, compile_backend=SCALE)
        self.assertEqual(report.status, PASSED, codes(report))


class TestErrorsBecomeFindings(unittest.TestCase):
    def test_gate1_result_without_a_tolerance(self):
        # A result the policy has no tolerance for (e.g. an unsupported dtype) is
        # unavailable, never guessed and never a crash.
        wl = Workload({"R": 4, "C": 8}, "float32")
        reference = F.model_reference(F.SCALE_SHIFT)
        with mock.patch("evograd.verification.gate1.runner.rule_dtype",
                        side_effect=ValueError("unsupported result dtype")):
            report = gate1.verify_pair(reference, F.scale_shift_pair(), device=CPU,
                                       cases=F.supplied_cases(reference, [wl]))
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("tolerance_unavailable", codes(report))

class TestGate3Seed(unittest.TestCase):
    def test_runs_at_different_seeds_do_not_align(self):
        from tests.verification.test_gate3 import PLAN, adapter

        native = gate3.measure_training(adapter(), Deployment.native(), PLAN, seed=0)
        other = gate3.measure_training(adapter(), Deployment.native(), PLAN, seed=1)
        comparison = gate3.compare_curves(other, native)
        self.assertIsNone(comparison.distances)
        self.assertIn("misaligned_seed", {f.code for f in comparison.findings})


if __name__ == "__main__":
    unittest.main()
