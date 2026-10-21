"""Regressions for defects found in review of the first implementation."""

from __future__ import annotations

import types
import unittest
import warnings

import torch

from evograd.opdecl import Workload
from evograd.verification import gate1, gate2, gate3
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Invocation, Sample, SiteHooks, candidate
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from evograd.verification.common.tensors import elementwise_error, relative_error
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.gate2.calibration import CalibrationError, Gate2Policy, calibrate
from tests.verification import _fixtures as F
from tests.verification.test_gate2 import synthetic_policy

warnings.filterwarnings("ignore")
CPU = "cpu"
FAST = "aot_eager"


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def samples(prefix, seeds):
    return [Sample(f"{prefix}/seed{s}", s) for s in seeds]


class TestOverwriteAllowance(unittest.TestCase):
    def test_declared_overwrite_is_honoured_at_both_gates(self):
        op = F.SCALE_SHIFT_OVERWRITE
        pair = F.scale_shift_pair(overwrite_x=True)
        self.assertEqual(gate1.verify_pair(op, pair, device=CPU).status, PASSED)
        scope = OperatorScope(op, op.correctness[0], device=CPU)
        cal = calibrate(scope, patch_set=("operator",), samples=scope.samples([1, 2]),
                        compile_backend=FAST)
        report = gate2.verify_integration(scope, scope.candidate(pair), policy=cal.policy,
                                          samples=scope.samples([9]))
        self.assertEqual(report.status, PASSED, codes(report))
        # Without the allowance the same pair is an input mutation.
        plain = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device=CPU)
        policy = synthetic_policy(plain, ("operator",), lambda s: {"E2": 1e-3, "Emax": 1e-2})
        refused = gate2.verify_integration(plain, plain.candidate(pair), policy=policy,
                                           samples=plain.samples([9]))
        self.assertIn("input_mutation", codes(refused))

    def test_outputs_are_judged_before_the_backward_reuses_them(self):
        pair = F.scale_shift_pair(clobber_output=True)
        self.assertEqual(gate1.verify_pair(F.SCALE_SHIFT, pair, device=CPU).status, PASSED)
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device=CPU)
        policy = synthetic_policy(scope, ("operator",), lambda s: {"E2": 1e-3, "Emax": 1e-2})
        report = gate2.verify_integration(scope, scope.candidate(pair), policy=policy,
                                          samples=scope.samples([9]))
        self.assertEqual(report.status, PASSED, codes(report))


class TestComparatorEdges(unittest.TestCase):
    def test_large_finite_float64_does_not_overflow(self):
        r = torch.full((8,), 1e160, dtype=torch.float64)
        c = r * (1 + 1e-9)
        err = relative_error(c, r)
        self.assertAlmostEqual(err.E2, 1e-9, delta=1e-12)
        check = gate2.judge(C.ResultSpec("out:z", C.OUTPUT), c, r, {"E2": 1e-6, "Emax": 1e-6})
        self.assertEqual(check.status, PASSED)
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
        fixed = samples("none", [1, 2])
        cal = calibrate(scope, patch_set=("norm",), samples=fixed, compile_backend=FAST)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                          policy=cal.policy, samples=fixed)
        self.assertEqual(report.status, PASSED, codes(report))

    def test_container_metadata_compares_exactly(self):
        self.assertTrue(C.exact_equal((torch.ones(3), {"k": [1, 2]}), (torch.ones(3), {"k": [1, 2]})))
        self.assertFalse(C.exact_equal((torch.ones(3),), (torch.zeros(3),)))
        self.assertFalse(C.exact_equal(None, torch.ones(1)))
        self.assertFalse(C.exact_equal(1, 1.0))


class TestSampleContent(unittest.TestCase):
    def blind_scope(self):
        """A recipe that ignores the sample seed: every sample is the same data."""
        def invocation(sample, device):
            return F.toy_invocation(Sample("fixed", 0), device)

        return ModuleScope(
            scope="model", name="blind", factory=lambda sites: F.ToyModel(sites),
            native_sites=dict(F.NATIVE_SITES), make_invocation=invocation,
            select_outputs=F.toy_outputs, output_names=("y", "aux", "tag"),
            metadata_outputs=("tag",), make_cotangents=lambda o, i, s: {
                k: torch.ones_like(v) for k, v in o.items() if k != "tag"},
            calls_per_execution={"norm": 2, "mix": 2}, device=CPU)

    def test_identical_calibration_samples_are_refused(self):
        cal = calibrate(self.blind_scope(), patch_set=("norm",), samples=samples("blind", [1, 2]),
                        compile_backend=FAST)
        self.assertIsNone(cal.policy)
        self.assertIn("duplicate_samples", {f.code for f in cal.findings})

    def test_matched_policy_needs_the_calibrated_sample_not_only_its_content(self):
        scope = self.blind_scope()
        cal = calibrate(scope, patch_set=("norm",), samples=samples("blind", [1]),
                        compile_backend=FAST)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        # Same content under another id/seed: not a calibrated sample, no budget.
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                          policy=cal.policy, samples=samples("blind/val", [7]))
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("sample_not_calibrated", codes(report))
        self.assertIn("required_sample_missing", codes(report))


def _checkpointed_block(sites):
    """The toy network with weights loaded from a checkpoint file (the factory's
    code, hence the reference identity, does not change with the file)."""
    module = F.ToyModel(sites)
    module.load_state_dict(torch.load(_CHECKPOINT["path"]))
    return module


_CHECKPOINT: dict = {}


class TestWeightsBinding(unittest.TestCase):
    def test_policy_refuses_other_weights(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/weights.pt"
            _CHECKPOINT["path"] = path
            torch.save(F.ToyModel(F.NATIVE_SITES).state_dict(), path)
            scope = F.toy_module_scope(factory=_checkpointed_block)
            fixed = samples("w", [1, 2])
            cal = calibrate(scope, patch_set=("norm",), samples=fixed, compile_backend=FAST)
            self.assertIsNotNone(cal.policy.weights["fixed"])
            same = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                            policy=cal.policy, samples=fixed)
            self.assertEqual(same.status, PASSED, codes(same))
            drifted = F.ToyModel(F.NATIVE_SITES)
            with torch.no_grad():
                drifted.mix_b[0].add_(0.5)
            torch.save(drifted.state_dict(), path)              # the checkpoint changed
            report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_spike}),
                                              policy=cal.policy, samples=fixed)
        self.assertEqual(report.status, INCOMPLETE)             # withheld, not judged
        self.assertIn("sample_identity_mismatch", codes(report))
        self.assertNotIn("policy_reference_mismatch", codes(report))
        self.assertEqual(report.cases[0].findings[0].details["differing"], ["initial_state"])


class TestBoundaryCapture(unittest.TestCase):
    def test_uninspectable_site_output_is_not_silently_checked(self):
        class Box:
            def __init__(self, value):
                self.value = value

        def boxed(h, weight, bias):
            return Box(F.mix_site(h, weight, bias))

        class BoxBlock(F.ToyModel):
            def forward(self, x, *, tag=7):
                for i in range(len(self.norm_w)):
                    h = self.sites["norm"](x, self.norm_w[i])
                    x = x + self.sites["mix"](h, self.mix_w[i], self.mix_b[i]).value
                return x, x.pow(2).mean(-1), torch.tensor(tag)

        scope = F.toy_module_scope(factory=lambda sites: BoxBlock(sites))
        scope.native_sites = {**scope.native_sites, "mix": boxed}
        cal = calibrate(scope, patch_set=("mix",), samples=samples("box", [1]),
                        compile_backend=FAST, boundaries=True)
        self.assertIsNone(cal.policy)
        self.assertIn("boundary_uncaptured", {f.code for f in cal.findings})

    def test_a_tensor_returned_twice_is_seeded_once(self):
        hooks = SiteHooks(record=True)
        site = hooks.wrap("dup", lambda x: (lambda y: (y, y))(x * 2))
        x = torch.ones(3, requires_grad=True)
        a, b = site(x)
        (2 * a.sum() + 10 * b.sum()).backward()
        upstream = hooks.calls[0].upstream
        self.assertEqual(sorted(upstream), ["out[0]"])
        self.assertTrue(torch.equal(upstream["out[0]"], torch.full((3,), 12.0)))


class TestAutocastBoundary(unittest.TestCase):
    def test_correct_candidate_passes_under_autocast_with_boundaries(self):
        scope = F.toy_module_scope()
        scope.autocast_dtype = "bfloat16"
        fixed = samples("ac", [1, 2])
        cal = calibrate(scope, patch_set=("norm",), samples=fixed,
                        compile_backend=FAST, boundaries=True)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                          policy=cal.policy, samples=fixed)
        self.assertEqual(report.status, PASSED, codes(report))


class TestErrorsBecomeFindings(unittest.TestCase):
    def test_gate1_workload_without_a_tolerance(self):
        wl = Workload({"R": 4, "C": 8}, "float16")
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU, workloads=[wl])
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("tolerance_unavailable", codes(report))

    def test_gate2_malformed_policies(self):
        scope = F.toy_module_scope()
        policy = synthetic_policy(scope, ("norm",), lambda s: {"E2": 1e-3, "Emax": 1e-2})
        nan = policy.to_dict()
        nan["gamma"] = float("nan")
        lacking = policy.to_dict()
        for key in ("weights", "policy_hash", "schema"):
            lacking.pop(key)
        with self.assertRaises(CalibrationError):          # hash-valid but missing fields
            Gate2Policy.build(lacking)
        for bad in (nan, {"schema": "evograd-verification-gate2-policy/1"}):
            report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                              policy=bad, samples=samples("m", [3]))
            self.assertEqual(report.status, INCOMPLETE)
            self.assertIn("policy_schema", codes(report))


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
