"""Gate 1: the local pair against the declared oracle, on CPU."""

from __future__ import annotations

import ast
import json
import pathlib
import unittest
import warnings

import torch

from evograd.opdecl import Workload
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import oracle
from evograd.verification import gate1
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from tests.verification import _fixtures as F

CPU = "cpu"
PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "src" / "evograd" / "verification"


def codes(report):
    return sorted({f.code for f in report.all_findings()})


class TestPairContract(unittest.TestCase):
    def test_correct_pair_passes_every_declared_result(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU)
        self.assertEqual(report.status, PASSED, codes(report))
        names = [r.name for r in report.cases[0].results]
        self.assertEqual(names, ["out:y", "grad_in:dx", "grad_in:dw", "grad_in:db"])
        self.assertEqual(len(report.cases), 2)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_multi_output_pair(self):
        report = gate1.verify_pair(F.SPLIT, F.split_pair(), device=CPU)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual([r.name for r in report.cases[0].results],
                         ["out:hi", "out:lo", "grad_in:dx", "grad_in:dw"])

    def test_wrong_forward_and_wrong_backward_are_rejected_by_name(self):
        fwd = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(fwd_bias=1e-3), device=CPU)
        self.assertEqual(fwd.status, FAILED)
        failing = {r.name for r in fwd.cases[0].results if r.status == FAILED}
        self.assertEqual(failing, {"out:y"})
        bwd = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(dw_scale=1.001), device=CPU)
        self.assertEqual({r.name for r in bwd.cases[0].results if r.status == FAILED},
                         {"grad_in:dw"})
        multi = gate1.verify_pair(F.SPLIT, F.split_pair(lo_bias=1.0), device=CPU)
        self.assertEqual({r.name for r in multi.cases[0].results if r.status == FAILED}, {"out:lo"})

    def test_missing_extra_and_none_gradients(self):
        for kwargs, code in (({"drop_grad": True}, "gradient_arity"),
                             ({"extra_grad": True}, "gradient_arity"),
                             ({"none_grad": True}, "missing_result")):
            report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(**kwargs), device=CPU)
            self.assertEqual(report.status, FAILED, kwargs)
            self.assertIn(code, codes(report), kwargs)
        arity = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(drop_grad=True), device=CPU)
        # Arity failure is not a silent truncation: every gradient is reported missing.
        missing = {r.name for r in arity.cases[0].results
                   if any(f.code == "missing_result" for f in r.findings)}
        self.assertEqual(missing, {"grad_in:dx", "grad_in:dw", "grad_in:db"})

    def test_shape_and_dtype(self):
        dtype = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(wrong_dtype=True), device=CPU)
        self.assertIn("dtype_mismatch", codes(dtype))
        shape = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(wrong_shape=True), device=CPU)
        self.assertIn("shape_mismatch", codes(shape))

    def test_non_finite_candidate_fails_and_json_stays_finite(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(nan_output=True), device=CPU)
        self.assertEqual(report.status, FAILED)
        self.assertIn("non_finite_candidate", codes(report))
        json.dumps(report.to_dict(), allow_nan=False)

    def test_non_finite_reference_is_incomplete_where_allclose_said_ok(self):
        """Finite hardening: ``allclose`` treats inf == inf as close, so the old
        comparator accepted this output. The new core calls the reference
        invalid and does not pass."""
        wl = F.SCALE_SHIFT.correctness[0]
        values = make_case_inputs(F.SCALE_SHIFT, wl, device=CPU)
        values["b"] = values["b"].clone()
        values["b"][0] = float("inf")
        case = gate1.Gate1Case("inf_input", wl, values)
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU, cases=[case])
        self.assertEqual(report.status, INCOMPLETE, codes(report))
        self.assertIn("invalid_reference", codes(report))
        self.assertNotIn("non_finite_candidate", codes(report))
        from evograd.opdecl.verify import _check as legacy_check
        y_ref, _ = oracle(F.SCALE_SHIFT, values)
        y_cand, _ = F.scale_shift_pair().forward_with_saved(values["x"], values["w"], values["b"])
        # The old comparator calls the inf-against-inf output correct.
        self.assertTrue(legacy_check("y", y_cand, y_ref, 1e-5, 1e-5).ok)

    def test_empty_case_set_never_passes(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU, cases=[])
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("no_cases", codes(report))

    def test_input_mutation_is_a_contract(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(mutate_input=True), device=CPU)
        self.assertIn("input_mutation", codes(report))
        upstream = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(mutate_upstream=True),
                                     device=CPU)
        self.assertIn("input_mutation", codes(upstream))
        recorded = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(mutate_upstream=True),
                                     device=CPU, input_mutation=gate1.RECORD)
        self.assertEqual(recorded.status, PASSED)
        self.assertEqual(recorded.cases[0].evidence["input_mutation"]["changed"], ["dy"])

    def test_verification_leaves_the_global_rng_alone(self):
        torch.manual_seed(123)
        state = torch.get_rng_state()
        gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))


class TestReferenceAndTolerance(unittest.TestCase):
    def test_reference_dtype_promotion_is_kept(self):
        # A candidate that computes in float32 and rounds once equals the
        # float32-then-cast oracle; the same-dtype oracle rounds x + 1000 to
        # bfloat16 and loses x.
        promoted = gate1.verify_pair(F.CANCEL_F32_REF, F.cancel_f32_pair(), device=CPU)
        self.assertEqual(promoted.status, PASSED, codes(promoted))
        same = gate1.verify_pair(F.CANCEL_SAME_DTYPE, F.cancel_f32_pair(), device=CPU)
        self.assertEqual(same.status, FAILED)
        self.assertEqual({r.name for r in same.cases[0].results if r.status == FAILED}, {"out:y"})
        values = make_case_inputs(F.CANCEL_F32_REF, F.CANCEL_F32_REF.correctness[0], device=CPU)
        y_ref, _ = oracle(F.CANCEL_F32_REF, values)
        self.assertEqual(y_ref.dtype, torch.bfloat16)               # cast back to the contract

    def test_per_result_multiplier_and_shape_hook_are_the_declarations(self):
        report = gate1.verify_pair(F.SPLIT, F.split_pair(), device=CPU)
        wl = F.SPLIT.correctness[0]
        for result in report.cases[0].results:
            declared = result.name.split(":", 1)[1]
            atol, rtol = F.SPLIT.tolerance_for(wl, declared)
            self.assertEqual((result.limits["atol"], result.limits["rtol"]), (atol, rtol))
        limits = {r.name: r.limits["atol"] for r in report.cases[0].results}
        self.assertAlmostEqual(limits["grad_in:dw"], 3e-5, places=15)            # multiplier
        self.assertAlmostEqual(limits["out:lo"], 1e-5 * wl.dims["C"], places=15)  # shape hook

    def test_an_error_inside_the_multiplied_tolerance_passes(self):
        wl = F.SPLIT.correctness[0]
        values = make_case_inputs(F.SPLIT, wl, device=CPU)
        _, grads = oracle(F.SPLIT, values)
        atol, rtol = F.SPLIT.tolerance_for(wl, "dw")
        bound = float((atol + rtol * grads["dw"].abs()).min())

        def shifted(delta):
            pair = F.split_pair()
            backward = pair.backward_from_saved

            def bwd(output_grads, saved, alpha=2.0):
                dx, dw = backward(output_grads, saved, alpha)
                return dx, dw + delta
            pair.backward_from_saved = bwd
            return pair

        self.assertEqual(gate1.verify_pair(F.SPLIT, shifted(0.5 * bound), device=CPU).status, PASSED)
        self.assertEqual(gate1.verify_pair(F.SPLIT, shifted(2.0 * atol + 1e-3), device=CPU).status,
                         FAILED)


class TestEntries(unittest.TestCase):
    def test_module_and_provider_reach_the_same_core(self):
        provider = F.CountingProvider(F.scale_shift_pair())
        via_provider = gate1.verify_pair(F.SCALE_SHIFT, provider, device=CPU)
        via_module = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU)
        self.assertEqual(via_provider.status, PASSED, codes(via_provider))
        self.assertEqual(len(provider.prepared), 2)                 # prepare really ran
        self.assertEqual(via_provider.subject["candidate_identity"]["adapter_kind"], "test_provider")
        self.assertEqual([r.metrics for r in via_provider.cases[0].results],
                         [r.metrics for r in via_module.cases[0].results])

    def test_declared_baseline_hook(self):
        import types

        pair = F.scale_shift_pair()
        hook = types.SimpleNamespace(
            pair_factory=lambda: (pair.forward_with_saved, pair.backward_from_saved),
            forward_args=("x", "w", "b"), backward_extras=())
        report = gate1.verify_pair(F.SCALE_SHIFT, gate1.pair_from_baseline(F.SCALE_SHIFT, "toy", hook),
                                   device=CPU)
        self.assertEqual(report.status, PASSED, codes(report))
        bad = F.scale_shift_pair(dw_scale=1.1)
        hook.pair_factory = lambda: (bad.forward_with_saved, bad.backward_from_saved)
        self.assertEqual(gate1.verify_pair(F.SCALE_SHIFT, gate1.pair_from_baseline(
            F.SCALE_SHIFT, "toy", hook), device=CPU).status, FAILED)

    def test_supplied_case_variants(self):
        op = F.SCALE_SHIFT
        wl = op.correctness[0]
        base = gate1.generated_case(op, wl, device=CPU)
        shifted = gate1.generated_case(op, wl, device=CPU, seed_offset=104729)
        self.assertEqual(sorted(shifted.provenance["changed_tensors"]), ["b", "dy", "w", "x"])
        cases = [base, shifted, *gate1.upstream_variants(op, base, device=CPU)]
        report = gate1.verify_pair(op, F.scale_shift_pair(), device=CPU, cases=cases)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(len(report.cases), 4)
        bad = gate1.verify_pair(op, F.scale_shift_pair(dw_scale=1.01), device=CPU, cases=cases)
        self.assertEqual(bad.counts()["failed"], 4)

    def test_stress_probe_with_invalid_reference_is_skipped_not_passed(self):
        op = F.SUMSQ_SAME_DTYPE
        base = gate1.generated_case(op, op.correctness[0], device=CPU)
        probes = gate1.stress_probes(op, base, ["x"], device=CPU, magnitude=1e30)
        report = gate1.verify_pair(op, F.sumsq_f32_pair(), device=CPU, cases=probes)
        self.assertEqual({c.status for c in report.cases}, {"skipped"})
        self.assertEqual(report.status, INCOMPLETE)                # nothing decisive ran

    def test_supplied_inputs_must_match_the_declaration(self):
        wl = F.SCALE_SHIFT.correctness[0]
        values = make_case_inputs(F.SCALE_SHIFT, wl, device=CPU)
        values["x"] = values["x"][:, :-1]
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU,
                                   cases=[gate1.Gate1Case("bad", wl, values)])
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("invalid_case_inputs", codes(report))

    def test_smoke_mode_is_never_a_pass(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(), device=CPU, mode=gate1.SMOKE)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("smoke_only", codes(report))
        broken = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(wrong_shape=True), device=CPU,
                                   mode=gate1.SMOKE)
        self.assertEqual(broken.status, FAILED)

    def test_evolution_summary_and_saved_state_observer(self):
        report = gate1.verify_pair(F.SCALE_SHIFT, F.scale_shift_pair(dw_scale=1.01), device=CPU,
                                   observers={"saved_state": gate1.saved_state_observer})
        summary = gate1.evolution_correctness(report, F.SCALE_SHIFT)
        self.assertEqual((summary["passed"], summary["total"]), (0, 2))
        self.assertEqual(summary["forward_correctness"], 1.0)
        self.assertEqual(summary["dw_correctness"], 0.0)
        self.assertEqual(summary["dx_correctness"], 1.0)
        saved = report.cases[0].evidence["saved_state"]
        self.assertGreater(saved["logical_saved_bytes"], 0)
        self.assertGreater(saved["retained_input_bytes"], 0)       # it saves x and w as-is


class TestAgainstTheOldVerifier(unittest.TestCase):
    """Targeted agreement on finite, legal samples of real declarations."""

    OPS = ("layernorm", "rmsnorm", "softmax", "swiglu", "cross_entropy", "geglu",
           "fused_add_rms_norm", "llama3_swiglu_mlp")

    def test_same_verdicts_for_correct_and_perturbed_pairs(self):
        from evograd.benchmark import get_task
        from evograd.evaluation.tier3.patch import eager_pair_for
        from evograd.opdecl.verify import verify as old_verify

        warnings.filterwarnings("ignore")
        for name in self.OPS:
            op = get_task(name)
            workloads = tuple(w for w in op.correctness if w.dtype == "float32")[:2]
            if not workloads:
                continue
            for scale in (1.0, 1.05):
                pair = eager_pair_for(op)
                if scale != 1.0:
                    backward = pair.backward_from_saved

                    def scaled(dout, saved, _b=backward, _s=scale, **kw):
                        grads = _b(dout, saved, **kw)
                        return (grads[0] * _s, *grads[1:])
                    pair.backward_from_saved = scaled
                new = gate1.verify_pair(op, pair, device=CPU, workloads=workloads)
                old = old_verify(op, pair, device=CPU, workloads=workloads)
                self.assertEqual(new.status == PASSED, old.ok, (name, scale, codes(new)))
                self.assertEqual(new.status, PASSED if scale == 1.0 else FAILED, (name, scale))


class TestIndependence(unittest.TestCase):
    """The new cores do not call back into the old verifiers or gates."""

    FORBIDDEN = ("evograd.opdecl.verify", "evograd.evolve", "evograd.pipelines.unified",
                 "evograd.evaluation.common.provider", "evograd.evaluation.tier3.gate",
                 "evograd.evaluation.tier3.runner", "evograd.evaluation.tier3.block",
                 "evograd.opdecl.baselines")
    FORBIDDEN_NAMES = {"check_module", "check_candidate", "model_correctness",
                       "check_model_correctness", "verify_pair_provider", "_run_correctness",
                       "validate_numerical_variants", "run_training", "training_distances"}

    def test_no_import_of_old_verifiers(self):
        for path in PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for bad in self.FORBIDDEN:
                        self.assertFalse(node.module == bad or node.module.startswith(bad + "."),
                                         f"{path.name} imports {node.module}")
                    names = {a.name for a in node.names}
                    self.assertFalse(names & self.FORBIDDEN_NAMES, f"{path.name}: {names}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for bad in self.FORBIDDEN:
                            self.assertFalse(alias.name.startswith(bad), path.name)


if __name__ == "__main__":
    unittest.main()
