"""Gate 2 regressions: live boundary gradients (fix A) and reference identity (fix B).

Policies here come from real ``torch.compile(backend="aot_eager")`` calibration
unless a test says otherwise.
"""

from __future__ import annotations

import json
import types
import unittest
import warnings

import torch
from torch import nn

from evograd.verification import gate2
from evograd.verification.common.execution import BoundaryCall, Invocation, Sample, SiteHooks, candidate
from evograd.verification.common.fingerprint import UnidentifiableReference, fingerprint
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.calibration import SCHEMA, CalibrationError, Gate2Policy, calibrate
from evograd.verification.gate2.runner import live_gradient_pairs

warnings.filterwarnings("ignore")
FAST = "aot_eager"
CAL = [Sample("c1", 1), Sample("c2", 2)]
#: Model scope uses matched inputs: fresh executions of the calibrated samples.
VAL = CAL


def native(x):
    return x * 2


def native_pair(a, b):
    return a * b


def invocation(sample, device):
    g = torch.Generator().manual_seed(int(sample.seed))
    return Invocation(args=(torch.randn(4, generator=g).to(device),),
                      differentiable_inputs={"x": "args[0]"})


def ones(out, invocation, sample):
    return {"y": torch.ones_like(out["y"])}


def scope_for(block, sites=None, name="audit", cotangents=ones, calls=None):
    return ModuleScope(scope="model", name=name, factory=block,
                       native_sites=sites or {"s": native}, make_invocation=invocation,
                       select_outputs=lambda y, i, s: {"y": y}, output_names=("y",),
                       make_cotangents=cotangents, calls_per_execution=calls or {})


class MaskingBlock(nn.Module):
    """The outer computation hides the site's input gradient (``x * 0``)."""

    def __init__(self, sites):
        super().__init__()
        self.site = sites["s"]

    def forward(self, x):
        return self.site(x * 0)


class BranchBlock(nn.Module):
    """``x`` reaches the site twice and also a branch outside it."""

    def __init__(self, sites):
        super().__init__()
        self.site = sites["s"]

    def forward(self, x):
        return self.site(x) + 5 * x + self.site(x)


class SharedArgBlock(nn.Module):
    """One tensor passed at two argument positions of one site call."""

    def __init__(self, sites):
        super().__init__()
        self.site = sites["p"]

    def forward(self, x):
        return self.site(x, x)


def scaled_vjp(scale, *, first_only=False):
    """A site whose forward is ``2x`` and whose backward is ``scale * dy`` (2 is right)."""
    state = {"calls": 0}

    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            state["calls"] += 1
            ctx.wrong = (state["calls"] == 1) if first_only else True
            return 2 * x

        @staticmethod
        def backward(ctx, dy):
            return (scale if ctx.wrong else 2.0) * dy

    return Fn.apply


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def failing(report):
    return sorted({r.name for c in report.cases for r in c.results if r.status == FAILED})


class TestLiveBoundaryGradients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = scope_for(MaskingBlock)
        cls.policy = calibrate(cls.scope, patch_set=("s",), samples=CAL, compile_backend=FAST,
                               boundaries=True).policy
        assert cls.policy is not None

    def test_wrong_first_actual_vjp_fails_at_the_live_input_gradient(self):
        report = gate2.verify_integration(self.scope, candidate("bad_once", {
            "s": scaled_vjp(1.0, first_only=True)}), policy=self.policy, samples=VAL)
        self.assertEqual(report.status, FAILED, codes(report))
        # Only the gradient the live backward emitted is wrong: the replays are
        # right and the masked scope gradient is zero on both sides.
        self.assertEqual(failing(report), ["boundary:s#0/grad_in:args[0]~live"])
        self.assertIn("boundary:s#0/grad_in:args[0]", self.policy.thresholds_for("c1"))

    def test_always_correct_candidate_passes_with_live_evidence(self):
        report = gate2.verify_integration(self.scope, candidate("good", {"s": scaled_vjp(2.0)}),
                                          policy=self.policy, samples=VAL)
        self.assertEqual(report.status, PASSED, codes(report))
        live = report.cases[0].result("boundary:s#0/grad_in:args[0]~live")
        self.assertTrue(live.compared)
        self.assertEqual(report.cases[0].evidence["boundaries"]["live_gradients"], {"compared": 1})

    def test_a_consistently_wrong_vjp_fails_live_and_in_replay(self):
        report = gate2.verify_integration(self.scope, candidate("bad", {"s": scaled_vjp(1.0)}),
                                          policy=self.policy, samples=VAL)
        self.assertIn("boundary:s#0/grad_in:args[0]~live", failing(report))
        self.assertIn("boundary:s#0/grad_in:args[0]", failing(report))


class TestAttribution(unittest.TestCase):
    def test_capture_is_this_calls_contribution_not_the_accumulated_grad(self):
        hooks = SiteHooks(record=True)
        block = BranchBlock({"s": hooks.wrap("s", native)})
        x = torch.randn(4, requires_grad=True)
        block(x).backward(torch.ones(4))
        self.assertTrue(torch.equal(x.grad, torch.full((4,), 9.0)))   # 2 + 5 + 2
        self.assertEqual(len(hooks.calls), 2)
        for call in hooks.calls:
            self.assertTrue(torch.equal(call.emitted["args[0]"], torch.full((4,), 2.0)))

    def test_shared_argument_positions_are_attributed_separately(self):
        hooks = SiteHooks(record=True)
        block = SharedArgBlock({"p": hooks.wrap("p", native_pair)})
        x = torch.randn(4, requires_grad=True)
        block(x).backward(torch.ones(4))
        call = hooks.calls[0]
        self.assertEqual(call.grad_paths, ("args[0]", "args[1]"))
        for path in call.grad_paths:
            self.assertTrue(torch.equal(call.emitted[path], x.detach()))
        self.assertTrue(torch.equal(x.grad, 2 * x.detach()))

    def test_branch_and_shared_scopes_end_to_end(self):
        branch = scope_for(BranchBlock, name="branch", calls={"s": 2})
        policy = calibrate(branch, patch_set=("s",), samples=CAL, compile_backend=FAST,
                           boundaries=True).policy
        self.assertIsNotNone(policy)
        good = gate2.verify_integration(branch, candidate("g", {"s": scaled_vjp(2.0)}),
                                        policy=policy, samples=VAL)
        self.assertEqual(good.status, PASSED, codes(good))
        bad = gate2.verify_integration(branch, candidate("b", {"s": scaled_vjp(3.0)}),
                                       policy=policy, samples=VAL)
        self.assertIn("boundary:s#0/grad_in:args[0]~live", failing(bad))
        self.assertIn("boundary:s#1/grad_in:args[0]~live", failing(bad))
        shared = scope_for(SharedArgBlock, sites={"p": native_pair}, name="shared")
        policy = calibrate(shared, patch_set=("p",), samples=CAL, compile_backend=FAST,
                           boundaries=True).policy
        self.assertIsNotNone(policy)
        report = gate2.verify_integration(shared, candidate("g", {"p": lambda a, b: a * b}),
                                          policy=policy, samples=VAL)
        self.assertEqual(report.status, PASSED, codes(report))

    def test_unattributable_and_unreached_calls_are_explicit(self):
        sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, 2.0]),
                                         (3,)).requires_grad_()
        hooks = SiteHooks(record=True)
        out = hooks.wrap("s", lambda t: t.to_dense() * 2)(sparse)
        out.sum().backward()
        call = hooks.calls[0]
        self.assertEqual(call.unattributable, ("args[0]",))
        self.assertIsNotNone(sparse.grad)                             # training flow untouched
        pairs, findings, status = live_gradient_pairs(call, {})
        self.assertEqual((pairs, status), ([], "no_gradient_inputs"))
        self.assertEqual([(f.code, f.outcome) for f in findings],
                         [("boundary_gradient_unattributable", INCOMPLETE)])
        unreached = BoundaryCall("s", 0, ((torch.ones(2),), {}), ("args[0]",), torch.ones(2),
                                 differentiable_outputs=True)
        self.assertEqual(live_gradient_pairs(unreached, {})[2], "no_backward")


def scaled_native(k):
    def native(x):
        return x * k
    return native


def native_with_default(x, k=2.0):
    return x * k


class TestReferenceIdentity(unittest.TestCase):
    def calibrated(self, sites):
        scope = scope_for(MaskingBlock, sites=sites, name="identity")
        result = calibrate(scope, patch_set=("s",), samples=CAL, compile_backend=FAST)
        self.assertIsNotNone(result.policy, result.to_dict())
        return scope, result.policy

    def verify(self, sites, policy, fn=native):
        scope = scope_for(MaskingBlock, sites=sites, name="identity")
        return gate2.verify_integration(scope, candidate("c", {"s": fn}), policy=policy,
                                        samples=VAL)

    def assert_mismatch(self, report):
        self.assertEqual(report.status, INCOMPLETE, codes(report))
        self.assertIn("policy_reference_mismatch", codes(report))
        self.assertEqual(report.evidence["thresholds_withheld"], ["policy_reference_mismatch"])

    def test_changed_reference_under_the_same_name_is_refused(self):
        _, policy = self.calibrated({"s": native})
        self.assertEqual(self.verify({"s": native}, policy).status, PASSED)   # unchanged: accepted

        def changed(x):
            return 3 * x
        changed.__name__, changed.__qualname__ = native.__name__, native.__qualname__
        changed.__module__ = native.__module__
        self.assert_mismatch(self.verify({"s": changed}, policy, fn=changed))

    def test_withheld_thresholds_cannot_fail_numerically(self):
        _, policy = self.calibrated({"s": native})
        report = self.verify({"s": scaled_native(3.0)}, policy, fn=lambda x: 4 * x)
        self.assert_mismatch(report)
        self.assertNotIn("threshold_exceeded", codes(report))
        self.assertTrue(all(r.metrics for r in report.cases[0].results))   # still measured

    def test_captured_constants_and_defaults_are_bound(self):
        _, policy = self.calibrated({"s": scaled_native(2.0)})
        self.assertEqual(self.verify({"s": scaled_native(2.0)}, policy).status, PASSED)
        self.assert_mismatch(self.verify({"s": scaled_native(3.0)}, policy))
        changed_default = types.FunctionType(native_with_default.__code__,
                                             native_with_default.__globals__,
                                             "native_with_default", (3.0,))
        self.assertNotEqual(fingerprint(native_with_default)["digest"],
                            fingerprint(changed_default)["digest"])

    def test_reached_global_values_are_bound(self):
        code = compile("def site(x):\n    return x * K\n", "<site>", "exec")
        made = []
        for k in (2.0, 3.0):
            scope = {"K": k, "__name__": "audit_globals"}
            exec(code, scope)
            made.append(fingerprint(scope["site"])["digest"])
        self.assertNotEqual(made[0], made[1])

    def test_policy_round_trip_keeps_the_binding(self):
        _, policy = self.calibrated({"s": native})
        reloaded = Gate2Policy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(reloaded.reference, policy.reference)
        self.assertEqual(self.verify({"s": native}, reloaded).status, PASSED)
        self.assert_mismatch(self.verify({"s": scaled_native(3.0)}, reloaded))

    def test_policies_without_the_binding_are_refused(self):
        _, policy = self.calibrated({"s": native})
        self.assertEqual(SCHEMA, "evograd-verification-gate2-policy/4")
        data = policy.to_dict()
        data["reference"].pop("implementation")
        data.pop("policy_hash")
        with self.assertRaises(CalibrationError):          # current schema, binding missing
            Gate2Policy.build(data)
        old = dict(data, schema="evograd-verification-gate2-policy/1")
        report = gate2.verify_integration(scope_for(MaskingBlock, name="identity"),
                                          candidate("c", {"s": native}), policy=old, samples=VAL)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("policy_schema", codes(report))

    def test_unidentifiable_reference_cannot_be_calibrated_unless_declared(self):
        class Scale:
            def __init__(self, k):
                self.k = k

            def __call__(self, x):
                return x * self.k

        with self.assertRaises(UnidentifiableReference):
            fingerprint(Scale(2.0))
        scope = scope_for(MaskingBlock, sites={"s": Scale(2.0)}, name="opaque")
        result = calibrate(scope, patch_set=("s",), samples=CAL, compile_backend=FAST)
        self.assertIsNone(result.policy)
        self.assertEqual([f.code for f in result.findings], ["reference_unidentifiable"])
        declared = Scale(2.0)
        declared.__verification_identity__ = {"kind": "Scale", "k": 2.0, "revision": 1}
        scope = scope_for(MaskingBlock, sites={"s": declared}, name="opaque")
        self.assertIsNotNone(calibrate(scope, patch_set=("s",), samples=CAL,
                                       compile_backend=FAST).policy)

    def test_identity_is_content_not_object_identity(self):
        a = scope_for(MaskingBlock, name="identity").reference_identity()
        b = scope_for(MaskingBlock, name="identity").reference_identity()   # new lambdas
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
