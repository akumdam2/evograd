"""Gate 2 at model scope: the ``matched_inputs`` sample protocol.

A model policy is calibrated on fixed samples and judges fresh executions of
those same samples, each against its own per-tensor thresholds, and only when
the sample's content (inputs, labels and other context, cotangents, initial
weights, module mode, seed) equals the calibrated one. The operator protocol
(disjoint holdout) is unchanged. CPU, ``aot_eager`` or a test backend.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
import warnings
from unittest import mock

import torch
from torch import nn

from evograd.verification import gate2
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Invocation, Sample, candidate
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from evograd.verification.common.tensors import relative_error
from evograd.verification.gate2.adapters.model import model_scope, next_token_mean_nll
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.gate2.calibration import CalibrationError, Gate2Policy, calibrate
from tests.verification import _fixtures as F

warnings.filterwarnings("ignore")
FAST = "aot_eager"
PERTURB = "evograd_test_input_dependent_perturbation"


def _perturbing_backend(gm, example_inputs):
    """A test 'compiler' whose result differs from eager by an amount that
    depends on the call's first input: trusted-control error that differs
    between samples. Test-only; never a verification control."""
    def run(*args):
        outs = gm(*args)
        x = next(a for a in args if torch.is_tensor(a) and a.is_floating_point())
        scale = 1.0 + 1e-2 * float(x.detach().abs().mean()) ** 2
        if isinstance(outs, (tuple, list)):
            return type(outs)(o * scale if torch.is_tensor(o) and o.is_floating_point() else o
                              for o in outs)
        return outs * scale
    return run


try:
    torch._dynamo.register_backend(_perturbing_backend, name=PERTURB)
except Exception:  # already registered in this process
    pass


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def failing(report):
    return sorted({r.name for c in report.cases for r in c.results if r.status == FAILED})


def fixed(prefix="lm/fixed", seeds=(41, 42)):
    return [Sample(f"{prefix}/seed{s}", s, {"recipe": "lm_batch"}) for s in seeds]


# ── configurable tiny LM: what the matched check must tell apart ─────────────

_STATE: dict = {}


class MaskedLM(F.TinyLM):
    """The tiny LM with an attention-mask-like input that gates the embeddings."""

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        if attention_mask is not None:
            h = h * attention_mask[..., None].to(h.dtype)
        for i in range(len(self.mix_w)):
            n = self.sites["norm"](h, self.norm_w[i])
            h = h + self.drop(self.sites["mix"](n, self.mix_w[i], self.mix_b[i]))
        return {"logits": self.head(self.sites["norm"](h, self.norm_w[-1]))}


def _factory(sites):
    module = MaskedLM(sites)
    if _STATE.get("checkpoint"):
        module.load_state_dict(torch.load(_STATE["checkpoint"]))
    return module


def _invocation(sample, device):
    """Content controlled by _STATE (read at run time, so the reference code --
    and its identity -- does not change between calibration and use)."""
    ids, labels = F.lm_batch(sample.seed)
    ids = (ids + _STATE.get("token_shift", 0)) % 32
    if _STATE.get("label_mask"):
        labels = labels.clone()
        labels[:, -_STATE["label_mask"]:] = -100
    mask = torch.ones_like(ids)
    if _STATE.get("mask_zero"):
        mask[:, :_STATE["mask_zero"]] = 0
    context = {"labels": labels.to(device)}
    if _STATE.get("loss_context") is not None:
        context["loss_note"] = _STATE["loss_context"]
    return Invocation(args=(ids.to(device),), kwargs={"attention_mask": mask.to(device)},
                      context=context)


def _loss(outputs, invocation, sample):
    return next_token_mean_nll(outputs["logits"], invocation.context["labels"])


def lm_scope(**overrides):
    from evograd.verification.gate2.adapters.module import ModuleScope

    def select(result, invocation, sample):
        out = {"logits": result["logits"]}
        out["loss"] = _loss(out, invocation, sample)
        return out

    def cotangents(outputs, invocation, sample):
        return {"loss": torch.ones_like(outputs["loss"]) * _STATE.get("cotangent_scale", 1.0)}

    args = dict(scope="model", name="masked_lm", factory=_factory, native_sites=dict(F.NATIVE_SITES),
                make_invocation=_invocation, select_outputs=select,
                output_names=("logits", "loss"), make_cotangents=cotangents,
                calls_per_execution={"norm": 3, "mix": 2},
                config={"vocab": 32, "width": 16, "layers": 2})
    args.update(overrides)
    return ModuleScope(**args)


class MatchedCase(unittest.TestCase):
    def setUp(self):
        _STATE.clear()
        self.addCleanup(_STATE.clear)


# ── 1, 2, 3: fresh executions of the same samples, per-sample budgets ────────


class TestPerSampleBudgets(MatchedCase):
    @classmethod
    def setUpClass(cls):
        _STATE.clear()
        cls.scope = lm_scope()
        cls.samples = fixed()
        cls.calibration = calibrate(cls.scope, patch_set=("norm",), samples=cls.samples,
                                    compile_backend=PERTURB)
        cls.policy = cls.calibration.policy
        cls.control, _ = gate2.compile_control(cls.scope, ("norm",), backend=PERTURB)

    def test_same_sample_fresh_candidate_is_judged_without_overlap(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        self.assertEqual(self.policy.sample_protocol, gate2.MATCHED)
        self.assertNotIn("thresholds", self.policy.to_dict())       # never pooled
        report = gate2.verify_integration(self.scope, candidate("same", {"norm": F.norm_site}),
                                          policy=self.policy, samples=self.samples)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertNotIn("sample_overlap", codes(report))
        for case in report.cases:
            self.assertTrue(all(r.limits for r in case.results if r.compared))
            matched = case.evidence["matched_sample"]
            self.assertTrue(matched["identical_content"])
            self.assertEqual(matched["calibrated_fingerprint"], matched["reference_fingerprint"])
            self.assertTrue(matched["reference_reproduces_calibration"])
        executions = {e for c in report.cases
                      for e in (c.evidence["reference"]["execution_id"],
                                c.evidence["candidate"]["execution_id"])}
        self.assertFalse(executions & set(self.policy.calibration_executions))

    def test_each_sample_has_its_own_thresholds_matching_an_independent_computation(self):
        tables = {s.sample_id: self.policy.thresholds_for(s.sample_id) for s in self.samples}
        a, b = (tables[s.sample_id] for s in self.samples)
        self.assertNotEqual(a["out:logits"]["E2"], b["out:logits"]["E2"])
        floors = self.policy.floors
        for sample in self.samples:
            native = gate2.execute(self.scope, Deployment.native(), sample)
            again = gate2.execute(self.scope, Deployment.native(), sample)
            compiled = gate2.execute(self.scope, self.control, sample)
            for name, entry in tables[sample.sample_id].items():
                family = C.FLOOR_FAMILY[entry["kind"]]
                e_rep = relative_error(again.results[name], native.results[name]).to_dict()
                e_cmp = relative_error(compiled.results[name], native.results[name]).to_dict()
                for metric in ("E2", "Emax"):
                    expected = 2.0 * max(e_rep[metric], e_cmp[metric], floors[family][metric])
                    self.assertEqual(entry[metric], expected, (sample.sample_id, name, metric))

    def test_fresh_compile_meets_its_own_budget_with_gamma_applied_once(self):
        for order in (self.samples, list(reversed(self.samples))):
            report = gate2.verify_integration(self.scope, self.control, policy=self.policy,
                                              samples=order)
            self.assertEqual(report.status, PASSED, codes(report))
            bound = 0
            for case in report.cases:
                own = self.policy.thresholds_for(case.case_id)
                for result in case.results:
                    entry = own.get(result.name)
                    if entry is None or not result.metrics:
                        continue
                    self.assertEqual(result.limits, {m: entry[m] for m in ("E2", "Emax")})
                    for metric in ("E2", "Emax"):
                        if entry["components"][metric]["binding"] == "trusted_compile":
                            bound += 1
                            # exactly 1/gamma: the sample's own budget, gamma once
                            self.assertEqual(result.metrics[metric] / result.limits[metric], 0.5)
            self.assertGreater(bound, 0)

    def test_candidate_results_cannot_move_the_frozen_thresholds(self):
        before = (self.policy.policy_hash, self.policy.to_dict())
        for fn in (F.norm_site, F.norm_spike, F.norm_scaled(1.5)):
            gate2.verify_integration(self.scope, candidate("c", {"norm": fn}), policy=self.policy,
                                     samples=self.samples)
        self.assertEqual((self.policy.policy_hash, self.policy.to_dict()), before)
        bad = gate2.verify_integration(self.scope, candidate("spike", {"norm": F.norm_spike}),
                                       policy=self.policy, samples=self.samples)
        self.assertEqual(bad.status, FAILED)
        self.assertEqual(bad.subject["policy"]["policy_hash"], before[0])


# ── 4, 5: content, not ids, binds the budget ─────────────────────────────────


class TestContentBinding(MatchedCase):
    @classmethod
    def setUpClass(cls):
        _STATE.clear()
        cls.scope = lm_scope()
        cls.samples = fixed()
        cls.policy = calibrate(cls.scope, patch_set=("norm",), samples=cls.samples,
                               compile_backend=FAST).policy

    def verify(self, fn=F.norm_spike, *, scope=None, samples=None):
        return gate2.verify_integration(scope or self.scope, candidate("c", {"norm": fn}),
                                        policy=self.policy, samples=samples or self.samples)

    def assert_withheld(self, report, code, differing=None):
        self.assertEqual(report.status, INCOMPLETE, codes(report))
        self.assertIn(code, codes(report))
        self.assertNotIn("threshold_exceeded", codes(report))      # the spike is not judged
        if differing is not None:
            found = [f for c in report.cases for f in c.findings if f.code == code]
            self.assertEqual(found[0].details["differing"], differing)

    def test_unchanged_content_is_judged(self):
        self.assertIsNotNone(self.policy)
        self.assertEqual(self.verify(F.norm_site).status, PASSED)
        self.assertEqual(self.verify().status, FAILED)

    def test_same_id_and_seed_with_other_content_is_refused(self):
        for state, component in (({"token_shift": 1}, "inputs"),
                                 ({"mask_zero": 2}, "inputs"),
                                 ({"label_mask": 3}, "context"),
                                 ({"loss_context": "other"}, "context"),
                                 ({"cotangent_scale": 2.0}, "cotangents")):
            _STATE.clear()
            _STATE.update(state)
            self.assert_withheld(self.verify(), "sample_identity_mismatch", [component])

    def test_other_initial_weights_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MaskedLM(F.NATIVE_SITES)
            with torch.no_grad():
                model.mix_b[0].add_(0.5)
            torch.save(model.state_dict(), f"{tmp}/w.pt")
            _STATE["checkpoint"] = f"{tmp}/w.pt"
            self.assert_withheld(self.verify(), "sample_identity_mismatch", ["initial_state"])

    def test_numerics_and_reference_changes_are_refused(self):
        autocast = lm_scope()
        autocast.autocast_dtype = "bfloat16"
        self.assert_withheld(self.verify(scope=autocast), "policy_numerics_mismatch")
        eval_mode = lm_scope(training=False)
        self.assert_withheld(self.verify(scope=eval_mode), "policy_numerics_mismatch")

        def other_norm(x, w):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        other = lm_scope(native_sites={**F.NATIVE_SITES, "norm": other_norm})
        self.assert_withheld(self.verify(scope=other), "policy_reference_mismatch")

    def test_uncalibrated_or_renamed_samples_and_missing_samples(self):
        same_seed_new_id = [Sample("lm/other/seed41", 41, {"recipe": "lm_batch"}), self.samples[1]]
        report = self.verify(F.norm_site, samples=same_seed_new_id)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("sample_not_calibrated", codes(report))
        self.assertIn("required_sample_missing", codes(report))
        # A wrong candidate: the calibrated sample is judged (and fails), the
        # renamed one -- same seed, same content -- gets no budget at all.
        report = self.verify(samples=same_seed_new_id)
        self.assertEqual(report.status, FAILED)
        unknown_case = next(c for c in report.cases if c.case_id == "lm/other/seed41")
        self.assertEqual(unknown_case.evidence["thresholds_withheld"], ["sample_not_calibrated"])
        self.assertNotIn("threshold_exceeded", {f.code for r in unknown_case.results
                                                for f in r.findings})
        same_id_new_seed = [Sample(self.samples[0].sample_id, 7, {"recipe": "lm_batch"}),
                            self.samples[1]]
        self.assertIn("sample_not_calibrated", codes(self.verify(samples=same_id_new_seed)))
        one = self.verify(F.norm_site, samples=self.samples[:1])
        self.assertEqual(one.status, INCOMPLETE)
        self.assertEqual(codes(one), ["required_sample_missing"])

    def test_sample_order_does_not_mismatch_budgets(self):
        forward = self.verify(F.norm_scaled(1.01))
        backward = self.verify(F.norm_scaled(1.01), samples=list(reversed(self.samples)))
        by_case = lambda r: {c.case_id: {x.name: (x.status, x.limits) for x in c.results}  # noqa: E731
                             for c in r.cases}
        self.assertEqual(by_case(forward), by_case(backward))

    def test_a_reused_execution_is_not_fresh_evidence(self):
        fixed_id = uuid.UUID(int=1)
        with mock.patch("evograd.verification.gate2.runner.uuid.uuid4", return_value=fixed_id):
            policy = calibrate(self.scope, patch_set=("norm",), samples=self.samples,
                               compile_backend=FAST).policy
            report = gate2.verify_integration(self.scope, candidate("c", {"norm": F.norm_spike}),
                                              policy=policy, samples=self.samples)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("calibration_execution_reused", codes(report))
        self.assertNotIn("threshold_exceeded", codes(report))


# ── 6, 7: calibration inputs and policy integrity ────────────────────────────


class TestCalibrationAndPolicy(MatchedCase):
    def test_calibration_takes_no_candidate_and_bad_controls_give_no_policy(self):
        import inspect

        self.assertNotIn("deployment", inspect.signature(calibrate).parameters)
        with self.assertRaises(CalibrationError):
            gate2.ControlMeasurement("candidate", "s", "out:a", C.OUTPUT, 0.0, 0.0)
        with self.assertRaises(CalibrationError):
            gate2.ControlMeasurement("trusted_compile", "s", "out:a", C.OUTPUT, float("nan"), 0.0)

        def nan_norm(x, w):
            return F.norm_site(x, w) * float("nan")
        bad_reference = lm_scope(native_sites={**F.NATIVE_SITES, "norm": nan_norm})
        result = calibrate(bad_reference, patch_set=("mix",), samples=fixed(), compile_backend=FAST)
        self.assertIsNone(result.policy)
        self.assertIn("invalid_reference", {f.code for f in result.findings})

    def test_policy_integrity(self):
        scope = lm_scope()
        policy = calibrate(scope, patch_set=("norm",), samples=fixed(), compile_backend=FAST).policy
        sid = fixed()[0].sample_id

        def rebuilt(mutate):
            data = policy.to_dict()
            mutate(data)
            data.pop("policy_hash")
            data.pop("schema")
            return Gate2Policy.build(data)

        tampered = policy.to_dict()
        tampered["thresholds_by_sample"][sid]["out:loss"]["E2"] *= 10
        with self.assertRaises(CalibrationError):                       # hash
            Gate2Policy.from_dict(tampered)
        with self.assertRaisesRegex(CalibrationError, "gamma \\* max"):  # re-hashed: formula
            rebuilt(lambda d: d["thresholds_by_sample"][sid]["out:loss"].update(E2=1.0))
        with self.assertRaisesRegex(CalibrationError, "exactly the calibration samples"):
            rebuilt(lambda d: d["thresholds_by_sample"].pop(sid))
        with self.assertRaisesRegex(CalibrationError, "exactly the calibration samples"):
            rebuilt(lambda d: d["sample_identities"].pop(sid))
        with self.assertRaisesRegex(CalibrationError, "lacks"):
            rebuilt(lambda d: d.pop("thresholds_by_sample"))
        with self.assertRaises(CalibrationError):
            rebuilt(lambda d: d["thresholds_by_sample"][sid]["out:loss"]["components"]["E2"]
                    .update(trusted_compile=float("inf")))
        v3 = dict(policy.to_dict(), schema="evograd-verification-gate2-policy/3")
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_spike}), policy=v3,
                                          samples=fixed())
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("policy_schema", codes(report))


# ── 8: the two protocols stay apart ──────────────────────────────────────────


class TestProtocolsStayApart(MatchedCase):
    def test_operator_protocol_is_not_relaxed(self):
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device="cpu")
        cal = calibrate(scope, patch_set=("operator",), samples=scope.samples([1, 2]),
                        compile_backend=FAST)
        self.assertEqual(cal.policy.sample_protocol, gate2.DISJOINT)
        good = scope.candidate(F.scale_shift_pair())
        self.assertEqual(gate2.verify_integration(scope, good, policy=cal.policy,
                                                  samples=scope.samples([11])).status, PASSED)
        same = gate2.verify_integration(scope, scope.candidate(F.scale_shift_pair(dw_scale=1.5)),
                                        policy=cal.policy, samples=scope.samples([1, 2]))
        self.assertEqual(same.status, INCOMPLETE)
        self.assertIn("sample_overlap", codes(same))

    def test_a_policy_for_the_other_protocol_is_refused(self):
        scope = lm_scope()
        policy = calibrate(scope, patch_set=("norm",), samples=fixed(), compile_backend=FAST).policy
        data = policy.to_dict()
        sid = fixed()[0].sample_id
        data["sample_protocol"] = gate2.DISJOINT          # forged: pooled, model scope
        data["thresholds"] = data.pop("thresholds_by_sample")[sid]
        data["formula"] = "T[z,m] = gamma * max(max_c E_m(native_repeat_c, native_c), " \
                          "max_c E_m(compile_c, native_c), floor[z,m])"
        for key in ("policy_hash", "schema"):
            data.pop(key)
        forged = Gate2Policy.build(data)
        report = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_spike}),
                                          policy=forged, samples=fixed(seeds=(7,)))
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("policy_protocol_mismatch", codes(report))
        self.assertNotIn("threshold_exceeded", codes(report))


# ── 9: model adapters and combination ────────────────────────────────────────


class TestModelAdapters(MatchedCase):
    def test_model_scope_helper_and_combination(self):
        lm = F.tiny_lm_scope()
        samples = fixed()
        policy = calibrate(lm, patch_set=("norm",), samples=samples, compile_backend=FAST).policy
        model = gate2.verify_integration(lm, candidate("c", {"norm": F.norm_site}), policy=policy,
                                         samples=samples)
        op_scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device="cpu")
        op_cal = calibrate(op_scope, patch_set=("operator",), samples=op_scope.samples([1, 2]),
                           compile_backend=FAST)
        operator = gate2.verify_integration(op_scope, op_scope.candidate(F.scale_shift_pair()),
                                            policy=op_cal.policy, samples=op_scope.samples([11]))
        both = gate2.combine_scopes(("operator", "model"), {"operator": operator, "model": model})
        self.assertEqual(both.status, PASSED, codes(both))
        identity = model.cases[0].evidence["reference"]["sample_identity"]
        self.assertEqual(sorted(identity["components"]), sorted(gate2.IDENTITY_COMPONENTS))
        self.assertEqual(identity["components"]["module_mode"], "train")
        self.assertEqual(identity["opaque"], [])
        self.assertIn("labels", identity["detail"]["context"]["dict"])


if __name__ == "__main__":
    unittest.main()
