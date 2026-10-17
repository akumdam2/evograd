"""Gate 2: one autograd forward/backward per scope, against native eager, on CPU.

Calibrated policies here use ``torch.compile(backend="aot_eager")`` -- a real
dynamo + AOTAutograd path, chosen for speed -- unless a test says otherwise.
``TestInductorControl`` runs the default inductor backend and skips, with the
reason, when this machine cannot build its C++ kernels (set ``CXX`` to a
working compiler). Tests marked *synthetic* freeze hand-written thresholds to
exercise arithmetic and binding; they are not calibration evidence.
"""

from __future__ import annotations

import json
import unittest
import warnings

import torch

from evograd.verification import gate1, gate2
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.common.reports import (
    FAILED,
    INCOMPLETE,
    PASSED,
    REPORT_FIRST,
    STRICT,
    continue_after,
)
from evograd.verification.common.tensors import module_state_digest
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.gate2.calibration import (
    DEFAULT_FLOORS,
    CalibrationError,
    ControlMeasurement,
    Gate2Policy,
    calibrate,
    derive_thresholds,
    freeze_policy,
)
from tests.verification import _fixtures as F

warnings.filterwarnings("ignore")
CPU = "cpu"
FAST = "aot_eager"


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def failing(report):
    return sorted({r.name for c in report.cases for r in c.results if r.status == FAILED})


def samples(prefix, seeds):
    return [Sample(f"{prefix}/seed{s}", s, {"recipe": "fixture"}) for s in seeds]


#: Floors low enough that hand-written thresholds follow the formula.
SYNTHETIC_FLOORS = {"output": {"E2": 1e-12, "Emax": 1e-12},
                    "gradient": {"E2": 1e-12, "Emax": 1e-12}}


def synthetic_entry(kind, limit, gamma=2.0):
    """A threshold entry whose components reproduce ``limit`` under the formula."""
    floor = SYNTHETIC_FLOORS[C.FLOOR_FAMILY[kind]]
    entry = {"kind": kind, "components": {}}
    for metric in ("E2", "Emax"):
        entry[metric] = limit[metric]
        entry["components"][metric] = {"native_repeat": limit[metric] / gamma,
                                       "trusted_compile": 0.0, "floor": floor[metric],
                                       "gamma": gamma, "binding": "native_repeat"}
    return entry


def synthetic_policy(scope, patch_set, thresholds_for, *, calibration=(("synthetic/cal", 900),),
                     boundaries=False):
    """Freeze hand-written thresholds through the real binding. Not calibration.

    The scope's sample protocol decides the layout: pooled thresholds for an
    operator, the same table for each (natively executed) sample for a model.
    """
    cal = [Sample(i, s) for i, s in calibration]
    natives = {s.sample_id: gate2.execute(scope, Deployment.native(), s) for s in cal}
    contract = gate2.integration_contract(natives[cal[0].sample_id])
    table = {s.name: synthetic_entry(s.kind, thresholds_for(s))
             for s in contract.compared() if not s.exact}
    matched = gate2.SAMPLE_PROTOCOLS[scope.scope] == gate2.MATCHED
    identities = {sid: {**ex.sample_identity, "reference_results_digest": ex.results_digest}
                  for sid, ex in natives.items()}
    return freeze_policy(scope, patch_set=patch_set, contract=contract, samples=cal,
                         thresholds=None if matched else table,
                         thresholds_by_sample={sid: table for sid in natives} if matched else None,
                         sample_identities=identities,
                         input_digests={sid: ex.input_digest for sid, ex in natives.items()},
                         controls={"synthetic": True}, gamma=2.0, floors=SYNTHETIC_FLOORS,
                         boundaries=boundaries)


class TestOperatorScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device=CPU)
        cls.calibration = calibrate(cls.scope, patch_set=("operator",),
                                    samples=cls.scope.samples([1, 2]), compile_backend=FAST)
        cls.mul_scope = OperatorScope(F.MUL, F.MUL.correctness[0], device=CPU)
        cls.mul_calibration = calibrate(cls.mul_scope, patch_set=("operator",),
                                        samples=cls.mul_scope.samples([1, 2]), compile_backend=FAST)

    def test_calibrated_policy_and_correct_legacy_pair(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        policy = self.calibration.policy
        self.assertEqual(sorted(policy.thresholds), ["grad:b", "grad:w", "grad_in:x", "out:y"])
        report = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair()),
                                          policy=policy, samples=self.scope.samples([11, 12]))
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(report.cases[0].evidence["candidate"]["deployment"]["identity"]
                         ["adapter_kind"], "legacy_bind_pair")
        self.assertEqual(report.subject["reference"]["kind"], "native_eager")
        json.dumps(report.to_dict(), allow_nan=False)

    def test_wrong_backward_fails_by_name(self):
        report = gate2.verify_integration(
            self.scope, self.scope.candidate(F.scale_shift_pair(dw_scale=1.01)),
            policy=self.calibration.policy, samples=self.scope.samples([11]))
        self.assertEqual(report.status, FAILED)
        self.assertEqual(failing(report), ["grad:w"])

    def test_correct_pair_but_wrong_autograd_routing_is_rejected(self):
        artifact = F.mul_artifact(swap=True)
        direct = gate1.verify_pair(F.MUL, artifact, device=CPU)
        self.assertEqual(direct.status, PASSED)                    # the pair itself is right
        good = gate2.verify_integration(self.mul_scope, self.mul_scope.candidate(F.mul_artifact()),
                                        policy=self.mul_calibration.policy,
                                        samples=self.mul_scope.samples([21]))
        self.assertEqual(good.status, PASSED, codes(good))
        self.assertEqual(good.cases[0].evidence["candidate"]["deployment"]["identity"]
                         ["adapter_kind"], "direct_deployment_entry")
        bad = gate2.verify_integration(self.mul_scope, self.mul_scope.candidate(artifact),
                                       policy=self.mul_calibration.policy,
                                       samples=self.mul_scope.samples([21]))
        self.assertEqual(bad.status, FAILED)
        self.assertEqual(failing(bad), ["grad:b", "grad_in:a"])

    def test_no_policy_measures_but_cannot_pass(self):
        report = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair()),
                                          policy=None, samples=self.scope.samples([11]))
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("no_policy", codes(report))
        self.assertTrue(all(r.metrics for r in report.cases[0].results))


class TestJudge(unittest.TestCase):
    """The single comparator, on hand-made tensors."""

    LIMIT = {"E2": 1e-2, "Emax": 1e-1}

    def judge(self, cand, ref, limit=LIMIT, **spec):
        return gate2.judge(C.ResultSpec("out:z", C.OUTPUT, **spec), cand, ref, limit)

    def test_small_e2_but_one_large_element_fails_on_emax(self):
        g = torch.Generator().manual_seed(0)
        ref = torch.randn(10_000, generator=g)
        cand = ref.clone()
        cand[17] += 0.5                                   # RMS(ref) ~ 1
        check = self.judge(cand, ref)
        self.assertLess(check.metrics["E2"], self.LIMIT["E2"])
        self.assertEqual([(f.code, f.details["metric"]) for f in check.findings],
                         [("threshold_exceeded", "Emax")])

    def test_zero_reference(self):
        zero = torch.zeros(5)
        self.assertEqual(self.judge(zero.clone(), zero).status, PASSED)
        self.assertEqual(self.judge(torch.full((5,), 1e-12), zero).status, FAILED)

    def test_non_finite_sides_and_exact_results(self):
        ref = torch.ones(4)
        nan = ref.clone()
        nan[0] = float("nan")
        self.assertEqual(self.judge(nan, ref).status, FAILED)
        self.assertEqual(self.judge(ref.clone(), nan).status, INCOMPLETE)
        ints = torch.tensor([1, 2])
        self.assertEqual(self.judge(ints.clone(), ints, None).status, PASSED)   # exact: no threshold
        self.assertEqual(self.judge(torch.tensor([1, 3]), ints, None).status, FAILED)

    def test_missing_or_invalid_threshold_is_incomplete(self):
        ref = torch.ones(3)
        self.assertEqual(self.judge(ref.clone(), ref, None).status, INCOMPLETE)
        self.assertEqual(self.judge(ref.clone(), ref, {"E2": float("nan"), "Emax": 1.0}).status,
                         INCOMPLETE)


class TestModuleScope(unittest.TestCase):
    """A small network at model scope: matched inputs, the same samples on both sides."""

    @classmethod
    def setUpClass(cls):
        cls.scope = F.toy_module_scope()
        cls.validation = samples("net/fixed", [31, 32])
        cls.calibration = calibrate(cls.scope, patch_set=("norm",), samples=cls.validation,
                                    compile_backend=FAST)
        cls.policy = cls.calibration.policy

    def verify(self, fn, scope=None):
        return gate2.verify_integration(scope or self.scope, candidate("cand", {"norm": fn}),
                                        policy=self.policy, samples=self.validation)

    def test_contract_and_correct_candidate(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        contract = self.policy.contract["results"]
        by_name = {r["name"]: r for r in contract}
        self.assertTrue(by_name["meta:tag"]["exact"])
        self.assertEqual(by_name["grad:unused"]["presence"], C.ABSENT)
        self.assertEqual(self.policy.sample_protocol, gate2.MATCHED)
        for sample in self.validation:
            self.assertNotIn("grad:unused", self.policy.thresholds_for(sample.sample_id))
        report = self.verify(lambda x, w: F.norm_site(x, w))
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(report.cases[0].evidence["candidate"]["site_calls"], {"norm": 2})

    def test_faults_are_rejected_each_by_its_own_finding(self):
        cases = {
            "spike": (F.norm_spike, "threshold_exceeded"),
            "nan": (F.norm_nan, "non_finite_candidate"),
            "detached": (F.norm_detached, "missing_result"),
            "misrouted": (F.norm_misrouted, "threshold_exceeded"),
            "writes_parameter": (F.norm_writes_parameter, "parameters_modified"),
        }
        for label, (fn, code) in cases.items():
            report = self.verify(fn)
            self.assertEqual(report.status, FAILED, label)
            self.assertIn(code, codes(report), label)
        misrouted = self.verify(F.norm_misrouted)
        self.assertEqual(failing(misrouted), ["grad:norm_w.0", "grad:norm_w.1"])

    def test_insufficient_coverage(self):
        broken = F.toy_module_scope(factory=lambda sites: F.ToyBlock(sites, skip_last_norm=True))
        report = self.verify(lambda x, w: F.norm_site(x, w), scope=broken)
        self.assertEqual(report.status, FAILED)
        self.assertIn("coverage_mismatch", codes(report))

    def test_no_optimizer_step_and_untouched_initial_state(self):
        fresh = F.ToyBlock(F.NATIVE_SITES)
        before = module_state_digest(fresh)
        torch.manual_seed(5)
        rng = torch.get_rng_state()
        report = self.verify(lambda x, w: F.norm_site(x, w))
        self.assertEqual(report.status, PASSED)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(module_state_digest(F.ToyBlock(F.NATIVE_SITES)), before)
        self.assertEqual(report.cases[0].evidence["candidate"]["state_changed"], [])

    def test_report_first_keeps_the_failure(self):
        report = self.verify(F.norm_spike)
        self.assertEqual(report.status, FAILED)
        self.assertTrue(continue_after(report, REPORT_FIRST))
        self.assertFalse(continue_after(report, STRICT))
        self.assertEqual(report.to_dict()["status"], FAILED)
        json.dumps(report.to_dict(), allow_nan=False)


class TestModelScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = F.tiny_lm_scope()
        cls.fixed = samples("lm/fixed", [41, 42])
        cls.calibration = calibrate(cls.scope, patch_set=("norm",), samples=cls.fixed,
                                    compile_backend=FAST)

    def test_tied_parameters_are_one_accumulated_result(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        results = {r["name"]: r for r in self.calibration.policy.contract["results"]}
        self.assertEqual(results["grad:embed.weight"]["aliases"], ["head.weight"])
        self.assertNotIn("grad:head.weight", results)
        sample = Sample("lm/val/seed41", 41)
        native = gate2.execute(self.scope, Deployment.native(), sample)
        model = F.TinyLM(F.NATIVE_SITES)
        ids, labels = F.lm_batch(41)
        from evograd.verification.gate2.adapters.model import next_token_mean_nll
        next_token_mean_nll(model(ids)["logits"], labels).backward()
        self.assertTrue(torch.equal(native.results["grad:embed.weight"], model.embed.weight.grad))

    def test_correct_and_wrong_candidates(self):
        policy = self.calibration.policy
        good = gate2.verify_integration(self.scope, candidate("same", {"norm": lambda x, w: F.norm_site(x, w)}),
                                        policy=policy, samples=self.fixed)
        self.assertEqual(good.status, PASSED, codes(good))
        self.assertEqual(good.cases[0].evidence["candidate"]["site_calls"], {"norm": 3})
        self.assertTrue(good.cases[0].evidence["matched_sample"]["identical_content"])
        self.assertTrue(good.cases[0].evidence["matched_sample"]["reference_reproduces_calibration"])
        bad = gate2.verify_integration(self.scope, candidate("scaled", {"norm": F.norm_scaled(1.01)}),
                                       policy=policy, samples=self.fixed)
        self.assertEqual(bad.status, FAILED)
        self.assertIn("out:loss", failing(bad))
        self.assertIn("grad:embed.weight", failing(bad))

    def test_dropout_randomness_is_matched_and_isolated(self):
        scope = F.tiny_lm_scope(dropout=0.2)
        a = gate2.execute(scope, Deployment.native(), Sample("d", 7))
        b = gate2.execute(scope, candidate("same", {"norm": lambda x, w: F.norm_site(x, w)}),
                          Sample("d", 7))
        self.assertTrue(torch.equal(a.results["out:loss"], b.results["out:loss"]))
        c = gate2.execute(scope, Deployment.native(), Sample("e", 8))
        self.assertFalse(torch.equal(a.results["out:loss"], c.results["out:loss"]))


class TestPolicyBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = F.toy_module_scope()
        cls.policy = synthetic_policy(cls.scope, ("norm",), lambda s: {"E2": 1e-3, "Emax": 1e-2})
        cls.good = candidate("same", {"norm": lambda x, w: F.norm_site(x, w)})

    def run_with(self, policy, *, scope=None, deployment=None, sample_list=None):
        return gate2.verify_integration(scope or self.scope, deployment or self.good, policy=policy,
                                        samples=sample_list or [Sample("synthetic/cal", 900)])

    def test_synthetic_policy_passes_when_everything_matches(self):
        self.assertEqual(self.run_with(self.policy).status, PASSED)

    def test_operator_sample_overlap_by_id_or_seed(self):
        # The operator protocol is unchanged: calibration and validation stay disjoint.
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device=CPU)
        policy = synthetic_policy(scope, ("operator",), lambda s: {"E2": 1e-3, "Emax": 1e-2})
        self.assertEqual(policy.sample_protocol, gate2.DISJOINT)
        good = scope.candidate(F.scale_shift_pair())
        self.assertEqual(gate2.verify_integration(scope, good, policy=policy,
                                                  samples=[Sample("other", 5)]).status, PASSED)
        by_id = gate2.verify_integration(scope, good, policy=policy,
                                         samples=[Sample("synthetic/cal", 5)])
        by_seed = gate2.verify_integration(scope, good, policy=policy, samples=[Sample("other", 900)])
        wrong = gate2.verify_integration(scope, scope.candidate(F.scale_shift_pair(dw_scale=1.5)),
                                         policy=policy, samples=[Sample("other", 900)])
        for report in (by_id, by_seed, wrong):
            self.assertEqual(report.status, INCOMPLETE)          # withheld, never judged
            self.assertIn("sample_overlap", codes(report))

    def test_environment_config_numerics_and_patch_set_mismatch(self):
        data = self.policy.to_dict()
        data["environment"] = dict(data["environment"], torch="0.0.0")
        data.pop("policy_hash")
        data.pop("schema")
        moved = Gate2Policy.build(data)
        self.assertIn("policy_environment_mismatch", codes(self.run_with(moved)))
        # A binding mismatch withholds the numbers: a wrong candidate is not judged either.
        wrong = self.run_with(moved, deployment=candidate("bad", {"norm": F.norm_spike}))
        self.assertEqual(wrong.status, INCOMPLETE)
        self.assertNotIn("threshold_exceeded", codes(wrong))
        other_config = F.toy_module_scope(calls={"norm": 2, "mix": 2})
        other_config.config = {**other_config.config, "width": 17}
        self.assertIn("policy_case_mismatch", codes(self.run_with(self.policy, scope=other_config)))
        autocast = F.toy_module_scope()
        autocast.autocast_dtype = "bfloat16"
        self.assertIn("policy_numerics_mismatch", codes(self.run_with(self.policy, scope=autocast)))
        both = candidate("both", {"norm": F.norm_site, "mix": F.mix_site})
        self.assertIn("policy_patch_set_mismatch", codes(self.run_with(self.policy, deployment=both)))

    def test_tampered_or_old_schema_policies_are_refused(self):
        data = self.policy.to_dict()
        data["thresholds_by_sample"]["synthetic/cal"]["out:y"]["E2"] = 1.0   # widened after freezing
        with self.assertRaises(CalibrationError):
            Gate2Policy.from_dict(data)
        report = self.run_with(data)
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("policy_schema", codes(report))
        # Re-hashed after widening: the entry no longer follows its own components.
        for key in ("policy_hash", "schema"):
            data.pop(key)
        with self.assertRaisesRegex(CalibrationError, "gamma \\* max"):
            Gate2Policy.build(data)
        formula = self.policy.to_dict()
        formula["formula"] = "T = anything"
        for key in ("policy_hash", "schema"):
            formula.pop(key)
        with self.assertRaisesRegex(CalibrationError, "formula"):
            Gate2Policy.build(formula)
        old = {"schema": "evograd-t3-block-policy/2", "entries": {}, "policy_hash": "x"}
        self.assertIn("policy_schema", codes(self.run_with(old)))
        v3 = self.policy.to_dict()
        v3["schema"] = "evograd-verification-gate2-policy/3"
        self.assertIn("policy_schema", codes(self.run_with(v3)))

    def test_policy_is_immutable(self):
        with self.assertRaises(Exception):
            self.policy.text = "{}"
        thresholds = self.policy.thresholds_for("synthetic/cal")
        thresholds["out:y"]["E2"] = 1.0
        self.assertEqual(self.policy.thresholds_for("synthetic/cal")["out:y"]["E2"], 1e-3)


class TestCalibrationArithmetic(unittest.TestCase):
    """*Synthetic* measurements: the formula, per-tensor matching, refusals."""

    def m(self, source, result, e2, emax, sample="s1", kind=C.OUTPUT):
        return ControlMeasurement(source, sample, result, kind, e2, emax)

    def test_formula_gamma_and_floors(self):
        ms = [self.m("native_repeat", "out:a", 1e-4, 5e-3), self.m("native_repeat", "out:a", 2e-4, 1e-3, "s2"),
              self.m("trusted_compile", "out:a", 3e-4, 1e-4),
              self.m("native_repeat", "grad:b", 0.0, 0.0, kind=C.PARAM_GRAD),
              self.m("trusted_compile", "grad:b", 0.0, 0.0, kind=C.PARAM_GRAD)]
        t = derive_thresholds(ms, results={"out:a": C.OUTPUT, "grad:b": C.PARAM_GRAD}, gamma=2.0)
        self.assertAlmostEqual(t["out:a"]["E2"], 2 * 3e-4)
        self.assertAlmostEqual(t["out:a"]["Emax"], 2 * 5e-3)
        self.assertEqual(t["out:a"]["components"]["E2"]["binding"], "trusted_compile")
        self.assertEqual((t["grad:b"]["E2"], t["grad:b"]["Emax"]), (2e-4, 2e-2))   # gradient floors
        self.assertEqual(t["grad:b"]["components"]["E2"]["binding"], "floor")

    def test_thresholds_are_per_tensor(self):
        ms = [self.m("native_repeat", "out:a", 0.5, 0.5), self.m("trusted_compile", "out:a", 0.5, 0.5),
              self.m("native_repeat", "out:b", 0.0, 0.0), self.m("trusted_compile", "out:b", 0.0, 0.0)]
        t = derive_thresholds(ms, results={"out:a": C.OUTPUT, "out:b": C.OUTPUT})
        self.assertEqual(t["out:a"]["E2"], 1.0)
        self.assertEqual(t["out:b"]["E2"], 2 * DEFAULT_FLOORS["output"]["E2"])

    def test_invalid_gamma_and_floors(self):
        ms = [self.m("native_repeat", "out:a", 0.0, 0.0), self.m("trusted_compile", "out:a", 0.0, 0.0)]
        for gamma in (-1.0, 0.5, float("nan"), float("inf"), True, "2"):
            with self.assertRaises(CalibrationError, msg=repr(gamma)):
                derive_thresholds(ms, results={"out:a": C.OUTPUT}, gamma=gamma)
        for bad in ({"output": {"E2": 0.0, "Emax": 1e-3}, "gradient": DEFAULT_FLOORS["gradient"]},
                    {"output": DEFAULT_FLOORS["output"]}):
            with self.assertRaises(CalibrationError):
                derive_thresholds(ms, results={"out:a": C.OUTPUT}, floors=bad)

    def test_candidate_measurements_cannot_calibrate(self):
        with self.assertRaises(CalibrationError):
            ControlMeasurement("candidate", "s", "out:a", C.OUTPUT, 0.0, 0.0)
        check = gate2.judge(C.ResultSpec("out:a", C.OUTPUT), torch.ones(2), torch.ones(2), None)
        with self.assertRaises(TypeError):
            derive_thresholds([check], results={"out:a": C.OUTPUT})

    def test_missing_control_evidence_is_an_error_not_a_fallback(self):
        with self.assertRaises(CalibrationError):
            derive_thresholds([self.m("native_repeat", "out:a", 0.0, 0.0)],
                              results={"out:a": C.OUTPUT})
        with self.assertRaises(CalibrationError):
            derive_thresholds([self.m("native_repeat", "out:x", 0.0, 0.0),
                               self.m("trusted_compile", "out:x", 0.0, 0.0)],
                              results={"out:a": C.OUTPUT})


class TestControls(unittest.TestCase):
    def test_failed_or_unavailable_compile_control_gives_no_policy(self):
        def breaks_graph(x, w):
            torch._dynamo.graph_break()
            return F.norm_site(x, w)

        scope = F.toy_module_scope()
        scope.native_sites = {**scope.native_sites, "norm": breaks_graph}
        result = calibrate(scope, patch_set=("norm",), samples=samples("ctl", [1]),
                           compile_backend=FAST)
        self.assertIsNone(result.policy)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual([f.code for f in result.findings], ["control_failed"])
        missing = calibrate(F.toy_module_scope(), patch_set=("norm",), samples=samples("ctl", [1]),
                            compile_backend="no_such_backend")
        self.assertIsNone(missing.policy)
        self.assertEqual([f.code for f in missing.findings], ["control_unavailable"])

    def test_calibration_inputs_are_checked(self):
        scope = F.toy_module_scope()
        for kwargs in ({"patch_set": ()}, {"patch_set": ("nope",)}, {"samples": []},
                       {"native_repeats": 1}, {"gamma": 0.1},
                       {"samples": [Sample("a", 1), Sample("b", 1)]}):
            args = {"patch_set": ("norm",), "samples": samples("c", [1]), **kwargs}
            self.assertIsNone(calibrate(scope, compile_backend=FAST, **args).policy, kwargs)


class TestBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = F.toy_module_scope()
        cls.fixed = samples("bnd", [1, 2])
        cls.calibration = calibrate(cls.scope, patch_set=("norm",), samples=cls.fixed,
                                    compile_backend=FAST, boundaries=True)

    def verify(self, fn):
        return gate2.verify_integration(self.scope, candidate("c", {"norm": fn}),
                                        policy=self.calibration.policy, samples=self.fixed)

    def test_boundary_thresholds_and_correct_candidate(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        table = self.calibration.policy.thresholds_for("bnd/seed1")
        names = sorted(n for n in table if n.startswith("boundary:"))
        self.assertEqual(names, ["boundary:norm#0/grad_in:args[0]", "boundary:norm#0/grad_in:args[1]",
                                 "boundary:norm#0/out:out", "boundary:norm#1/grad_in:args[0]",
                                 "boundary:norm#1/grad_in:args[1]", "boundary:norm#1/out:out"])
        report = self.verify(lambda x, w: F.norm_site(x, w))
        self.assertEqual(report.status, PASSED, codes(report))

    def test_stateful_and_misrouted_sites_fail_at_their_boundary(self):
        stateful = self.verify(F.StatefulNorm())
        self.assertEqual(stateful.status, FAILED)
        self.assertTrue(any(n.endswith("~repeat") for n in failing(stateful)), failing(stateful))
        misrouted = self.verify(F.norm_misrouted)
        self.assertIn("boundary:norm#0/grad_in:args[1]", failing(misrouted))
        self.assertNotIn("boundary:norm#0/grad_in:args[0]", failing(misrouted))


class TestDeclaredScopes(unittest.TestCase):
    def test_required_scopes_are_fixed_before_results(self):
        scope = F.toy_module_scope()
        fixed = [Sample("scopes", 1)]
        policy = synthetic_policy(scope, ("norm",), lambda s: {"E2": 1e-3, "Emax": 1e-2},
                                  calibration=(("scopes", 1),))
        model = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                         policy=policy, samples=fixed)
        self.assertEqual(gate2.combine_scopes(("model",), {"model": model}).status, PASSED)
        missing = gate2.combine_scopes(("operator", "model"), {"model": model})
        self.assertEqual(missing.status, INCOMPLETE)
        self.assertIn("required_scope_missing", codes(missing))
        failed_extra = gate2.verify_integration(scope, candidate("bad", {"norm": F.norm_spike}),
                                                policy=policy, samples=fixed)
        kept = gate2.combine_scopes(("model",), {"model": model, "extra": failed_extra})
        self.assertEqual(kept.status, PASSED)
        self.assertIn("extra", kept.evidence["undeclared_scopes"])

    def test_block_is_not_a_scope(self):
        self.assertEqual(gate2.SCOPES, ("operator", "model"))
        scope = F.toy_module_scope()
        policy = synthetic_policy(scope, ("norm",), lambda s: {"E2": 1e-3, "Emax": 1e-2},
                                  calibration=(("scopes", 1),))
        model = gate2.verify_integration(scope, candidate("c", {"norm": F.norm_site}),
                                         policy=policy, samples=[Sample("scopes", 1)])
        block = gate2.combine_scopes(("block",), {"block": model})
        self.assertEqual(block.status, INCOMPLETE)
        self.assertIn("unknown_scope", codes(block))


class TestInductorControl(unittest.TestCase):
    """The default control: torch.compile with inductor, executed for real."""

    def test_operator_and_model_scope_with_inductor(self):
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device=CPU)
        result = calibrate(scope, patch_set=("operator",), samples=scope.samples([1, 2]))
        if result.policy is None:
            reasons = "; ".join(f.message[:200] for f in result.findings)
            self.skipTest(f"inductor control unavailable on this machine: {reasons}")
        self.assertGreater(result.policy.controls["compile_graphs"]["operator"], 0)
        self.assertEqual(result.policy.controls["compile"]["identity"]["backend"], "inductor")
        report = gate2.verify_integration(scope, scope.candidate(F.scale_shift_pair()),
                                          policy=result.policy, samples=scope.samples([11, 12]))
        self.assertEqual(report.status, PASSED, codes(report))
        model = F.toy_module_scope()
        fixed = samples("ind", [1, 2])
        calibrated = calibrate(model, patch_set=("norm",), samples=fixed)
        self.assertEqual(calibrated.status, "calibrated", calibrated.to_dict())
        bad = gate2.verify_integration(model, candidate("spike", {"norm": F.norm_spike}),
                                       policy=calibrated.policy, samples=fixed)
        self.assertEqual(bad.status, FAILED)


if __name__ == "__main__":
    unittest.main()
