"""Gate 2 on CPU: the operator scope against the float32 truth, the model scope
against native eager.

``SCALE`` is a test-only "compiler": dynamo's graph run eagerly with its floating
outputs scaled by ``1 + 3e-4``, a trusted control whose error is known and
nonzero; ``SAME`` runs dynamo's graph unchanged, the "compile reproduces native"
case. ``TestInductorControl`` runs the
default inductor backend and skips, with the reason, when this machine cannot
build its C++ kernels (set ``CXX`` to a working compiler).
"""

from __future__ import annotations

import gc
import json
import math
import types
import unittest
import warnings
import weakref

import torch

from evograd.opdecl import Workload
from evograd.verification import gate2
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.common.reports import FAILED, INCOMPLETE, PASSED
from evograd.verification.common.tensors import tensors_in
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.gate2.model import FLOORS as MODEL_FLOORS, e2, verify_model
from evograd.verification.gate2.operator import FLOORS, errors, verify_operator
from tests.verification import _fixtures as F

warnings.filterwarnings("ignore")
CPU = "cpu"
FAST = "aot_eager"
SCALE = "evograd_test_scale_3e-4"
#: The same at a scale bfloat16 keeps (3e-4 rounds away in a bfloat16 model).
SCALE_BF16 = "evograd_test_scale_2^-6"
SAME = "evograd_test_graph_as_is"


def _scaling(factor):
    def backend(gm, example_inputs):
        """Test-only: the graph run eagerly, its floating outputs scaled by ``factor``."""
        def run(*args):
            outs = gm(*args)
            if isinstance(outs, (tuple, list)):
                return type(outs)(o * factor if torch.is_tensor(o) and o.is_floating_point() else o
                                  for o in outs)
            return outs * factor
        return run
    return backend


def _graph_as_is(gm, example_inputs):
    """Test-only: dynamo's graph run eagerly, unchanged (native bit for bit)."""
    return gm.forward


for _backend, _name in ((_scaling(1 + 3e-4), SCALE), (_scaling(1 + 2 ** -6), SCALE_BF16),
                        (_graph_as_is, SAME)):
    try:
        torch._dynamo.register_backend(_backend, name=_name)
    except Exception:  # already registered in this process
        pass


def codes(report):
    return sorted({f.code for f in report.all_findings()})


def failing(report):
    return sorted({r.name for c in report.cases for r in c.results if r.status == FAILED})


def samples(prefix, seeds):
    return [Sample(f"{prefix}/seed{s}", s, {"recipe": "fixture"}) for s in seeds]


class TestOperatorScope(unittest.TestCase):
    """Operator Gate 2: every sample against the float32 truth, within twice native's
    and compile's largest error on the samples, or the dtype's floor."""

    @classmethod
    def setUpClass(cls):
        cls.scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device=CPU)
        cls.mul_scope = OperatorScope(F.MUL, F.MUL.benchmark[0], device=CPU)
        cls.bf16_scope = OperatorScope(F.SCALE_SHIFT, Workload({"R": 64, "C": 32}, "bfloat16"),
                                       device=CPU)

    def verify(self, scope, artifact, seeds=(1, 2)):
        return verify_operator(scope, scope.candidate(artifact), samples=scope.samples(list(seeds)),
                               compile_backend=FAST)

    def test_correct_legacy_pair_passes(self):
        report = self.verify(self.scope, F.scale_shift_pair())
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(sorted(report.subject["thresholds"]), ["grad:b", "grad:w", "grad_in:x", "out:y"])
        self.assertEqual(report.subject["deployment"]["identity"]["adapter_kind"], "legacy_bind_pair")
        self.assertEqual(report.subject["reference"]["kind"], "float32_truth")
        self.assertEqual((report.subject["rule"]["factor"], report.subject["rule"]["floors"]),
                         (2.0, FLOORS))
        self.assertEqual(sorted(report.cases[0].evidence["trusted"]), ["compile", "native"])
        json.dumps(report.to_dict(), allow_nan=False)

    def test_a_float32_call_is_its_own_truth_and_the_floor_decides(self):
        # Nothing to widen: native is the truth (error 0), and aot_eager reproduces it, so
        # every threshold is the float32 floor 1e-5.
        report = self.verify(self.scope, F.scale_shift_pair())
        for name, entry in report.subject["thresholds"].items():
            self.assertEqual((entry["dtype"], entry["E2"]["threshold"], entry["E2"]["from"]),
                             ("float32", 1e-5, "floor"), name)
        self.assertEqual(self.verify(self.scope, F.scale_shift_pair(dw_scale=1 + 2e-6)).status, PASSED)
        small = self.verify(self.scope, F.scale_shift_pair(dw_scale=1 + 5e-5))
        self.assertEqual((small.status, failing(small)), (FAILED, ["grad:w"]))

    def test_wrong_backward_fails_by_name(self):
        report = self.verify(self.scope, F.scale_shift_pair(dw_scale=1.01))
        self.assertEqual(report.status, FAILED)
        self.assertEqual(failing(report), ["grad:w"])
        self.assertIn("threshold_exceeded", codes(report))

    def test_a_bfloat16_call_is_held_to_twice_the_trusted_error_against_the_truth(self):
        report = self.verify(self.bf16_scope, F.scale_shift_pair())
        self.assertEqual(report.status, PASSED, codes(report))
        thresholds = report.subject["thresholds"]
        for name, entry in thresholds.items():
            self.assertEqual(entry["dtype"], "bfloat16", name)
            for metric in ("E2", "M"):
                own = entry[metric]
                self.assertEqual(own["threshold"], max(2 * own["trusted"], FLOORS["bfloat16"]))
                self.assertGreater(own["trusted"], 0.0)      # native rounds where the truth does not
        scaled = self.verify(self.bf16_scope, F.scale_shift_pair(dw_scale=1.05))
        self.assertEqual(failing(scaled), ["grad:w"])

    def test_correct_pair_but_wrong_autograd_routing_is_rejected(self):
        artifact = F.mul_artifact(swap=True)
        direct = F.gate1_report(F.MUL, artifact)
        self.assertEqual(direct.status, PASSED)                    # the pair itself is right
        good = self.verify(self.mul_scope, F.mul_artifact())
        self.assertEqual(good.status, PASSED, codes(good))
        self.assertEqual(good.subject["deployment"]["identity"]["adapter_kind"],
                         "direct_deployment_entry")
        bad = self.verify(self.mul_scope, artifact)
        self.assertEqual(bad.status, FAILED)
        self.assertEqual(failing(bad), ["grad:b", "grad_in:a"])

    def test_without_a_compile_control_native_alone_is_trusted(self):
        from unittest import mock

        with mock.patch("evograd.verification.gate2.operator.compile_control",
                        side_effect=RuntimeError("no compiler")):
            report = self.verify(self.bf16_scope, F.scale_shift_pair())
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertIn("no compiler", report.evidence["compile"]["unavailable"])
        self.assertEqual(sorted(report.cases[0].evidence["trusted"]), ["native"])
        self.assertTrue(all(e[m]["from"] in ("native", "floor")
                            for e in report.subject["thresholds"].values() for m in ("E2", "M")))

    def test_errors_against_the_truth(self):
        t = torch.tensor([1.0, -0.5, 0.0, 0.25])
        e = errors(t * 1.01, t)
        self.assertAlmostEqual(e["E2"], 0.01)
        self.assertAlmostEqual(e["M"], 0.01)
        spike = t.clone()
        spike[3] += 0.1                                  # one element, a tenth of the largest
        self.assertAlmostEqual(errors(spike, t)["M"], 0.1)
        zero = torch.zeros(3)
        self.assertEqual(errors(zero.clone(), zero), {"E2": 0.0, "M": 0.0})
        self.assertEqual(errors(torch.tensor([0.0, 1e-30, 0.0]), zero), {"E2": math.inf, "M": math.inf})


class TestModelScope(unittest.TestCase):
    """A small network at model scope: native, compile and the candidate on the same samples."""

    def setUp(self):
        self.scope = F.toy_module_scope()
        self.fixed = samples("net/fixed", [31, 32])

    def verify(self, fn, scope=None, backend=SCALE, gamma=4.0):
        return verify_model(scope or self.scope, candidate("cand", {"norm": fn}),
                            samples=self.fixed, gamma=gamma, compile_backend=backend)

    def test_correct_candidate_and_its_thresholds(self):
        report = self.verify(lambda x, w: F.norm_site(x, w))
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(report.subject["reference"]["kind"], "native_eager")
        self.assertEqual(report.cases[0].evidence["candidate"]["site_calls"], {"norm": 2})
        for name, entry in report.subject["thresholds"].items():
            floor = MODEL_FLOORS["gradient" if name.startswith("grad") else "output"]
            self.assertEqual(entry["threshold"], 4.0 * max(entry["compile"], floor), name)
        self.assertGreater(max(e["compile"] for e in report.subject["thresholds"].values()), 1e-4)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_faults_are_rejected_each_by_its_own_finding(self):
        cases = {
            "spike": (F.norm_spike, "threshold_exceeded"),
            "nan": (F.norm_nan, "non_finite_candidate"),
            "detached": (F.norm_detached, "missing_result"),
            "misrouted": (F.norm_misrouted, "threshold_exceeded"),
            "scaled": (F.norm_scaled(1.01), "threshold_exceeded"),
        }
        for label, (fn, code) in cases.items():
            report = self.verify(fn)
            self.assertEqual(report.status, FAILED, label)
            self.assertIn(code, codes(report), label)
        self.assertEqual(failing(self.verify(F.norm_misrouted)), ["grad:norm_w.0", "grad:norm_w.1"])

    def test_insufficient_coverage(self):
        broken = F.toy_module_scope(factory=lambda sites: F.ToyModel(sites, skip_last_norm=True))
        report = self.verify(lambda x, w: F.norm_site(x, w), scope=broken)
        self.assertEqual(report.status, FAILED)
        self.assertIn("coverage_mismatch", codes(report))

    def test_a_compile_that_reproduces_native_leaves_coverage_only(self):
        # A control that reproduces native: no trusted alternative, so the numbers are
        # Gate 1's and operator Gate 2's; coverage is still checked.
        report = self.verify(F.norm_spike, backend=SAME)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertIn("compile_bitwise", report.evidence["numeric"])
        self.assertTrue(all(r.limits.get("not_judged") == "compile_bitwise"
                            for c in report.cases for r in c.results if r.metrics))
        broken = F.toy_module_scope(factory=lambda sites: F.ToyModel(sites, skip_last_norm=True))
        self.assertIn("coverage_mismatch", codes(self.verify(F.norm_spike, scope=broken, backend=SAME)))

    def test_without_a_trusted_control_the_report_is_incomplete(self):
        report = self.verify(lambda x, w: F.norm_site(x, w), backend="no_such_backend")
        self.assertEqual(report.status, INCOMPLETE)
        self.assertIn("no_trusted_control", codes(report))

    def test_the_callers_rng_is_untouched(self):
        torch.manual_seed(5)
        rng = torch.get_rng_state()
        self.assertEqual(self.verify(lambda x, w: F.norm_site(x, w)).status, PASSED)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_e2_against_native(self):
        r = torch.tensor([1.0, -2.0, 2.0])
        self.assertAlmostEqual(e2(r * 1.01, r), 0.01)
        self.assertEqual(e2(r.clone(), r), 0.0)
        self.assertEqual(e2(torch.zeros(3), torch.zeros(3)), 0.0)
        self.assertEqual(e2(torch.tensor([0.0, 1e-30, 0.0]), torch.zeros(3)), math.inf)


class TestTinyLM(unittest.TestCase):
    def setUp(self):
        self.scope = F.tiny_lm_scope()
        self.fixed = samples("lm/fixed", [41, 42])

    def test_tied_parameters_are_one_accumulated_result(self):
        native = gate2.execute(self.scope, Deployment.native(), Sample("lm/val/seed41", 41))
        results = {s.name: s for s in gate2.integration_contract(native).specs}
        self.assertEqual(results["grad:embed.weight"].aliases, ("head.weight",))
        self.assertNotIn("grad:head.weight", results)
        model = F.TinyLM(F.NATIVE_SITES)
        ids, labels = F.lm_batch(41)
        F.next_token_mean_nll(model(ids)["logits"], labels).backward()
        self.assertTrue(torch.equal(native.results["grad:embed.weight"], model.embed.weight.grad))

    def test_correct_and_wrong_candidates(self):
        good = verify_model(self.scope, candidate("same", {"norm": lambda x, w: F.norm_site(x, w)}),
                            samples=self.fixed, gamma=4.0, compile_backend=SCALE)
        self.assertEqual(good.status, PASSED, codes(good))
        self.assertEqual(good.cases[0].evidence["candidate"]["site_calls"], {"norm": 3})
        bad = verify_model(self.scope, candidate("scaled", {"norm": F.norm_scaled(1.01)}),
                           samples=self.fixed, gamma=4.0, compile_backend=SCALE)
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


def _alive(refs):
    """The objects behind ``refs`` that are still alive, described."""
    out = []
    for ref in refs:
        obj = ref()
        if obj is not None:
            out.append(f"{type(obj).__name__}{tuple(obj.shape) if torch.is_tensor(obj) else ''}")
    return out


class _WatchedOperator(OperatorScope):
    """The operator scope, watching by weak reference every module it builds, the
    module's inputs and its output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.watched = []

    def build(self, deployment, sample, hooks):
        module = super().build(deployment, sample, hooks)
        watched = self.watched
        watched.append(weakref.ref(module))

        def inputs(_module, args):
            watched.extend(weakref.ref(a) for a in args)

        def output(_module, _args, out):
            watched.append(weakref.ref(out))

        module.register_forward_pre_hook(inputs)
        module.register_forward_hook(output)
        return module


def _watching(fn, watched):
    """``fn`` as a site, watching its arguments, the tensors they view (a module's own
    parameters aside) and its output."""
    def site(*args):
        for t in args:
            watched.append(weakref.ref(t))
            if t._base is not None and not isinstance(t._base, torch.nn.Parameter):
                watched.append(weakref.ref(t._base))
        out = fn(*args)
        watched.append(weakref.ref(out))
        return out
    return site


class _CyclicModels:
    """Toy models that hold themselves in a reference cycle, as a tier-3 workload's
    patched layers do (``forward`` bound to the instance); the factory keeps the last
    one it built, as a workload keeps its last build. ``stale`` describes every
    earlier model still alive when a model starts its forward."""

    __verification_identity__ = {"factory": "tests: cyclic toy models, the last one kept"}

    def __init__(self):
        self.watched, self.stale, self.last = [], [], None

    def __call__(self, sites):
        model = F.ToyModel(sites)
        model.forward = types.MethodType(F.ToyModel.forward, model)
        watched, stale = self.watched, self.stale

        def check(module, _args):
            stale.extend(_alive([r for r in watched if r() is not module]))

        model.register_forward_pre_hook(check)
        watched.append(weakref.ref(model))
        self.last = model
        return model


class TestReleasedRuns(unittest.TestCase):
    """What a run allocates is freed once nothing refers to it, with the cyclic
    collector's automatic runs disabled: they are triggered by counts of Python
    objects, not by memory, and an operator calibration on MegaFold payloads ran a
    95 GiB GPU out of memory on its fourth sample waiting for them (2026-11-11).
    Operator builds hold no cycles at all, so operator executions skip the collector."""

    def setUp(self):
        gc.collect()
        gc.disable()
        self.addCleanup(gc.enable)

    def test_a_dropped_execution_leaves_nothing_alive(self):
        scope = _WatchedOperator(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device=CPU)
        sample = scope.samples([1])[0]
        control, _graphs = gate2.compile_control(scope, ("operator",), backend=FAST)
        runs = ((Deployment.native(), False),
                (candidate("c", {"operator": _watching(F.scale_shift_runtime, scope.watched)}),
                 True),
                (control, True))
        for deployment, _record in runs:
            ex = gate2.execute(scope, deployment, sample)
            self.assertIsNone(ex.error, ex.error)
            scope.watched.extend(weakref.ref(t) for _, t in tensors_in([ex.results]))
            del ex
            self.assertEqual(_alive(scope.watched), [], deployment.name)

    def test_operator_judging_leaves_nothing_alive(self):
        scope = _WatchedOperator(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device=CPU)
        fixed = scope.samples([1, 2])
        report = verify_operator(
            scope, candidate("c", {"operator": _watching(F.scale_shift_runtime, scope.watched)}),
            samples=fixed, compile_backend=FAST)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(_alive(scope.watched), [])

    def test_model_scope_frees_each_model_before_the_next_one_runs(self):
        models = _CyclicModels()
        scope = F.toy_module_scope(factory=models)
        replayed = []
        report = verify_model(scope, candidate("c", {"norm": _watching(F.norm_site, replayed)}),
                              samples=samples("mem", [1, 2]), gamma=4.0, compile_backend=SCALE)
        self.assertEqual(report.status, PASSED, codes(report))
        self.assertEqual(models.stale, [])
        self.assertEqual(_alive(models.watched), ["ToyModel"])     # the factory's last build
        # Only the last build's own parameters, which the candidate's site was called with.
        self.assertEqual([t for t in _alive(replayed) if not t.startswith("Parameter")], [])


class TestInductorControl(unittest.TestCase):
    """The default control: torch.compile with inductor, executed for real."""

    def test_operator_and_model_scope_with_inductor(self):
        scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device=CPU)
        report = verify_operator(scope, scope.candidate(F.scale_shift_pair()),
                                 samples=scope.samples([1, 2]))
        compiled = [c.evidence.get("compile") for c in report.cases if c.evidence.get("compile")]
        if "unavailable" in report.evidence["compile"] or compiled:
            self.skipTest(f"inductor control unavailable on this machine: "
                          f"{report.evidence['compile'].get('unavailable') or compiled[0]['failed_at']}")
        self.assertGreater(report.evidence["compile"]["graphs"]["operator"], 0)
        self.assertEqual(report.evidence["compile"]["deployment"]["identity"]["backend"], "inductor")
        self.assertEqual(report.status, PASSED, codes(report))
        model = F.toy_module_scope()
        bad = verify_model(model, candidate("spike", {"norm": F.norm_spike}),
                           samples=samples("ind", [1, 2]), gamma=4.0)
        self.assertEqual(bad.status, FAILED, codes(bad))


if __name__ == "__main__":
    unittest.main()
