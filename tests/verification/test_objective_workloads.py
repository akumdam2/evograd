"""The gates on a workload that is not a language model, on a toy CPU model.

* ``ObjectiveModelScope``: calibrates on the workload's own loss, primary
  output and parameter gradients; passes the bound native pair and rejects a
  1% fault; counts site calls from ``expected_site_calls``;
* ``workload_objective_training``: the recorded objective is the workload loss,
  and a replaced site runs every step;
* ``observed_cases`` with several Level-2 configurations, case-id uniqueness,
  and ``case_for`` resolving shape-identical cases by component;
* ``isolated_rng`` seeds and restores Python's ``random``.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from evograd.verification import gate2, gate3
from evograd.verification.common.execution import Deployment, Sample, candidate
from evograd.verification.common.reports import FAILED, PASSED

ROWS, HIDDEN, COLS, CLASSES = 8, 12, 16, 5
FAST = "aot_eager"


def bound_eager(op_name, scale=1.0):
    from evograd.benchmark import get_task
    from evograd.evaluation.tier3.patch import eager_pair_for, kernel_from_pair

    op = get_task(op_name)
    kernel = kernel_from_pair(op, eager_pair_for(op))
    if scale == 1.0:
        return kernel
    return lambda *args: kernel(*args) * scale


class ToyNet(torch.nn.Module):
    def __init__(self, act):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.up = torch.nn.Parameter(torch.randn(HIDDEN, 2 * COLS, generator=g) * 0.3)
        self.down = torch.nn.Parameter(torch.randn(COLS, CLASSES, generator=g) * 0.3)
        self.act = act

    def forward(self, x):
        a, b = (x @ self.up).chunk(2, dim=-1)
        return self.act(a.contiguous(), b.contiguous()) @ self.down


class ToyWorkload:
    """A classification workload in the tier-3 protocol plus the verification interface."""

    name = "toy_classifier"
    unit_name = "examples"
    primary_output_name = "logits"
    objective_definition = "mean cross entropy over the batch's class logits"

    def __init__(self):
        from evograd.benchmark import get_task
        from evograd.evaluation.tier3.patch import Site, SiteRegistry
        from evograd.opdecl.activity import Workload
        from evograd.opdecl.oracle import resolve_runtime_forward

        self.spec = SimpleNamespace(device="cpu", dtype="float32")
        self.site_registry = SiteRegistry("toy", (Site(
            "act", "swiglu", resolve_runtime_forward(get_task("swiglu")),
            (Workload(dims={"rows": ROWS, "cols": COLS}, dtype="float32"),)),))

    def units_per_step(self):
        return ROWS

    def build_patched(self, kernels):
        return ToyNet(kernels.kernel_for("act")), None

    def build(self, kernels):
        return self.build_patched(kernels)[0]

    def batch_for(self, *, seed):
        g = torch.Generator().manual_seed(int(seed))
        return {"x": torch.randn(ROWS, HIDDEN, generator=g),
                "labels": torch.randint(0, CLASSES, (ROWS,), generator=g)}

    def primary_output(self, model, batch):
        logits = model(batch["x"])
        return torch.nn.functional.cross_entropy(logits, batch["labels"]), logits

    def loss(self, model, batch):
        return self.primary_output(model, batch)[0]

    def expected_site_calls(self):
        return {"act": 1}

    def describe(self):
        return {"name": self.name, "rows": ROWS}


class Batches:
    def __init__(self, workload, offset):
        self.workload, self.offset = workload, offset

    def batch(self, index):
        return self.workload.batch_for(seed=self.offset + index)

    def describe(self):
        return {"source": "toy", "offset": self.offset}


def codes(report):
    return sorted({f.code for f in report.all_findings()})


class ObjectiveScopeTests(unittest.TestCase):
    def setUp(self):
        from evograd.verification.gate2.adapters.model import ObjectiveModelScope

        self.workload = ToyWorkload()
        self.scope = ObjectiveModelScope(self.workload)
        self.samples = self.scope.samples([1, 2])

    def test_calibrates_and_judges_on_loss_primary_and_parameter_gradients(self):
        cal = gate2.calibrate(self.scope, patch_set=("act",), samples=self.samples,
                              compile_backend=FAST, boundaries=True)
        self.assertEqual(cal.status, "calibrated", cal.to_dict())
        names = set(cal.policy.thresholds_for(self.samples[0].sample_id))
        self.assertTrue({"out:logits", "out:loss", "grad:model.up", "grad:model.down"} <= names, names)
        good = gate2.verify_integration(self.scope, candidate("bound", {"act": bound_eager("swiglu")}),
                                        policy=cal.policy, samples=self.samples)
        self.assertEqual(good.status, PASSED, codes(good))
        self.assertEqual(good.cases[0].evidence["candidate"]["site_calls"], {"act": 1})
        bad = gate2.verify_integration(self.scope, candidate("off", {"act": bound_eager("swiglu", 1.01)}),
                                       policy=cal.policy, samples=self.samples)
        self.assertEqual(bad.status, FAILED)
        failed = {r.name for c in bad.cases for r in c.results if r.status == FAILED}
        self.assertIn("out:logits", failed)

    def test_identity_names_the_objective(self):
        identity = self.scope.identity()
        self.assertEqual(identity["outputs"], ["logits", "loss"])
        self.assertEqual(identity["loss"], ToyWorkload.objective_definition)
        self.assertIn("act", identity["site_references"])


class ObjectiveTrainingTests(unittest.TestCase):
    def adapter(self):
        workload = ToyWorkload()
        return gate3.workload_objective_training(workload, train_data=Batches(workload, 100),
                                                 optimizer=gate3.adamw(lr=1e-2))

    def test_objective_is_the_workload_loss(self):
        plan = gate3.TrainingPlan(steps=3, window=1, checkpoints=(), validation=False)
        run = gate3.measure_training(self.adapter(), Deployment.native(), plan, seed=0)
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        from evograd.evaluation.tier3.patch import KernelSet

        workload = ToyWorkload()
        model = workload.build(KernelSet(registry=workload.site_registry))
        first = workload.loss(model, workload.batch_for(seed=100)).detach()
        self.assertEqual(run.train_steps[0]["objective"], float(first))
        self.assertEqual(run.train_steps[0]["weight"], 1.0)

    def test_candidate_runs_its_site_every_step(self):
        plan = gate3.TrainingPlan(steps=3, window=1, checkpoints=(), validation=False)
        run = gate3.measure_training(self.adapter(), candidate("bound", {"act": bound_eager("swiglu")}),
                                     plan, seed=0)
        self.assertTrue(run.completed, [f.to_dict() for f in run.findings])
        self.assertEqual(run.site_calls["total"], {"act": 3})

    def test_validation_plans_are_refused(self):
        plan = gate3.TrainingPlan(steps=2, window=1, checkpoints=(0,), validation=True)
        run = gate3.measure_training(self.adapter(), Deployment.native(), plan, seed=0)
        self.assertFalse(run.completed)


class CaseTableTests(unittest.TestCase):
    @staticmethod
    def _workload(component, dims):
        from evograd.opdecl.activity import Provenance, Workload

        return Workload(dims=dims, dtype="bfloat16",
                        provenance=Provenance(model="toy", component=component, free={}))

    def _cases(self, l2_components):
        from evograd.verification.integrations.base import Model, observed_cases

        l1 = [self._workload("fc1", {"M": 4, "K": 8, "N": 16})]
        l2 = [self._workload(c, {"rows": 4 * (i + 1), "cols": 8}) for i, c in enumerate(l2_components)]
        snapshot = {"level1": {"linear": {"configurations": [{"dims": dict(l1[0].dims),
                                                              "frequency": 3}]}},
                    "tasks": {"toy_mlp": {"configurations": [{"dims": dict(w.dims), "frequency": 2 + i}
                                                             for i, w in enumerate(l2)],
                                          "frequency": 99}}}
        ops = {"linear": SimpleNamespace(benchmark_suites={"toy_observed": tuple(l1)}),
               "toy_mlp": SimpleNamespace(benchmark_suites={"toy_observed": tuple(l2)})}
        model = Model(name="toy", suite="toy_observed", workload_config={}, load_workload=None,
                      train_batches=None, live_capture=None, invocations=None, reference_op=None,
                      contract="toy", source_identity=dict)
        with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: snapshot), \
                mock.patch("evograd.benchmark.get_task", lambda task: ops[task]):
            return observed_cases(model)

    def test_level2_task_with_several_configurations(self):
        cases = {c.case_id: c for c in self._cases(["stage0", "stage1"])}
        self.assertEqual(sorted(cases), ["linear/fc1", "toy_mlp/stage0", "toy_mlp/stage1"])
        self.assertEqual(cases["toy_mlp/stage1"].frequency, 3)
        self.assertEqual(cases["toy_mlp/stage0"].level, 2)

    def test_repeated_case_ids_are_refused(self):
        with self.assertRaisesRegex(ValueError, "case ids repeat"):
            self._cases(["toy_mlp", "toy_mlp"])

    def test_component_resolves_shape_identical_cases(self):
        from evograd.opdecl.activity import Workload
        from evograd.verification.integrations.base import Case, case_for

        dims = {"B": 1, "n_heads": 2, "T": 4, "head_dim": 8}
        cases = [Case("rope/rope", "rope", 1, Workload(dims=dims, dtype="float32"), 1),
                 Case("rope/rope_kv", "rope", 1, Workload(dims=dims, dtype="float32"), 1)]
        args = {"x": torch.zeros(1, 2, 4, 8), "cos": torch.zeros(4, 8), "sin": torch.zeros(4, 8)}
        with self.assertRaisesRegex(KeyError, "2 observed cases"):
            case_for(cases, "rope", args)
        self.assertEqual(case_for(cases, "rope", args, component="rope_kv").case_id, "rope/rope_kv")
        # The module path the harvest recorded decides first.
        tied = [Case("rope/a", "rope", 1, Workload(dims=dims, dtype="float32"), 1, ("m.x",)),
                Case("rope/b", "rope", 1, Workload(dims=dims, dtype="float32"), 1, ("m.y", "m.z"))]
        self.assertEqual(case_for(tied, "rope", args, path="m.z#2").case_id, "rope/b")
        # A role no tied case carries does not eliminate them all.
        with self.assertRaisesRegex(KeyError, "2 observed cases"):
            case_for(cases, "rope", args, component="rope_merged_elsewhere")


class PythonRandomTests(unittest.TestCase):
    def test_isolated_rng_seeds_and_restores_python_random(self):
        from evograd.verification.common.tensors import isolated_rng

        random.seed(123)
        expected_outside = random.random()
        random.seed(123)
        with isolated_rng(7):
            first = random.random()
        self.assertEqual(random.random(), expected_outside)
        with isolated_rng(7):
            self.assertEqual(random.random(), first)


if __name__ == "__main__":
    unittest.main()
