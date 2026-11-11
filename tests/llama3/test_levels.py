"""Llama's levels 3-1: the operator it owns, and what the others consume.

Almost nothing here can be *run* today. Level 3 needs a GPU capture, level 2
needs that capture plus a snapshot, and level 1 needs the snapshot. What can be
checked without any of them is the part that would otherwise be checked only by
running it: that the modules import, that the declaration Llama introduced is
self-consistent, that the composition table describes Llama's decoder rather
than Qwen3's, and -- the one that actually matters -- that a stage with no
artifact **refuses by name** instead of substituting a synthetic stand-in.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from evograd.benchmark.topdown import UnharvestedWorkload, has_snapshot
from evograd.opdecl import Workload
from tests._small_cases import SMALL


class TestQkvRopeDeclaration(unittest.TestCase):
    """``llama3_qkv_rope``: the one operator this workload had to introduce."""

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark import get_task

        cls.op = get_task("llama3_qkv_rope")

    def test_it_has_no_per_head_norm_weights(self):
        """LlamaAttention has no per-head norms: four active arguments, four
        gradients."""
        names = {arg.name for arg in self.op.args}
        self.assertNotIn("q_norm_weight", names)
        self.assertNotIn("k_norm_weight", names)
        self.assertEqual(
            self.op.grad_names(), ("dx", "dq_weight", "dk_weight", "dv_weight")
        )

    def test_it_carries_no_eps(self):
        """There is no normalization inside the boundary for one to belong to."""
        self.assertNotIn("eps", {arg.name for arg in self.op.args})

    def test_it_returns_q_k_and_v_in_that_order(self):
        self.assertEqual(self.op.output_names, ("q", "k", "v"))

    def test_the_benchmark_shape_is_llama_3_2_1bs(self):
        """Derived from the published configuration, not written out."""
        from evograd.opdecl.models import LLAMA_3_2_1B

        self.assertEqual(
            self.op.benchmark[0].dims,
            LLAMA_3_2_1B.qkv_norm_rope_dims(batch=2, seq=2048),
        )

    def test_the_declared_layout_is_the_head_major_view(self):
        """The model reaches q/k/v by view-then-transpose, so what a kernel
        receives and must return is a non-contiguous view."""
        from evograd.ops._common import is_head_major_view

        self.assertTrue(is_head_major_view(self.op.benchmark[0]))

    def test_the_observed_suite_appears_only_once_harvested(self):
        """A suite named ``observed`` must come from an observation. It is wired
        and inert; the harvest is what turns it on, with no edit here."""
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2.qkv_rope import task as decl

        self.assertEqual(decl.HARVESTED, has_snapshot("llama_3_2_1b"))
        self.assertEqual(
            "llama_3_2_1b_observed" in self.op.benchmark_suites, decl.HARVESTED
        )

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_eager_pair_passes_gate1_against_the_declared_reference(self):
        """The production spelling, as the eager pair, against the declared
        forward under a model-reference contract, by Gate 1's rule."""
        from evograd.evaluation.tier3.patch import eager_pair_for
        from tests.verification import _fixtures as F

        report = F.gate1_report(self.op, eager_pair_for(self.op),
                                workloads=SMALL["llama3_qkv_rope"])
        self.assertEqual(report.status, "passed", report.to_dict())

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_pair_runs_forward_and_backward_on_small_cases(self):
        from evograd.evaluation.tier3.patch import eager_pair_for

        pair = eager_pair_for(self.op)
        for workload in SMALL["llama3_qkv_rope"]:
            with self.subTest(dims=workload.dims, dtype=workload.dtype):
                values = self.op.make_inputs(torch, self.op, workload, device="cpu")
                args = tuple(values[a.name] for a in self.op.args)
                outputs, saved = pair.forward_with_saved(*args)
                dims = workload.dims
                # The head-major stride signature, which is part of the
                # contract rather than an implementation detail.
                self.assertEqual(
                    outputs[0].stride(),
                    (dims["T"] * dims["HQ"] * dims["D"], dims["D"],
                     dims["HQ"] * dims["D"], 1),
                )
                grads = pair.backward_from_saved(
                    tuple(values[f"d{n}"] for n in self.op.output_names), saved
                )
                self.assertEqual(len(grads), len(self.op.grad_names()))
                for name, grad in zip(self.op.grad_names(), grads):
                    self.assertIsNotNone(grad, name)


#: What ``mapping()`` reports as the composition, once a harvest exists.
COMPOSES_INTO_EXPECTED = {
    "linear_no_bias": ("llama3_qkv_rope", "llama3_attention", "llama3_swiglu_mlp"),
    "rmsnorm": ("llama3_residual_rmsnorm",),
    "rope": ("llama3_qkv_rope",),
    "swiglu": ("llama3_swiglu_mlp",),
    "causal_gqa_attention": ("llama3_attention",),
    "cross_entropy": (),
}


class TestResidualRmsnormEpsilon(unittest.TestCase):
    """The model case runs the model's epsilon; any other case the generic one."""

    def setUp(self):
        from evograd.benchmark import get_task

        self.op = get_task("llama3_residual_rmsnorm")

    def test_the_model_case_carries_the_published_rms_norm_eps(self):
        import torch

        from evograd.benchmark.topdown.llama3_2_1b.levels.level2.residual_rmsnorm import task as decl
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.spec import LLAMA_3_2_1B
        from evograd.opdecl.inputs import make_case_inputs

        self.assertEqual(decl.MODEL_EPS, LLAMA_3_2_1B["rms_norm_eps"])
        self.assertNotEqual(decl.MODEL_EPS, decl.GRID_EPS)
        for workload in self.op.benchmark:
            self.assertTrue(decl.is_model_case(workload))
            values = make_case_inputs(self.op, workload, device="cpu")
            self.assertEqual(values["eps"], decl.MODEL_EPS)
            del values
        observed = self.op.benchmark_workloads(suite="llama_3_2_1b_observed")
        self.assertTrue(observed)
        for workload in observed:
            self.assertEqual(decl.eps_for(workload), decl.MODEL_EPS)

    def test_a_case_without_model_provenance_keeps_the_generic_epsilon(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2.residual_rmsnorm import task as decl
        from evograd.opdecl.inputs import make_case_inputs

        for workload in (Workload(dims={"rows": 8, "cols": 64}, dtype="float32"),
                         Workload(dims={"rows": 32, "cols": 256}, dtype="bfloat16")):
            self.assertFalse(decl.is_model_case(workload))
            self.assertEqual(make_case_inputs(self.op, workload, device="cpu")["eps"], decl.GRID_EPS)
        # The declared default is the generic one, so a candidate that ignores the
        # argument it is handed is caught by the model case, not hidden by it.
        eps_arg = next(arg for arg in self.op.args if arg.name == "eps")
        self.assertEqual(eps_arg.default, decl.GRID_EPS)

    def test_no_pair_baseline_is_declared(self):
        self.assertFalse(self.op.performance_baselines)


class TestLevel1Mapping(unittest.TestCase):
    def test_rmsnorm_reaches_only_the_residual_fusion(self):
        """One edge, not Qwen3's two: Llama-3's only RMSNorm is the decoder's."""
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
            COMPOSES_INTO,
        )

        self.assertEqual(COMPOSES_INTO["rmsnorm"], ("llama3_residual_rmsnorm",))
        self.assertEqual(COMPOSES_INTO["rope"], ("llama3_qkv_rope",))

    def test_every_named_operator_exists(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
            COMPOSES_INTO,
            OBSERVED_BINDINGS,
            OPERATORS,
        )
        from evograd.ops import PRIMITIVES

        # Every name on the left of the composition is a reusable primitive,
        # and every name on the right is an executable task -- the two sides
        # are different kinds of thing and the registries keep them apart.
        for name in OPERATORS:
            self.assertIn(name, PRIMITIVES)
        for primitive, fused in COMPOSES_INTO.items():
            self.assertIn(primitive, PRIMITIVES)
            for target in fused:
                self.assertIn(target, TASKS)
                self.assertEqual(TASKS[target].level, 2)

        # The three tables that name this model's primitives agree. They are
        # separate statements -- what it runs, what each composes into, and
        # where its observed cases attach -- so a primitive added to one and
        # forgotten in another is caught here.
        self.assertEqual(set(OPERATORS), set(COMPOSES_INTO))
        self.assertEqual(
            set(OPERATORS), {binding.task for binding in OBSERVED_BINDINGS}
        )

    def test_the_mapping_refuses_by_name_until_the_harvest_is_run(self):
        """No snapshot means no report -- not an empty one.

        An unrun harvest and an empty harvest are different answers, and the
        one thing a workload package must never do is fabricate the second.
        """
        from evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot import SnapshotError
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import mapping

        if has_snapshot("llama_3_2_1b"):
            report = mapping()
            self.assertIn("snapshot_hash", report)
            self.assertEqual(report["composes_into"], COMPOSES_INTO_EXPECTED)
            return
        with self.assertRaises(SnapshotError) as raised:
            mapping()
        self.assertIn("snapshot.json", str(raised.exception))


class TestLevel2Package(unittest.TestCase):
    def test_the_projection_boundary_is_llamas_own_declaration(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest

        self.assertEqual(manifest.SITE_TASKS["qkv_rope"], "llama3_qkv_rope")
        self.assertEqual(manifest.task_key("qkv_rope"), "llama3_qkv_rope")

    def test_every_calibrated_operator_exists(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest

        for name in manifest.SITE_TASKS.values():
            self.assertIn(name, TASKS)
            self.assertEqual(TASKS[name].level, 2)

    def test_each_module_names_the_operator_it_calibrates(self):
        import importlib

        from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest

        for module_name, op_name in manifest.SITE_TASKS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(
                    f"evograd.benchmark.topdown.llama3_2_1b.levels.level2.{module_name}"
                )
                # The site package serves the declaration itself, so this
                # compares what the registry will register rather than a
                # second string that could drift away from it.
                self.assertEqual(module.op.name, op_name)
                capture = importlib.import_module(
                    f"evograd.benchmark.topdown.llama3_2_1b.levels.level2."
                    f"{module_name}.capture"
                )
                self.assertEqual(capture.TASK_NAME, op_name)


class TestLevel3Package(unittest.TestCase):
    def test_the_representative_layer_is_this_architectures(self):
        """Half depth, as Qwen3 picks half of 28. Naming one is what makes two
        captures comparable."""
        from evograd.benchmark.topdown.llama3_2_1b.layer import capture
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.spec import LLAMA_3_2_1B

        self.assertEqual(
            capture.CANONICAL_LAYER_INDEX, LLAMA_3_2_1B["num_hidden_layers"] // 2
        )

    def test_the_schemas_are_llamas_own(self):
        """A capture of another model relabelled as a Llama one would otherwise load."""
        from evograd.benchmark.topdown.llama3_2_1b.layer import artifact

        self.assertIn("llama3", artifact.SCHEMA_VERSION)


class TestUnharvestedStagesRefuseByName(unittest.TestCase):
    """The load-bearing property of everything above.

    A stage whose artifact does not exist must say which command produces it.
    Falling back to synthetic tensors would produce a number that looks like a
    measurement of Llama-3 and is not.
    """

    @unittest.skipIf(has_snapshot("llama_3_2_1b"), "a snapshot now exists")
    def test_loading_the_snapshot_names_the_harvest_command(self):
        from evograd.benchmark.topdown import load_snapshot

        with self.assertRaises(UnharvestedWorkload) as caught:
            load_snapshot("llama_3_2_1b")
        message = str(caught.exception)
        self.assertIn("harvest.harvest", message)
        self.assertIn("harvest.snapshot", message)


if __name__ == "__main__":
    unittest.main()
