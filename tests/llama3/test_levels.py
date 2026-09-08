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


class TestQkvRopeDeclaration(unittest.TestCase):
    """``llama3_qkv_rope``: the one operator this workload had to introduce."""

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark import get_task

        cls.op = get_task("llama3_qkv_rope")

    def test_it_has_no_per_head_norm_weights(self):
        """The whole reason it is not ``qwen3_qkv_norm_rope``. Four active
        arguments, not six; four gradients, not six."""
        from evograd.benchmark import get_task

        names = {arg.name for arg in self.op.args}
        self.assertNotIn("q_norm_weight", names)
        self.assertNotIn("k_norm_weight", names)
        self.assertEqual(
            self.op.grad_names(), ("dx", "dq_weight", "dk_weight", "dv_weight")
        )
        qwen = get_task("qwen3_qkv_norm_rope")
        self.assertEqual(len(qwen.grad_names()) - len(self.op.grad_names()), 2)

    def test_it_carries_no_eps(self):
        """There is no normalization inside the boundary for one to belong to."""
        self.assertNotIn("eps", {arg.name for arg in self.op.args})

    def test_it_returns_q_k_and_v_in_that_order(self):
        self.assertEqual(self.op.output_names, ("q", "k", "v"))

    def test_the_benchmark_shape_is_llama_3_8bs(self):
        """Derived from the published configuration, not written out."""
        from evograd.opdecl.models import LLAMA_3_8B

        self.assertEqual(
            self.op.benchmark[0].dims,
            LLAMA_3_8B.qkv_norm_rope_dims(batch=2, seq=2048),
        )

    def test_the_declared_layout_is_the_head_major_view(self):
        """The model reaches q/k/v by view-then-transpose, so what a kernel
        receives and must return is a non-contiguous view."""
        from evograd.ops._common import is_head_major_view

        self.assertTrue(is_head_major_view(self.op.benchmark[0]))

    def test_the_observed_suite_appears_only_once_harvested(self):
        """A suite named ``observed`` must come from an observation. It is wired
        and inert; the harvest is what turns it on, with no edit here."""
        from evograd.benchmark.topdown.llama3_8b.levels.level2.qkv_rope import task as decl

        self.assertEqual(decl.HARVESTED, has_snapshot("llama_3_8b"))
        self.assertEqual(
            "llama_3_8b_observed" in self.op.benchmark_suites, decl.HARVESTED
        )

    def test_both_tolerance_questions_are_answered_by_measurement(self):
        """The grid and the observed shape are two questions, and both are shut.

        The grid is declared in this file and both spellings are in
        ``forward_ref``, so comparing them needs no GPU -- the multipliers are
        that measurement. The observed shape reduces over 4096 tokens where the
        grid's longest is 64, and what that costs in bfloat16 is what the hook
        encodes; its sufficiency was measured on a GH200 at the model's widths,
        which is what ``CALIBRATED`` now records."""
        from evograd.benchmark.topdown.llama3_8b.levels.level2.qkv_rope import task as decl

        self.assertTrue(decl.GRID_CALIBRATED)
        self.assertEqual(
            set(self.op.tolerance_multipliers), {"dx", "dq_weight", "dk_weight"}
        )
        self.assertTrue(decl.CALIBRATED)
        self.assertIsNotNone(self.op.tolerance_hook)

    def test_the_gate_admits_the_measured_requirement_at_the_observed_shape(self):
        """The numbers `CALIBRATED` cites, pinned so a retune cannot drop below
        them silently. `required_t` is the smallest base tolerance that accepts
        the declaration's own reference at the model's width; the declared atol
        has to stay above it with margin, on every result that has one."""
        observed = self.op.benchmark[0]
        required_t = {
            "q": 1.039e-02,
            "k": 1.004e-02,
            "dx": 2.415e-02,
            "dq_weight": 1.795e-01,
            "dk_weight": 1.437e-01,
        }
        # `dx` sits almost exactly on the 1.5x policy (1.51x): the element-count
        # term supplies 1.83x where the measured growth from the grid was 2.16x,
        # and the declared 1.2 multiplier covers the rest. It has the least slack
        # of anything here, so a change to `gain` or to that multiplier will
        # surface as this assertion before it surfaces as a rejected kernel.
        for name, required in required_t.items():
            with self.subTest(result=name):
                ma = self.op.tolerance_multipliers.get(name, (1.0, 1.0))[0]
                atol = self.op.tolerance_for(observed, name)[0]
                self.assertGreater(atol / ma, required * 1.5, name)
        # `v` and `dv_weight` measured exactly 0.0: the value path has no
        # rotation, so the two spellings are the same computation there. They
        # keep the hook's term anyway -- that is a fact about the oracle pair,
        # not about a candidate accumulating over 4096 tokens in bfloat16.
        for name in ("v", "dv_weight"):
            self.assertNotIn(name, self.op.tolerance_multipliers)
            self.assertGreater(self.op.tolerance_for(observed, name)[0], 2e-2)

    def test_the_hook_is_exactly_the_identity_on_the_correctness_grid(self):
        """The property that makes it safe to add to a calibrated declaration.

        The anchor is the grid's largest case on every term, so nothing that
        passes today becomes easier to pass. Checked at both dtypes and on every
        result, because a hook that widened one float32 case would be loosening
        a gate that measured exactly 0.0 disagreement.
        """
        for workload in self.op.correctness:
            base = self.op.tolerances[workload.dtype]
            for name in (*self.op.output_names, *self.op.grad_names()):
                with self.subTest(dims=workload.dims, result=name):
                    ma, mr = self.op.tolerance_multipliers.get(name, (1.0, 1.0))
                    self.assertEqual(
                        self.op.tolerance_for(workload, name),
                        (base[0] * ma, base[1] * mr),
                    )

    def test_the_observed_shape_gets_the_reduction_term_the_grid_cannot_see(self):
        """And the gate has to separate the results that reduce from those that
        do not, or a constant is doing the work of a law.

        The three projection gradients accumulate one term per token: 4096 at
        the model's shape against the anchor's 64, an 8x random walk before the
        element-count term. ``q``, ``k``, ``v`` and ``dx`` keep the token axis
        and pick up the element count alone.
        """
        observed = self.op.benchmark[0]
        base_atol = self.op.tolerances["bfloat16"][0]
        widening = {
            name: self.op.tolerance_for(observed, name)[0]
            / (base_atol * self.op.tolerance_multipliers.get(name, (1.0, 1.0))[0])
            for name in (*self.op.output_names, *self.op.grad_names())
        }
        for name in ("q", "k", "v", "dx"):
            self.assertLess(widening[name], 2.5, name)
        for name in ("dq_weight", "dk_weight", "dv_weight"):
            self.assertGreater(widening[name], 15.0, name)
        # rtol is untouched everywhere: relative error is what stays constant,
        # and it is what catches a systematically scaled result.
        for name in (*self.op.output_names, *self.op.grad_names()):
            self.assertEqual(self.op.tolerance_for(observed, name)[1], 2e-2)

    def test_only_the_reductions_carry_multipliers(self):
        """The base is set by the forward outputs, so a candidate's primary
        numbers are gated by the base alone. A multiplier on ``q`` would hide
        the very result it is judged on."""
        for name in self.op.output_names:
            self.assertNotIn(name, self.op.tolerance_multipliers)

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_declared_pair_passes_its_own_correctness_gate(self):
        """The tolerance has to admit the declaration's own reference. This is
        what caught the inherited tolerances: at multiplier 1.0 the two weight
        gradients failed, which is the placeholder's stated failure mode --
        visible, and a correct kernel rejected rather than a wrong one taken."""
        from evograd.evaluation.tier3.patch import eager_pair_for
        from evograd.opdecl.oracle import resolve_forward

        pair = eager_pair_for(self.op)
        oracle = resolve_forward(self.op)
        for workload in self.op.correctness:
            with self.subTest(dims=workload.dims, dtype=workload.dtype):
                values = self.op.make_inputs(torch, self.op, workload, device="cpu")
                args = tuple(values[a.name] for a in self.op.args)
                outputs, saved = pair.forward_with_saved(*args)
                grads = pair.backward_from_saved(
                    tuple(values[f"d{n}"] for n in self.op.output_names), saved
                )
                reference = oracle(*args)
                for name, got, want in zip(self.op.output_names, outputs, reference):
                    atol, rtol = self.op.tolerance_for(workload, name)
                    self.assertTrue(
                        torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol),
                        f"{name} at {workload.dims}",
                    )
                self.assertEqual(len(grads), len(self.op.grad_names()))

    def test_a_2_to_1_grouping_case_is_in_the_correctness_grid(self):
        """Llama-3-8B is 4:1 and has ``QO == H``. A grid of only those two
        coincidences would pass a kernel that folded the query fan-out into the
        hidden size, or hard-coded the ratio."""
        ratios = {w.dims["HQ"] // w.dims["HK"] for w in self.op.correctness}
        self.assertIn(2, ratios)
        self.assertTrue(
            any(w.dims["QO"] != w.dims["H"] for w in self.op.correctness)
        )

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_two_spellings_agree(self):
        """``forward`` is the float32 oracle; ``runtime_forward`` is the exact
        ``LlamaAttention`` spelling, and is what the eager baseline is timed
        through. Their gap is the floor no correct kernel can beat."""
        from evograd.opdecl.baselines import verify_runtime_forward

        verify_runtime_forward(self.op, device="cpu")

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_pair_runs_forward_and_backward_on_every_correctness_case(self):
        from evograd.evaluation.tier3.patch import eager_pair_for

        pair = eager_pair_for(self.op)
        for workload in self.op.correctness:
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


class TestLevel1Mapping(unittest.TestCase):
    def test_rmsnorm_reaches_only_the_residual_fusion(self):
        """One edge, not Qwen3's two: Llama-3's only RMSNorm is the decoder's."""
        from evograd.benchmark.topdown.llama3_8b.levels.level1.manifest import (
            COMPOSES_INTO,
        )

        self.assertEqual(COMPOSES_INTO["rmsnorm"], ("llama3_residual_rmsnorm",))
        self.assertEqual(COMPOSES_INTO["rope"], ("llama3_qkv_rope",))

    def test_every_named_operator_exists(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown.llama3_8b.levels.level1.manifest import (
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

    def test_the_level_one_evaluation_side_imports(self):
        """Every module the Level-1 command is built from, actually imported.

        ``compileall`` cannot see this: each of these names is resolved at
        import time from a *different* package, and the split that moved the
        verdict functions to :mod:`evograd.evaluation` is exactly the kind of
        move that leaves one of them behind. Importing them is the only check
        that the two halves still fit together.
        """
        import importlib

        for module in ("verify", "calibrate", "cli"):
            with self.subTest(module=module):
                importlib.import_module(
                    f"evograd.evaluation.workloads.llama3_8b.level1.{module}"
                )

    def test_the_mapping_refuses_by_name_until_the_harvest_is_run(self):
        """No snapshot means no report -- not an empty one.

        An unrun harvest and an empty harvest are different answers, and the
        one thing a workload package must never do is fabricate the second.
        """
        from evograd.benchmark.topdown.llama3_8b.harvest.snapshot import SnapshotError
        from evograd.benchmark.topdown.llama3_8b.levels.level1.manifest import mapping

        if has_snapshot("llama_3_8b"):
            report = mapping()
            self.assertIn("snapshot_hash", report)
            self.assertEqual(report["composes_into"], COMPOSES_INTO_EXPECTED)
            return
        with self.assertRaises(SnapshotError) as raised:
            mapping()
        self.assertIn("snapshot.json", str(raised.exception))


class TestLevel2Package(unittest.TestCase):
    def test_the_projection_boundary_is_llamas_own_declaration(self):
        from evograd.benchmark.topdown.llama3_8b.levels.level2 import manifest

        self.assertEqual(manifest.SITE_TASKS["qkv_rope"], "llama3_qkv_rope")
        self.assertEqual(manifest.task_key("qkv_rope"), "llama3_qkv_rope")

    def test_every_calibrated_operator_exists(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown.llama3_8b.levels.level2 import manifest

        for name in manifest.SITE_TASKS.values():
            self.assertIn(name, TASKS)
            self.assertEqual(TASKS[name].level, 2)

    def test_each_module_names_the_operator_it_calibrates(self):
        import importlib

        from evograd.benchmark.topdown.llama3_8b.levels.level2 import manifest

        for module_name, op_name in manifest.SITE_TASKS.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(
                    f"evograd.benchmark.topdown.llama3_8b.levels.level2.{module_name}"
                )
                # The site package serves the declaration itself, so this
                # compares what the registry will register rather than a
                # second string that could drift away from it.
                self.assertEqual(module.op.name, op_name)
                capture = importlib.import_module(
                    f"evograd.benchmark.topdown.llama3_8b.levels.level2."
                    f"{module_name}.capture"
                )
                self.assertEqual(capture.TASK_NAME, op_name)


class TestLevel3Package(unittest.TestCase):
    def test_the_representative_layer_is_this_architectures(self):
        """Half depth, as Qwen3 picks half of 28. Naming one is what makes two
        captures comparable."""
        from evograd.benchmark.topdown.llama3_8b.levels.level3 import capture
        from evograd.benchmark.topdown.llama3_8b.levels.level4.spec import LLAMA_3_8B

        self.assertEqual(
            capture.CANONICAL_LAYER_INDEX, LLAMA_3_8B["num_hidden_layers"] // 2
        )

    def test_the_schemas_are_llamas_own(self):
        """A Qwen3 capture relabelled as a Llama one would otherwise load."""
        from evograd.benchmark.topdown.llama3_8b.levels.level3 import artifact
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3 import (
            artifact as qwen_artifact,
        )
        from evograd.evaluation.workloads.llama3_8b.level3 import replay

        self.assertIn("llama3", artifact.SCHEMA_VERSION)
        self.assertIn("llama3", replay.REPORT_SCHEMA)
        self.assertNotEqual(artifact.SCHEMA_VERSION, qwen_artifact.SCHEMA_VERSION)

    def test_replay_builds_one_decoder_layer_and_nothing_above_it(self):
        from evograd.evaluation.workloads.llama3_8b.level3 import replay

        live = replay.live_model_instances()
        self.assertIn("LlamaForCausalLM", live)
        self.assertIn("LlamaDecoderLayer", live)


class TestUnharvestedStagesRefuseByName(unittest.TestCase):
    """The load-bearing property of everything above.

    A stage whose artifact does not exist must say which command produces it.
    Falling back to synthetic tensors would produce a number that looks like a
    measurement of Llama-3 and is not.
    """

    @unittest.skipIf(has_snapshot("llama_3_8b"), "a snapshot now exists")
    def test_loading_the_snapshot_names_the_harvest_command(self):
        from evograd.benchmark.topdown import load_snapshot

        with self.assertRaises(UnharvestedWorkload) as caught:
            load_snapshot("llama_3_8b")
        message = str(caught.exception)
        self.assertIn("harvest.harvest", message)
        self.assertIn("harvest.snapshot", message)

    def test_the_tier3_gate_names_the_calibration_command(self):
        from evograd.evaluation.tier3.workloads.llama3_8b import gate

        with self.assertRaises(gate.CalibrationUnavailable) as caught:
            gate.load_policy()
        self.assertIn("calibrate", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
