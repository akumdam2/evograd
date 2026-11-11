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


class TestAttentionBlockDeclaration(unittest.TestCase):
    """``llama3_attention_block``: the whole ``LlamaAttention`` module."""

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark import get_task

        cls.op = get_task("llama3_attention_block")

    def test_it_has_no_per_head_norm_weights_and_no_eps(self):
        """Llama-3 has no per-head query/key RMSNorm: the four projection
        weights are the only parameters, and there is no normalization for an
        ``eps`` to belong to."""
        names = {arg.name for arg in self.op.args}
        self.assertFalse(names & {"q_norm_weight", "k_norm_weight", "eps"})
        self.assertEqual(self.op.parameter_args,
                         ("q_weight", "k_weight", "v_weight", "o_weight"))
        self.assertEqual(self.op.grad_names(),
                         ("dx", "dq_weight", "dk_weight", "dv_weight", "do_weight"))

    def test_it_returns_the_module_output(self):
        self.assertEqual(self.op.output_names, ("out",))

    def test_the_benchmark_shape_is_llama_3_2_1bs(self):
        """Derived from the published configuration, not written out."""
        from evograd.opdecl.models import LLAMA_3_2_1B

        self.assertEqual(
            self.op.benchmark[0].dims,
            LLAMA_3_2_1B.attention_block_dims(batch=2, seq=2048),
        )

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_eager_pair_passes_gate1_against_the_declared_reference(self):
        """The production spelling, as the eager pair, against the declared
        forward under a model-reference contract, by Gate 1's rule."""
        from evograd.evaluation.tier3.patch import eager_pair_for
        from tests.verification import _fixtures as F

        report = F.gate1_report(self.op, eager_pair_for(self.op),
                                workloads=SMALL["llama3_attention_block"])
        self.assertEqual(report.status, "passed", report.to_dict())

    @unittest.skipIf(torch is None, "torch is required")
    def test_the_runtime_spelling_is_llamaattention_bit_for_bit(self):
        """The model reference every gate judges against is the module's own
        arithmetic: a live ``LlamaAttention`` (SDPA, train mode, no mask) and
        the runtime forward agree exactly, forward and every gradient."""
        try:
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import (
                LlamaAttention, LlamaRotaryEmbedding)
        except ImportError:  # pragma: no cover
            self.skipTest("transformers is not installed")
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.integrations.base import shared_row

        forward = resolve_runtime_forward(self.op)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                config = LlamaConfig(hidden_size=64, num_attention_heads=4,
                                     num_key_value_heads=1, head_dim=16,
                                     intermediate_size=128, num_hidden_layers=1,
                                     attention_dropout=0.0)
                config._attn_implementation = "sdpa"
                torch.manual_seed(0)
                module = LlamaAttention(config, layer_idx=0).to(dtype).train()
                x = torch.randn(2, 24, 64, dtype=dtype, requires_grad=True)
                ids = torch.arange(24)[None].expand(2, -1)
                cos, sin = LlamaRotaryEmbedding(config)(x, ids)
                live, _ = module(hidden_states=x, position_embeddings=(cos, sin),
                                 attention_mask=None)
                dout = torch.randn_like(live)
                weights = [module.q_proj.weight, module.k_proj.weight,
                           module.v_proj.weight, module.o_proj.weight]
                live_grads = torch.autograd.grad(live, [x, *weights], dout)
                ours = forward(x, *weights, shared_row(cos), shared_row(sin))
                our_grads = torch.autograd.grad(ours, [x, *weights], dout)
                self.assertTrue(torch.equal(ours, live))
                for name, a, b in zip(self.op.grad_names(), our_grads, live_grads):
                    self.assertTrue(torch.equal(a, b), name)


#: What ``mapping()`` reports as the composition, once a harvest exists.
COMPOSES_INTO_EXPECTED = {
    "linear_no_bias": ("llama3_attention_block", "llama3_swiglu_mlp"),
    "rmsnorm": ("llama3_residual_rmsnorm",),
    "rope": ("llama3_attention_block",),
    "swiglu": ("llama3_swiglu_mlp",),
    "causal_gqa_attention": ("llama3_attention_block",),
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
        self.assertEqual(COMPOSES_INTO["rope"], ("llama3_attention_block",))

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
    def test_the_attention_boundary_is_llamas_own_declaration(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest

        self.assertEqual(manifest.SITES, ("attention_block", "swiglu_mlp", "residual_rmsnorm"))
        self.assertEqual(manifest.task_key("attention_block"), "llama3_attention_block")
        self.assertEqual(manifest.frequency("attention_block"), 16)

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
