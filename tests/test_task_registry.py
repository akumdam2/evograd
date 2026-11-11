"""What the three registries contain, and what must not change about them.

The refactor that split ``evograd.ops`` into a primitive registry and
``evograd.benchmark`` into the executable task registry moved a lot of files.
None of it was allowed to move a *contract*: the same task names resolve, to
the same arguments, outputs, gradients, cases and provenance as before. These
tests pin that, and pin the composition of the registries themselves so a
later move cannot quietly drop a task or register one twice.

Counts are asserted against the registries rather than against a table copied
into documentation, which is the drift this whole layout exists to prevent.
"""

from __future__ import annotations

import unittest

from evograd.benchmark import TASKS, get_task, load_task, tasks_at_level
from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest as llama_manifest
from evograd.benchmark.topdown.qwen3_0_6b.levels.level2 import manifest
from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest as qwen3_5_manifest
from evograd.ops import PRIMITIVES, get_primitive
from tests.qwen3 import QWEN3_SNAPSHOT_STALE, qwen3_snapshot_current

QWEN_LEVEL2 = (
    "fused_add_rms_norm",
    "qwen3_attention_block",
    "qwen3_lm_head_ce",
    "qwen3_swiglu_mlp",
)
#: Llama-3.2-1B's three. A harvested architecture owns its Level-2 identities
#: rather than borrowing another's, so these are more tasks and not more
#: suites on Qwen3's.
LLAMA_LEVEL2 = (
    "llama3_attention_block",
    "llama3_residual_rmsnorm",
    "llama3_swiglu_mlp",
)
#: Qwen3.5-4B's seven sites (lm_head_ce deferred).
QWEN3_5_LEVEL2 = (
    "qwen3_5_gated_attention",
    "qwen3_5_gated_delta_rule",
    "qwen3_5_gdn_in_proj",
    "qwen3_5_gdn_norm_out",
    "qwen3_5_qkv_norm_partial_rope",
    "qwen3_5_residual_rmsnorm",
    "qwen3_5_swiglu_mlp",
)
#: OLMoE-1B-7B's four sites (the router and lm_head_ce are not sites).
OLMOE_LEVEL2 = (
    "olmoe_attention",
    "olmoe_moe_experts",
    "olmoe_qkv_norm_rope",
    "olmoe_residual_rmsnorm",
)
VISION_LEVEL2 = (
    "convnextv2_tiny_convnext_mlp",
    "dit_s_2_dit_mlp",
    "dit_s_2_dit_modulation",
    "qwen2_vl_2b_text_mlp",
    "qwen2_vl_2b_vision_mlp",
)
MEGAFOLD_LEVEL2 = (
    "megafold_evoattention",
    "megafold_layernorm_linear",
    "megafold_msa_pair_weighted_averaging",
    "megafold_outer_product_mean",
    "megafold_transition",
    "megafold_triangle_attention_ending",
    "megafold_triangle_attention_starting",
    "megafold_triangle_multiplication_incoming",
    "megafold_triangle_multiplication_outgoing",
)
DLRM_LEVEL2 = ("torchrec_dlrm_dlrm_interaction",)
MODEL_LEVEL2 = (
    QWEN_LEVEL2 + LLAMA_LEVEL2 + QWEN3_5_LEVEL2 + OLMOE_LEVEL2 + VISION_LEVEL2
    + MEGAFOLD_LEVEL2 + DLRM_LEVEL2
)
DELETED_LEGACY = ("af3_single_repr_block", "llama3_decoder_layer")


class TestPrimitiveRegistry(unittest.TestCase):
    def test_holds_only_level_one(self):
        for name, op in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                self.assertEqual(op.level, 1)

    def test_holds_the_expected_primitives(self):
        self.assertEqual(
            sorted(PRIMITIVES),
            [
                "causal_conv1d", "causal_gqa_attention", "cross_entropy", "embedding_bag",
                "gated_rmsnorm", "gdn_gating", "gelu", "gemma_rmsnorm", "glu", "grn", "l2norm",
                "layernorm", "layernorm_no_affine", "layernorm_no_bias", "linear",
                "linear_no_bias", "partial_rope", "quick_gelu", "relu", "rmsnorm", "rope",
                "sigmoid", "softmax", "spatial_conv2d", "swiglu",
            ],
        )

    def test_no_fused_or_model_specific_task_appears(self):
        for name in MODEL_LEVEL2 + DELETED_LEGACY:
            with self.subTest(op=name):
                self.assertNotIn(name, PRIMITIVES)

    def test_get_primitive_refuses_a_task_that_is_not_one(self):
        with self.assertRaises(KeyError):
            get_primitive("qwen3_attention_block")

    def test_the_primitive_contract_carries_no_model_suite(self):
        """A primitive's own declaration must not name a harvested workload.

        The observed cases are real and still reported -- they are bound on by
        :mod:`evograd.benchmark.cases`. What must not happen is the primitive
        package carrying them, because that is the coupling that made ``ops``
        import a snapshot.
        """
        for name, op in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                for suite in op.benchmark_suites:
                    self.assertNotIn("qwen3_0_6b", suite)
                    self.assertNotIn("llama_3_8b", suite)
                    self.assertNotIn("llama_3_2_1b", suite)
                    self.assertNotIn("qwen3_5_4b", suite)


class TestTaskRegistry(unittest.TestCase):
    def test_contains_every_executable_task_exactly_once(self):
        self.assertEqual(len(TASKS), len(set(TASKS)))
        self.assertEqual(len(TASKS), len(PRIMITIVES) + len(MODEL_LEVEL2))
        self.assertEqual(len(TASKS), 58)

    def test_contains_the_qwen_level_two_tasks(self):
        for name in QWEN_LEVEL2:
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 2)

    def test_contains_the_llama_level_two_tasks(self):
        for name in LLAMA_LEVEL2:
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 2)

    def test_contains_the_qwen3_5_level_two_tasks(self):
        for name in QWEN3_5_LEVEL2:
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 2)
        self.assertEqual(sorted(qwen3_5_manifest.SITE_TASKS.values()), sorted(QWEN3_5_LEVEL2))
        self.assertNotIn("qwen3_5_lm_head_ce", TASKS)

    def test_contains_the_olmoe_level_two_tasks(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest as olmoe_manifest

        for name in OLMOE_LEVEL2:
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 2)
        self.assertEqual(sorted(olmoe_manifest.SITE_TASKS.values()), sorted(OLMOE_LEVEL2))
        self.assertNotIn("olmoe_router", TASKS)

    def test_the_two_models_share_no_level_two_task(self):
        """Each architecture's boundaries are its own.

        Llama's MLP and residual boundaries compute the same mathematics as
        Qwen3's, and the implementation is shared. The *task* is not: a report row, a
        candidate program and a calibration file all key off the name, so one
        key serving two models' widths would make each of them ambiguous.
        """
        self.assertEqual(set(QWEN_LEVEL2) & set(LLAMA_LEVEL2), set())
        self.assertEqual(
            sorted(llama_manifest.SITE_TASKS.values()), sorted(LLAMA_LEVEL2)
        )
        self.assertEqual(sorted(manifest.SITE_TASKS.values()), sorted(QWEN_LEVEL2))

    def test_the_deleted_legacy_blocks_are_absent(self):
        for name in DELETED_LEGACY:
            with self.subTest(op=name):
                self.assertNotIn(name, TASKS)
        self.assertEqual(tasks_at_level(3), {})

    def test_levels_present_are_one_and_two_only(self):
        self.assertEqual({op.level for op in TASKS.values()}, {1, 2})

    def test_get_task_resolves_every_registered_name(self):
        for name in TASKS:
            with self.subTest(op=name):
                self.assertIs(get_task(name), TASKS[name])

    def test_get_task_refuses_a_deleted_legacy_name(self):
        for name in DELETED_LEGACY:
            with self.subTest(op=name):
                with self.assertRaises(KeyError):
                    get_task(name)

    def test_external_declarations_still_load(self):
        self.assertTrue(callable(load_task))
        with self.assertRaises(ValueError):
            load_task("no_colon_here")


class TestObservedCaseBinding(unittest.TestCase):
    """The cases moved owner without changing value."""

    OBSERVED = ("causal_gqa_attention", "cross_entropy", "linear_no_bias",
                "rmsnorm", "rope", "swiglu")

    def test_the_bound_task_serves_the_observed_suite(self):
        for name in self.OBSERVED:
            with self.subTest(op=name):
                suites = TASKS[name].benchmark_suites
                self.assertIn("qwen3_0_6b_observed", suites)
                self.assertTrue(suites["qwen3_0_6b_observed"])

    def test_every_observed_case_carries_qwen_provenance(self):
        for name in self.OBSERVED:
            for case in TASKS[name].benchmark_suites["qwen3_0_6b_observed"]:
                with self.subTest(op=name, dims=tuple(sorted(case.dims.items()))):
                    self.assertIsNotNone(case.provenance)
                    self.assertEqual(case.provenance.model, "qwen3_0_6b")
                    self.assertEqual(case.provenance.source, "hf_config")

    def test_the_observed_cases_reach_untimed_coverage(self):
        for name in self.OBSERVED:
            with self.subTest(op=name):
                observed = TASKS[name].benchmark_suites["qwen3_0_6b_observed"]
                for case in observed:
                    self.assertIn(case, TASKS[name].coverage)

    def test_binding_does_not_mutate_the_primitive(self):
        for name in self.OBSERVED:
            with self.subTest(op=name):
                self.assertNotIn("qwen3_0_6b_observed", PRIMITIVES[name].benchmark_suites)

class TestQwenLevelTwoInvariance(unittest.TestCase):
    """Site-to-task mapping, output names and gradient order are contracts."""

    EXPECTED_SITES = {
        "attention_block": "qwen3_attention_block",
        "swiglu_mlp": "qwen3_swiglu_mlp",
        "residual_rmsnorm": "fused_add_rms_norm",
        "lm_head_ce": "qwen3_lm_head_ce",
    }
    EXPECTED_OUTPUTS = {
        "qwen3_attention_block": ("out",),
        "qwen3_swiglu_mlp": ("out",),
        "fused_add_rms_norm": ("out", "summed"),
        "qwen3_lm_head_ce": ("loss",),
    }
    EXPECTED_GRADS = {
        "qwen3_attention_block": ("dx", "dq_weight", "dk_weight", "dv_weight", "do_weight",
                                  "dq_norm_weight", "dk_norm_weight"),
        "qwen3_swiglu_mlp": ("dx", "dgate_weight", "dup_weight", "ddown_weight"),
        "qwen3_lm_head_ce": ("dx", "dweight"),
    }

    def test_the_site_to_task_mapping_is_unchanged(self):
        self.assertEqual(manifest.SITE_TASKS, self.EXPECTED_SITES)

    def test_the_manifest_and_the_tier3_registry_agree(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import sites

        self.assertEqual(sites.qwen3_sites().site_ops, manifest.SITE_TASKS)

    def test_structured_outputs_are_unchanged(self):
        for task, names in self.EXPECTED_OUTPUTS.items():
            with self.subTest(op=task):
                self.assertEqual(TASKS[task].output_names, names)
                self.assertEqual(TASKS[task].is_multi_output, len(names) > 1)

    def test_gradient_order_is_unchanged(self):
        for task, order in self.EXPECTED_GRADS.items():
            with self.subTest(op=task):
                self.assertEqual(TASKS[task].grad_order, order)

    def test_the_reference_resolves_from_the_site_package(self):
        for site, task in self.EXPECTED_SITES.items():
            with self.subTest(site=site):
                self.assertTrue(
                    TASKS[task].forward.startswith(
                        "evograd.benchmark.topdown.qwen3_0_6b.levels.level2."
                        f"{site}.reference:"
                    ),
                    TASKS[task].forward,
                )

    @unittest.skipUnless(qwen3_snapshot_current(), QWEN3_SNAPSHOT_STALE)
    def test_the_manifest_reads_frequency_from_the_snapshot(self):
        for site in manifest.SITES:
            with self.subTest(site=site):
                self.assertGreater(manifest.frequency(site), 0)
                self.assertTrue(manifest.module_paths(site))

    def test_the_manifest_names_the_canonical_workload(self):
        self.assertEqual(
            manifest.workload_id(),
            "qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad",
        )


class TestQwen3_5LevelTwoInvariance(unittest.TestCase):
    """Block kinds, outputs and gradient order are contracts."""

    EXPECTED_OUTPUTS = {
        "qwen3_5_gdn_in_proj": ("q", "k", "v", "z", "beta", "g"),
        "qwen3_5_gated_delta_rule": ("o",),
        "qwen3_5_gdn_norm_out": ("y",),
        "qwen3_5_qkv_norm_partial_rope": ("q", "k", "v", "gate"),
        "qwen3_5_gated_attention": ("y",),
        "qwen3_5_residual_rmsnorm": ("normed", "summed"),
        "qwen3_5_swiglu_mlp": ("out",),
    }
    EXPECTED_GRADS = {
        "qwen3_5_gdn_in_proj": ("dx", "dw_qkv", "dw_z", "dw_b", "dw_a", "dconv_w", "dA_log",
                                "ddt_bias"),
        "qwen3_5_gated_delta_rule": ("dq", "dk", "dv", "dg", "dbeta"),
        "qwen3_5_gdn_norm_out": ("do", "dz", "dw_norm", "dw_out"),
        "qwen3_5_qkv_norm_partial_rope": ("dx", "dq_weight", "dk_weight", "dv_weight",
                                          "dq_norm_weight", "dk_norm_weight"),
        "qwen3_5_gated_attention": ("dq", "dk", "dv", "dgate", "do_weight"),
        "qwen3_5_residual_rmsnorm": ("dbranch", "dresidual", "dweight"),
        "qwen3_5_swiglu_mlp": ("dx", "dgate_weight", "dup_weight", "ddown_weight"),
    }

    def test_sites_by_block_kind(self):
        self.assertEqual(
            qwen3_5_manifest.SITES_BY_BLOCK["linear_attention"],
            ("gdn_in_proj", "gated_delta_rule", "gdn_norm_out", "residual_rmsnorm", "swiglu_mlp"))
        self.assertEqual(
            qwen3_5_manifest.SITES_BY_BLOCK["full_attention"],
            ("qkv_norm_partial_rope", "gated_attention", "residual_rmsnorm", "swiglu_mlp"))

    def test_outputs_and_gradient_order(self):
        for task, names in self.EXPECTED_OUTPUTS.items():
            with self.subTest(op=task):
                self.assertEqual(TASKS[task].output_names, names)
                self.assertEqual(TASKS[task].grad_names(), self.EXPECTED_GRADS[task])

    def test_every_task_resolves_from_its_site(self):
        for site, task in qwen3_5_manifest.SITE_TASKS.items():
            with self.subTest(site=site):
                self.assertTrue(TASKS[task].forward.startswith(
                    f"evograd.benchmark.topdown.qwen3_5_4b.levels.level2.{site}.reference:"))
                self.assertIn("qwen3_5_4b_observed", TASKS[task].benchmark_suites)


if __name__ == "__main__":
    unittest.main()
