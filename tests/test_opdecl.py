"""Typed declaration derivation and validation tests."""

import unittest

from evograd.opdecl import Active, Inactive, Provenance, Workload, declare_op
from evograd.benchmark import TASKS, get_task


class TestNativeContractRendering(unittest.TestCase):
    def test_layernorm_signatures_come_directly_from_declaration(self):
        op = get_task("layernorm")
        self.assertEqual(op.forward_fn_name, "layernorm_forward_with_saved")
        self.assertEqual(op.forward_parameters(), "x, weight, bias, eps=1e-5")
        self.assertEqual(op.backward_fn_name, "layernorm_backward_from_saved")
        self.assertEqual(op.backward_parameters(), "dy, saved_tensors, eps=1e-5")
        self.assertEqual(op.backward_returns(), "dx, dweight, dbias")

class TestDerivedNaming(unittest.TestCase):
    def test_grad_name_override(self):
        op = get_task("megafold_evoattention")
        self.assertEqual(op.grad_names(), ("dq", "dk", "dv", "d_pair_bias"))
        self.assertEqual(op.upstream_grad_name, "do")

    def test_grad_order_override(self):
        op = _minimal(
            dims=("M", "K", "N"),
            args=(Active("x", "[M, K]"), Active("weight", "[K]"), Active("bias", "[K]"),
                  Active("linear_weight", "[K, N]")),
            output=Active("out", "[M, N]"),
            grad_order=("dx", "dlinear_weight", "dweight", "dbias"),
        )
        self.assertEqual(op.grad_names(), ("dx", "dlinear_weight", "dweight", "dbias"))
        self.assertEqual(op.upstream_grad_name, "dout")

    def test_inactive_tensor_is_no_grad_input(self):
        op = get_task("megafold_evoattention")
        self.assertEqual([c.name for c in op.tensor_inactive_args()], ["key_mask"])

    def test_registry_is_grouped_by_benchmark_level(self):
        by_level = {}
        for name, op in TASKS.items():
            by_level.setdefault(op.level, []).append(name)
        self.assertEqual(
            {level: sorted(names) for level, names in sorted(by_level.items())},
            {
                1: [
                    # The Gated DeltaNet's short convolution (Qwen3.5).
                    "causal_conv1d",
                    # The one primitive a decoder-only step runs that had no
                    # generic task before the Qwen3 Level-1 mapping.
                    "causal_gqa_attention",
                    "cross_entropy",
                    "embedding_bag",
                    # The GDN output norm and gates.
                    "gated_rmsnorm",
                    "gdn_gating",
                    "gelu",
                    # The zero-centred (1 + w) RMSNorm.
                    "gemma_rmsnorm",
                    "glu",
                    "grn",
                    # The delta rule's q/k normalization.
                    "l2norm",
                    "layernorm",
                    "layernorm_no_affine",
                    "layernorm_no_bias",
                    "linear",
                    # Biasless projections -- what a modern decoder runs.
                    "linear_no_bias",
                    # Gated attention's 64-of-256-channel rotary embedding.
                    "partial_rope",
                    "quick_gelu",
                    "relu",
                    "rmsnorm",
                    "rope",
                    "sigmoid",
                    # MegaFold's attention probabilities.
                    "softmax",
                    "spatial_conv2d",
                    "swiglu",
                ],
                2: [
                    # ConvNeXt V2's two boundaries in each ConvNextV2Layer.
                    "convnextv2_tiny_convnext_mlp",
                    "convnextv2_tiny_dwconv_layernorm",
                    # DiT-S/2's four boundaries in each block.
                    "dit_s_2_dit_adaln_residual",
                    "dit_s_2_dit_attention",
                    "dit_s_2_dit_mlp",
                    "dit_s_2_dit_modulation",
                    # Qwen3-0.6B's residual fusion, under the generic key it
                    # was declared with before a second architecture existed.
                    "fused_add_rms_norm",
                    # Each harvested workload's own four decoder boundaries,
                    # rather than operators specified up front. Three of the
                    # four compute the same mathematics in both models and
                    # share one implementation; the tasks stay distinct because
                    # the cases, widths, provenance and tolerances are the
                    # model's, and a report row keys off the name.
                    "llama3_attention_block",
                    "llama3_residual_rmsnorm",
                    "llama3_swiglu_mlp",
                    # MegaFold's mixed-precision LayerNorm+Linear and
                    # attention: what its autocast trunk computes there.
                    "megafold_evoattention",
                    "megafold_layernorm_linear",
                    "megafold_msa_pair_weighted_averaging",
                    "megafold_outer_product_mean",
                    "megafold_transition",
                    "megafold_triangle_attention_ending",
                    "megafold_triangle_attention_starting",
                    "megafold_triangle_multiplication_incoming",
                    "megafold_triangle_multiplication_outgoing",
                    # OLMoE-1B-7B's three sites: the whole attention module
                    # (full-width q/k norms), the routed experts (the router
                    # stays native) and the residual fusion.
                    "olmoe_attention_block",
                    "olmoe_moe_experts",
                    "olmoe_residual_rmsnorm",
                    # Qwen2-VL-2B's three boundaries in each of its two
                    # towers: attention, MLP and the residual add + norm.
                    "qwen2_vl_2b_text_attention_block",
                    "qwen2_vl_2b_text_mlp",
                    "qwen2_vl_2b_text_residual_rmsnorm",
                    "qwen2_vl_2b_vision_attention",
                    "qwen2_vl_2b_vision_mlp",
                    "qwen2_vl_2b_vision_residual_layernorm",
                    # Qwen3.5-4B's six sites (two layer kinds, dense MLP).
                    "qwen3_5_attention_block",
                    "qwen3_5_gated_delta_rule",
                    "qwen3_5_gdn_in_proj",
                    "qwen3_5_gdn_norm_out",
                    "qwen3_5_residual_rmsnorm",
                    "qwen3_5_swiglu_mlp",
                    # Qwen3-0.6B's whole attention module and its LM head with
                    # the loss (with fused_add_rms_norm and the MLP above).
                    "qwen3_attention_block",
                    "qwen3_lm_head_ce",
                    "qwen3_swiglu_mlp",
                    "torchrec_dlrm_dlrm_interaction",
                ],
            },
        )

    def test_no_operator_claims_level_three(self):
        """Level 3 is the whole model, a ``WorkloadDecl``; an OpDecl is level 1 or 2."""
        self.assertFalse([name for name, op in TASKS.items() if (op.level or 0) >= 3])
        with self.assertRaises(ValueError):
            _minimal(level=3, family="synthetic_block")

    def test_registry_is_discovery_based(self):
        import inspect
        import evograd.ops

        source = inspect.getsource(evograd.ops)
        self.assertIn("pkgutil.iter_modules", source)
        self.assertIn("module_info.ispkg", source)
        self.assertNotIn("from evograd.ops import", source)

    def test_every_task_is_a_self_contained_package(self):
        """A task's package holds its contract and its reference, and its
        ``forward`` names that package.

        Where the package lives depends on what the task is: a primitive under
        ``ops/level1``, a model-specific one under that model's top-down
        package. The declaration's ``forward`` is what ties the two cases
        together, so the check is written against it rather than against a
        directory guess.
        """
        from pathlib import Path

        import evograd

        src_root = Path(evograd.__file__).parent
        for name, op in TASKS.items():
            with self.subTest(op=name):
                module_path, _, _symbol = op.forward.partition(":")
                self.assertTrue(
                    module_path.startswith("evograd."),
                    f"{name}: forward must name an evograd module, got {op.forward!r}",
                )
                reference = src_root.joinpath(
                    *module_path.split(".")[1:]
                ).with_suffix(".py")
                self.assertTrue(
                    reference.is_file(), f"{name}: no reference at {reference}"
                )
                self.assertTrue(
                    (reference.parent / "__init__.py").is_file(),
                    f"{name}: {reference.parent} is not a package",
                )

    def test_a_task_lives_where_its_kind_says_it_does(self):
        """Primitives under ops, fusions under benchmark. The registry is the
        authority on the kind; this pins the directory to it."""
        for name, op in TASKS.items():
            with self.subTest(op=name):
                if op.level == 1:
                    self.assertTrue(
                        op.forward.startswith(f"evograd.ops.level1.{name}."),
                        f"{name} is a primitive but its reference is {op.forward!r}",
                    )
                else:
                    self.assertTrue(
                        op.forward.startswith("evograd.benchmark."),
                        f"{name} is a level-{op.level} task but its reference is "
                        f"{op.forward!r}; fused tasks belong to evograd.benchmark",
                    )


def _minimal(**overrides):
    kwargs = dict(
        name="toy",
        forward="toy.forward_ref:toy_forward_ref",
        dims=("M", "N"),
        args=(Active("x", "[M, N]"), Inactive("eps", default=1e-5)),
        output=Active("y", "[M, N]"),
        forward_semantics="f",
        backward_semantics="b",
    )
    kwargs.update(overrides)
    return declare_op(**kwargs)


class TestValidation(unittest.TestCase):
    def test_duplicate_arg_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate argument names"):
            _minimal(args=(Active("x", "[M, N]"), Active("x", "[M, N]")))

    def test_unknown_shape_dim_rejected(self):
        with self.assertRaisesRegex(ValueError, "neither a declared dim"):
            _minimal(args=(Active("x", "[M, Q]"),))

    def test_bad_grad_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a\\s+permutation"):
            _minimal(grad_order=("dx", "dweight"))

    def test_bad_forward_ref_rejected(self):
        with self.assertRaisesRegex(ValueError, "module.path:callable"):
            _minimal(forward="no_colon_here")

    def test_integer_literal_shape_dims_allowed(self):
        op = _minimal(args=(Active("x", "[M, 1, N]"), Inactive("eps", default=1e-5)))
        self.assertEqual(op.active_args()[0].shape, "[M, 1, N]")

    def test_scalar_tensor_shape_is_allowed(self):
        op = _minimal(args=(Active("alpha", "[]"),))
        self.assertEqual(op.active_args()[0].shape, "[]")

    def test_memory_inputs_must_name_tensor_inputs(self):
        with self.assertRaisesRegex(ValueError, "memory_inputs"):
            _minimal(memory_inputs=("eps",))

class TestLigerSuiteMigration(unittest.TestCase):
    # Fork-era benchmark cases. The timed suites moved onto model-derived grids
    # in v1, so the fork counts now live in each declaration's ``legacy``
    # ablation suite. Asserting them there is a stronger check than the
    # original: it proves the migration *preserved* the historical grids rather
    # than silently discarding them.
    _FORK_COUNTS = {
        "fused_add_rms_norm": 16,
    }

    def test_final_fork_workload_counts_are_preserved_as_ablation_suites(self):
        for name, benchmark in self._FORK_COUNTS.items():
            with self.subTest(op=name):
                op = get_task(name)
                self.assertEqual(
                    len(op.benchmark_workloads("legacy")),
                    benchmark,
                    f"{name}: the fork's benchmark grid must survive as 'legacy'",
                )

    def test_timed_suites_are_model_derived(self):
        for name in self._FORK_COUNTS:
            with self.subTest(op=name):
                op = get_task(name)
                sources = {case.provenance.source for case in op.benchmark}
                self.assertEqual(sources, {"hf_config"}, name)

    def test_shape_regime_suites_partition_full_suite(self):
        for name in ("fused_add_rms_norm",):
            with self.subTest(op=name):
                op = get_task(name)
                full = op.benchmark_workloads("full")
                small = op.benchmark_workloads("small")
                large = op.benchmark_workloads("large")
                self.assertEqual(len(full), len(small) + len(large))
                self.assertTrue(small)
                self.assertTrue(large)

    def test_inactive_targets_do_not_count_toward_memory_budget(self):
        self.assertEqual(get_task("cross_entropy").memory_input_names(), ("logits",))

    def test_no_task_declares_a_liger_baseline(self):
        """Liger was retired; nothing registers it."""
        self.assertFalse(
            [name for name, op in TASKS.items() if "liger" in op.performance_baselines]
        )


if __name__ == "__main__":
    unittest.main()
