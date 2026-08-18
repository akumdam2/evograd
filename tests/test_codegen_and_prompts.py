"""Pipeline codegen/prompt derivations — all pure string work, no torch.

Covers the pieces that replaced the old string conventions:
wrapper codegen (grad_reorder / "d"+name matching), example-input derivation
(hand-typed README strings), prompt rendering, and config templating."""

import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evograd.opdecl.activity import example_input_spec
from evograd.atenir.primitive_triton.dispatch import make_kernel
from evograd.ops import get_op
from evograd.pipelines.b_dispatch.program_codegen import (
    _ProgramBuilder,
    _ordered_arg_exprs,
)
from evograd.pipelines.b_dispatch.wrapper_codegen import (
    grad_indices,
    render_autograd_pair_wrapper,
)


class TestWrapperCodegen(unittest.TestCase):
    def test_layernorm_identity_order_returns_directly(self):
        op = get_op("layernorm")
        self.assertEqual(grad_indices(op), [0, 1, 2])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn("def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):", wrapper)
        self.assertIn("def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):", wrapper)
        # Grads are cast to their source input's dtype (autograd contract).
        self.assertIn(
            "return (_grads[0].to(x.dtype), _grads[1].to(weight.dtype), "
            "_grads[2].to(bias.dtype))",
            wrapper,
        )

    def test_single_tensor_op_unpacks_rather_than_binding_the_tuple(self):
        # `x = saved_tensors[:1]` is a plain assignment, so a one-tensor op
        # would bind the whole tuple and fail on the first `.contiguous()`.
        for name in ("softmax", "relu_squared", "sparsemax"):
            op = get_op(name)
            wrapper = render_autograd_pair_wrapper("m:f", op)
            self.assertIn("x, = saved_tensors[:1]", wrapper, name)

    def test_generated_backward_wrapper_executes(self):
        import torch

        for name in ("softmax", "layernorm"):
            op = get_op(name)
            source = (
                "def run_graph_program(*a):\n"
                "    return tuple(torch.zeros_like(t) for t in a[1:])\n"
                + render_autograd_pair_wrapper("m:f", op)
            )
            namespace = {"torch": torch}
            exec(compile(source, "<generated>", "exec"), namespace)
            n_tensors = len(
                [a for a in op.args if getattr(a, "shape", None) is not None]
            )
            saved = tuple(torch.zeros(2, 3) for _ in range(n_tensors))
            grads = namespace[op.backward_fn_name](torch.zeros(2, 3), saved)
            self.assertEqual(len(grads), len(op.grad_names()), name)

    def test_evoattention_drops_inactive_gradient(self):
        op = get_op("evoattention")
        # res_mask is tensor arg index 3; its graph gradient is skipped.
        self.assertEqual(grad_indices(op), [0, 1, 2, 4])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn(
            "return (_grads[0].to(q.dtype), _grads[1].to(k.dtype), "
            "_grads[2].to(v.dtype), _grads[4].to(pair_bias.dtype))",
            wrapper,
        )
        self.assertIn("def evoattention_backward_from_saved(do, saved_tensors):", wrapper)
        self.assertNotIn("eps", wrapper)  # no scalar consts on this op

    def test_layernorm_linear_honors_grad_order(self):
        op = get_op("layernorm_linear")
        # contract order: dx, dlinear_weight, dweight, dbias
        self.assertEqual(grad_indices(op), [0, 3, 1, 2])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn(
            "return (_grads[0].to(x.dtype), _grads[3].to(linear_weight.dtype), "
            "_grads[1].to(weight.dtype), _grads[2].to(bias.dtype))",
            wrapper,
        )

    def test_inactive_tensor_does_not_create_graph_gradient_slot(self):
        op = get_op("fused_moe_swiglu")
        self.assertEqual(grad_indices(op), [0, 1, 2, 3])
        wrapper = render_autograd_pair_wrapper(
            "evograd.ops.level2.fused_moe_swiglu.forward_ref:"
            "fused_moe_swiglu_forward_ref",
            op,
        )
        self.assertNotIn("_grads[4]", wrapper)

    def test_wrapper_compiles_for_every_op(self):
        from evograd.ops import OPS

        for name, op in OPS.items():
            with self.subTest(op=name):
                wrapper = render_autograd_pair_wrapper("m:f", op)
                compile("def run_graph_program(*a): pass\n" + wrapper, f"<{name}>", "exec")


class TestDispatchProgramCodegen(unittest.TestCase):
    def test_ordered_args_preserve_scalar_constants(self):
        self.assertEqual(
            _ordered_arg_exprs(
                [
                    {"kind": "node", "name": "mean"},
                    {"kind": "scalar", "value": 1e-5},
                    {
                        "kind": "shape_list",
                        "items": [
                            {"kind": "sym_node", "name": "rows"},
                            {"kind": "scalar", "value": 64},
                        ],
                    },
                ]
            ),
            ["t_mean", "1e-05", "(t_rows, 64)"],
        )

    def test_dispatch_closure_drops_already_baked_scalar_argument(self):
        def factory(scalar):
            def op(a):
                return a + scalar

            return op

        builder = _ProgramBuilder()
        expression = builder.call_expr_for_node(
            {
                "target": "aten.add.Scalar",
                "args_ordered": [
                    {"kind": "node", "name": "input"},
                    {"kind": "scalar", "value": 2.0},
                ],
            },
            factory(2.0),
            ["t_input", "2.0"],
        )
        self.assertEqual(expression, "_k0(t_input)")

        scalar_first = builder.call_expr_for_node(
            {
                "target": "aten.sub.Tensor",
                "args_ordered": [
                    {"kind": "scalar", "value": 1},
                    {"kind": "node", "name": "input"},
                ],
            },
            factory(1.0),
            ["1", "t_input"],
        )
        self.assertEqual(scalar_first, "_k1(t_input)")

    def test_symbolic_conv_and_moe_nodes_have_handwritten_routes(self):
        cases = (
            {
                "name": "conv_bwd",
                "target": "aten.convolution_backward.default",
                "args_ordered": [
                    {"kind": "node", "name": "dy"},
                    {"kind": "node", "name": "x"},
                    {"kind": "node", "name": "weight"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "sym_node", "name": "channels"}],
                    },
                    {"kind": "scalar", "value": [1, 1]},
                    {"kind": "scalar", "value": [0, 0]},
                    {"kind": "scalar", "value": [1, 1]},
                    {"kind": "scalar", "value": False},
                    {"kind": "scalar", "value": [0, 0]},
                    {"kind": "scalar", "value": 1},
                    {"kind": "scalar", "value": [True, True, True]},
                ],
            },
            {
                "name": "expert_index",
                "target": "aten.index.Tensor",
                "args_ordered": [
                    {"kind": "node", "name": "experts"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "node", "name": "routing"}],
                    },
                ],
            },
            {
                "name": "expert_index_put",
                "target": "aten.index_put.default",
                "args_ordered": [
                    {"kind": "node", "name": "base"},
                    {
                        "kind": "shape_list",
                        "items": [{"kind": "node", "name": "routing"}],
                    },
                    {"kind": "node", "name": "values"},
                    {"kind": "scalar", "value": True},
                ],
            },
        )
        for node in cases:
            with self.subTest(target=node["target"]):
                kernel = make_kernel(node)
                self.assertNotIn("fallback", getattr(kernel, "_dispatch_tag", ""))


class TestExampleInputSpec(unittest.TestCase):
    def test_layernorm_matches_legacy_readme_string(self):
        self.assertEqual(
            example_input_spec(get_op("layernorm")),
            "[(8,64) f32, (64) f32, (64) f32]",
        )

    def test_rmsnorm_matches_legacy_readme_string(self):
        self.assertEqual(
            example_input_spec(get_op("rmsnorm")),
            "[(8,64) f32, (64) f32]",
        )

    def test_evoattention_includes_inactive_mask_with_fixed_dtype(self):
        # First correctness workload with every dim >= 2 (so no symbolic dim
        # specializes to 1 during extraction): B=2 S=2 H=4 N=128 D=64, bfloat16.
        self.assertEqual(
            example_input_spec(get_op("evoattention")),
            "[(2,2,128,4,64) bf16, (2,2,128,4,64) bf16, (2,2,128,4,64) bf16, "
            "(2,2,1,1,128) f32, (2,1,4,128,128) f32]",
        )

    def test_cross_entropy_includes_integer_labels_and_scalar_output(self):
        self.assertEqual(
            example_input_spec(get_op("cross_entropy")),
            "[(8,512) f32, (8) i64]",
        )


class TestPromptRendering(unittest.TestCase):
    def test_pipeline_a_rules_include_contract_and_no_grad(self):
        from evograd.pipelines.a_atenir_llm.prompts import render_pair_rules

        rules = render_pair_rules(get_op("evoattention"))
        self.assertIn("def evoattention_forward_with_saved(q, k, v, res_mask, pair_bias):", rules)
        self.assertIn("return dq, dk, dv, d_pair_bias", rules)
        self.assertIn("`res_mask`", rules)  # no-grad warning present

    def test_pipeline_c_rules_include_contract(self):
        from evograd.pipelines.c_forward_only.prompts import render_pair_rules

        rules = render_pair_rules(get_op("layernorm_linear"))
        self.assertIn("return dx, dlinear_weight, dweight, dbias", rules)


class TestEvolveConfigRendering(unittest.TestCase):
    def test_all_placeholders_replaced(self):
        from evograd.evolve.run import render_config

        for name in ("layernorm", "evoattention"):
            with self.subTest(op=name):
                config = render_config(get_op(name), iterations=7)
                self.assertNotIn("__", config)
                self.assertIn("max_iterations: 7", config)
                self.assertIn(get_op(name).forward_fn_name, config)
                self.assertIn("models:", config)
                self.assertNotIn("primary_model:", config)

    def test_default_map_elites_feature_dimensions(self):
        from evograd.evolve.run import render_config

        config = render_config(get_op("layernorm"))
        self.assertIn(
            'feature_dimensions: ["complexity", "saved_memory_ratio"]', config
        )
        self.assertIn("feature_bins: 10", config)

    def test_custom_feature_dimensions_render_and_validate(self):
        from evograd.evolve.run import render_config

        config = render_config(
            get_op("layernorm"),
            feature_dimensions=("diversity", "speedup"),
            feature_bins=6,
        )
        self.assertIn('feature_dimensions: ["diversity", "speedup"]', config)
        self.assertIn("feature_bins: 6", config)
        with self.assertRaises(ValueError):
            render_config(get_op("layernorm"), feature_dimensions=("not_a_metric",))
        with self.assertRaises(ValueError):
            render_config(get_op("layernorm"), feature_dimensions=())
        with self.assertRaises(ValueError):
            render_config(get_op("layernorm"), feature_bins=0)

    def test_grid_density_controls_render_and_validate(self):
        from evograd.evolve.run import render_config

        config = render_config(get_op("layernorm"))
        self.assertIn("num_islands: 2", config)
        self.assertIn("archive_size: 20", config)
        dense = render_config(
            get_op("layernorm"),
            feature_dimensions=("saved_memory_ratio", "shape_specialization"),
            feature_bins=3,
            num_islands=1,
            archive_size=8,
        )
        self.assertIn("num_islands: 1", dense)
        self.assertIn("archive_size: 8", dense)
        self.assertIn("feature_bins: 3", dense)
        # OpenEvolve clamps int bins up to ceil(archive^(1/dims)): an archive
        # bigger than the grid would silently undo the coarse-bin request.
        with self.assertRaises(ValueError):
            render_config(
                get_op("layernorm"),
                feature_dimensions=("saved_memory_ratio", "shape_specialization"),
                feature_bins=3,
                archive_size=10,
            )

    def test_evaluator_failure_results_carry_feature_metrics(self):
        # OpenEvolve raises if a configured custom feature dimension is missing
        # from any program's metrics, so failure paths must include them too.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from evograd.evolve.evaluator import build_evaluate
        from evograd.evolve.scoring import CUSTOM_FEATURE_DIMENSION_DEFAULTS, get_policy

        evaluate = build_evaluate(get_op("layernorm"), get_policy("speed_memory"))
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = evaluate("/nonexistent/candidate.py")
        for dim, default in CUSTOM_FEATURE_DIMENSION_DEFAULTS.items():
            self.assertEqual(result.metrics.get(dim), default)

    def test_run_wrapper_propagates_declaration_native_benchmark_selection(self):
        from evograd.evolve.run import run_evolve

        with TemporaryDirectory() as td:
            root = Path(td)
            seed = root / "seed.py"
            seed.write_text("# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END\n")
            observed_env = {}

            def fake_run_evolution(**kwargs):
                observed_env.update(os.environ)
                return mock.Mock(best_code="")

            with (
                mock.patch("openevolve.run_evolution", side_effect=fake_run_evolution),
                mock.patch(
                    "evograd.opdecl.baselines.resolve_performance_baseline",
                    return_value="liger",
                ),
            ):
                code = run_evolve(
                    get_op("layernorm"),
                    seed_path=seed,
                    output_dir=root / "out",
                    benchmark_suite="tb_i12",
                    benchmark_dtypes=("bfloat16",),
                    performance_baseline="liger",
                )
            self.assertEqual(code, 1)
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_SUITE"], "tb_i12")
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_DTYPES"], "bfloat16")
            self.assertEqual(observed_env["EVOGRAD_PERFORMANCE_BASELINE"], "liger")


if __name__ == "__main__":
    unittest.main()
