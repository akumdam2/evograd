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
from evograd.ops import get_op
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
        self.assertIn("return run_graph_program(", wrapper)
        self.assertNotIn("_grads[", wrapper)  # identity: no index selection needed

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
        self.assertIn("return (_grads[0], _grads[1], _grads[2], _grads[4])", wrapper)
        self.assertIn("def evoattention_backward_from_saved(do, saved_tensors):", wrapper)
        self.assertNotIn("eps", wrapper)  # no scalar consts on this op

    def test_layernorm_linear_honors_grad_order(self):
        op = get_op("layernorm_linear")
        # contract order: dx, dlinear_weight, dweight, dbias
        self.assertEqual(grad_indices(op), [0, 3, 1, 2])
        wrapper = render_autograd_pair_wrapper("m:f", op)
        self.assertIn("return (_grads[0], _grads[3], _grads[1], _grads[2])", wrapper)

    def test_wrapper_compiles_for_every_op(self):
        from evograd.ops import OPS

        for name, op in OPS.items():
            with self.subTest(op=name):
                wrapper = render_autograd_pair_wrapper("m:f", op)
                compile("def run_graph_program(*a): pass\n" + wrapper, f"<{name}>", "exec")


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
        # First correctness workload: B=1 S=1 H=4 N=23 D=8, float16.
        self.assertEqual(
            example_input_spec(get_op("evoattention")),
            "[(1,1,23,4,8) f16, (1,1,23,4,8) f16, (1,1,23,4,8) f16, "
            "(1,1,1,1,23) f32, (1,1,4,23,23) f32]",
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
                    benchmark_dtypes=("float16",),
                    performance_baseline="liger",
                )
            self.assertEqual(code, 1)
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_SUITE"], "tb_i12")
            self.assertEqual(observed_env["EVOGRAD_BENCHMARK_DTYPES"], "float16")
            self.assertEqual(observed_env["EVOGRAD_PERFORMANCE_BASELINE"], "liger")


if __name__ == "__main__":
    unittest.main()
