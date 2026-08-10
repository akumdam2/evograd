"""Typed declaration derivation and validation tests."""

import unittest

from evograd.opdecl import Active, Inactive, declare_op
from evograd.ops import OPS, get_op


class TestNativeContractRendering(unittest.TestCase):
    def test_layernorm_signatures_come_directly_from_declaration(self):
        op = get_op("layernorm")
        self.assertEqual(op.forward_fn_name, "layernorm_forward_with_saved")
        self.assertEqual(op.forward_parameters(), "x, weight, bias, eps=1e-5")
        self.assertEqual(op.backward_fn_name, "layernorm_backward_from_saved")
        self.assertEqual(op.backward_parameters(), "dy, saved_tensors, eps=1e-5")
        self.assertEqual(op.backward_returns(), "dx, dweight, dbias")

    def test_per_output_tolerance_multipliers(self):
        evo = get_op("evoattention")
        case = evo.correctness[0]
        self.assertEqual(evo.tolerance_for(case), (2e-2, 2e-2))
        self.assertEqual(evo.tolerance_for(case, "d_pair_bias"), (4e-2, 2e-2))

        fused = get_op("layernorm_linear")
        case = fused.correctness[0]
        self.assertEqual(fused.tolerance_for(case, "dweight"), (1.6e-1, 2e-2))

    def test_layernorm_declares_all_legacy_shape_suites(self):
        op = get_op("layernorm")
        for suite in ("mixed", "small", "large", "tb_sweep", "tb_i27"):
            self.assertTrue(op.benchmark_workloads(suite))

    def test_dense_compute_benchmarks_use_explicit_baseline_provenance(self):
        self.assertEqual(
            {case.dtype for case in get_op("matmul").benchmark},
            {"bfloat16"},
        )
        large_matmul = get_op("matmul").benchmark_workloads("large_bf16")
        self.assertEqual(len(large_matmul), 4)
        self.assertTrue(all(case.dims["K"] == 4096 for case in large_matmul))
        extreme_matmul = get_op("matmul").benchmark_workloads("extreme_bf16")
        self.assertEqual(len(extreme_matmul), 1)
        self.assertEqual(
            extreme_matmul[0].dims,
            {"M": 8192, "K": 4096, "N": 14336},
        )
        self.assertEqual(
            set(get_op("matmul").performance_baselines),
            {"cublas_pair"},
        )
        self.assertFalse(get_op("conv2d").performance_baselines)
        self.assertEqual(
            set(get_op("gemm_leaky_relu").performance_baselines),
            {"triton_tutorial"},
        )
        self.assertIn(
            "liger",
            get_op("fused_moe_swiglu").performance_baselines,
        )

    def test_conv2d_pipeline_b_contract_is_scalar_specialized(self):
        op = get_op("conv2d")
        self.assertEqual(op.forward_parameters(), "x, weight, bias")
        for case in (*op.correctness, *op.benchmark):
            dims = case.dims
            self.assertEqual(dims["OH"], dims["H"] - dims["KH"] + 1)
            self.assertEqual(dims["OW"], dims["W"] - dims["KW"] + 1)


class TestDerivedNaming(unittest.TestCase):
    def test_grad_name_override(self):
        op = get_op("evoattention")
        self.assertEqual(op.grad_names(), ("dq", "dk", "dv", "d_pair_bias"))
        self.assertEqual(op.upstream_grad_name, "do")

    def test_grad_order_override(self):
        op = get_op("layernorm_linear")
        self.assertEqual(op.grad_names(), ("dx", "dlinear_weight", "dweight", "dbias"))
        self.assertEqual(op.upstream_grad_name, "dout")

    def test_inactive_tensor_is_no_grad_input(self):
        op = get_op("evoattention")
        self.assertEqual([c.name for c in op.tensor_inactive_args()], ["res_mask"])

    def test_registry_covers_all_twenty_one(self):
        self.assertEqual(
            sorted(OPS),
            [
                "conv2d",
                "cross_entropy",
                "dyt",
                "evoattention",
                "fused_add_rms_norm",
                "fused_moe_swiglu",
                "geglu",
                "gemm_leaky_relu",
                "jsd",
                "kl_div",
                "layernorm",
                "layernorm_linear",
                "linear",
                "matmul",
                "poly_norm",
                "relu_squared",
                "rmsnorm",
                "softmax",
                "sparsemax",
                "swiglu",
                "tvd",
            ],
        )

    def test_registry_is_discovery_based(self):
        import inspect
        import evograd.ops

        source = inspect.getsource(evograd.ops)
        self.assertIn("pkgutil.iter_modules", source)
        self.assertIn("module_info.ispkg", source)
        self.assertNotIn("from evograd.ops import", source)

    def test_every_operator_is_a_self_contained_package(self):
        from pathlib import Path
        from evograd import ops as ops_module

        ops_root = Path(ops_module.__file__).parent
        for name, op in OPS.items():
            with self.subTest(op=name):
                package = ops_root / name
                self.assertTrue((package / "__init__.py").is_file())
                self.assertTrue((package / "forward_ref.py").is_file())
                self.assertTrue(op.forward.startswith(f"evograd.ops.{name}."))


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

    def test_unknown_tolerance_gradient_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown gradients"):
            _minimal(tolerance_multipliers={"dmissing": (2.0, 1.0)})


class TestLigerSuiteMigration(unittest.TestCase):
    def test_final_fork_workload_counts(self):
        expected = {
            "cross_entropy": (6, 42),
            "dyt": (6, 15),
            "fused_add_rms_norm": (6, 16),
            "fused_moe_swiglu": (2, 6),
            "geglu": (6, 42),
            "jsd": (6, 14),
            "kl_div": (6, 39),
            "poly_norm": (6, 14),
            "relu_squared": (6, 14),
            "softmax": (6, 36),
            "sparsemax": (4, 15),
            "swiglu": (6, 42),
            "tvd": (6, 14),
        }
        for name, counts in expected.items():
            with self.subTest(op=name):
                op = get_op(name)
                self.assertEqual((len(op.correctness), len(op.benchmark)), counts)

    def test_shape_regime_suites_partition_full_suite(self):
        for name in (
            "cross_entropy",
            "dyt",
            "fused_add_rms_norm",
            "geglu",
            "jsd",
            "kl_div",
            "poly_norm",
            "relu_squared",
            "softmax",
            "sparsemax",
            "swiglu",
            "tvd",
        ):
            with self.subTest(op=name):
                op = get_op(name)
                full = op.benchmark_workloads("full")
                small = op.benchmark_workloads("small")
                large = op.benchmark_workloads("large")
                self.assertEqual(len(full), len(small) + len(large))
                self.assertTrue(small)
                self.assertTrue(large)

    def test_inactive_targets_do_not_count_toward_memory_budget(self):
        self.assertEqual(get_op("cross_entropy").memory_input_names(), ("logits",))
        self.assertEqual(get_op("kl_div").memory_input_names(), ("y_pred",))
        self.assertEqual(get_op("jsd").memory_input_names(), ("log_q",))

    def test_liger_hooks_keep_the_reviewed_pair_for_the_runtime_gate(self):
        names = (
            "cross_entropy",
            "dyt",
            "fused_add_rms_norm",
            "fused_moe_swiglu",
            "geglu",
            "jsd",
            "kl_div",
            "layernorm",
            "poly_norm",
            "relu_squared",
            "softmax",
            "sparsemax",
            "swiglu",
            "tvd",
        )
        for name in names:
            with self.subTest(op=name):
                hook = get_op(name).performance_baselines["liger"]
                self.assertTrue(callable(getattr(hook, "pair_factory", None)))
                self.assertIsInstance(hook.forward_args, tuple)


if __name__ == "__main__":
    unittest.main()
