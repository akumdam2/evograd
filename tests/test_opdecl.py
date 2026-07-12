"""Typed declaration derivation and validation tests."""

import unittest

from evograd.opdecl import Const, Duplicated, declare_op
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


class TestDerivedNaming(unittest.TestCase):
    def test_grad_name_override(self):
        op = get_op("evoattention")
        self.assertEqual(op.grad_names(), ("dq", "dk", "dv", "d_pair_bias"))
        self.assertEqual(op.upstream_grad_name, "do")

    def test_grad_order_override(self):
        op = get_op("layernorm_linear")
        self.assertEqual(op.grad_names(), ("dx", "dlinear_weight", "dweight", "dbias"))
        self.assertEqual(op.upstream_grad_name, "dout")

    def test_const_tensor_is_no_grad_input(self):
        op = get_op("evoattention")
        self.assertEqual([c.name for c in op.tensor_const_args()], ["res_mask"])

    def test_registry_covers_all_six(self):
        self.assertEqual(
            sorted(OPS),
            ["evoattention", "layernorm", "layernorm_linear", "linear", "matmul", "rmsnorm"],
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
        args=(Duplicated("x", "[M, N]"), Const("eps", default=1e-5)),
        output=Duplicated("y", "[M, N]"),
        forward_semantics="f",
        backward_semantics="b",
    )
    kwargs.update(overrides)
    return declare_op(**kwargs)


class TestValidation(unittest.TestCase):
    def test_duplicate_arg_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate argument names"):
            _minimal(args=(Duplicated("x", "[M, N]"), Duplicated("x", "[M, N]")))

    def test_unknown_shape_dim_rejected(self):
        with self.assertRaisesRegex(ValueError, "neither a declared dim"):
            _minimal(args=(Duplicated("x", "[M, Q]"),))

    def test_bad_grad_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a\\s+permutation"):
            _minimal(grad_order=("dx", "dweight"))

    def test_bad_forward_ref_rejected(self):
        with self.assertRaisesRegex(ValueError, "module.path:callable"):
            _minimal(forward="no_colon_here")

    def test_integer_literal_shape_dims_allowed(self):
        op = _minimal(args=(Duplicated("x", "[M, 1, N]"), Const("eps", default=1e-5)))
        self.assertEqual(op.duplicated_args()[0].shape, "[M, 1, N]")

    def test_unknown_tolerance_gradient_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown gradients"):
            _minimal(tolerance_multipliers={"dmissing": (2.0, 1.0)})


if __name__ == "__main__":
    unittest.main()
