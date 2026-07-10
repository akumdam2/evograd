"""Fidelity tests: OpDecl-derived specs must match the legacy contract exactly.

``tests/fixtures/*_spec.json`` are unmodified snapshots of the old repo's
``pipeline/autograd_pair_fusion_agent/<op>_spec.json`` files. If these tests
pass, pipelines consuming ``to_operator_spec(op)`` behave identically to
pipelines loading the legacy JSONs.
"""

import json
import unittest
from pathlib import Path

from evograd.opdecl import Const, Duplicated, declare_op, to_operator_spec, to_spec_dict
from evograd.ops import OPS, get_op

FIXTURES = Path(__file__).parent / "fixtures"

JSON_BACKED_OPS = ("rmsnorm", "matmul", "linear", "layernorm_linear", "evoattention")


class TestLegacySpecFidelity(unittest.TestCase):
    def test_all_json_specs_roundtrip(self):
        for name in JSON_BACKED_OPS:
            with self.subTest(op=name):
                expected = json.loads(
                    (FIXTURES / f"{name}_spec.json").read_text(encoding="utf-8")
                )
                self.assertEqual(to_spec_dict(get_op(name)), expected)

    def test_layernorm_matches_prompts_py_constant(self):
        # layernorm's legacy spec is LAYERNORM_SPEC in prompts.py, not a JSON.
        spec = to_operator_spec(get_op("layernorm"))
        self.assertEqual(spec.forward_fn_name, "layernorm_forward_with_saved")
        self.assertEqual(spec.forward_args, "x, weight, bias, eps=1e-5")
        self.assertEqual(spec.backward_fn_name, "layernorm_backward_from_saved")
        self.assertEqual(spec.backward_args, "dy, saved_tensors, eps=1e-5")
        self.assertEqual(spec.backward_returns, "dx, dweight, dbias")
        self.assertEqual(spec.no_grad_inputs, ())
        self.assertEqual(spec.extra_constraints, "")


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


if __name__ == "__main__":
    unittest.main()
