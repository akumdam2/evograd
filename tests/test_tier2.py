"""Tier 2 wiring: the declaration surface and provider composition.

Nothing here touches a GPU. What it pins is the part that decides *what* gets
measured — which providers exist, and which arguments become module state —
because getting that wrong produces a benchmark that runs fine and compares
the wrong things.
"""

from __future__ import annotations

import unittest

from evograd.bench.tier2 import (
    ProviderSpec,
    _require_declared_split,
    default_provider_specs,
)
from evograd.opdecl.activity import Active, Inactive, Workload, declare_op
from evograd.ops import get_op


def _decl(**overrides):
    base = dict(
        name="toy",
        forward="evograd.ops.level1.layernorm.forward_ref:layernorm_forward_ref",
        dims=("rows", "hidden"),
        args=(
            Active("x", "[rows, hidden]"),
            Active("weight", "[hidden]"),
            Inactive("eps", default=1e-5),
        ),
        output=Active("y", "[rows, hidden]"),
        forward_semantics="s",
        backward_semantics="s",
        tolerances={"bfloat16": (1e-2, 1e-2)},
        correctness=(Workload(dims={"rows": 4, "hidden": 8}, dtype="bfloat16"),),
        benchmark=(Workload(dims={"rows": 4, "hidden": 8}, dtype="bfloat16"),),
    )
    base.update(overrides)
    return declare_op(**base)


class TestParameterArgs(unittest.TestCase):
    def test_a_parameter_must_be_a_declared_active_arg(self):
        with self.assertRaises(ValueError) as caught:
            _decl(parameter_args=("gamma",))
        self.assertIn("gamma", str(caught.exception))

    def test_an_inactive_arg_cannot_be_a_parameter(self):
        # eps is Inactive: it takes no gradient, so it is not module state the
        # optimizer ever sees. It rides along as a scalar instead.
        with self.assertRaises(ValueError):
            _decl(parameter_args=("eps",))

    def test_naming_every_active_arg_leaves_no_activation(self):
        # An nn.Module takes activations as call arguments. If every Active arg
        # is parameter state there is nothing left to pass to forward().
        with self.assertRaises(ValueError) as caught:
            _decl(parameter_args=("x", "weight"))
        self.assertIn("no activation", str(caught.exception))

    def test_undeclared_is_none_not_empty(self):
        # The two must stay distinguishable: `None` means nobody has said, and
        # `()` means someone said "no parameters". Collapsing them would let
        # layernorm be measured with its weight as an activation.
        self.assertIsNone(_decl().parameter_args)
        self.assertEqual(_decl(parameter_args=()).parameter_args, ())

    def test_the_tier_refuses_an_undeclared_operator(self):
        with self.assertRaises(ValueError) as caught:
            _require_declared_split(_decl())
        self.assertIn("parameter_args", str(caught.exception))

    def test_a_parameter_free_operator_is_accepted(self):
        # geglu, softmax, swiglu: no module state, but the autograd engine
        # still charges graph recording and AccumulateGrad on the activations,
        # which is the whole difference between this tier and tier 1.
        _require_declared_split(_decl(parameter_args=()))


class TestDeclaredOperators(unittest.TestCase):
    def test_layernorm_splits_activations_from_parameters(self):
        op = get_op("layernorm")
        self.assertEqual(op.parameter_args, ("weight", "bias"))
        activations = [
            a.name for a in op.active_args() if a.name not in op.parameter_args
        ]
        self.assertEqual(activations, ["x"])

    def test_rmsnorm_has_one_parameter(self):
        self.assertEqual(get_op("rmsnorm").parameter_args, ("weight",))

    def test_every_operator_declares_the_split(self):
        # A missing declaration is not a latent bug here, it is an operator the
        # tier cannot measure — so the registry is the right place to enforce it.
        from evograd.ops import OPS

        undeclared = sorted(n for n, op in OPS.items() if op.parameter_args is None)
        self.assertEqual(undeclared, [])

    def test_a_parameter_free_operator_keeps_all_its_args_as_activations(self):
        op = get_op("geglu")
        self.assertEqual(op.parameter_args, ())
        self.assertEqual([a.name for a in op.active_args()], ["a", "b"])


class TestProviderComposition(unittest.TestCase):
    def test_the_default_set_is_the_four_the_paper_compares(self):
        specs = default_provider_specs(candidate_module=object(), baseline="liger")
        self.assertEqual(
            [(s.name, s.kind) for s in specs],
            [
                ("eager", "eager"),
                ("torch_compile", "compile"),
                ("liger", "baseline_pair"),
                ("candidate", "candidate_pair"),
            ],
        )

    def test_without_a_candidate_the_set_is_the_reference_line(self):
        specs = default_provider_specs(baseline="liger")
        self.assertNotIn("candidate", [s.name for s in specs])

    def test_compile_can_be_dropped(self):
        specs = default_provider_specs(compile_baseline=False, baseline=None)
        self.assertEqual([s.name for s in specs], ["eager"])

    def test_a_declared_baseline_other_than_liger_is_reachable(self):
        specs = default_provider_specs(baseline="cublas_pair")
        self.assertIn(
            ProviderSpec(name="cublas_pair", kind="baseline_pair", baseline="cublas_pair"),
            specs,
        )


class TestIntegratedUsesTheDeclaration(unittest.TestCase):
    """`bench.integrated` wraps the same operators and must split them the same way.

    It used to infer the split — "the first Active tensor is the input, the rest
    are weights" — which is right for LayerNorm and wrong for nine other
    declarations. Nothing raised: the module built, and the training step it
    measured was not the one the operator describes.
    """

    def test_the_split_matches_parameter_args_for_every_operator(self):
        from evograd.bench.integrated import activation_and_parameter_args
        from evograd.ops import OPS

        for name, op in OPS.items():
            with self.subTest(op=name):
                activations, parameters = activation_and_parameter_args(op)
                self.assertEqual(
                    tuple(a.name for a in parameters), tuple(op.parameter_args)
                )
                self.assertEqual(
                    tuple(a.name for a in activations),
                    tuple(
                        a.name
                        for a in op.active_args()
                        if a.name not in op.parameter_args
                    ),
                )

    def test_a_parameter_free_operator_keeps_both_activations(self):
        from evograd.bench.integrated import activation_and_parameter_args

        activations, parameters = activation_and_parameter_args(get_op("geglu"))
        self.assertEqual([a.name for a in activations], ["a", "b"])
        self.assertEqual(parameters, ())

    def test_the_residual_is_an_activation_not_a_parameter(self):
        from evograd.bench.integrated import activation_and_parameter_args

        activations, parameters = activation_and_parameter_args(
            get_op("fused_add_rms_norm")
        )
        self.assertEqual([a.name for a in activations], ["x", "r"])
        self.assertEqual([a.name for a in parameters], ["weight"])

    def test_an_undeclared_split_raises(self):
        from evograd.bench.integrated import activation_and_parameter_args

        with self.assertRaises(ValueError) as caught:
            activation_and_parameter_args(_decl())
        self.assertIn("parameter_args", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
