"""Model-reference contracts preserve the numerical computation used by the gates."""

from dataclasses import replace
import unittest

import torch

from evograd.opdecl import Active, Workload, declare_op
from evograd.opdecl.oracle import oracle, reference_identity, resolve_runtime_forward
from tests.verification import _fixtures as F


CONTRACT = "evograd-model-reference/fixture/1"


def source_identity():
    return {"contract": CONTRACT, "revision": "fixture-source-v1"}


def rounded_forward(x):
    return (x + 1000.0) - 1000.0


def model_op(**changes):
    args = dict(
        name="vt_model_reference", forward=f"{__name__}:rounded_forward",
        reference_contract=CONTRACT, dims=("N",),
        args=(Active("x", "[N]"),), output=Active("y", "[N]"),
        forward_semantics="two operations at the operand dtype",
        backward_semantics="dx = dy",
        correctness=(Workload({"N": 4}, "bfloat16"),), parameter_args=(),
        tolerances={"bfloat16": (1e-5, 1.6e-2)},
    )
    args.update(changes)
    return declare_op(**args)


class TestReferenceContract(unittest.TestCase):
    def test_declaration_accepts_one_reference_and_records_its_source(self):
        op = model_op()
        self.assertIs(resolve_runtime_forward(op), rounded_forward)
        identity = reference_identity(op)
        self.assertEqual(identity["kind"], "model_reference")
        self.assertEqual(identity["reference_contract"], CONTRACT)
        self.assertEqual(identity["source"], source_identity())
        self.assertIs(resolve_runtime_forward(model_op(runtime_forward=op.forward)), rounded_forward)

    def test_model_declaration_rejects_alternative_semantics(self):
        for changes in (
            {"reference_dtype": "float32"},
            {"runtime_forward": F.SCALE_SHIFT.forward},
            {"reference_contract": " "},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model_op(**changes)

    def test_oracle_does_not_promote_model_operands(self):
        x = torch.ones(4, dtype=torch.bfloat16)
        cotangent = torch.ones_like(x)
        output, grads = oracle(model_op(), {"x": x}, dout=cotangent)
        self.assertTrue(torch.equal(output, torch.zeros_like(x)))
        self.assertTrue(torch.equal(grads["dx"], cotangent))
        generic = replace(model_op(), reference_contract=None, reference_dtype="float32")
        widened, _ = oracle(generic, {"x": x}, dout=cotangent)
        self.assertTrue(torch.equal(widened, x))

    def test_oracle_rejects_promotion_even_if_validation_was_bypassed(self):
        op = replace(model_op(), reference_dtype="float32")
        with self.assertRaisesRegex(ValueError, "would change its numerical target"):
            oracle(op, {"x": torch.ones(4, dtype=torch.bfloat16)})

    def test_generic_declaration_keeps_its_reference_contract(self):
        op = F.SCALE_SHIFT
        self.assertIsNone(op.reference_contract)
        self.assertEqual(reference_identity(op)["kind"], "declared_forward_oracle")
        self.assertIs(resolve_runtime_forward(op), F.scale_shift_runtime)
