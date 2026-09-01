"""Regression tests for the two Pipeline B / FLCE defects repaired in this study.

Both were silent: the first produced a seed that imported and read plausibly but
raised on its first call, the second was a contract that three places disagreed
about without anything checking.
"""

from __future__ import annotations

import unittest

from evograd.ops import get_op


class TestGenericFallbackArity(unittest.TestCase):
    """A serialized aten fallback must be emitted with its schema's arity.

    `_ordered_arg_exprs` renders every positional argument including scalars;
    `_generic_fallback_call` used to re-interleave the scalars on top of that,
    emitting each one twice. For `aten._log_softmax.default` that produced
    `(mm, 1, False, 1, False)` against a three-argument schema, so the generated
    seed raised on its first call.
    """

    def _emit(self, node: dict) -> str:
        from evograd.pipelines.b_dispatch.program_codegen import (
            _ProgramBuilder,
            _ordered_arg_exprs,
        )

        builder = _ProgramBuilder()
        arg_exprs = _ordered_arg_exprs(node["args_ordered"])
        return builder._generic_fallback_call(node["target"], arg_exprs, node)

    def test_scalar_arguments_are_emitted_exactly_once(self):
        node = {
            "target": "aten._log_softmax.default",
            "args_ordered": [
                {"kind": "node", "name": "mm"},
                {"kind": "scalar", "value": 1},
                {"kind": "scalar", "value": False},
            ],
        }
        call = self._emit(node)
        self.assertEqual(call.count("False"), 1, call)
        inner = call[call.index("(", call.index(")(")) + 1 : call.rindex(")")]
        self.assertEqual(len(inner.split(", ")), 3, call)

    def test_emitted_arity_matches_the_aten_schema(self):
        import torch

        node = {
            "target": "aten._log_softmax.default",
            "args_ordered": [
                {"kind": "node", "name": "mm"},
                {"kind": "scalar", "value": 1},
                {"kind": "scalar", "value": False},
            ],
        }
        call = self._emit(node)
        inner = call[call.index("(", call.index(")(")) + 1 : call.rindex(")")]
        emitted = len(inner.split(", "))
        schema = torch.ops.aten._log_softmax.default._schema
        self.assertEqual(emitted, len(schema.arguments), f"{call} vs {schema}")

    def test_multiple_scalars_and_tensors_keep_their_slots(self):
        node = {
            "target": "aten.some_op.default",
            "args_ordered": [
                {"kind": "scalar", "value": 0},
                {"kind": "node", "name": "a"},
                {"kind": "scalar", "value": True},
                {"kind": "node", "name": "b"},
            ],
        }
        call = self._emit(node)
        inner = call[call.index("(", call.index(")(")) + 1 : call.rindex(")")]
        self.assertEqual(inner.split(", "), ["0", "t_a", "True", "t_b"], call)


class TestLossDtypeContract(unittest.TestCase):
    """The loss is float32 everywhere: reference, declaration text, and dloss.

    The three used to disagree — the reference returned float32, the declaration
    said the scalar was "cast to x's dtype", and the input builder created dloss
    in the workload dtype.
    """

    def setUp(self):
        self.op = get_op("fused_linear_cross_entropy")

    @staticmethod
    def _inputs(op, workload):
        from evograd.opdecl.inputs import make_case_inputs

        return make_case_inputs(op, workload, device="cpu")

    def test_declaration_does_not_claim_the_loss_is_cast_to_x_dtype(self):
        text = self.op.extra_constraints
        self.assertIn("float32 scalar", text)
        # The old text asserted the cast; the new one denies it. Match on the
        # denial rather than on absence of the phrase, which the denial contains.
        self.assertIn("NOT cast to x's dtype", text)
        self.assertNotIn("float32-accumulated scalar cast to", text)

    def test_dloss_is_float32_for_every_declared_workload(self):
        import torch

        for workload in self.op.correctness:
            values = self._inputs(self.op, workload)
            self.assertEqual(
                values[self.op.upstream_grad_name].dtype,
                torch.float32,
                f"{workload.dims} {workload.dtype}",
            )

    def test_reference_returns_float32_regardless_of_input_dtype(self):
        import torch

        from evograd.opdecl.oracle import resolve_forward

        forward = resolve_forward(self.op)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(4, 8, dtype=dtype)
            weight = torch.randn(6, 8, dtype=dtype)
            target = torch.randint(0, 6, (4,), dtype=torch.int64)
            self.assertEqual(forward(x, weight, target).dtype, torch.float32, str(dtype))


if __name__ == "__main__":
    unittest.main()
