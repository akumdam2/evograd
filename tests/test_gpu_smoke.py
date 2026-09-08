"""CPU tests for the declared-output reporting in ``scripts/gpu_smoke.py``.

The smoke script used to format one result per operator as ``tuple(y.shape)``,
which is only true of a one-output declaration. Every structured-output
operator therefore failed with ``'tuple' object has no attribute 'shape'``
before a single kernel had been exercised -- a defect in the reporting
harness, not in any operator.

These tests pin the normalization the script now performs through the
declaration contract itself. They construct tensors directly rather than
running ``oracle``, so nothing here needs CUDA, triton, or a compute node.
"""

import importlib.util
import unittest
from pathlib import Path

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from evograd.opdecl import Active, Workload, declare_op

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_smoke.py"


def _gpu_smoke():
    """Load the script by path -- ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("gpu_smoke_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single_forward_ref(x):
    return x


def structured_forward_ref(x):
    return x, x, x


def _single_output_op():
    return declare_op(
        name="smoke_single",
        forward="tests.test_gpu_smoke:single_forward_ref",
        dims=("M", "N"),
        args=(Active("x", "[M, N]"),),
        output=Active("y", "[M, N]"),
        forward_semantics="identity",
        backward_semantics="identity",
        correctness=(Workload(dims={"M": 4, "N": 8}, dtype="float32"),),
        tolerances={"float32": (1e-5, 1e-5)},
    )


def _structured_output_op():
    """Three outputs whose names are deliberately not in sorted order."""
    return declare_op(
        name="smoke_structured",
        forward="tests.test_gpu_smoke:structured_forward_ref",
        dims=("M", "N"),
        args=(Active("x", "[M, N]"),),
        output=(
            Active("q", "[M, N]"),
            Active("k", "[M]"),
            Active("v", "[N, M]"),
        ),
        forward_semantics="identity",
        backward_semantics="identity",
        correctness=(Workload(dims={"M": 4, "N": 8}, dtype="float32"),),
        tolerances={"float32": (1e-5, 1e-5)},
    )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class OutputShapeReporting(unittest.TestCase):
    def setUp(self):
        self.smoke = _gpu_smoke()

    def test_single_output_reports_its_declared_name(self):
        op = _single_output_op()
        shapes = self.smoke.output_shapes(op, torch.zeros(4, 8))
        self.assertEqual(shapes, {"y": (4, 8)})

    def test_structured_output_reports_every_declared_output(self):
        op = _structured_output_op()
        result = (torch.zeros(4, 8), torch.zeros(4), torch.zeros(8, 4))
        shapes = self.smoke.output_shapes(op, result)
        self.assertEqual(shapes, {"q": (4, 8), "k": (4,), "v": (8, 4)})

    def test_declared_output_order_is_preserved(self):
        op = _structured_output_op()
        result = (torch.zeros(4, 8), torch.zeros(4), torch.zeros(8, 4))
        shapes = self.smoke.output_shapes(op, result)
        self.assertEqual(list(shapes), list(op.output_names))
        self.assertEqual(list(shapes), ["q", "k", "v"])

    def test_arity_disagreement_is_raised_not_quietly_formatted(self):
        """A contract violation must stay a failure the smoke run reports."""
        op = _structured_output_op()
        with self.assertRaises(ValueError):
            self.smoke.output_shapes(op, torch.zeros(4, 8))
        with self.assertRaises(ValueError):
            self.smoke.output_shapes(op, (torch.zeros(4, 8), torch.zeros(4)))

    def test_registry_structured_operators_report_all_their_outputs(self):
        """The real declarations that exposed the defect, without an oracle run."""
        from evograd.ops import OPS

        structured = {n: op for n, op in OPS.items() if op.is_multi_output}
        self.assertTrue(structured, "registry declares no structured-output operator")
        for name, op in sorted(structured.items()):
            with self.subTest(op=name):
                result = tuple(
                    torch.zeros(2 + i, 3 + i) for i in range(len(op.output_names))
                )
                shapes = self.smoke.output_shapes(op, result)
                self.assertEqual(list(shapes), list(op.output_names))
                self.assertEqual(len(shapes), len(op.output_names))


if __name__ == "__main__":
    unittest.main()
