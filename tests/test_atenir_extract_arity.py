"""AtenIR autograd extraction must trace every output, not just the first."""
import unittest

import torch
from torch import fx

from evograd.atenir.extract import extract_autograd


def _two_output(x, r, weight):
    summed = x + r
    return summed * weight, summed


def _one_output(x, weight):
    return x * weight


class TestAutogradExtractionArity(unittest.TestCase):
    def test_a_single_output_forward_still_traces(self):
        gm = extract_autograd(f"{__name__}:_one_output",
                              [torch.randn(4, 8), torch.randn(8)])
        self.assertIsInstance(gm, fx.GraphModule)

    def test_a_two_output_forward_traces_both(self):
        gm = extract_autograd(f"{__name__}:_two_output",
                              [torch.randn(4, 8), torch.randn(4, 8), torch.randn(8)])
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        # One upstream gradient per output, then the three forward inputs.
        self.assertEqual(len(placeholders), 5)

    def test_the_second_output_reaches_the_traced_gradients(self):
        # `summed` feeds `out`, so tracing only the first output would still
        # produce a graph -- just the wrong one. Differentiating with a zero
        # gradient on `out` isolates the second output's contribution, which a
        # first-output-only trace cannot represent at all.
        x = torch.randn(4, 8, requires_grad=True)
        r = torch.randn(4, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        out, summed = _two_output(x, r, w)
        grads = torch.autograd.grad(
            (out, summed), [x, r, w],
            grad_outputs=(torch.zeros_like(out), torch.ones_like(summed)),
        )
        self.assertTrue(bool((grads[0] != 0).any()))
        self.assertTrue(bool((grads[2] == 0).all()))


if __name__ == "__main__":
    unittest.main()
