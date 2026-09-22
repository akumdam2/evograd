"""AtenIR autograd extraction must trace every output, not just the first."""
import unittest

import torch
from torch import fx

from evograd.atenir.extract import _serialise, extract_autograd, extract_named_op


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


class TestJointForwardOutputs(unittest.TestCase):
    """The trace returns the forward outputs as well as the gradients.

    Without them the forward's nodes exist only as backward intermediates and
    no consumer can emit a forward implementation -- which is why Pipeline B's
    seed called the eager reference instead.
    """

    def test_a_single_output_forward_is_recorded_and_returned_first(self):
        gm = extract_autograd(f"{__name__}:_one_output",
                              [torch.randn(4, 8), torch.randn(8)])
        graph = _serialise(gm)
        self.assertEqual(graph["num_forward_outputs"], 1)
        self.assertEqual(graph["num_grad_placeholders"], 1)
        out_args = graph["nodes"][-1]["args"][0]
        # one forward output, then one gradient per differentiable input
        self.assertEqual(len(out_args), 1 + 2)

    def test_a_two_output_forward_records_both(self):
        gm = extract_autograd(f"{__name__}:_two_output",
                              [torch.randn(4, 8), torch.randn(4, 8), torch.randn(8)])
        graph = _serialise(gm)
        self.assertEqual(graph["num_forward_outputs"], 2)
        self.assertEqual(graph["num_grad_placeholders"], 2)
        self.assertEqual(len(graph["nodes"][-1]["args"][0]), 2 + 3)

    def test_a_named_op_graph_carries_no_forward_outputs(self):
        """Back-compat: the op mode is gradients-only and must stay unmarked.

        Every previously extracted JSON is in this shape, and consumers read a
        missing key as zero.
        """
        gm = extract_named_op(
            "aten._softmax_backward_data",
            [torch.randn(4, 8), torch.softmax(torch.randn(4, 8), -1), -1, torch.float32],
        )
        graph = _serialise(gm)
        self.assertNotIn("num_forward_outputs", graph)
