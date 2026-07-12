"""CPU tests for the torch-facing opdecl stack (oracle, bind, verify).

Self-skips when torch is unavailable (dev boxes); runs on any node with torch
installed — no GPU or triton needed, everything here is pure torch on CPU.
Kernel-level validation still requires scripts/gpu_smoke.py on a CUDA node.
"""

import types
import unittest

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from evograd.opdecl import Active, Inactive, Workload, declare_op


def toy_op():
    """y = (x * scale_vec) @ w, with an inactive additive mask and scalar eps."""
    return declare_op(
        name="toy",
        forward="tests.test_opdecl_torch:toy_forward_ref",
        dims=("M", "N"),
        args=(
            Active("x", "[M, N]"),
            Active("w", "[N, N]"),
            Inactive("mask", "[M, N]"),
            Inactive("eps", default=1e-5),
        ),
        output=Active("y", "[M, N]"),
        forward_semantics="toy",
        backward_semantics="toy",
        correctness=(Workload(dims={"M": 4, "N": 8}, dtype="float32"),),
        tolerances={"float32": (1e-5, 1e-5)},
    )


def toy_forward_ref(x, w, mask, eps=1e-5):
    return (x + mask) @ w + eps


def _toy_seed_module(correct=True, forward_offset=0.0, save_scalar=False):
    """A pure-torch seed implementing the autograd-pair contract by hand."""

    def forward_with_saved(x, w, mask, eps=1e-5):
        y = (x + mask) @ w + eps + forward_offset
        saved = (x, w, mask, x.shape[0]) if save_scalar else (x, w, mask)
        return y, saved

    def backward_from_saved(dy, saved_tensors, eps=1e-5):
        x, w, mask = saved_tensors[:3]
        dx = dy @ w.T
        dw = (x + mask).T @ dy
        if not correct:
            dw = dw * 2.0
        return dx, dw

    module = types.ModuleType("toy_seed")
    module.forward_with_saved = forward_with_saved
    module.backward_from_saved = backward_from_saved
    return module


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestOracle(unittest.TestCase):
    def test_oracle_matches_closed_form(self):
        from evograd.opdecl import make_case_inputs, oracle

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        y, grads = oracle(op, inputs)

        x, w, mask, dy = inputs["x"], inputs["w"], inputs["mask"], inputs["dy"]
        torch.testing.assert_close(y, (x + mask) @ w + 1e-5)
        self.assertEqual(list(grads), ["dx", "dw"])  # Active only, mask excluded
        torch.testing.assert_close(grads["dx"], dy @ w.T)
        torch.testing.assert_close(grads["dw"], (x + mask).T @ dy)

    def test_oracle_never_grads_inactive(self):
        from evograd.opdecl import make_case_inputs, oracle

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        _, grads = oracle(op, inputs)
        self.assertNotIn("dmask", grads)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestBind(unittest.TestCase):
    def test_bound_function_matches_plain_autograd(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module())

        x = inputs["x"].detach().clone().requires_grad_(True)
        w = inputs["w"].detach().clone().requires_grad_(True)
        y = fn(x, w, inputs["mask"])
        y.backward(inputs["dy"])

        x2 = inputs["x"].detach().clone().requires_grad_(True)
        w2 = inputs["w"].detach().clone().requires_grad_(True)
        toy_forward_ref(x2, w2, inputs["mask"]).backward(inputs["dy"])

        torch.testing.assert_close(x.grad, x2.grad)
        torch.testing.assert_close(w.grad, w2.grad)

    def test_inactive_slot_gets_none(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module())

        mask = inputs["mask"].detach().clone().requires_grad_(True)
        y = fn(inputs["x"], inputs["w"], mask)
        y.backward(inputs["dy"])
        self.assertIsNone(mask.grad)  # Inactive: harness emitted None for its slot

    def test_plain_scalar_saved_state_round_trips(self):
        from evograd.opdecl import bind, make_case_inputs

        op = toy_op()
        inputs = make_case_inputs(op, op.correctness[0], device="cpu")
        fn = bind(op, _toy_seed_module(save_scalar=True))
        x = inputs["x"].detach().clone().requires_grad_(True)
        w = inputs["w"].detach().clone().requires_grad_(True)
        fn(x, w, inputs["mask"]).backward(inputs["dy"])
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestVerify(unittest.TestCase):
    def test_correct_seed_passes(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(), device="cpu")
        self.assertTrue(report.ok, report.to_dict())

    def test_wrong_gradient_fails_named_check(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(correct=False), device="cpu")
        self.assertFalse(report.ok)
        failing = [c.name for case in report.cases for c in case.checks if not c.ok]
        self.assertEqual(failing, ["dw"])  # exactly the sabotaged gradient

    def test_wrong_forward_fails_even_with_correct_backward(self):
        from evograd.opdecl import verify

        report = verify(toy_op(), _toy_seed_module(forward_offset=1.0), device="cpu")
        self.assertFalse(report.ok)
        failing = [c.name for case in report.cases for c in case.checks if not c.ok]
        self.assertEqual(failing, ["y"])


if __name__ == "__main__":
    unittest.main()
