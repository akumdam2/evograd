"""The built-in torch.compile performance baseline.

``pytorch_autograd`` measures eager PyTorch, which a Pipeline D seed beats by
construction — it is captured from the same Inductor that ``torch.compile``
runs. These tests pin the parts that can be checked without a GPU: name
resolution, the hook's metadata, and (the substantive one) that the compiled
forward is timed in setup while only ``autograd.grad`` sits in the timed region,
so the baseline is measured the same way the candidate is.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from evograd.opdecl.baselines import (
    available_baselines,
    baseline_hook,
    resolve_performance_baseline,
)
from evograd.opdecl.compiled import BUILTIN_MODES, _leaf_args, make_compiled_baseline
from evograd.ops import get_op


class _FakeTensor:
    """Enough tensor for leaf construction: identity detach, counting clone."""

    def __init__(self, tag, requires_grad=False):
        self.tag = tag
        self.requires_grad = requires_grad
        self.clones = 0

    def detach(self):
        return self

    def clone(self):
        copy = _FakeTensor(self.tag)
        copy.clones = self.clones + 1
        return copy

    def requires_grad_(self, flag=True):
        self.requires_grad = flag
        return self


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.op = get_op("layernorm")

    def test_builtin_names_resolve_without_declaration_support(self):
        self.assertNotIn("torch_compile", self.op.performance_baselines)
        for name in BUILTIN_MODES:
            self.assertEqual(resolve_performance_baseline(self.op, name), name)

    def test_auto_never_picks_torch_compile(self):
        # Compiling costs minutes; it is a deliberate comparison, not a default.
        self.assertNotIn(
            resolve_performance_baseline(self.op, "auto"), set(BUILTIN_MODES)
        )

    def test_unknown_name_lists_the_builtins(self):
        with self.assertRaises(KeyError) as caught:
            resolve_performance_baseline(self.op, "nope")
        self.assertIn("torch_compile", str(caught.exception))
        self.assertIn("torch_compile", available_baselines(self.op))

    def test_hook_carries_mode_and_a_verification_entry_point(self):
        default = baseline_hook(self.op, "torch_compile")
        tuned = baseline_hook(self.op, "torch_compile_max_autotune")
        self.assertIsNone(default.compiled_mode)
        self.assertEqual(tuned.compiled_mode, "max-autotune")
        self.assertFalse(default.compiled_dynamic)
        # reference_run is what verify_performance_baseline checks against the
        # eager oracle before any timing is trusted.
        self.assertTrue(callable(default.reference_run))
        self.assertTrue(default.available())


class TestLeafArgs(unittest.TestCase):
    def test_active_args_become_fresh_leaves_inactive_pass_through(self):
        op = get_op("layernorm")
        inputs = {arg.name: _FakeTensor(arg.name) for arg in op.args}
        positional, leaves = _leaf_args(op, inputs)

        self.assertEqual(len(positional), len(op.args))
        self.assertEqual([name for name, _ in leaves], list(op.grad_names()))
        for name, leaf in leaves:
            self.assertTrue(leaf.requires_grad, name)
            self.assertEqual(leaf.clones, 1, name)  # fresh clone, not the input
        for arg, value in zip(op.args, positional):
            if not any(leaf is value for _, leaf in leaves):
                self.assertFalse(getattr(value, "requires_grad", False), arg.name)

    def test_each_call_gets_its_own_leaves(self):
        op = get_op("layernorm")
        inputs = {arg.name: _FakeTensor(arg.name) for arg in op.args}
        _p1, first = _leaf_args(op, inputs)
        _p2, second = _leaf_args(op, inputs)
        # Reusing a leaf would accumulate into an already-populated .grad.
        for (_, a), (_, b) in zip(first, second):
            self.assertIsNot(a, b)


class TestTimingSplit(unittest.TestCase):
    def test_forward_runs_in_setup_only_backward_is_timed(self):
        op = get_op("layernorm")
        inputs = {arg.name: _FakeTensor(arg.name) for arg in op.args}
        inputs[op.upstream_grad_name] = _FakeTensor("dout")
        calls = []

        def fake_compiled(*args):
            calls.append("forward")
            return _FakeTensor("y")

        def fake_grad(y, leaves, grad_outputs=None):
            calls.append("backward")
            return tuple(_FakeTensor("g") for _ in leaves)

        fake_torch = SimpleNamespace(autograd=SimpleNamespace(grad=fake_grad))

        timed_region = {}

        def median_ms_timed_region(setup, timed, warmup, reps):
            state = setup()
            del calls[:]  # only what the timed callable does counts
            timed(state)
            timed_region["calls"] = list(calls)
            return 1.0

        def median_ms(fn, warmup, reps):
            del calls[:]
            fn()
            timed_region["full_step"] = list(calls)
            return 2.0

        hook = make_compiled_baseline(op)
        with mock.patch(
            "evograd.opdecl.compiled.compile_forward", return_value=fake_compiled
        ):
            backward_ms, full_ms = hook(
                fake_torch,
                inputs,
                inputs[op.upstream_grad_name],
                3,
                5,
                median_ms,
                median_ms_timed_region,
            )

        self.assertEqual((backward_ms, full_ms), (1.0, 2.0))
        # The timed backward region must not contain the forward — that is the
        # asymmetry that makes speedup_vs_baseline_backward unreadable.
        self.assertEqual(timed_region["calls"], ["backward"])
        self.assertEqual(timed_region["full_step"], ["forward", "backward"])

    def test_mode_and_dynamic_reach_torch_compile(self):
        import torch

        from evograd.opdecl.compiled import compile_forward

        op = get_op("layernorm")
        with mock.patch.object(torch, "compile") as compile_call:
            compile_forward(op, "max-autotune", False)
        _args, kwargs = compile_call.call_args
        self.assertEqual(kwargs["mode"], "max-autotune")
        self.assertIs(kwargs["dynamic"], False)


class TestFairProvider(unittest.TestCase):
    def test_compiled_pair_matches_declared_gradient_order(self):
        import torch

        from evograd.bench.fair import torch_compile_provider

        op = get_op("layernorm")

        def fake_compiled(x, weight, bias, eps):
            return x * weight + bias + eps

        with mock.patch(
            "evograd.opdecl.compiled.compile_forward", return_value=fake_compiled
        ):
            provider = torch_compile_provider(
                op, name="torch_compile", mode=None, dynamic=False
            )

        values = {
            "x": torch.randn(3, 4),
            "weight": torch.randn(4),
            "bias": torch.randn(4),
            "eps": 1e-5,
        }
        dout = torch.randn(3, 4)
        output, saved = provider.forward(values)
        grads = provider.backward(dout, saved, values)

        self.assertEqual(provider.name, "torch_compile")
        self.assertEqual(provider.adapter_kind, "torch_compile_default_static")
        self.assertEqual(len(grads), 3)
        torch.testing.assert_close(grads[0], dout * values["weight"])
        torch.testing.assert_close(grads[1], (dout * values["x"]).sum(0))
        torch.testing.assert_close(grads[2], dout.sum(0))

    def test_exact_workload_provider_gate(self):
        import torch

        from evograd.bench.fair import PairProvider, verify_pair_provider
        from evograd.opdecl.activity import Workload

        op = get_op("layernorm")

        def forward(values):
            x = values["x"].detach().requires_grad_(True)
            weight = values["weight"].detach().requires_grad_(True)
            bias = values["bias"].detach().requires_grad_(True)
            values.update(x=x, weight=weight, bias=bias)
            output = torch.nn.functional.layer_norm(
                x, (x.shape[-1],), weight, bias, values["eps"]
            )
            return output, (output,)

        def backward(dout, saved, values):
            return torch.autograd.grad(
                saved[0],
                (values["x"], values["weight"], values["bias"]),
                dout,
            )

        provider = PairProvider("compiled", forward, backward, "hash", "test")
        workloads = (Workload({"rows": 3, "hidden": 7}, "float32"),)
        report = verify_pair_provider(op, provider, workloads, device="cpu")
        self.assertTrue(report.ok, report.to_dict())
        self.assertEqual(report.cases[0].dims, {"rows": 3, "hidden": 7})


if __name__ == "__main__":
    unittest.main()
