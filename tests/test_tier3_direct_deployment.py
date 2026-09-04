"""A candidate that ships its own autograd Function is called directly.

The binder exists so an arbitrary generated pair can be trained without a
hand-written wrapper. A candidate that already wired its own
``torch.autograd.Function`` needs none of that, and routing it through the
binder anyway measures EvoGrad's generic argument routing rather than the
kernel being deployed.
"""

from __future__ import annotations

import types
import unittest

import torch

from evograd.bench.tier3_patch import (
    DEPLOYMENT_ENTRY_ATTR,
    deployment_entry,
    patched_kernels,
)


def _build_artifact(op, *, source_only=False):
    """A complete artifact: stub pair + the real generated deployment layer."""
    from evograd.pipelines.shared.artifact import render_deployment_layer

    args = ", ".join(a.name for a in op.args)
    outs = ", ".join(op.output_names)
    ret = f"({outs})" if len(op.output_names) > 1 else outs
    grads = ", ".join(a.grad_name for a in op.active_args())
    stub = (
        "import torch\n\n"
        f"def {op.forward_fn_name}({args}):\n"
        f"    {outs} = " + ", ".join(
            ["x + 0"] * len(op.output_names)) + "\n"
        f"    return {ret}, (x,)\n\n"
        f"def {op.backward_fn_name}(grads, saved, "
        + ", ".join(f"{c.name}={c.default!r}" for c in op.scalar_inactive_args())
        + "):\n"
        f"    (x,) = saved\n"
        f"    return " + ", ".join(
            ["torch.zeros_like(x)"] * len(op.grad_names())) + "\n"
    )
    source = stub + render_deployment_layer(op)
    if source_only:
        return source
    module = types.ModuleType("artifact_under_test")
    exec(compile(source, "<artifact>", "exec"), module.__dict__)
    return module


class TestDeploymentEntryDetection(unittest.TestCase):
    def test_a_named_callable_is_found(self):
        module = types.SimpleNamespace(DEPLOYMENT_ENTRY="run", run=lambda *a: None)
        self.assertIs(deployment_entry(module), module.run)

    def test_a_legacy_pair_module_has_none(self):
        self.assertIsNone(deployment_entry(types.SimpleNamespace()))

    def test_a_name_that_resolves_to_nothing_is_not_an_entry(self):
        # A typo must fall back to the binder, not crash the patcher.
        self.assertIsNone(
            deployment_entry(types.SimpleNamespace(DEPLOYMENT_ENTRY="absent"))
        )

    def test_a_non_string_entry_is_ignored(self):
        module = types.SimpleNamespace(**{DEPLOYMENT_ENTRY_ATTR: lambda: None})
        self.assertIsNone(deployment_entry(module))


class TestPatchedKernelsRouting(unittest.TestCase):
    def _registry_and_ops(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import qwen3_sites
        from evograd.ops import OPS

        return qwen3_sites(), OPS

    def test_a_direct_candidate_is_patched_as_itself(self):
        registry, ops = self._registry_and_ops()
        module = _build_artifact(ops["fused_add_rms_norm"])
        kernels = patched_kernels({"residual_rmsnorm": module}, ops,
                                  registry=registry, origin="candidate")
        entry = getattr(module, module.DEPLOYMENT_ENTRY)
        self.assertIs(kernels.kernel_for("residual_rmsnorm"), entry)
        self.assertEqual([s.origin for s in kernels.sources],
                         ["candidate:direct_deployment"])
        self.assertEqual(list(kernels.patched), ["residual_rmsnorm"])

    def test_a_malformed_direct_artifact_is_rejected_not_demoted(self):
        from evograd.pipelines.shared.artifact import ArtifactError

        registry, ops = self._registry_and_ops()
        # Declares an entry, exports nothing else: silently falling back to the
        # binder here would report a path the candidate never ran.
        broken = types.SimpleNamespace(DEPLOYMENT_ENTRY="go", go=lambda *a: None)
        with self.assertRaises(ArtifactError):
            patched_kernels({"residual_rmsnorm": broken}, ops,
                            registry=registry, origin="candidate")

    def test_a_legacy_pair_candidate_still_goes_through_the_binder(self):
        registry, ops = self._registry_and_ops()

        def fused_add_rms_norm_forward_with_saved(x, r, weight, eps=1e-6):
            summed = x + r
            return (summed * weight, summed), (summed, weight)

        def fused_add_rms_norm_backward_from_saved(grads, saved, eps=1e-6):
            dout, dsummed = grads
            summed, weight = saved
            return dout, dout, (dout * summed).sum(0)

        module = types.SimpleNamespace(
            fused_add_rms_norm_forward_with_saved=fused_add_rms_norm_forward_with_saved,
            fused_add_rms_norm_backward_from_saved=fused_add_rms_norm_backward_from_saved,
        )
        kernels = patched_kernels({"residual_rmsnorm": module}, ops,
                                  registry=registry, origin="candidate")
        self.assertIsNot(kernels.kernel_for("residual_rmsnorm"),
                         fused_add_rms_norm_forward_with_saved)
        self.assertEqual([s.origin for s in kernels.sources],
                         ["candidate:legacy_bind"])


class TestStructuredStaticBackward(unittest.TestCase):
    """The fixed contract the deployment template pins, tested on CPU."""

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, residual, weight, eps):
            summed = x + residual
            ctx.save_for_backward(summed, weight)
            ctx.eps = eps
            return summed * weight, summed

        @staticmethod
        def backward(ctx, dout, dsummed):
            summed, weight = ctx.saved_tensors
            dx = dout * weight + dsummed
            return dx, dx, (dout * summed).sum(0), None

    def test_both_outputs_and_both_upstream_gradients_participate(self):
        x = torch.randn(4, 8, requires_grad=True)
        r = torch.randn(4, 8, requires_grad=True)
        w = torch.randn(8, requires_grad=True)
        out, summed = self._Fn.apply(x, r, w, 1e-6)
        dout, dsummed = torch.randn_like(out), torch.randn_like(summed)
        torch.autograd.backward((out, summed), (dout, dsummed))
        # dsummed reaches dx only if the second upstream gradient is consumed.
        self.assertTrue(torch.allclose(x.grad, dout * w + dsummed))
        self.assertTrue(torch.equal(x.grad, r.grad))

    def test_the_scalar_argument_returns_no_gradient(self):
        x = torch.randn(2, 4, requires_grad=True)
        r = torch.randn(2, 4, requires_grad=True)
        w = torch.randn(4, requires_grad=True)
        out, summed = self._Fn.apply(x, r, w, 1e-6)
        grads = torch.autograd.grad(
            (out, summed), [x, r, w],
            (torch.ones_like(out), torch.ones_like(summed)),
        )
        self.assertEqual(len(grads), 3)          # eps contributes none
        self.assertEqual(grads[2].shape, w.shape)

    def test_a_second_call_does_not_reuse_the_first_saved_state(self):
        x = torch.randn(3, 5, requires_grad=True)
        r = torch.randn(3, 5, requires_grad=True)
        w = torch.randn(5, requires_grad=True)
        first = self._Fn.apply(x, r, w, 1e-6)
        second = self._Fn.apply(x * 2, r, w, 1e-6)
        self.assertFalse(torch.equal(first[1], second[1]))


if __name__ == "__main__":
    unittest.main()
