"""Boundary hooks must record external edges, not internal multi-output paths."""

import unittest

import torch
from torch import nn

from evograd.verification import gate2
from evograd.verification.common.execution import Invocation, Sample, SiteHooks, candidate, capture_grad
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.runner import run_callable


def connected_outputs(x):
    z = x * 2
    return z.square(), z


class Consumer(nn.Module):
    def __init__(self, sites):
        super().__init__()
        self.site = sites["s"]

    def forward(self, x):
        y, z = self.site(x)
        return y + 3 * z


class TestExternalCotangents(unittest.TestCase):
    def test_none_gradient_hook_does_not_record_a_tensor(self):
        store = {}
        self.assertIsNone(capture_grad(store, "unused")(None))
        self.assertEqual(store, {})

    def test_internal_path_is_not_counted_as_an_external_seed(self):
        x = torch.tensor([1., 2., 3.], requires_grad=True)
        hooks = SiteHooks(record=True)
        y, z = hooks.wrap("s", connected_outputs)(x)
        (y + 3 * z).sum().backward()
        call = hooks.calls[0]
        self.assertTrue(torch.equal(call.upstream["out[0]"], torch.ones_like(x)))
        self.assertTrue(torch.equal(call.upstream["out[1]"], 3 * torch.ones_like(x)))
        self.assertTrue(torch.equal(x.grad, 8 * x.detach() + 6))
        replay = run_callable(connected_outputs, call.inputs, call.grad_paths, call.upstream)
        self.assertTrue(torch.equal(call.emitted["args[0]"], replay["grad_in:args[0]"]))

    def test_unused_output_is_zero_not_internal_gradient(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        hooks = SiteHooks(record=True)
        y, unused = hooks.wrap("s", connected_outputs)(x)
        y.sum().backward()
        call = hooks.calls[0]
        self.assertNotIn("out[1]", call.upstream)
        replay = run_callable(connected_outputs, call.inputs, call.grad_paths, call.upstream)
        self.assertTrue(torch.equal(replay["grad_in:args[0]"], x.grad))

    def test_alias_return_keeps_one_external_seed_and_storage(self):
        def site(x):
            z = x * 2
            return z.square(), z, z

        x = torch.randn(2, 3).t().detach().requires_grad_(True)
        hooks = SiteHooks(record=True)
        y, z, alias = hooks.wrap("s", site)(x)
        self.assertIs(z, alias)
        (y + 3 * z + 5 * alias).sum().backward()
        call = hooks.calls[0]
        self.assertTrue(torch.equal(call.upstream["out[1]"], torch.full_like(x, 8.)))
        self.assertNotIn("out[2]", call.upstream)
        self.assertEqual(z.stride(), call.outputs[1].stride())
        replay = run_callable(site, call.inputs, call.grad_paths, call.upstream)
        self.assertTrue(torch.equal(replay["grad_in:args[0]"], x.grad))

    def test_eager_multioutput_site_passes_real_calibration_and_matched_check(self):
        scope = ModuleScope(
            scope="model", name="external-cotangent-test", factory=Consumer,
            native_sites={"s": connected_outputs},
            make_invocation=lambda s, d: Invocation(args=(torch.randn(4),),
                                                    differentiable_inputs={"x": "args[0]"}),
            select_outputs=lambda y, i, s: {"y": y}, output_names=("y",))
        fixed = [Sample("a", 1), Sample("b", 2)]
        cal = gate2.calibrate(scope, patch_set=("s",), samples=fixed,
                             boundaries=True, compile_backend="aot_eager")
        self.assertIsNotNone(cal.policy, cal.to_dict())
        report = gate2.verify_integration(scope, candidate("eager", {"s": connected_outputs}),
                                          policy=cal.policy, samples=fixed)
        self.assertEqual(report.status, "passed", report.to_dict())
        live = [r for c in report.cases for r in c.results if "grad_in" in r.name and r.name.endswith("~live")]
        self.assertTrue(live)
        self.assertTrue(all(r.status == "passed" for r in live))

    def test_tiny_qwen_final_residual_unused_output_calibrates(self):
        from tests.verification.test_bridges import ARCH, bound_eager
        from evograd.verification.gate2.adapters.model import WorkloadModelScope
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

        w = Qwen3Workload(device="cpu", dtype="float32", batch_size=1, seq_len=16,
                         arch_overrides={**ARCH, "num_key_value_heads": 2})
        scope = WorkloadModelScope(w)
        cal = gate2.calibrate(scope, patch_set=("residual_rmsnorm",), samples=scope.samples([1, 2]),
                             compile_backend="aot_eager", boundaries=True)
        self.assertIsNotNone(cal.policy, cal.to_dict())
        report = gate2.verify_integration(
            scope, candidate("bound", {"residual_rmsnorm": bound_eager("fused_add_rms_norm")}),
            policy=cal.policy, samples=scope.samples([1, 2]))
        self.assertEqual(report.status, "passed", report.to_dict())
