"""The OLMoE Level-3 Tier-3 workload, on a 4-layer tiny model (CPU).

Covers: the per-site patch counts (the residual site's ``2 * layers``, through
every decoder layer and ``model.norm``), structural identity against the
unpatched model (loss and every gradient, bitwise), the model reference at every
site through the adapters' kernel ABI (bitwise too: that is what model-scope
gates install as the native), the runner on the workload, the bound pair at the
experts site, and the experts kernel in the identity and the read-back.
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark import TASKS
from evograd.evaluation.tier3.patch import KernelSet, patch
from evograd.evaluation.tier3.workloads.olmoe_1b_7b import sites
from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload

from tests.olmoe_1b_7b._tiny import SEQ, TINY_ARCH


def tiny_workload(dtype: str = "float32", **kwargs) -> OlmoeWorkload:
    return OlmoeWorkload(device="cpu", seq_len=SEQ, batch_size=1, arch_overrides=TINY_ARCH,
                         dtype=dtype, **kwargs)


def _step(workload, kernels):
    model, provenance = workload.build_patched(kernels)
    loss = workload.loss(model, workload.batch_for(seed=0))
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return loss.detach(), grads, workload.last_build


def _reference_kernels(registry):
    """Every site holding its model reference in the declared ABI, as model-scope
    Gate 2 and Gate 3 install the native."""
    from evograd.opdecl.oracle import resolve_runtime_forward
    from evograd.verification.integrations.olmoe_1b_7b.reference import reference_op

    kernels = KernelSet(registry=registry)
    for site in registry.names:
        kernels = patch(kernels, site,
                        resolve_runtime_forward(reference_op(registry.require(site).op)))
    return kernels


class TestCounts(unittest.TestCase):
    def test_the_published_counts(self):
        self.assertEqual(sites.expected_counts(16),
                         {"attention_block": 16, "residual_rmsnorm": 32, "moe_experts": 16})
        self.assertEqual(sites.olmoe_sites().names,
                         ("attention_block", "residual_rmsnorm", "moe_experts"))

    def test_every_site_patched_on_every_layer(self):
        workload = tiny_workload()
        _, _, build = _step(workload, sites.structural_identity_kernels(workload.site_registry))
        self.assertEqual(build.observed(),
                         {"attention_block": 4, "residual_rmsnorm": 8, "moe_experts": 4})
        self.assertEqual(build.count_problems(), [])
        paths = build.provenance.paths
        self.assertEqual(len(paths["moe_experts"]), 4)
        self.assertTrue(all(".mlp.experts" in p for p in paths["moe_experts"]))
        self.assertEqual(paths["attention_block"],
                         tuple(f"model.layers.{i}.self_attn" for i in range(4)))
        self.assertEqual(paths["residual_rmsnorm"],
                         (*(f"model.layers.{i}" for i in range(4)), "model.norm"))
        self.assertIsNotNone(build.carrier)
        self.assertIsNone(build.carrier.pending, "model.norm took the last layer's branch")

    def test_one_site_patched_counts_only_its_adapter(self):
        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload()
        for site, calls in (("moe_experts", 4), ("attention_block", 4), ("residual_rmsnorm", 8)):
            with self.subTest(site=site):
                control = restrict(sites.structural_identity_kernels(workload.site_registry),
                                   (site,))
                _, _, build = _step(workload, control)
                self.assertEqual(build.observed(), {site: calls})
                self.assertEqual(build.count_problems(), [])
                self.assertEqual(build.carrier is None, site != "residual_rmsnorm")

    def test_the_residual_calls_are_the_three_wirings(self):
        """Four layers: 4 post-attention fusions, 3 into the next layer's input norm,
        1 into model.norm -- each handed to a listener under (layer, category)."""
        from collections import Counter

        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload()
        control = restrict(sites.structural_identity_kernels(workload.site_registry),
                           ("residual_rmsnorm",))
        model, _ = workload.build_patched(control)
        seen = []
        sites.set_tap(model, lambda site, inputs, outputs, upstream, emitted, layer:
                      seen.append(layer))
        workload.loss(model, workload.batch_for(seed=0)).backward()
        self.assertEqual(Counter(category for _layer, category in seen),
                         {"post_attention": 4, "mlp_to_next_input": 3, "final_model_norm": 1})
        self.assertEqual(sorted(l for l, c in seen if c == "mlp_to_next_input"), [1, 2, 3])


class TestStructuralIdentity(unittest.TestCase):
    def test_bitwise_the_unpatched_model(self):
        for dtype in ("float32", "bfloat16"):
            for experts in ("grouped_mm", "eager"):
                workload = tiny_workload(dtype, experts_implementation=experts)
                with self.subTest(dtype=dtype, experts=experts):
                    loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
                    loss1, grads1, _ = _step(
                        workload, sites.structural_identity_kernels(workload.site_registry))
                    self.assertTrue(torch.equal(loss0, loss1))
                    self.assertEqual(sorted(grads0), sorted(grads1))
                    for name in grads0:
                        self.assertTrue(torch.equal(grads0[name], grads1[name]), name)

    def test_the_model_reference_at_every_site_is_bitwise_native(self):
        """The model reference in each site's declared ABI -- the attention block's
        and the residual fusions' kernel calls, the carrier's wiring -- reproduces
        the unpatched loss and every gradient bit for bit."""
        for dtype in ("float32", "bfloat16"):
            workload = tiny_workload(dtype)
            with self.subTest(dtype=dtype):
                loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
                loss1, grads1, build = _step(workload, _reference_kernels(workload.site_registry))
                self.assertEqual(build.count_problems(), [])
                self.assertTrue(torch.equal(loss0, loss1))
                for name in grads0:
                    self.assertTrue(torch.equal(grads0[name], grads1[name]), name)


class TestRunnerAndBoundPair(unittest.TestCase):
    def test_the_runner_measures_every_provider(self):
        from evograd.evaluation.tier3.runner import measure_provider

        workload = tiny_workload()
        for name, kernels in (("eager", KernelSet(registry=workload.site_registry)),
                              ("structural_identity",
                               sites.structural_identity_kernels(workload.site_registry))):
            with self.subTest(provider=name):
                row = measure_provider(workload, kernels, provider=name, warmup=0, steps=1,
                                       blocks=1, loss_steps=2, device="cpu")
                self.assertEqual(row["optimizer"]["name"], "AdamW")
                self.assertTrue(all(l == l for l in row["losses"]))

    def test_the_bound_pair_runs_the_experts_site(self):
        """The declared pair through ``bind`` at the experts site: the oracle's
        dense arithmetic in place of the grouped GEMM, within tolerance."""
        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload()
        bound = restrict(sites.bound_pair_identity_kernels(TASKS, None, workload.site_registry),
                         ("moe_experts",))
        loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
        loss1, grads1, build = _step(workload, bound)
        self.assertEqual(build.observed(), {"moe_experts": 4})
        self.assertTrue(torch.allclose(loss0, loss1, atol=1e-4, rtol=1e-4))
        for name in grads0:
            self.assertTrue(torch.allclose(grads0[name], grads1[name], atol=1e-4, rtol=1e-3), name)


class TestWorkloadProtocol(unittest.TestCase):
    def test_config_round_trip_and_canonical_identity(self):
        canonical = OlmoeWorkload()
        self.assertTrue(canonical.spec.is_canonical)
        self.assertEqual((canonical.spec.batch_size, canonical.spec.seq_len), (2, 2048))
        self.assertEqual(canonical.to_config()["experts_implementation"], "grouped_mm")
        rebuilt = OlmoeWorkload.from_config(tiny_workload().to_config())
        self.assertEqual(rebuilt.spec.workload_id, tiny_workload().spec.workload_id)
        self.assertFalse(rebuilt.spec.is_canonical)
        eager = OlmoeWorkload.from_config(tiny_workload(experts_implementation="eager").to_config())
        self.assertEqual(eager.spec.experts_implementation, "eager")
        self.assertNotEqual(eager.spec.workload_id, rebuilt.spec.workload_id)

    def test_describe_and_runtime_report(self):
        workload = tiny_workload()
        described = workload.describe()
        self.assertEqual(described["experts"]["num_experts"], 4)
        self.assertEqual(described["experts_implementation"], "grouped_mm")
        self.assertEqual(described["expected_site_counts"],
                         {"attention_block": 4, "residual_rmsnorm": 8, "moe_experts": 4})
        self.assertIs(described["model_gate"]["implemented"], False)
        model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
        report = workload.runtime_report(model)
        self.assertEqual(report["experts_implementation"], "grouped_mm")
        self.assertEqual(report["experts_implementation_per_module"], ["grouped_mm"])
        self.assertEqual(report["attn_implementation"], "sdpa")
        eager = tiny_workload(experts_implementation="eager")
        model, _ = eager.build_patched(KernelSet(registry=eager.site_registry))
        self.assertEqual(eager.runtime_report(model)["experts_implementation"], "eager")

    def test_whole_model_compile_is_a_baseline_of_the_unpatched_model(self):
        from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import (
            whole_model_compile_kernels)

        workload = tiny_workload()
        model, provenance = workload.build_patched(
            whole_model_compile_kernels(workload.site_registry))
        self.assertEqual(provenance.method, "whole_model_torch_compile")
        self.assertIn("whole_model_torch_compile", workload.runtime_report(model))
        mixed = patch(whole_model_compile_kernels(workload.site_registry), "moe_experts",
                      sites.PRODUCTION["moe_experts"])
        with self.assertRaises(ValueError):
            workload.build_patched(mixed)

if __name__ == "__main__":
    unittest.main()
