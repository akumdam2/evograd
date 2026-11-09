"""The Qwen3.5 Level-3 Tier-3 workload, on an 8-layer tiny model (CPU).

Eight layers in the published pattern -- six Gated DeltaNet, two full attention
-- so the per-site patch counts are exactly a quarter of Qwen3.5-4B's
24 / 8 / 32 / 32. Covers: the counts, structural identity against the unpatched
model (loss and every gradient, bitwise), and the runner on the workload.
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
from evograd.evaluation.tier3.patch import KernelSet, patch
from evograd.evaluation.tier3.workloads.qwen3_5_4b import sites
from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

from tests.qwen3_5_4b._tiny import SEQ, TINY_OVERRIDES

ARCH8 = {**with_layers(QWEN3_5_4B, 8), **TINY_OVERRIDES}


def tiny_workload(dtype: str = "float32") -> Qwen3_5Workload:
    return Qwen3_5Workload(device="cpu", seq_len=SEQ, batch_size=1, arch_overrides=ARCH8,
                           dtype=dtype)


def _step(workload, kernels):
    model, provenance = workload.build_patched(kernels)
    loss = workload.loss(model, workload.batch_for(seed=0))
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return loss.detach(), grads, workload.last_build


class TestCounts(unittest.TestCase):
    def test_the_published_counts(self):
        from evograd.benchmark.topdown.qwen3_5_4b.architecture import layer_types

        self.assertEqual(sites.expected_counts(layer_types()),
                         {"gdn_in_proj": 24, "gated_delta_rule": 24, "gdn_norm_out": 24,
                          "qkv_norm_partial_rope": 8, "gated_attention": 8,
                          "residual_rmsnorm": 32, "swiglu_mlp": 32})

    def test_every_site_patched_on_every_layer_of_its_type(self):
        workload = tiny_workload()
        _, _, build = _step(workload, sites.structural_identity_kernels(workload.site_registry))
        observed = build.observed()
        self.assertEqual((observed["gated_delta_rule"], observed["gated_attention"],
                          observed["residual_rmsnorm"], observed["swiglu_mlp"]), (6, 2, 8, 8))
        self.assertEqual(observed["gdn_in_proj"], 6)
        self.assertEqual(observed["qkv_norm_partial_rope"], 2)
        self.assertEqual(build.count_problems(), [])
        self.assertEqual(len(build.provenance.paths["gated_delta_rule"]), 6)
        self.assertEqual(len(build.provenance.paths["gated_attention"]), 2)
        self.assertTrue(all(".linear_attn" in p for p in build.provenance.paths["gdn_in_proj"]))
        self.assertTrue(all(".self_attn" in p
                            for p in build.provenance.paths["qkv_norm_partial_rope"]))

    def test_one_site_patched_counts_only_its_adapter_group(self):
        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload()
        control = restrict(sites.structural_identity_kernels(workload.site_registry),
                           ("swiglu_mlp",))
        _, _, build = _step(workload, control)
        self.assertEqual(build.observed(), {"swiglu_mlp": 8})
        self.assertEqual(build.count_problems(), [])


class TestStructuralIdentity(unittest.TestCase):
    def test_bitwise_the_unpatched_model(self):
        for dtype in ("float32", "bfloat16"):
            workload = tiny_workload(dtype)
            with self.subTest(dtype=dtype):
                loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
                loss1, grads1, _ = _step(workload,
                                         sites.structural_identity_kernels(workload.site_registry))
                self.assertTrue(torch.equal(loss0, loss1))
                self.assertEqual(sorted(grads0), sorted(grads1))
                for name in grads0:
                    self.assertTrue(torch.equal(grads0[name], grads1[name]), name)

class TestRunner(unittest.TestCase):
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

class TestWorkloadProtocol(unittest.TestCase):
    def test_config_round_trip_and_canonical_identity(self):
        canonical = Qwen3_5Workload()
        self.assertTrue(canonical.spec.is_canonical)
        self.assertEqual((canonical.spec.batch_size, canonical.spec.seq_len), (1, 2048))
        rebuilt = Qwen3_5Workload.from_config(tiny_workload().to_config())
        self.assertEqual(rebuilt.spec.workload_id, tiny_workload().spec.workload_id)
        self.assertFalse(rebuilt.spec.is_canonical)

    def test_describe_and_runtime_report(self):
        workload = tiny_workload()
        described = workload.describe()
        self.assertEqual(described["layer_types"], {"linear_attention": 6, "full_attention": 2})
        self.assertEqual(described["expected_site_counts"]["gated_delta_rule"], 6)
        model, _ = workload.build_patched(KernelSet(registry=workload.site_registry))
        report = workload.runtime_report(model)
        self.assertEqual(report["gated_delta_rule_dispatch"], "torch")
        self.assertEqual(report["causal_conv1d_dispatch"], "torch")
        self.assertEqual(report["attn_implementation"], "sdpa")

    def test_whole_model_compile_is_a_baseline_of_the_unpatched_model(self):
        from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import (
            whole_model_compile_kernels)

        workload = tiny_workload()
        model, provenance = workload.build_patched(
            whole_model_compile_kernels(workload.site_registry))
        self.assertEqual(provenance.method, "whole_model_torch_compile")
        self.assertIn("whole_model_torch_compile", workload.runtime_report(model))
        mixed = patch(whole_model_compile_kernels(workload.site_registry), "swiglu_mlp",
                      sites.PRODUCTION["swiglu_mlp"])
        with self.assertRaises(ValueError):
            workload.build_patched(mixed)

if __name__ == "__main__":
    unittest.main()
