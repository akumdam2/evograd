"""The Qwen3.5 Level-3 Tier-3 workload, on an 8-layer tiny model (CPU).

Eight layers in the published pattern -- six Gated DeltaNet, two full attention
-- so the per-site patch counts are exactly a quarter of Qwen3.5-4B's
24 / 8 / 32 / 64. Covers: the counts, structural identity and the model
reference deployed at every site against the unpatched model (loss and every
gradient, bitwise), the pinned torch dispatch (T-dispatch), and the runner on
the workload.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import torch

from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
from evograd.evaluation.tier3.patch import KernelSet, patch
from evograd.evaluation.tier3.workloads.qwen3_5_4b import sites
from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload

from tests.qwen3_5_4b._tiny import SEQ, TINY_OVERRIDES

ARCH8 = {**with_layers(QWEN3_5_4B, 8), **TINY_OVERRIDES}
REPO = Path(__file__).resolve().parents[2]


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
                          "attention_block": 8, "residual_rmsnorm": 64, "swiglu_mlp": 32})

    def test_every_site_patched_on_every_layer_of_its_type(self):
        workload = tiny_workload()
        _, _, build = _step(workload, sites.structural_identity_kernels(workload.site_registry))
        self.assertEqual(build.observed(), {"gdn_in_proj": 6, "gated_delta_rule": 6,
                                            "gdn_norm_out": 6, "attention_block": 2,
                                            "residual_rmsnorm": 16, "swiglu_mlp": 8})
        self.assertEqual(build.count_problems(), [])
        paths = build.provenance.paths
        self.assertEqual(len(paths["gated_delta_rule"]), 6)
        self.assertTrue(all(".linear_attn" in p for p in paths["gdn_in_proj"]))
        self.assertEqual(paths["attention_block"],
                         ("model.layers.3.self_attn", "model.layers.7.self_attn"))
        # The residual site rebinds every decoder layer and the final norm.
        self.assertEqual(paths["residual_rmsnorm"],
                         tuple(f"model.layers.{i}" for i in range(8)) + ("model.norm",))
        self.assertIsInstance(build.carrier, sites.ResidualCarrier)
        self.assertIsNone(build.carrier.pending)

    def test_one_site_patched_counts_only_its_adapter_group(self):
        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload()
        control = restrict(sites.structural_identity_kernels(workload.site_registry),
                           ("swiglu_mlp",))
        _, _, build = _step(workload, control)
        self.assertEqual(build.observed(), {"swiglu_mlp": 8})
        self.assertEqual(build.count_problems(), [])


def model_reference_kernels(registry):
    """Every site patched with its task's runtime spelling, called through the
    task ABI: what model-scope Gate 2 and Gate 3 deploy as the native candidate."""
    from evograd.benchmark import get_task
    from evograd.evaluation.tier3.patch import KernelSource
    from evograd.opdecl.oracle import resolve_runtime_forward

    kernels = KernelSet(registry=registry)
    for site in registry.sites:
        kernels = patch(kernels, site.name, resolve_runtime_forward(get_task(site.op)),
                        source=KernelSource(site=site.name, op_name=site.op, module=None,
                                            origin="model_reference"))
    return kernels


class TestStructuralIdentity(unittest.TestCase):
    def test_bitwise_the_unpatched_model(self):
        """Production spellings, and the declared tasks' own spellings through the
        adapters (the attention block, the carried residual fusion): the loss and
        every gradient bit for bit."""
        for dtype in ("float32", "bfloat16"):
            workload = tiny_workload(dtype)
            loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
            for name, kernels in (
                    ("structural_identity",
                     sites.structural_identity_kernels(workload.site_registry)),
                    ("model_reference", model_reference_kernels(workload.site_registry))):
                with self.subTest(dtype=dtype, kernels=name):
                    loss1, grads1, build = _step(workload, kernels)
                    self.assertEqual(build.count_problems(), [])
                    self.assertTrue(torch.equal(loss0, loss1))
                    self.assertEqual(sorted(grads0), sorted(grads1))
                    for param in grads0:
                        self.assertTrue(torch.equal(grads0[param], grads1[param]), param)

    def test_each_site_alone_is_bitwise_the_unpatched_model(self):
        from evograd.evaluation.tier3.patch import restrict

        workload = tiny_workload("bfloat16")
        loss0, grads0, _ = _step(workload, KernelSet(registry=workload.site_registry))
        reference = model_reference_kernels(workload.site_registry)
        for site in ("attention_block", "residual_rmsnorm"):
            with self.subTest(site=site):
                loss1, grads1, build = _step(workload, restrict(reference, (site,)))
                self.assertEqual(build.observed(), {site: build.expected[site]})
                self.assertTrue(torch.equal(loss0, loss1))
                for param in grads0:
                    self.assertTrue(torch.equal(grads0[param], grads1[param]), param)

class TestDispatch(unittest.TestCase):
    def test_an_importable_fla_does_not_reach_the_production_spelling(self):
        """T-dispatch, in a fresh interpreter: a fake ``fla`` is importable before
        ``modeling_qwen3_5`` is, so its module-level names dispatch to it. The
        dispatch read-back must say so, and the pinned production spellings and the
        task's runtime spelling must still compute with torch."""
        script = textwrap.dedent("""
            import json, sys, types
            import torch
            called = []
            fla = types.ModuleType("fla")
            def chunk_gated_delta_rule(*a, **k):
                called.append("fla"); raise RuntimeError("fake FLA was called")
            fla.chunk_gated_delta_rule = chunk_gated_delta_rule
            fla.fused_recurrent_gated_delta_rule = chunk_gated_delta_rule
            sys.modules["fla"] = fla
            from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
            from evograd.benchmark import TASKS
            from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import (
                build_model, dispatch_settings)
            from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL
            from evograd.opdecl.oracle import resolve_runtime_forward
            from evograd.evaluation.tier3.workloads.qwen3_5_4b import sites
            from tests.qwen3_5_4b._tiny import TINY_ARCH, GDN_LAYER
            model = build_model(CANONICAL.replace(arch=TINY_ARCH, device="cpu", seq_len=8,
                                                  dtype="float32"))
            gdn = model.model.layers[GDN_LAYER].linear_attn
            x = torch.randn(1, 8, TINY_ARCH["hidden_size"])
            with torch.no_grad():
                q, k, v, z, beta, g = sites.production_gdn_in_proj(gdn, x)
                out = sites.production_gated_delta_rule(gdn, q, k, v, g, beta)
                task = resolve_runtime_forward(TASKS["qwen3_5_gated_delta_rule"])(q, k, v, g, beta)
                torch_out, _ = modeling.torch_chunk_gated_delta_rule.__wrapped__(
                    q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            try:
                modeling.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                                      use_qk_l2norm_in_kernel=True)
                native = "torch"
            except RuntimeError:
                native = "fla"
            print(json.dumps({"dispatch": dispatch_settings()["gated_delta_rule_dispatch"],
                              "native": native,
                              "production_equal": bool(torch.equal(out, torch_out)),
                              "task_equal": bool(torch.equal(task, torch_out)),
                              "fla_calls_before_native": called[:-1] if native == "fla" else called}))
        """)
        env = {"PYTHONPATH": f"{REPO / 'src'}:{REPO}", "PATH": "/usr/bin:/bin",
               "HOME": str(Path.home())}
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                cwd=REPO, env=env, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["native"], "fla", "the trap did not trigger; the test is void")
        self.assertNotEqual(report["dispatch"], "torch")
        self.assertTrue(report["production_equal"])
        self.assertTrue(report["task_equal"])
        self.assertEqual(report["fla_calls_before_native"], [])

    def test_the_sidecar_imports_qwen3_5_before_fla(self):
        from evograd.opdecl.sidecar import SIDECARS

        self.assertIn("transformers.models.qwen3_5.modeling_qwen3_5",
                      SIDECARS["fla"].dispatches_first)


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
        self.assertEqual(described["expected_site_counts"]["residual_rmsnorm"], 16)
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
