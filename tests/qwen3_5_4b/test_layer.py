"""Qwen3.5's representative decoder layer as a data source (``topdown/qwen3_5_4b/layer``).

The two traps a bare decoder layer falls into: T-init (a layer built outside the
model skips ``post_init``) and T-dispatch (an importable FLA must never reach the
production spelling). The layer feeds the Level-2 sites' real invocations.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import torch

from evograd.benchmark.topdown.qwen3_5_4b.layer import prepare
from evograd.evaluation.tier3.workloads.qwen3_5_4b import sites

from tests.qwen3_5_4b._tiny import BATCH, FULL_LAYER, GDN_LAYER, SEQ, TINY_ARCH

REPO = Path(__file__).resolve().parents[2]


class TestTInit(unittest.TestCase):
    def test_a_bare_layer_is_not_the_models(self):
        """The trap: no post_init, so the linears keep PyTorch's default scale."""
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

        layer = modeling.Qwen3_5DecoderLayer(prepare.build_config(TINY_ARCH), GDN_LAYER)
        weights = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        problems = prepare.check_initialization(weights)
        self.assertTrue(any("not the initializer's" in p for p in problems), problems)

    def test_explicit_rules_give_the_models_layer(self):
        for index in (GDN_LAYER, FULL_LAYER):
            with self.subTest(layer=index):
                weights = prepare.initial_weights(TINY_ARCH, index, dtype="float32", seed=0)
                self.assertEqual(prepare.check_initialization(weights), [])
                norms = [k for k in weights if k.endswith("layernorm.weight")]
                self.assertTrue(norms and all(bool((weights[k] == 0).all()) for k in norms))
        gdn = prepare.initial_weights(TINY_ARCH, GDN_LAYER, dtype="float32", seed=0)
        self.assertTrue(bool((gdn["linear_attn.dt_bias"] == 1).all()))
        self.assertTrue(bool((gdn["linear_attn.norm.weight"] == 1).all()))
        a = gdn["linear_attn.A_log"].exp()
        self.assertTrue(bool(((a >= 0.01 - 1e-4) & (a <= 16 + 1e-3)).all()))

    def test_the_rules_are_the_full_models_post_init(self):
        """Every rule matches what ``_from_config`` (post_init) produces: same
        constants, same distributions (checked by moments, not bits -- the RNG
        streams differ)."""
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import build_model
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL

        model = build_model(CANONICAL.replace(arch=TINY_ARCH, device="cpu", seq_len=SEQ,
                                              dtype="float32"))
        full = {k: v.detach() for k, v in model.model.layers[GDN_LAYER].state_dict().items()}
        self.assertEqual(prepare.check_initialization(full), [])

    def test_initialization_is_seeded_and_per_layer(self):
        a = prepare.initial_weights(TINY_ARCH, GDN_LAYER, dtype="float32", seed=0)
        b = prepare.initial_weights(TINY_ARCH, GDN_LAYER, dtype="float32", seed=0)
        c = prepare.initial_weights(TINY_ARCH, GDN_LAYER, dtype="float32", seed=1)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(torch.equal(a["mlp.gate_proj.weight"], c["mlp.gate_proj.weight"]))


class TestPins(unittest.TestCase):
    def test_effective_settings_are_the_pinned_ones(self):
        for index in (GDN_LAYER, FULL_LAYER):
            weights = prepare.initial_weights(TINY_ARCH, index, dtype="bfloat16", seed=0)
            layer = prepare.build_single_layer(TINY_ARCH, index, device="cpu", dtype="bfloat16",
                                               weights=weights)
            effective = prepare.effective_settings(layer)
            with self.subTest(layer=index):
                self.assertEqual(prepare.check_effective_settings(effective, dtype="bfloat16"), [])
                self.assertEqual(effective["gated_delta_rule_dispatch"], "torch")
                self.assertEqual(effective["causal_conv1d_dispatch"], "torch")
            layer.train(False)
            self.assertTrue(any("training" in p for p in prepare.check_effective_settings(
                prepare.effective_settings(layer), dtype="bfloat16")))

    def test_an_importable_fla_does_not_reach_the_production_spelling(self):
        """T-dispatch, in a fresh interpreter: a fake ``fla`` is importable before
        ``modeling_qwen3_5`` is, so its module-level names dispatch to it. The
        effective settings must say so, and every pinned production spelling must
        still compute with torch."""
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
            from evograd.benchmark.topdown.qwen3_5_4b.layer import prepare
            from evograd.benchmark import TASKS
            from evograd.opdecl.oracle import resolve_runtime_forward
            from evograd.evaluation.tier3.workloads.qwen3_5_4b import sites
            from tests.qwen3_5_4b._tiny import TINY_ARCH, GDN_LAYER
            w = prepare.initial_weights(TINY_ARCH, GDN_LAYER, dtype="float32", seed=0)
            layer = prepare.build_single_layer(TINY_ARCH, GDN_LAYER, device="cpu",
                                               dtype="float32", weights=w)
            eff = prepare.effective_settings(layer)
            x = torch.randn(1, 8, TINY_ARCH["hidden_size"])
            q, k, v, z, beta, g = sites.production_gdn_in_proj(layer.linear_attn, x)
            out = sites.production_gated_delta_rule(layer.linear_attn, q, k, v, g, beta)
            task = resolve_runtime_forward(TASKS["qwen3_5_gated_delta_rule"])(q, k, v, g, beta)
            torch_out, _ = modeling.torch_chunk_gated_delta_rule.__wrapped__(
                q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
            try:
                modeling.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                                      use_qk_l2norm_in_kernel=True)
                native = "torch"
            except RuntimeError:
                native = "fla"
            print(json.dumps({"dispatch": eff["gated_delta_rule_dispatch"],
                              "problems": prepare.check_effective_settings(eff, dtype="float32"),
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
        self.assertTrue(any("gated_delta_rule_dispatch" in p for p in report["problems"]))
        self.assertTrue(report["production_equal"])
        self.assertTrue(report["task_equal"])
        self.assertEqual(report["fla_calls_before_native"], [])

    def test_the_sidecar_imports_qwen3_5_before_fla(self):
        from evograd.opdecl.sidecar import SIDECARS

        self.assertIn("transformers.models.qwen3_5.modeling_qwen3_5",
                      SIDECARS["fla"].dispatches_first)




if __name__ == "__main__":
    unittest.main()
