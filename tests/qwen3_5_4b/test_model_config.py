"""The Qwen3.5-4B record in :mod:`evograd.opdecl.models`, the architecture it
mirrors, and the parameter counts derived from either.

Every shape a Qwen3.5 case declares is computed from this record, so it is
checked against the published ``text_config`` and against the HF modules a
config builds; the analytic parameter count is held to the numbers the review
measured (4.206B total, 0.636B embedding, 112.92M per GDN layer, 107.48M per
full-attention layer).
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

from evograd.benchmark.topdown.common.spec import analytic_parameter_count
from evograd.benchmark.topdown.qwen3_5_4b.architecture import (
    NON_ARCHITECTURAL,
    QWEN3_5_4B as ARCH,
    REPRESENTATIVE_LAYERS,
    config_record,
    derived_layer_types,
    layer_kind,
    layer_types,
    with_layers,
)
from evograd.opdecl import Provenance
from evograd.opdecl.models import MODELS, QWEN3_5_4B as CFG, rederive_dims

#: ``text_config`` of ``Qwen/Qwen3.5-4B``'s ``config.json`` as fetched for the
#: 2026-10-10 review, verbatim. A test-side copy, independent of
#: ``architecture.py``: the two are compared field by field.
PUBLISHED_TEXT_CONFIG = json.loads("""
{
    "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true,
    "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4,
    "head_dim": 256, "hidden_act": "silu", "hidden_size": 2560,
    "initializer_range": 0.02, "intermediate_size": 9216,
    "layer_types": [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 32, "linear_value_head_dim": 128,
    "max_position_embeddings": 262144, "mlp_only_layers": [], "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 16, "num_hidden_layers": 32, "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "use_cache": true,
    "vocab_size": 248320, "mamba_ssm_dtype": "float32",
    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                        "rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25}
}
""")


class TestArchitecture(unittest.TestCase):
    def test_every_published_field_is_recorded_exactly_once(self):
        recorded = {**ARCH, **NON_ARCHITECTURAL}
        self.assertEqual(set(recorded), set(PUBLISHED_TEXT_CONFIG))
        self.assertFalse(set(ARCH) & set(NON_ARCHITECTURAL))
        for key, value in PUBLISHED_TEXT_CONFIG.items():
            with self.subTest(key=key):
                self.assertEqual(recorded[key], value)

    def test_layer_types_match_the_interval_rule(self):
        self.assertEqual(layer_types(), derived_layer_types())
        self.assertEqual(layer_types().count("linear_attention"), 24)
        self.assertEqual([i for i, t in enumerate(layer_types()) if t == "full_attention"],
                         [3, 7, 11, 15, 19, 23, 27, 31])
        self.assertEqual(layer_kind(REPRESENTATIVE_LAYERS["linear_attention"]),
                         "linear_attention")
        self.assertEqual(layer_kind(REPRESENTATIVE_LAYERS["full_attention"]), "full_attention")

    def test_shrinking_the_depth_keeps_the_pattern(self):
        small = with_layers(ARCH, 6)
        self.assertEqual(small["layer_types"], list(layer_types())[:6])
        self.assertEqual(small["num_hidden_layers"], 6)
        with self.assertRaises(ValueError):
            with_layers(ARCH, 33)

    def test_record_is_the_architecture(self):
        self.assertEqual(dataclasses.replace(config_record(), source=CFG.source), CFG)
        self.assertEqual(CFG.layer_types, layer_types())


class TestRecord(unittest.TestCase):
    def test_registered_under_its_own_key(self):
        self.assertIs(MODELS["qwen3_5_4b"], CFG)
        self.assertEqual(len(MODELS), 6)

    def test_derived_widths(self):
        self.assertEqual(CFG.key_dim, 2048)
        self.assertEqual(CFG.value_dim, 4096)
        self.assertEqual(CFG.conv_dim, 8192)
        self.assertEqual(CFG.q_gate_out, 8192)
        self.assertEqual(CFG.q_out, 4096)
        self.assertEqual(CFG.kv_out, 1024)
        self.assertEqual(CFG.rotary_dim, 64)
        self.assertEqual(CFG.rope_theta, 1e7)
        self.assertEqual(CFG.layer_count("linear_attention"), 24)
        self.assertEqual(CFG.layer_count("full_attention"), 8)


class TestDims(unittest.TestCase):
    """Canonical B1 x T2048."""

    def test_level2_sites(self):
        self.assertEqual(CFG.gdn_in_proj_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "H": 2560, "C": 8192, "VD": 4096, "HV": 32,
                          "K": 4, "D": 128})
        self.assertEqual(CFG.gated_delta_rule_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "HV": 32, "D": 128})
        self.assertEqual(CFG.gdn_norm_out_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "HV": 32, "D": 128, "VD": 4096, "H": 2560})
        self.assertEqual(CFG.qkv_norm_partial_rope_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "H": 2560, "HQ": 16, "HK": 4, "D": 256, "R": 64,
                          "QG": 8192, "KVO": 1024, "QO": 4096})
        self.assertEqual(CFG.gated_attention_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "HQ": 16, "HK": 4, "D": 256, "QO": 4096,
                          "H": 2560})
        self.assertEqual(CFG.residual_rmsnorm_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "H": 2560})
        self.assertEqual(CFG.swiglu_mlp_dims(batch=1, seq=2048),
                         {"B": 1, "T": 2048, "H": 2560, "I": 9216})

    def test_level1_components(self):
        self.assertEqual(CFG.gdn_conv1d_dims(batch=1, seq=2048),
                         {"B": 1, "C": 8192, "T": 2048, "K": 4})
        self.assertEqual(CFG.gdn_qk_l2norm_dims(batch=1, seq=2048),
                         {"rows": 2048 * 32, "cols": 128})
        self.assertEqual(CFG.gdn_gated_norm_dims(tokens=2048), {"rows": 2048 * 32, "hidden": 128})
        self.assertEqual(CFG.logits_dims(tokens=2048), {"rows": 2048, "cols": 248320})
        self.assertEqual(CFG.lm_head_dims(tokens=2048), {"M": 2048, "K": 2560, "N": 248320})
        self.assertEqual(CFG.gdn_in_ba_dims(tokens=2048), {"M": 2048, "K": 2560, "N": 32})
        # in_proj_qkv and q_proj have the same shape, as do out_proj and o_proj:
        # a harvest collapses each pair into one configuration.
        self.assertEqual(CFG.gdn_in_qkv_dims(tokens=2048), CFG.attn_q_gate_proj_dims(tokens=2048))
        self.assertEqual(CFG.gdn_out_proj_dims(tokens=2048), CFG.attn_out_proj_dims(tokens=2048))

    def test_rederive_reaches_every_component(self):
        free_by_kind = {"batch_seq": {"batch": 1, "seq": 2048}, "tokens": {"tokens": 2048}}
        for name in dir(CFG):
            if not name.endswith("_dims"):
                continue
            component = name[: -len("_dims")]
            builder = getattr(CFG, name)
            kind = "tokens" if "tokens" in builder.__code__.co_varnames else "batch_seq"
            free = free_by_kind[kind]
            with self.subTest(component=component):
                provenance = Provenance(model=CFG.name, component=component, free=free)
                self.assertEqual(rederive_dims(provenance), builder(**free))


class TestParameterCount(unittest.TestCase):
    def test_published_counts(self):
        counts = analytic_parameter_count(ARCH)
        self.assertEqual(counts["per_layer"], {"linear_attention": 112_923_840,
                                               "full_attention": 107_484_672})
        self.assertEqual(counts["layers_by_type"], {"linear_attention": 24, "full_attention": 8})
        self.assertEqual(counts["embedding"], 635_699_200)
        self.assertEqual(counts["lm_head"], 0)  # tied
        self.assertEqual(counts["total"], 4_205_751_296)
        self.assertAlmostEqual(counts["total"] / 1e9, 4.206, places=3)

    def test_count_matches_a_built_model(self):
        try:
            import torch
            from transformers import Qwen3_5ForCausalLM
        except ImportError:  # pragma: no cover
            self.skipTest("transformers with qwen3_5 is not installed")
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import build_config

        from ._tiny import TINY_ARCH
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL

        spec = CANONICAL.replace(arch=TINY_ARCH, device="cpu", seq_len=16)
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(build_config(spec))
        built = sum(p.numel() for p in model.parameters())
        self.assertEqual(analytic_parameter_count(TINY_ARCH)["total"], built)
        layer = model.model.layers
        counts = analytic_parameter_count(TINY_ARCH)["per_layer"]
        self.assertEqual(sum(p.numel() for p in layer[2].parameters()), counts["linear_attention"])
        self.assertEqual(sum(p.numel() for p in layer[3].parameters()), counts["full_attention"])

    def test_a_routed_moe_takes_the_moe_counter_not_the_hybrid_one(self):
        """``num_experts`` routes to the routed-MoE counter (OLMoE's), which
        reads no hybrid field; a routed MoE missing its top-k is refused by
        name rather than counted as dense."""
        from evograd.benchmark.topdown.common.spec import moe_parameter_count

        from ._tiny import TINY_ARCH

        moe = {**TINY_ARCH, "num_experts": 8, "num_experts_per_tok": 2}
        self.assertEqual(analytic_parameter_count(moe), moe_parameter_count(moe))
        self.assertNotEqual(analytic_parameter_count(moe)["total"],
                            analytic_parameter_count(TINY_ARCH)["total"])
        with self.assertRaises(KeyError):
            analytic_parameter_count({**TINY_ARCH, "num_experts": 8})


if __name__ == "__main__":
    unittest.main()
