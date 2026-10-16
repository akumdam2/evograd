"""The OLMoE-1B-7B record in :mod:`evograd.opdecl.models`, the architecture it
mirrors, and the parameter counts derived from either.

Every shape an OLMoE case declares is computed from this record, so it is
checked against the published ``config.json`` (a test-side copy, compared field
by field) and the analytic parameter count is held to the model card's
numbers (6.92B total, 1.28B active per token).
"""

from __future__ import annotations

import dataclasses
import json
import unittest

from evograd.benchmark.topdown.common.spec import analytic_parameter_count, moe_parameter_count
from evograd.benchmark.topdown.olmoe_1b_7b.architecture import (
    NON_ARCHITECTURAL,
    OLMOE_1B_7B as ARCH,
    PUBLISHED_OMITTED,
    PUBLISHED_OVERRIDDEN,
    REPRESENTATIVE_LAYER,
    config_record,
    layer_kind,
    with_layers,
)
from evograd.opdecl.models import MODELS, OLMOE_1B_7B as CFG, rederive_dims

#: ``allenai/OLMoE-1B-7B-0924``'s ``config.json`` as downloaded on 2026-10-16
#: to ``/projects/bcjw/akumdam/models/OLMoE-1B-7B-0924/config.json``, verbatim.
PUBLISHED_CONFIG = json.loads("""
{
  "architectures": ["OlmoeForCausalLM"],
  "attention_bias": false, "attention_dropout": 0.0, "clip_qkv": null,
  "eos_token_id": 50279, "hidden_act": "silu", "hidden_size": 2048,
  "initializer_range": 0.02, "intermediate_size": 1024,
  "max_position_embeddings": 4096, "model_type": "olmoe", "norm_topk_prob": false,
  "num_attention_heads": 16, "num_experts": 64, "num_experts_per_tok": 8,
  "num_hidden_layers": 16, "num_key_value_heads": 16, "output_router_logits": false,
  "pad_token_id": 1, "rope_scaling": null, "rope_theta": 10000.0,
  "router_aux_loss_coef": 0.01, "tie_word_embeddings": false,
  "torch_dtype": "bfloat16", "transformers_version": "4.43.0.dev0",
  "use_cache": true, "vocab_size": 50304
}
""")


class TestArchitecture(unittest.TestCase):
    def test_every_published_field_is_recorded_exactly_once(self):
        recorded = {**ARCH, **NON_ARCHITECTURAL}
        self.assertEqual(set(recorded) - set(PUBLISHED_OMITTED), set(PUBLISHED_CONFIG))
        self.assertFalse(set(ARCH) & set(NON_ARCHITECTURAL))
        for key, value in PUBLISHED_CONFIG.items():
            with self.subTest(key=key):
                if key in PUBLISHED_OVERRIDDEN:
                    self.assertEqual(PUBLISHED_OVERRIDDEN[key], value, "the published value")
                    self.assertNotEqual(recorded[key], value, "the stated departure")
                else:
                    self.assertEqual(recorded[key], value)

    def test_the_two_departures_are_stated(self):
        """The aux loss is pinned on; the norm epsilon is the library default
        the published file omits. Both are the architecture's and both say so."""
        self.assertIs(ARCH["output_router_logits"], True)
        self.assertIs(PUBLISHED_CONFIG["output_router_logits"], False)
        self.assertEqual(PUBLISHED_OVERRIDDEN, {"output_router_logits": False})
        self.assertNotIn("rms_norm_eps", PUBLISHED_CONFIG)
        self.assertEqual(ARCH["rms_norm_eps"], PUBLISHED_OMITTED["rms_norm_eps"])

    def test_the_record_matches_the_architecture(self):
        self.assertEqual(dataclasses.replace(config_record(), source=CFG.source), CFG)
        self.assertIs(MODELS["olmoe_1b_7b"], CFG)
        self.assertEqual((CFG.q_out, CFG.kv_out, CFG.gate_up_out), (2048, 2048, 2048))
        self.assertEqual(CFG.n_heads, CFG.n_kv_heads, "OLMoE-1B-7B is multi-head attention")
        self.assertFalse(CFG.norm_topk_prob)
        self.assertFalse(CFG.tie_word_embeddings)

    def test_every_layer_is_a_decoder_layer(self):
        self.assertEqual(layer_kind(REPRESENTATIVE_LAYER), "decoder_layer")
        with self.assertRaises(ValueError):
            layer_kind(16)
        self.assertEqual(with_layers(ARCH, 4)["num_hidden_layers"], 4)
        self.assertEqual(config_record(with_layers(ARCH, 4), name="tiny").layers, 4)


class TestParameterCount(unittest.TestCase):
    def test_total_and_active_match_the_model_card(self):
        counts = analytic_parameter_count(ARCH)
        self.assertEqual(counts, moe_parameter_count(ARCH))
        self.assertAlmostEqual(counts["total"] / 1e9, 6.92, places=2)
        self.assertAlmostEqual(counts["active"] / 1e9, 1.28, places=2)
        self.assertEqual(counts["embedding"], 50304 * 2048)
        self.assertEqual(counts["lm_head"], counts["embedding"], "untied")
        self.assertEqual(counts["per_expert"], 3 * 1024 * 2048)
        self.assertEqual(counts["experts_per_layer"], 64 * counts["per_expert"])
        self.assertEqual(counts["router_per_layer"], 64 * 2048)

    def test_the_dense_counter_is_not_used_for_a_moe(self):
        dense = dict(ARCH)
        dense.pop("num_experts")
        self.assertNotEqual(analytic_parameter_count(dense)["total"],
                            analytic_parameter_count(ARCH)["total"])


class TestSiteDims(unittest.TestCase):
    def test_every_site_rederives_from_the_record(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest
        from evograd.opdecl import Provenance

        for site in manifest.SITES:
            with self.subTest(site=site):
                dims = manifest.dims_for(site, CFG, batch=2, seq=2048)
                provenance = Provenance(model="olmoe_1b_7b", component=site,
                                        free={"batch": 2, "seq": 2048})
                self.assertEqual(rederive_dims(provenance), dims)
        self.assertEqual(manifest.dims_for("moe_experts", CFG, batch=2, seq=2048),
                         {"T": 4096, "H": 2048, "E": 64, "G": 2048, "I": 1024, "K": 8})
        self.assertEqual(manifest.dims_for("qkv_norm_rope", CFG, batch=2, seq=2048)["QO"], 2048)

    def test_level1_components_rederive(self):
        self.assertEqual(CFG.moe_activation_dims(tokens=4096), {"rows": 32768, "cols": 1024})
        self.assertEqual(CFG.attn_q_norm_dims(tokens=4096), {"rows": 4096, "hidden": 2048})
        self.assertEqual(CFG.attn_k_norm_dims(tokens=4096), {"rows": 4096, "hidden": 2048})
        self.assertEqual(CFG.lm_head_dims(tokens=4096), {"M": 4096, "K": 2048, "N": 50304})


if __name__ == "__main__":
    unittest.main()
