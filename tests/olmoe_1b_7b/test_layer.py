"""OLMoE's representative decoder layer as a data source (``topdown/olmoe_1b_7b/layer``).

The trap a bare decoder layer falls into: T-init (a layer built outside the
model skips ``post_init``, so the router and the 3-D expert tensors keep their
constructor values). The layer feeds the Level-2 sites' real invocations, and
its experts kernel is read back rather than assumed.
"""

from __future__ import annotations

import unittest

import torch

from evograd.benchmark.topdown.olmoe_1b_7b.layer import prepare

from tests.olmoe_1b_7b._tiny import LAYER, SEQ, TINY_ARCH


class TestTInit(unittest.TestCase):
    def test_a_bare_layer_is_not_the_models(self):
        """The trap: no post_init, so the linears keep PyTorch's default scale
        and the router stays at its constructor zeros."""
        from transformers.models.olmoe import modeling_olmoe as modeling

        layer = modeling.OlmoeDecoderLayer(prepare.build_config(TINY_ARCH), LAYER)
        weights = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        problems = prepare.check_initialization(weights)
        self.assertTrue(any("not the initializer's" in p for p in problems), problems)
        self.assertTrue(any("mlp.gate.weight" in p for p in problems), problems)

    def test_explicit_rules_give_the_models_layer(self):
        weights = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=0)
        self.assertEqual(prepare.check_initialization(weights), [])
        for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                     "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            self.assertTrue(bool((weights[name] == 1).all()), name)
        self.assertEqual(tuple(weights["mlp.experts.gate_up_proj"].shape), (4, 64, 64))
        self.assertEqual(tuple(weights["mlp.experts.down_proj"].shape), (4, 64, 32))
        self.assertEqual(tuple(weights["mlp.gate.weight"].shape), (4, 64))

    def test_the_rules_are_the_full_models_post_init(self):
        """Every rule matches what ``_from_config`` (post_init) produces: same
        constants, same distributions (checked by moments, not bits -- the RNG
        streams differ)."""
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.model import build_model
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.spec import CANONICAL

        model = build_model(CANONICAL.replace(arch=TINY_ARCH, device="cpu", seq_len=SEQ,
                                              dtype="float32"))
        full = {k: v.detach() for k, v in model.model.layers[LAYER].state_dict().items()}
        self.assertEqual(prepare.check_initialization(full), [])
        self.assertEqual(set(full), set(prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32")))

    def test_initialization_is_seeded_and_per_layer(self):
        a = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=0)
        b = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=0)
        c = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=1)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertFalse(torch.equal(a["mlp.experts.gate_up_proj"], c["mlp.experts.gate_up_proj"]))


class TestPins(unittest.TestCase):
    def test_effective_settings_are_the_pinned_ones(self):
        weights = prepare.initial_weights(TINY_ARCH, LAYER, dtype="bfloat16", seed=0)
        layer = prepare.build_single_layer(TINY_ARCH, LAYER, device="cpu", dtype="bfloat16",
                                           weights=weights)
        effective = prepare.effective_settings(layer)
        self.assertEqual(prepare.check_effective_settings(effective, dtype="bfloat16"), [])
        self.assertEqual(effective["experts_implementation"], "grouped_mm")
        layer.train(False)
        self.assertTrue(any("training" in p for p in prepare.check_effective_settings(
            prepare.effective_settings(layer), dtype="bfloat16")))

    def test_an_eager_layer_is_flagged_against_the_canonical_pin(self):
        weights = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=0)
        layer = prepare.build_single_layer(TINY_ARCH, LAYER, device="cpu", dtype="float32",
                                           weights=weights, experts_implementation="eager")
        effective = prepare.effective_settings(layer)
        self.assertEqual(effective["experts_implementation"], "eager")
        self.assertTrue(any("experts_implementation" in p for p in
                            prepare.check_effective_settings(effective, dtype="float32")))
        self.assertEqual(prepare.check_effective_settings(effective, dtype="float32",
                                                          experts_implementation="eager"), [])

    def test_the_two_kernels_agree_on_one_layer_call(self):
        """Same weights, same inputs, grouped GEMM against the eager loop: the
        same mathematics to float32 rounding (the grouped path sums the top-k
        contributions in float32; the loop accumulates in the model dtype)."""
        weights = prepare.initial_weights(TINY_ARCH, LAYER, dtype="float32", seed=0)
        inputs = prepare.block_inputs(TINY_ARCH, LAYER, batch=1, seq=SEQ, dtype="float32")
        outs = []
        for experts in ("grouped_mm", "eager"):
            layer = prepare.build_single_layer(TINY_ARCH, LAYER, device="cpu", dtype="float32",
                                               weights=weights, experts_implementation=experts)
            with torch.no_grad():
                outs.append(layer(inputs["hidden_states"],
                                  position_embeddings=inputs["position_embeddings"]))
        self.assertTrue(torch.allclose(outs[0], outs[1], atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
