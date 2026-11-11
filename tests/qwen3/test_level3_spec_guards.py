"""Qwen3-0.6B's Level-3 spec: the canonical identity and the pinned checkpoint."""

from __future__ import annotations

import unittest


class TestCanonicalIdentity(unittest.TestCase):
    def test_the_synthetic_canonical_identity_is_unchanged_by_the_new_fields(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import CANONICAL
        self.assertTrue(CANONICAL.workload_id.endswith("6e7919ad"))
        self.assertNotIn("weights", CANONICAL.to_dict())
        self.assertNotIn("data", CANONICAL.to_dict())


class TestPretrainedArchitectureGuard(unittest.TestCase):
    def test_the_two_rope_spellings_are_the_same_rotation(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import rope_settings
        old = {"rope_theta": 1000000.0, "rope_scaling": None}
        new = {"rope_theta": None, "rope_scaling": {"rope_theta": 1000000, "rope_type": "default"}}
        self.assertEqual(rope_settings(old), rope_settings(new))
        self.assertEqual(rope_settings(new), (1000000.0, "default"))

    def test_a_different_theta_or_type_is_still_a_disagreement(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import rope_settings
        base = {"rope_theta": 1000000.0, "rope_scaling": None}
        self.assertNotEqual(rope_settings(base), rope_settings({"rope_theta": 10000.0, "rope_scaling": None}))
        self.assertNotEqual(rope_settings(base), rope_settings(
            {"rope_theta": None, "rope_scaling": {"rope_theta": 1000000, "rope_type": "yarn"}}))

    def test_the_declared_qwen3_arch_matches_the_pinned_checkpoint_config(self):
        import json, os
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.model import rope_settings
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import QWEN3_0_6B
        snapshot = ("/u/wzhan/.cache/evograd-mF/hf/models--Qwen--Qwen3-0.6B/snapshots/"
                    "c1899de289a04d12100db370d81485cdf75e47ca/config.json")
        if not os.path.exists(snapshot):
            self.skipTest("pinned checkpoint not present locally")
        loaded = json.load(open(snapshot))
        self.assertEqual(rope_settings(QWEN3_0_6B), rope_settings(loaded))
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
                    "tie_word_embeddings", "attention_bias", "rms_norm_eps"):
            self.assertEqual(loaded.get(key), QWEN3_0_6B[key], key)


class TestSharedBuildersRefuseRealRuns(unittest.TestCase):
    """The shared builders make random weights and synthetic tokens only; a spec
    naming a checkpoint or a dataset is refused rather than run on random ones
    under its pretrained/real-text id."""

    def test_a_pretrained_or_real_text_spec_is_refused(self):
        from evograd.benchmark.topdown.common.harvest import run_harvest
        from evograd.benchmark.topdown.qwen3_0_6b.declaration import WORKLOAD
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import WorkloadSpecError
        from evograd.verification.integrations.qwen3_0_6b.capture import PRETRAINED, REAL_TEXT

        from tests.qwen3.test_level3_workload import HAVE_TRANSFORMERS, tiny_spec

        spec = tiny_spec()
        for label, real in (("weights", spec.replace(weights=PRETRAINED)),
                            ("data", spec.replace(data=REAL_TEXT)),
                            ("both", spec.replace(weights=PRETRAINED, data=REAL_TEXT))):
            with self.subTest(label):
                with self.assertRaisesRegex(WorkloadSpecError, "random weights and synthetic"):
                    WORKLOAD.build_model(real)
                with self.assertRaisesRegex(WorkloadSpecError, "random weights and synthetic"):
                    WORKLOAD.make_inputs(real)
                if HAVE_TRANSFORMERS:
                    with self.assertRaises(WorkloadSpecError):
                        run_harvest(WORKLOAD, real)
