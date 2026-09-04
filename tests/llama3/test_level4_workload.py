"""The Llama-3 workload is a *specification* before it is a run.

Almost everything Llama-3 needs is shared with Qwen3 -- the spec machinery, the
builder, the observer, the manifest, the snapshot reader. What is Llama's is a
short list of declarations, and a declaration that is wrong fails quietly: the
model still builds, the harvest still runs, and every shape downstream is a
number for something else.

So these check the declarations rather than the machinery. Most are CPU-only and
build nothing; the ones that construct a model use a deliberately tiny variant
and say so.
"""

from __future__ import annotations

import json
import unittest

from evograd.bench.workloads.common.spec import analytic_parameter_count
from evograd.bench.workloads.llama3.levels.level4.spec import (
    CANONICAL,
    LLAMA_3_8B,
    MODEL_NAME,
    WorkloadSpec,
    WorkloadSpecError,
)

try:
    import transformers  # noqa: F401

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - depends on the machine
    HAVE_TRANSFORMERS = False


def tiny_spec(**overrides) -> WorkloadSpec:
    """A variant small enough to build on a CPU in a test."""
    base = {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seq_len": 64,
        "arch": {"num_hidden_layers": 2},
    }
    base.update(overrides)
    return CANONICAL.replace(**base)


class TestPublishedArchitecture(unittest.TestCase):
    """The architecture is written out, so it can be wrong. These re-derive it."""

    def test_the_parameter_count_matches_the_published_model(self):
        """8.03B total is the number on the model card. It is a function of the
        config alone, so reproducing it checks every width at once -- and no
        network or checkpoint is needed to do it."""
        counts = analytic_parameter_count(LLAMA_3_8B)
        self.assertAlmostEqual(counts["total"] / 1e9, 8.03, places=2)

    def test_the_embedding_is_not_tied(self):
        """Unlike Qwen3-0.6B. If this were wrong the lm_head would vanish from
        the parameter count and from every gradient check."""
        self.assertFalse(LLAMA_3_8B["tie_word_embeddings"])
        counts = analytic_parameter_count(LLAMA_3_8B)
        self.assertEqual(counts["lm_head"], LLAMA_3_8B["vocab_size"] * LLAMA_3_8B["hidden_size"])

    def test_the_rope_base_is_llama_3s_and_not_llama_2s(self):
        """500000, not 10000. Getting this wrong produces a RoPE kernel that is
        self-consistent and completely wrong."""
        self.assertEqual(LLAMA_3_8B["rope_theta"], 500000.0)

    def test_grouped_query_attention_is_declared(self):
        self.assertEqual(LLAMA_3_8B["num_attention_heads"], 32)
        self.assertEqual(LLAMA_3_8B["num_key_value_heads"], 8)
        self.assertEqual(
            LLAMA_3_8B["num_attention_heads"] % LLAMA_3_8B["num_key_value_heads"], 0
        )

    def test_head_dim_times_heads_equals_hidden(self):
        """True for Llama-3-8B and *not* for Qwen3-0.6B, which fans out. It is
        why this model's q_proj and o_proj deduplicate into one configuration."""
        self.assertEqual(
            LLAMA_3_8B["num_attention_heads"] * LLAMA_3_8B["head_dim"],
            LLAMA_3_8B["hidden_size"],
        )

    def test_there_is_no_per_head_qk_normalization(self):
        """Qwen3's distinguishing feature, which Llama-3 does not have. The
        observer, the level-1 mapping and any future level-2 task all depend on
        this being true."""
        self.assertNotIn("qk_norm", LLAMA_3_8B)


class TestCanonicalSpec(unittest.TestCase):
    def test_the_canonical_values(self):
        self.assertEqual(CANONICAL.model_name, MODEL_NAME)
        self.assertEqual(CANONICAL.batch_size, 2)
        self.assertEqual(CANONICAL.seq_len, 2048)
        self.assertEqual(CANONICAL.token_count, 4096)
        self.assertEqual(CANONICAL.dtype, "bfloat16")
        self.assertEqual(CANONICAL.device, "cuda")
        self.assertEqual(CANONICAL.attn_implementation, "sdpa")
        self.assertFalse(CANONICAL.use_cache)
        self.assertFalse(CANONICAL.gradient_checkpointing)
        self.assertTrue(CANONICAL.training)
        self.assertTrue(CANONICAL.is_canonical)

    def test_the_defaults_are_the_canonical_spec(self):
        self.assertEqual(WorkloadSpec(), CANONICAL)

    def test_the_sequence_fits_the_architecture(self):
        self.assertLessEqual(CANONICAL.seq_len, LLAMA_3_8B["max_position_embeddings"])

    def test_the_workload_id_is_stable_and_names_the_model(self):
        self.assertTrue(CANONICAL.workload_id.startswith("meta-llama-3-8b.train.bs2.seq2048.bf16.cuda.sdpa."))
        self.assertEqual(CANONICAL.workload_id, WorkloadSpec().workload_id)

    def test_it_does_not_collide_with_the_other_workload(self):
        """Two workloads sharing a hash would make every downstream artifact
        ambiguous about which model produced it."""
        from evograd.bench.workloads.qwen3.levels.level4.spec import CANONICAL as QWEN

        self.assertNotEqual(CANONICAL.workload_id, QWEN.workload_id)
        self.assertNotEqual(CANONICAL.workload_hash, QWEN.workload_hash)
        self.assertNotEqual(CANONICAL.config_hash, QWEN.config_hash)

    def test_the_three_rules_are_refused_here_too(self):
        for override in ({"use_cache": True},
                         {"gradient_checkpointing": True},
                         {"training": False}):
            with self.subTest(**override), self.assertRaises(WorkloadSpecError):
                CANONICAL.replace(**override)

    def test_a_sequence_past_the_context_window_is_refused(self):
        with self.assertRaises(WorkloadSpecError):
            CANONICAL.replace(seq_len=LLAMA_3_8B["max_position_embeddings"] + 1)

    def test_any_override_makes_the_run_non_canonical(self):
        for override in ({"batch_size": 1}, {"seq_len": 512},
                         {"dtype": "float32"}, {"device": "cpu"},
                         {"arch": {"num_hidden_layers": 4}}):
            with self.subTest(**override):
                other = CANONICAL.replace(**override)
                self.assertFalse(other.is_canonical)
                self.assertNotEqual(other.workload_hash, CANONICAL.workload_hash)


class TestDeclaration(unittest.TestCase):
    """The declaration is what every shared stage reads."""

    def setUp(self):
        from evograd.bench.workloads.llama3.declaration import WORKLOAD

        self.workload = WORKLOAD

    def test_it_names_the_registry_key(self):
        from evograd.bench.workloads import WORKLOADS

        self.assertIn(self.workload.name, WORKLOADS)

    def test_its_canonical_spec_is_the_modules(self):
        self.assertEqual(self.workload.canonical, CANONICAL)

    def test_the_observation_plan_names_llamas_classes(self):
        classes = {b.class_name for b in self.workload.plan.boundaries}
        self.assertEqual(
            classes,
            {"LlamaDecoderLayer", "LlamaAttention", "LlamaMLP", "LlamaRMSNorm",
             "Linear", "LlamaRotaryEmbedding"},
        )
        self.assertEqual(
            self.workload.plan.modeling_module,
            "transformers.models.llama.modeling_llama",
        )

    def test_no_class_name_belongs_to_another_architecture(self):
        for boundary in self.workload.plan.boundaries:
            self.assertNotIn("Qwen", boundary.class_name)

    def test_the_patched_call_sites_match_the_plan(self):
        """The capture scope records these as what ran. If the RoPE entry named
        a different module than the observer patches, the manifest would
        describe a run that did not happen."""
        rope = [w for w in self.workload.function_wrappers if "rotary" in w]
        self.assertEqual(len(rope), 1)
        self.assertTrue(rope[0].startswith(self.workload.plan.modeling_module))

    def test_its_schemas_are_its_own(self):
        from evograd.bench.workloads.qwen3.declaration import WORKLOAD as QWEN

        self.assertNotEqual(self.workload.smoke_schema, QWEN.smoke_schema)
        self.assertNotEqual(self.workload.manifest_schema, QWEN.manifest_schema)
        for schema in (self.workload.smoke_schema, self.workload.manifest_schema):
            self.assertIn("llama3", schema)


class TestSnapshotState(unittest.TestCase):
    """No snapshot exists yet, and that is a state to be honest about."""

    def test_the_level1_mapping_covers_every_role_llama_presents(self):
        """An unmapped role raises during extraction rather than being dropped,
        so this is what stands between a harvest and a silent gap."""
        from evograd.bench.workloads.llama3.harvest.snapshot import LEVEL1_SOURCES

        linear = LEVEL1_SOURCES["linear_no_bias"]["component_by_role"]
        for role in ("q_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "lm_head"):
            self.assertIn(role, linear)
        self.assertIn("input_layernorm", LEVEL1_SOURCES["rmsnorm"]["component_by_role"])

    def test_every_level1_component_re_derives_from_the_published_config(self):
        """The provenance claim is mechanical: the component named must
        reproduce the dims the harvest will record."""
        from evograd.opdecl.models import LLAMA_3_8B as CONFIG

        for component in ("attn_qkv", "attn_out_proj", "attn_kv_proj",
                          "mlp_up", "mlp_down", "lm_head", "rmsnorm"):
            with self.subTest(component=component):
                dims = getattr(CONFIG, f"{component}_dims")(tokens=64)
                self.assertTrue(dims)

    def test_the_merged_projection_is_consistent_either_way(self):
        """At Llama-3-8B's widths q_proj and o_proj are both 4096->4096, so the
        harvest deduplicates them into one configuration and the extraction has
        to pick one component. This checks the choice cannot matter."""
        from evograd.opdecl.models import LLAMA_3_8B as CONFIG

        self.assertEqual(CONFIG.attn_qkv_dims(tokens=64),
                         CONFIG.attn_out_proj_dims(tokens=64))

    def test_level2_tasks_only_name_operators_that_exist(self):
        """A task pointing at an undeclared operator would produce a snapshot
        nothing can read."""
        from evograd.ops import OPS
        from evograd.bench.workloads.llama3.harvest.snapshot import TASK_SOURCES

        for name in TASK_SOURCES:
            self.assertIn(name, OPS, f"{name} has no declaration")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TestItActuallyBuilds(unittest.TestCase):
    """CPU-only, two layers, 64 tokens: enough to prove the wiring, not a run.

    Built once for the class. Shrinking the layer count barely helps here --
    Llama-3's embedding and untied lm_head are 128256x4096 each and dominate the
    construction whatever the depth -- so the saving is in not repeating it.
    """

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.common.smoke import run_smoke
        from evograd.bench.workloads.llama3.declaration import WORKLOAD

        cls.workload = WORKLOAD
        cls.report = run_smoke(WORKLOAD, tiny_spec())

    def test_the_model_builds_and_the_step_produces_finite_gradients(self):
        self.assertTrue(self.report.ok, self.report.failure)
        self.assertTrue(self.report.result["loss_is_finite"])
        self.assertTrue(self.report.result["grads_all_finite"])
        self.assertEqual(self.report.result["missing_grad_params"], [])

    def test_every_trainable_parameter_is_accounted_for(self):
        """Two layers of nine parameters, plus the embedding, the final norm and
        an untied lm_head. Qwen3 at two layers has 24, because it adds q_norm
        and k_norm per layer and ties the lm_head -- the count is a fingerprint
        of the architecture."""
        self.assertEqual(self.report.result["trainable_params"], 2 * 9 + 3)

    def test_the_report_carries_llamas_schema_and_says_it_is_not_canonical(self):
        self.assertEqual(self.report.schema_version, self.workload.smoke_schema)
        self.assertFalse(self.report.workload["canonical"])
        self.assertEqual(
            self.report.workload["canonical_workload_id"], CANONICAL.workload_id
        )


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TestItActuallyHarvests(unittest.TestCase):
    """The end the whole package exists for: a manifest, then a snapshot."""

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.common.harvest import run_harvest
        from evograd.bench.workloads.llama3.declaration import WORKLOAD

        cls.workload = WORKLOAD
        cls.manifest = run_harvest(WORKLOAD, tiny_spec())

    def test_every_mandatory_boundary_produced_events(self):
        counts = self.manifest["counts_by_task"]
        for task in self.workload.plan.mandatory_tasks:
            with self.subTest(task=task):
                self.assertGreater(counts.get(task, 0), 0)

    def test_the_capture_scope_records_llamas_call_sites(self):
        wrappers = self.manifest["capture_scope"]["mechanism"]["function_wrappers"]
        self.assertIn(
            "transformers.models.llama.modeling_llama.apply_rotary_pos_emb", wrappers
        )
        for entry in wrappers:
            self.assertNotIn("qwen3", entry)

    def test_the_manifest_declares_llamas_schema(self):
        self.assertEqual(self.manifest["schema_version"], self.workload.manifest_schema)

    def test_the_manifest_hash_is_self_consistent(self):
        from evograd.bench.workloads.common.manifest import semantic_hash

        restored = json.loads(json.dumps(self.manifest))
        self.assertEqual(semantic_hash(restored), restored["manifest_hash"])

    def test_rms_norm_deduplicates_into_one_configuration(self):
        """Llama has no per-head q/k norm, so every RMSNorm in the model is at
        the residual width and collapses to a single configuration. Qwen3 has
        three. This is the clearest structural difference between them."""
        norms = [c for c in self.manifest["configurations"] if c["task"] == "rms_norm"]
        self.assertEqual(len(norms), 1)
        self.assertEqual(
            set(norms[0]["roles"]),
            {"input_layernorm", "post_attention_layernorm", "norm"},
        )

    def test_the_snapshot_extraction_maps_every_observed_configuration(self):
        """The step that turns a run into task shapes. An unmapped role raises,
        so reaching the end is the assertion."""
        from evograd.bench.workloads.llama3.harvest.snapshot import extract

        snapshot = extract(self.manifest, layer_index=1)
        self.assertEqual(
            sorted(snapshot["level1"]),
            ["causal_gqa_attention", "cross_entropy", "linear_no_bias",
             "rmsnorm", "rope", "swiglu"],
        )
        for entry in snapshot["level1"].values():
            self.assertTrue(entry["configurations"])
            for config in entry["configurations"]:
                self.assertTrue(config["provenance"]["component"])

    def test_the_extracted_snapshot_names_llama_not_qwen(self):
        from evograd.bench.workloads.llama3.harvest.snapshot import extract

        snapshot = extract(self.manifest, layer_index=1)
        self.assertEqual(snapshot["model"]["name"], MODEL_NAME)
        self.assertIn("llama3", snapshot["schema_version"])


if __name__ == "__main__":
    unittest.main()
