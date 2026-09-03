"""The Qwen3 workload harvest: what was observed, how it deduplicates, and that
observing changes nothing.

Everything runs on the tiny CPU Qwen3 from ``test_level4_workload``. The point of
these tests is structure -- boundaries, ordering, provenance, determinism -- and
structure is the same at two layers as at twenty-eight.
"""

from __future__ import annotations

import json
import unittest

import torch

from tests.qwen3.test_level4_workload import HAVE_TRANSFORMERS, tiny_spec

if HAVE_TRANSFORMERS:
    from evograd.bench.workloads.qwen3.harvest.harvest import run_harvest
    from evograd.bench.workloads.qwen3.harvest.manifest import (
        SCHEMA_VERSION,
        deduplicate,
        semantic_hash,
        summarize,
    )
    from evograd.bench.workloads.qwen3.levels.level4.model import build_model, make_inputs, training_step
    from evograd.bench.workloads.qwen3.harvest.observe import (
        MANDATORY_TASKS,
        MandatoryBoundaryError,
        ObserverError,
        observe,
    )


def walk(node, path="$"):
    """Every leaf in a nested structure, with the path that reached it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from walk(value, f"{path}[{index}]")
    else:
        yield path, node


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class HarvestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = tiny_spec()
        cls.manifest = run_harvest(cls.spec)
        cls.events = cls.manifest["events"]
        cls.configs = cls.manifest["configurations"]
        cls.layers = cls.spec.arch["num_hidden_layers"]

    def events_for(self, task):
        return [e for e in self.events if e["task"] == task]

    def configs_for(self, task):
        return [c for c in self.configs if c["task"] == task]


class TestBoundariesObserved(HarvestFixture):
    """Every boundary the manifest is defined to contain is in it, at the
    frequency the architecture implies."""

    def test_every_mandatory_boundary_produced_events(self):
        counts = self.manifest["counts_by_task"]
        for task in MANDATORY_TASKS:
            with self.subTest(task=task):
                self.assertGreater(counts.get(task, 0), 0, counts)

    def test_per_layer_boundaries_fire_once_per_layer(self):
        counts = self.manifest["counts_by_task"]
        for task in ("decoder_layer", "attention", "mlp", "sdpa", "rope_apply", "silu"):
            with self.subTest(task=task):
                self.assertEqual(counts[task], self.layers)

    def test_linear_and_rms_norm_counts_follow_from_the_architecture(self):
        """7 Linears and 4 RMSNorms per layer, plus the lm_head and the final
        norm. Derived here, never written into the manifest."""
        counts = self.manifest["counts_by_task"]
        self.assertEqual(counts["linear"], 7 * self.layers + 1)
        self.assertEqual(counts["rms_norm"], 4 * self.layers + 1)

    def test_the_loss_boundary_fires_exactly_once(self):
        self.assertEqual(self.manifest["counts_by_task"]["causal_cross_entropy"], 1)

    def test_sdpa_records_the_attention_configuration(self):
        attrs = self.events_for("sdpa")[0]["attrs"]
        self.assertTrue(attrs["is_causal"])
        self.assertEqual(attrs["dropout_p"], 0.0)
        self.assertIn("enable_gqa", attrs)
        self.assertIn("scale", attrs)

    def test_softmax_is_not_claimed_as_an_observed_primitive(self):
        self.assertNotIn("softmax", self.manifest["counts_by_task"])
        scope = " ".join(self.manifest["capture_scope"]["not_observed"])
        self.assertIn("softmax", scope)

    def test_capture_scope_states_that_backward_ran_but_was_not_observed(self):
        scope = self.manifest["capture_scope"]
        self.assertEqual(scope["phases_observed"], ["forward"])
        self.assertTrue(scope["backward_executed"])
        self.assertFalse(scope["backward_observed"])
        self.assertTrue(all(e["phase"] == "forward" for e in self.events))


class TestEventProvenance(HarvestFixture):
    def test_ordinals_are_dense_and_ordered(self):
        ordinals = [e["ordinal"] for e in self.events]
        self.assertEqual(ordinals, sorted(ordinals))
        self.assertEqual(ordinals, list(range(len(ordinals))))

    def test_order_is_invocation_order_parent_before_child(self):
        """A decoder layer's ordinal precedes every event inside it."""
        for index in range(self.layers):
            layer = next(
                e for e in self.events_for("decoder_layer") if e["layer_index"] == index
            )
            inner = [
                e
                for e in self.events
                if e["module_path"]
                and e["module_path"].startswith(f"model.layers.{index}.")
            ]
            self.assertTrue(inner)
            self.assertTrue(all(e["ordinal"] > layer["ordinal"] for e in inner))

    def test_module_paths_and_roles_are_correct(self):
        by_path = {e["module_path"]: e for e in self.events if e["module_path"]}
        self.assertEqual(by_path["model.layers.0.self_attn.q_proj"]["role"], "q_proj")
        self.assertEqual(by_path["model.layers.0.self_attn.q_proj"]["task"], "linear")
        self.assertEqual(by_path["model.layers.0.self_attn.q_norm"]["role"], "q_norm")
        self.assertEqual(by_path["model.layers.0.self_attn.q_norm"]["task"], "rms_norm")
        self.assertEqual(by_path["model.layers.0.mlp.gate_proj"]["role"], "gate_proj")
        self.assertEqual(by_path["model.norm"]["role"], "norm")
        self.assertEqual(by_path["lm_head"]["role"], "lm_head")

    def test_layer_index_comes_from_the_module_path(self):
        for event in self.events:
            path = event["module_path"] or ""
            if path.startswith("model.layers."):
                expected = int(path.split(".")[2])
                self.assertEqual(event["layer_index"], expected, path)
        self.assertIsNone(next(e for e in self.events if e["module_path"] == "model.norm")["layer_index"])

    def test_functional_events_inherit_the_enclosing_layer(self):
        """RoPE and SDPA have no ``self``; their provenance comes from the
        module stack, so it must still name the attention module and layer."""
        for task in ("rope_apply", "sdpa"):
            for event in self.events_for(task):
                self.assertIsNotNone(event["layer_index"], event)
                self.assertEqual(
                    event["module_path"], f"model.layers.{event['layer_index']}.self_attn"
                )

    def test_every_event_carries_workload_provenance(self):
        for event in self.events:
            self.assertEqual(event["provenance"]["kind"], "observed")
            self.assertEqual(event["provenance"]["workload_id"], self.spec.workload_id)
            self.assertEqual(event["provenance"]["config_hash"], self.spec.config_hash)

    def test_parameter_metadata_is_present_where_parameters_exist(self):
        linear = next(e for e in self.events if e["module_path"] == "model.layers.0.mlp.gate_proj")
        self.assertIn("weight", linear["params"])
        self.assertEqual(
            linear["params"]["weight"]["shape"],
            [self.spec.arch["intermediate_size"], self.spec.arch["hidden_size"]],
        )
        norm = next(e for e in self.events if e["module_path"] == "model.layers.0.self_attn.q_norm")
        self.assertEqual(norm["params"]["weight"]["shape"], [self.spec.arch["head_dim"]])

    def test_tensor_metadata_carries_layout(self):
        event = self.events_for("linear")[0]
        meta = event["inputs"][0]
        for key in ("shape", "dtype", "device", "requires_grad", "stride", "contiguous", "numel"):
            self.assertIn(key, meta)
        self.assertEqual(meta["device"], "cpu")


class TestDeduplication(HarvestFixture):
    def test_frequencies_sum_to_the_event_count(self):
        self.assertEqual(sum(c["frequency"] for c in self.configs), len(self.events))
        for task, count in self.manifest["counts_by_task"].items():
            self.assertEqual(sum(c["frequency"] for c in self.configs_for(task)), count)

    def test_config_ids_are_unique(self):
        ids = [c["config_id"] for c in self.configs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_identical_layers_collapse_to_one_configuration(self):
        for task in ("decoder_layer", "attention", "mlp", "sdpa", "rope_apply", "silu"):
            with self.subTest(task=task):
                configs = self.configs_for(task)
                self.assertEqual(len(configs), 1, [c["roles"] for c in configs])
                self.assertEqual(configs[0]["frequency"], self.layers)
                self.assertEqual(
                    configs[0]["layer_indices"], list(range(self.layers)), configs[0]
                )

    def test_gate_and_up_projections_share_one_configuration(self):
        """The example from the milestone: same generic Linear task, two roles,
        and every module path kept."""
        config = next(
            c for c in self.configs_for("linear") if set(c["roles"]) == {"gate_proj", "up_proj"}
        )
        self.assertEqual(config["frequency"], 2 * self.layers)
        self.assertEqual(len(config["module_paths"]), 2 * self.layers)
        self.assertEqual(config["layer_indices"], list(range(self.layers)))
        for index in range(self.layers):
            self.assertIn(f"model.layers.{index}.mlp.gate_proj", config["module_paths"])
            self.assertIn(f"model.layers.{index}.mlp.up_proj", config["module_paths"])

    def test_residual_norms_and_the_final_norm_share_a_configuration(self):
        config = next(c for c in self.configs_for("rms_norm") if "norm" in c["roles"])
        self.assertEqual(
            set(config["roles"]), {"input_layernorm", "post_attention_layernorm", "norm"}
        )
        self.assertEqual(config["frequency"], 2 * self.layers + 1)
        # The final norm belongs to no decoder layer, and the record says so.
        self.assertIn(None, config["layer_indices"])
        self.assertEqual(config["attrs"]["normalized_size"], self.spec.arch["hidden_size"])

    def test_q_and_k_head_norms_stay_distinct(self):
        """Both normalize 128 head-dim elements, but over a different number of
        heads, so they are different kernels and must not merge."""
        q = next(c for c in self.configs_for("rms_norm") if c["roles"] == ["q_norm"])
        k = next(c for c in self.configs_for("rms_norm") if c["roles"] == ["k_norm"])
        self.assertNotEqual(q["config_id"], k["config_id"])
        for record in (q, k):
            self.assertEqual(record["attrs"]["normalized_size"], self.spec.arch["head_dim"])
            self.assertEqual(record["frequency"], self.layers)
        self.assertEqual(q["inputs"][0]["shape"][2], self.spec.arch["num_attention_heads"])
        self.assertEqual(k["inputs"][0]["shape"][2], self.spec.arch["num_key_value_heads"])

    def test_the_key_excludes_path_role_layer_and_ordinal(self):
        """Directly: two events that differ only in provenance produce one
        configuration."""
        from evograd.bench.workloads.qwen3.harvest.observe import Event

        def make(ordinal, path, role, layer):
            return Event(
                ordinal=ordinal,
                phase="forward",
                task="linear",
                module_path=path,
                module_class="Linear",
                role=role,
                layer_index=layer,
                inputs=[{"kind": "tensor", "shape": [2, 4]}],
                input_kwargs={},
                outputs=[{"kind": "tensor", "shape": [2, 8]}],
                params={},
                attrs={"in_features": 4, "out_features": 8, "bias": False},
                provenance={},
            )

        records = deduplicate(
            [make(0, "a.gate_proj", "gate_proj", 0), make(9, "b.up_proj", "up_proj", 7)],
            workload_id="w",
            config_hash="c",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].frequency, 2)
        self.assertEqual(records[0].roles, ["gate_proj", "up_proj"])
        self.assertEqual(records[0].layer_indices, [0, 7])

    def test_a_differing_attribute_splits_the_configuration(self):
        from evograd.bench.workloads.qwen3.harvest.observe import Event

        def make(out_features):
            return Event(
                ordinal=0,
                phase="forward",
                task="linear",
                module_path="p",
                module_class="Linear",
                role="r",
                layer_index=0,
                inputs=[],
                input_kwargs={},
                outputs=[],
                params={},
                attrs={"out_features": out_features},
                provenance={},
            )

        self.assertEqual(len(deduplicate([make(8), make(16)], workload_id="w", config_hash="c")), 2)

    def test_each_configuration_keeps_provenance_kind_and_workload_identity(self):
        for config in self.configs:
            self.assertEqual(config["provenance"]["kind"], "observed")
            self.assertEqual(config["provenance"]["workload_id"], self.spec.workload_id)
            self.assertEqual(config["provenance"]["config_hash"], self.spec.config_hash)


class TestManifestIntegrity(HarvestFixture):
    def test_no_tensor_survives_anywhere_in_the_manifest(self):
        offenders = [
            (path, type(value).__name__)
            for path, value in walk(self.manifest)
            if isinstance(value, torch.Tensor) or isinstance(value, torch.nn.Module)
        ]
        self.assertEqual(offenders, [])

    def test_the_manifest_is_json_serializable_without_coercion(self):
        """No ``default=`` fallback: if anything needed coercing, it was not
        plain metadata."""
        text = json.dumps(self.manifest)
        self.assertEqual(json.loads(text)["manifest_hash"], self.manifest["manifest_hash"])

    def test_json_round_trip_preserves_the_hash(self):
        restored = json.loads(json.dumps(self.manifest))
        self.assertEqual(semantic_hash(restored), restored["manifest_hash"])

    def test_no_object_addresses_leaked_into_the_manifest(self):
        for path, value in walk(self.manifest):
            if isinstance(value, str):
                self.assertNotIn("0x", value, path)
                self.assertNotIn(" object at ", value, path)

    def test_schema_and_identity_fields(self):
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.manifest["workload_id"], self.spec.workload_id)
        self.assertEqual(self.manifest["config_hash"], self.spec.config_hash)

    def test_the_tiny_harvest_is_marked_non_canonical(self):
        self.assertFalse(self.manifest["workload"]["canonical"])
        self.assertNotEqual(
            self.manifest["workload_id"], self.manifest["workload"]["canonical_workload_id"]
        )

    def test_summary_mentions_the_hash_and_the_counts(self):
        text = summarize(self.manifest)
        self.assertIn(self.manifest["manifest_hash"], text)
        self.assertIn("raw events", text)
        self.assertIn("deduplicated configurations", text)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestDeterminism(unittest.TestCase):
    def test_two_harvests_of_the_same_workload_hash_identically(self):
        spec = tiny_spec()
        first, second = run_harvest(spec), run_harvest(spec)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(first["events"], second["events"])
        self.assertEqual(first["configurations"], second["configurations"])

    def test_environment_and_diagnostics_are_outside_the_hash(self):
        spec = tiny_spec()
        manifest = run_harvest(spec)
        baseline = manifest["manifest_hash"]
        manifest["environment"]["gpu_name"] = "a different machine"
        manifest["diagnostics"]["wall_time_s"] = 999.0
        manifest["validation"]["loss"] = -1.0
        self.assertEqual(semantic_hash(manifest), baseline)

    def test_changing_structure_changes_the_hash(self):
        manifest = run_harvest(tiny_spec())
        baseline = manifest["manifest_hash"]
        manifest["events"][0]["task"] = "something_else"
        self.assertNotEqual(semantic_hash(manifest), baseline)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestObserverIsolation(unittest.TestCase):
    """The observer must be invisible once its context closes -- in the process
    it patched, and in the numbers the model produces."""

    def _originals(self):
        import transformers.loss.loss_utils as loss_utils
        import transformers.models.qwen3.modeling_qwen3 as modeling

        return {
            "rope": modeling.apply_rotary_pos_emb,
            "sdpa": torch.nn.functional.scaled_dot_product_attention,
            "loss": loss_utils.LOSS_MAPPING["ForCausalLM"],
            "flat": loss_utils.fixed_cross_entropy,
        }

    def test_everything_is_restored_after_success(self):
        before = self._originals()
        spec = tiny_spec()
        model = build_model(spec)
        ids, labels = make_inputs(spec)
        with observe(model, workload_id="w", config_hash="c") as obs:
            training_step(model, ids, labels)
        self.assertGreater(len(obs.events), 0)
        self.assertEqual(self._originals(), before)

    def test_everything_is_restored_after_a_failure(self):
        before = self._originals()
        model = build_model(tiny_spec())

        class Boom(RuntimeError):
            pass

        with self.assertRaises(Boom):
            with observe(model, workload_id="w", config_hash="c"):
                raise Boom("the observed step exploded")
        self.assertEqual(self._originals(), before)

    def test_module_hooks_are_removed(self):
        model = build_model(tiny_spec())
        with observe(model, workload_id="w", config_hash="c"):
            pass
        for name, module in model.named_modules():
            self.assertEqual(len(module._forward_hooks), 0, name)
            self.assertEqual(len(module._forward_pre_hooks), 0, name)

    def test_a_run_after_the_context_records_nothing(self):
        spec = tiny_spec()
        model = build_model(spec)
        ids, labels = make_inputs(spec)
        with observe(model, workload_id="w", config_hash="c") as obs:
            training_step(model, ids, labels)
        recorded = len(obs.events)
        model.zero_grad(set_to_none=True)
        training_step(model, ids, labels)
        self.assertEqual(len(obs.events), recorded)

    def test_nesting_is_refused_rather_than_double_counted(self):
        model = build_model(tiny_spec())
        with observe(model, workload_id="w", config_hash="c"):
            with self.assertRaises(ObserverError):
                with observe(model, workload_id="w", config_hash="c"):
                    pass
        # The refused attempt must not have damaged the outer installation.
        self.assertEqual(
            self._originals()["sdpa"], torch.nn.functional.scaled_dot_product_attention
        )

    def test_two_sequential_observations_record_the_same_amount(self):
        spec = tiny_spec()
        model = build_model(spec)
        ids, labels = make_inputs(spec)
        counts = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            with observe(model, workload_id="w", config_hash="c") as obs:
                training_step(model, ids, labels)
            counts.append(obs.counts_by_task())
        self.assertEqual(counts[0], counts[1])

    def test_observation_changes_neither_loss_nor_gradients(self):
        from evograd.bench.workloads.qwen3.levels.level4.smoke import run_smoke

        spec = tiny_spec()
        unobserved = run_smoke(spec)
        observed = run_harvest(spec)
        self.assertTrue(unobserved.ok, unobserved.failure)
        self.assertEqual(observed["validation"]["loss"], unobserved.result["loss"])
        for key in (
            "trainable_params",
            "params_with_grad",
            "missing_grad_params",
            "grads_all_finite",
        ):
            self.assertEqual(observed["validation"][key], unobserved.result[key], key)

    def test_a_missing_mandatory_boundary_is_an_error(self):
        from evograd.bench.workloads.qwen3.harvest.observe import (
            Observation,
            check_mandatory_boundaries,
        )

        empty = Observation(workload_id="w", config_hash="c")
        with self.assertRaises(MandatoryBoundaryError) as ctx:
            check_mandatory_boundaries(empty)
        self.assertIn("rope_apply", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
