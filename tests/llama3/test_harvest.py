"""The Llama-3 harvest: architecture-derived counts, deduplication, and that
observing changes nothing.

Everything runs on the tiny CPU Llama from ``test_level4_workload`` -- two
layers at the published widths, because Llama's embedding and untied
``lm_head`` are 128256x4096 each and dominate construction whatever the depth.
The point is structure -- boundaries, ordering, provenance, determinism -- and
structure is the same at two layers as at thirty-two.

**What is deliberately not here.** ``test_level4_workload.TestItActuallyHarvests``
already pins the manifest's self-consistent hash, the capture scope naming
Llama's call sites, the single ``rms_norm`` configuration, and the whole
snapshot extraction. This file covers what nothing else does: counts re-derived
from the architecture, event provenance, the deduplication *key*, manifest
integrity, determinism, and the observer's isolation from the process it
patched.

**What is Llama's own.** Two dedup outcomes that Qwen3 cannot produce. Every
RMSNorm here is at the residual width, so ``rms_norm`` is ``2 * layers + 1``
invocations rather than Qwen3's ``4 * layers + 1`` -- there is no per-head q/k
normalization to observe. And ``n_heads * head_dim == hidden``, so ``q_proj``
and ``o_proj`` are both 4096->4096 and collapse into one configuration, where
Qwen3's fan-out keeps them apart.
"""

from __future__ import annotations

import json
import unittest

import torch

from tests.llama3.test_level4_workload import HAVE_TRANSFORMERS, tiny_spec

if HAVE_TRANSFORMERS:
    from evograd.benchmark.topdown.common.manifest import semantic_hash
    from evograd.benchmark.topdown.llama3_8b.harvest.harvest import run_harvest
    from evograd.benchmark.topdown.llama3_8b.harvest.observe import (
        MANDATORY_TASKS,
        MandatoryBoundaryError,
        Observation,
        ObserverError,
        check_mandatory_boundaries,
        observe,
    )
    from evograd.benchmark.topdown.llama3_8b.levels.level4.model import (
        build_model,
        make_inputs,
        training_step,
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
    """One harvest, shared. Building the model is the expensive part."""

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


class TestCountsFollowFromTheArchitecture(HarvestFixture):
    """Every count is re-derived here, never read out of the manifest twice."""

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

    def test_seven_linears_per_layer_plus_the_untied_lm_head(self):
        """q, k, v, o, gate, up, down -- and Llama does not tie its output
        embedding, so ``lm_head`` is a Linear of its own."""
        self.assertEqual(
            self.manifest["counts_by_task"]["linear"], 7 * self.layers + 1
        )

    def test_two_rms_norms_per_layer_plus_the_final_norm(self):
        """The clearest count-level difference from Qwen3, which has four per
        layer: Llama has no per-head query or key normalization at all."""
        self.assertEqual(
            self.manifest["counts_by_task"]["rms_norm"], 2 * self.layers + 1
        )

    def test_no_per_head_norm_is_observed_anywhere(self):
        roles = {e["role"] for e in self.events_for("rms_norm")}
        self.assertEqual(
            roles, {"input_layernorm", "post_attention_layernorm", "norm"}
        )
        self.assertNotIn("q_norm", roles)
        self.assertNotIn("k_norm", roles)

    def test_the_loss_boundary_fires_exactly_once(self):
        self.assertEqual(self.manifest["counts_by_task"]["causal_cross_entropy"], 1)

    def test_sdpa_records_the_attention_configuration(self):
        attrs = self.events_for("sdpa")[0]["attrs"]
        self.assertTrue(attrs["is_causal"])
        self.assertEqual(attrs["dropout_p"], 0.0)
        self.assertIn("enable_gqa", attrs)
        self.assertIn("scale", attrs)

    def test_only_the_forward_phase_is_observed(self):
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
        layer_events = self.events_for("decoder_layer")
        for index in range(self.layers):
            layer = next(e for e in layer_events if e["layer_index"] == index)
            inside = [
                e for e in self.events
                if e["layer_index"] == index and e["task"] != "decoder_layer"
            ]
            self.assertTrue(inside, f"layer {index} recorded nothing inside it")
            for event in inside:
                self.assertGreater(event["ordinal"], layer["ordinal"])

    def test_layer_index_comes_from_the_module_path(self):
        for event in self.events:
            path = event["module_path"]
            if path and ".layers." in path:
                index = int(path.split(".layers.")[1].split(".")[0])
                with self.subTest(path=path):
                    self.assertEqual(event["layer_index"], index)

    def test_functional_events_inherit_the_enclosing_layer(self):
        """``apply_rotary_pos_emb`` and SDPA are module-level functions with no
        path of their own; without the enclosing layer they would be
        unattributable."""
        for task in ("rope_apply", "sdpa", "silu"):
            with self.subTest(task=task):
                indices = sorted(e["layer_index"] for e in self.events_for(task))
                self.assertEqual(indices, list(range(self.layers)))

    def test_the_final_norm_is_outside_every_layer(self):
        final = [e for e in self.events_for("rms_norm") if e["role"] == "norm"]
        self.assertEqual(len(final), 1)
        self.assertIsNone(final[0]["layer_index"])

    def test_every_event_carries_workload_provenance(self):
        for event in self.events:
            self.assertEqual(
                event["provenance"]["workload_id"], self.manifest["workload_id"]
            )
            self.assertEqual(
                event["provenance"]["config_hash"], self.manifest["config_hash"]
            )

    def test_parameter_metadata_is_present_where_parameters_exist(self):
        for event in self.events_for("linear"):
            self.assertIn("weight", event["params"])
        for event in self.events_for("sdpa"):
            self.assertEqual(event["params"], {})

    def test_tensor_metadata_carries_layout(self):
        entry = self.events_for("linear")[0]["inputs"][0]
        for key in ("shape", "dtype"):
            self.assertIn(key, entry)


class TestDeduplication(HarvestFixture):
    def test_frequencies_sum_to_the_event_count(self):
        self.assertEqual(
            sum(c["frequency"] for c in self.configs), len(self.events)
        )

    def test_config_ids_are_unique(self):
        ids = [c["config_id"] for c in self.configs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_identical_layers_collapse_to_one_configuration(self):
        """Two layers, one configuration each: the layers are structurally the
        same, so keeping them apart would inflate every downstream shape count.
        """
        for task in ("decoder_layer", "attention", "mlp", "sdpa"):
            with self.subTest(task=task):
                configs = self.configs_for(task)
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0]["frequency"], self.layers)

    def test_gate_and_up_projections_share_one_configuration(self):
        """Same in-width, same out-width, same module class: one shape, two
        call sites, and the roles record both."""
        merged = [
            c for c in self.configs_for("linear")
            if set(c["roles"]) >= {"gate_proj", "up_proj"}
        ]
        self.assertEqual(len(merged), 1, [c["roles"] for c in self.configs_for("linear")])

    def test_the_square_projections_merge_because_llama_does_not_fan_out(self):
        """``n_heads * head_dim == hidden`` for Llama-3, so ``q_proj`` and
        ``o_proj`` are both 4096->4096 and are one configuration. Qwen3's 2048
        against 1024 keeps its equivalents apart, which is why this assertion
        has no counterpart there."""
        merged = [
            c for c in self.configs_for("linear")
            if set(c["roles"]) >= {"q_proj", "o_proj"}
        ]
        self.assertEqual(len(merged), 1, [c["roles"] for c in self.configs_for("linear")])

    def test_the_key_excludes_path_role_layer_and_ordinal(self):
        """Two invocations that differ only in *where* they happened are the
        same configuration; the record keeps the difference as provenance."""
        for config in self.configs:
            if config["frequency"] > 1:
                with self.subTest(config=config["config_id"]):
                    self.assertGreater(
                        len(set(config["module_paths"])
                            | set(config["roles"])
                            | {str(i) for i in config["layer_indices"]}),
                        1,
                    )

    def test_each_configuration_keeps_its_workload_identity(self):
        for config in self.configs:
            self.assertEqual(
                config["provenance"]["workload_id"], self.manifest["workload_id"]
            )


class TestManifestIntegrity(HarvestFixture):
    def test_no_tensor_survives_anywhere_in_the_manifest(self):
        for path, leaf in walk(self.manifest):
            self.assertNotIsInstance(leaf, torch.Tensor, path)

    def test_the_manifest_is_json_serializable_without_coercion(self):
        json.dumps(self.manifest)

    def test_json_round_trip_preserves_the_hash(self):
        restored = json.loads(json.dumps(self.manifest))
        self.assertEqual(semantic_hash(restored), self.manifest["manifest_hash"])

    def test_no_object_addresses_leaked_into_the_manifest(self):
        for path, leaf in walk(self.manifest):
            if isinstance(leaf, str):
                self.assertNotIn("0x", leaf, path)
                self.assertNotIn(" object at ", leaf, path)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestDeterminism(unittest.TestCase):
    def test_two_harvests_of_the_same_workload_hash_identically(self):
        spec = tiny_spec()
        first, second = run_harvest(spec), run_harvest(spec)
        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(first["events"], second["events"])
        self.assertEqual(first["configurations"], second["configurations"])

    def test_environment_and_diagnostics_are_outside_the_hash(self):
        """A harvest run on another machine describes the same architecture, so
        the machine must not enter the identity."""
        manifest = run_harvest(tiny_spec())
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
    it patched, and in the numbers the model produces.

    Llama's plan names ``transformers.models.llama.modeling_llama`` as the
    module holding ``apply_rotary_pos_emb``, so restoration is checked against
    *that* module. The shared observer is exercised by Qwen3's equivalent; what
    is untested without this is Llama's own plan driving it.
    """

    def _originals(self):
        import transformers.loss.loss_utils as loss_utils
        import transformers.models.llama.modeling_llama as modeling

        return {
            "rope": modeling.apply_rotary_pos_emb,
            "sdpa": torch.nn.functional.scaled_dot_product_attention,
            "loss": loss_utils.LOSS_MAPPING["ForCausalLM"],
            # Patched only where the installed Transformers has it, so absence
            # is recorded rather than raising here.
            "flat": getattr(loss_utils, "fixed_cross_entropy", None),
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
            with self.subTest(module=name):
                self.assertEqual(len(module._forward_hooks), 0)
                self.assertEqual(len(module._forward_pre_hooks), 0)

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
            self._originals()["sdpa"],
            torch.nn.functional.scaled_dot_product_attention,
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
        """The load-bearing one. A harvest that perturbed the step would
        describe a model nobody trains."""
        from evograd.benchmark.topdown.llama3_8b.levels.level4.smoke import run_smoke

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
            with self.subTest(key=key):
                self.assertEqual(
                    observed["validation"][key], unobserved.result[key]
                )

    def test_a_missing_mandatory_boundary_is_an_error(self):
        """Exporting a manifest with a boundary missing would silently drop
        every invocation of it from the snapshot."""
        empty = Observation(workload_id="w", config_hash="c")
        with self.assertRaises(MandatoryBoundaryError) as ctx:
            check_mandatory_boundaries(empty)
        self.assertIn("rope_apply", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
