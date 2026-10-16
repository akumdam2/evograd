"""The OLMoE Level-3 smoke, the whole-model harvest, its snapshot, the layer
capture and replay, and the Level-1 bindings -- on a tiny model (CPU).

The canonical run of each is a GPU session's; these hold the machinery: every
layer is held to the decoder block's mandatory boundaries (attention, RoPE,
SDPA, the MoE block, its router, its experts, their SiLU), the snapshot's site
dims agree with the declared ones, the whole-model counts are observed, a
captured layer replays the full model bitwise, and the experts kernel is read
back and held to the spec.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.benchmark.topdown.olmoe_1b_7b.architecture import config_record
from evograd.benchmark.topdown.olmoe_1b_7b.harvest import snapshot
from evograd.benchmark.topdown.olmoe_1b_7b.harvest.harvest import run_harvest
from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.spec import CANONICAL

from tests.olmoe_1b_7b._tiny import LAYER, LAYERS, SEQ, TINY_ARCH


def tiny_spec(**kwargs):
    return CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=SEQ, **kwargs)


_MANIFEST: dict = {}


def tiny_manifest():
    if "m" not in _MANIFEST:
        _MANIFEST["m"] = json.loads(json.dumps(run_harvest(tiny_spec()), default=str))
    return _MANIFEST["m"]


class TestSpecAndSmoke(unittest.TestCase):
    def test_canonical_is_b2_t2048_grouped_mm_and_says_why(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3 import spec as module

        self.assertEqual((CANONICAL.batch_size, CANONICAL.seq_len, CANONICAL.dtype,
                          CANONICAL.experts_implementation),
                         (2, 2048, "bfloat16", "grouped_mm"))
        self.assertTrue(CANONICAL.is_canonical)
        self.assertTrue(CANONICAL.workload_id.startswith("olmoe-1b-7b.train.bs2.seq2048.bf16"))
        self.assertIn(".grouped_mm.", CANONICAL.workload_id)
        for phrase in ("batch 2", "96 GB", "grouped_mm", "13.8"):
            self.assertIn(phrase, module.__doc__)
        self.assertEqual(CANONICAL.to_dict()["experts_implementation"], "grouped_mm")

    def test_the_experts_kernel_is_part_of_the_identity(self):
        eager = CANONICAL.replace(experts_implementation="eager")
        self.assertNotEqual(eager.workload_hash, CANONICAL.workload_hash)
        self.assertEqual(eager.config_hash, CANONICAL.config_hash, "the architecture is the same")
        self.assertIn(".eager.", eager.workload_id)
        self.assertFalse(eager.is_canonical)
        with self.assertRaises(Exception):
            CANONICAL.replace(experts_implementation="batched_mm")
        with self.assertRaises(Exception):
            CANONICAL.replace(arch={"clip_qkv": 8.0})
        with self.assertRaises(Exception):
            CANONICAL.replace(arch={"num_experts_per_tok": 65})

    def test_the_config_carries_every_field_and_the_experts_kernel(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.model import build_config

        config = build_config(CANONICAL)
        self.assertEqual(config.num_experts, 64)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertIs(config.output_router_logits, True)
        self.assertIs(config.tie_word_embeddings, False)
        self.assertFalse(config.use_cache)
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(config._experts_implementation, "grouped_mm")
        self.assertEqual(config.rope_parameters["rope_theta"], 10000.0)
        self.assertEqual(CANONICAL.arch["rope_theta"], 10000.0,
                         "Transformers' in-place RoPE normalization reached the declaration")
        self.assertEqual(build_config(CANONICAL.replace(experts_implementation="eager"))
                         ._experts_implementation, "eager")

    def test_tiny_smoke_reads_back_the_experts_kernel(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level3.smoke import run_smoke

        for experts in ("grouped_mm", "eager"):
            with self.subTest(experts=experts):
                report = run_smoke(tiny_spec(experts_implementation=experts))
                self.assertEqual(report.status, "ok", report.failure)
                self.assertEqual(report.effective["experts_implementation"], experts)
                self.assertEqual(report.effective["experts_implementation_per_module"], [experts])
                self.assertTrue(report.result["grads_all_finite"])
                self.assertEqual(report.schema_version, "evograd-olmoe-smoke/1")

    def test_the_other_experts_kernel_is_refused_by_the_read_back(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.declaration import WORKLOAD

        effective = {"experts_implementation": "eager", "experts_implementation_per_module": ["eager"],
                     "param_dtypes": ["torch.bfloat16"], "param_devices": ["cpu"],
                     "attn_implementation": "sdpa", "attn_implementation_per_module": ["sdpa"],
                     "use_cache": False, "gradient_checkpointing": False, "training_mode": True}
        problems = WORKLOAD.check_settings(effective, tiny_spec())
        self.assertEqual(len([p for p in problems if "experts" in p]), 1, problems)
        self.assertEqual(WORKLOAD.check_settings(
            {**effective, "experts_implementation": "grouped_mm",
             "experts_implementation_per_module": ["grouped_mm"]}, tiny_spec()), [])


class TestHarvest(unittest.TestCase):
    def test_every_layer_holds_the_decoder_boundaries(self):
        manifest = tiny_manifest()
        counts = manifest["counts_by_task"]
        self.assertEqual(counts["decoder_layer"], 4)
        self.assertEqual(counts["moe"], 4)
        self.assertEqual(counts["router"], 4)
        self.assertEqual(counts["experts"], 4)
        self.assertEqual(counts["silu"], 4, "one packed activation per layer under grouped_mm")
        self.assertEqual(counts["sdpa"], 4)
        self.assertEqual(counts["rope_apply"], 4)
        self.assertNotIn("mlp", counts)
        self.assertEqual(manifest["schema_version"], "evograd-olmoe-harvest/1")
        self.assertIn("transformers.models.olmoe.modeling_olmoe.apply_rotary_pos_emb",
                      manifest["capture_scope"]["mechanism"]["function_wrappers"])
        self.assertTrue(any("AdamW" in item for item in manifest["capture_scope"]["not_observed"]))
        self.assertTrue(any("router" in item for item in manifest["capture_scope"]["not_observed"]))

    def test_the_eager_loop_activates_once_per_hit_expert(self):
        manifest = json.loads(json.dumps(run_harvest(tiny_spec(experts_implementation="eager")),
                                         default=str))
        self.assertEqual(manifest["counts_by_task"]["experts"], 4)
        self.assertGreater(manifest["counts_by_task"]["silu"], 4)

    def test_a_layer_missing_a_boundary_fails_the_harvest(self):
        from evograd.benchmark.topdown.common.observe import MandatoryBoundaryError
        from evograd.benchmark.topdown.olmoe_1b_7b.harvest import observe as plan

        class _Observation:
            def __init__(self, events):
                self.events = events

            def counts_by_task(self):
                out = {}
                for e in self.events:
                    out[e.task] = out.get(e.task, 0) + 1
                return out

        class _Event:
            def __init__(self, task, layer):
                self.task, self.layer_index = task, layer

        events = [_Event(task, i) for i in range(TINY_ARCH["num_hidden_layers"])
                  for task in plan.LAYER_TASKS]
        plan.check_layers(_Observation(events), tiny_spec())  # complete: passes
        without = [e for e in events if not (e.layer_index == LAYER and e.task == "router")]
        with self.assertRaises(MandatoryBoundaryError):
            plan.check_layers(_Observation(without), tiny_spec())
        extra = events + [_Event("experts", LAYER)]
        with self.assertRaises(MandatoryBoundaryError):
            plan.check_layers(_Observation(extra), tiny_spec())

    def test_snapshot_extraction_and_declared_dims(self):
        manifest = tiny_manifest()
        snap = snapshot.extract(manifest, representative_layers=LAYERS)
        self.assertEqual(snap["schema_version"], "evograd-olmoe-snapshot/1")
        self.assertEqual(snap["provenance"], "observed")
        self.assertEqual(snap["snapshot_hash"], snapshot.snapshot_hash(snap))
        self.assertEqual(sorted(snap["level1"]), sorted(
            ["causal_gqa_attention", "cross_entropy", "linear_no_bias", "rmsnorm", "rope",
             "swiglu"]))
        self.assertEqual(snap["model"]["num_experts"], 4)
        record = config_record(TINY_ARCH, name="tiny")
        declared = {site: getattr(record, f"{site}_dims")(batch=1, seq=SEQ)
                    for site in snapshot.site_dims(snap)}
        self.assertEqual(snapshot.check_declared_dims(snap, declared), [])
        wrong = dict(declared, moe_experts={**declared["moe_experts"], "K": 1})
        self.assertTrue(snapshot.check_declared_dims(snap, wrong))

    def test_whole_model_counts_are_observed(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest as sites

        snap = snapshot.extract(tiny_manifest(), representative_layers=LAYERS)
        for site in sites.SITES:
            entry = snap["tasks"][sites.task_key(site)]["observed"]
            with self.subTest(site=site):
                self.assertEqual(entry["kind"], "observed")
                self.assertEqual(entry["whole_model_invocations"], 4)
                self.assertEqual(entry["whole_model_invocations"],
                                 sites.expected_model_frequency(site, TINY_ARCH))
                self.assertEqual(sites.observed_model_frequency(site, snap), 4)

    def test_every_level1_component_rederives(self):
        from evograd.opdecl.models import MODELS

        snap = snapshot.extract(tiny_manifest(), representative_layers=LAYERS)
        record = config_record(TINY_ARCH, name="tiny")
        seen = set()
        for generic, entry in snap["level1"].items():
            for config in entry["configurations"]:
                component = config["provenance"]["component"]
                seen.add(component)
                with self.subTest(primitive=generic, component=component):
                    builder = getattr(record, f"{component}_dims")
                    self.assertEqual(builder(**config["provenance"]["free"]), config["dims"])
        self.assertIn("moe_activation", seen)
        activation = next(c for c in snap["level1"]["swiglu"]["configurations"])
        self.assertEqual(activation["dims"], {"rows": SEQ * 2, "cols": 32})
        self.assertNotIn("tiny", MODELS)

    def test_no_snapshot_yet_means_no_observed_cases(self):
        """Until the GPU harvest is frozen the bindings contribute nothing, and
        the workload is registered all the same."""
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown import TOPDOWN_WORKLOADS, has_snapshot

        self.assertEqual(TOPDOWN_WORKLOADS["olmoe_1b_7b"], "olmoe_1b_7b")
        if not has_snapshot("olmoe_1b_7b"):
            self.assertNotIn("olmoe_1b_7b_observed", TASKS["swiglu"].benchmark_suites)
        else:  # once frozen: the hash verifies and the dims are the declared ones
            from evograd.benchmark.topdown.olmoe_1b_7b.levels.level2 import manifest as sites
            from evograd.opdecl.models import OLMOE_1B_7B

            snap = snapshot.load()
            declared = {site: sites.dims_for(site, OLMOE_1B_7B, batch=2, seq=2048)
                        for site in sites.SITES}
            self.assertEqual(snapshot.check_declared_dims(snap, declared), [])
            self.assertIn("olmoe_1b_7b_observed", TASKS["swiglu"].benchmark_suites)


class TestCaptureAndReplay(unittest.TestCase):
    def test_capture_then_standalone_replay_is_bitwise(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.layer.artifact import LayerArtifact
        from evograd.benchmark.topdown.olmoe_1b_7b.layer.capture import (
            run_capture, write_captures)
        from evograd.evaluation.workloads.olmoe_1b_7b.layer.replay import run_replay

        spec = tiny_spec(dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            written = write_captures(run_capture(spec, manifest_path=harvest, layers=[LAYER]),
                                     Path(tmp))
            self.assertEqual(len(written), 2)
            artifact = LayerArtifact.load(Path(tmp) / f"layer{LAYER}.pt")
            self.assertEqual(artifact.identity["layer_kind"], "decoder_layer")
            self.assertEqual(artifact.identity["experts_implementation"], "grouped_mm")
            self.assertIn("position_embeddings", artifact.payload["kwargs"])
            self.assertIn("mlp.experts.gate_up_proj", artifact.payload["state_dict"])
            report = run_replay(artifact, device="cpu", noise_repeats=0)
            self.assertEqual(report["status"], "pass", report.get("failures"))
            self.assertTrue(report["comparisons"]["output"]["bitwise_identical"])

    def test_a_relabelled_artifact_is_refused(self):
        from evograd.benchmark.topdown.olmoe_1b_7b.layer.artifact import (
            ArtifactError, LayerArtifact)
        from evograd.benchmark.topdown.olmoe_1b_7b.layer.capture import run_capture

        spec = tiny_spec(dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            artifact, _ = run_capture(spec, manifest_path=harvest, layers=[LAYER])[LAYER]
            artifact.payload["identity"]["experts_implementation"] = "eager"
            path = artifact.save(Path(tmp) / "relabelled.pt")
            with self.assertRaises(ArtifactError):
                LayerArtifact.load(path)


class TestLevel1Bindings(unittest.TestCase):
    def test_six_bindings_under_one_suite(self):
        from evograd.benchmark.core.registry import _observed_bindings
        from evograd.benchmark.topdown.olmoe_1b_7b.levels.level1.manifest import (
            COMPOSES_INTO, OBSERVED_BINDINGS)

        self.assertEqual(len(OBSERVED_BINDINGS), 6)
        self.assertEqual({b.suite for b in OBSERVED_BINDINGS}, {"olmoe_1b_7b_observed"})
        self.assertEqual(set(COMPOSES_INTO), {b.task for b in OBSERVED_BINDINGS})
        everything = _observed_bindings()
        start = everything.index(OBSERVED_BINDINGS[0])
        self.assertEqual(everything[start:start + 6], OBSERVED_BINDINGS)
        self.assertNotIn("linear", [b.task for b in OBSERVED_BINDINGS],
                         "the expert GEMMs are not nn.Linear modules")


class TestHarvestedInvocation(unittest.TestCase):
    def test_the_descriptor_hook_on_a_tiny_layer(self):
        from evograd.benchmark import TASKS
        from evograd.evaluation.workloads.olmoe_1b_7b.descriptor import WORKLOAD
        from evograd.evaluation.workloads.olmoe_1b_7b.level2.harvested import captured_invocation

        self.assertEqual(WORKLOAD.layer_for(), 8)
        self.assertEqual(WORKLOAD.artifact_path.name, "layer8.pt")
        for task in ("olmoe_qkv_norm_rope", "olmoe_moe_experts"):
            with self.subTest(task=task):
                values, workload, label = captured_invocation(
                    TASKS[task], "cpu", arch=TINY_ARCH, layers=LAYERS, batch=1, seq=SEQ)
                self.assertIn("config-derived", label)
                self.assertIn("grouped_mm", label)
                self.assertEqual(workload.dims["T"], SEQ)
        self.assertIsNone(captured_invocation(TASKS["rope"], "cpu"))


if __name__ == "__main__":
    unittest.main()
