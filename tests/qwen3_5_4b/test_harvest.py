"""The Qwen3.5 Level-3 smoke, the whole-model harvest, its snapshot, the layer
capture and replay, and the Level-1 bindings -- on a tiny model (CPU).

The canonical run of each is a GPU session's; these hold the machinery: every
layer is held to its block kind's mandatory boundaries, the snapshot's site
dims agree with the declared ones, the whole-model counts are observed, a
captured layer replays the full model bitwise, and the observed Level-1 cases
carry the model's own RoPE theta.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from evograd.benchmark.topdown.qwen3_5_4b.architecture import config_record
from evograd.benchmark.topdown.qwen3_5_4b.harvest import snapshot
from evograd.benchmark.topdown.qwen3_5_4b.harvest.harvest import run_harvest
from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.spec import CANONICAL

from tests.qwen3_5_4b._tiny import FULL_LAYER, GDN_LAYER, SEQ, TINY_ARCH

LAYERS = {"linear_attention": GDN_LAYER, "full_attention": FULL_LAYER}


def tiny_spec(**kwargs):
    return CANONICAL.replace(arch=TINY_ARCH, device="cpu", batch_size=1, seq_len=SEQ, **kwargs)


_MANIFEST: dict = {}


def tiny_manifest():
    if "m" not in _MANIFEST:
        _MANIFEST["m"] = json.loads(json.dumps(run_harvest(tiny_spec()), default=str))
    return _MANIFEST["m"]


class TestSpecAndSmoke(unittest.TestCase):
    def test_canonical_is_b1_t2048_and_says_why(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3 import spec as module

        self.assertEqual((CANONICAL.batch_size, CANONICAL.seq_len, CANONICAL.dtype),
                         (1, 2048, "bfloat16"))
        self.assertTrue(CANONICAL.is_canonical)
        self.assertTrue(CANONICAL.workload_id.startswith("qwen3.5-4b.train.bs1.seq2048.bf16"))
        for phrase in ("batch 1", "96 GB", "73 GiB", "global batch 2"):
            self.assertIn(phrase, module.__doc__)
        self.assertEqual(CANONICAL.weights, "random")
        self.assertEqual(CANONICAL.data, "synthetic")

    def test_the_config_carries_every_published_field(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import build_config

        config = build_config(CANONICAL)
        self.assertEqual(list(config.layer_types), list(CANONICAL.arch["layer_types"]))
        self.assertEqual(config.rope_parameters["rope_theta"], 10000000)
        self.assertEqual(config.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertEqual(config.rope_parameters["mrope_section"], [11, 11, 10])
        self.assertIs(config.attn_output_gate, True)
        self.assertIs(config.tie_word_embeddings, True)
        self.assertFalse(config.use_cache)
        self.assertEqual(config._attn_implementation, "sdpa")
        self.assertEqual(CANONICAL.arch["rope_parameters"]["rope_theta"], 10000000,
                         "Transformers' in-place RoPE normalization reached the declaration")

    def test_tiny_smoke_reports_the_pinned_dispatch(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.smoke import run_smoke

        report = run_smoke(tiny_spec())
        self.assertEqual(report.status, "ok", report.failure)
        self.assertEqual(report.effective["gated_delta_rule_dispatch"], "torch")
        self.assertEqual(report.effective["causal_conv1d_dispatch"], "torch")
        self.assertTrue(report.result["grads_all_finite"])
        self.assertEqual(report.schema_version, "evograd-qwen3_5-smoke/1")

    def test_a_non_torch_dispatch_is_refused(self):
        from evograd.benchmark.topdown.qwen3_5_4b.declaration import WORKLOAD

        effective = {"gated_delta_rule_dispatch": "fla.ops.chunk_gated_delta_rule",
                     "causal_conv1d_dispatch": "torch"}
        pinned = [p for p in WORKLOAD.check_settings({**effective, "param_dtypes": [],
                                                      "param_devices": [],
                                                      "attn_implementation": None,
                                                      "attn_implementation_per_module": [],
                                                      "use_cache": False,
                                                      "gradient_checkpointing": False,
                                                      "training_mode": True}, tiny_spec())
                  if "dispatch" in p]
        self.assertEqual(len(pinned), 1)

    def test_a_pretrained_spec_is_loaded_from_the_local_cache_only(self):
        import os
        import tempfile
        from unittest import mock

        from huggingface_hub.errors import LocalEntryNotFoundError

        from evograd.benchmark.topdown.qwen3_5_4b.levels.level3.model import build_model

        spec = tiny_spec(weights="pretrained:Qwen/Qwen3.5-4B@main")
        self.assertEqual(spec.pretrained_repo, ("Qwen/Qwen3.5-4B", "main"))
        with tempfile.TemporaryDirectory() as empty, \
                mock.patch.dict(os.environ, {"EVOGRAD_HF_CACHE": empty}):
            with self.assertRaises(LocalEntryNotFoundError):     # never fetched
                build_model(spec)


class TestHarvest(unittest.TestCase):
    def test_every_layer_holds_its_kinds_boundaries(self):
        manifest = tiny_manifest()
        counts = manifest["counts_by_task"]
        self.assertEqual(counts["decoder_layer"], 4)
        self.assertEqual(counts["gated_delta_rule"], 3)
        self.assertEqual(counts["sdpa"], 1)
        self.assertEqual(counts["l2norm"], 6)
        self.assertEqual(manifest["schema_version"], "evograd-qwen3_5-harvest/1")
        self.assertIn("transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule",
                      manifest["capture_scope"]["mechanism"]["function_wrappers"])
        self.assertTrue(any("AdamW" in item for item in manifest["capture_scope"]["not_observed"]))

    def test_a_layer_missing_a_boundary_fails_the_harvest(self):
        from evograd.benchmark.topdown.common.observe import MandatoryBoundaryError
        from evograd.benchmark.topdown.qwen3_5_4b.harvest import observe as plan

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

        events = [_Event(task, i) for i, kind in enumerate(TINY_ARCH["layer_types"])
                  for task in plan.BLOCK_TASKS[kind]]
        plan.check_layers(_Observation(events), tiny_spec())  # complete: passes
        without = [e for e in events if not (e.layer_index == GDN_LAYER and e.task == "l2norm")]
        with self.assertRaises(MandatoryBoundaryError):
            plan.check_layers(_Observation(without), tiny_spec())

    def test_snapshot_extraction_and_declared_dims(self):
        manifest = tiny_manifest()
        snap = snapshot.extract(manifest, representative_layers=LAYERS)
        self.assertEqual(snap["schema_version"], "evograd-qwen3_5-snapshot/1")
        self.assertEqual(snap["provenance"], "observed")
        self.assertEqual(snap["snapshot_hash"], snapshot.snapshot_hash(snap))
        self.assertEqual(sorted(snap["level1"]), sorted(
            ["causal_conv1d", "causal_gqa_attention", "cross_entropy", "gated_rmsnorm",
             "gdn_gating", "gemma_rmsnorm", "l2norm", "linear_no_bias", "partial_rope",
             "swiglu"]))
        record = config_record(TINY_ARCH, name="tiny")
        declared = {site: getattr(record, f"{site}_dims")(batch=1, seq=SEQ)
                    for site in snapshot.site_dims(snap)}
        self.assertEqual(snapshot.check_declared_dims(snap, declared), [])
        wrong = dict(declared, swiglu_mlp={**declared["swiglu_mlp"], "I": 1})
        self.assertTrue(snapshot.check_declared_dims(snap, wrong))

    def test_whole_model_counts_are_observed(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest as sites

        snap = snapshot.extract(tiny_manifest(), representative_layers=LAYERS)
        for site in sites.SITES:
            entry = snap["tasks"][sites.task_key(site)]["observed"]
            with self.subTest(site=site):
                self.assertEqual(entry["kind"], "observed")
                self.assertEqual(entry["whole_model_invocations"],
                                 sites.expected_model_frequency(site, TINY_ARCH))
                self.assertEqual(sites.observed_model_frequency(site, snap),
                                 entry["whole_model_invocations"])

    def test_every_level1_component_rederives(self):
        from evograd.opdecl.models import MODELS

        snap = snapshot.extract(tiny_manifest(), representative_layers=LAYERS)
        record = config_record(TINY_ARCH, name="tiny")
        for generic, entry in snap["level1"].items():
            for config in entry["configurations"]:
                component = config["provenance"]["component"]
                with self.subTest(primitive=generic, component=component):
                    builder = getattr(record, f"{component}_dims")
                    self.assertEqual(builder(**config["provenance"]["free"]), config["dims"])
        self.assertNotIn("tiny", MODELS)

    def test_no_snapshot_yet_means_no_observed_cases(self):
        """Until the GPU harvest is frozen the bindings contribute nothing, and
        the workload is registered all the same."""
        from evograd.benchmark import TASKS
        from evograd.benchmark.topdown import TOPDOWN_WORKLOADS, has_snapshot

        self.assertEqual(TOPDOWN_WORKLOADS["qwen3_5_4b"], "qwen3_5_4b")
        if not has_snapshot("qwen3_5_4b"):
            self.assertNotIn("qwen3_5_4b_observed", TASKS["partial_rope"].benchmark_suites)
        else:  # once frozen: the hash verifies and the dims are the declared ones
            from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest as sites
            from evograd.opdecl.models import QWEN3_5_4B

            snap = snapshot.load()
            declared = {site: sites.dims_for(site, QWEN3_5_4B, batch=1, seq=2048)
                        for site in sites.SITES}
            self.assertEqual(snapshot.check_declared_dims(snap, declared), [])
            self.assertIn("qwen3_5_4b_observed", TASKS["partial_rope"].benchmark_suites)


class TestCaptureAndReplay(unittest.TestCase):
    def test_capture_then_standalone_replay_is_bitwise(self):
        from evograd.benchmark.topdown.qwen3_5_4b.layer.artifact import LayerArtifact
        from evograd.benchmark.topdown.qwen3_5_4b.layer.capture import (
            run_capture, write_captures)
        from evograd.evaluation.workloads.qwen3_5_4b.layer.replay import run_replay

        spec = tiny_spec(dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            written = write_captures(run_capture(spec, manifest_path=harvest,
                                                 layers=[GDN_LAYER, FULL_LAYER]), Path(tmp))
            self.assertEqual(len(written), 4)
            for index, kind in ((GDN_LAYER, "linear_attention"), (FULL_LAYER, "full_attention")):
                artifact = LayerArtifact.load(Path(tmp) / f"layer{index}.pt")
                with self.subTest(layer=index):
                    self.assertEqual(artifact.identity["layer_kind"], kind)
                    self.assertIn("position_embeddings", artifact.payload["kwargs"])
                    report = run_replay(artifact, device="cpu", noise_repeats=0)
                    self.assertEqual(report["status"], "pass", report.get("failures"))
                    self.assertTrue(report["comparisons"]["output"]["bitwise_identical"])

    def test_a_relabelled_artifact_is_refused(self):
        from evograd.benchmark.topdown.qwen3_5_4b.layer.artifact import (
            ArtifactError, LayerArtifact)
        from evograd.benchmark.topdown.qwen3_5_4b.layer.capture import run_capture

        spec = tiny_spec(dtype="float32")
        with tempfile.TemporaryDirectory() as tmp:
            harvest = Path(tmp) / "harvest.json"
            harvest.write_text(json.dumps(run_harvest(spec), default=str))
            artifact, _ = run_capture(spec, manifest_path=harvest, layers=[GDN_LAYER])[GDN_LAYER]
            artifact.payload["identity"]["layer_index"] = 1
            path = artifact.save(Path(tmp) / "relabelled.pt")
            with self.assertRaises(ArtifactError):
                LayerArtifact.load(path)


class TestLevel1Bindings(unittest.TestCase):
    def test_ten_bindings_under_one_suite(self):
        from evograd.benchmark.core.registry import _observed_bindings
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level1.manifest import (
            COMPOSES_INTO, OBSERVED_BINDINGS)

        self.assertEqual(len(OBSERVED_BINDINGS), 10)
        self.assertEqual({b.suite for b in OBSERVED_BINDINGS}, {"qwen3_5_4b_observed"})
        self.assertEqual(set(COMPOSES_INTO), {b.task for b in OBSERVED_BINDINGS})
        self.assertEqual(_observed_bindings()[-10:], OBSERVED_BINDINGS)

    def test_partial_rope_cases_rotate_at_the_models_theta(self):
        from evograd.benchmark import TASKS
        from evograd.opdecl import Provenance, Workload
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.ops.level1.partial_rope import rotation_base

        dims = {"B": 1, "H": 2, "T": 16, "D": 32, "R": 8}
        observed = Workload(dims=dims, dtype="float32",
                            provenance=Provenance(model="qwen3_5_4b", component="partial_rope",
                                                  free={"batch": 1, "seq": 16},
                                                  source="hf_config"))
        self.assertEqual(rotation_base(observed), 1e7)
        self.assertEqual(rotation_base(Workload(dims=dims, dtype="float32")), 1e4)
        values = make_case_inputs(TASKS["partial_rope"], observed, device="cpu")
        inv_freq = 1.0 / (1e7 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        angles = torch.arange(16, dtype=torch.float32)[:, None] * inv_freq[None]
        want = torch.cat((angles, angles), dim=-1)[None].cos()
        self.assertTrue(torch.allclose(values["cos"], want))

    def test_cross_entropy_runs_on_full_width_rows(self):
        """The 248,320-wide rows of the loss: the oracle and production agree (a
        few rows, on CPU)."""
        from evograd.benchmark import TASKS
        from evograd.opdecl import Workload
        from evograd.opdecl.inputs import as_output_tuple, make_case_inputs
        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward

        op = TASKS["cross_entropy"]
        workload = Workload(dims={"rows": 4, "cols": 248320}, dtype="float32")
        values = make_case_inputs(op, workload, device="cpu")
        args = [values.get(a.name, getattr(a, "default", None)) for a in op.args]
        ref = as_output_tuple(op, resolve_forward(op)(*args))
        prod = as_output_tuple(op, resolve_runtime_forward(op)(*args))
        for a, b in zip(ref, prod):
            self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))


class TestHarvestedInvocation(unittest.TestCase):
    def test_the_descriptor_hook_on_a_tiny_block(self):
        from evograd.benchmark import TASKS
        from evograd.evaluation.workloads.qwen3_5_4b.descriptor import WORKLOAD
        from evograd.evaluation.workloads.qwen3_5_4b.level2.harvested import captured_invocation

        self.assertEqual(WORKLOAD.layer_for("full_attention"), 15)
        for task in ("qwen3_5_gdn_in_proj", "qwen3_5_gated_attention", "qwen3_5_swiglu_mlp"):
            with self.subTest(task=task):
                values, workload, label = captured_invocation(
                    TASKS[task], "cpu", arch=TINY_ARCH, layers=LAYERS, batch=1, seq=SEQ)
                self.assertIn("config-derived", label)
                self.assertEqual(workload.dims["T"], SEQ)
        self.assertIsNone(captured_invocation(TASKS["partial_rope"], "cpu"))


if __name__ == "__main__":
    unittest.main()
