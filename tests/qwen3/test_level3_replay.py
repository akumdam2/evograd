"""Capturing one decoder layer from the canonical run, and replaying it alone.

The tiny CPU Qwen3 stands in for the 0.6B model: what these tests check is that
the captured call is the observed one, that the artifact is complete and
self-describing, that a replay builds exactly one layer, and that it reproduces
the full model's numbers. None of that depends on the model being large.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.qwen3.test_level4_workload import HAVE_TRANSFORMERS, tiny_spec

if HAVE_TRANSFORMERS:
    from evograd.bench.workloads.qwen3.levels.level3.artifact import (
        SCHEMA_VERSION,
        ArtifactError,
        LayerArtifact,
        content_hash,
        describe,
    )
    from evograd.bench.workloads.qwen3.levels.level3.capture import (
        CaptureError,
        capture_decoder_layer,
        load_manifest,
        run_capture,
        select_layer_event,
    )
    from evograd.bench.workloads.qwen3.harvest.harvest import run_harvest
    from evograd.bench.workloads.qwen3.harvest.manifest import write_manifest
    from evograd.bench.workloads.qwen3.levels.level4.model import build_model, make_inputs, training_step
    from evograd.bench.workloads.qwen3.levels.level3.artifact import artifact_hash, load_canonical
    from evograd.bench.workloads.qwen3.levels.level3.replay import (
        BF16_EPS,
        BF16_UNIT_ROUNDOFF,
        FORWARD_TOL,
        GRADIENT_TOL,
        REPORT_SCHEMA,
        compare_tensors,
        run_replay,
        validate_noise_repeats,
    )

#: The tiny model has two layers; 1 is the deeper one, the analogue of 14.
TINY_LAYER = 1


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class CaptureFixture(unittest.TestCase):
    """One harvest and one capture, shared -- both are real runs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.spec = tiny_spec()
        cls.manifest = run_harvest(cls.spec)
        cls.manifest_path = write_manifest(cls.manifest, root / "harvest.json")
        cls.artifact, cls.metadata = run_capture(
            cls.spec,
            manifest_path=cls.manifest_path,
            layer_index=TINY_LAYER,
            expect_workload_id=cls.spec.workload_id,
            expect_manifest_hash=cls.manifest["manifest_hash"],
        )
        cls.artifact_path = cls.artifact.save(root / "layer.pt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestLayerSelection(CaptureFixture):
    """The layer is chosen from the manifest, so it cannot be one that never ran."""

    def test_the_selected_event_is_a_real_observed_invocation(self):
        event = select_layer_event(self.manifest, TINY_LAYER)
        self.assertEqual(event["task"], "decoder_layer")
        self.assertEqual(event["layer_index"], TINY_LAYER)
        self.assertEqual(event["module_path"], f"model.layers.{TINY_LAYER}")
        self.assertIn(event, self.manifest["events"])

    def test_an_absent_layer_is_rejected_and_names_what_exists(self):
        with self.assertRaises(CaptureError) as ctx:
            select_layer_event(self.manifest, 14)
        message = str(ctx.exception)
        self.assertIn("14", message)
        self.assertIn("observed indices", message)

    def test_a_wrong_workload_identity_is_rejected(self):
        with self.assertRaises(CaptureError) as ctx:
            select_layer_event(self.manifest, TINY_LAYER, expect_workload_id="not-this-one")
        self.assertIn("not-this-one", str(ctx.exception))

    def test_a_wrong_manifest_hash_is_rejected(self):
        with self.assertRaises(CaptureError) as ctx:
            select_layer_event(self.manifest, TINY_LAYER, expect_manifest_hash="0" * 64)
        self.assertIn("manifest hash", str(ctx.exception))

    def test_an_edited_manifest_is_rejected_before_anything_runs(self):
        edited = json.loads(self.manifest_path.read_text())
        edited["events"][0]["task"] = "tampered"
        path = self.manifest_path.with_name("tampered.json")
        path.write_text(json.dumps(edited))
        with self.assertRaises(CaptureError) as ctx:
            load_manifest(path)
        self.assertIn("does not match its own content", str(ctx.exception))

    def test_capturing_against_a_manifest_for_another_workload_is_refused(self):
        other = json.loads(self.manifest_path.read_text())
        with self.assertRaises(CaptureError):
            run_capture(
                self.spec.replace(seed=99),
                manifest_path=self.manifest_path,
                layer_index=TINY_LAYER,
            )
        self.assertEqual(other["workload_id"], self.spec.workload_id)


class TestCapturedArguments(CaptureFixture):
    """Exactly what Transformers passed, with its structure intact."""

    def test_the_captured_signature_matches_the_observed_event(self):
        event = select_layer_event(self.manifest, TINY_LAYER)
        captured = [describe(v) for v in self.artifact.payload["args"]]
        for got, observed in zip(captured, event["inputs"]):
            self.assertEqual(got["shape"], observed["shape"])
            self.assertEqual(got["dtype"], observed["dtype"])
        self.assertEqual(
            sorted(self.artifact.payload["kwargs"]), sorted(event["input_kwargs"])
        )

    def test_every_argument_transformers_passes_is_present(self):
        kwargs = self.artifact.payload["kwargs"]
        for name in (
            "attention_mask",
            "position_embeddings",
            "position_ids",
            "past_key_values",
            "use_cache",
        ):
            self.assertIn(name, kwargs)

    def test_an_absent_attention_mask_stays_absent(self):
        """``attention_mask=None`` is what sends SDPA down its causal path; a
        format that dropped it would replay a different kernel."""
        self.assertIsNone(self.artifact.payload["kwargs"]["attention_mask"])
        self.assertIsNone(self.artifact.payload["kwargs"]["past_key_values"])
        self.assertIs(self.artifact.payload["kwargs"]["use_cache"], False)

    def test_rotary_embeddings_keep_their_tuple_structure(self):
        pe = self.artifact.payload["kwargs"]["position_embeddings"]
        self.assertIsInstance(pe, tuple)
        self.assertEqual(len(pe), 2)
        for tensor in pe:
            self.assertTrue(torch.is_tensor(tensor))
            self.assertEqual(tensor.shape[-1], self.spec.arch["head_dim"])

    def test_gradients_of_input_and_output_were_captured(self):
        payload = self.artifact.payload
        for key in ("output", "grad_output", "grad_input"):
            self.assertTrue(torch.is_tensor(payload[key]), key)
            self.assertEqual(list(payload[key].shape), list(payload["args"][0].shape))
            self.assertTrue(torch.isfinite(payload[key]).all(), key)
        self.assertGreater(float(payload["grad_output"].abs().max()), 0.0)

    def test_state_and_gradients_cover_every_layer_parameter(self):
        payload = self.artifact.payload
        self.assertEqual(sorted(payload["state_dict"]), sorted(payload["param_grads"]))
        self.assertEqual(
            len(payload["state_dict"]), self.metadata["parameters"]["count"]
        )
        self.assertGreater(self.metadata["parameters"]["count"], 0)

    def test_nothing_captured_holds_a_live_graph_reference(self):
        def walk(value):
            if torch.is_tensor(value):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)

        for key in ("args", "kwargs", "output", "grad_output", "grad_input", "state_dict", "param_grads"):
            for tensor in walk(self.artifact.payload[key]):
                self.assertEqual(tensor.device.type, "cpu", key)
                self.assertFalse(tensor.requires_grad, key)
                self.assertIsNone(tensor.grad_fn, key)


class TestArtifactIntegrity(CaptureFixture):
    def test_content_hash_is_deterministic(self):
        self.assertEqual(
            content_hash(self.artifact.payload), content_hash(self.artifact.payload)
        )
        self.assertEqual(
            self.artifact.payload["content_hash"], content_hash(self.artifact.payload)
        )

    def test_round_trip_through_disk_preserves_the_hash(self):
        loaded = LayerArtifact.load(self.artifact_path)
        self.assertEqual(loaded.payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded.verify_content(), self.artifact.payload["content_hash"])
        for key in ("output", "grad_output", "grad_input"):
            self.assertTrue(torch.equal(loaded.payload[key], self.artifact.payload[key]))

    def test_a_modified_artifact_is_detected(self):
        loaded = LayerArtifact.load(self.artifact_path)
        loaded.payload["grad_output"] = loaded.payload["grad_output"] + 1
        with self.assertRaises(ArtifactError) as ctx:
            loaded.verify_content()
        self.assertIn("content hash mismatch", str(ctx.exception))

    def test_identity_is_checked_field_by_field(self):
        loaded = LayerArtifact.load(self.artifact_path)
        loaded.verify_identity(
            workload_id=self.spec.workload_id,
            config_hash=self.spec.config_hash,
            manifest_hash=self.manifest["manifest_hash"],
            layer_index=TINY_LAYER,
        )
        for field, bad in (
            ("workload_id", "wrong"),
            ("config_hash", "wrong"),
            ("manifest_hash", "wrong"),
            ("layer_index", 99),
        ):
            with self.subTest(field=field), self.assertRaises(ArtifactError):
                loaded.verify_identity(**{field: bad})

    def test_provenance_points_at_the_source_manifest(self):
        identity = self.artifact.identity
        self.assertEqual(identity["manifest_hash"], self.manifest["manifest_hash"])
        self.assertEqual(identity["workload_id"], self.manifest["workload_id"])
        self.assertEqual(identity["config_hash"], self.manifest["config_hash"])
        self.assertEqual(identity["module_path"], f"model.layers.{TINY_LAYER}")
        self.assertEqual(identity["provenance_kind"], "captured")
        event = select_layer_event(self.manifest, TINY_LAYER)
        self.assertEqual(identity["event_ordinal"], event["ordinal"])

    def test_metadata_serializes(self):
        payload = json.loads(json.dumps(self.metadata, default=str))
        for key in (
            "schema_version",
            "identity",
            "content_hash",
            "signature",
            "parameters",
            "environment",
            "effective",
        ):
            self.assertIn(key, payload)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestCaptureHooks(unittest.TestCase):
    def _model(self):
        spec = tiny_spec()
        model = build_model(spec)
        ids, labels = make_inputs(spec)
        return spec, model, ids, labels

    def _hook_counts(self, model, path):
        layer = model.get_submodule(path)
        return len(layer._forward_hooks), len(layer._forward_pre_hooks)

    def test_hooks_are_removed_after_success(self):
        spec, model, ids, labels = self._model()
        path = f"model.layers.{TINY_LAYER}"
        with capture_decoder_layer(model, path) as capture:
            training_step(model, ids, labels)
        self.assertIsNotNone(capture.output)
        self.assertEqual(self._hook_counts(model, path), (0, 0))
        self.assertEqual(capture.handles, [])

    def test_hooks_are_removed_after_a_failure(self):
        spec, model, ids, labels = self._model()
        path = f"model.layers.{TINY_LAYER}"

        class Boom(RuntimeError):
            pass

        with self.assertRaises(Boom):
            with capture_decoder_layer(model, path):
                raise Boom("the captured step exploded")
        self.assertEqual(self._hook_counts(model, path), (0, 0))

    def test_a_run_after_the_context_captures_nothing_new(self):
        spec, model, ids, labels = self._model()
        path = f"model.layers.{TINY_LAYER}"
        with capture_decoder_layer(model, path) as capture:
            training_step(model, ids, labels)
        calls = capture.forward_calls
        model.zero_grad(set_to_none=True)
        training_step(model, ids, labels)
        self.assertEqual(capture.forward_calls, calls)

    def test_a_second_forward_inside_one_capture_is_refused(self):
        spec, model, ids, labels = self._model()
        path = f"model.layers.{TINY_LAYER}"
        with self.assertRaises(CaptureError):
            with capture_decoder_layer(model, path):
                model(input_ids=ids, labels=labels, use_cache=False)
                model(input_ids=ids, labels=labels, use_cache=False)
        self.assertEqual(self._hook_counts(model, path), (0, 0))


class TestReplay(CaptureFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report = run_replay(
            LayerArtifact.load(cls.artifact_path), device="cpu", noise_repeats=4
        )

    def test_the_replay_passes(self):
        self.assertEqual(self.report["status"], "pass", self.report["failures"])
        self.assertEqual(self.report["failures"], [])

    def test_only_one_decoder_layer_was_constructed(self):
        instances = self.report["construction"]["live_instances"]
        self.assertEqual(instances["Qwen3DecoderLayer"], 1, instances)
        self.assertEqual(instances["Qwen3Attention"], 1, instances)
        self.assertEqual(instances["Qwen3ForCausalLM"], 0, instances)
        self.assertEqual(instances["Qwen3Model"], 0, instances)
        self.assertEqual(self.report["construction"]["module_class"], "Qwen3DecoderLayer")

    def test_one_layer_is_a_small_fraction_of_the_model(self):
        """A replay that had quietly built the whole model would show the whole
        model's parameter count."""
        one_layer = self.report["construction"]["parameter_elements"]
        whole = self.metadata["full_model_validation"]["trainable_elements"]
        self.assertLess(one_layer, whole)
        self.assertEqual(one_layer, self.metadata["parameters"]["elements"])

    def test_forward_output_agrees(self):
        record = self.report["comparisons"]["output"]
        self.assertTrue(record["within_tolerance"], record)
        self.assertTrue(record["shape_match"] and record["dtype_match"])
        self.assertTrue(record["stride_match"])
        self.assertLessEqual(record["max_rel_err_vs_scale"], FORWARD_TOL)

    def test_input_gradient_agrees(self):
        record = self.report["comparisons"]["grad_input"]
        self.assertTrue(record["within_tolerance"], record)
        self.assertLessEqual(record["max_rel_err_vs_scale"], GRADIENT_TOL)

    def test_every_parameter_gradient_agrees(self):
        per_param = self.report["comparisons"]["param_grads"]
        self.assertEqual(sorted(per_param), sorted(self.artifact.payload["param_grads"]))
        for name, record in per_param.items():
            with self.subTest(param=name):
                self.assertTrue(record["within_tolerance"], record)
                self.assertTrue(record["actual_all_finite"])
                self.assertLessEqual(record["max_rel_err_vs_scale"], GRADIENT_TOL)
        self.assertEqual(self.report["summary"]["missing_param_grads"], [])
        self.assertEqual(self.report["summary"]["non_finite_param_grads"], [])

    def test_the_noise_floor_is_measured_and_reported(self):
        noise = self.report["noise_floor"]
        self.assertEqual(noise["repeats"], 4)
        self.assertTrue(noise["measured"])
        for key in ("output", "grad_input", "param_grads_max"):
            self.assertIn(key, noise)
            self.assertGreaterEqual(noise[key], 0.0)

    def test_tolerances_are_stated_in_the_report(self):
        """BF16 keeps a 7-bit explicit mantissa: the spacing near 1.0 is 2^-7 and
        the unit roundoff is half that. The report must name them correctly."""
        tolerances = self.report["tolerances"]
        self.assertEqual(tolerances["bf16_eps"], 2.0**-7)
        self.assertEqual(tolerances["bf16_unit_roundoff"], 2.0**-8)
        self.assertEqual(BF16_EPS, 2.0**-7)
        self.assertEqual(BF16_UNIT_ROUNDOFF, 2.0**-8)
        self.assertEqual(tolerances["forward"], FORWARD_TOL)
        self.assertEqual(tolerances["gradient"], GRADIENT_TOL)
        self.assertIn("unit roundoff", tolerances["forward_meaning"])
        self.assertIn("spacing", tolerances["gradient_meaning"])
        self.assertIn("max|a-b|", tolerances["metric"])

    def test_the_report_serializes_and_carries_provenance(self):
        payload = json.loads(json.dumps(self.report, default=str))
        self.assertEqual(payload["schema_version"], REPORT_SCHEMA)
        self.assertEqual(payload["identity"]["manifest_hash"], self.manifest["manifest_hash"])
        self.assertEqual(payload["identity"]["layer_index"], TINY_LAYER)
        self.assertEqual(
            payload["artifact"]["content_hash"], self.artifact.payload["content_hash"]
        )
        self.assertIn("forward_backward_wall_time_s", payload["diagnostics"])

    def test_a_corrupted_artifact_fails_the_replay(self):
        loaded = LayerArtifact.load(self.artifact_path)
        loaded.payload["state_dict"]["mlp.down_proj.weight"] += 1.0
        with self.assertRaises(ArtifactError):
            run_replay(loaded, device="cpu", noise_repeats=0)

    def test_a_perturbed_weight_is_detected_numerically(self):
        """The negative control. Everything above reports zero error, which is
        only meaningful if a wrong replay would report a non-zero one -- so
        perturb one weight, re-hash so the integrity check passes, and require
        the comparison to catch it."""
        loaded = LayerArtifact.load(self.artifact_path)
        weight = loaded.payload["state_dict"]["mlp.down_proj.weight"]
        loaded.payload["state_dict"]["mlp.down_proj.weight"] = weight * 1.05
        loaded.payload["content_hash"] = content_hash(loaded.payload)
        loaded.payload["artifact_hash"] = artifact_hash(loaded.payload)
        report = run_replay(loaded, device="cpu", noise_repeats=0)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["failures"])
        self.assertGreater(report["summary"]["output_max_rel_err_vs_scale"], FORWARD_TOL)
        self.assertFalse(report["comparisons"]["output"]["bitwise_identical"])


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestComparisonVerdict(unittest.TestCase):
    """The verdict must fail for every way two tensors can disagree.

    A comparison that only ever reports agreement is worse than none, because
    everything downstream is then justified by a check that cannot fail.
    """

    def test_a_zero_reference_with_a_nonzero_replay_fails(self):
        """The failure this class exists for: normalizing by a zero scale used
        to yield a relative error of 0.0 -- a perfect score for the worst
        possible result."""
        expected = torch.zeros(4, 8)
        actual = torch.full((4, 8), 0.5)
        record = compare_tensors(actual, expected, GRADIENT_TOL)
        self.assertFalse(record["within_tolerance"])
        self.assertTrue(record["zero_reference_mismatch"])
        self.assertIsNone(record["max_rel_err_vs_scale"])
        self.assertEqual(record["max_abs_err"], 0.5)

    def test_two_zero_tensors_agree_exactly(self):
        record = compare_tensors(torch.zeros(4, 8), torch.zeros(4, 8), GRADIENT_TOL)
        self.assertTrue(record["within_tolerance"])
        self.assertFalse(record["zero_reference_mismatch"])
        self.assertEqual(record["max_rel_err_vs_scale"], 0.0)

    def test_a_stride_mismatch_fails_even_with_identical_values(self):
        expected = torch.randn(4, 8)
        actual = expected.t().contiguous().t()  # same values, transposed strides
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(actual.stride(), expected.stride())
        record = compare_tensors(actual, expected, GRADIENT_TOL)
        self.assertFalse(record["stride_match"])
        self.assertEqual(record["max_abs_err"], 0.0)
        self.assertFalse(record["within_tolerance"])

    def test_a_dtype_mismatch_fails(self):
        expected = torch.zeros(4, dtype=torch.bfloat16)
        record = compare_tensors(torch.zeros(4, dtype=torch.float32), expected, GRADIENT_TOL)
        self.assertFalse(record["dtype_match"])
        self.assertFalse(record["within_tolerance"])

    def test_a_non_finite_reference_fails(self):
        """A capture holding a NaN is unusable. A replay that faithfully
        reproduced the NaN would otherwise be scored as agreement."""
        expected = torch.tensor([1.0, float("nan"), 3.0])
        record = compare_tensors(expected.clone(), expected, GRADIENT_TOL)
        self.assertFalse(record["expected_all_finite"])
        self.assertFalse(record["within_tolerance"])

    def test_a_non_finite_replay_fails(self):
        expected = torch.tensor([1.0, 2.0, 3.0])
        actual = torch.tensor([1.0, float("inf"), 3.0])
        record = compare_tensors(actual, expected, GRADIENT_TOL)
        self.assertFalse(record["actual_all_finite"])
        self.assertFalse(record["within_tolerance"])

    def test_a_shape_mismatch_fails_before_anything_is_computed(self):
        record = compare_tensors(torch.zeros(4), torch.zeros(5), GRADIENT_TOL)
        self.assertFalse(record["shape_match"])
        self.assertFalse(record["within_tolerance"])

    def test_agreement_within_tolerance_still_passes(self):
        expected = torch.randn(64, 64)
        actual = expected + expected.abs().max() * (GRADIENT_TOL / 4)
        record = compare_tensors(actual, expected, GRADIENT_TOL)
        self.assertTrue(record["within_tolerance"], record)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestNoiseRepeatValidation(unittest.TestCase):
    def test_valid_values(self):
        for value in (0, 2, 3, 16):
            self.assertEqual(validate_noise_repeats(value), value)

    def test_one_repeat_is_refused_because_it_measures_nothing(self):
        with self.assertRaises(ValueError) as ctx:
            validate_noise_repeats(1)
        self.assertIn("nothing to be compared with", str(ctx.exception))

    def test_negative_and_non_integer_values_are_refused(self):
        for value in (-1, -5, 2.5, "3", None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_noise_repeats(value)

    def test_the_cli_refuses_an_invalid_count(self):
        import contextlib
        import io

        from evograd.bench.workloads.qwen3.levels.level3.replay import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            main(["--artifact", "unused.pt", "--noise-repeats", "1"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("noise_repeats=1", stderr.getvalue())


class TestIdentityBoundHash(CaptureFixture):
    """Content and identity are hashed separately, and both are checked."""

    def test_both_hashes_are_stored_and_verified_together(self):
        loaded = LayerArtifact.load(self.artifact_path)
        hashes = loaded.verify()
        self.assertEqual(hashes["content_hash"], self.artifact.payload["content_hash"])
        self.assertEqual(hashes["artifact_hash"], self.artifact.payload["artifact_hash"])
        self.assertNotEqual(hashes["content_hash"], hashes["artifact_hash"])

    def test_a_tampered_identity_is_caught_although_the_content_is_intact(self):
        """The failure the identity hash exists for: a correct capture wearing
        someone else's label."""
        loaded = LayerArtifact.load(self.artifact_path, verify=False)
        loaded.payload["identity"]["layer_index"] = 99
        loaded.verify_content()  # the numbers are untouched
        with self.assertRaises(ArtifactError) as ctx:
            loaded.verify_artifact()
        self.assertIn("identity hash mismatch", str(ctx.exception))

    def test_every_identity_field_is_bound(self):
        for field, value in (
            ("workload_id", "other"),
            ("config_hash", "other"),
            ("manifest_hash", "other"),
            ("layer_index", 0),
            ("module_path", "model.layers.0"),
            ("event_ordinal", 1),
            ("module_class", "Other"),
            ("provenance_kind", "invented"),
        ):
            with self.subTest(field=field):
                loaded = LayerArtifact.load(self.artifact_path, verify=False)
                loaded.payload["identity"][field] = value
                with self.assertRaises(ArtifactError):
                    loaded.verify_artifact()

    def test_a_missing_identity_field_is_refused(self):
        loaded = LayerArtifact.load(self.artifact_path, verify=False)
        del loaded.payload["identity"]["manifest_hash"]
        with self.assertRaises(ArtifactError) as ctx:
            artifact_hash(loaded.payload)
        self.assertIn("manifest_hash", str(ctx.exception))

    def test_loading_verifies_by_default(self):
        tampered = self.artifact_path.with_name("tampered.pt")
        payload = dict(self.artifact.payload)
        payload["identity"] = {**payload["identity"], "layer_index": 99}
        torch.save(payload, tampered)
        with self.assertRaises(ArtifactError):
            LayerArtifact.load(tampered)

    def test_a_canonical_consumer_cannot_skip_validation(self):
        """``load_canonical`` takes no argument that disables a check, and the
        tiny artifact is not the canonical one, so it must be refused."""
        import inspect

        signature = inspect.signature(load_canonical)
        self.assertNotIn("verify", signature.parameters)
        with self.assertRaises(ArtifactError) as ctx:
            load_canonical(self.artifact_path)
        self.assertIn("workload_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
