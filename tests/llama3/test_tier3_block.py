"""Llama-3's decoder layer through the block-scope executor, on a two-layer CPU model.

The tiny model stands in for the 1B one exactly as it does for the model
scope: the *wiring* is what is pinned -- which modules an adapter reaches, that
each site runs once, that a single-site and a combined patch both install and
count, that structural identity is bitwise, that the captured artifact's
numbers are reproduced, that a wrong backward is rejected before anything is
timed, and that the config-derived case needs no snapshot and no full model.
The adapter is Llama's own (``qkv_rope``, no per-head norm, Llama's rotary
tables with rope scaling); the executor, gate, CLI and report are the shared
ones and are not changed by this workload.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.llama3.test_level4_workload import HAVE_TRANSFORMERS, tiny_spec

if HAVE_TRANSFORMERS:
    from evograd.benchmark import TASKS
    from evograd.evaluation.tier3 import block as B
    from evograd.evaluation.tier3.gate import block as G
    from evograd.evaluation.tier3.gate.faults import scale_grad
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch, restrict
    from evograd.evaluation.tier3.workloads.llama3_2_1b import block as lblock
    from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import (
        SITE_ATTENTION, SITE_MLP, SITE_QKV, SITE_RESIDUAL,
        structural_identity_kernels, bound_pair_identity_kernels,
    )

_skip = unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")

#: Llama's own shape family: 4:1 grouped-query attention and
#: ``n_heads * head_dim == hidden``.
TINY = {
    "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16,
    "vocab_size": 256, "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1,
}
TINY_LAYER = 1


def _config_adapter(seed=0, **overrides):
    kwargs = dict(batch=2, seq=16, layer_index=TINY_LAYER, dtype="float32", seed=seed,
                  arch_overrides=TINY)
    kwargs.update(overrides)
    return lblock.config_case(**kwargs)


def _measure(adapter, name, kernels, *, role, policy=None, reference=None, **options):
    defaults = dict(device="cpu", ops=TASKS, policy=policy, reference=reference, verify=False,
                    purity=False, noise_repeats=1, warmup=0, samples=1, blocks=1)
    defaults.update(options)
    return B.measure_block_one(adapter, name, kernels, role=role, **defaults)


@_skip
class TestConfigCaseNeedsNoSnapshot(unittest.TestCase):
    def test_it_builds_from_the_architecture_alone(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.prepare import live_model_instances

        adapter = _config_adapter()
        case = adapter.case
        self.assertEqual(case.architecture, "llama_3_2_1b")
        self.assertEqual(case.source_mode, "config")
        self.assertEqual(case.dims["H"], 64)
        self.assertEqual(case.sites[SITE_QKV], "llama3_qkv_rope")
        for field in ("weights_hash", "inputs_hash", "cotangents_hash"):
            self.assertIsNotNone(getattr(case, field))
        # Other test modules may keep a full model alive on the heap; what is
        # pinned is that building the block adds none.
        before = live_model_instances()
        built = adapter.build(device="cpu")
        live = live_model_instances()
        self.assertEqual(live.get("LlamaForCausalLM", 0), before.get("LlamaForCausalLM", 0))
        self.assertEqual(live.get("LlamaModel", 0), before.get("LlamaModel", 0))
        self.assertEqual(live.get("LlamaDecoderLayer", 0), before.get("LlamaDecoderLayer", 0) + 1)
        # q, k, v, o, gate, up, down, input_layernorm, post_attention_layernorm:
        # Llama has no per-head norm weights, so nine rather than Qwen3's eleven.
        self.assertEqual(len(built.parameters), 9)
        inv = adapter.prepare(built, device="cpu")
        self.assertEqual(inv.outputs, ("result",))
        self.assertEqual(inv.differentiable_inputs, ("args[0]",))
        self.assertIsNone(inv.kwargs["attention_mask"])
        self.assertEqual(tuple(inv.kwargs["position_embeddings"][0].shape), (1, 16, 16))

    def test_identity_is_deterministic_and_seed_sensitive(self):
        a, b, c = _config_adapter(), _config_adapter(), _config_adapter(seed=1)
        self.assertEqual(a.case.case_hash, b.case.case_hash)
        self.assertEqual(a.case.weights_hash, b.case.weights_hash)
        self.assertNotEqual(a.case.case_hash, c.case.case_hash)
        self.assertNotEqual(a.case.weights_hash, c.case.weights_hash)
        self.assertNotEqual(a.case.inputs_hash, c.case.inputs_hash)
        self.assertEqual(a.case.case_id, "llama_3_2_1b/decoder_layer@1/config/B2.T16.float32")

    def test_a_layer_outside_the_architecture_is_refused(self):
        with self.assertRaises(ValueError):
            _config_adapter(layer_index=5)

    def test_from_args_builds_the_config_case_the_cli_asked_for(self):
        from evograd.evaluation.tier3 import cli

        args = cli._parser().parse_args(
            ["--scope", "block", "--model", "llama_3_2_1b", "--device", "cpu", "--dtype", "float32",
             "--batch", "1", "--tokens", "8", "--layer-index", str(TINY_LAYER),
             *[f"--arch-override={k}={v}" for k, v in TINY.items()]])
        cli.check_options(args)
        adapter = lblock.from_args(args)
        self.assertEqual(adapter.layer_index, TINY_LAYER)
        self.assertEqual(adapter.case.dims["B"], 1)
        self.assertEqual(adapter.case.dims["T"], 8)
        self.assertEqual(adapter.case.source["arch_overrides"], TINY)


@_skip
class TestSitesInsideTheBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = _config_adapter()
        cls.registry = cls.adapter.registry
        cls.reference, _, _ = B.native_reference(cls.adapter, device="cpu")

    def _run(self, kernels):
        built = self.adapter.build(device="cpu")
        installed = self.adapter.install(built, kernels)
        inv = self.adapter.prepare(built, device="cpu")
        reset = lambda: (built.module.zero_grad(set_to_none=True), installed.counters.reset())  # noqa: E731
        result = B.run_vjp(built.module, inv, built.parameters, reset=reset)
        return result, installed, built

    def test_a_single_site_patch_installs_one_adapter_and_runs_once(self):
        kernels = restrict(structural_identity_kernels(self.registry), (SITE_MLP,))
        result, installed, built = self._run(kernels)
        self.assertEqual(installed.counters.snapshot(), {SITE_MLP: 1})
        self.assertEqual(installed.provenance.paths, {SITE_MLP: ("mlp",)})
        self.assertEqual(self.adapter.expected_invocations(kernels), {SITE_MLP: 1})
        self.assertTrue(torch.equal(result.outputs["result"], self.reference.outputs["result"]))

    def test_patching_qkv_makes_attention_live_too(self):
        kernels = restrict(structural_identity_kernels(self.registry), (SITE_QKV,))
        result, installed, _ = self._run(kernels)
        self.assertEqual(installed.counters.snapshot(), {SITE_QKV: 1, SITE_ATTENTION: 1})
        self.assertEqual(self.adapter.expected_invocations(kernels), {SITE_QKV: 1, SITE_ATTENTION: 1})

    def test_the_combined_patch_is_bitwise_and_keeps_the_parameter_list(self):
        kernels = structural_identity_kernels(self.registry)
        result, installed, built = self._run(kernels)
        self.assertEqual(installed.counters.snapshot(),
                         {SITE_QKV: 1, SITE_ATTENTION: 1, SITE_MLP: 1, SITE_RESIDUAL: 1})
        self.assertEqual(installed.provenance.paths[SITE_RESIDUAL], ("post_attention_layernorm",))
        self.assertEqual(list(built.parameters), list(self.reference.param_grads))
        self.assertTrue(torch.equal(result.outputs["result"], self.reference.outputs["result"]))
        self.assertTrue(torch.equal(result.input_grads["args[0]"], self.reference.input_grads["args[0]"]))
        for name, grad in self.reference.param_grads.items():
            self.assertTrue(torch.equal(result.param_grads[name], grad), name)

    def test_the_block_never_builds_the_model_loop(self):
        kernels = structural_identity_kernels(self.registry)
        _, _, built = self._run(kernels)
        self.assertFalse(hasattr(built.module, "model"))
        self.assertIsNone(getattr(built.module, "_evograd_llama3_state", {}).get("carrier"))

    def test_the_residual_site_receives_the_models_epsilon(self):
        """The site hands the kernel ``variance_epsilon`` -- the model's 1e-5, not
        the declaration's generic 1e-6 -- and the block adapter keeps that."""
        seen = {}

        def spy(x, r, weight, eps):
            seen["eps"] = eps
            summed = r + x
            return torch.nn.functional.rms_norm(summed, (summed.shape[-1],), weight=weight, eps=eps), summed

        kernels = patch(KernelSet(registry=self.registry), SITE_RESIDUAL, spy,
                        source=KernelSource(SITE_RESIDUAL, "llama3_residual_rmsnorm", None, "test:spy"))
        self._run(kernels)
        self.assertEqual(seen["eps"], self.adapter.arch["rms_norm_eps"])
        self.assertEqual(seen["eps"], 1e-5)


@_skip
class TestProvidersThroughTheExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = _config_adapter()
        cls.registry = cls.adapter.registry
        mlp_only = restrict(structural_identity_kernels(cls.registry), (SITE_MLP,))
        cls.policy = G.calibrate_policy(
            cls.adapter, {G.patch_key(mlp_only): mlp_only,
                          G.patch_key(structural_identity_kernels(cls.registry)):
                          structural_identity_kernels(cls.registry)},
            ops=TASKS, device="cpu", noise_repeats=1)
        cls.reference, _, _ = B.native_reference(cls.adapter, device="cpu")

    def test_the_policy_holds_both_patch_sets_from_controls_only(self):
        entries = self.policy["entries"]
        self.assertEqual(sorted(entries), [
            "attention+qkv_rope+residual_rmsnorm+swiglu_mlp", "swiglu_mlp"])
        for entry in entries.values():
            self.assertTrue(entry["controls_ok"], entry["controls"])
            self.assertTrue(entry["controls"]["structural_identity"]["bitwise"])
            self.assertIn("bound_pair_identity", entry["controls"])

    def test_native_is_the_reference_and_passes_its_own_gate(self):
        entry = _measure(self.adapter, "native", KernelSet(registry=self.registry), role="reference",
                         policy=self.policy)
        self.assertTrue(entry["ok"], entry.get("error"))
        block = entry["correctness"]["block"]
        self.assertTrue(block["repeatability"]["ok"])
        self.assertTrue(block["capture_agreement"]["skipped"])
        self.assertIsNotNone(entry["latency"]["forward_backward_ms"])
        self.assertIsNone(entry["memory"]["execution_peak_bytes"])
        self.assertGreater(entry["memory"]["saved_state_bytes"], 0)

    def test_single_and_combined_structural_patches_pass_with_local_checks(self):
        for sites in ((SITE_MLP,), None):
            kernels = structural_identity_kernels(self.registry)
            if sites:
                kernels = restrict(kernels, sites)
            with self.subTest(sites=sites or "all"):
                entry = _measure(self.adapter, "structural", kernels, role="control",
                                 policy=self.policy, reference=self.reference)
                self.assertTrue(entry["ok"], entry.get("error"))
                self.assertTrue(entry["correctness"]["live_boundary"]["ok"])
                self.assertEqual(entry["correctness"]["live_boundary"]["checked_invocations"],
                                 len(self.adapter.expected_invocations(kernels)))
                self.assertTrue(entry["correctness"]["patch_coverage"]["ok"])
                self.assertTrue(entry["correctness"]["block"]["comparison"]["bitwise"])

    def test_the_bound_pair_route_passes_the_calibrated_envelope(self):
        kernels = bound_pair_identity_kernels(TASKS, (SITE_MLP,), registry=self.registry)
        entry = _measure(self.adapter, "bound", kernels, role="control",
                         policy=self.policy, reference=self.reference)
        self.assertTrue(entry["ok"], entry.get("error"))
        self.assertEqual(entry["correctness"]["block"]["policy"]["patch_key"], "swiglu_mlp")

    def test_a_wrong_backward_is_rejected_before_timing(self):
        def faulty(x, gate_w, up_w, down_w):
            import torch.nn.functional as F
            out = F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)
            return scale_grad(out, 1.5)

        kernels = patch(KernelSet(registry=self.registry), SITE_MLP, faulty,
                        source=KernelSource(SITE_MLP, "llama3_swiglu_mlp", None, "fault:grad_scale@1.5"))
        entry = _measure(self.adapter, "faulty", kernels, role="candidate",
                         policy=self.policy, reference=self.reference)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "block_correctness")
        self.assertIsNone(entry["latency"])

    def test_a_provider_whose_patch_set_was_not_calibrated_is_reported_alone(self):
        kernels = restrict(structural_identity_kernels(self.registry), (SITE_RESIDUAL,))
        entry = _measure(self.adapter, "residual_only", kernels, role="control",
                         policy=self.policy, reference=self.reference)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "no_policy")


@_skip
class TestCapturedCase(unittest.TestCase):
    """The tiny artifact, captured the way layer8.pt is, replayed as a block."""

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.llama3_2_1b.harvest.harvest import run_harvest
        from evograd.benchmark.topdown.llama3_2_1b.harvest.manifest import write_manifest
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.capture import run_capture

        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        spec = tiny_spec()
        manifest = run_harvest(spec)
        manifest_path = write_manifest(manifest, root / "harvest.json")
        artifact, _ = run_capture(spec, manifest_path=manifest_path, layer_index=TINY_LAYER,
                                  expect_workload_id=spec.workload_id,
                                  expect_manifest_hash=manifest["manifest_hash"])
        cls.artifact_path = artifact.save(root / "layer.pt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_native_block_reproduces_the_capture(self):
        adapter = lblock.captured_case(self.artifact_path, canonical=False)
        self.assertEqual(adapter.case.source_mode, "captured")
        self.assertFalse(adapter.case.source["canonical_identity_verified"])
        self.assertEqual(adapter.case.block_index, TINY_LAYER)
        entry = _measure(adapter, "native", KernelSet(registry=adapter.registry), role="reference")
        self.assertTrue(entry["ok"], entry.get("error"))
        agreement = entry["correctness"]["block"]["capture_agreement"]
        self.assertTrue(agreement["ok"], agreement)
        self.assertTrue(agreement["outputs_and_input_gradients"]["out:result"]["within_tolerance"])
        self.assertEqual(agreement["param_grads"]["count"], 9)

    def test_the_wrong_layer_index_is_refused(self):
        with self.assertRaises(B.InvocationError):
            lblock.captured_case(self.artifact_path, layer_index=0, canonical=False)

    def test_a_canonical_claim_is_checked_against_the_snapshot(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level3.artifact import ArtifactError

        with self.assertRaises(ArtifactError):
            lblock.captured_case(self.artifact_path, canonical=True)


@_skip
class TestCli(unittest.TestCase):
    OVERRIDES = [f"--arch-override={k}={v}" for k, v in TINY.items()]

    def _argv(self, *extra):
        return ["--scope", "block", "--model", "llama_3_2_1b", "--device", "cpu", "--dtype", "float32",
                "--batch", "2", "--tokens", "16", "--layer-index", str(TINY_LAYER), *self.OVERRIDES,
                "--baseline", "none", "--no-purity", *extra]

    def test_llama_declares_the_block_scope(self):
        from evograd.evaluation.tier3.workloads import tier3_adapter

        self.assertIn("block", tier3_adapter("llama_3_2_1b").scopes)

    def test_model_only_options_are_refused_by_name_in_block_scope(self):
        from evograd.evaluation.tier3 import cli

        for flag in (["--loss-steps", "3"], ["--layers", "1"]):
            with self.subTest(flag=flag[0]):
                with self.assertRaises(SystemExit):
                    cli.check_options(cli._parser().parse_args(self._argv(*flag)))

    def test_an_in_process_run_writes_a_readable_block_report(self):
        from evograd.evaluation.tier3 import cli
        from evograd.evaluation.tier3.report import identify_report, provider_rows

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "block.json"
            rc = cli.main(self._argv("--structural-identity", "--sites", SITE_MLP, "--no-isolate",
                                     "--samples", "1", "--blocks", "1", "--warmup", "0",
                                     "--noise-repeats", "1", "--out", str(out)))
            self.assertEqual(rc, 0)
            report = json.loads(out.read_text())
            self.assertEqual(identify_report(report).execution_scope, "block")
            rows = {r.provider: r for r in provider_rows(report)}
            self.assertTrue(rows["native"].ok, report["providers"]["native"].get("error"))
            self.assertTrue(rows["structural_identity"].ok,
                            report["providers"]["structural_identity"].get("error"))
            self.assertEqual(report["providers"]["structural_identity"]["patched"], [SITE_MLP])
            self.assertTrue(Path(report["policy_file"]).is_file())
            self.assertEqual(report["case"]["architecture"], "llama_3_2_1b")
            self.assertEqual(report["case"]["source_mode"], "config")
            self.assertEqual(report["case"]["source"]["arch_overrides"]["hidden_size"], 64)


if __name__ == "__main__":
    unittest.main()
