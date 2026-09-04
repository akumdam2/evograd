"""The first Level-2 task derived from an observed workload.

Two halves. The first needs nothing but the tracked snapshot -- no GPU, no
Transformers, no ``results/`` directory -- because that is the whole point of
freezing a snapshot: a declaration whose shapes came from a real training step
must still import and be checkable on a machine that has never run one. The
second extracts a real ``Qwen3MLP`` invocation from a tiny replay and checks the
declaration's reference against it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evograd.bench.workloads.qwen3.harvest import snapshot as snapshot_module
from evograd.opdecl.inputs import make_case_inputs
from evograd.ops import OPS, get_op
from evograd.ops.level2.qwen3_swiglu_mlp import (
    FREQUENCY,
    HARVEST,
    PROVENANCE_CHAIN,
)

from tests.qwen3.test_level4_workload import HAVE_TRANSFORMERS, tiny_spec

REPO_ROOT = Path(__file__).resolve().parents[2]

if HAVE_TRANSFORMERS:
    from evograd.bench.workloads.qwen3.levels.level3.artifact import ArtifactError
    from evograd.bench.workloads.qwen3.levels.level3.capture import run_capture
    from evograd.bench.workloads.qwen3.harvest.harvest import run_harvest
    from evograd.bench.workloads.qwen3.harvest.manifest import write_manifest
    from evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp import (
        CONTENT_KEYS,
        IDENTITY_KEYS,
        MlpExtractionError,
        check_provenance,
        declaration_problems,
        derive_mlp_invocation,
        run_verify,
    )


class TestSnapshot(unittest.TestCase):
    """The tracked extract, without which none of this is importable offline."""

    def test_the_snapshot_loads_and_its_hash_verifies(self):
        payload = snapshot_module.load()
        self.assertEqual(payload["schema_version"], snapshot_module.SCHEMA_VERSION)
        self.assertEqual(
            payload["snapshot_hash"], snapshot_module.snapshot_hash(payload)
        )

    def test_the_snapshot_carries_what_a_task_needs(self):
        payload = snapshot_module.load()
        for key in (
            "workload_id",
            "config_hash",
            "manifest_hash",
            "model",
            "batch_size",
            "seq_len",
            "dtype",
            "representative_layer",
            "tasks",
        ):
            self.assertIn(key, payload)
        task = payload["tasks"]["qwen3_swiglu_mlp"]
        for key in (
            "config_id",
            "frequency",
            "module_paths",
            "layer_indices",
            "input_shapes",
            "output_shapes",
            "dtype",
            "attrs",
            "supporting",
        ):
            self.assertIn(key, task)

    def test_an_edited_snapshot_is_rejected(self):
        payload = snapshot_module.load()
        payload["tasks"]["qwen3_swiglu_mlp"]["frequency"] = 27
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.json"
            path.write_text(json.dumps(payload))
            with self.assertRaises(snapshot_module.SnapshotError) as ctx:
                snapshot_module.load(path)
            self.assertIn("snapshot hash mismatch", str(ctx.exception))

    def test_the_snapshot_module_imports_nothing_heavy(self):
        """Same check ``test_provenance`` makes of the model registry: the module
        a declaration reads at import time must not pull in torch, Transformers,
        or anything from the capture machinery.

        Both halves are checked. Reading a snapshot lives in the shared
        ``workloads.common.snapshot``; deriving one from a harvest stays here.
        A declaration reaches the first at import time, so a heavy import in
        *either* would break every operator on a machine without a GPU.

        Checked by AST rather than by blocking the modules in a subprocess, since
        ``evograd/__init__.py`` imports ``evograd.api`` and therefore torch for
        any ``evograd.*`` import at all -- a repository-wide property that has
        nothing to do with this file.
        """
        import ast

        from evograd.bench.workloads.common import snapshot as common_snapshot

        def toplevel_imports(module) -> set[str]:
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            return {
                node.module.split(".")[0]
                for node in ast.walk(tree)
                # `node.level` is the number of leading dots: a relative import
                # names a sibling in this package, not a third-party dependency.
                if isinstance(node, ast.ImportFrom) and node.module and not node.level
            } | {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                for alias in getattr(node, "names", [])
                if isinstance(node, ast.Import)
            }

        self.assertEqual(
            toplevel_imports(snapshot_module),
            {"__future__", "argparse", "hashlib", "json", "sys", "pathlib", "typing"},
        )
        self.assertEqual(
            toplevel_imports(common_snapshot),
            {"__future__", "hashlib", "json", "re", "pathlib", "typing"},
        )

    def test_the_snapshot_loads_without_transformers(self):
        script = (
            "import sys; sys.modules['transformers'] = None;"
            " from evograd.bench.workloads.qwen3.harvest.snapshot import load;"
            " print(load()['tasks']['qwen3_swiglu_mlp']['frequency'])"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "28")

    def test_the_snapshot_is_tracked_beside_the_code(self):
        self.assertTrue(snapshot_module.SNAPSHOT_PATH.is_file())
        self.assertNotIn("results", snapshot_module.SNAPSHOT_PATH.parts)


class TestDeclaration(unittest.TestCase):
    def test_the_operator_is_registered_at_level_two(self):
        self.assertIn("qwen3_swiglu_mlp", OPS)
        op = get_op("qwen3_swiglu_mlp")
        self.assertEqual(op.level, 2)
        self.assertEqual(op.family, "mlp")

    def test_the_canonical_benchmark_shape_is_the_observed_one(self):
        op = get_op("qwen3_swiglu_mlp")
        self.assertEqual(len(op.benchmark), 1)
        case = op.benchmark[0]
        self.assertEqual(case.dims, {"B": 2, "T": 2048, "H": 1024, "I": 3072})
        self.assertEqual(case.dtype, "bfloat16")

    def test_the_declaration_agrees_with_the_snapshot(self):
        from evograd.bench.workloads.qwen3.levels.level2 import swiglu_mlp as mlp_module

        self.assertEqual(mlp_module.declaration_problems(), [])

    def test_provenance_is_recomputable_from_the_published_config(self):
        from evograd.opdecl.models import rederive_dims

        case = get_op("qwen3_swiglu_mlp").benchmark[0]
        self.assertEqual(case.provenance.source, "hf_config")
        self.assertEqual(case.provenance.model, "qwen3_0_6b")
        self.assertEqual(case.dims, rederive_dims(case.provenance))

    def test_frequency_and_module_paths_are_structured_not_prose(self):
        self.assertEqual(FREQUENCY, 28)
        self.assertEqual(len(HARVEST["module_paths"]), 28)
        self.assertEqual(HARVEST["layer_indices"], list(range(28)))
        self.assertIn("model.layers.14.mlp", HARVEST["module_paths"])
        self.assertEqual(HARVEST["harvested_task"], "mlp")

    def test_the_provenance_chain_is_complete(self):
        joined = " | ".join(PROVENANCE_CHAIN)
        for link in (
            "qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad",
            "3ab24571b6d5860859eb5c947daef94f30dfee4d949ec3cf0dea518ad9c7fabc",
            "model.layers.14",
            "replay",
            "Qwen3MLP",
            "qwen3_swiglu_mlp",
        ):
            self.assertIn(link, joined)

    def test_the_supporting_configurations_are_recorded(self):
        supporting = HARVEST["supporting"]
        self.assertEqual(supporting["gate_up_projection"]["frequency"], 56)
        self.assertEqual(supporting["down_projection"]["frequency"], 28)
        self.assertEqual(supporting["activation"]["frequency"], 28)
        self.assertEqual(
            supporting["gate_up_projection"]["attrs"]["out_features"], 3072
        )

    def test_the_declaration_imports_without_transformers_or_results(self):
        script = (
            "import sys; sys.modules['transformers'] = None;"
            " from evograd.ops import get_op;"
            " op = get_op('qwen3_swiglu_mlp');"
            " print(op.benchmark[0].dims['H'], op.level)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            # A directory with no results/ tree, so a hidden dependency on the
            # untracked harvest would fail here.
            cwd=tempfile.gettempdir(),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "1024 2")


class TestForwardReference(unittest.TestCase):
    def test_the_reference_matches_the_declared_contract(self):
        import torch
        import torch.nn.functional as F

        from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
            qwen3_swiglu_mlp_forward_ref,
        )

        torch.manual_seed(0)
        x = torch.randn(2, 5, 8)
        gate_w = torch.randn(16, 8)
        up_w = torch.randn(16, 8)
        down_w = torch.randn(8, 16)

        expected = F.linear(
            (F.silu(F.linear(x, gate_w).float()) * F.linear(x, up_w).float()).to(x.dtype),
            down_w,
        )
        got = qwen3_swiglu_mlp_forward_ref(x, gate_w, up_w, down_w)
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(got.shape, x.shape)

    def test_mismatched_weights_are_rejected(self):
        import torch

        from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
            qwen3_swiglu_mlp_forward_ref,
        )

        x = torch.randn(1, 2, 8)
        with self.assertRaises(ValueError):
            qwen3_swiglu_mlp_forward_ref(x, torch.randn(16, 8), torch.randn(12, 8), torch.randn(8, 16))
        with self.assertRaises(ValueError):
            qwen3_swiglu_mlp_forward_ref(x, torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8))

    def test_backward_produces_all_four_gradients(self):
        import torch

        from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
            qwen3_swiglu_mlp_forward_ref,
        )

        torch.manual_seed(0)
        tensors = {
            "x": torch.randn(2, 5, 8, requires_grad=True),
            "gate_weight": torch.randn(16, 8, requires_grad=True),
            "up_weight": torch.randn(16, 8, requires_grad=True),
            "down_weight": torch.randn(8, 16, requires_grad=True),
        }
        out = qwen3_swiglu_mlp_forward_ref(**tensors)
        out.backward(torch.randn_like(out))
        for name, tensor in tensors.items():
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor.grad)
                self.assertEqual(tensor.grad.shape, tensor.shape)
                self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_the_hf_spelling_differs_only_by_rounding(self):
        """The declared contract accumulates in float32; Transformers does not.
        The difference is real and small, and is reported rather than hidden."""
        import torch

        from evograd.ops.level2.qwen3_swiglu_mlp.forward_ref import (
            qwen3_swiglu_mlp_forward_hf,
            qwen3_swiglu_mlp_forward_ref,
        )

        torch.manual_seed(0)
        args = (
            torch.randn(2, 32, 16, dtype=torch.bfloat16),
            (torch.randn(48, 16) * 0.25).to(torch.bfloat16),
            (torch.randn(48, 16) * 0.25).to(torch.bfloat16),
            (torch.randn(16, 48) * 0.14).to(torch.bfloat16),
        )
        declared = qwen3_swiglu_mlp_forward_ref(*args).float()
        spelled = qwen3_swiglu_mlp_forward_hf(*args).float()
        scale = float(declared.abs().max())
        self.assertGreater(scale, 0.0)
        self.assertLess(float((declared - spelled).abs().max()) / scale, 2.0**-5)


class TestTimedBaselineAndGate(unittest.TestCase):
    """``runtime_forward`` is what the eager baseline is timed through, and the
    tolerance that guards it is small enough to reject a wrong implementation."""

    def test_runtime_forward_resolves_to_the_hf_spelling(self):
        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
        from evograd.ops.level2.qwen3_swiglu_mlp import forward_ref

        op = get_op("qwen3_swiglu_mlp")
        self.assertIs(resolve_runtime_forward(op), forward_ref.qwen3_swiglu_mlp_forward_hf)
        self.assertIs(resolve_forward(op), forward_ref.qwen3_swiglu_mlp_forward_ref)
        self.assertIsNot(resolve_runtime_forward(op), resolve_forward(op))

    def test_the_declared_tolerances_are_the_calibrated_ones(self):
        op = get_op("qwen3_swiglu_mlp")
        self.assertEqual(op.tolerances["bfloat16"], (1e-2, 1e-2))
        self.assertEqual(op.tolerances["float32"], (2e-5, 2e-5))
        self.assertEqual(
            op.tolerance_multipliers,
            {
                "dx": (2.3, 1.0),
                "dgate_weight": (3.7, 1.0),
                "dup_weight": (4.9, 1.0),
                "ddown_weight": (6.5, 1.0),
            },
        )

    def test_the_real_pair_passes_verify_runtime_forward(self):
        from evograd.opdecl import baselines

        baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_swiglu_mlp")
        baselines.verify_runtime_forward(get_op("qwen3_swiglu_mlp"), device="cpu")

    def test_a_materially_perturbed_implementation_is_rejected(self):
        """The negative control for the tightened gate.

        A 2% error in the SwiGLU intermediate is far larger than any rounding
        difference between the two spellings, and far smaller than a broken
        kernel. The old 8e-2 tolerance accepted it; 1e-2 does not.
        """
        import torch

        from evograd.opdecl import baselines
        from evograd.ops.level2.qwen3_swiglu_mlp import forward_ref

        def perturbed(x, gate_weight, up_weight, down_weight):
            gate = torch.nn.functional.linear(x, gate_weight)
            up = torch.nn.functional.linear(x, up_weight)
            hidden = (torch.nn.functional.silu(gate.float()) * up.float() * 1.02).to(x.dtype)
            return torch.nn.functional.linear(hidden, down_weight)

        original = forward_ref.qwen3_swiglu_mlp_forward_hf
        forward_ref.qwen3_swiglu_mlp_forward_hf = perturbed
        try:
            baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_swiglu_mlp")
            with self.assertRaises(RuntimeError) as ctx:
                baselines.verify_runtime_forward(get_op("qwen3_swiglu_mlp"), device="cpu")
            self.assertIn("disagrees with forward", str(ctx.exception))

            # And the tolerance this replaced would have let it through.
            op = get_op("qwen3_swiglu_mlp")
            case = next(w for w in op.correctness if w.dtype == "bfloat16")
            values = make_case_inputs(op, case, device="cpu")
            args = (
                values["x"],
                values["gate_weight"],
                values["up_weight"],
                values["down_weight"],
            )
            reference = forward_ref.qwen3_swiglu_mlp_forward_ref(*args)
            wrong = perturbed(*args)
            self.assertTrue(
                torch.allclose(wrong.float(), reference.float(), atol=8e-2, rtol=8e-2),
                "the old 8e-2 tolerance should have accepted this, or the test "
                "is not demonstrating a tightening",
            )
            self.assertFalse(
                torch.allclose(wrong.float(), reference.float(), atol=1e-2, rtol=1e-2)
            )
        finally:
            forward_ref.qwen3_swiglu_mlp_forward_hf = original
            baselines._RUNTIME_FORWARD_VERIFIED.discard("qwen3_swiglu_mlp")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers not installed on this machine")
class TestExtractionAndVerification(unittest.TestCase):
    """A real MLP invocation, taken from a real (tiny) layer replay."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.spec = tiny_spec()
        manifest = run_harvest(cls.spec)
        manifest_path = write_manifest(manifest, root / "harvest.json")
        cls.layer_index = cls.spec.arch["num_hidden_layers"] - 1
        # A snapshot of the tiny workload, so the canonical loader has something
        # to validate this artifact against.
        cls.snapshot_path = root / "snapshot.json"
        derived = snapshot_module.extract(manifest, layer_index=cls.layer_index)
        cls.snapshot_path.write_text(json.dumps(derived, indent=2))
        artifact, _ = run_capture(
            cls.spec, manifest_path=manifest_path, layer_index=cls.layer_index
        )
        cls.layer_path = artifact.save(root / "layer.pt")
        cls.payload, cls.metadata = derive_mlp_invocation(
            cls.layer_path, device="cpu", snapshot_path=cls.snapshot_path
        )
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_capture_holds_every_tensor_the_task_needs(self):
        for key in CONTENT_KEYS:
            self.assertIn(key, self.payload)
        for name in ("gate_weight", "up_weight", "down_weight"):
            self.assertIn(name, self.payload["weights"])
            self.assertIn(name, self.payload["weight_grads"])

    def test_shapes_follow_the_contract(self):
        hidden = self.payload["arch"]["hidden_size"]
        intermediate = self.payload["arch"]["intermediate_size"]
        self.assertEqual(list(self.payload["input"].shape)[-1], hidden)
        self.assertEqual(list(self.payload["output"].shape), list(self.payload["input"].shape))
        self.assertEqual(list(self.payload["weights"]["gate_weight"].shape), [intermediate, hidden])
        self.assertEqual(list(self.payload["weights"]["up_weight"].shape), [intermediate, hidden])
        self.assertEqual(list(self.payload["weights"]["down_weight"].shape), [hidden, intermediate])

    def test_nothing_captured_is_on_a_device_or_in_a_graph(self):
        import torch

        for key in CONTENT_KEYS:
            values = self.payload[key]
            tensors = values.values() if isinstance(values, dict) else [values]
            for tensor in tensors:
                self.assertEqual(tensor.device.type, "cpu", key)
                self.assertFalse(tensor.requires_grad, key)
                self.assertIsNone(tensor.grad_fn, key)

    def test_the_provenance_chain_is_recorded_and_valid(self):
        self.assertEqual(check_provenance(self.payload, snapshot_path=self.snapshot_path), [])
        self.assertEqual(len(self.payload["provenance_chain"]), 6)
        self.assertEqual(
            self.payload["identity"]["provenance_kind"], "captured_from_verified_replay"
        )
        self.assertEqual(
            self.payload["identity"]["module_path"], f"model.layers.{self.layer_index}.mlp"
        )

    def test_the_source_artifact_hashes_are_carried_forward(self):
        from evograd.bench.workloads.qwen3.levels.level3.artifact import LayerArtifact

        source = LayerArtifact.load(self.layer_path)
        self.assertEqual(
            self.payload["identity"]["source_content_hash"], source.payload["content_hash"]
        )
        self.assertEqual(
            self.payload["identity"]["source_artifact_hash"], source.payload["artifact_hash"]
        )

    def test_no_second_tensor_file_is_written(self):
        """``layer14.pt`` is the authoritative tensor store. A derived ``.pt``
        would be the same numbers under a second name, with nothing forcing the
        two to stay equal."""
        self.assertFalse(self.metadata["tensors_written"])
        self.assertEqual(list(self.root.glob("*.pt")), [self.layer_path])

    def test_the_derivation_is_reproducible(self):
        again, _ = derive_mlp_invocation(
            self.layer_path, device="cpu", snapshot_path=self.snapshot_path
        )
        self.assertEqual(again["content_hash"], self.payload["content_hash"])
        self.assertEqual(again["derivation_hash"], self.payload["derivation_hash"])
        self.assertNotEqual(again["content_hash"], again["derivation_hash"])

    def test_a_wrong_snapshot_makes_provenance_fail(self):
        problems = check_provenance(self.payload)  # the canonical snapshot
        self.assertTrue(problems)
        self.assertTrue(any("workload_id" in p for p in problems), problems)

    def test_extraction_refuses_a_non_canonical_source(self):
        """``derive_mlp_invocation`` goes through ``load_canonical``: an artifact
        that does not match the snapshot it is checked against cannot be
        derived from."""
        with self.assertRaises(ArtifactError):
            derive_mlp_invocation(self.layer_path, device="cpu")  # canonical snapshot

    def test_the_reference_reproduces_the_captured_invocation(self):
        report = run_verify(
            self.payload, device="cpu", noise_repeats=2, snapshot_path=self.snapshot_path
        )
        self.assertEqual(report["status"], "pass", report["failures"])
        self.assertTrue(report["provenance_validated"])
        comparisons = report["comparisons"]
        for name in ("output", "grad_input"):
            with self.subTest(quantity=name):
                record = comparisons[name]
                self.assertTrue(record["within_tolerance"], record)
                self.assertTrue(record["shape_match"])
                self.assertTrue(record["dtype_match"])
                self.assertTrue(record["stride_match"])
                self.assertTrue(record["actual_all_finite"])
                self.assertTrue(record["expected_all_finite"])
        for name, record in comparisons["weight_grads"].items():
            with self.subTest(gradient=name):
                self.assertTrue(record["within_tolerance"], record)
        self.assertEqual(sorted(comparisons["weight_grads"]), ["down_weight", "gate_weight", "up_weight"])

    def test_the_bf16_spelling_reproduces_the_capture_exactly(self):
        """The wiring check. The declared reference upcasts to float32 and the
        model does not, so only this comparison can be held to a rounding-level
        tolerance -- and a wrong weight mapping or a missing transpose would
        break it."""
        report = run_verify(
            self.payload, device="cpu", noise_repeats=0, snapshot_path=self.snapshot_path
        )
        hf = report["hf_spelling_comparisons"]
        self.assertTrue(hf["output"]["within_tolerance"], hf["output"])
        self.assertTrue(hf["grad_input"]["within_tolerance"], hf["grad_input"])
        for name, record in hf["weight_grads"].items():
            with self.subTest(gradient=name):
                self.assertTrue(record["within_tolerance"], record)

    def test_the_two_comparisons_use_separately_justified_tolerances(self):
        report = run_verify(
            self.payload, device="cpu", noise_repeats=0, snapshot_path=self.snapshot_path
        )
        tolerances = report["tolerances"]
        self.assertIn("declared_reference", tolerances)
        self.assertIn("hf_spelling", tolerances)
        # The declaration's gate is an (atol, rtol) pair, not a single number,
        # and it is looser than the replay's, and says why.
        atol, rtol = tolerances["declared_reference"]["values"]["output"]
        self.assertGreater(atol, tolerances["hf_spelling"]["forward"])
        self.assertGreater(rtol, 0.0)
        self.assertIn("float32", tolerances["declared_reference"]["why"])
        self.assertIn("same computation", tolerances["hf_spelling"]["why"])

    def test_the_declared_gate_is_the_harness_gate(self):
        """The verdict is ``allclose`` at the declared tolerance, not a
        scale-normalized proxy that could accept what the harness rejects."""
        report = run_verify(
            self.payload, device="cpu", noise_repeats=0, snapshot_path=self.snapshot_path
        )
        record = report["comparisons"]["output"]
        self.assertIn("allclose", record)
        self.assertIn("required_t", record)
        self.assertLessEqual(record["required_t"], record["declared_base"])
        self.assertTrue(record["allclose"])

    def test_the_verification_measures_noise(self):
        report = run_verify(
            self.payload, device="cpu", noise_repeats=3, snapshot_path=self.snapshot_path
        )
        noise = report["noise_floor"]
        self.assertTrue(noise["measured"])
        self.assertEqual(noise["repeats"], 3)
        self.assertIn("weight_grads", noise)

    def test_the_verification_also_reports_the_hf_spelling(self):
        report = run_verify(
            self.payload, device="cpu", noise_repeats=0, snapshot_path=self.snapshot_path
        )
        self.assertIn("hf_spelling_comparisons", report)
        self.assertIn("output", report["hf_spelling_comparisons"])

    def test_a_second_mlp_call_inside_one_extraction_is_refused(self):
        from evograd.bench.workloads.qwen3.levels.level2.swiglu_mlp import capture_mlp
        from evograd.bench.workloads.qwen3.levels.level3.replay import build_single_layer

        from evograd.bench.workloads.qwen3.levels.level3.artifact import LayerArtifact

        layer_payload = LayerArtifact.load(self.layer_path).payload
        layer = build_single_layer(
            layer_payload["arch"], self.layer_index, device="cpu", dtype="float32"
        )
        mlp = layer.get_submodule("mlp")
        import torch

        with self.assertRaises(MlpExtractionError):
            with capture_mlp(layer):
                x = torch.randn(1, 4, layer_payload["arch"]["hidden_size"])
                mlp(x)
                mlp(x)
        self.assertEqual(len(mlp._forward_hooks), 0)
        self.assertEqual(len(mlp._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
