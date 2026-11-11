"""The first Level-2 task derived from an observed workload.

Everything here needs nothing but the tracked snapshot -- no GPU, no
Transformers, no ``results/`` directory -- because that is the whole point of
freezing a snapshot: a declaration whose shapes came from a real training step
must still import and be checkable on a machine that has never run one. Its
real invocations are the real-payload capture's
(``tests/verification/test_integration_qwen3.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evograd.benchmark.topdown.qwen3_0_6b.harvest import snapshot as snapshot_module
from evograd.opdecl.inputs import make_case_inputs
from evograd.benchmark import TASKS, get_task
from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.task import (
    FREQUENCY,
    HARVEST,
    PROVENANCE_CHAIN,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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

        Checked by AST rather than by blocking the modules in a subprocess, so
        the check is about these modules' own imports and not about whatever an
        ``evograd.*`` import happens to pull in.
        """
        import ast

        from evograd.benchmark.topdown.common import snapshot as common_snapshot

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
            " from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import load;"
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
        self.assertIn("qwen3_swiglu_mlp", TASKS)
        op = get_task("qwen3_swiglu_mlp")
        self.assertEqual(op.level, 2)
        self.assertEqual(op.family, "mlp")

    def test_the_canonical_benchmark_shape_is_the_observed_one(self):
        op = get_task("qwen3_swiglu_mlp")
        self.assertEqual(len(op.benchmark), 1)
        case = op.benchmark[0]
        self.assertEqual(case.dims, {"B": 2, "T": 2048, "H": 1024, "I": 3072})
        self.assertEqual(case.dtype, "bfloat16")

    def test_provenance_is_recomputable_from_the_published_config(self):
        from evograd.opdecl.models import rederive_dims

        case = get_task("qwen3_swiglu_mlp").benchmark[0]
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
            "qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.pretrained.realtext.09d9a051",
            "7ba7c1019f24a3dd135f01a71088052d86d0613e478f6431313ac670d98915b1",
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
            " from evograd.benchmark import get_task;"
            " op = get_task('qwen3_swiglu_mlp');"
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

        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference import (
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

        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference import (
            qwen3_swiglu_mlp_forward_ref,
        )

        x = torch.randn(1, 2, 8)
        with self.assertRaises(ValueError):
            qwen3_swiglu_mlp_forward_ref(x, torch.randn(16, 8), torch.randn(12, 8), torch.randn(8, 16))
        with self.assertRaises(ValueError):
            qwen3_swiglu_mlp_forward_ref(x, torch.randn(16, 8), torch.randn(16, 8), torch.randn(16, 8))

    def test_backward_produces_all_four_gradients(self):
        import torch

        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference import (
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

        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp.reference import (
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
    """``runtime_forward`` is what the eager baseline is timed through."""

    def test_runtime_forward_resolves_to_the_hf_spelling(self):
        from evograd.opdecl.oracle import resolve_forward, resolve_runtime_forward
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.swiglu_mlp import reference as forward_ref
        op = get_task("qwen3_swiglu_mlp")
        self.assertIs(resolve_runtime_forward(op), forward_ref.qwen3_swiglu_mlp_forward_hf)
        self.assertIs(resolve_forward(op), forward_ref.qwen3_swiglu_mlp_forward_ref)
        self.assertIsNot(resolve_runtime_forward(op), resolve_forward(op))

    def test_the_observed_case_is_this_models_own_width(self):
        """Llama-3 has its own task, `llama3_swiglu_mlp`, at its own width; this
        task's observed case stays Qwen3's 3072-wide intermediate."""
        op = get_task("qwen3_swiglu_mlp")
        observed = op.benchmark_workloads(suite="qwen3_0_6b_observed")[0]
        self.assertEqual(observed.dims["I"], 3072)
        self.assertNotIn("llama_3_2_1b_observed", op.benchmark_suites)


if __name__ == "__main__":
    unittest.main()
