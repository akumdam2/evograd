"""Where a declaration lives must agree with what kind of thing it is.

Three packages own declarations and the boundary between them is the point of
the layout:

* ``evograd/ops/level1/<name>/`` -- reusable primitives, and nothing else;
* ``evograd/benchmark/operator_suite/tasks/level<N>/<name>/`` -- fusions that
  belong to no single model;
* ``evograd/benchmark/topdown/<model>/levels/level<N>/<site>/`` -- tasks whose
  identity comes from one captured model.

The declaration's ``level`` is the authority on which of those it is; these
tests make the directory follow it, so a half-finished move fails loudly
instead of leaving the tree quietly wrong. They also pin the boundary itself:
``ops/level2`` and ``ops/level3`` must not come back.
"""

import unittest
from pathlib import Path

from evograd.benchmark import TASKS, WORKLOADS
from evograd.ops import PRIMITIVES

SRC = Path(__file__).resolve().parents[1] / "src" / "evograd"
OPS_ROOT = SRC / "ops"
SUITE_TASKS = SRC / "benchmark" / "operator_suite" / "tasks"
QWEN_L2 = SRC / "benchmark" / "topdown" / "qwen3_0_6b" / "levels" / "level2"

#: Site package -> the task key it serves. The two differ on purpose.
QWEN_SITE_TASKS = {
    "qkv_norm_rope": "qwen3_qkv_norm_rope",
    "attention": "qwen3_attention",
    "swiglu_mlp": "qwen3_swiglu_mlp",
    "residual_rmsnorm": "fused_add_rms_norm",
}


class TestPrimitivesOwnOnlyLevelOne(unittest.TestCase):
    def test_every_primitive_sits_under_level1(self):
        for name in sorted(PRIMITIVES):
            with self.subTest(op=name):
                self.assertTrue(
                    (OPS_ROOT / "level1" / name / "__init__.py").is_file(),
                    f"{name} is registered as a primitive but has no level1 package",
                )

    def test_the_primitive_registry_holds_only_level_one(self):
        for name, op in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                self.assertEqual(op.level, 1)

    def test_no_fused_task_is_registered_as_a_primitive(self):
        for name, op in sorted(TASKS.items()):
            if op.level == 1:
                continue
            with self.subTest(op=name):
                self.assertNotIn(name, PRIMITIVES)

    def test_the_deleted_level_directories_do_not_come_back(self):
        """``ops`` owns mathematics; a level directory above 1 would mean it had
        started owning benchmarks again."""
        for group in ("level2", "level3"):
            with self.subTest(group=group):
                self.assertFalse(
                    (OPS_ROOT / group).exists(),
                    f"ops/{group} exists again; fused and model-specific tasks "
                    f"belong to evograd.benchmark",
                )

    def test_no_primitive_package_sits_directly_under_ops(self):
        """A package left at the top level would still be discovered, so a
        half-finished move must fail here rather than pass unnoticed."""
        stray = [
            entry.name
            for entry in OPS_ROOT.iterdir()
            if entry.is_dir()
            and entry.name != "level1"
            and not entry.name.startswith(("_", "."))
            and (entry / "__init__.py").is_file()
        ]
        self.assertEqual(stray, [], f"primitive packages outside level1/: {stray}")

    def test_the_group_package_declares_no_operator(self):
        """``level1/`` is a grouping package. If it ever exposed an ``op`` the
        registry would register it under the group's name."""
        import importlib

        module = importlib.import_module("evograd.ops.level1")
        self.assertIsNone(getattr(module, "op", None))

    def test_directory_contents_match_the_primitive_registry(self):
        """No orphan in either direction."""
        on_disk = {
            entry.name
            for entry in (OPS_ROOT / "level1").iterdir()
            if entry.is_dir()
            and (entry / "__init__.py").is_file()
            and not entry.name.startswith("_")
        }
        self.assertEqual(on_disk, set(PRIMITIVES))


class TestFusedTasksLiveInBenchmark(unittest.TestCase):
    def test_generic_level_two_tasks_resolve_from_the_operator_suite(self):
        for name in ("fused_linear_cross_entropy", "fused_moe_swiglu",
                     "gemm_leaky_relu", "layernorm_linear"):
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 2)
                self.assertTrue(
                    (SUITE_TASKS / "level2" / name / "__init__.py").is_file(),
                    f"{name} is a generic fusion and belongs to operator_suite/tasks",
                )

    def test_qwen_level_two_tasks_resolve_from_the_model_package(self):
        for site, task in sorted(QWEN_SITE_TASKS.items()):
            with self.subTest(site=site):
                self.assertIn(task, TASKS)
                self.assertEqual(TASKS[task].level, 2)
                for part in ("__init__.py", "task.py", "reference.py", "capture.py"):
                    self.assertTrue(
                        (QWEN_L2 / site / part).is_file(),
                        f"{site}/{part} missing; the site owns its contract, "
                        f"reference and capture",
                    )

    def test_a_site_package_may_be_named_for_the_place_not_the_task(self):
        """``residual_rmsnorm`` serves ``fused_add_rms_norm``. Their differing
        so is the reason the manifest exists, and the reason the model-side
        discovery does not require the two to match."""
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2 import manifest

        self.assertEqual(manifest.SITE_TASKS, QWEN_SITE_TASKS)
        self.assertEqual(manifest.task_key("residual_rmsnorm"), "fused_add_rms_norm")


class TestRegistryComposition(unittest.TestCase):
    def test_expected_counts_per_level(self):
        """20 reusable primitives, 4 generic fusions, 4 Qwen and 4 Llama ones.

        The legacy direct-block tasks -- ``llama3_decoder_layer`` and
        ``af3_single_repr_block`` -- have been deleted, so no task declares
        level 3. Whole-model tasks are WorkloadDecls, not pair contracts, and
        are counted on their own registry.

        Each harvested architecture owns its four Level-2 identities rather
        than sharing another model's. ``llama3_qkv_rope`` is a separate
        declaration for a separate reason as well: ``LlamaAttention`` has no
        per-head query/key RMSNorm, so it is the same three projections and the
        same rotation with two fewer weights and two fewer gradients -- a
        different computation, not a second suite on Qwen's.
        """
        counts: dict[int, int] = {}
        for op in TASKS.values():
            counts[op.level] = counts.get(op.level, 0) + 1
        self.assertEqual(counts, {1: 20, 2: 12})
        self.assertEqual(len(PRIMITIVES), 20)
        self.assertEqual(len(TASKS), 32)
        for name in ("qwen3_swiglu_mlp", "qwen3_attention", "qwen3_qkv_norm_rope",
                     "llama3_qkv_rope", "llama3_attention", "llama3_swiglu_mlp",
                     "llama3_residual_rmsnorm"):
            self.assertIn(name, TASKS)
            self.assertEqual(TASKS[name].level, 2)
        self.assertEqual(
            {name: decl.level for name, decl in WORKLOADS.items()},
            {"alphafold3": 4},
        )

    def test_the_deleted_legacy_blocks_are_not_registered(self):
        for name in ("llama3_decoder_layer", "af3_single_repr_block"):
            with self.subTest(op=name):
                self.assertNotIn(name, TASKS)
                self.assertNotIn(name, PRIMITIVES)

    def test_every_primitive_is_also_an_executable_task(self):
        """Binding a model's observed cases must not drop a primitive."""
        self.assertEqual(
            set(PRIMITIVES), {n for n, op in TASKS.items() if op.level == 1}
        )


if __name__ == "__main__":
    unittest.main()
