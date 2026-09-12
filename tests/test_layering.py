"""Static dependency checks for the benchmark/evaluation package split."""

from __future__ import annotations

import ast
import pathlib
import unittest

EVOGRAD = pathlib.Path(__file__).resolve().parents[1] / "src" / "evograd"
SHARED = (
    EVOGRAD / "benchmark" / "topdown" / "common",
    EVOGRAD / "evaluation" / "tier3" / "gate",
)
WORKLOAD_MARKERS = ("qwen3_0_6b", "llama3_2_1b", "alphafold3")


def _resolved_imports(path: pathlib.Path) -> set[str]:
    """Return imports with relative names resolved against ``evograd``."""
    parts = path.relative_to(EVOGRAD.parent).with_suffix("").parts
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                package = list(parts[: len(parts) - node.level])
                suffix = (node.module or "").split(".") if node.module else []
                found.add(".".join(package + suffix))
            elif node.module:
                found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _python_files(root: pathlib.Path):
    return sorted(root.rglob("*.py"))


class TestSharedLayers(unittest.TestCase):
    def test_shared_directories_are_nonempty_and_workload_neutral(self):
        for root in SHARED:
            with self.subTest(root=root):
                files = [
                    path for path in _python_files(root)
                    if path.name != "__init__.py"
                ]
                self.assertTrue(files, f"{root} holds no implementation modules")
                for path in files:
                    for imported in _resolved_imports(path):
                        self.assertFalse(
                            any(marker in imported for marker in WORKLOAD_MARKERS),
                            f"{path} imports workload-specific module {imported}",
                        )

    def test_generic_tier3_modules_import_no_specific_workload(self):
        root = EVOGRAD / "evaluation" / "tier3"
        for name in ("cli.py", "model.py", "patch.py", "runner.py"):
            path = root / name
            for imported in _resolved_imports(path):
                self.assertFalse(
                    any(marker in imported for marker in WORKLOAD_MARKERS),
                    f"{path} imports workload-specific module {imported}",
                )


class TestDependencyDirection(unittest.TestCase):
    def test_declaration_layers_do_not_import_evaluation(self):
        for area in ("opdecl", "ops"):
            for path in _python_files(EVOGRAD / area):
                offenders = [
                    name for name in _resolved_imports(path)
                    if name.startswith("evograd.evaluation")
                ]
                self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_benchmark_imports_no_evaluation_anywhere(self):
        """The whole benchmark package stays below the evaluation runners.

        There is no exemption. ``operator_suite/cli.py`` used to be one: it was
        the composition root that asked evaluation to run what the benchmark
        selected. That coordination now lives at the root, in
        :mod:`evograd.suite_cli`, so the rule holds for every file here.
        """
        for path in _python_files(EVOGRAD / "benchmark"):
            offenders = [
                name for name in _resolved_imports(path)
                if name.startswith("evograd.evaluation")
            ]
            self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_no_suite_cli_is_left_inside_benchmark(self):
        """No forwarding module: one canonical entry point, at the root."""
        self.assertFalse((EVOGRAD / "benchmark" / "operator_suite" / "cli.py").exists())
        self.assertTrue((EVOGRAD / "suite_cli.py").is_file())

    def test_ops_imports_no_benchmark_at_all(self):
        """The primitive layer sits below the benchmark, with no bridge back.

        ``ops`` used to reach into ``benchmark.topdown`` for a frozen snapshot,
        so a package that owns mathematics depended on one model's captured
        run. The observed cases are bound on the benchmark side now
        (:mod:`evograd.benchmark.cases`), and this is what stops the bridge
        being rebuilt.
        """
        for path in _python_files(EVOGRAD / "ops"):
            offenders = [
                name for name in _resolved_imports(path)
                if name.startswith("evograd.benchmark")
            ]
            self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_ops_imports_no_workload_package(self):
        for path in _python_files(EVOGRAD / "ops"):
            for imported in _resolved_imports(path):
                self.assertFalse(
                    any(marker in imported for marker in WORKLOAD_MARKERS),
                    f"{path} imports workload-specific module {imported}",
                )

    def test_the_deleted_ops_level_directories_are_gone(self):
        for group in ("level2", "level3"):
            with self.subTest(group=group):
                self.assertFalse((EVOGRAD / "ops" / group).exists())

    def test_no_active_import_names_the_deleted_ops_levels(self):
        """No compatibility shim, no alias, no tombstone -- one canonical path.

        Scanned by source text across the whole repository rather than by
        resolved import, because ``tests/`` and ``scripts/`` sit outside the
        package root that relative names resolve against.
        """
        repo = pathlib.Path(__file__).resolve().parents[1]
        # Assembled rather than written out, so this file does not match itself.
        pattern = tuple("evograd.ops." + group for group in ("level2", "level3"))
        for root in (EVOGRAD, repo / "tests", repo / "scripts", repo / "tools"):
            if not root.exists():
                continue
            for path in _python_files(root):
                source = path.read_text(encoding="utf-8")
                for name in pattern:
                    self.assertNotIn(name, source, f"{path} names {name}")


class TestQwenOwnershipSplit(unittest.TestCase):
    """Capture describes the case; evaluation decides the verdict."""

    QWEN_BENCH = EVOGRAD / "benchmark" / "topdown" / "qwen3_0_6b"
    QWEN_EVAL = EVOGRAD / "evaluation" / "workloads" / "qwen3_0_6b"

    def test_qwen_benchmark_side_imports_no_evaluation(self):
        for path in _python_files(self.QWEN_BENCH):
            offenders = [
                name for name in _resolved_imports(path)
                if name.startswith("evograd.evaluation")
            ]
            self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_the_evaluation_side_exists_and_owns_the_verdicts(self):
        for relative in ("level1/verify.py", "level1/calibrate.py", "level1/cli.py",
                         "level2/qkv_norm_rope.py", "level2/attention.py",
                         "level2/swiglu_mlp.py", "level2/residual_rmsnorm.py",
                         "level2/calibrate.py", "level2/negative_controls.py",
                         "level3/replay.py"):
            with self.subTest(module=relative):
                self.assertTrue((self.QWEN_EVAL / relative).is_file())

    def test_the_benchmark_side_keeps_capture_and_artifact(self):
        for relative in ("levels/level1/manifest.py",
                         "levels/level2/manifest.py",
                         "levels/level3/artifact.py",
                         "levels/level3/capture.py",
                         "levels/level3/prepare.py"):
            with self.subTest(module=relative):
                self.assertTrue((self.QWEN_BENCH / relative).is_file())

    def test_no_verdict_function_is_left_on_the_benchmark_side(self):
        """``run_verify``/``run_calibration`` decide whether something passes."""
        for path in _python_files(self.QWEN_BENCH):
            source = path.read_text(encoding="utf-8")
            for marker in ("def run_verify(", "def run_calibration(",
                           "def declared_gate(", "def required_tolerance("):
                self.assertNotIn(
                    marker, source,
                    f"{path} defines {marker.strip('def (')}; judgment belongs to "
                    f"evograd.evaluation",
                )


class TestLlamaOwnershipSplit(unittest.TestCase):
    """The same split, applied to the second harvested architecture.

    The Qwen3 class above pins one model's boundary. Repeating it here is not
    duplication for its own sake: the split is a property of *each* workload
    package, and a second architecture integrated without it would leave the
    layering test passing on the model that was checked while the new one drifts.
    """

    LLAMA_BENCH = EVOGRAD / "benchmark" / "topdown" / "llama3_2_1b"
    LLAMA_EVAL = EVOGRAD / "evaluation" / "workloads" / "llama3_2_1b"

    def test_llama_benchmark_side_imports_no_evaluation(self):
        for path in _python_files(self.LLAMA_BENCH):
            offenders = [
                name for name in _resolved_imports(path)
                if name.startswith("evograd.evaluation")
            ]
            self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_the_evaluation_side_exists_and_owns_the_verdicts(self):
        for relative in ("level1/verify.py", "level1/calibrate.py", "level1/cli.py",
                         "level2/qkv_rope.py", "level2/attention.py",
                         "level2/swiglu_mlp.py", "level2/residual_rmsnorm.py",
                         "level2/calibrate.py", "level2/negative_controls.py",
                         "level3/replay.py"):
            with self.subTest(module=relative):
                self.assertTrue((self.LLAMA_EVAL / relative).is_file())

    def test_the_benchmark_side_keeps_capture_and_artifact(self):
        for relative in ("levels/level1/manifest.py",
                         "levels/level2/manifest.py",
                         "levels/level3/artifact.py",
                         "levels/level3/capture.py",
                         "levels/level3/prepare.py"):
            with self.subTest(module=relative):
                self.assertTrue((self.LLAMA_BENCH / relative).is_file())

    def test_each_site_package_owns_its_contract_reference_and_capture(self):
        """Four site packages, as Qwen3 has, rather than four flat modules."""
        from evograd.benchmark.topdown.llama3_2_1b.levels.level2 import manifest

        for site in manifest.SITES:
            for part in ("__init__.py", "task.py", "reference.py", "capture.py"):
                with self.subTest(site=site, part=part):
                    self.assertTrue(
                        (self.LLAMA_BENCH / "levels" / "level2" / site / part).is_file(),
                        f"{site}/{part} missing",
                    )

    def test_no_verdict_function_is_left_on_the_benchmark_side(self):
        for path in _python_files(self.LLAMA_BENCH):
            source = path.read_text(encoding="utf-8")
            for marker in ("def run_verify(", "def run_calibration(",
                           "def declared_gate(", "def required_tolerance("):
                self.assertNotIn(
                    marker, source,
                    f"{path} defines {marker.strip('def (')}; judgment belongs to "
                    f"evograd.evaluation",
                )


class TestTaskNamesAreUnique(unittest.TestCase):
    """Aggregating three sources is where a name collision would first bite."""

    def test_no_task_name_is_claimed_twice(self):
        from evograd.benchmark import TASKS
        from evograd.benchmark.core.registry import _discover

        self.assertEqual(len(TASKS), len(set(TASKS)))
        self.assertEqual(set(_discover()), set(TASKS))

    def test_a_duplicate_name_fails_loudly(self):
        from evograd.benchmark.core.registry import DuplicateTask, _register
        from evograd.ops import PRIMITIVES

        primitive = PRIMITIVES["rmsnorm"]
        discovered: dict = {}
        _register(discovered, primitive, "owner.one")
        with self.assertRaises(DuplicateTask):
            _register(discovered, primitive, "owner.two")

    def test_every_primitive_reaches_the_task_registry_exactly_once(self):
        from evograd.benchmark import TASKS
        from evograd.ops import PRIMITIVES

        for name in PRIMITIVES:
            with self.subTest(op=name):
                self.assertIn(name, TASKS)
                self.assertEqual(TASKS[name].level, 1)


if __name__ == "__main__":
    unittest.main()
