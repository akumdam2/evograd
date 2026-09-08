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
WORKLOAD_MARKERS = ("qwen3_0_6b", "llama3_8b", "alphafold3")


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

    def test_benchmark_definitions_do_not_import_evaluation(self):
        """Core and top-down declarations stay below evaluation runners.

        ``operator_suite/cli.py`` is the composition root that invokes an
        evaluation mode, so it is intentionally outside this rule.
        """
        roots = (
            EVOGRAD / "benchmark" / "core",
            EVOGRAD / "benchmark" / "topdown",
        )
        for path in [file for root in roots for file in _python_files(root)]:
            offenders = [
                name for name in _resolved_imports(path)
                if name.startswith("evograd.evaluation")
            ]
            self.assertEqual(offenders, [], f"{path}: {offenders}")

    def test_ops_snapshot_bridge_names_no_workload_package(self):
        """Model-derived shapes use only the neutral top-down registry."""
        for path in _python_files(EVOGRAD / "ops"):
            for imported in _resolved_imports(path):
                if not imported.startswith("evograd.benchmark"):
                    continue
                self.assertEqual(
                    imported,
                    "evograd.benchmark.topdown",
                    f"{path} bypasses the neutral snapshot registry: {imported}",
                )


if __name__ == "__main__":
    unittest.main()
