"""The dependency direction, enforced rather than described.

This repository has two kinds of shared code and one kind of specific code:

    bench/tier3_gate/        evaluation. Asks questions about a *kernel*.
    bench/workloads/common/  workload machinery. Describes *a* model, not one.
    bench/workloads/<name>/  one model: its classes, shapes, adapters, snapshot.

The value of that split is entirely in its direction. A workload package
imports the shared halves; the shared halves import nothing from any workload.
The moment one of them does, a threshold or an aggregation has acquired a
branch on a model name, and the next architecture inherits a number measured on
something else.

That is not a property a docstring can hold. Checked by AST rather than by
importing, so a module that is heavy or optional is still covered, and so the
failure names the file and the import rather than surfacing as a cycle.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

BENCH = pathlib.Path(__file__).resolve().parents[1] / "src" / "evograd" / "bench"

#: Directories that must not know which model they are serving.
SHARED = ("tier3_gate", "workloads/common")

#: Substrings that mark an import as belonging to one workload. Extend this with
#: each workload package added under ``bench/workloads/``.
WORKLOAD_MARKERS = ("qwen3", "llama")


def _resolved_imports(path: pathlib.Path) -> set[str]:
    """Every module this file imports, with relative imports resolved.

    A relative import is the one that actually matters here: ``from .sites
    import ...`` inside shared code is exactly the regression this guards
    against, and it carries no package name to grep for.
    """
    parts = path.relative_to(BENCH.parents[2]).with_suffix("").parts
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
            found |= {alias.name for alias in node.names}
    return found


class TestSharedCodeKnowsNoWorkload(unittest.TestCase):
    def test_shared_directories_import_no_workload_package(self):
        for area in SHARED:
            for path in sorted((BENCH / area).rglob("*.py")):
                for module in sorted(_resolved_imports(path)):
                    for marker in WORKLOAD_MARKERS:
                        with self.subTest(file=str(path.name), imports=module):
                            self.assertNotIn(
                                marker, module,
                                f"{path} imports {module}: shared code must not "
                                f"depend on a workload package",
                            )

    def test_the_shared_directories_are_not_empty(self):
        """A guard that passes because there is nothing to check is not a guard."""
        for area in SHARED:
            with self.subTest(area=area):
                modules = [
                    p for p in (BENCH / area).rglob("*.py")
                    if p.name != "__init__.py"
                ]
                self.assertTrue(modules, f"{area} holds no modules")

    def test_the_tier3_cli_names_no_architecture(self):
        """The CLI reaches a workload through the registry, by name."""
        source = (BENCH / "tier3_cli.py").read_text(encoding="utf-8")
        for marker in ("qwen3", "Qwen3", "Qwen"):
            self.assertNotIn(marker, source, marker)

    def test_the_runner_and_patcher_name_no_architecture_in_code(self):
        """``tier3_runner`` and ``tier3_patch`` may *mention* a model in prose --
        they explain themselves with examples -- but must not import one."""
        for name in ("tier3_runner.py", "tier3_patch.py", "tier3_model.py"):
            with self.subTest(module=name):
                for module in _resolved_imports(BENCH / name):
                    self.assertNotIn("qwen3", module)


class TestWorkloadRegistryIsTheOnlyDoor(unittest.TestCase):
    def test_ops_reaches_workloads_only_through_the_registry(self):
        """``evograd.ops`` declarations state the shapes a real model ran. They
        must reach them by workload *name*, so a second harvested architecture
        needs no edit to any operator."""
        ops = BENCH.parents[0] / "ops"
        offenders = []
        for path in sorted(ops.rglob("*.py")):
            for module in _resolved_imports(path):
                if "workloads." in module and "workloads.common" not in module:
                    offenders.append(f"{path.name}: {module}")
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
