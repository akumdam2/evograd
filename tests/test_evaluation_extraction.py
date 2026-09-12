"""Shared evaluation modules must not reference names they do not bind.

Extracting a per-model module into a shared one moves function bodies away from
the imports that fed them. A helper that quietly used a symbol imported at the
top of the old file then raises ``NameError`` at runtime -- after the model has
been built and the artifact loaded, which on a real workload is minutes in.
This catches it at test time instead, and is the guard for every remaining
extraction.
"""

from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1] / "src/evograd/evaluation/workloads/common"


def unresolved_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound |= {(a.asname or a.name) for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    used = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return sorted(used - bound)


class TestSharedEvaluationModules(unittest.TestCase):
    def test_every_shared_module_binds_what_it_uses(self):
        modules = [p for p in SHARED.glob("*.py") if p.name != "__init__.py"]
        self.assertTrue(modules, "no shared evaluation modules found")
        for path in modules:
            with self.subTest(module=path.name):
                self.assertEqual(unresolved_names(path), [], f"{path.name} uses unbound names")

    def test_no_shared_module_hardcodes_one_model(self):
        """A correct name with the wrong value is invisible to the check above.

        Six report schemas were left as ``evograd-llama3-…`` literals during
        extraction, which would have stamped every Qwen report with Llama's
        schema. Nothing structural catches that, so it is checked by name.
        """
        import re

        pattern = re.compile(r"evograd-(llama3|qwen3)|topdown\.(llama3_2_1b|qwen3_0_6b)")
        for path in SHARED.glob("*.py"):
            if path.name in ("__init__.py", "descriptor.py"):
                continue
            hits = [
                f"{path.name}:{i}"
                for i, line in enumerate(path.read_text().splitlines(), 1)
                if pattern.search(line) and not line.lstrip().startswith("#")
            ]
            with self.subTest(module=path.name):
                self.assertEqual(hits, [], "shared module names one model")

    def test_local_calls_pass_every_required_parameter(self):
        """Threading a parameter into a signature must update its call sites.

        ``build_parser()`` kept being called with no argument after
        ``descriptor`` became required -- a TypeError that only surfaces when
        the CLI runs, which is minutes into a GPU session.
        """
        for path in SHARED.glob("*.py"):
            if path.name == "__init__.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            required = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    pos = [a.arg for a in node.args.args]
                    n_default = len(node.args.defaults)
                    needed_pos = len(pos) - n_default
                    kw_required = [
                        a.arg
                        for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                        if d is None
                    ]
                    required[node.name] = (needed_pos, set(kw_required))
            problems = []
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                spec = required.get(node.func.id)
                if spec is None:
                    continue
                needed_pos, kw_required = spec
                given_kw = {k.arg for k in node.keywords if k.arg}
                starred = any(k.arg is None for k in node.keywords) or any(
                    isinstance(a, ast.Starred) for a in node.args
                )
                if len(node.args) < needed_pos and not starred:
                    problems.append(f"{node.func.id}() line {node.lineno}: too few positional")
                missing = kw_required - given_kw
                if missing and not starred:
                    problems.append(f"{node.func.id}() line {node.lineno}: missing {sorted(missing)}")
            with self.subTest(module=path.name):
                self.assertEqual(problems, [])

    def test_the_check_would_catch_a_lifted_helper(self):
        """The guard itself, on the shape of the bug that motivated it."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "lifted.py"
            bad.write_text("def f(x):\n    return tensor_meta(x)\n")
            self.assertEqual(unresolved_names(bad), ["tensor_meta"])


if __name__ == "__main__":
    unittest.main()
