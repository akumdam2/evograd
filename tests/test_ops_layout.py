"""The ops/ directory layout must agree with the declarations it holds.

Operators live under ``evograd/ops/level<N>/<name>/`` so the tree reads like the
benchmark hierarchy. That makes the level visible in two places — the directory
and ``OpDecl.level`` — and two sources of truth drift. The declaration is the
authority; these tests make the directory follow it, so moving an operator
between levels without moving its files fails loudly instead of leaving the tree
quietly wrong.
"""

import unittest
from pathlib import Path

from evograd.ops import OPS, WORKLOADS

OPS_ROOT = Path(__file__).resolve().parents[1] / "src" / "evograd" / "ops"
GROUPS = ("level1", "level2", "level3", "level4")


class TestOpsLayout(unittest.TestCase):
    def test_every_operator_sits_in_the_directory_its_level_declares(self):
        for name, op in sorted(OPS.items()):
            with self.subTest(op=name):
                expected = OPS_ROOT / f"level{op.level}" / name
                self.assertTrue(
                    expected.is_dir(),
                    f"{name} declares level {op.level} but "
                    f"{expected.relative_to(OPS_ROOT)} does not exist",
                )
        for name, workload in sorted(WORKLOADS.items()):
            with self.subTest(workload=name):
                expected = OPS_ROOT / f"level{workload.level}" / name
                self.assertTrue(expected.is_dir())

    def test_no_operator_package_sits_directly_under_ops(self):
        """A package left at the top level would still be discovered, so a
        half-finished move must fail here rather than pass unnoticed."""
        stray = [
            entry.name
            for entry in OPS_ROOT.iterdir()
            if entry.is_dir()
            and entry.name not in GROUPS
            and not entry.name.startswith("_")
            and not entry.name.startswith(".")
            and (entry / "__init__.py").is_file()
        ]
        self.assertEqual(stray, [], f"operator packages outside level dirs: {stray}")

    def test_group_packages_declare_no_operator(self):
        """level1/ and friends are grouping packages. If one ever exposed an
        ``op`` the registry would register it under the group's name."""
        import importlib

        for group in GROUPS:
            with self.subTest(group=group):
                module = importlib.import_module(f"evograd.ops.{group}")
                self.assertIsNone(getattr(module, "op", None))

    def test_directory_contents_match_the_registry(self):
        """Every operator directory is registered, and every registered
        operator has a directory — no orphans in either direction."""
        on_disk = {
            entry.name
            for group in GROUPS
            for entry in (OPS_ROOT / group).iterdir()
            if entry.is_dir()
            and (entry / "__init__.py").is_file()
            and not entry.name.startswith("_")
        }
        self.assertEqual(on_disk, set(OPS) | set(WORKLOADS))

    def test_expected_counts_per_level(self):
        """The benchmark specification states 18 / 5 / 2, plus the operators
        derived from an observed workload rather than specified up front --
        currently ``qwen3_swiglu_mlp``, ``qwen3_attention`` and
        ``qwen3_qkv_norm_rope``, extracted from the
        Qwen3-0.6B Level-4 harvest. Counted separately so the specified suite
        stays legible. Whole-model tasks are WorkloadDecls, not operators, and
        are counted on their own registry."""
        counts = {level: 0 for level in (1, 2, 3)}
        for op in OPS.values():
            counts[op.level] += 1
        self.assertEqual(counts, {1: 20, 2: 8, 3: 2})
        for name in ("qwen3_swiglu_mlp", "qwen3_attention", "qwen3_qkv_norm_rope"):
            self.assertIn(name, OPS)
            self.assertEqual(OPS[name].level, 2)
        self.assertEqual(
            {name: decl.level for name, decl in WORKLOADS.items()},
            {"alphafold3": 4},
        )


if __name__ == "__main__":
    unittest.main()
