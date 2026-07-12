"""Every source file must at least compile — the only whole-tree check
possible on dev boxes without torch/triton (see README testing notes)."""

import compileall
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSourcesCompile(unittest.TestCase):
    def test_src_tree_compiles(self):
        self.assertTrue(
            compileall.compile_dir(str(REPO_ROOT / "src"), quiet=2, force=True)
        )

    def test_scripts_compile(self):
        self.assertTrue(
            compileall.compile_dir(str(REPO_ROOT / "scripts"), quiet=2, force=True)
        )

    def test_pipeline_b_uses_installed_evograd_namespace(self):
        source = (
            REPO_ROOT / "src/evograd/pipelines/b_dispatch/program_codegen.py"
        ).read_text(encoding="utf-8")
        self.assertIn("evograd.atenir.primitive_triton", source)
        self.assertNotIn('import_module("atenir.', source)
        self.assertNotIn("from atenir.", source)


if __name__ == "__main__":
    unittest.main()
