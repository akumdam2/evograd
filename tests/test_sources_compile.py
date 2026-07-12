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


if __name__ == "__main__":
    unittest.main()
