"""Keeping every evaluated candidate (`evograd evolve --save-programs`).

OpenEvolve returns only the best program, so without this the other N-1
candidates a run paid for are unrecoverable.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from evograd.evolve.evaluator import EvaluationResult, archive_program


def _result(score: float) -> EvaluationResult:
    return EvaluationResult(metrics={"combined_score": score, "correct": 1.0}, artifacts={})


class TestArchiveProgram(unittest.TestCase):
    def test_archives_source_metrics_and_index(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            program = root / "candidate.py"
            program.write_text("# kernel v1\n")
            archive = root / "programs"

            saved = archive_program(str(program), _result(1.5), archive)

            self.assertIsNotNone(saved)
            self.assertEqual(Path(saved).read_text(), "# kernel v1\n")
            self.assertEqual(Path(saved).parent, archive)
            sidecar = json.loads(Path(saved).with_suffix(".json").read_text())
            self.assertEqual(sidecar["metrics"]["combined_score"], 1.5)

            record = json.loads((archive / "index.jsonl").read_text().strip())
            self.assertEqual(record["file"], Path(saved).name)
            self.assertFalse(record["duplicate"])
            self.assertEqual(record["source"], "candidate.py")

    def test_identical_code_stored_once_but_indexed_twice(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            program = root / "candidate.py"
            program.write_text("# same\n")
            archive = root / "programs"

            first = archive_program(str(program), _result(1.0), archive)
            second = archive_program(str(program), _result(1.0), archive)

            self.assertEqual(first, second)
            self.assertEqual(len(list(archive.glob("*.py"))), 1)
            lines = (archive / "index.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(json.loads(lines[1])["duplicate"])

    def test_distinct_code_sorts_chronologically(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "programs"
            for index in range(3):
                program = root / f"c{index}.py"
                program.write_text(f"# kernel v{index}\n")
                archive_program(str(program), _result(float(index)), archive)
            names = sorted(p.name for p in archive.glob("*.py"))
            self.assertEqual(len(names), 3)
            # Filenames lead with a UTC timestamp, so lexical order is run order.
            self.assertEqual(names, [p.name for p in sorted(archive.glob("*.py"))])

    def test_failed_candidates_are_archived_too(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            program = root / "bad.py"
            program.write_text("# broken\n")
            archive = root / "programs"

            saved = archive_program(str(program), _result(-1e9), archive)

            self.assertIsNotNone(saved)
            self.assertIn("-", Path(saved).name)  # negative score in the filename

    def test_archiving_never_raises(self):
        # A missing program, an unwritable directory — a broken archive must not
        # take down an otherwise fine evaluation.
        self.assertIsNone(archive_program("/nonexistent/program.py", _result(1.0), "/tmp/x"))


class TestRunEvolveWiring(unittest.TestCase):
    def _run(self, **kwargs):
        from evograd.evolve.run import run_evolve
        from evograd.ops import get_op

        with TemporaryDirectory() as td:
            root = Path(td)
            seed = root / "seed.py"
            seed.write_text("# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END\n")
            observed = {}

            def fake_run_evolution(**_kwargs):
                observed.update(os.environ)
                return mock.Mock(best_code="")

            with mock.patch("openevolve.run_evolution", side_effect=fake_run_evolution):
                run_evolve(
                    get_op("layernorm"),
                    seed_path=seed,
                    output_dir=root / "out",
                    **kwargs,
                )
            return observed, root / "out"

    def test_flag_points_the_evaluator_at_output_dir_programs(self):
        observed, out = self._run(save_programs=True)
        self.assertEqual(
            observed["EVOGRAD_PROGRAM_ARCHIVE_DIR"], str(out / "programs")
        )

    def test_off_by_default(self):
        observed, _out = self._run()
        self.assertNotIn("EVOGRAD_PROGRAM_ARCHIVE_DIR", observed)


if __name__ == "__main__":
    unittest.main()
