import tempfile
from pathlib import Path
import unittest

from evograd.gepa_backend.archive import (
    best_generalist_index,
    pareto_union,
    shape_winner_map,
)
from evograd.gepa_backend.candidate import EvolveBlockTemplate
from evograd.gepa_backend.dispatch import optimize_contiguous
from evograd.gepa_backend.evaluator import ShapeBatchEvaluator, examples_for_suite
from evograd.gepa_backend.worker import _score_case


BODY = """
def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    return x, (x, weight)

def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    x, weight = saved_tensors
    return dy, weight, weight
""".strip()


class TestCandidateContract(unittest.TestCase):
    def setUp(self):
        self.source = (
            "import torch\n# EVOLVE-BLOCK-START\n"
            + BODY
            + "\n# EVOLVE-BLOCK-END\nOUTSIDE = 1\n"
        )
        self.template = EvolveBlockTemplate.from_source(self.source)

    def test_round_trip_and_scope(self):
        rendered = self.template.render(BODY + "\n# harmless")
        self.template.assert_scope(rendered)
        self.assertIn("# harmless", rendered)

    def test_rejects_markers_and_high_level_fallback(self):
        with self.assertRaises(ValueError):
            self.template.render(BODY + "\n# EVOLVE-BLOCK-END")
        with self.assertRaises(ValueError):
            self.template.render(BODY + "\ntorch.autograd.grad(x, x)")


class TestShapeScoring(unittest.TestCase):
    def test_case_score_is_min_speedup_with_memory_penalty(self):
        case = {
            "dims": {"rows": 8, "hidden": 1024},
            "dtype": "bfloat16",
            "speedup_vs_baseline_backward": 2.0,
            "speedup_vs_baseline_raw_full_step": 1.5,
            "saved_bytes": 50.0,
            "input_bytes": 100.0,
            "backward_from_saved_ms": 1.0,
            "raw_forward_backward_full_step_ms": 2.0,
            "baseline_backward_ms": 2.0,
            "baseline_raw_full_step_ms": 3.0,
        }
        score, info = _score_case(case)
        self.assertAlmostEqual(score, 1.5 / 1.025)
        self.assertAlmostEqual(info["saved_memory_ratio"], 0.5)

    def test_layernorm_suite_catalogs(self):
        self.assertEqual(len(examples_for_suite("layernorm", "tb_mixed")), 9)
        self.assertEqual(len(examples_for_suite("layernorm", "tb_sweep")), 16)

    def test_persistent_shape_cache(self):
        class FakeEvaluator(ShapeBatchEvaluator):
            calls = 0

            def _run_worker(self, *, source, shapes, run_gate):
                del source, run_gate
                self.calls += 1
                return {
                    "gate_ok": True,
                    "shapes": {
                        shape["id"]: {
                            "score": 1.25,
                            "info": {"shape_id": shape["id"], "status": "ok"},
                        }
                        for shape in shapes
                    },
                }

        with tempfile.TemporaryDirectory() as temporary:
            seed = Path(temporary) / "seed.py"
            seed.write_text(
                "import torch\n# EVOLVE-BLOCK-START\n"
                + BODY
                + "\n# EVOLVE-BLOCK-END\n",
                encoding="utf-8",
            )
            evaluator = FakeEvaluator(seed_path=seed, cache_dir=Path(temporary) / "cache")
            shape = {"id": "r4_h1024", "dims": {"rows": 4, "hidden": 1024}, "dtype": "bfloat16"}
            self.assertEqual(evaluator.evaluate(BODY, shape)[0], 1.25)
            self.assertEqual(evaluator.evaluate(BODY, shape)[0], 1.25)
            self.assertEqual(evaluator.calls, 1)


class TestArchiveAndDispatch(unittest.TestCase):
    def test_archive_helpers(self):
        result = {
            "val_aggregate_scores": [1.0, 1.5, 1.2],
            "per_val_instance_best_candidates": {"0": [1], "1": [2]},
        }
        valset = [{"id": "small"}, {"id": "large"}]
        self.assertEqual(best_generalist_index(result), 1)
        self.assertEqual(pareto_union(result), [1, 2])
        self.assertEqual(shape_winner_map(result, valset), {"small": [1], "large": [2]})

    def test_contiguous_optimizer_preserves_specialists(self):
        rows = [4, 16, 64, 256]
        score, segments = optimize_contiguous(
            rows,
            {
                0: [2.0, 2.0, 0.8, 0.8],
                1: [0.8, 0.8, 2.0, 2.0],
                2: [1.1, 1.1, 1.1, 1.1],
            },
            max_segments=2,
        )
        self.assertGreater(score, 0.0)
        self.assertEqual(segments, [(0, 2, 0), (2, 4, 1)])


if __name__ == "__main__":
    unittest.main()
