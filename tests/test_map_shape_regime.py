"""Unit tests for regime MAP metrics and archive elite harvest."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evograd.evolve.map_harvest import harvest_regime_elites
from evograd.evolve.scoring import regime_speedup_metrics
from evograd.opdecl.activity import Workload
from evograd.ops import get_op


class TestRegimeSpeedupMetrics(unittest.TestCase):
    def test_partitions_cases_by_regime_feature(self):
        op = get_op("layernorm")

        def feature(workload: Workload) -> float:
            return float(workload.dims["rows"])

        cases = [
            {
                "dims": {"rows": 128, "hidden": 1024},
                "dtype": "bfloat16",
                "speedup_vs_baseline_raw_full_step": 2.0,
            },
            {
                "dims": {"rows": 256, "hidden": 1024},
                "dtype": "bfloat16",
                "speedup_vs_baseline_raw_full_step": 8.0,
            },
            {
                "dims": {"rows": 8192, "hidden": 1024},
                "dtype": "bfloat16",
                "speedup_vs_baseline_raw_full_step": 3.0,
            },
        ]
        metrics = regime_speedup_metrics(
            cases, regime_feature=feature, regime_split=4096.0
        )
        self.assertAlmostEqual(metrics["small_regime_speedup"], 4.0)
        self.assertAlmostEqual(metrics["large_regime_speedup"], 3.0)
        self.assertEqual(op.regime_split, 4096.0)

    def test_missing_cases_emit_zeros(self):
        metrics = regime_speedup_metrics(
            None, regime_feature=lambda w: 1.0, regime_split=4096.0
        )
        self.assertEqual(metrics["small_regime_speedup"], 0.0)
        self.assertEqual(metrics["large_regime_speedup"], 0.0)


class TestMapHarvest(unittest.TestCase):
    def test_selects_regime_elites_from_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "checkpoint_3"
            programs_dir = checkpoint / "programs"
            programs_dir.mkdir(parents=True)

            def write_program(program_id: str, metrics: dict, code: str):
                payload = {"id": program_id, "metrics": metrics, "code": code}
                (programs_dir / f"{program_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            write_program(
                "a",
                {
                    "combined_score": 1.5,
                    "small_regime_speedup": 1.1,
                    "large_regime_speedup": 1.0,
                },
                "code_a = 1\n",
            )
            write_program(
                "b",
                {
                    "combined_score": 1.2,
                    "small_regime_speedup": 2.5,
                    "large_regime_speedup": 0.9,
                },
                "code_b = 1\n",
            )
            write_program(
                "c",
                {
                    "combined_score": 1.1,
                    "small_regime_speedup": 0.8,
                    "large_regime_speedup": 3.0,
                },
                "code_c = 1\n",
            )
            write_program(
                "d",
                {
                    "combined_score": 9.0,
                    "correct": 0.0,
                    "small_regime_speedup": 9.0,
                    "large_regime_speedup": 9.0,
                },
                "code_d = 1\n",
            )
            for program_id in ("a", "b", "c"):
                path = programs_dir / f"{program_id}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["metrics"]["correct"] = 1.0
                path.write_text(json.dumps(payload), encoding="utf-8")
            (checkpoint / "metadata.json").write_text(
                json.dumps({"archive": ["a", "b", "c", "d"]}),
                encoding="utf-8",
            )

            report = harvest_regime_elites(root, archive_only=True)
            self.assertEqual(report["selected"]["full"]["id"], "a")
            self.assertEqual(report["selected"]["small"]["id"], "b")
            self.assertEqual(report["selected"]["large"]["id"], "c")
            self.assertEqual(report["correct_pool_size"], 3)
            self.assertTrue(report["distinct_regime_elites"])
            self.assertIn("code_b", Path(report["programs"]["small"]).read_text())
            self.assertIn("code_c", Path(report["programs"]["large"]).read_text())

    def test_rejects_archive_without_correct_programs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "checkpoint_1"
            programs_dir = checkpoint / "programs"
            programs_dir.mkdir(parents=True)
            (programs_dir / "bad.json").write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "metrics": {
                            "correct": 0.0,
                            "combined_score": -1e6,
                            "small_regime_speedup": 0.0,
                            "large_regime_speedup": 0.0,
                        },
                        "code": "bad = True\n",
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "metadata.json").write_text(
                json.dumps({"archive": ["bad"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "no correct programs"):
                harvest_regime_elites(root)


if __name__ == "__main__":
    unittest.main()
