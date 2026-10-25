"""The benchmark protocol end to end on the toy integration (CPU).

``evaluate`` verifies, then times every executable candidate, then judges and
times the combination:

* a native-equivalent candidate is accepted, timed at Tier 1 and Tier 3, and
  -- when its single-site interval is above one -- forms the combination that
  passes model Gate 2 and Gate 3 and is counted at every level;
* a 5% gradient fault is executable, so it is timed, but it is not accepted:
  it appears with its runtime and is counted nowhere;
* a candidate that raises is not executable and is not timed;
* rerunning the same command reuses every stage.

Tier 1 needs CUDA events, so a fixed timer stands in for it; Tier 3 runs for
real on the toy model.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

warnings.filterwarnings("ignore")

FAST = "aot_eager"
TIER3 = {"warmup": 1, "steps": 2, "blocks": 3, "loss_steps": 2, "learning_rate": 1e-4, "seed": 0}

RAISES = '''
import torch


def forward_with_saved(a, b):
    return torch.nn.functional.silu(a) * b, (a, b)


def backward_from_saved(dc, saved):
    raise RuntimeError("illegal memory access")
'''

#: The toy pair spelled within the static rules (no torch.nn.functional).
RULED = '''
import torch


def forward_with_saved(a, b):
    return a * torch.sigmoid(a) * b, (a, b)


def backward_from_saved(dc, saved):
    a, b = saved
    s = torch.sigmoid(a)
    return dc * b * (s * (1 + a * (1 - s))), dc * a * s
'''

#: A submission's shape: its module inside one evolve block.
BLOCK = "# EVOLVE-BLOCK-START\n{}\n# EVOLVE-BLOCK-END\n"


def fixed_timer(speedup):
    def time_tier1(op, module, workload):
        return {"eager": {"route": "autograd", "speedup": speedup, "ci95": [speedup, speedup]},
                "torch_compile": {"route": "autograd", "speedup": speedup / 2,
                                  "ci95": [speedup / 2, speedup / 2]}}
    return time_tier1


def faster_than_eager(report, reference="eager", *, seed=0):
    """Every provider's single-site interval read as above one."""
    from evograd.evaluation.tier3 import runner

    real = runner.__dict__["_real_speedup_intervals"](report, reference, seed=seed)
    for row in (real.get("vs_reference") or {}).values():
        row["ci95"] = {"iterations": 1, "low": 1.1, "high": 1.3}
    return real


class Evaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3 import runner

        runner.__dict__.setdefault("_real_speedup_intervals", runner.speedup_intervals)
        from evograd.verification.integrations import base
        from tests.verification import _toy_integration as T

        cls.T = T
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model = T.model()
        with T.toy_environment():
            cls.manifest = base.capture(cls.model, root / "corpus", device="cpu", log=lambda *a: None)
        cls.programs = {}
        for name, source in (("good", T.CANDIDATE.format(scale=1.0)),
                             ("fault", T.CANDIDATE.format(scale=1.05)), ("raises", RAISES),
                             ("violates", BLOCK.format(T.CANDIDATE.format(scale=1.0))),
                             ("ruled", BLOCK.format(RULED))):
            path = root / f"{name}.py"
            path.write_text(source)
            cls.programs[name] = str(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def evaluate(self, program, out, *, timer=fixed_timer(1.5), intervals=faster_than_eager,
                 rules=False):
        """The toy pair is a bare module, not an artifact: the static rules are
        off unless a test asks for them."""
        from evograd.evaluation.protocol import evaluate

        with self.T.toy_environment(), \
                mock.patch("evograd.evaluation.tier3.runner.speedup_intervals", intervals):
            return evaluate(self.model, {self.T.CASE_ID: self.programs[program]}, self.manifest,
                            out, state=Path(self.tmp.name) / "state", device="cpu",
                            compile_backend=FAST, log=lambda *a: None, tier3_options=TIER3,
                            time_tier1=timer, isolate_tier3=False, rules=rules)

    def test_an_accepted_candidate_is_timed_combined_and_counted(self):
        out = Path(self.tmp.name) / "good"
        result = self.evaluate("good", out)
        case = result["cases"][self.T.CASE_ID]
        self.assertTrue(case["accepted"] and case["executable"], case)
        self.assertEqual(case["tier1"]["eager"]["speedup"], 1.5)
        self.assertTrue(case["tier3_single_site"]["ok"], case["tier3_single_site"])
        combination = result["tier3"]["combination"]
        self.assertEqual(combination["cases"], [self.T.CASE_ID])
        self.assertEqual((combination["gate2_model"], combination["gate3"]), ("passed", "passed"))
        self.assertTrue(combination["accepted"])
        self.assertTrue(combination["tier3"]["ok"], combination["tier3"])
        summary = result["summary"]
        self.assertEqual(summary["level2"]["coverage"], 1.0)
        self.assertAlmostEqual(summary["level2"]["speedup"]["eager"]["geomean"], 1.5)
        self.assertAlmostEqual(summary["level2"]["speedup"]["torch_compile"]["geomean"], 0.75)
        self.assertEqual(summary["level2"]["funnel"],
                         {"configurations": 1, "submitted": 1, "violation": 0, "executable": 1,
                          "gate1": 1, "accepted": 1})
        self.assertEqual(summary["level3"]["coverage"], 1.0)
        self.assertIsNotNone(summary["level3"]["models"]["toy"]["speedup"])
        self.assertTrue((out / "evaluation.json").exists())

        # Rerun: every stage comes back from DIR; nothing is timed again.
        def never(*args, **kwargs):
            raise AssertionError("a finished stage was timed again")

        again = self.evaluate("good", out, timer=never)
        self.assertEqual(again["cases"], result["cases"])
        self.assertEqual(again["summary"], result["summary"])

    def test_a_rejected_candidate_is_timed_but_counted_nowhere(self):
        result = self.evaluate("fault", Path(self.tmp.name) / "fault")
        case = result["cases"][self.T.CASE_ID]
        self.assertTrue(case["executable"])
        self.assertFalse(case["accepted"])
        self.assertEqual(case["gate2_operator"], "failed")
        self.assertEqual(case["tier1"]["eager"]["speedup"], 1.5)
        self.assertIn("tier3_single_site", case)
        self.assertEqual(result["tier3"]["combination"]["cases"], [])
        summary = result["summary"]
        self.assertEqual(summary["level2"]["coverage"], 0.0)
        self.assertIsNone(summary["level2"]["speedup"]["eager"]["geomean"])
        self.assertEqual(summary["level2"]["funnel"]["executable"], 1)
        self.assertEqual(summary["level2"]["funnel"]["accepted"], 0)
        self.assertEqual(summary["level3"]["coverage"], 0.0)

    def test_a_candidate_that_raises_is_not_timed(self):
        def never(*args, **kwargs):
            raise AssertionError("a non-executable candidate was timed")

        result = self.evaluate("raises", Path(self.tmp.name) / "raises", timer=never)
        case = result["cases"][self.T.CASE_ID]
        self.assertFalse(case["executable"])
        self.assertIn("illegal memory access", case["error"])
        self.assertNotIn("tier1", case)
        self.assertNotIn("tier3_single_site", case)
        self.assertFalse(case["accepted"])
        self.assertEqual(result["summary"]["level2"]["funnel"]["executable"], 0)

    def test_a_submission_breaking_the_static_rules_is_neither_verified_nor_timed(self):
        def never(*args, **kwargs):
            raise AssertionError("a violating candidate was timed")

        # The toy pair calls torch.nn.functional, which the rules deny.
        result = self.evaluate("violates", Path(self.tmp.name) / "violates", timer=never,
                               rules=True)
        case = result["cases"][self.T.CASE_ID]
        self.assertIn("nn.functional", case["violation"])
        self.assertFalse(case["accepted"] or case["executable"])
        for key in ("gate1_payloads", "gate2_operator", "tier1", "tier3_single_site"):
            self.assertNotIn(key, case)
        self.assertEqual(result["tier3"]["combination"]["cases"], [])
        funnel = result["summary"]["level2"]["funnel"]
        self.assertEqual((funnel["violation"], funnel["executable"]), (1, 0))

        # A bare module has no block for the rules to span.
        bare = self.evaluate("good", Path(self.tmp.name) / "bare", timer=never, rules=True)
        self.assertIn("EVOLVE-BLOCK", bare["cases"][self.T.CASE_ID]["violation"])

        # The same math spelled within the rules goes on to be verified and timed.
        ruled = self.evaluate("ruled", Path(self.tmp.name) / "ruled", rules=True)
        case = ruled["cases"][self.T.CASE_ID]
        self.assertNotIn("violation", case)
        self.assertTrue(case["executable"])
        self.assertIn("gate1_payloads", case)

    def test_an_interval_that_reaches_one_leaves_the_candidate_out_of_the_combination(self):
        def straddles(report, reference="eager", *, seed=0):
            real = faster_than_eager(report, reference, seed=seed)
            for row in (real.get("vs_reference") or {}).values():
                row["ci95"]["low"] = 0.98
            return real

        result = self.evaluate("good", Path(self.tmp.name) / "straddles", intervals=straddles)
        self.assertTrue(result["cases"][self.T.CASE_ID]["accepted"])
        self.assertEqual(result["tier3"]["combination"]["cases"], [])
        self.assertNotIn("tier3", result["tier3"]["combination"])
        self.assertEqual(result["summary"]["level2"]["coverage"], 1.0)
        self.assertEqual(result["summary"]["level3"]["coverage"], 0.0)

    def test_summaries_pool_models_by_level(self):
        from evograd.evaluation.protocol import summarize

        good = self.evaluate("good", Path(self.tmp.name) / "pooled")
        other = json.loads(json.dumps(good))
        other["model"] = "toy2"
        other["cases"][self.T.CASE_ID]["tier1"]["eager"]["speedup"] = 6.0
        pooled = summarize([good, other])
        self.assertEqual(pooled["models"], ["toy", "toy2"])
        self.assertEqual(pooled["level2"]["funnel"]["configurations"], 2)
        # One task: the geometric mean of 1.5 and 6.0.
        self.assertAlmostEqual(pooled["level2"]["speedup"]["eager"]["geomean"], 3.0)
        self.assertEqual(pooled["level3"]["speedup"]["models_covered"], 2)


class Executable(unittest.TestCase):
    def test_wrong_output_shape_is_not_executable(self):
        import types

        from evograd.benchmark import get_task
        from evograd.opdecl.activity import Workload
        from evograd.evaluation.protocol import NotExecutable, check_executable

        module = types.ModuleType("wrong_shape")
        exec('''
import torch
def forward_with_saved(a, b):
    return (torch.nn.functional.silu(a) * b)[:1], (a, b)
def backward_from_saved(dc, saved):
    a, b = saved
    return torch.zeros_like(a), torch.zeros_like(b)
''', module.__dict__)
        op = get_task("swiglu")
        with self.assertRaises(NotExecutable):
            check_executable(op, module, Workload(dims={"rows": 4, "cols": 8}, dtype="float32"),
                             device="cpu")


if __name__ == "__main__":
    unittest.main()
