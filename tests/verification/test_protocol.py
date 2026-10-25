"""The benchmark protocol end to end on the toy integration (CPU).

``evaluate`` verifies, then times every executable candidate, then judges and
times the combination:

* a native-equivalent candidate is accepted, timed at Tier 2 and Tier 3, and
  -- when its single-site interval is above one -- forms the combination that
  passes model Gate 2 and Gate 3 and is counted at every level;
* a 5% gradient fault is executable, so it is timed, but it is not accepted:
  it appears with its runtime and is counted nowhere;
* a candidate that raises is not executable and is not timed;
* rerunning the same command reuses every stage.

Tier 2 needs CUDA events, so a fixed timer stands in for it; Tier 3 runs for
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
    def time_tier2(op, module, workload):
        return {"eager": {"route": "autograd", "speedup": speedup, "ci95": [speedup, speedup]},
                "torch_compile": {"route": "autograd", "speedup": speedup / 2,
                                  "ci95": [speedup / 2, speedup / 2]}}
    return time_tier2


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
                            time_tier2=timer, isolate_tier3=False, rules=rules)

    def test_an_accepted_candidate_is_timed_combined_and_counted(self):
        out = Path(self.tmp.name) / "good"
        legacy_cache = out / "tier1" / f"{self.T.CASE_ID.replace('/', '__')}.json"
        legacy_cache.parent.mkdir(parents=True, exist_ok=True)
        legacy_cache.write_text(json.dumps({"result": {"tier1": {"eager": {"speedup": 999}}}}))
        result = self.evaluate("good", out)
        case = result["cases"][self.T.CASE_ID]
        self.assertTrue(case["accepted"] and case["executable"], case)
        self.assertEqual(case["tier2"]["eager"]["speedup"], 1.5)
        self.assertNotIn("tier1", case)
        self.assertNotIn("tier1", result["settings"])
        cache = json.loads((out / "tier2" / legacy_cache.name).read_text())
        self.assertEqual(cache["stage_key"]["protocol"], result["protocol"])
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
        single = summary["level2"]["tier3_single_site"]["toy"][self.T.CASE_ID]
        self.assertTrue(single["positive"] and single["selected_for_combination"])
        self.assertEqual(single["gate3"], "passed")
        self.assertEqual(single["tier3"], case["tier3_single_site"])
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
        self.assertEqual(case["tier2"]["eager"]["speedup"], 1.5)
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
        self.assertNotIn("tier2", case)
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
        for key in ("gate1_payloads", "gate2_operator", "tier2", "tier3_single_site"):
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
        other["cases"][self.T.CASE_ID]["tier2"]["eager"]["speedup"] = 6.0
        pooled = summarize([good, other])
        self.assertEqual(pooled["models"], ["toy", "toy2"])
        self.assertEqual(pooled["level2"]["funnel"]["configurations"], 2)
        # One task: the geometric mean of 1.5 and 6.0.
        self.assertAlmostEqual(pooled["level2"]["speedup"]["eager"]["geomean"], 3.0)
        self.assertEqual(pooled["level3"]["speedup"]["models_covered"], 2)


class Tier2Timing(unittest.TestCase):
    def test_eager_compile_and_library_use_the_same_deployment_autograd_route(self):
        from dataclasses import replace
        import types

        import torch

        from evograd.benchmark import get_task
        from evograd.evaluation.common.provider import clone_values, pytorch_autograd_provider
        from evograd.evaluation.protocol import TIER2_OPTIONS, time_case
        from evograd.opdecl.activity import Workload
        from evograd.opdecl.inputs import make_case_inputs, upstream_grad_values
        from evograd.pipelines.shared.artifact import render_deployment_layer
        from tests.verification import _toy_integration as T

        op = get_task("swiglu")
        module = types.ModuleType("tier2_direct_deployment")
        source = (T.CANDIDATE.format(scale=1.0)
                  + f"\n{op.forward_fn_name} = forward_with_saved\n"
                  + f"{op.backward_fn_name} = backward_from_saved\n"
                  + render_deployment_layer(op))
        exec(compile(source, "<tier2-test>", "exec"), module.__dict__)
        hook = types.SimpleNamespace(
            pair_factory=lambda: (module.forward_with_saved, module.backward_from_saved),
            forward_args=("a", "b"), backward_extras=())
        op = replace(op, performance_baselines={"library": hook})
        workload = Workload(dims={"rows": 4, "cols": 8}, dtype="float32")
        measured = []

        def measure(op, candidate, baseline, *, workloads, device, **options):
            self.assertEqual(options, TIER2_OPTIONS)
            self.assertEqual(candidate.adapter_kind, "candidate_direct_deployment_autograd")
            self.assertIn("autograd", baseline.adapter_kind)
            values = make_case_inputs(op, workloads[0], device=device)
            outputs, gradients = [], []
            for provider in (candidate, baseline):
                own = clone_values(values)
                output, saved = provider.forward(own)
                self.assertIsNotNone(output.grad_fn)
                outputs.append(output)
                gradients.append(provider.backward(upstream_grad_values(op, own), saved, own))
            torch.testing.assert_close(outputs[0], outputs[1])
            for actual, expected in zip(*gradients):
                torch.testing.assert_close(actual, expected)
            measured.append(baseline.name)
            return {"aggregate": {
                "speedup_pair_full": 1.2, "speedup_pair_full_ci95": {"low": 1.1, "high": 1.3},
                "candidate_pair_full_ms": 1.0, "baseline_pair_full_ms": 1.2,
                "speedup_forward": 1.1, "speedup_backward": 1.3}}

        with mock.patch("evograd.evaluation.tier1.fair.run_fair_benchmarks", measure), \
                mock.patch("evograd.evaluation.common.provider.torch_compile_provider",
                           side_effect=lambda op, **kwargs: pytorch_autograd_provider(op)), \
                mock.patch("evograd.opdecl.baselines.verify_performance_baseline"), \
                mock.patch("torch.autograd.grad", wraps=torch.autograd.grad) as backward:
            result = time_case(op, module, workload, device="cpu")
        self.assertEqual(set(result), {"eager", "torch_compile", "library"})
        self.assertEqual(len(measured), 3, result)
        self.assertEqual(backward.call_count, 6)
        for row in result.values():
            self.assertEqual(row["route"], "autograd")
            self.assertEqual(row["speedup"], 1.2)

    def test_summary_rejects_legacy_tier1_instead_of_relabeling_it(self):
        from evograd.evaluation.protocol import summarize

        with self.assertRaisesRegex(ValueError, "Legacy Tier 1"):
            summarize([{"protocol": "evograd-evaluation-v1", "model": "legacy", "cases": {}}])


class Summary(unittest.TestCase):
    def evaluation(self, *, selected=True):
        from evograd.evaluation.protocol import PROTOCOL

        return {"protocol": PROTOCOL, "model": "toy", "cases": {
            "swiglu/site": {"task": "swiglu", "level": 2, "family": "activation",
                            "submitted": True, "executable": True, "accepted": True,
                            "gate1_payloads": "passed", "gate2_operator": "passed",
                            "gate2_model": "passed", "gate3": "passed",
                            "tier2": {"eager": {"error": "timer failed"},
                                      "torch_compile": {"error": "compile failed"}},
                            "tier3_single_site": {"ok": True, "speedup": 1.2, "step_ms": 10,
                                                  "ci95": {"low": 1.1, "high": 1.3}}}},
                "tier3": {"combination": {"cases": ["swiglu/site"] if selected else []}}}

    def test_correctness_coverage_does_not_require_successful_tier2_timing(self):
        from evograd.evaluation.protocol import summarize

        level2 = summarize([self.evaluation()])["level2"]
        self.assertEqual(level2["coverage"], 1.0)
        self.assertEqual(level2["funnel"]["accepted"], 1)
        for baseline in ("eager", "torch_compile"):
            self.assertEqual(level2["speedup"][baseline]["configurations_covered"], 0)
            self.assertIsNone(level2["speedup"][baseline]["geomean"])

    def test_single_sites_keep_gates_intervals_and_selection_without_pooling(self):
        from evograd.evaluation.protocol import summarize

        good = self.evaluation()
        other = self.evaluation(selected=False)
        other["model"] = "other"
        single = summarize([good, other])["level2"]["tier3_single_site"]
        self.assertEqual(set(single), {"toy", "other"})
        for model in single:
            row = single[model]["swiglu/site"]
            self.assertTrue(row["accepted"] and row["positive"])
            self.assertEqual(row["selected_for_combination"], model == "toy")
            self.assertEqual(row["tier3"]["ci95"], {"low": 1.1, "high": 1.3})
            for gate in ("gate1_payloads", "gate2_operator", "gate2_model", "gate3"):
                self.assertEqual(row[gate], "passed")
        self.assertNotIn("geomean", single)

    def test_rejected_or_unproven_single_sites_are_not_positive(self):
        from evograd.evaluation.protocol import summarize

        for condition in ("rejected", "interval_touches_one", "timing_failed", "not_submitted"):
            with self.subTest(condition=condition):
                evaluation = self.evaluation(selected=False)
                row = evaluation["cases"]["swiglu/site"]
                if condition == "rejected":
                    row.update(accepted=False, gate3="failed")
                elif condition == "interval_touches_one":
                    row["tier3_single_site"]["ci95"]["low"] = 1.0
                elif condition == "timing_failed":
                    row["tier3_single_site"] = {"ok": False, "error": "timing failed"}
                else:
                    row = evaluation["cases"]["swiglu/site"] = {
                        "task": "swiglu", "level": 2, "family": "activation", "submitted": False}
                single = summarize([evaluation])["level2"]["tier3_single_site"]["toy"]["swiglu/site"]
                self.assertFalse(single["positive"] or single["selected_for_combination"])
                self.assertEqual(single["tier3"], row.get("tier3_single_site", {}))


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
