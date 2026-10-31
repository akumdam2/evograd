"""Case-major ordering and one eager/compile measurement across agents and combinations."""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evograd.evaluation import case_group, protocol
from evograd.opdecl.activity import Workload
from evograd.verification.integrations.base import Case
from evograd.verification.integrations.gates import Run, Suspended
from tests.verification import _toy_integration as T


class FakeCorpus:
    sha256 = "same"
    manifest_path = Path("manifest.json")

    def binding(self):
        return {"corpus": "same"}

    def load(self, record, **_kwargs):
        return {"case": record["case"]}

    def _read_storage(self, key, device):
        raise AssertionError("No tensor storage needed in scheduler tests")


class CaseGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.model = T.model()
        self.cases = [Case("swiglu/a", "swiglu", 2, Workload({"rows": 8, "cols": 16}, "float32"), 1),
                      Case("swiglu/b", "swiglu", 1, Workload({"rows": 8, "cols": 16}, "float32"), 1)]
        self.plan = {"model": "toy", "state": str(root / "state"), "jobs": []}
        for agent in ("gepa", "codex"):
            program = root / f"{agent}.py"
            program.write_text("# frozen test candidate\n")
            self.plan["jobs"].append({"out": str(root / agent), "selection": {
                "agent": agent, "corpus": "manifest.json", "candidates": {
                    c.case_id: {"path": str(program)} for c in self.cases}}})
        self.events = []

    def open_run(self, model, paths, corpus, out, *, state, log, device, compile_backend):
        agent = Path(out).name
        return Run(model, Path(out), Path(state), FakeCorpus(), Path(corpus),
                   {c.case_id: c for c in self.cases}, {cid: "hash" for cid in paths}, paths,
                   {cid: SimpleNamespace(agent=agent) for cid in paths}, device=device,
                   log=log, _workload=T.ToyWorkload())

    def verify(self, run):
        cid = next(iter(run.modules))
        agent = run.modules[cid].agent
        self.events.append(("verify", cid, agent))
        return {"candidates": {cid: {"level": run.cases[cid].level, "accepted": True,
                "gate1_payloads": "passed", "gate2_operator": "passed",
                "gate2_model": "passed", "gate3": "passed"}}}

    def timing(self, op, modules, inputs, **kwargs):
        cid = inputs[0][1]()["case"]
        self.events.append(("tier2", cid, tuple(modules)))
        return {"rows": {agent: {baseline: {"speedup": 2, "baseline_ms": 2, "candidate_ms": 1}
                                 for baseline in ("eager", "torch_compile")} for agent in modules}}

    def provider(self, run, name, spec, options):
        self.events.append(("tier3", name))
        ms = 100 if name == "eager" else 70 if name == "torch_compile" else 60 if name.endswith("combination") else 80
        return {"ok": True, "step_ms": ms, "per_block_ms": [ms] * 3,
                "peak_memory_bytes": 100, "cpu_bound_fraction": .1,
                **({"correctness": {"status": "passed"}} if name == "torch_compile" else {})}

    def combination(self, run, chosen):
        self.events.append(("combination", run.out.name, tuple(chosen)))
        return {"cases": chosen, "accepted": bool(chosen), "gate2_model": "passed", "gate3": "passed"}

    def execute(self, provider=None):
        with ExitStack() as stack:
            stack.enter_context(T.toy_environment())
            stack.enter_context(patch("evograd.verification.integrations.base.observed_cases", return_value=self.cases))
            stack.enter_context(patch("evograd.verification.integrations.gates.open_run", side_effect=self.open_run))
            stack.enter_context(patch("evograd.verification.integrations.gates.verify_cases", side_effect=self.verify))
            stack.enter_context(patch("evograd.verification.integrations.gates.verify_combination", side_effect=self.combination))
            stack.enter_context(patch.object(protocol, "tier2_payloads", side_effect=lambda corpus, op, case: [{"case": case.case_id}]))
            stack.enter_context(patch.object(protocol, "tier2_binding", return_value={"order": 303}))
            stack.enter_context(patch.object(protocol, "check_executable"))
            stack.enter_context(patch.object(protocol, "whole_model_compile", return_value=object()))
            return case_group.evaluate_group(self.plan, model=self.model, deadline=time.time() + 60,
                                             time_tier2=self.timing, time_provider=provider or self.provider,
                                             rules=False, check_imports=False, device="cpu", log=lambda *a: None)

    def test_cases_then_all_single_sites_then_combinations_and_one_model_baseline(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.events[:6], [
            ("verify", "swiglu/a", "gepa"), ("verify", "swiglu/a", "codex"),
            ("tier2", "swiglu/a", ("gepa", "codex")),
            ("verify", "swiglu/b", "gepa"), ("verify", "swiglu/b", "codex"),
            ("tier2", "swiglu/b", ("gepa", "codex"))])
        names = [event[1] for event in self.events if event[0] == "tier3"]
        self.assertEqual(names.count("eager"), 1)
        self.assertEqual(names.count("torch_compile"), 1)
        self.assertEqual(set(names), {"eager", "torch_compile", "gepa::swiglu/a", "codex::swiglu/a",
                                      "gepa::combination", "codex::combination"})
        for job in self.plan["jobs"]:
            result = json.loads((Path(job["out"]) / "evaluation.json").read_text())
            self.assertEqual(result["tier3"]["combination"]["cases"], ["swiglu/a"])
            self.assertAlmostEqual(result["tier3"]["combination"]["speedup_vs_torch_compile"], 70 / 60)
            self.assertEqual(result["settings"]["group_execution"], case_group.GROUP)
        self.events.clear()
        self.assertEqual(self.execute(), 0)
        self.assertFalse(any(e[0] in ("tier2", "tier3") for e in self.events))

    def test_resume_does_not_remeasure_finished_baselines(self):
        fail = [True]
        successful = Counter()

        def provider(run, name, spec, options):
            if name == "gepa::swiglu/a" and fail[0]:
                fail[0] = False
                raise Suspended("allocation deadline")
            successful[name] += 1
            return self.provider(run, name, spec, options)

        self.assertEqual(self.execute(provider), 3)
        self.assertEqual(self.execute(provider), 0)
        self.assertEqual(successful["eager"], 1)
        self.assertEqual(successful["torch_compile"], 1)
        self.assertTrue(all(count == 1 for count in successful.values()))

    def test_empty_selection_keeps_baselines_in_agent_report(self):
        with patch.object(protocol, "_positive", return_value=False):
            self.assertEqual(self.execute(), 0)
        names = [e[1] for e in self.events if e[0] == "tier3"]
        self.assertEqual(names.count("torch_compile"), 1)
        self.assertFalse(any(name.endswith("::combination") for name in names))
        for job in self.plan["jobs"]:
            result = json.loads((Path(job["out"]) / "evaluation.json").read_text())
            self.assertTrue(result["tier3"]["baselines"]["torch_compile"]["ok"])
            self.assertEqual(result["tier3"]["combination"]["cases"], [])
            self.assertIn("torch_compile", result["summary"]["level3"]["models"]["toy"]["baselines"])

    def test_no_executable_candidates_still_measures_baselines(self):
        for job in self.plan["jobs"]:
            job["selection"]["candidates"] = {}
        self.assertEqual(self.execute(), 0)
        names = [e[1] for e in self.events if e[0] == "tier3"]
        self.assertEqual(set(names), {"eager", "torch_compile"})
        for job in self.plan["jobs"]:
            result = json.loads((Path(job["out"]) / "evaluation.json").read_text())
            self.assertTrue(result["tier3"]["baselines"]["torch_compile"]["ok"])
            self.assertFalse(result["tier3"]["single_site"])

    def test_compile_failure_is_explicit_and_candidates_continue(self):
        def provider(run, name, spec, options):
            if name == "torch_compile":
                return {"ok": False, "failed_at": "compile", "error": "compiler failed"}
            return self.provider(run, name, spec, options)
        self.assertEqual(self.execute(provider), 0)
        for job in self.plan["jobs"]:
            result = json.loads((Path(job["out"]) / "evaluation.json").read_text())
            self.assertEqual(result["tier3"]["baselines"]["torch_compile"]["error"], "compiler failed")
            self.assertTrue(result["tier3"]["combination"]["tier3"]["ok"])
