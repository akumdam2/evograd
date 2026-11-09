"""Seed acceptance: Gate 1 on a task pack's search payloads, then the deployment."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from evograd.adbench_tools import TASK_SCHEMA
from evograd.adbench_tools.task import AgentTask
from evograd.evaluation.common import seed
from evograd.opdecl import Active, Provenance, Workload, declare_op
from evograd.pipelines.shared import runner
from evograd.pipelines.shared.artifact import render_deployment_layer
from evograd.verification.payloads import gate1 as payload_gate1

CASE = Workload(dims={"rows": 8, "hidden": 4}, dtype="float32",
                provenance=Provenance(model="synthetic", component="affine", source="handpicked"))


def affine_reference(x, weight, bias):
    return x * weight + bias


def fixture_op():
    """A pack's scoped declaration: one workload, the case's."""
    return declare_op(
        name="seed_affine", level=1, family="test", dims=("rows", "hidden"),
        args=(Active("x", "[rows, hidden]"), Active("weight", "[hidden]"),
              Active("bias", "[hidden]")),
        output=Active("y", "[rows, hidden]"), parameter_args=("weight", "bias"),
        forward="tests.test_seed_acceptance:affine_reference",
        forward_semantics="Elementwise affine transform.",
        backward_semantics="Return dx, dweight, dbias.",
        benchmark=(CASE,),
    )


def split_reference(x, weight):
    scaled = x * weight
    return scaled, scaled.sum(-1)


def multi_output_op():
    return declare_op(
        name="seed_split", level=1, family="test", dims=("rows", "hidden"),
        args=(Active("x", "[rows, hidden]"), Active("weight", "[hidden]")),
        output=(Active("scaled", "[rows, hidden]"), Active("total", "[rows]")),
        parameter_args=("weight",),
        forward="tests.test_seed_acceptance:split_reference",
        forward_semantics="Scale, and also reduce over the last axis.",
        backward_semantics="Return dx and dweight.",
        benchmark=(CASE,),
    )


def _module(name: str, source: str):
    module = types.ModuleType(name)
    exec(source, module.__dict__)
    module.fixture_source = source
    return module


def fixture_candidate(op, *, row_limit=100):
    return _module("seed_fixture", f"import torch\nrow_limit = {row_limit!r}\n" + """
def seed_affine_forward_with_saved(x, weight, bias):
    if x.shape[0] > row_limit:
        raise ValueError('row count unsupported')
    return x * weight + bias, (x, weight, bias)

def seed_affine_backward_from_saved(dy, saved_tensors):
    x, weight, bias = saved_tensors
    return dy * weight, (dy * x).sum(0), dy.sum(0)
""" + render_deployment_layer(op))


def multi_output_candidate(op, *, bad_dtype=False):
    return _module("seed_split_fixture", f"import torch\nbad_dtype = {bad_dtype!r}\n" + """
def seed_split_forward_with_saved(x, weight):
    scaled = x * weight
    total = scaled.sum(-1)
    if bad_dtype:
        total = total.to(torch.float64)
    return (scaled, total), (x, weight)

def seed_split_backward_from_saved(output_grads, saved_tensors):
    dscaled, dtotal = output_grads
    x, weight = saved_tensors
    grad = dscaled + dtotal.unsqueeze(-1)
    return grad * weight, (grad * x).sum(0)
""" + render_deployment_layer(op))


def payload_values(op):
    """What ``Corpus.load`` returns for one recorded call: arguments and cotangents."""
    generator = torch.Generator().manual_seed(0)
    dims = dict(CASE.dims)
    shapes = {"x": (dims["rows"], dims["hidden"]), "weight": (dims["hidden"],),
              "bias": (dims["hidden"],), "y": (dims["rows"], dims["hidden"]),
              "scaled": (dims["rows"], dims["hidden"]), "total": (dims["rows"],)}
    names = [a.name for a in op.args] + [o.name for o in op.outputs]
    grad_names = [a.name for a in op.args] + list(op.upstream_grad_names)
    return {g: torch.randn(shapes[n], generator=generator) for n, g in zip(names, grad_names)}


SPEC = {"rule": payload_gate1.SEARCH_RULE, "case_id": "affine/affine",
        "corpus": {"manifest": "/corpus/manifest.json", "corpus_id": "c"},
        "payload_ids": ["p0", "p1", "p2"]}


class FakeCorpus:
    def __init__(self, values):
        self.values = values
        self.loaded = []

    def load(self, payload_id, *, device):
        self.loaded.append(payload_id)
        return {k: v.clone() for k, v in self.values.items()}


def run_accept(op, candidate, summary, *, spec=SPEC):
    task = mock.Mock(spec=AgentTask)
    task.root = Path("/pack")
    task.spec = {"search_correctness": spec}
    task.op.return_value = op
    corpus = FakeCorpus(payload_values(op))
    judged = []
    with mock.patch.object(payload_gate1, "search_correctness",
                           lambda o, m, s, device: judged.append(s) or summary), \
            mock.patch("evograd.verification.payloads.corpus.Corpus.open",
                       return_value=corpus):
        accepted, report = seed.accept(task, candidate, device="cpu")
    return accepted, report, judged, corpus


class TestAccept(unittest.TestCase):
    def test_payloads_and_deployment_accept_a_correct_seed(self):
        op = fixture_op()
        accepted, report, judged, corpus = run_accept(op, fixture_candidate(op),
                                                      {"passed": 6, "total": 6})
        self.assertTrue(accepted, report)
        self.assertEqual(judged, [SPEC])
        # The deployment runs on the first search payload, not on generated inputs.
        self.assertEqual(corpus.loaded, ["p0"])
        self.assertEqual(report["deployment"]["adapter_kind"], "evolved_direct_autograd_module")

    def test_a_gate1_failure_rejects_before_the_deployment(self):
        op = fixture_op()
        accepted, report, _, corpus = run_accept(op, fixture_candidate(op),
                                                 {"passed": 5, "total": 6})
        self.assertFalse(accepted)
        self.assertNotIn("deployment", report)
        self.assertEqual(corpus.loaded, [])

    def test_a_deployment_failure_rejects_a_seed_gate1_accepts(self):
        op = fixture_op()
        accepted, report, _, _ = run_accept(op, fixture_candidate(op, row_limit=4),
                                            {"passed": 6, "total": 6})
        self.assertFalse(accepted)
        self.assertIn("row count unsupported", report["deployment"]["error_message"])

    def test_a_deployment_that_drops_a_gradient_is_rejected(self):
        op = fixture_op()
        candidate = fixture_candidate(op)
        setattr(candidate, candidate.DEPLOYMENT_ENTRY,
                lambda x, weight, bias: x.detach() * weight + bias)
        accepted, report, _, _ = run_accept(op, candidate, {"passed": 6, "total": 6})
        self.assertFalse(accepted)
        self.assertIn("dx", report["deployment"]["error_message"])

    def test_every_declared_output_and_its_dtype_are_checked(self):
        op = multi_output_op()
        accepted, report, _, _ = run_accept(op, multi_output_candidate(op),
                                            {"passed": 6, "total": 6})
        self.assertTrue(accepted, report)
        accepted, report, _, _ = run_accept(op, multi_output_candidate(op, bad_dtype=True),
                                            {"passed": 6, "total": 6})
        self.assertFalse(accepted)
        self.assertIn("total", report["deployment"]["error_message"])

    def test_a_pack_without_the_payload_rule_is_refused(self):
        op = fixture_op()
        with self.assertRaises(ValueError):
            run_accept(op, fixture_candidate(op), {"passed": 1, "total": 1},
                       spec={"rule": "declared_tolerances"})


class TestMain(unittest.TestCase):
    def write_pack(self, root: Path) -> None:
        (root / "declaration.py").write_text(
            "from tests.test_seed_acceptance import fixture_op\nop = fixture_op()\n")
        (root / "task.json").write_text(json.dumps({
            "schema": TASK_SCHEMA, "task_id": "t", "op": "seed_affine",
            "search_correctness": SPEC}))

    def test_main_reports_the_pack_rule_and_the_deployment(self):
        op = fixture_op()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_pack(root)
            candidate = root / "candidate.py"
            candidate.write_text(fixture_candidate(op).fixture_source)
            out = io.StringIO()
            with mock.patch.object(payload_gate1, "search_correctness",
                                   return_value={"passed": 6, "total": 6}), \
                    mock.patch("evograd.verification.payloads.corpus.Corpus.open",
                               return_value=FakeCorpus(payload_values(op))), \
                    contextlib.redirect_stdout(out):
                code = seed.main(["--pack", str(root), "--device", "cpu", str(candidate)])
        report = json.loads(out.getvalue())
        self.assertEqual((code, report["metrics"]["correct"]), (0, 1.0), report)
        self.assertEqual(report["search_correctness"], {"passed": 6, "total": 6})
        self.assertTrue(report["deployment"]["ok"])

    def test_main_rejects_when_the_pack_cannot_be_loaded(self):
        with TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = seed.main(["--pack", tmp, "--device", "cpu", str(Path(tmp) / "c.py")])
        report = json.loads(out.getvalue())
        self.assertEqual((code, report["metrics"]["correct"]), (1, 0.0))
        self.assertIn("error", report)


class TestVerifyCandidate(unittest.TestCase):
    def test_the_child_is_seed_acceptance_on_the_pack(self):
        done = subprocess.CompletedProcess([], 0, stdout='{"metrics":{"correct":1}}', stderr="")
        with mock.patch.object(runner, "_run", return_value=done) as run:
            report = runner.verify_candidate(python="python", pack=Path("/pack"),
                                             program_path=Path("seed.py"))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:3], ["python", "-m", "evograd.evaluation.common.seed"])
        self.assertEqual(cmd[cmd.index("--pack") + 1], str(Path("/pack").resolve()))
        self.assertTrue(runner.report_passed(report))

    def test_no_pack_is_refused_before_any_subprocess(self):
        with mock.patch.object(runner, "_run") as spawned:
            with self.assertRaises(ValueError):
                runner.verify_candidate(python=sys.executable, pack=None,
                                        program_path=Path("x.py"))
        spawned.assert_not_called()

    def test_timeout_is_repair_feedback_not_a_pass(self):
        with mock.patch.object(runner, "_run", return_value=subprocess.CompletedProcess(
            [], -9, stdout="", stderr="candidate killed after 120s")):
            report = runner.verify_candidate(python="python", pack=Path("/pack"),
                                             program_path=Path("x"))
        self.assertFalse(runner.report_passed(report))
        self.assertIn("killed", report["stderr"])

    def test_success_json_followed_by_crash_is_not_accepted(self):
        with mock.patch.object(runner, "_run", return_value=subprocess.CompletedProcess(
            [], 1, stdout='{"metrics":{"correct":1}}', stderr="crashed")):
            report = runner.verify_candidate(python="python", pack=Path("/pack"),
                                             program_path=Path("x"))
        self.assertFalse(runner.report_passed(report))

    def test_acceptance_cases_reads_the_gate1_summary(self):
        self.assertEqual(runner.acceptance_cases({"search_correctness": {"passed": 5, "total": 6}}),
                         (5, 6))
        self.assertEqual(runner.acceptance_cases({"error": "x"}), (0, 0))


if __name__ == "__main__":
    unittest.main()
