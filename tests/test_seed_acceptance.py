"""Seed acceptance checks declared dtypes, deployment, and required shapes."""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch

from evograd.evaluation.common.seed import check_seed_acceptance
from evograd.opdecl import Active, Provenance, Workload, declare_op
from evograd.opdecl.verify import verify
from evograd.pipelines.shared import runner
from evograd.pipelines.shared.artifact import render_deployment_layer


def affine_reference(x, weight, bias):
    return x * weight + bias


def fixture_op():
    return declare_op(
        name="seed_affine", level=1, family="test", dims=("rows", "hidden"),
        args=(Active("x", "[rows, hidden]"), Active("weight", "[hidden]"),
              Active("bias", "[hidden]")),
        output=Active("y", "[rows, hidden]"), parameter_args=("weight", "bias"),
        forward="tests.test_seed_acceptance:affine_reference",
        forward_semantics="Elementwise affine transform.",
        backward_semantics="Return dx, dweight, dbias.",
        correctness=tuple(Workload(dims={"rows": 2, "hidden": 4}, dtype=d)
                          for d in ("float32", "float16", "bfloat16")),
        benchmark=(Workload(dims={"rows": 8, "hidden": 4}, dtype="bfloat16",
                            provenance=Provenance(model="synthetic", component="affine",
                                                  source="handpicked")),),
        tolerances={"float32": (1e-5, 1e-5), "float16": (0.1, 0.1),
                    "bfloat16": (0.1, 0.1)},
    )


def split_reference(x, weight):
    scaled = x * weight
    return scaled, scaled.sum(-1)


def multi_output_op():
    """Two declared outputs and two gradients, so the zip in `_check_outputs`
    and the gradient sweep are exercised on more than one of each."""
    return declare_op(
        name="seed_split", level=1, family="test", dims=("rows", "hidden"),
        args=(Active("x", "[rows, hidden]"), Active("weight", "[hidden]")),
        output=(Active("scaled", "[rows, hidden]"), Active("total", "[rows]")),
        parameter_args=("weight",),
        forward="tests.test_seed_acceptance:split_reference",
        forward_semantics="Scale, and also reduce over the last axis.",
        backward_semantics="Return dx and dweight.",
        correctness=(Workload(dims={"rows": 2, "hidden": 4}, dtype="float32"),),
        benchmark=(Workload(dims={"rows": 8, "hidden": 4}, dtype="float32",
                            provenance=Provenance(model="synthetic", component="split",
                                                  source="handpicked")),),
        tolerances={"float32": (1e-5, 1e-5)},
    )


def multi_output_candidate(op, *, drop_second_output=False, bad_dtype=False):
    module = types.ModuleType("seed_split_fixture")
    source = (f"import torch\ndrop_second = {drop_second_output!r}\n"
              f"bad_dtype = {bad_dtype!r}\n" + """
def seed_split_forward_with_saved(x, weight):
    scaled = x * weight
    total = scaled.sum(-1)
    if bad_dtype:
        total = total.to(torch.float64)
    if drop_second:
        return scaled, (x, weight)
    return (scaled, total), (x, weight)

def seed_split_backward_from_saved(output_grads, saved_tensors):
    dscaled, dtotal = output_grads
    x, weight = saved_tensors
    grad = dscaled + dtotal.unsqueeze(-1)
    return grad * weight, (grad * x).sum(0)
""" + render_deployment_layer(op))
    exec(source, module.__dict__)
    module.fixture_source = source
    return module


def fixture_candidate(op, *, fp32_only=False, row_limit=100):
    module = types.ModuleType("seed_fixture")
    source = (f"import torch\nfp32_only = {fp32_only!r}\nrow_limit = {row_limit!r}\n" + """
def seed_affine_forward_with_saved(x, weight, bias):
    if fp32_only and x.dtype != torch.float32:
        raise TypeError('float32 only')
    if x.shape[0] > row_limit:
        raise ValueError('row count unsupported')
    return x * weight + bias, (x, weight, bias)

def seed_affine_backward_from_saved(dy, saved_tensors):
    x, weight, bias = saved_tensors
    return (dy * weight, (dy.float() * x.float()).sum(0).to(weight.dtype),
            dy.float().sum(0).to(bias.dtype))
""" + render_deployment_layer(op))
    exec(source, module.__dict__)
    module.fixture_source = source
    return module


class TestSeedAcceptance(unittest.TestCase):
    def test_valid_multidtype_seed_uses_native_deployment(self):
        op = fixture_op()
        candidate = fixture_candidate(op)
        self.assertTrue(verify(op, candidate, device="cpu").ok)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertTrue(report["ok"], report)
        self.assertEqual(len(report["cases"]), 5)
        for row in report["cases"]:
            if row["stage"] != "benchmark_pair":
                self.assertEqual(row["adapter_kind"], "evolved_direct_autograd_module")

    def test_fp32_only_program_is_rejected_on_declared_dtypes(self):
        op = fixture_op()
        report = verify(op, fixture_candidate(op, fp32_only=True), device="cpu")
        self.assertFalse(report.ok)
        self.assertEqual([case.ok for case in report.cases], [True, False, False])

    def test_small_correctness_pass_does_not_accept_benchmark_rejection(self):
        op = fixture_op()
        candidate = fixture_candidate(op, row_limit=4)
        self.assertTrue(verify(op, candidate, device="cpu").ok)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        self.assertEqual(report["failure"]["stage"], "benchmark_pair")
        self.assertEqual(report["failure"]["dims"]["rows"], 8)
        self.assertEqual(report["failure"]["dtype"], "bfloat16")
        self.assertIn("row count unsupported", report["failure"]["error_message"])

    def test_benchmark_native_route_failure_is_not_hidden_by_valid_pair(self):
        op = fixture_op()
        candidate = fixture_candidate(op)
        original = getattr(candidate, candidate.DEPLOYMENT_ENTRY)

        def native(x, weight, bias):
            if x.shape[0] > 4:
                raise ValueError("native large shape unsupported")
            return original(x, weight, bias)

        setattr(candidate, candidate.DEPLOYMENT_ENTRY, native)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        self.assertEqual(report["failure"]["stage"], "benchmark_deployment")

    def test_wrong_deployment_gradient_fails_existing_numerical_gate(self):
        op = fixture_op()
        candidate = fixture_candidate(op)

        def native(x, weight, bias):
            return x.detach() * weight + bias

        setattr(candidate, candidate.DEPLOYMENT_ENTRY, native)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        self.assertEqual(report["failure"]["stage"], "deployment_correctness")
        self.assertFalse(report["failure"]["checks"]["dx"]["ok"])

    def test_selected_suite_is_the_only_benchmark_smoked(self):
        op = fixture_op()
        op = replace(op, benchmark_suites={"tiny": (op.correctness[2],)})
        report = check_seed_acceptance(op, fixture_candidate(op, row_limit=4),
                                       benchmark_suite="tiny", device="cpu")
        self.assertTrue(report["ok"], report)


class TestVerificationSubprocess(unittest.TestCase):
    def test_actual_child_rejects_large_shape_and_accepts_repaired_program(self):
        op = fixture_op()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            declaration = root / "declaration.py"
            declaration.write_text("from tests.test_seed_acceptance import fixture_op\nop = fixture_op()\n")
            candidate = root / "candidate.py"
            candidate.write_text(fixture_candidate(op, row_limit=4).fixture_source)
            kwargs = dict(python=sys.executable, op_name=op.name, program_path=candidate,
                          declaration=f"{declaration}:op", device="cpu", timeout=60)
            # The original direct-pair verifier alone passes the small cases.
            self.assertTrue(runner.report_passed(runner.verify_candidate(**kwargs)))
            report = runner.verify_candidate(**kwargs, seed_acceptance=True)
            self.assertFalse(runner.report_passed(report), report)
            self.assertEqual(report["seed_acceptance"]["failure"]["dims"]["rows"], 8)
            # Same filepath, corrected body: a fresh child must read the new code.
            candidate.write_text(fixture_candidate(op).fixture_source)
            report = runner.verify_candidate(**kwargs, seed_acceptance=True)
            self.assertTrue(runner.report_passed(report), report)

    def test_unified_seed_rejects_overflow_that_passes_default_oracle(self):
        from tests.test_unified_validation import _mean_op

        op = _mean_op()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            declaration = root / "declaration.py"
            declaration.write_text(
                "from tests.test_unified_validation import _mean_op\nop = _mean_op()\n"
            )
            candidate = root / "candidate.py"
            candidate.write_text(
                "import torch\nUNIFIED_DEPLOYMENT_CONFIG = {}\n"
                "def validation_mean_forward_with_saved(x):\n"
                "    return x.sum(-1, keepdim=True) / x.shape[-1], (x,)\n"
                "def validation_mean_backward_from_saved(dy, saved_tensors):\n"
                "    x, = saved_tensors\n"
                "    return (dy.expand_as(x) / x.shape[-1],)\n"
                + render_deployment_layer(op)
            )
            kwargs = dict(
                python=sys.executable, op_name=op.name, program_path=candidate,
                declaration=f"{declaration}:op", device="cpu", timeout=60,
            )
            self.assertTrue(runner.report_passed(runner.verify_candidate(**kwargs)))
            report = runner.verify_candidate(**kwargs, seed_acceptance=True)
            self.assertFalse(runner.report_passed(report), report)
            self.assertTrue(report["verify"]["ok"])
            self.assertTrue(report["seed_acceptance"]["ok"])
            variants = report["numerical_variants"]
            large = next(case for case in variants["cases"]
                         if case["variant"] == "stress_large_constant")
            self.assertEqual(large["status"], "failed_candidate")
            self.assertGreater(variants["coverage"]["failed"], 0)

    def test_flags_are_opt_in_and_suite_is_forwarded(self):
        done = subprocess.CompletedProcess([], 0, stdout='{"metrics":{"correct":1}}', stderr="")
        with mock.patch.object(runner, "_run", return_value=done) as run:
            runner.verify_candidate(python="python", op_name="layernorm", program_path=Path("seed.py"),
                                    seed_acceptance=True, benchmark_suite="full")
            cmd = run.call_args.args[0]
            self.assertIn("evograd.evaluation.common.seed", cmd)
            self.assertEqual(cmd[cmd.index("--benchmark-suite") + 1], "full")
            self.assertNotIn("--dtype", cmd)
            runner.verify_candidate(python="python", op_name="layernorm", program_path=Path("seed.py"))
            self.assertIn("evograd.opdecl.verify_cli", run.call_args.args[0])

    def test_timeout_is_repair_feedback_not_a_pass(self):
        with mock.patch.object(runner, "_run", return_value=subprocess.CompletedProcess(
            [], -9, stdout="", stderr="candidate killed after 120s")):
            report = runner.verify_candidate(python="python", op_name="layernorm", program_path=Path("x"))
        self.assertFalse(runner.report_passed(report))
        self.assertIn("killed", report["stderr"])

    def test_success_json_followed_by_crash_is_not_accepted(self):
        with mock.patch.object(runner, "_run", return_value=subprocess.CompletedProcess(
            [], 1, stdout='{"metrics":{"correct":1}}', stderr="crashed")):
            report = runner.verify_candidate(python="python", op_name="layernorm", program_path=Path("x"))
        self.assertFalse(runner.report_passed(report))


class TestMultiOutputTasks(unittest.TestCase):
    """Every declared output and every declared gradient, not just the first."""

    def test_a_correct_multi_output_seed_is_accepted(self):
        op = multi_output_op()
        candidate = multi_output_candidate(op)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertTrue(report["ok"], report.get("failure"))
        stages = {row["stage"] for row in report["cases"]}
        self.assertEqual(
            stages,
            {"deployment_correctness", "benchmark_pair", "benchmark_deployment"},
        )

    def test_a_missing_second_output_is_rejected(self):
        op = multi_output_op()
        candidate = multi_output_candidate(op, drop_second_output=True)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertFalse(report["ok"])

    def test_a_wrong_output_dtype_is_rejected_against_the_declaration(self):
        """The expected dtype comes from the declaration, not from whichever
        tensor the case builder happened to allocate for the gradient."""
        op = multi_output_op()
        candidate = multi_output_candidate(op, bad_dtype=True)
        report = check_seed_acceptance(op, candidate, device="cpu")
        self.assertFalse(report["ok"])
        # Rejected by the existing numerical/contract gate, and the failing
        # output is named in the message the repair prompt leads with.
        failure = report["failure"]
        self.assertIn("total", failure["error_message"])
        self.assertFalse(failure["checks"]["total"]["ok"])


class TestVerifyCandidateArgumentValidation(unittest.TestCase):
    """Misuse is refused before a subprocess command is assembled."""

    def test_seed_acceptance_refuses_a_dtype_filter(self):
        with self.assertRaises(ValueError) as raised:
            runner.verify_candidate(
                python=sys.executable, op_name="layernorm", program_path=Path("x.py"),
                seed_acceptance=True, dtypes=("bfloat16",))
        self.assertIn("every declared correctness dtype", str(raised.exception))

    def test_a_benchmark_suite_without_seed_acceptance_is_refused(self):
        with self.assertRaises(ValueError):
            runner.verify_candidate(
                python=sys.executable, op_name="layernorm", program_path=Path("x.py"),
                benchmark_suite="full")

    def test_validation_happens_before_any_subprocess_is_started(self):
        with mock.patch.object(runner, "_run") as spawned:
            with self.assertRaises(ValueError):
                runner.verify_candidate(
                    python=sys.executable, op_name="layernorm",
                    program_path=Path("x.py"), seed_acceptance=True,
                    dtypes=("bfloat16",))
        spawned.assert_not_called()


if __name__ == "__main__":
    unittest.main()
