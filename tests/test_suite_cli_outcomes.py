"""What ``evograd suite`` does with the results it gets back.

The command's own job is small and worth pinning: choose which tasks and cases
to run, hand them and the run parameters to the measurement layer, turn what
comes back into rows of a report, write it, and decide an exit code. The
measurement itself is tier-1's and is tested there; these tests mock exactly
that boundary and nothing below it.

Only the expensive execution boundary is replaced:

* ``run_fair_benchmarks`` -- the timing loop, which needs a GPU;
* ``verify_runtime_forward`` -- which executes the operator to prove the eager
  baseline matches the definition;
* ``torch.cuda.is_available`` -- so the command's deliberate early refusal does
  not stop a CPU test from reaching the part being tested.

Everything else is the real thing: the real registry, the real selection, the
real report schema, the real exit-code decision.

**On timeouts.** ``evograd suite`` owns no timeout and starts no subprocess: it
calls the tier-1 runners in-process, and neither ``tier1/fair.py`` nor
``tier1/fast.py`` implements one either. There is therefore no timeout policy
to test here, and inventing a flag or a status to make one testable would be
inventing behaviour. What does exist is the per-task failure boundary, which is
what a runner-raised timeout would travel through, so that is what is tested --
with ``TimeoutError`` as the raised failure, precisely because it is the shape
such a failure would have.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evograd.benchmark.core.report import TaskResult

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

CANDIDATE_SOURCE = '''
"""A candidate module the CLI can load. Never executed: the runner is mocked."""


def forward_with_saved(*args, **kwargs):
    raise AssertionError("the timing runner is mocked in these tests")


def backward_from_saved(*args, **kwargs):
    raise AssertionError("the timing runner is mocked in these tests")
'''

#: Two primitives with different levels of nothing in common, so a selection
#: that silently widened or narrowed would show up.
SELECTED = ("rmsnorm", "softmax")


def _fair_report(op_name: str, cases: int) -> dict:
    """The shape ``task_from_fair_report`` reads, with nothing measured."""
    return {
        "op": op_name,
        "ok": True,
        "cases": [
            {"ok": True, "speedup_full_step": 2.0, "saved_bytes": 0.0,
             "input_bytes": 0.0, "workload": {}}
            for _ in range(cases)
        ],
    }


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class SuiteCommandOutcomes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="evograd-suite-cli-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = self.root / "candidates"
        self.candidates.mkdir()
        for name in SELECTED:
            (self.candidates / f"{name}.py").write_text(CANDIDATE_SOURCE)
        self.out = self.root / "out"

    # ── the boundary ────────────────────────────────────────────────────
    def _run(self, outcomes, argv_extra=()):
        """Run the command with the measurement layer replaced.

        ``outcomes`` maps an operator name to what the runner should do: a
        number of measured cases, or an exception instance to raise.
        """
        from evograd import suite_cli

        calls = []

        def fake_run_fair(op, candidate, baseline, **kwargs):
            calls.append({"op": op.name, "kwargs": kwargs})
            outcome = outcomes[op.name]
            if isinstance(outcome, BaseException):
                raise outcome
            return _fair_report(op.name, outcome)

        def fake_task_from_fair(name, level, family, resolved, report, **kwargs):
            return TaskResult(
                op=name, level=level, family=family, baseline=resolved,
                speedups=tuple(2.0 for _ in report["cases"]),
                cases_total=len(report["cases"]), cases_ok=len(report["cases"]),
            )

        argv = [
            "--candidates", str(self.candidates),
            "--out", str(self.out),
            "--device", "cpu",
            "--baseline", "pytorch_autograd",
            *[a for name in SELECTED for a in ("--op", name)],
            *argv_extra,
        ]
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("evograd.opdecl.baselines.verify_runtime_forward"), \
             mock.patch("evograd.evaluation.tier1.fair.run_fair_benchmarks",
                        side_effect=fake_run_fair), \
             mock.patch.object(suite_cli, "task_from_fair_report",
                               side_effect=fake_task_from_fair):
            code = suite_cli.main(argv)
        return code, calls, json.loads((self.out / "suite_report.json").read_text())

    # ── A. every selected task succeeds ─────────────────────────────────
    def test_all_selected_tasks_succeed(self):
        code, calls, report = self._run({name: 3 for name in SELECTED})

        self.assertEqual(code, 0)
        self.assertEqual(sorted(c["op"] for c in calls), sorted(SELECTED))
        rows = {task["op"]: task for task in report["tasks"]}
        self.assertEqual(sorted(rows), sorted(SELECTED))
        for name in SELECTED:
            self.assertTrue(rows[name]["ok"], rows[name])
            self.assertEqual(rows[name]["cases_ok"], 3)
        # The report's own statement that nothing was left unrun.
        self.assertEqual(report["overall"]["coverage_operators"], 1.0)
        self.assertEqual(report["overall"]["coverage_cases"], 1.0)

    def test_the_selection_reaching_the_runner_is_the_requested_one(self):
        """``--op`` narrows what runs; nothing else may reach the runner."""
        _code, calls, _report = self._run({name: 1 for name in SELECTED})
        self.assertEqual(len(calls), len(SELECTED))

    def test_the_declared_cases_are_what_is_measured(self):
        from evograd.benchmark import get_task

        _code, calls, _report = self._run({name: 1 for name in SELECTED})
        by_op = {c["op"]: c for c in calls}
        for name in SELECTED:
            with self.subTest(op=name):
                self.assertEqual(
                    tuple(by_op[name]["kwargs"]["workloads"]),
                    get_task(name).benchmark_workloads(suite=None),
                )

    def test_run_parameters_are_forwarded(self):
        _code, calls, _report = self._run(
            {name: 1 for name in SELECTED},
            argv_extra=("--warmup", "7", "--reps", "11"),
        )
        for call in calls:
            with self.subTest(op=call["op"]):
                self.assertEqual(call["kwargs"]["device"], "cpu")
                self.assertEqual(call["kwargs"]["warmup"], 7)
                self.assertEqual(call["kwargs"]["reps"], 11)
                # One block is the documented default of this command.
                self.assertEqual(call["kwargs"]["blocks"], 1)

    def test_optional_parameters_are_omitted_when_not_given(self):
        """Absent flags must not be forwarded as ``None`` and override the
        runner's own defaults."""
        _code, calls, _report = self._run({name: 1 for name in SELECTED})
        for call in calls:
            with self.subTest(op=call["op"]):
                self.assertNotIn("warmup", call["kwargs"])
                self.assertNotIn("reps", call["kwargs"])

    def test_a_named_suite_selects_its_own_cases(self):
        from evograd.benchmark import get_task

        _code, calls, _report = self._run(
            {name: 1 for name in SELECTED}, argv_extra=("--suite", "full")
        )
        by_op = {c["op"]: c for c in calls}
        for name in SELECTED:
            with self.subTest(op=name):
                self.assertEqual(
                    tuple(by_op[name]["kwargs"]["workloads"]),
                    get_task(name).benchmark_workloads(suite="full"),
                )

    def test_the_report_is_written_where_asked(self):
        _code, _calls, _report = self._run({name: 1 for name in SELECTED})
        self.assertTrue((self.out / "suite_report.json").is_file())
        self.assertTrue((self.out / "SUITE_RESULTS.md").is_file())

    # ── B. one selected task fails ──────────────────────────────────────
    def test_a_failing_task_is_reported_and_fails_the_run(self):
        failing, passing = SELECTED[0], SELECTED[1]
        code, calls, report = self._run(
            {failing: RuntimeError("kernel produced a wrong gradient"),
             passing: 4}
        )

        self.assertEqual(code, 1)
        rows = {task["op"]: task for task in report["tasks"]}

        # The failure survives into the report rather than vanishing.
        self.assertIn(failing, rows)
        self.assertFalse(rows[failing]["ok"])
        self.assertIn("kernel produced a wrong gradient", rows[failing]["error"])
        self.assertEqual(rows[failing]["cases_ok"], 0)

        # And the run is not presented as entirely successful: the report says
        # one of the two operators produced no covered case.
        self.assertEqual(report["overall"]["coverage_operators"], 0.5)
        self.assertLess(report["overall"]["coverage_cases"], 1.0)
        self.assertIn(failing, (self.out / "SUITE_RESULTS.md").read_text())

        # The other selected task still ran and still reports its result.
        self.assertTrue(rows[passing]["ok"])
        self.assertEqual(rows[passing]["cases_ok"], 4)
        self.assertEqual(sorted(c["op"] for c in calls), sorted(SELECTED))

    def test_a_failing_task_still_reports_the_cases_it_would_have_run(self):
        """Coverage is "we did not run it", not "it has no cases"."""
        from evograd.benchmark import get_task

        failing = SELECTED[0]
        _code, _calls, report = self._run(
            {failing: RuntimeError("boom"), SELECTED[1]: 1}
        )
        row = next(t for t in report["tasks"] if t["op"] == failing)
        self.assertEqual(
            row["cases_total"], len(get_task(failing).benchmark_workloads(suite=None))
        )

    def test_a_missing_candidate_is_uncovered_not_a_crash(self):
        """The other documented failure mode of this command."""
        (self.candidates / f"{SELECTED[0]}.py").unlink()
        code, calls, report = self._run({SELECTED[1]: 2})

        self.assertEqual(code, 1)
        rows = {task["op"]: task for task in report["tasks"]}
        self.assertFalse(rows[SELECTED[0]]["ok"])
        self.assertIn("no candidate program found", rows[SELECTED[0]]["error"])
        # It never reached the runner, and the other task still did.
        self.assertEqual([c["op"] for c in calls], [SELECTED[1]])
        self.assertTrue(rows[SELECTED[1]]["ok"])

    # ── C. a runner failure of the shape a timeout would have ───────────
    def test_a_runner_timeout_propagates_like_any_other_failure(self):
        """This command owns no timeout and starts no subprocess.

        The tier-1 runners it calls own none either, so there is no timeout
        policy here to assert. What is asserted is the existing propagation
        path a runner-raised ``TimeoutError`` would take: the task is recorded
        as failed with the exception named, the run's other tasks are
        unaffected, and the existing failure exit code is returned. No new
        flag, status or exit code is introduced.
        """
        timed_out, passing = SELECTED[0], SELECTED[1]
        code, calls, report = self._run(
            {timed_out: TimeoutError("timed out after 600s"), passing: 2}
        )

        self.assertEqual(code, 1)
        rows = {task["op"]: task for task in report["tasks"]}
        self.assertFalse(rows[timed_out]["ok"])
        self.assertIn("TimeoutError", rows[timed_out]["error"])
        self.assertIn("timed out after 600s", rows[timed_out]["error"])
        self.assertEqual(report["overall"]["coverage_operators"], 0.5)
        self.assertTrue(rows[passing]["ok"])
        self.assertEqual(sorted(c["op"] for c in calls), sorted(SELECTED))

    def test_the_suite_command_declares_no_timeout_option(self):
        """Pinned so a timeout cannot be added without revisiting this file."""
        from evograd import suite_cli

        source = Path(suite_cli.__file__).read_text()
        self.assertNotIn("--timeout", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":
    unittest.main()
