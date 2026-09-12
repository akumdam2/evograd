"""Who owns a benchmark case, and what must not change when it moves.

A Level-1 primitive declares mathematics and the generic cases that prove an
implementation correct. It does not decide which shapes are worth timing, where
the shape regime splits, how cases are weighted, or what a named suite selects
-- those are benchmark decisions, and they live in
:mod:`evograd.benchmark.operator_suite.cases` (grids computed from a published
model configuration) or in a model's own manifest (cases observed in a captured
run).

Moving them was allowed to change exactly one thing: where the code lives. The
tests here pin the rest -- the case lists, their order and multiplicity, their
dtypes, tolerances, layouts and provenance, the weighting applied to them, and
the fact that binding produces a *new* declaration rather than editing the
shared one.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from evograd.benchmark import TASKS, get_task
from evograd.benchmark.cases import (
    ObservedBinding,
    bind_benchmark_cases,
    bind_observed_cases,
    bind_suite_cases,
    index_bindings,
)
from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import (
    OBSERVED_BINDINGS as QWEN3_0_6B_OBSERVED,
)
from evograd.benchmark.operator_suite.cases import SuiteCases, all_cases, cases_for
from evograd.ops import PRIMITIVES

REPO = Path(__file__).resolve().parents[1]

#: Every primitive whose timed grid comes from a published model configuration.
#: ``conv2d``'s grid is hand-picked ResNet-style rather than model-derived, and
#: is here for the same reason: it is a performance grid, so the suite owns it.
SUITE_CASE_OWNERS = frozenset(PRIMITIVES)


class TestPrimitivesOwnNoPerformanceCases(unittest.TestCase):
    def test_no_primitive_declares_a_timed_grid(self):
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                self.assertEqual(primitive.benchmark, ())
                self.assertEqual(primitive.coverage, ())
                self.assertEqual(primitive.benchmark_suites, {})

    def test_no_primitive_declares_regime_or_weighting(self):
        """Both are selection settings: they say which cases count and by how
        much, which is a benchmark question even though the feature reads a
        dimension of the primitive's own shape."""
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                self.assertIsNone(primitive.regime_feature)
                self.assertIsNone(primitive.regime_split)
                self.assertIsNone(primitive.case_weight)

    def test_every_primitive_keeps_its_generic_correctness_cases(self):
        """Correctness is the primitive's own: it proves the mathematics."""
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                self.assertTrue(primitive.correctness, f"{name} lost its correctness cases")

    def test_no_primitive_names_a_model_configuration(self):
        """One documented exception, and it is a numerical rule rather than a
        case list: RoPE's input generator reads ``rope_theta`` from each
        workload's own provenance, and needs the published configuration table
        to do so."""
        offenders = []
        for package in sorted((REPO / "src/evograd/ops/level1").iterdir()):
            init = package / "__init__.py"
            if init.is_file() and "opdecl.models" in init.read_text():
                offenders.append(package.name)
        self.assertEqual(offenders, ["rope"])

    def test_importing_primitives_loads_no_benchmark_or_evaluation(self):
        """Checked in a fresh interpreter: the ownership rule is about what a
        process actually loads, not only about what a module names."""
        code = (
            "import sys, evograd.ops;"
            "print(len(evograd.ops.PRIMITIVES));"
            "print(sorted(m for m in sys.modules"
            " if m.startswith(('evograd.benchmark', 'evograd.evaluation'))))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        count, loaded = result.stdout.strip().splitlines()
        self.assertEqual(count, "20")
        self.assertEqual(loaded, "[]", f"importing primitives loaded {loaded}")


class TestSuiteOwnsThePerformanceCases(unittest.TestCase):
    def test_every_primitive_has_suite_cases(self):
        defined = all_cases()
        self.assertEqual(set(defined), SUITE_CASE_OWNERS)
        for name, suite in sorted(defined.items()):
            with self.subTest(op=name):
                self.assertIsInstance(suite, SuiteCases)

    def test_a_primitive_the_suite_does_not_define_is_returned_unchanged(self):
        self.assertIsNone(cases_for("not_a_primitive"))

    def test_the_suite_cases_are_what_the_task_serves(self):
        mirrored = {
            b.task: b.mirrors_coverage for b in QWEN3_0_6B_OBSERVED if b.mirrors_coverage
        }
        for name, suite in sorted(all_cases().items()):
            with self.subTest(op=name):
                task = TASKS[name]
                self.assertEqual(task.benchmark, tuple(suite.benchmark))
                for suite_name, cases in suite.suites.items():
                    if mirrored.get(name) == suite_name:
                        # Declared to follow the coverage, which a model's
                        # observed cases join. Pinned by its own test.
                        continue
                    self.assertEqual(task.benchmark_suites[suite_name], tuple(cases))

    def test_suite_names_keep_their_declaration_order(self):
        for name, suite in sorted(all_cases().items()):
            if not suite.suites:
                continue
            with self.subTest(op=name):
                declared = list(suite.suites)
                served = [k for k in TASKS[name].benchmark_suites if k in set(declared)]
                self.assertEqual(served, declared)


class TestBindingDoesNotMutateThePrimitive(unittest.TestCase):
    def test_binding_returns_a_new_declaration(self):
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                bound = bind_benchmark_cases(primitive, QWEN3_0_6B_OBSERVED)
                self.assertIsNot(bound, primitive)
                self.assertEqual(primitive.benchmark, ())
                self.assertEqual(primitive.benchmark_suites, {})
                self.assertIsNone(primitive.case_weight)

    def test_binding_twice_gives_the_same_result(self):
        """Binding reads the primitive; it must not accumulate into it."""
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                first = bind_benchmark_cases(primitive, QWEN3_0_6B_OBSERVED)
                second = bind_benchmark_cases(primitive, QWEN3_0_6B_OBSERVED)
                self.assertEqual(first.benchmark, second.benchmark)
                self.assertEqual(first.coverage, second.coverage)
                self.assertEqual(
                    sorted(first.benchmark_suites), sorted(second.benchmark_suites)
                )

    def test_a_primitive_declaring_its_own_grid_is_refused(self):
        """One owner per case collection, and the collision is loud."""
        import dataclasses

        primitive = PRIMITIVES["rmsnorm"]
        conflicting = dataclasses.replace(primitive, benchmark=(primitive.correctness[0],))
        with self.assertRaisesRegex(ValueError, "has one owner"):
            bind_suite_cases(conflicting)

    def test_the_registry_serves_the_bound_declaration(self):
        for name in sorted(PRIMITIVES):
            with self.subTest(op=name):
                self.assertIs(get_task(name), TASKS[name])
                self.assertIsNot(TASKS[name], PRIMITIVES[name])
                self.assertEqual(TASKS[name].name, PRIMITIVES[name].name)
                self.assertEqual(TASKS[name].forward, PRIMITIVES[name].forward)


class TestObservedCasesStillJoinLast(unittest.TestCase):
    OBSERVED = tuple(binding.task for binding in QWEN3_0_6B_OBSERVED)

    def test_observed_cases_are_bound_after_the_suite_cases(self):
        """Order is the contract: the suite establishes the coverage a model's
        observed cases then join."""
        for name in self.OBSERVED:
            with self.subTest(op=name):
                suite_only = bind_suite_cases(PRIMITIVES[name])
                both = bind_observed_cases(suite_only, QWEN3_0_6B_OBSERVED)
                observed = both.benchmark_suites["qwen3_0_6b_observed"]
                self.assertTrue(observed)
                for case in observed:
                    self.assertIn(case, both.coverage)
                    self.assertNotIn(case, suite_only.coverage)

    def test_the_coverage_mirroring_suite_follows_the_bound_coverage(self):
        """``rmsnorm`` serves its untimed coverage under a name as well as a
        field. The two must not drift when the observed cases join."""
        binding = next(b for b in QWEN3_0_6B_OBSERVED if b.mirrors_coverage)
        self.assertEqual(binding.task, "rmsnorm")
        task = TASKS["rmsnorm"]
        self.assertEqual(task.benchmark_suites[binding.mirrors_coverage], task.coverage)

    def test_only_the_declared_mirror_is_extended(self):
        """A suite that merely happens to equal the coverage is left alone."""
        suite_only = bind_suite_cases(PRIMITIVES["rmsnorm"])
        bound = bind_observed_cases(suite_only, QWEN3_0_6B_OBSERVED)
        for name, cases in bound.benchmark_suites.items():
            if name in ("coverage", "qwen3_0_6b_observed"):
                continue
            with self.subTest(suite=name):
                self.assertEqual(cases, suite_only.benchmark_suites[name])


class TestWeightingAndRegimesSurvivedTheMove(unittest.TestCase):
    """The moved functions are exercised, not merely counted."""

    def test_every_weighted_task_weights_its_own_cases(self):
        weighted = {n: op for n, op in TASKS.items() if op.case_weight is not None}
        self.assertTrue(weighted)
        for name, op in sorted(weighted.items()):
            with self.subTest(op=name):
                for case in op.benchmark:
                    weight = op.case_weight(case)
                    self.assertIsInstance(weight, float)
                    self.assertGreater(weight, 0.0)

    def test_regime_feature_and_split_travel_together(self):
        for name, op in sorted(TASKS.items()):
            with self.subTest(op=name):
                self.assertEqual(
                    op.regime_feature is None, op.regime_split is None,
                    "a regime feature without a split describes nothing",
                )

    def test_the_regime_feature_separates_the_declared_suites(self):
        """``small``/``large`` are what the split means, so they must agree."""
        for name, op in sorted(TASKS.items()):
            if op.regime_split is None:
                continue
            suites = op.benchmark_suites
            if not {"small", "large"} <= set(suites):
                continue
            with self.subTest(op=name):
                for case in suites["small"]:
                    self.assertLessEqual(op.regime_feature(case), op.regime_split)
                for case in suites["large"]:
                    # ``regime_suites`` puts the boundary case in ``large``.
                    self.assertGreaterEqual(op.regime_feature(case), op.regime_split)

    def test_weighting_is_defined_exactly_where_the_regime_is(self):
        for name, op in sorted(TASKS.items()):
            if op.level != 1:
                continue
            with self.subTest(op=name):
                self.assertEqual(op.case_weight is None, op.regime_split is None)


class TestSuiteCommandOwnership(unittest.TestCase):
    def test_the_root_cli_dispatches_suite_to_the_root_module(self):
        import ast

        source = (REPO / "src/evograd/cli.py").read_text()
        tree = ast.parse(source)
        handler = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_suite"
        )
        imported = [
            node.module for node in ast.walk(handler)
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertEqual(imported, ["evograd.suite_cli"])

    def test_the_suite_command_is_still_registered(self):
        from evograd.cli import _COMMANDS

        self.assertIn("suite", _COMMANDS)

    def test_the_canonical_module_exposes_main(self):
        from evograd import suite_cli

        self.assertTrue(callable(suite_cli.main))

    def test_a_missing_required_argument_still_exits_two(self):
        """argparse's usage error, unchanged by the move."""
        from evograd import suite_cli

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                suite_cli.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_a_run_without_a_device_is_refused_before_any_work(self):
        """The suite times with CUDA events, so it refuses early rather than
        failing per operator. Same message, same exit code, new owner.

        The absent device is stated rather than inherited from the host: this
        asserts what the command does when there is no CUDA device, and on a
        GPU machine the real answer is that there is one. Reading the host
        would make the test assert something different depending on where it
        ran, which is the opposite of pinning a behaviour.
        """
        import torch

        from evograd import suite_cli

        stderr = io.StringIO()
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    suite_cli.main(
                        ["--candidate-baseline", "liger", "--out", "/tmp",
                         "--op", "not_an_operator"]
                    )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("needs a CUDA device", stderr.getvalue())



class TestObservedBindingOwnership(unittest.TestCase):
    """Which primitives a model binds is the model's; how to bind is shared."""

    SHARED = REPO / "src/evograd/benchmark/cases.py"
    MANIFEST = (REPO / "src/evograd/benchmark/topdown/qwen3_0_6b"
                / "levels/level1/manifest.py")
    ASSEMBLY = REPO / "src/evograd/benchmark/core/registry.py"

    def test_the_shared_binder_names_no_model(self):
        """No task list, no `if workload == ...`, not even the word."""
        source = self.SHARED.read_text()
        for marker in ("qwen", "Qwen", "QWEN", "llama", "Llama", "alphafold"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, source)

    def test_the_configuration_lives_in_the_qwen_manifest(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level1 import manifest

        self.assertTrue(manifest.OBSERVED_BINDINGS)
        for binding in manifest.OBSERVED_BINDINGS:
            with self.subTest(task=binding.task):
                self.assertIsInstance(binding, ObservedBinding)
                self.assertEqual(binding.workload, "qwen3_0_6b")

    def test_there_is_one_authoritative_configuration(self):
        """The manifest is the only place the table is written down."""
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level1 import manifest

        owners = {
            self.MANIFEST,
            REPO / "src/evograd/benchmark/topdown/llama3_2_1b/levels/level1/manifest.py",
        }
        declaring = []
        for path in (REPO / "src/evograd").rglob("*.py"):
            if "__pycache__" in path.parts or path in owners:
                continue
            if "ObservedBinding(" in path.read_text():
                declaring.append(str(path.relative_to(REPO)))
        self.assertEqual(declaring, [])
        self.assertEqual(len(manifest.OBSERVED_BINDINGS), 6)

    def test_the_registry_is_the_assembly_point(self):
        """A concrete-model import is acceptable here and nowhere else."""
        source = self.ASSEMBLY.read_text()
        self.assertIn("levels.level1.manifest", source)
        self.assertIn("OBSERVED_BINDINGS", source)

    def test_no_other_shared_helper_took_over_the_branch(self):
        """The dispatch must not have moved into another generic module."""
        shared = [
            REPO / "src/evograd/benchmark/cases.py",
            REPO / "src/evograd/benchmark/core/tasks.py",
            REPO / "src/evograd/benchmark/operator_suite/cases/__init__.py",
        ]
        for path in shared:
            with self.subTest(module=path.name):
                self.assertNotIn("qwen3_0_6b", path.read_text())


class TestTheBinderTakesSuppliedConfiguration(unittest.TestCase):
    """A synthetic binding: no registered model, no snapshot, no model name.

    The point is that :func:`bind_observed_cases` applies whatever it is given.
    The stand-in overrides only where the cases come from; every rule the
    binder implements -- suite naming, append/prepend, the coverage mirror, the
    refusal to overwrite an existing suite -- is exercised unchanged.
    """

    class _Supplied(ObservedBinding):
        """An ObservedBinding whose cases are handed over rather than read."""

        def workloads(self, primitive):
            from evograd.opdecl import Workload

            return (
                Workload(dims=dict(rows=3, hidden=5), dtype="float32"),
                Workload(dims=dict(rows=7, hidden=9), dtype="float32"),
            )

    def _primitive(self):
        return bind_suite_cases(PRIMITIVES["rmsnorm"])

    def test_supplied_cases_reach_the_named_suite(self):
        task = bind_observed_cases(
            self._primitive(),
            [self._Supplied("a_synthetic_workload", "rmsnorm", "synthetic_observed")],
        )
        cases = task.benchmark_suites["synthetic_observed"]
        self.assertEqual([c.dims for c in cases],
                         [{"rows": 3, "hidden": 5}, {"rows": 7, "hidden": 9}])

    def test_append_and_prepend_are_honoured(self):
        base = self._primitive()
        appended = bind_observed_cases(
            base, [self._Supplied("w", "rmsnorm", "s", coverage="append")]
        )
        prepended = bind_observed_cases(
            base, [self._Supplied("w", "rmsnorm", "s", coverage="prepend")]
        )
        supplied = appended.benchmark_suites["s"]
        self.assertEqual(appended.coverage[-len(supplied):], supplied)
        self.assertEqual(prepended.coverage[:len(supplied)], supplied)
        self.assertEqual(len(appended.coverage), len(prepended.coverage))

    def test_the_coverage_mirror_follows_the_result(self):
        task = bind_observed_cases(
            self._primitive(),
            [self._Supplied("w", "rmsnorm", "s", coverage="prepend",
                            mirrors_coverage="coverage")],
        )
        self.assertEqual(task.benchmark_suites["coverage"], task.coverage)

    def test_a_primitive_no_binding_names_is_untouched(self):
        base = self._primitive()
        self.assertIs(
            bind_observed_cases(base, [self._Supplied("w", "softmax", "s")]), base
        )

    def test_binding_over_an_existing_suite_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exactly one place"):
            bind_observed_cases(
                self._primitive(),
                [self._Supplied("w", "rmsnorm", "legacy")],
            )

    def test_two_workloads_may_bind_one_primitive_under_distinct_suites(self):
        """The supported case: one contract, two architectures, two suites."""
        indexed = index_bindings([
            ObservedBinding("model_a", "rmsnorm", "a_observed"),
            ObservedBinding("model_b", "rmsnorm", "b_observed"),
        ])
        self.assertEqual(len(indexed["rmsnorm"]), 2)
        self.assertEqual([b.suite for b in indexed["rmsnorm"]],
                         ["a_observed", "b_observed"])

    def test_the_same_primitive_and_suite_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "under the suite name"):
            index_bindings([
                ObservedBinding("model_a", "rmsnorm", "shared_observed"),
                ObservedBinding("model_b", "rmsnorm", "shared_observed"),
            ])

    def test_no_bindings_at_all_is_a_no_op(self):
        base = self._primitive()
        self.assertIs(bind_observed_cases(base), base)
        self.assertIs(bind_observed_cases(base, []), base)


class TestTwoModelsBindTheSamePrimitive(unittest.TestCase):
    """Qwen3 and Llama-3 both run RMSNorm; each contributes its own suite.

    Synthetic bindings supply their own cases, so this exercises the binder's
    rules -- distinct suites coexisting, ordering, coverage mirroring, refusal
    of a real collision, and no mutation -- without needing either model's
    harvest to have been run.
    """

    class _Supplied(ObservedBinding):
        """A binding whose cases are handed over rather than read."""

        def workloads(self, primitive):
            from evograd.opdecl import Workload

            tag = len(self.workload)
            return (
                Workload(dims=dict(rows=tag, hidden=5), dtype="float32"),
                Workload(dims=dict(rows=tag + 1, hidden=9), dtype="float32"),
            )

    def _base(self):
        return bind_suite_cases(PRIMITIVES["rmsnorm"])

    def _both(self, **kwargs):
        return (
            self._Supplied("model_one", "rmsnorm", "one_observed", **kwargs),
            self._Supplied("model_twoo", "rmsnorm", "two_observed"),
        )

    def test_distinct_suites_coexist_on_one_primitive(self):
        task = bind_observed_cases(self._base(), self._both())
        self.assertIn("one_observed", task.benchmark_suites)
        self.assertIn("two_observed", task.benchmark_suites)
        self.assertNotEqual(
            task.benchmark_suites["one_observed"],
            task.benchmark_suites["two_observed"],
        )

    def test_the_same_primitive_and_suite_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "under the suite name"):
            index_bindings([
                ObservedBinding("model_a", "rmsnorm", "shared_observed"),
                ObservedBinding("model_b", "rmsnorm", "shared_observed"),
            ])

    def test_ordering_follows_the_supplied_order(self):
        first, second = self._both()
        forward = bind_observed_cases(self._base(), (first, second))
        backward = bind_observed_cases(self._base(), (second, first))
        base_len = len(self._base().coverage)
        self.assertEqual(
            forward.coverage[base_len:],
            forward.benchmark_suites["one_observed"]
            + forward.benchmark_suites["two_observed"],
        )
        self.assertEqual(
            backward.coverage[base_len:],
            backward.benchmark_suites["two_observed"]
            + backward.benchmark_suites["one_observed"],
        )

    def test_the_coverage_mirror_follows_both_contributions(self):
        task = bind_observed_cases(
            self._base(), self._both(mirrors_coverage="coverage")
        )
        self.assertEqual(task.benchmark_suites["coverage"], task.coverage)
        for suite in ("one_observed", "two_observed"):
            for case in task.benchmark_suites[suite]:
                self.assertIn(case, task.coverage)

    def test_neither_binding_mutates_the_primitive(self):
        primitive = PRIMITIVES["rmsnorm"]
        bind_observed_cases(self._base(), self._both())
        self.assertEqual(primitive.benchmark_suites, {})
        self.assertEqual(primitive.coverage, ())

    def test_an_unharvested_workload_contributes_nothing(self):
        """Llama-3-8B's snapshot does not exist yet, so its suites are empty
        and no declaration had to be edited to say so."""
        from evograd.benchmark.topdown import has_snapshot
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
            OBSERVED_BINDINGS,
        )

        self.assertFalse(has_snapshot("llama_3_2_1b"))
        base = self._base()
        self.assertIs(bind_observed_cases(base, OBSERVED_BINDINGS), base)
        self.assertNotIn("llama_3_2_1b_observed", TASKS["rmsnorm"].benchmark_suites)

    def test_the_registry_supplies_both_models(self):
        from evograd.benchmark.core.registry import _observed_bindings

        workloads = {b.workload for b in _observed_bindings()}
        self.assertEqual(workloads, {"qwen3_0_6b", "llama_3_2_1b"})


if __name__ == "__main__":
    unittest.main()
