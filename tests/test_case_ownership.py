"""Who owns a benchmark case, and what must not change when it moves.

A Level-1 primitive declares mathematics. It does not decide which shapes are
worth timing, where the shape regime splits, how cases are weighted, or what a
named suite selects -- those are benchmark decisions, and a primitive's cases
are the ones a model's own manifest observed in a captured run.

The tests here pin how those cases are bound -- their order and multiplicity,
the suites they are served under, and the fact that binding produces a *new*
declaration rather than editing the shared one.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from evograd.benchmark import TASKS, get_task
from evograd.benchmark.cases import (
    ObservedBinding,
    bind_observed_cases,
    index_bindings,
)
from evograd.benchmark.topdown.qwen3_0_6b.levels.level1.manifest import (
    OBSERVED_BINDINGS as QWEN3_0_6B_OBSERVED,
)
from evograd.ops import PRIMITIVES

REPO = Path(__file__).resolve().parents[1]

#: The primitives Qwen3-0.6B binds its observed cases to.
QWEN3_BOUND = tuple(binding.task for binding in QWEN3_0_6B_OBSERVED)


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

    def test_no_primitive_names_a_model_configuration(self):
        """Two documented exceptions, both a numerical rule rather than a case
        list: the RoPE and partial-RoPE input generators read ``rope_theta`` from
        each workload's own provenance, and need the published configuration
        table to do so."""
        offenders = []
        for package in sorted((REPO / "src/evograd/ops/level1").iterdir()):
            init = package / "__init__.py"
            if init.is_file() and "opdecl.models" in init.read_text():
                offenders.append(package.name)
        self.assertEqual(offenders, ["partial_rope", "rope"])

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
        self.assertEqual(count, "19")
        self.assertEqual(loaded, "[]", f"importing primitives loaded {loaded}")


class TestBindingDoesNotMutateThePrimitive(unittest.TestCase):
    def test_binding_returns_a_new_declaration(self):
        for name in sorted(QWEN3_BOUND):
            primitive = PRIMITIVES[name]
            with self.subTest(op=name):
                bound = bind_observed_cases(primitive, QWEN3_0_6B_OBSERVED)
                self.assertIsNot(bound, primitive)
                self.assertEqual(primitive.benchmark, ())
                self.assertEqual(primitive.benchmark_suites, {})
                self.assertIsNone(primitive.case_weight)

    def test_binding_twice_gives_the_same_result(self):
        """Binding reads the primitive; it must not accumulate into it."""
        for name, primitive in sorted(PRIMITIVES.items()):
            with self.subTest(op=name):
                first = bind_observed_cases(primitive, QWEN3_0_6B_OBSERVED)
                second = bind_observed_cases(primitive, QWEN3_0_6B_OBSERVED)
                self.assertEqual(first.benchmark, second.benchmark)
                self.assertEqual(first.coverage, second.coverage)
                self.assertEqual(
                    sorted(first.benchmark_suites), sorted(second.benchmark_suites)
                )

    def test_the_registry_serves_the_bound_declaration(self):
        for name in sorted(PRIMITIVES):
            with self.subTest(op=name):
                self.assertIs(get_task(name), TASKS[name])
                self.assertIsNot(TASKS[name], PRIMITIVES[name])
                self.assertEqual(TASKS[name].name, PRIMITIVES[name].name)
                self.assertEqual(TASKS[name].forward, PRIMITIVES[name].forward)


class TestObservedCasesJoinTheCoverage(unittest.TestCase):
    def test_observed_cases_are_served_as_a_suite_and_in_the_coverage(self):
        for name in QWEN3_BOUND:
            with self.subTest(op=name):
                bound = bind_observed_cases(PRIMITIVES[name], QWEN3_0_6B_OBSERVED)
                observed = bound.benchmark_suites["qwen3_0_6b_observed"]
                self.assertTrue(observed)
                for case in observed:
                    self.assertIn(case, bound.coverage)

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
            REPO / "src/evograd/benchmark/topdown/qwen3_5_4b/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/olmoe_1b_7b/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/convnextv2_tiny/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/qwen2_vl_2b/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/dit_s_2/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/torchrec_dlrm/levels/level1/manifest.py",
            REPO / "src/evograd/benchmark/topdown/megafold/levels/level1/manifest.py",
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
        ]
        for path in shared:
            with self.subTest(module=path.name):
                self.assertNotIn("qwen3_0_6b", path.read_text())


class TestTheBinderTakesSuppliedConfiguration(unittest.TestCase):
    """A synthetic binding: no registered model, no snapshot, no model name.

    The point is that :func:`bind_observed_cases` applies whatever it is given.
    The stand-in overrides only where the cases come from; every rule the
    binder implements -- suite naming, append/prepend, the refusal to overwrite
    an existing suite -- is exercised unchanged.
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
        return PRIMITIVES["rmsnorm"]

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

    def test_a_primitive_no_binding_names_is_untouched(self):
        base = self._primitive()
        self.assertIs(
            bind_observed_cases(base, [self._Supplied("w", "layernorm", "s")]), base
        )

    def test_binding_over_an_existing_suite_is_refused(self):
        bound = bind_observed_cases(self._primitive(), [self._Supplied("w", "rmsnorm", "s")])
        with self.assertRaisesRegex(ValueError, "exactly one place"):
            bind_observed_cases(bound, [self._Supplied("v", "rmsnorm", "s")])

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
    rules -- distinct suites coexisting, ordering, refusal of a real
    collision, and no mutation -- without needing either model's
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
        return PRIMITIVES["rmsnorm"]

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

    def test_neither_binding_mutates_the_primitive(self):
        primitive = PRIMITIVES["rmsnorm"]
        bind_observed_cases(self._base(), self._both())
        self.assertEqual(primitive.benchmark_suites, {})
        self.assertEqual(primitive.coverage, ())

    def test_an_unharvested_workload_contributes_nothing(self):
        """A model package can exist before its harvest does: a binding whose
        workload has no tracked snapshot adds no suite and no declaration had
        to be edited to say so. Llama-3.2-1B's snapshot is tracked now, so the
        unharvested state is produced by hiding it rather than assumed."""
        from unittest import mock

        from evograd.benchmark import cases as cases_module
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
            OBSERVED_BINDINGS,
        )

        base = self._base()
        with mock.patch.object(cases_module, "has_snapshot", return_value=False) \
                if hasattr(cases_module, "has_snapshot") else \
                mock.patch("evograd.benchmark.topdown.has_snapshot", return_value=False):
            self.assertIs(bind_observed_cases(base, OBSERVED_BINDINGS), base)

    def test_a_harvested_workload_contributes_its_observed_suite(self):
        """The Llama-3.2-1B snapshot (df3e806) is tracked, so the same binding
        now yields a suite whose shapes are the harvested model's -- 2048 wide,
        not Llama-3-8B's 4096 -- and the registry serves it on the primitive."""
        from evograd.benchmark.topdown import has_snapshot
        from evograd.benchmark.topdown.llama3_2_1b.levels.level1.manifest import (
            OBSERVED_BINDINGS,
        )

        self.assertTrue(has_snapshot("llama_3_2_1b"))
        bound = bind_observed_cases(self._base(), OBSERVED_BINDINGS)
        observed = bound.benchmark_suites["llama_3_2_1b_observed"]
        self.assertTrue(observed)
        for workload in observed:
            self.assertEqual(workload.dims["hidden"], 2048, workload.dims)
            self.assertEqual(workload.provenance.model, "llama_3_2_1b")
        self.assertIn("llama_3_2_1b_observed", TASKS["rmsnorm"].benchmark_suites)

    def test_the_registry_supplies_every_model(self):
        from evograd.benchmark.core.registry import _observed_bindings

        workloads = {b.workload for b in _observed_bindings()}
        self.assertEqual(workloads, {"qwen3_0_6b", "llama_3_2_1b", "qwen3_5_4b", "olmoe_1b_7b",
                                     "convnextv2_tiny", "qwen2_vl_2b", "dit_s_2",
                                     "torchrec_dlrm", "megafold"})


if __name__ == "__main__":
    unittest.main()
