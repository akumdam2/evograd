"""The three gaps the whole-model envelope alone could not close.

A provider that is a function of its arguments, checked by calling it more
times than the model will; every one of the model's invocations checked against
the declaration on the model's own tensors; and the four quantities a training
step produces judged apart instead of pooled. Each of those is a different way
for a wrong kernel to look right, and each is tested here on a model a CPU can
run in a second.
"""

from __future__ import annotations

import unittest

import torch

from evograd.bench.workloads.qwen3.evaluation.tier3 import boundary
from evograd.bench.tier3_gate import numerics
from evograd.bench.workloads.qwen3.evaluation.tier3 import purity
from evograd.bench.tier3_gate.numerics import (
    KIND_EXP_AVG,
    KIND_EXP_AVG_SQ,
    KIND_GRADIENT,
    KIND_STEP,
    KIND_UPDATE,
    check_against,
    derive_envelope,
    group_key,
)
from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
    bound_pair_identity_kernels,
    expected_counts,
    qwen3_sites,
    set_tap,
    structural_identity_kernels,
)
from evograd.bench.workloads.qwen3.evaluation.tier3.workload import Qwen3Workload
from evograd.ops import OPS

#: Two layers, 64 hidden. Small enough to run everywhere, structurally
#: complete: two of every site and four residual fusions across all three
#: categories, which is what the coverage rules are written against.
SMALL = dict(
    device="cpu", dtype="float32", batch_size=1, seq_len=16,
    arch_overrides={
        "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "vocab_size": 128, "max_position_embeddings": 64,
    },
)


def _workload():
    return Qwen3Workload(**SMALL)


def _residual_kernel():
    registry = qwen3_sites()
    kernels = bound_pair_identity_kernels(OPS, None, registry)
    return registry.require("residual_rmsnorm").op, kernels.kernel_for("residual_rmsnorm")


# ── 1. purity ────────────────────────────────────────────────────────────────


class TestPurityCoverage(unittest.TestCase):
    def test_every_site_is_called_more_often_than_the_model_calls_it(self):
        # 28 layers: the model makes 28 calls to three sites and 56 to the
        # fourth. A gate that stopped at the model's count could not tell
        # "correct throughout" from "correct exactly as far as we looked".
        model_calls = expected_counts(28)
        for site, minimum in purity.MIN_CALLS.items():
            with self.subTest(site=site):
                self.assertGreaterEqual(minimum, 2 * model_calls[site])

    def test_the_checkpoints_bracket_the_counts_that_matter(self):
        marks = purity.checkpoints(112)
        for boundary in (8, 28, 56, 112):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, marks)
        self.assertEqual(marks[0], 1)
        self.assertEqual(marks[-1], 112)

    def test_a_shorter_run_keeps_only_the_checkpoints_it_reaches(self):
        marks = purity.checkpoints(9)
        self.assertEqual(marks, [1, 2, 3, 7, 8, 9])


class TestPurityDetection(unittest.TestCase):
    def test_a_pure_kernel_passes_and_is_called_bitwise(self):
        op_name, kernel = _residual_kernel()
        report = purity.check_site("residual_rmsnorm", op_name, kernel,
                                      device="cpu", calls=10)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["determinism_regime"], "deterministic")
        self.assertIsNone(report["first_drift"])
        self.assertEqual(report["median_consecutive_spread"], 0.0)

    def test_a_kernel_that_turns_wrong_after_eight_calls_is_rejected(self):
        # The control the whole-model envelope could not catch: eight correct
        # calls, then 2%. Two percent is inside the operator's declared 8%
        # rtol, so only the determinism question finds it.
        op_name, kernel = _residual_kernel()
        state = {"n": 0}

        def stateful(*args):
            state["n"] += 1
            out = kernel(*args)
            return (out[0] * 1.02, out[1]) if state["n"] > 8 else out

        report = purity.check_site("residual_rmsnorm", op_name, stateful,
                                      device="cpu", calls=12)
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_drift"]["call"], 9)
        self.assertIsNotNone(report["first_drift"]["result"])

    def test_the_declared_tolerance_is_what_a_wrong_answer_is_judged_by(self):
        op_name, kernel = _residual_kernel()
        state = {"n": 0}

        def wrong(*args):
            state["n"] += 1
            out = kernel(*args)
            return (out[0] * 10.0, out[1]) if state["n"] > 2 else out

        report = purity.check_site("residual_rmsnorm", op_name, wrong,
                                      device="cpu", calls=5)
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_divergence"]["call"], 3)
        self.assertIn("atol", report["first_divergence"])

    def test_a_kernel_that_mutates_an_input_is_rejected_not_crashed_through(self):
        op_name, kernel = _residual_kernel()

        def mutating(x, r, weight, eps):
            x.add_(1.0)
            return kernel(x, r, weight, eps)

        report = purity.check_site("residual_rmsnorm", op_name, mutating,
                                      device="cpu", calls=4)
        self.assertFalse(report["ok"])
        self.assertIn("reason", report)

    def test_gradients_are_compared_and_not_only_forward_outputs(self):
        op_name, kernel = _residual_kernel()
        report = purity.check_site("residual_rmsnorm", op_name, kernel,
                                      device="cpu", calls=3)
        self.assertTrue(any(n.startswith("grad:") for n in report["results_checked"]))
        self.assertTrue(any(n.startswith("out:") for n in report["results_checked"]))

    def test_the_production_default_is_named_as_not_a_provider(self):
        # The registry's own spelling is the model's code and is not callable
        # with the declared signature at all -- saying so beats inventing a
        # subject to test.
        registry = qwen3_sites()
        kernels = structural_identity_kernels(registry)
        report = purity.check_site(
            "residual_rmsnorm", registry.require("residual_rmsnorm").op,
            kernels.kernel_for("residual_rmsnorm"), device="cpu", calls=3,
        )
        self.assertTrue(report["ok"])
        self.assertIn("skipped", report)


class TestPurityIsolation(unittest.TestCase):
    def test_a_rebuildable_provider_gets_a_child_process_spec(self):
        spec = purity.spec_for(structural_identity_kernels(qwen3_sites()))
        self.assertEqual(spec["provider"], "structural")
        self.assertEqual(set(spec["sites"]), set(purity.MIN_CALLS))

    def test_a_provider_with_no_reconstructible_origin_says_so(self):
        from evograd.bench.tier3_patch import KernelSet, KernelSource, patch

        registry = qwen3_sites()
        kernels = patch(
            KernelSet(registry=registry), "swiglu_mlp", lambda *a: None,
            source=KernelSource(site="swiglu_mlp", op_name="qwen3_swiglu_mlp",
                                module=None, origin="handwritten"),
        )
        self.assertIsNone(purity.spec_for(kernels))


# ── 2. the live boundary ─────────────────────────────────────────────────────


class TestBoundaryCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workload = _workload()
        cls.report = boundary.validate_all_invocations(
            workload, structural_identity_kernels(workload.site_registry)
        )

    def test_every_invocation_of_every_site_is_checked(self):
        self.assertEqual(self.report["observed_counts"],
                         self.report["expected_counts"])
        self.assertEqual(self.report["checked_invocations"],
                         sum(self.report["expected_counts"].values()))
        self.assertTrue(self.report["coverage_ok"])

    def test_the_three_residual_fusions_are_counted_apart(self):
        categories = self.report["residual_categories"]
        self.assertEqual(categories["post_attention"], 2)
        self.assertEqual(categories["mlp_to_next_input"], 1)
        self.assertEqual(categories["final_model_norm"], 1)

    def test_the_structural_provider_matches_its_declaration_everywhere(self):
        self.assertTrue(self.report["ok"], self.report["failures"][:2])
        self.assertEqual(self.report["failure_count"], 0)
        self.assertEqual(self.report["errors"], [])

    def test_summed_is_the_live_residual_stream_not_a_recomputation(self):
        self.assertTrue(self.report["summed_is_the_residual_stream"])

    def test_no_parameter_gradient_is_attributed_to_two_invocations(self):
        self.assertEqual(self.report["shared_parameter_boundaries"], [])

    def test_the_report_carries_summaries_and_no_tensors(self):
        def walk(value):
            if isinstance(value, dict):
                return any(walk(v) for v in value.values())
            if isinstance(value, list):
                return any(walk(v) for v in value)
            return torch.is_tensor(value)

        self.assertFalse(walk(self.report))


class TestBoundaryIdentity(unittest.TestCase):
    def test_an_invocation_id_names_layer_category_and_ordinal(self):
        self.assertEqual(
            boundary.invocation_id("residual_rmsnorm", (7, "post_attention"), 31),
            "residual_rmsnorm:layer7:post_attention:#31",
        )
        self.assertEqual(boundary.invocation_id("swiglu_mlp", 3, 12),
                         "swiglu_mlp:layer3:#12")

    def test_a_repeated_identity_is_reported_rather_than_overwritten(self):
        report = boundary.BoundaryReport()
        report.ids.add("swiglu_mlp:layer0:#1")
        listener = boundary.make_validator(
            lambda site: None, workload_case=lambda op: None, report=report
        )
        listener("swiglu_mlp", 0, {}, ())
        self.assertEqual(report.duplicates, ["swiglu_mlp:layer0:#1"])


class TestBoundaryIsShadowOnly(unittest.TestCase):
    def test_attaching_the_validator_changes_no_value_the_model_produces(self):
        # The probes are `view_as` aliases: identity in value and storage, a
        # distinct node in the graph. If that were not exact the whole gate
        # would be measuring its own instrumentation.
        workload = _workload()

        def run(with_tap):
            kernels = structural_identity_kernels(workload.site_registry)
            model, _ = workload.build_patched(kernels)
            if with_tap:
                from evograd.ops import get_op

                registry = workload.site_registry
                set_tap(model, boundary.make_validator(
                    lambda site: get_op(registry.require(site).op),
                    workload_case=lambda op: (
                        op.benchmark_workloads(suite="qwen3_0_6b_observed")
                        or op.benchmark
                    )[0],
                    report=boundary.BoundaryReport(),
                ))
            ids, labels = workload.batch_for(seed=0)
            outputs = model(input_ids=ids, labels=labels, use_cache=False)
            outputs.loss.backward()
            return (outputs.loss.detach().clone(),
                    {n: p.grad.clone() for n, p in model.named_parameters()
                     if p.grad is not None})

        plain_loss, plain_grads = run(False)
        tapped_loss, tapped_grads = run(True)
        self.assertTrue(torch.equal(plain_loss, tapped_loss))
        self.assertEqual(set(plain_grads), set(tapped_grads))
        for name, expected in plain_grads.items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(expected, tapped_grads[name]))


class TestBoundaryDetection(unittest.TestCase):
    def _reject(self, fault_name: str, magnitude: float = 0.02):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import catalogue

        workload = _workload()
        fault = next(f for f in catalogue(magnitudes=(magnitude,))
                     if f.name == fault_name)
        kernels = fault.apply(workload, structural_identity_kernels(
            workload.site_registry))
        return boundary.validate_all_invocations(workload, kernels)

    def test_a_two_percent_backward_error_is_caught_at_the_boundary(self):
        report = self._reject("grad_scale")
        self.assertFalse(report["ok"])
        self.assertTrue(any(f["result"].startswith("d") for f in report["failures"]))

    def test_a_two_percent_forward_error_is_caught_at_the_boundary(self):
        report = self._reject("output_scale")
        self.assertFalse(report["ok"])

    def test_a_single_role_error_is_caught_and_named(self):
        report = self._reject("one_role")
        self.assertFalse(report["ok"])

    def test_non_finite_output_is_caught_at_the_boundary(self):
        report = self._reject("non_finite")
        self.assertFalse(report["ok"])

    def test_a_failure_names_the_invocation_it_happened_in(self):
        report = self._reject("grad_scale")
        first = report["failures"][0]
        self.assertRegex(first["id"], r"^[a-z_]+:layer[0-9?]+(:[a-z_]+)?:#\d+$")
        self.assertIn("max_abs_err", first)


# ── 3. the separated envelopes ───────────────────────────────────────────────


def _step_pair():
    workload = _workload()
    kernels = structural_identity_kernels(workload.site_registry)
    from evograd.bench.workloads.qwen3.evaluation.tier3.gate import _step

    return (_step(workload, kernels, data_seed=0, learning_rate=1e-4),
            _step(workload, kernels, data_seed=0, learning_rate=1e-4))


class TestSeparatedEnvelopes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import _compare

        cls.compare = staticmethod(_compare)
        cls.candidate, cls.reference = _step_pair()
        cls.samples = _compare(cls.candidate, cls.reference)
        cls.envelopes = derive_envelope(cls.samples)

    def test_every_parameter_contributes_to_every_family(self):
        count = len(self.candidate["parameter_names"])
        for family in ("grads", "updates", "exp_avg", "exp_avg_sq", "steps"):
            with self.subTest(family=family):
                self.assertEqual(len(self.candidate[family]), count)
        self.assertEqual(self.candidate["missing_grads"], [])
        self.assertEqual(self.candidate["stateless_parameters"], [])

    def test_the_update_is_the_change_the_step_made_in_float32(self):
        update = next(iter(self.candidate["updates"].values()))
        self.assertEqual(update.dtype, torch.float32)
        # Non-zero: an update captured by aliasing the live parameter reads
        # after-minus-after and is silently zero everywhere.
        self.assertGreater(float(update.abs().max()), 0.0)
        # One AdamW step moves each element by about the learning rate; a
        # stepped *parameter* would be orders of magnitude larger.
        self.assertLess(float(update.abs().max()), 1e-3)

    def test_the_five_kinds_each_get_their_own_namespaces(self):
        kinds = {group.split("|", 1)[0] for group in self.envelopes}
        self.assertEqual(
            kinds, {"output", KIND_GRADIENT, KIND_UPDATE, KIND_EXP_AVG, KIND_EXP_AVG_SQ},
        )

    def test_a_step_counter_gets_a_verdict_and_never_an_envelope(self):
        self.assertNotIn(group_key(KIND_STEP, "up_proj"), self.envelopes)
        counters = [s for s in self.samples if s.get("kind") == KIND_STEP]
        self.assertTrue(counters)
        self.assertTrue(all(s["exact"] for s in counters))

    def test_a_role_is_never_judged_against_another_kinds_samples(self):
        one = derive_envelope([
            {"name": "a", "role": "up_proj", "kind": KIND_GRADIENT,
             "group": group_key(KIND_GRADIENT, "up_proj"),
             "rel_l2": 0.02, "max_abs_over_rms": 0.2, "finite": True},
        ])
        verdict = check_against(one, [
            {"name": "update:a", "role": "up_proj", "kind": KIND_UPDATE,
             "group": group_key(KIND_UPDATE, "up_proj"),
             "rel_l2": 0.01, "max_abs_over_rms": 0.1, "finite": True},
        ])
        self.assertFalse(verdict["ok"])
        self.assertIn("no envelope", verdict["exceeded"][0]["reason"])

    def test_the_identity_provider_fits_its_own_envelope(self):
        self.assertTrue(check_against(self.envelopes, self.samples)["ok"])

    def _corrupt(self, family: str, factor: float = 1.5):
        candidate = dict(self.candidate)
        candidate[family] = {n: t * factor for n, t in self.candidate[family].items()}
        return check_against(self.envelopes,
                             self.compare(candidate, self.reference))

    def test_a_wrong_update_with_correct_gradients_is_caught(self):
        verdict = self._corrupt("updates")
        self.assertFalse(verdict["ok"])
        kinds = {e.get("kind") for e in verdict["exceeded"]}
        self.assertEqual(kinds, {KIND_UPDATE})

    def test_a_corrupted_first_moment_is_caught_in_its_own_namespace(self):
        verdict = self._corrupt("exp_avg")
        self.assertFalse(verdict["ok"])
        self.assertEqual({e.get("kind") for e in verdict["exceeded"]}, {KIND_EXP_AVG})

    def test_a_corrupted_second_moment_is_caught_in_its_own_namespace(self):
        verdict = self._corrupt("exp_avg_sq")
        self.assertFalse(verdict["ok"])
        self.assertEqual({e.get("kind") for e in verdict["exceeded"]},
                         {KIND_EXP_AVG_SQ})

    def test_a_wrong_step_counter_is_caught_exactly(self):
        candidate = dict(self.candidate)
        candidate["steps"] = {n: v + 1 for n, v in self.candidate["steps"].items()}
        verdict = check_against(self.envelopes,
                                self.compare(candidate, self.reference))
        self.assertFalse(verdict["ok"])
        self.assertEqual({e.get("metric") for e in verdict["exceeded"]}, {"exact"})

    def test_a_single_roles_gradient_error_stays_in_that_role(self):
        candidate = dict(self.candidate)
        candidate["grads"] = {
            n: (t * 1.5 if "up_proj" in n else t)
            for n, t in self.candidate["grads"].items()
        }
        verdict = check_against(self.envelopes,
                                self.compare(candidate, self.reference))
        self.assertFalse(verdict["ok"])
        self.assertEqual({e.get("group") for e in verdict["exceeded"]},
                         {group_key(KIND_GRADIENT, "up_proj")})

    def test_an_omitted_optimizer_moment_is_a_failure_not_an_absence(self):
        candidate = dict(self.candidate)
        first = next(iter(self.candidate["exp_avg"]))
        candidate["exp_avg"] = {n: t for n, t in self.candidate["exp_avg"].items()
                                if n != first}
        samples = self.compare(candidate, self.reference)
        omitted = [s for s in samples if s["name"] == f"exp_avg:{first}"]
        self.assertEqual(len(omitted), 1)
        self.assertEqual(omitted[0]["rel_l2"], float("inf"))
        self.assertFalse(check_against(self.envelopes, samples)["ok"])

    def test_a_gradient_sized_floor_never_becomes_an_updates_threshold(self):
        gradient = numerics.floor_for(group_key(KIND_GRADIENT, "up_proj"))
        update = numerics.floor_for(group_key(KIND_UPDATE, "up_proj"))
        self.assertLess(update["rel_l2"], gradient["rel_l2"])
        self.assertLess(update["max_abs_over_rms"], gradient["max_abs_over_rms"])


if __name__ == "__main__":
    unittest.main()


# ── 4. the gate's order, and what it lets through ────────────────────────────


class _Spy:
    """Stands in for the timer, and records whether anything reached it."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return {"timed": True}


class TestGateOrder(unittest.TestCase):
    """The order is the point: a stage only runs if every earlier one passed."""

    def test_the_declared_order_is_the_one_the_gate_runs(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import STAGES

        self.assertEqual(STAGES, (
            "site_preflight", "provider_purity", "live_boundary",
            "numerical_envelopes", "loss_trajectory", "counts_and_provenance",
        ))

    def test_a_failed_preflight_stops_before_the_provider_is_ever_called(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import gate

        purity = _Spy()
        boundary = _Spy()
        with _patched(gate, purity=_module(run_for=purity),
                      boundary=_module(validate_all_invocations=boundary)):
            verdict = gate.check_model_correctness(
                _workload(), structural_identity_kernels(qwen3_sites()),
                policy=_fake_policy(), check_trajectory=False,
                preflight={"ok": False, "reason": "shape not supported"},
            )
        self.assertEqual(verdict["failed_at"], "site_preflight")
        self.assertEqual(purity.calls, 0)
        self.assertEqual(boundary.calls, 0)

    def test_an_impure_provider_never_reaches_the_model_or_the_boundary(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import gate

        boundary = _Spy()
        impure = {"ok": False, "sites": [
            {"site": "swiglu_mlp", "first_drift": {
                "call": 9, "result": "out:out", "max_abs_err": 0.4,
                "median_consecutive_spread": 0.0, "bound": 0.01,
                "regime": "deterministic"}},
        ]}
        with _patched(gate,
                      purity=_module(run_for=lambda *a, **k: impure),
                      boundary=_module(validate_all_invocations=boundary)):
            verdict = gate.check_model_correctness(
                _workload(), structural_identity_kernels(qwen3_sites()),
                policy=_fake_policy(), check_trajectory=False,
            )
        self.assertEqual(verdict["failed_at"], "provider_purity")
        self.assertIn("depends on its history", verdict["reason"])
        self.assertEqual(boundary.calls, 0)

    def test_a_boundary_failure_stops_before_any_whole_model_step(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import gate

        step = _Spy()
        broken = {"ok": False, "errors": [], "coverage_ok": True,
                  "failure_count": 3, "checked_invocations": 140,
                  "failures": [{"id": "swiglu_mlp:layer3:#17", "result": "out",
                                "max_abs_err": 0.5, "atol": 0.01}]}
        with _patched(gate,
                      purity=_module(run_for=lambda *a, **k: {"ok": True}),
                      boundary=_module(validate_all_invocations=lambda *a, **k: broken),
                      _step=step):
            verdict = gate.check_model_correctness(
                _workload(), structural_identity_kernels(qwen3_sites()),
                policy=_fake_policy(), check_trajectory=False,
            )
        self.assertEqual(verdict["failed_at"], "live_boundary")
        self.assertIn("swiglu_mlp:layer3:#17", verdict["reason"])
        self.assertEqual(step.calls, 0)


class TestTimingIsGatedOnCorrectness(unittest.TestCase):
    """Whether a timer would have run. No timing is collected either way."""

    def _run(self, verdict):
        from evograd.bench import tier3_runner

        timer = _Spy()
        workload = _Fixture(verdict)
        kernels = structural_identity_kernels(qwen3_sites())
        with _patched(tier3_runner, measure_step=timer):
            try:
                tier3_runner.model_correctness_check(
                    workload, kernels, verify=True, device="cpu"
                )
                admitted = True
            except tier3_runner.ModelCorrectnessFailure:
                admitted = False
        return admitted, timer

    def test_a_failing_gate_means_the_timer_is_never_reached(self):
        admitted, timer = self._run({"ok": False, "reason": "purity",
                                     "failed_at": "provider_purity"})
        self.assertFalse(admitted)
        self.assertEqual(timer.calls, 0)

    def test_a_passing_gate_admits_the_provider_to_timing(self):
        admitted, timer = self._run({"ok": True, "failed_at": None})
        self.assertTrue(admitted)
        # Admission is the claim being tested; the timer itself is not run here.
        self.assertEqual(timer.calls, 0)

    def test_the_failure_carries_the_stage_that_refused(self):
        from evograd.bench import tier3_runner

        with self.assertRaises(tier3_runner.ModelCorrectnessFailure) as caught:
            tier3_runner.model_correctness_check(
                _Fixture({"ok": False, "failed_at": "live_boundary",
                          "reason": "invocation 17 disagreed"}),
                structural_identity_kernels(qwen3_sites()),
                verify=True, device="cpu",
            )
        self.assertIn("invocation 17 disagreed", str(caught.exception))


class _Fixture:
    """A workload whose only job is to answer the gate with a fixed verdict."""

    def __init__(self, verdict):
        self._verdict = verdict

    def model_correctness(self, kernels, *, device):
        return self._verdict


def _module(**attributes):
    return type("stub", (), attributes)()


class _patched:
    """Swap module attributes for the duration of a block, then put them back."""

    def __init__(self, module, **attributes):
        self.module = module
        # No name filtering. `_step` is private and is precisely the attribute
        # a stage-order test must replace; silently dropping it made a spy that
        # could never fire and a test that passed without testing anything.
        self.attributes = dict(attributes)
        self.saved = {}

    #: Modules the gate reaches with `from . import <name>`, which reads the
    #: package attribute rather than `sys.modules` -- so that is where a
    #: stand-in has to go.
    SIBLINGS = ("purity", "boundary")

    def __enter__(self):
        import evograd.bench.workloads.qwen3.evaluation.tier3 as package

        for name, value in self.attributes.items():
            target = package if name in self.SIBLINGS else self.module
            self.saved[name] = (target, getattr(target, name, None))
            setattr(target, name, value)
        return self

    def __exit__(self, *exc):
        for name, (target, value) in self.saved.items():
            setattr(target, name, value)
        return False


def _fake_policy():
    from evograd.bench.tier3_gate.numerics import (
        NumericsPolicy,
        SCHEMA_VERSION,
        TrajectoryPolicy,
    )

    return NumericsPolicy(
        schema_version=SCHEMA_VERSION, workload_id="test", workload_hash="0000",
        environment={}, environment_hash="0000", envelopes={},
        trajectory=TrajectoryPolicy(horizon=1, optimizer="AdamW",
                                    learning_rate=1e-4, max_abs_delta=1.0,
                                    max_rel_delta=1.0),
    )


# ── 5. optimizer-update controls, and what bfloat16 can express ──────────────


class TestUlpPerturbation(unittest.TestCase):
    """A fault defined in units of the storage format cannot be rounded away."""

    def test_every_element_moves_away_from_zero(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import perturb_ulps

        clean = torch.tensor([1.0, -1.0, 0.013, -0.013, 0.0, -0.0],
                             dtype=torch.bfloat16)
        moved = perturb_ulps(clean, 2)
        self.assertTrue(bool((clean.view(torch.int16) != moved.view(torch.int16)).all()))
        for before, after in zip(clean.tolist(), moved.tolist()):
            with self.subTest(value=before):
                self.assertGreater(abs(after), abs(before))

    def test_two_ulps_at_the_projection_scale_is_one_adamw_step(self):
        # |p| ~ 1.3e-2 puts the bfloat16 ULP at 6.1e-5, and one AdamW step at
        # lr=1e-4 moves the weight by 1.22e-4. The control is therefore a whole
        # step's worth of error, not an arbitrary number.
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import perturb_ulps

        value = torch.tensor([0.013], dtype=torch.bfloat16)
        delta = float(perturb_ulps(value, 2)) - float(value)
        self.assertAlmostEqual(delta, 1.22e-4, delta=1e-6)


class TestObservability(unittest.TestCase):
    """"Not detected" and "nothing happened" must never read alike."""

    def _capture(self, values):
        return {name: torch.tensor(v, dtype=torch.bfloat16)
                for name, v in values.items()}

    def test_a_sub_ulp_update_perturbation_changes_no_stored_bit(self):
        # The measured root cause, in miniature: a norm weight sits at 1.0
        # where the bfloat16 ULP is 7.8e-3, so an AdamW step of 1e-4 rounds
        # away entirely and the realized update is exactly zero. Scaling zero
        # by 1.02 is still zero.
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import observability

        clean = self._capture({"model.norm.weight": [1.0, 1.0, 1.0]})
        before = clean["model.norm.weight"].float()
        stepped = (before + 1e-4 * 1.02).to(torch.bfloat16)
        evidence = observability(clean, {"model.norm.weight": stepped})
        self.assertFalse(evidence["observable"])
        self.assertEqual(evidence["stored_elements_changed"], 0)

    def test_a_ulp_perturbation_changes_every_stored_bit(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import (
            observability,
            perturb_ulps,
        )

        clean = self._capture({"model.layers.0.mlp.up_proj.weight": [0.013, -0.02, 1.0]})
        damaged = {n: perturb_ulps(t, 2) for n, t in clean.items()}
        evidence = observability(clean, damaged)
        self.assertTrue(evidence["observable"])
        self.assertEqual(evidence["stored_fraction_changed"], 1.0)
        self.assertEqual(evidence["roles_with_no_stored_change"], [])

    def test_the_classification_separates_the_two_failures_to_reject(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.controls import (
            DETECTED,
            MISSED,
            UNOBSERVABLE,
            classify,
        )

        nothing = {"observable": False, "roles_with_no_stored_change": ["final_norm"]}
        everything = {"observable": True, "roles_with_no_stored_change": []}
        self.assertEqual(classify(nothing, rejected=False), UNOBSERVABLE)
        self.assertEqual(classify(everything, rejected=False), MISSED)
        self.assertEqual(classify(everything, rejected=True), DETECTED)


class TestUpdateControlPolicy(unittest.TestCase):
    def test_the_ulp_control_is_required_and_wrong_update_is_a_diagnostic(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import (
            diagnostic_catalogue,
            state_catalogue,
        )

        required = {f.name for f in state_catalogue()}
        diagnostic = {f.name for f in diagnostic_catalogue()}
        self.assertIn("stored_param_ulp", required)
        self.assertNotIn("wrong_update", required)
        self.assertEqual(diagnostic, {"wrong_update"})

    def test_the_ulp_fault_moves_the_update_by_what_it_moved_the_parameter(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import state_catalogue

        fault = next(f for f in state_catalogue() if f.name == "stored_param_ulp")
        stored = torch.tensor([0.013, -0.013], dtype=torch.bfloat16)
        captured = {"stored": {"w": stored},
                    "updates": {"w": torch.zeros(2, dtype=torch.float32)}}
        damaged = fault.apply(captured)
        moved = damaged["stored"]["w"].float() - stored.float()
        self.assertTrue(torch.equal(damaged["updates"]["w"], moved))
        self.assertTrue(bool((moved != 0).all()))


class TestObservableUpdateFaultIsRejected(unittest.TestCase):
    """The end the gate is responsible for: an observable fault must fail."""

    @classmethod
    def setUpClass(cls):
        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import _compare, _step

        cls.compare = staticmethod(_compare)
        workload = _workload()
        kernels = structural_identity_kernels(workload.site_registry)
        cls.candidate = _step(workload, kernels, data_seed=0, learning_rate=1e-4)
        cls.reference = _step(workload, kernels, data_seed=0, learning_rate=1e-4)
        cls.envelopes = derive_envelope(_compare(cls.candidate, cls.reference))

    def test_a_ulp_corrupted_update_is_rejected(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import state_catalogue

        fault = next(f for f in state_catalogue() if f.name == "stored_param_ulp")
        verdict = check_against(
            self.envelopes, self.compare(fault.apply(self.candidate), self.reference)
        )
        self.assertFalse(verdict["ok"])
        self.assertEqual({e.get("kind") for e in verdict["exceeded"]},
                         {KIND_UPDATE})

    def test_it_is_observable_in_stored_state(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import (
            observability,
            state_catalogue,
        )

        fault = next(f for f in state_catalogue() if f.name == "stored_param_ulp")
        evidence = observability(self.candidate["stored"],
                                 fault.apply(self.candidate)["stored"])
        self.assertTrue(evidence["observable"])
        self.assertEqual(evidence["stored_fraction_changed"], 1.0)

    def test_the_gate_fails_at_numerical_envelopes(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import gate as gate_module
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import state_catalogue

        fault = next(f for f in state_catalogue() if f.name == "stored_param_ulp")
        damaged = fault.apply(self.candidate)
        policy = _fake_policy()
        policy.envelopes.update(self.envelopes)
        with _patched(gate_module,
                      purity=_module(run_for=lambda *a, **k: {"ok": True}),
                      boundary=_module(validate_all_invocations=lambda *a, **k: {
                          "ok": True, "errors": [], "coverage_ok": True,
                          "failure_count": 0, "checked_invocations": 10,
                          "failures": []}),
                      _step=lambda *a, **k: damaged):
            verdict = gate_module.check_model_correctness(
                _workload(), structural_identity_kernels(qwen3_sites()),
                policy=policy, check_trajectory=False,
                references={"eager": self.reference, "bound": self.reference},
            )
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "numerical_envelopes")

    def test_a_rejected_provider_never_reaches_timing(self):
        from evograd.bench import tier3_runner

        timer = _Spy()
        with _patched(tier3_runner, measure_step=timer):
            with self.assertRaises(tier3_runner.ModelCorrectnessFailure):
                tier3_runner.model_correctness_check(
                    _Fixture({"ok": False, "failed_at": "numerical_envelopes",
                              "reason": "update:model.layers.0... outside envelope"}),
                    structural_identity_kernels(qwen3_sites()),
                    verify=True, device="cpu",
                )
        self.assertEqual(timer.calls, 0)


class TestUpdateFaultReachesStoredState(unittest.TestCase):
    """A wrong update is only wrong once it has been stored."""

    def test_a_two_percent_update_fault_is_re_stored_before_it_is_measured(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import (
            diagnostic_catalogue,
            observability,
        )

        fault = diagnostic_catalogue(0.02)[0]
        # A projection weight: |p| ~ 1.3e-2, bfloat16 ULP 6.1e-5, and an AdamW
        # step of 1.22e-4 is exactly two of them. A norm weight sits at 1.0
        # where the ULP is 7.8e-3 and the same step rounds away completely.
        stored = torch.tensor([0.013] * 64 + [1.0] * 64, dtype=torch.bfloat16)
        before = stored.float() - 1.22e-4
        captured = {
            "stored": {"model.layers.0.self_attn.q_proj.weight": stored},
            "updates": {"model.layers.0.self_attn.q_proj.weight":
                        stored.float() - before},
        }
        damaged = fault.apply(captured)
        evidence = observability(captured["stored"], damaged["stored"])
        # The point of re-storing: some elements move, most do not.
        self.assertLess(evidence["stored_fraction_changed"], 1.0)

    def test_a_sub_ulp_update_fault_stores_nothing_and_is_classified_so(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.controls import (
            UNOBSERVABLE,
            classify,
        )
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import (
            diagnostic_catalogue,
            observability,
        )

        fault = diagnostic_catalogue(0.02)[0]
        stored = torch.full((128,), 1.0, dtype=torch.bfloat16)
        captured = {"stored": {"model.norm.weight": stored},
                    "updates": {"model.norm.weight": torch.zeros(128)}}
        damaged = fault.apply(captured)
        evidence = observability(captured["stored"], damaged["stored"])
        self.assertFalse(evidence["observable"])
        self.assertEqual(classify(evidence, rejected=False), UNOBSERVABLE)

    def test_an_optimizer_state_fault_is_not_called_unobservable(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.controls import (
            OPTIMIZER_SCOPE,
            classify,
        )
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import state_catalogue

        moment = next(f for f in state_catalogue() if f.name == "corrupt_exp_avg")
        self.assertEqual(moment.scope, "optimizer_state")
        verdict = classify({"observable": False, "roles_with_no_stored_change": []},
                           rejected=True, scope=moment.scope)
        self.assertIn(OPTIMIZER_SCOPE, verdict)
        self.assertNotIn("unobservable", verdict)


class TestTrajectoryLimitsCombine(unittest.TestCase):
    """The loss curve is bounded the way the tensors are: drift plus drift."""

    def _policy(self, ee, sb):
        from evograd.bench.workloads.qwen3.evaluation.tier3.numerics import (
            TrajectoryPolicy,
        )

        make = lambda a, r: TrajectoryPolicy(  # noqa: E731
            horizon=5, optimizer="AdamW", learning_rate=1e-4,
            max_abs_delta=a, max_rel_delta=r, margin=2.0,
        )
        return make(*ee), make(*sb)

    def test_the_two_halves_are_summed(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.numerics import (
            combined_trajectory,
        )

        hardware, integration = self._policy((6.5e-4, 5.3e-5), (6.6e-4, 5.4e-5))
        combined = combined_trajectory(hardware, integration)
        self.assertAlmostEqual(combined.max_abs_delta, 1.31e-3, places=5)
        self.assertAlmostEqual(combined.max_rel_delta, 1.07e-4, places=6)

    def test_a_deterministic_configuration_does_not_demand_a_bitwise_loss_curve(self):
        # The smoke config is small enough that eager-vs-eager is bitwise, so
        # the E/E limit derives to exactly zero. Held to that alone, a correct
        # provider is rejected for one ULP on one loss.
        from evograd.bench.workloads.qwen3.evaluation.tier3.numerics import (
            combined_trajectory,
        )

        hardware, integration = self._policy((0.0, 0.0), (1.04e-3, 8.6e-5))
        # A fresh Qwen3-0.6B starts near a loss of 11, which is what makes
        # 4.6e-4 a relative deviation of 4e-5 rather than 2e-4.
        curve = [11.5, 11.4, 11.3, 11.2, 11.1]
        drifted = [11.5, 11.4 + 4.6e-4, 11.3, 11.2, 11.1]
        self.assertFalse(hardware.check(curve, drifted)["ok"])
        self.assertTrue(combined_trajectory(hardware, integration)
                        .check(curve, drifted)["ok"])

    def test_a_real_divergence_is_still_rejected(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.numerics import (
            combined_trajectory,
        )

        hardware, integration = self._policy((6.5e-4, 5.3e-5), (6.6e-4, 5.4e-5))
        combined = combined_trajectory(hardware, integration)
        self.assertFalse(
            combined.check([11.5, 11.4, 11.3, 11.2, 11.1],
                           [11.5, 11.35, 11.3, 11.2, 11.1])["ok"]
        )

    def test_a_policy_without_an_integration_half_is_unchanged(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.numerics import (
            combined_trajectory,
        )

        hardware, _ = self._policy((6.5e-4, 5.3e-5), (0.0, 0.0))
        self.assertIs(combined_trajectory(hardware, None), hardware)
