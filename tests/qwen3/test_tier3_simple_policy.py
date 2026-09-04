"""The simplified, patch-set-matched Tier-3 numerical gate.

What these defend: that a threshold describes the run it is applied to. The
detailed policy derived its integration term from a bound-pair model patched at
*every* site and then applied it to candidates that replace one, so the number a
QKV candidate was measured against included three sites' worth of drift it never
caused. Here the trusted replacement is built from the candidate's own patch
set, the binding is asserted rather than assumed, and no candidate can reach the
derivation.
"""

from __future__ import annotations

import math
import unittest

import torch

from evograd.bench.workloads.qwen3.evaluation.tier3.simple import (
    FLOORS,
    HARD_METRICS,
    SAFETY_MARGIN,
    SCHEMA_VERSION,
    PatchSet,
    PolicyMismatch,
    SimplePolicy,
    check,
    derive_simple_policy,
    global_grad_rel_l2,
    matched_trusted_kernels,
)

QKV = PatchSet(patched=("qkv_norm_rope",), supporting=("attention",),
               expected_counts={"attention": 4, "qkv_norm_rope": 4})
RESIDUAL = PatchSet(patched=("residual_rmsnorm",), supporting=(),
                    expected_counts={"residual_rmsnorm": 8})


def _policy(patch_set=QKV, *, noise=0.0, drift=3e-3, margin=SAFETY_MARGIN,
            workload_id="smoke", workload_hash="wh", dtype="bfloat16",
            environment_hash="env") -> SimplePolicy:
    return derive_simple_policy(
        reference_noise=[{m: noise for m in HARD_METRICS}],
        trusted_drift=[{m: drift for m in HARD_METRICS}],
        workload_id=workload_id, workload_hash=workload_hash, dtype=dtype,
        environment_hash=environment_hash, patch_set=patch_set, margin=margin,
    )


def _metrics(logits=0.0, grads=0.0, **overrides):
    base = {
        "logits_rel_l2": logits, "global_grad_rel_l2": grads,
        "missing_grads": [], "grad_presence": {"missing": [], "shape_mismatch": []},
        "finite": {"logits": True, "loss": True, "gradients": True, "ok": True},
    }
    base.update(overrides)
    return base


class TestMatchedPatchSets(unittest.TestCase):
    def test_the_trusted_reference_patches_the_candidate_s_sites_only(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import build_registry
        from evograd.ops import OPS

        registry = build_registry()
        trusted = matched_trusted_kernels(dict(OPS), QKV, registry)
        self.assertEqual(trusted.patched, ("qkv_norm_rope",))

    def test_the_all_sites_reference_is_a_different_object(self):
        # The exact substitution this work replaces: `sites=None` is every site.
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import (
            bound_pair_identity_kernels, build_registry,
        )
        from evograd.ops import OPS

        registry = build_registry()
        every = bound_pair_identity_kernels(dict(OPS), None, registry=registry)
        self.assertEqual(len(every.patched), 4)
        self.assertEqual(len(matched_trusted_kernels(dict(OPS), QKV, registry).patched), 1)

    def test_a_residual_patch_set_carries_nothing(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.sites import build_registry
        from evograd.ops import OPS

        trusted = matched_trusted_kernels(dict(OPS), RESIDUAL, build_registry())
        self.assertEqual(trusted.patched, ("residual_rmsnorm",))
        self.assertEqual(RESIDUAL.supporting, ())

    def test_the_patch_set_key_names_the_calibration(self):
        self.assertEqual(QKV.key, "qkv_norm_rope")
        self.assertEqual(PatchSet((), (), {}).key, "eager")
        self.assertEqual(
            PatchSet(("attention", "qkv_norm_rope"), (), {}).key,
            "attention+qkv_norm_rope",
        )


class TestBindingRefusal(unittest.TestCase):
    def _bind(self, policy, **overrides):
        arguments = {"workload_id": "smoke", "workload_hash": "wh",
                     "dtype": "bfloat16", "environment_hash": "env",
                     "patch_set": QKV}
        arguments.update(overrides)
        policy.require_binding(**arguments)

    def test_a_matching_binding_is_accepted(self):
        self._bind(_policy())

    def test_a_different_patch_set_is_refused(self):
        with self.assertRaises(PolicyMismatch) as caught:
            self._bind(_policy(), patch_set=RESIDUAL)
        self.assertIn("patch set", str(caught.exception))

    def test_an_all_sites_candidate_cannot_use_a_single_site_policy(self):
        every = PatchSet(("attention", "qkv_norm_rope", "residual_rmsnorm",
                          "swiglu_mlp"), (), {})
        with self.assertRaises(PolicyMismatch):
            self._bind(_policy(), patch_set=every)

    def test_the_supporting_sites_are_part_of_the_identity(self):
        # Same patched site, different carried set: not the same run.
        detached = PatchSet(("qkv_norm_rope",), (), {"qkv_norm_rope": 4})
        with self.assertRaises(PolicyMismatch):
            self._bind(_policy(), patch_set=detached)

    def test_canonical_and_smoke_are_different_workloads(self):
        for field, value in (("workload_id", "canonical"),
                             ("workload_hash", "other"),
                             ("dtype", "float16"),
                             ("environment_hash", "another-gpu")):
            with self.assertRaises(PolicyMismatch, msg=field):
                self._bind(_policy(), **{field: value})

    def test_expected_counts_are_part_of_the_identity(self):
        four_layers = QKV
        twenty_eight = PatchSet(("qkv_norm_rope",), ("attention",),
                                {"attention": 28, "qkv_norm_rope": 28})
        self.assertFalse(four_layers.matches(twenty_eight))


class TestFormula(unittest.TestCase):
    def test_threshold_is_the_max_of_three_terms_times_the_margin(self):
        policy = _policy(noise=1e-3, drift=4e-3)
        for metric in HARD_METRICS:
            self.assertAlmostEqual(policy.thresholds[metric], 4e-3 * SAFETY_MARGIN)
            self.assertEqual(policy.derivation[metric]["binding_term"], "trusted_drift")

    def test_reference_noise_can_be_the_binding_term(self):
        policy = _policy(noise=9e-3, drift=1e-3)
        for metric in HARD_METRICS:
            self.assertAlmostEqual(policy.thresholds[metric], 9e-3 * SAFETY_MARGIN)
            self.assertEqual(policy.derivation[metric]["binding_term"], "reference_noise")

    def test_the_floor_binds_when_both_measurements_are_zero(self):
        policy = _policy(noise=0.0, drift=0.0)
        for metric in HARD_METRICS:
            self.assertAlmostEqual(policy.thresholds[metric],
                                   FLOORS[metric] * SAFETY_MARGIN)
            self.assertEqual(policy.derivation[metric]["binding_term"], "floor")

    def test_the_margin_is_two_and_is_recorded(self):
        self.assertEqual(SAFETY_MARGIN, 2.0)
        policy = _policy()
        self.assertEqual(policy.margin, 2.0)
        for metric in HARD_METRICS:
            self.assertEqual(policy.derivation[metric]["margin"], 2.0)

    def test_the_maximum_is_taken_not_a_quantile(self):
        policy = derive_simple_policy(
            reference_noise=[{m: 0.0 for m in HARD_METRICS}],
            trusted_drift=[{m: v for m in HARD_METRICS} for v in (1e-3, 5e-3, 2e-3)],
            workload_id="w", workload_hash="h", dtype="bfloat16",
            environment_hash="e", patch_set=QKV,
        )
        for metric in HARD_METRICS:
            self.assertAlmostEqual(policy.thresholds[metric], 5e-3 * 2.0)


class TestCalibrationIsCandidateFree(unittest.TestCase):
    def test_derivation_takes_only_reference_samples(self):
        import inspect

        parameters = set(inspect.signature(derive_simple_policy).parameters)
        self.assertIn("reference_noise", parameters)
        self.assertIn("trusted_drift", parameters)
        for forbidden in ("candidate", "candidate_samples", "program", "kernels"):
            self.assertNotIn(forbidden, parameters)

    def test_both_reference_sample_sets_are_required(self):
        with self.assertRaises(ValueError):
            derive_simple_policy(
                reference_noise=[], trusted_drift=[{m: 0.0 for m in HARD_METRICS}],
                workload_id="w", workload_hash="h", dtype="bfloat16",
                environment_hash="e", patch_set=QKV,
            )

    def test_the_calibration_driver_takes_a_patch_set_not_a_program(self):
        import inspect

        from evograd.bench.workloads.qwen3.evaluation.tier3 import calibrate_simple

        parameters = set(inspect.signature(calibrate_simple.calibrate).parameters)
        self.assertIn("sites", parameters)
        for forbidden in ("candidate", "program", "program_path"):
            self.assertNotIn(forbidden, parameters)

    def test_holdout_seeds_do_not_overlap_the_calibration_seeds(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import calibrate_simple

        self.assertFalse(set(calibrate_simple.CALIBRATION_SEEDS)
                         & set(calibrate_simple.HOLDOUT_SEEDS))


class TestStreamingGlobalGradient(unittest.TestCase):
    def test_it_matches_the_closed_form(self):
        reference = {"a": torch.ones(4, 4), "b": torch.full((8,), 2.0)}
        candidate = {"a": torch.ones(4, 4) + 0.5, "b": torch.full((8,), 2.0)}
        expected = math.sqrt(16 * 0.25) / math.sqrt(16 * 1.0 + 8 * 4.0)
        self.assertAlmostEqual(global_grad_rel_l2(candidate, reference)["rel_l2"],
                               expected, places=12)

    def test_identical_gradients_give_zero(self):
        reference = {"a": torch.randn(6, 7), "b": torch.randn(3)}
        candidate = {k: v.clone() for k, v in reference.items()}
        self.assertEqual(global_grad_rel_l2(candidate, reference)["rel_l2"], 0.0)

    def test_it_never_concatenates_the_gradients(self):
        import inspect

        source = inspect.getsource(global_grad_rel_l2)
        for forbidden in ("torch.cat", "torch.stack", "torch.hstack"):
            self.assertNotIn(forbidden, source)

    def test_it_accumulates_in_float64(self):
        # A float32 running sum loses the small summands across 300+ tensors of
        # very different scale. One huge and many tiny is the shape that shows it.
        reference = {"big": torch.full((1000,), 1e4)}
        candidate = {"big": torch.full((1000,), 1e4)}
        for index in range(200):
            reference[f"small{index}"] = torch.full((10,), 1e-4)
            candidate[f"small{index}"] = torch.full((10,), 1e-4 + 1e-8)
        result = global_grad_rel_l2(candidate, reference)
        self.assertGreater(result["rel_l2"], 0.0)
        self.assertTrue(math.isfinite(result["rel_l2"]))

    def test_a_missing_gradient_is_reported_not_skipped(self):
        reference = {"a": torch.ones(4), "b": torch.ones(4)}
        result = global_grad_rel_l2({"a": torch.ones(4)}, reference)
        self.assertEqual(result["missing"], ["b"])
        self.assertEqual(result["missing_count"], 1)
        self.assertFalse(result["ok"])

    def test_a_non_finite_gradient_is_reported(self):
        reference = {"a": torch.ones(4)}
        candidate = {"a": torch.tensor([1.0, float("nan"), 1.0, 1.0])}
        result = global_grad_rel_l2(candidate, reference)
        self.assertEqual(result["non_finite"], ["a"])
        self.assertFalse(result["ok"])

    def test_a_shape_mismatch_is_reported(self):
        result = global_grad_rel_l2({"a": torch.ones(3)}, {"a": torch.ones(4)})
        self.assertEqual(result["shape_mismatch"], ["a"])
        self.assertFalse(result["ok"])

    def test_extra_candidate_gradients_are_recorded(self):
        result = global_grad_rel_l2({"a": torch.ones(4), "z": torch.ones(2)},
                                    {"a": torch.ones(4)})
        self.assertEqual(result["extra"], ["z"])


class TestTheGateItself(unittest.TestCase):
    def test_a_trusted_sized_deviation_passes(self):
        policy = _policy(drift=3e-3)          # thresholds 6e-3
        verdict = check(policy, _metrics(logits=3e-3, grads=3e-3))
        self.assertTrue(verdict["ok"], verdict["reason"])

    def test_just_below_the_threshold_passes(self):
        policy = _policy(drift=3e-3)
        threshold = policy.thresholds["logits_rel_l2"]
        verdict = check(policy, _metrics(logits=threshold * 0.999))
        self.assertTrue(verdict["ok"], verdict["reason"])

    def test_just_above_the_threshold_fails(self):
        policy = _policy(drift=3e-3)
        threshold = policy.thresholds["logits_rel_l2"]
        verdict = check(policy, _metrics(logits=threshold * 1.001))
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "logits_rel_l2")

    def test_exactly_at_the_threshold_passes(self):
        policy = _policy(drift=3e-3)
        verdict = check(policy, _metrics(logits=policy.thresholds["logits_rel_l2"]))
        self.assertTrue(verdict["ok"])

    def test_a_global_gradient_excursion_fails(self):
        policy = _policy(drift=3e-3)
        threshold = policy.thresholds["global_grad_rel_l2"]
        verdict = check(policy, _metrics(grads=threshold * 1.5))
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["failed_at"], "global_grad_rel_l2")

    def test_a_missing_gradient_outranks_a_magnitude(self):
        policy = _policy(drift=3e-3)
        verdict = check(policy, _metrics(
            logits=1.0, missing_grads=["model.layers.0.self_attn.q_proj.weight"]))
        self.assertEqual(verdict["failed_at"], "presence")

    def test_non_finite_outranks_a_magnitude(self):
        policy = _policy(drift=3e-3)
        verdict = check(policy, _metrics(
            logits=1.0,
            finite={"logits": False, "loss": True, "gradients": True, "ok": False}))
        self.assertEqual(verdict["failed_at"], "finite")

    def test_only_the_four_questions_are_hard(self):
        self.assertEqual(HARD_METRICS, ("logits_rel_l2", "global_grad_rel_l2"))
        policy = _policy(drift=3e-3)
        # A wild per-role diagnostic does not fail the gate.
        verdict = check(policy, _metrics(
            logits=1e-4, grads=1e-4,
            loss_rel_delta=0.9, logits_max_abs_over_rms=50.0))
        self.assertTrue(verdict["ok"])
        self.assertIn("loss_rel_delta", verdict["diagnostics"])

    def test_the_verdict_carries_values_thresholds_and_ratios(self):
        policy = _policy(drift=3e-3)
        verdict = check(policy, _metrics(logits=3e-3, grads=1.5e-3))
        self.assertEqual(set(verdict["measured"]), set(HARD_METRICS))
        self.assertEqual(set(verdict["thresholds"]), set(HARD_METRICS))
        self.assertAlmostEqual(verdict["ratios"]["logits_rel_l2"], 0.5)


class TestSchemaAndLegacy(unittest.TestCase):
    def test_the_schema_is_version_three(self):
        self.assertEqual(SCHEMA_VERSION, "evograd-qwen3-t3-numerics/3")

    def test_a_policy_round_trips(self):
        policy = _policy()
        restored = SimplePolicy.from_dict(policy.to_dict())
        self.assertEqual(restored.thresholds, policy.thresholds)
        self.assertTrue(restored.patch_set.matches(policy.patch_set))

    def test_an_older_schema_is_refused(self):
        payload = _policy().to_dict()
        payload["schema"] = "evograd-qwen3-t3-numerics/2"
        with self.assertRaises(PolicyMismatch):
            SimplePolicy.from_dict(payload)

    def test_the_detailed_policy_still_loads_and_is_unchanged(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3 import numerics

        self.assertEqual(numerics.SAFETY_MARGIN, 2.0)
        self.assertEqual(numerics.GATED_METRICS, ("rel_l2", "max_abs_over_rms"))

    def test_the_gate_accepts_a_simple_policy_in_either_role(self):
        import inspect

        from evograd.bench.workloads.qwen3.evaluation.tier3 import gate

        parameters = inspect.signature(gate.check_model_correctness).parameters
        self.assertIn("simple_policy", parameters)
        self.assertIn("simple_primary", parameters)
        self.assertIs(parameters["simple_policy"].default, None)
        self.assertIs(parameters["simple_primary"].default, False)

    def test_the_local_boundary_gate_is_untouched(self):
        # Part 5: the elementwise layer must not be weakened by any of this.
        from evograd.bench.workloads.qwen3.evaluation.tier3 import boundary

        self.assertEqual(boundary.SCHEMA_VERSION, "evograd-qwen3-t3-boundary/2")
        source = inspect_source(boundary.BoundaryReport.to_dict)
        self.assertIn("failures", source)
        self.assertIn("not failures", source)


def inspect_source(function) -> str:
    import inspect

    return inspect.getsource(function)


if __name__ == "__main__":
    unittest.main()


class TestTrustedProvidersAndNegativeControls(unittest.TestCase):
    """The acceptance criteria, pinned to what was actually measured.

    Values are the smoke (4-layer, 256-token) measurements recorded in
    ``simple-policy-20260904-113848``. They are here so the policy's *decisions*
    are regression-tested on CPU: the numbers came from the GPU sweep, and a
    change that silently flips one of these verdicts fails here.
    """

    # patch set -> (logits threshold, global-grad threshold), as calibrated
    THRESHOLDS = {
        "residual_rmsnorm": (6.8569e-03, 1.6178e-02),
        "qkv_norm_rope": (2.0000e-05, 2.0000e-04),
    }

    # provider -> (patch set, logits rel_l2, global grad rel_l2, must pass)
    MEASURED = {
        # trusted references and providers
        "structural":          ("qkv_norm_rope",    0.0,        0.0,        True),
        "bound_qkv":           ("qkv_norm_rope",    0.0,        0.0,        True),
        "bound_residual":      ("residual_rmsnorm", 3.4285e-03, 8.0890e-03, True),
        "trusted_liger_resid": ("residual_rmsnorm", 2.6259e-03, 8.4133e-03, True),
        # an independent trusted provider that does NOT fit the qkv policy
        "trusted_compile_qkv": ("qkv_norm_rope",    1.0971e-02, 1.5695e-02, False),
        # deliberate defects: every one must be rejected
        "ctl_wrong_output_once":  ("qkv_norm_rope", 1.2982e-01, 1.8303e-01, False),
        "ctl_wrong_output_all":   ("qkv_norm_rope", 3.5038e-02, 5.0228e-02, False),
        "ctl_q_norm_scale":       ("qkv_norm_rope", 1.4054e-02, 2.1482e-02, False),
        "ctl_k_norm_scale":       ("qkv_norm_rope", 1.4151e-02, 2.1694e-02, False),
        "ctl_rope_sign":          ("qkv_norm_rope", 6.0482e-01, 8.7736e-01, False),
        "ctl_dropped_gradient":   ("qkv_norm_rope", 0.0,        2.9991e-01, False),
        "ctl_duplicated_grad":    ("qkv_norm_rope", 0.0,        3.6922e-01, False),
    }

    def _verdict(self, patch_key, logits, grads):
        logits_threshold, grad_threshold = self.THRESHOLDS[patch_key]
        policy = SimplePolicy(
            workload_id="smoke", workload_hash="wh", dtype="bfloat16",
            environment_hash="env",
            patch_set=QKV if patch_key == "qkv_norm_rope" else RESIDUAL,
            thresholds={"logits_rel_l2": logits_threshold,
                        "global_grad_rel_l2": grad_threshold},
            derivation={},
        )
        return check(policy, _metrics(logits=logits, grads=grads))

    def test_every_recorded_provider_gets_its_recorded_verdict(self):
        for name, (patch_key, logits, grads, expected) in self.MEASURED.items():
            with self.subTest(provider=name):
                verdict = self._verdict(patch_key, logits, grads)
                self.assertEqual(verdict["ok"], expected,
                                 f"{name}: {verdict.get('reason')}")

    def test_the_trusted_residual_providers_are_accepted(self):
        for name in ("bound_residual", "trusted_liger_resid"):
            patch_key, logits, grads, _ = self.MEASURED[name]
            self.assertTrue(self._verdict(patch_key, logits, grads)["ok"], name)

    def test_every_negative_control_is_rejected(self):
        controls = {k: v for k, v in self.MEASURED.items() if k.startswith("ctl_")}
        self.assertEqual(len(controls), 7)
        for name, (patch_key, logits, grads, _) in controls.items():
            with self.subTest(control=name):
                self.assertFalse(self._verdict(patch_key, logits, grads)["ok"], name)

    def test_a_gradient_only_defect_is_caught_by_the_global_gradient_metric(self):
        # Both dropped and duplicated contributions leave the logits bit-exact;
        # without the global gradient term the gate would pass them.
        for name in ("ctl_dropped_gradient", "ctl_duplicated_grad"):
            patch_key, logits, grads, _ = self.MEASURED[name]
            self.assertEqual(logits, 0.0)
            verdict = self._verdict(patch_key, logits, grads)
            self.assertEqual(verdict["failed_at"], "global_grad_rel_l2", name)

    def test_the_qkv_policy_rejects_a_trusted_provider_which_blocks_promotion(self):
        # The recorded blocker: torch.compile of the declared spelling is a
        # correct, independent implementation and the calibrated qkv threshold
        # rejects it, so the policy may not be promoted for that patch set.
        patch_key, logits, grads, expected = self.MEASURED["trusted_compile_qkv"]
        self.assertFalse(expected)
        verdict = self._verdict(patch_key, logits, grads)
        self.assertFalse(verdict["ok"])
        self.assertGreater(verdict["ratios"]["logits_rel_l2"], 100.0)

    def test_the_qkv_threshold_is_floor_driven_and_the_residual_one_is_not(self):
        qkv_logits, qkv_grad = self.THRESHOLDS["qkv_norm_rope"]
        self.assertAlmostEqual(qkv_logits, FLOORS["logits_rel_l2"] * SAFETY_MARGIN)
        self.assertAlmostEqual(qkv_grad, FLOORS["global_grad_rel_l2"] * SAFETY_MARGIN)
        residual_logits, _ = self.THRESHOLDS["residual_rmsnorm"]
        self.assertGreater(residual_logits, FLOORS["logits_rel_l2"] * SAFETY_MARGIN)
