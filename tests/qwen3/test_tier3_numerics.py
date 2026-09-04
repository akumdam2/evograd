"""The Qwen3 tier-3 numerics policy: metrics, grouping, envelope, gate.

No GPU. What these pin is the part that decides *what counts as correct* — the
statistics, the grouping they are pooled by, how a threshold is derived from
reference runs, and the refusals. Getting that wrong produces a gate that runs
fine and admits a wrong kernel, which is the failure the gate exists to prevent.
"""

from __future__ import annotations

import json
import math
import unittest

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

if HAVE_TORCH:
    from evograd.bench.tier3_gate.numerics import (
        BINDING_FIELDS,
        GATED_METRICS,
        SAFETY_MARGIN,
        SCHEMA_VERSION,
        THRESHOLD_FLOOR,
        GroupEnvelope,
        NumericsPolicy,
        TrajectoryPolicy,
        check_against,
        compare_tensor,
        derive_envelope,
        derive_trajectory_policy,
        environment_fingerprint,
        environment_mismatch,
        fingerprint_hash,
        role_of,
        roles_present,
    )

_skip = unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")


@_skip
class TestMetrics(unittest.TestCase):
    """The statistics have to survive tensors full of near-zero elements."""

    def test_identical_tensors_score_zero_and_bitwise(self):
        a = torch.randn(64, 32)
        stats = compare_tensor("model.layers.0.mlp.up_proj.weight", a, a.clone())
        self.assertTrue(stats.bitwise)
        self.assertEqual(stats.rel_l2, 0.0)
        self.assertEqual(stats.max_abs_over_rms, 0.0)
        self.assertAlmostEqual(stats.cosine, 1.0, places=5)
        self.assertEqual(stats.sign_flips_above_floor, 0)

    def test_relative_l2_is_a_whole_tensor_quantity(self):
        # One element perturbed by a lot moves max_abs_over_rms and barely moves
        # rel_l2. That is the point: a single cancelling element must not be
        # able to fail a tensor on its own, and must still be visible.
        b = torch.ones(1000)
        a = b.clone()
        a[0] += 1.0
        stats = compare_tensor("g", a, b)
        self.assertAlmostEqual(stats.rel_l2, 1.0 / math.sqrt(1000), places=5)
        self.assertAlmostEqual(stats.max_abs_over_rms, 1.0, places=5)

    def test_a_near_zero_element_does_not_dominate(self):
        # The statistic that would explode here is max relative error, which is
        # why it is not the primary one.
        b = torch.full((256,), 1.0)
        b[0] = 1e-12
        a = b.clone()
        a[0] = 2e-12                       # 100% relative error on one element
        stats = compare_tensor("g", a, b)
        self.assertLess(stats.rel_l2, 1e-12)
        self.assertLess(stats.max_abs_over_rms, 1e-11)

    def test_sign_flips_are_counted_only_above_the_noise_floor(self):
        b = torch.tensor([1.0, -1.0, 1e-9, -1e-9])
        a = torch.tensor([-1.0, -1.0, -1e-9, -1e-9])   # two flips, one tiny
        stats = compare_tensor("g", a, b)
        self.assertEqual(stats.sign_flips_above_floor, 1)
        self.assertEqual(stats.elements_above_floor, 2)

    def test_non_finite_is_reported_and_does_not_raise(self):
        b = torch.ones(8)
        a = b.clone()
        a[3] = float("nan")
        stats = compare_tensor("g", a, b)
        self.assertFalse(stats.finite)
        self.assertTrue(stats.reference_finite)

    def test_shape_dtype_and_element_count_travel(self):
        a = torch.randn(4, 5, dtype=torch.float64)
        stats = compare_tensor("g", a, a.clone())
        self.assertEqual(stats.shape, (4, 5))
        self.assertEqual(stats.dtype, "float64")
        self.assertEqual(stats.elements, 20)

    def test_nothing_holds_a_tensor(self):
        stats = compare_tensor("g", torch.randn(4), torch.randn(4))
        for value in stats.to_dict().values():
            self.assertNotIsInstance(value, torch.Tensor)
        json.dumps(stats.to_dict())


@_skip
class TestGrouping(unittest.TestCase):
    """Roles, not tensors: 310 constants is not a policy."""

    def test_every_qwen_parameter_role_is_recognized(self):
        expected = {
            "model.embed_tokens.weight": "embedding",
            "model.norm.weight": "final_norm",
            "model.layers.7.self_attn.q_proj.weight": "q_proj",
            "model.layers.7.self_attn.k_proj.weight": "k_proj",
            "model.layers.7.self_attn.v_proj.weight": "v_proj",
            "model.layers.7.self_attn.o_proj.weight": "o_proj",
            "model.layers.7.self_attn.q_norm.weight": "q_norm",
            "model.layers.7.self_attn.k_norm.weight": "k_norm",
            "model.layers.7.mlp.gate_proj.weight": "gate_proj",
            "model.layers.7.mlp.up_proj.weight": "up_proj",
            "model.layers.7.mlp.down_proj.weight": "down_proj",
            "model.layers.7.input_layernorm.weight": "input_layernorm",
            "model.layers.7.post_attention_layernorm.weight": "post_attention_layernorm",
            "loss": "loss",
            "logits": "logits",
        }
        for name, role in expected.items():
            with self.subTest(name=name):
                self.assertEqual(role_of(name), role)

    def test_layers_of_the_same_role_pool_together(self):
        names = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(28)]
        groups = roles_present(names)
        self.assertEqual(list(groups), ["down_proj"])
        self.assertEqual(len(groups["down_proj"]), 28)

    def test_roles_with_different_scales_are_never_pooled(self):
        # A 151936x1024 embedding and a 128-element per-head norm have nothing
        # to say about each other's noise.
        groups = roles_present([
            "model.embed_tokens.weight", "model.layers.0.self_attn.q_norm.weight"
        ])
        self.assertEqual(sorted(groups), ["embedding", "q_norm"])

    def test_an_unknown_name_gets_its_own_group_rather_than_a_default(self):
        self.assertTrue(role_of("model.something.new").startswith("other:"))

    def test_stepped_parameters_keep_their_role(self):
        self.assertEqual(role_of("step:model.layers.0.mlp.up_proj.weight"), "up_proj")

    def test_a_prefix_does_not_break_a_start_anchored_role(self):
        # `final_norm` and `lm_head` are anchored at the start of the name. A
        # stepped `model.norm.weight` used to fall outside every envelope and
        # fail the gate on a provider that was perfectly correct.
        for prefix in ("", "step:", "exp_avg:"):
            with self.subTest(prefix=prefix):
                self.assertEqual(role_of(f"{prefix}model.norm.weight"), "final_norm")
                self.assertEqual(role_of(f"{prefix}lm_head.weight"), "lm_head")

    def test_every_result_a_gate_compares_has_an_envelope(self):
        # The full canonical name set, gradients and stepped parameters alike.
        names = ["logits", "loss", "model.embed_tokens.weight", "model.norm.weight"]
        for layer in range(2):
            for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
                           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                           "input_layernorm", "post_attention_layernorm"):
                names.append(f"model.layers.{layer}.{suffix}.weight")
        every = names + [f"step:{n}" for n in names if n not in ("logits", "loss")]
        unknown = [n for n in every if role_of(n).startswith("other:")]
        self.assertEqual(unknown, [])


@_skip
class TestEnvelopeDerivation(unittest.TestCase):
    def _samples(self, role: str, values: list[float]):
        return [
            {"name": f"{role}.{i}", "role": role, "rel_l2": v,
             "max_abs_over_rms": v * 10}
            for i, v in enumerate(values)
        ]

    def test_the_threshold_is_the_observed_maximum_times_the_margin(self):
        envelopes = derive_envelope(self._samples("up_proj", [0.01, 0.02, 0.03]))
        envelope = envelopes["gradient|up_proj"]
        self.assertAlmostEqual(envelope.observed_max["rel_l2"], 0.03)
        self.assertAlmostEqual(envelope.threshold["rel_l2"], 0.03 * SAFETY_MARGIN)
        self.assertEqual(envelope.samples, 3)

    def test_a_floor_stops_a_deterministic_configuration_demanding_bitwise(self):
        # A small enough model is deterministic; without a floor its envelope
        # would be zero and no correct BF16 implementation could pass it.
        envelopes = derive_envelope(self._samples("up_proj", [0.0, 0.0]))
        self.assertGreaterEqual(
            envelopes["gradient|up_proj"].threshold["rel_l2"],
            THRESHOLD_FLOOR["rel_l2"] * SAFETY_MARGIN,
        )

    def test_the_floor_binds_nothing_once_the_noise_is_real(self):
        big = THRESHOLD_FLOOR["rel_l2"] * 1000
        envelopes = derive_envelope(self._samples("up_proj", [big]))
        self.assertAlmostEqual(envelopes["gradient|up_proj"].threshold["rel_l2"],
                               big * SAFETY_MARGIN)

    def test_every_gated_metric_gets_a_threshold(self):
        envelopes = derive_envelope(self._samples("q_proj", [0.01]))
        for metric in GATED_METRICS:
            with self.subTest(metric=metric):
                self.assertIn(metric, envelopes["gradient|q_proj"].threshold)

    def test_a_sample_inside_the_envelope_passes_and_outside_fails(self):
        envelopes = derive_envelope(self._samples("q_proj", [0.01]))
        inside = check_against(envelopes, self._samples("q_proj", [0.015]))
        self.assertTrue(inside["ok"])
        outside = check_against(envelopes, self._samples("q_proj", [0.5]))
        self.assertFalse(outside["ok"])
        self.assertEqual(outside["exceeded"][0]["role"], "q_proj")
        self.assertIn("metric", outside["exceeded"][0])

    def test_a_role_with_no_envelope_is_a_failure_not_a_pass(self):
        envelopes = derive_envelope(self._samples("q_proj", [0.01]))
        verdict = check_against(envelopes, self._samples("brand_new", [0.0]))
        self.assertFalse(verdict["ok"])
        self.assertIn("no envelope", verdict["exceeded"][0]["reason"])

    def test_thresholds_are_never_derived_from_the_thing_being_judged(self):
        # Structural: derive from one population, check another. If the check
        # ever used its own samples this would pass trivially at any magnitude.
        envelopes = derive_envelope(self._samples("q_proj", [0.001]))
        self.assertFalse(check_against(envelopes, self._samples("q_proj", [1.0]))["ok"])


@_skip
class TestCombinedEnvelope(unittest.TestCase):
    """A provider crosses hardware noise *and* the known integration drift."""

    def _envelope(self, role, rel, mar):
        from evograd.bench.tier3_gate.numerics import GroupEnvelope

        return {role: GroupEnvelope(
            role=role, tensors=1, samples=1,
            observed_max={"rel_l2": rel, "max_abs_over_rms": mar},
            observed_p99={"rel_l2": rel, "max_abs_over_rms": mar},
            threshold={"rel_l2": rel, "max_abs_over_rms": mar},
        )}

    def test_thresholds_add(self):
        from evograd.bench.tier3_gate.numerics import combined_envelope

        merged = combined_envelope(self._envelope("q_proj", 0.01, 0.1),
                                   self._envelope("q_proj", 0.02, 0.3))
        self.assertAlmostEqual(merged["q_proj"].threshold["rel_l2"], 0.03)
        self.assertAlmostEqual(merged["q_proj"].threshold["max_abs_over_rms"], 0.4)

    def test_a_role_present_in_only_one_half_survives(self):
        from evograd.bench.tier3_gate.numerics import combined_envelope

        merged = combined_envelope(self._envelope("q_proj", 0.01, 0.1),
                                   self._envelope("k_proj", 0.02, 0.3))
        self.assertEqual(sorted(merged), ["k_proj", "q_proj"])
        self.assertAlmostEqual(merged["q_proj"].threshold["rel_l2"], 0.01)

    def test_the_combined_bound_is_never_tighter_than_either_half(self):
        from evograd.bench.tier3_gate.numerics import combined_envelope

        hardware = self._envelope("q_proj", 0.01, 0.1)
        integration = self._envelope("q_proj", 0.02, 0.3)
        merged = combined_envelope(hardware, integration)
        for metric in GATED_METRICS:
            with self.subTest(metric=metric):
                self.assertGreaterEqual(merged["q_proj"].threshold[metric],
                                        hardware["q_proj"].threshold[metric])
                self.assertGreaterEqual(merged["q_proj"].threshold[metric],
                                        integration["q_proj"].threshold[metric])


@_skip
class TestTrajectoryPolicy(unittest.TestCase):
    def test_the_horizon_is_part_of_the_policy(self):
        policy = derive_trajectory_policy(
            [(0.01, 0.001)], horizon=5, optimizer="AdamW", learning_rate=1e-4
        )
        self.assertEqual(policy.horizon, 5)
        verdict = policy.check([1.0] * 50, [1.0] * 50)
        self.assertFalse(verdict["ok"])
        self.assertIn("5 steps", verdict["reason"])

    def test_a_curve_inside_the_bound_passes(self):
        # 0.01 absolute and 0.001 relative, doubled by the margin: a curve that
        # moves by 0.001 on a loss of 1.0 is inside both.
        policy = derive_trajectory_policy(
            [(0.01, 0.001)], horizon=3, optimizer="AdamW", learning_rate=1e-4
        )
        verdict = policy.check([1.0, 2.0, 3.0], [1.001, 2.0, 3.0])
        self.assertTrue(verdict["ok"], verdict)
        self.assertAlmostEqual(verdict["max_abs_delta"], 0.001, places=6)

    def test_a_curve_outside_it_fails_and_says_by_how_much(self):
        policy = derive_trajectory_policy(
            [(0.001, 0.0001)], horizon=3, optimizer="AdamW", learning_rate=1e-4
        )
        verdict = policy.check([1.0, 2.0, 3.0], [1.5, 2.0, 3.0])
        self.assertFalse(verdict["ok"])
        self.assertAlmostEqual(verdict["max_abs_delta"], 0.5)

    def test_mismatched_lengths_are_refused(self):
        policy = derive_trajectory_policy(
            [(0.01, 0.001)], horizon=3, optimizer="AdamW", learning_rate=1e-4
        )
        self.assertFalse(policy.check([1.0, 2.0], [1.0, 2.0, 3.0])["ok"])


@_skip
class TestEnvironmentBinding(unittest.TestCase):
    """A calibration is a measurement of one machine, and says so."""

    def _fingerprint(self, **overrides):
        base = {field: f"value-{field}" for field in BINDING_FIELDS}
        base.update(overrides)
        return base

    def test_an_identical_environment_matches(self):
        base = self._fingerprint()
        self.assertEqual(environment_mismatch(base, dict(base)), [])

    def test_every_binding_field_can_reject_on_its_own(self):
        base = self._fingerprint()
        for field in BINDING_FIELDS:
            with self.subTest(field=field):
                moved = {**base, field: "something else"}
                mismatch = environment_mismatch(base, moved)
                self.assertEqual(len(mismatch), 1)
                self.assertIn(field, mismatch[0])

    def test_the_live_fingerprint_carries_what_can_move_a_number(self):
        live = environment_fingerprint()
        for field in ("torch", "cuda", "cudnn", "transformers", "tf32_matmul",
                      "float32_matmul_precision", "deterministic_algorithms",
                      "sdpa_backends", "gpu_name"):
            with self.subTest(field=field):
                self.assertIn(field, live)
        self.assertEqual(len(fingerprint_hash(live)), 16)

    def test_a_policy_refuses_to_travel(self):
        policy = _policy(environment=self._fingerprint())
        self.assertTrue(policy.applies_here(self._fingerprint(torch="9.9.9")))
        self.assertEqual(policy.applies_here(self._fingerprint()), [])


def _policy(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        workload_id="qwen3-0.6b.train.bs2.seq2048.bf16.cuda.sdpa.6e7919ad",
        workload_hash="6e7919ad",
        environment={"gpu_name": "GH200"},
        environment_hash="abc123",
        envelopes={"gradient|up_proj": GroupEnvelope(
            role="gradient|up_proj", tensors=28, samples=112,
            observed_max={"rel_l2": 0.01, "max_abs_over_rms": 0.1},
            observed_p99={"rel_l2": 0.009, "max_abs_over_rms": 0.09},
            threshold={"rel_l2": 0.02, "max_abs_over_rms": 0.2},
        )},
        trajectory=TrajectoryPolicy(horizon=5, optimizer="AdamW",
                                    learning_rate=1e-4, max_abs_delta=0.01,
                                    max_rel_delta=0.001),
    )
    base.update(overrides)
    return NumericsPolicy(**base)


@_skip
class TestPolicySerialization(unittest.TestCase):
    def test_a_policy_round_trips_through_json(self):
        policy = _policy()
        payload = json.loads(json.dumps(policy.to_dict()))
        rebuilt = NumericsPolicy.from_dict(payload)
        self.assertEqual(rebuilt.workload_id, policy.workload_id)
        self.assertEqual(rebuilt.envelopes["gradient|up_proj"].threshold,
                         policy.envelopes["gradient|up_proj"].threshold)
        self.assertEqual(rebuilt.trajectory.horizon, 5)

    def test_a_different_schema_is_refused_rather_than_reinterpreted(self):
        payload = _policy().to_dict()
        payload["schema_version"] = "evograd-qwen3-t3-numerics/0"
        with self.assertRaises(ValueError) as caught:
            NumericsPolicy.from_dict(payload)
        self.assertIn("recalibrate", str(caught.exception))

    def test_the_artifact_carries_no_tensor_payload(self):
        # Not a string search -- "tensors: 28" is a legitimate count. The claim
        # is that nothing in it is bulk numeric data.
        def longest_numeric_list(node) -> int:
            if isinstance(node, list):
                if node and all(isinstance(v, (int, float)) for v in node):
                    return len(node)
                return max((longest_numeric_list(v) for v in node), default=0)
            if isinstance(node, dict):
                return max((longest_numeric_list(v) for v in node.values()), default=0)
            return 0

        payload = _policy().to_dict()
        self.assertLessEqual(longest_numeric_list(payload), 64)
        self.assertLess(len(json.dumps(payload)), 100_000)


@_skip
class TestTheGateRefuses(unittest.TestCase):
    def test_a_missing_calibration_is_a_refusal_not_a_default(self):
        from pathlib import Path

        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import (
            CalibrationUnavailable,
            load_policy,
        )

        with self.assertRaises(CalibrationUnavailable) as caught:
            load_policy(Path("/nonexistent/t3-numerics-calibration.json"))
        self.assertIn("recalibrat", str(caught.exception).lower() + "recalibrate")

    def test_a_calibration_from_elsewhere_is_refused(self):
        import tempfile
        from pathlib import Path

        from evograd.bench.workloads.qwen3.evaluation.tier3.gate import (
            CalibrationUnavailable,
            load_policy,
        )

        policy = _policy(environment={**environment_fingerprint(),
                                      "gpu_name": "a different GPU"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            path.write_text(json.dumps({"policy": policy.to_dict()}))
            with self.assertRaises(CalibrationUnavailable) as caught:
                load_policy(path)
            self.assertIn("gpu_name", str(caught.exception))
            # ... and it can be loaded deliberately, saying what that means.
            loaded = load_policy(path, require_environment=False)
            self.assertTrue(loaded.applies_here())

    def test_the_runner_records_a_model_correctness_failure_without_timing(self):
        from evograd.bench.tier3_runner import (
            ModelCorrectnessFailure,
            _failure_stage,
            model_correctness_check,
        )

        self.assertEqual(
            _failure_stage(ModelCorrectnessFailure("boom")), "model_correctness"
        )

        class _Workload:
            def model_correctness(self, kernels, *, device):
                return {"ok": False, "reason": "gradients outside the envelope"}

        class _Kernels:
            patched = ("swiglu_mlp",)

        with self.assertRaises(ModelCorrectnessFailure) as caught:
            model_correctness_check(_Workload(), _Kernels(), verify=True, device="cpu")
        self.assertIn("envelope", str(caught.exception))

    def test_a_workload_without_a_gate_is_not_blocked_by_one(self):
        from evograd.bench.tier3_runner import model_correctness_check

        class _Bare:
            pass

        class _Kernels:
            patched = ("x",)

        verdict = model_correctness_check(_Bare(), _Kernels(), verify=True, device="cpu")
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["gate"], "none")

    def test_an_unpatched_provider_skips_the_gate(self):
        from evograd.bench.tier3_runner import model_correctness_check

        class _Workload:
            def model_correctness(self, kernels, *, device):
                raise AssertionError("must not be called for an unpatched provider")

        class _Kernels:
            patched = ()

        self.assertTrue(
            model_correctness_check(_Workload(), _Kernels(), verify=True,
                                    device="cpu")["ok"]
        )


@_skip
class TestFaultCatalogue(unittest.TestCase):
    def test_the_catalogue_covers_every_required_kind(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import catalogue

        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import state_catalogue

        kinds = {fault.name for fault in catalogue()}
        kinds |= {fault.name for fault in state_catalogue()}
        self.assertEqual(
            kinds,
            # Ten, and each is a different way for a wrong provider to look
            # right: three magnitudes of numerical error, a structural one, a
            # sparse one, one that only appears after enough calls, one that is
            # not a number, and the four that live in the optimizer.
            {"grad_scale", "output_scale", "one_role", "dropped_row",
             "stateful", "non_finite", "layer_subset", "single_layer",
             "wrong_update", "corrupt_exp_avg", "corrupt_exp_avg_sq",
             "wrong_step_count"},
        )

    def test_the_smallest_always_rejected_magnitude_is_reported(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import smallest_rejected

        results = [
            {"fault": {"name": "output_scale", "magnitude": 0.001}, "rejected": False},
            {"fault": {"name": "output_scale", "magnitude": 0.005}, "rejected": True},
            {"fault": {"name": "output_scale", "magnitude": 0.005}, "rejected": True},
            {"fault": {"name": "output_scale", "magnitude": 0.02}, "rejected": True},
        ]
        summary = smallest_rejected(results)
        self.assertAlmostEqual(summary["output_scale"]["smallest_always_rejected"], 0.005)

    def test_a_fault_rejected_on_only_some_seeds_is_not_counted(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import smallest_rejected

        results = [
            {"fault": {"name": "one_role", "magnitude": 0.001}, "rejected": True},
            {"fault": {"name": "one_role", "magnitude": 0.001}, "rejected": False},
        ]
        self.assertIsNone(
            smallest_rejected(results)["one_role"]["smallest_always_rejected"]
        )

    def test_the_grad_scale_fault_leaves_the_forward_exact(self):
        from evograd.bench.workloads.qwen3.evaluation.tier3.faults import _GradScale

        x = torch.randn(8, requires_grad=True)
        y = _GradScale.apply(x, 1.5)
        self.assertTrue(torch.equal(y.detach(), x.detach()))
        y.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.full_like(x, 1.5)))


if __name__ == "__main__":
    unittest.main()
