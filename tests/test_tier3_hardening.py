"""Tier 3 hardening: correctness before timing, honest measurement, isolation.

The ported tier-3 wiring measured whatever it was handed, in a fixed order,
with the patch record written back onto the workload. These pin the four things
that changed:

* nothing is timed until it has passed the tier-1 pair gate and produced finite
  scalar losses;
* every declared output is rank-adapted and differentiated, not just the first;
* provider order is seeded, randomized and recorded, and ratios carry an
  interval;
* what a provider actually patched travels with that provider.

All CPU. ``qwen3_qkv_norm_rope`` and ``fused_add_rms_norm`` appear as
multi-output fixtures. Neither is a model patch site yet -- the point is that
the adapters are written against ``op.outputs`` rather than against the three
sites that happen to have one output each today.
"""

from __future__ import annotations

import json
import unittest

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

if HAVE_TORCH:
    from torch import nn

    from evograd.bench.tier3 import (
        LLAMA_SITES,
        LLAMA_SITE_OPS,
        KernelSet,
        KernelSource,
        ModulePatch,
        ModuleWorkload,
        NonFiniteLoss,
        PreflightFailure,
        build_with_provenance,
        by_construction_provenance,
        check_loss,
        eager_pair_for,
        identity_control_kernels,
        kernel_from_pair,
        loss_agreement,
        loss_trajectory,
        measure_one,
        patch,
        patched_kernels,
        preflight,
        provider_order,
        run_tier3,
        speedup_intervals,
        verification_policy,
    )
    from evograd.bench.tier3_runner import _bootstrap_ratio
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import OPS, get_op

#: Not patch sites. Fixtures, chosen because they are the only declarations in
#: the repository that return more than one tensor.
MULTI_OUTPUT = ("fused_add_rms_norm", "qwen3_qkv_norm_rope")


# ── a workload small enough to drive a whole run on CPU ──────────────────────


class _ToyModel(nn.Module):
    """One weight, one kernel-set call, a scalar loss."""

    def __init__(self, kernels: KernelSet, *, blow_up: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4))
        self.kernels = kernels
        self.blow_up = blow_up

    def forward(self, x):
        y = self.kernels.rms_norm(x, self.weight, 1e-5)
        loss = (y * y).mean()
        return loss * float("inf") if self.blow_up else loss


class _ToyWorkload:
    """A ``TrainingWorkload`` that needs no GPU, so the runner can be tested."""

    unit_name = "rows"
    #: It reaches for `rms_norm`, so it owns Llama's registry. Every workload
    #: must name one -- tier 3 refuses to guess which model's sites it has.
    site_registry = LLAMA_SITES

    def __init__(self, *, blow_up: bool = False, threshold: float | None = None):
        self.name = "toy"
        self.blow_up = blow_up
        self.loss_delta_threshold = threshold
        self.builds: list[tuple[str, ...]] = []

    def units_per_step(self) -> int:
        return 8

    def build(self, kernels: KernelSet) -> nn.Module:
        self.builds.append(kernels.patched)
        torch.manual_seed(0)
        return _ToyModel(kernels, blow_up=self.blow_up)

    def batch_for(self, *, seed: int):
        return torch.randn(8, 4, generator=torch.Generator().manual_seed(seed))

    def loss(self, model, batch) -> torch.Tensor:
        return model(batch)

    def describe(self) -> dict:
        return {"workload": "toy", "name": self.name}


def _toy_options(**overrides):
    options = {
        "warmup": 1, "steps": 2, "blocks": 2, "loss_steps": 3,
        "seed": 0, "device": "cpu", "ops": OPS,
    }
    options.update(overrides)
    return options


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestRankAdaptation(unittest.TestCase):
    """A model carries leading dimensions; a declaration is written for rows.

    The adapter flattens to the declared rank and restores afterwards. With one
    output that is a bare Tensor in and a bare Tensor out; with several it has
    to be decided per output, because a multi-output operator can mix a per-row
    result with a scalar and restoring the wrong one is a silent reshape.
    """

    def _control(self, name):
        op = get_op(name)
        return op, kernel_from_pair(op, eager_pair_for(op))

    def test_a_single_output_stays_a_bare_tensor(self):
        op, kernel = self._control("rmsnorm")
        self.assertFalse(op.is_multi_output)
        x = torch.randn(2, 5, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        y = kernel(x, weight, 1e-5)
        self.assertTrue(torch.is_tensor(y))
        self.assertEqual(tuple(y.shape), (2, 5, 8))

    def test_two_outputs_are_each_restored(self):
        op, kernel = self._control("fused_add_rms_norm")
        self.assertTrue(op.is_multi_output)
        x = torch.randn(2, 5, 8, requires_grad=True)
        r = torch.randn(2, 5, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        outputs = kernel(x, r, weight, 1e-6)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual([tuple(t.shape) for t in outputs], [(2, 5, 8), (2, 5, 8)])

    def test_three_outputs_with_a_rank_three_declaration(self):
        # x is declared [B, T, H]; handing it a fourth leading dimension is what
        # exercises the adapter on an operator that is already batched.
        op = get_op("qwen3_qkv_norm_rope")
        values = make_case_inputs(op, op.correctness[0], device="cpu")
        kernel = kernel_from_pair(op, eager_pair_for(op))
        x = values["x"].detach().unsqueeze(0).repeat(2, 1, 1, 1).requires_grad_(True)
        weights = [
            values[n].detach().requires_grad_(True)
            for n in ("q_weight", "k_weight", "v_weight",
                      "q_norm_weight", "k_norm_weight")
        ]
        q, k, v = kernel(x, *weights, values["cos"], values["sin"], values["eps"])
        dims = op.correctness[0].dims
        self.assertEqual(tuple(q.shape), (2, dims["B"], dims["HQ"], dims["T"], dims["D"]))
        self.assertEqual(tuple(k.shape), (2, dims["B"], dims["HK"], dims["T"], dims["D"]))
        self.assertEqual(tuple(v.shape), tuple(k.shape))

    def test_a_scalar_output_is_never_unflattened(self):
        op, kernel = self._control("fused_linear_cross_entropy")
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        weight = torch.randn(32, 16, requires_grad=True)
        target = torch.randint(0, 32, (2, 5))
        loss = kernel(hidden, weight, target)
        self.assertEqual(tuple(loss.shape), ())

    def test_the_adapter_matches_calling_the_declaration_directly(self):
        # Flatten-and-restore must be a no-op on the mathematics, or the patched
        # provider is computing something the unpatched one is not.
        op = get_op("fused_add_rms_norm")
        kernel = kernel_from_pair(op, eager_pair_for(op))
        from evograd.opdecl.oracle import resolve_runtime_forward

        reference = resolve_runtime_forward(op)
        x = torch.randn(2, 5, 8)
        r = torch.randn(2, 5, 8)
        weight = torch.randn(8)
        adapted = kernel(x, r, weight, 1e-6)
        direct = reference(x.reshape(-1, 8), r.reshape(-1, 8), weight, 1e-6)
        for got, want in zip(adapted, direct):
            self.assertTrue(torch.allclose(got.reshape(-1, 8), want, atol=1e-6))


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestGradientRouting(unittest.TestCase):
    """Every output's gradient has to reach every input it depends on."""

    def test_one_output_reaches_every_active_input(self):
        op = get_op("rmsnorm")
        kernel = kernel_from_pair(op, eager_pair_for(op))
        x = torch.randn(2, 5, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        kernel(x, weight, 1e-5).sum().backward()
        self.assertEqual(tuple(x.grad.shape), (2, 5, 8))
        self.assertEqual(tuple(weight.grad.shape), (8,))

    def test_both_outputs_contribute_to_the_input_gradients(self):
        # `summed` feeds the residual stream. A backward that dropped it would
        # produce a strictly smaller dr and no error at all.
        op = get_op("fused_add_rms_norm")
        kernel = kernel_from_pair(op, eager_pair_for(op))

        def dr_for(use_summed: bool):
            x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
            r = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(1))
            r.requires_grad_(True)
            weight = torch.ones(8, requires_grad=True)
            out, summed = kernel(x.requires_grad_(True), r, weight, 1e-6)
            total = out.sum() + (summed.sum() if use_summed else 0.0)
            total.backward()
            return r.grad.clone()

        self.assertFalse(torch.allclose(dr_for(True), dr_for(False)))

    def test_three_outputs_all_route_back(self):
        op = get_op("qwen3_qkv_norm_rope")
        values = make_case_inputs(op, op.correctness[0], device="cpu")
        kernel = kernel_from_pair(op, eager_pair_for(op))
        x = values["x"].detach().requires_grad_(True)
        names = ("q_weight", "k_weight", "v_weight", "q_norm_weight", "k_norm_weight")
        weights = [values[n].detach().requires_grad_(True) for n in names]
        q, k, v = kernel(x, *weights, values["cos"], values["sin"], values["eps"])
        (q.sum() + k.sum() + v.sum()).backward()
        self.assertIsNotNone(x.grad)
        for name, weight in zip(names, weights):
            with self.subTest(weight=name):
                self.assertIsNotNone(weight.grad)

    def test_the_control_refuses_a_mismatched_gradient_count(self):
        op = get_op("fused_add_rms_norm")
        _forward, backward = (
            eager_pair_for(op).forward_with_saved,
            eager_pair_for(op).backward_from_saved,
        )
        saved = (torch.randn(4, 8), torch.randn(4, 8), torch.ones(8), 1e-6)
        with self.assertRaises(ValueError) as caught:
            backward(torch.randn(4, 8), saved)   # one gradient, two outputs
        self.assertIn("2 outputs", str(caught.exception))


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestCorrectnessBeforeTiming(unittest.TestCase):
    """A wrong kernel at this tier does not raise. It returns a throughput."""

    def test_the_control_passes_the_tier_one_gate(self):
        kernels = identity_control_kernels(OPS, ("rms_norm", "swiglu"))
        report = preflight(kernels, OPS, device="cpu")
        self.assertEqual([c["site"] for c in report["checked"]], ["rms_norm", "swiglu"])
        self.assertTrue(all(c["ok"] for c in report["checked"]))

    def test_a_wrong_kernel_is_rejected_before_any_timing(self):
        op = get_op("rmsnorm")
        honest = eager_pair_for(op)

        class _Wrong:
            @staticmethod
            def rmsnorm_forward_with_saved(x, weight, eps=1e-5):
                y, saved = honest.forward_with_saved(x, weight, eps)
                return y * 2.0, saved       # plausible, and wrong

            rmsnorm_backward_from_saved = staticmethod(honest.backward_from_saved)

        kernels = patched_kernels({"rms_norm": _Wrong}, OPS)
        with self.assertRaises(PreflightFailure) as caught:
            preflight(kernels, OPS, device="cpu")
        self.assertIn("rms_norm", str(caught.exception))

    def test_a_provider_that_fails_preflight_is_not_timed(self):
        op = get_op("rmsnorm")
        honest = eager_pair_for(op)

        class _Wrong:
            @staticmethod
            def rmsnorm_forward_with_saved(x, weight, eps=1e-5):
                y, saved = honest.forward_with_saved(x, weight, eps)
                return y + 1.0, saved

            rmsnorm_backward_from_saved = staticmethod(honest.backward_from_saved)

        entry = measure_one(
            _ToyWorkload(), "candidate", patched_kernels({"rms_norm": _Wrong}, OPS),
            **_toy_options(),
        )
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "preflight")
        self.assertNotIn("step_ms", entry)

    def test_a_raw_callable_is_reported_as_unverifiable_not_verified(self):
        kernels = patch(KernelSet(), "rms_norm", lambda x, w, eps: x)
        report = preflight(kernels, OPS, device="cpu")
        self.assertEqual(report["checked"], [])
        self.assertEqual([u["site"] for u in report["unverifiable"]], ["rms_norm"])

    def test_the_policy_is_recorded(self):
        policy = verification_policy(_ToyWorkload())
        self.assertIn("verify_pair_provider", policy["preflight"])
        self.assertIn("finite scalar", policy["loss_scalar_and_finite"])
        self.assertIsNone(policy["loss_delta_threshold"])
        self.assertIn("diagnostic only", policy["loss_trajectory"])

    def test_a_declared_threshold_turns_the_trajectory_into_a_gate(self):
        policy = verification_policy(_ToyWorkload(threshold=0.25))
        self.assertEqual(policy["loss_delta_threshold"], 0.25)
        self.assertIn("gated", policy["loss_trajectory"])

    def test_skipping_verification_says_so(self):
        self.assertIn("skipped", verification_policy(_ToyWorkload(), verify=False)["preflight"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestFiniteLosses(unittest.TestCase):
    """A NaN loss has a wall time, and that wall time divides into tokens/s."""

    def test_nan_and_inf_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                with self.assertRaises(NonFiniteLoss):
                    check_loss(torch.tensor(bad), where="test")

    def test_a_non_scalar_loss_is_rejected(self):
        with self.assertRaises(NonFiniteLoss) as caught:
            check_loss(torch.zeros(4), where="test")
        self.assertIn("scalar", str(caught.exception))

    def test_a_non_tensor_loss_is_rejected(self):
        with self.assertRaises(NonFiniteLoss):
            check_loss(1.0, where="test")

    def test_a_finite_scalar_passes_and_comes_back_as_a_float(self):
        self.assertAlmostEqual(check_loss(torch.tensor(1.5), where="test"), 1.5)

    def test_the_trajectory_fails_on_the_step_that_diverged(self):
        workload = _ToyWorkload(blow_up=True)
        model = workload.build(KernelSet())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with self.assertRaises(NonFiniteLoss) as caught:
            loss_trajectory(workload, model, optimizer, steps=3, seed=0)
        self.assertIn("step 0", str(caught.exception))

    def test_a_diverging_provider_is_failed_rather_than_measured(self):
        entry = measure_one(
            _ToyWorkload(blow_up=True), "candidate", KernelSet(), **_toy_options()
        )
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "loss_finiteness")

    def test_a_provider_that_diverges_only_after_the_optimizer_moves_is_caught(self):
        # A single pre-check would miss this: the losses are finite until the
        # timed blocks have run, and only then does the model produce NaN.
        class _LateDiverger(_ToyWorkload):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def loss(self, model, batch):
                self.calls += 1
                value = model(batch)
                return value * float("nan") if self.calls > 8 else value

        entry = measure_one(
            _LateDiverger(), "candidate", KernelSet(), **_toy_options()
        )
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "loss_finiteness")
        self.assertIn("after the timed blocks", entry["error"])

    def test_the_policy_says_why_the_timed_blocks_are_not_read(self):
        policy = verification_policy(_ToyWorkload())
        self.assertIn("synchronization per step", policy["loss_scalar_and_finite"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestMeasurementIntegrity(unittest.TestCase):
    def test_provider_order_is_seeded_and_reproducible(self):
        names = ["eager", "eager_through_bind", "liger", "candidate"]
        self.assertEqual(provider_order(names, seed=7), provider_order(names, seed=7))
        self.assertCountEqual(provider_order(names, seed=7), names)

    def test_some_seed_reorders_the_providers(self):
        names = ["eager", "eager_through_bind", "liger", "candidate"]
        self.assertTrue(
            any(provider_order(names, seed=s) != names for s in range(8)),
            "the order is never shuffled; clock drift would be indistinguishable "
            "from a kernel result",
        )

    def test_the_report_records_the_order_that_ran(self):
        report = run_tier3(
            _ToyWorkload(), {"eager": KernelSet(), "control": KernelSet()},
            **_toy_options(),
        )
        self.assertCountEqual(report["provider_order"], ["eager", "control"])
        self.assertEqual(list(report["providers"]), report["provider_order"])

    def test_every_provider_gets_the_same_settings(self):
        report = run_tier3(
            _ToyWorkload(), {"eager": KernelSet(), "control": KernelSet()},
            **_toy_options(),
        )
        protocol = report["timing_protocol"]
        self.assertEqual(protocol["warmup_steps"], 1)
        self.assertEqual(protocol["timed_steps_per_block"], 2)
        self.assertEqual(protocol["blocks"], 2)
        self.assertEqual(protocol["loss_trajectory_steps"], 3)
        self.assertEqual(protocol["optimizer"], "AdamW")
        self.assertEqual(protocol["optimizer_config"]["learning_rate"], 1e-4)
        self.assertEqual(protocol["optimizer_config"]["betas"], [0.9, 0.999])
        self.assertEqual(report["seed"], 0)
        for entry in report["providers"].values():
            self.assertEqual(entry["optimizer"]["name"], "AdamW")

    def test_l2_is_not_flushed_inside_a_step(self):
        report = run_tier3(_ToyWorkload(), {"eager": KernelSet()}, **_toy_options())
        self.assertIn("never flushed", report["timing_protocol"]["l2_policy"])

    def test_cpu_bound_fraction_is_described_conservatively(self):
        report = run_tier3(_ToyWorkload(), {"eager": KernelSet()}, **_toy_options())
        described = report["timing_protocol"]["cpu_bound_fraction"]
        self.assertIn("submission-and-blocking", described)
        self.assertIn("Not a dispatch-only measurement", described)

    def test_block_samples_and_latency_statistics_are_reported(self):
        report = run_tier3(_ToyWorkload(), {"eager": KernelSet()}, **_toy_options())
        entry = report["providers"]["eager"]
        self.assertEqual(len(entry["per_block_ms"]), 2)
        self.assertEqual(entry["latency"]["count"], 2)
        for key in ("median_ms", "min_ms", "q20_ms", "q80_ms"):
            self.assertIn(key, entry["latency"])

    def test_a_ratio_carries_a_bootstrap_interval(self):
        report = run_tier3(
            _ToyWorkload(), {"eager": KernelSet(), "control": KernelSet()},
            **_toy_options(),
        )
        intervals = report["speedup_intervals"]
        self.assertTrue(intervals["available"])
        entry = intervals["vs_reference"]["control"]
        self.assertGreater(entry["ci95"]["iterations"], 0)
        self.assertLessEqual(entry["ci95"]["low"], entry["ci95"]["high"])

    def test_the_interval_serializes(self):
        report = run_tier3(
            _ToyWorkload(), {"eager": KernelSet(), "control": KernelSet()},
            **_toy_options(),
        )
        decoded = json.loads(json.dumps(report, sort_keys=True))
        self.assertIn("ci95", decoded["speedup_intervals"]["vs_reference"]["control"])

    def test_identical_blocks_bracket_one(self):
        interval = _bootstrap_ratio([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], seed=0)
        self.assertAlmostEqual(interval["low"], 1.0)
        self.assertAlmostEqual(interval["high"], 1.0)

    def test_a_failed_reference_leaves_no_intervals(self):
        report = {"providers": {"eager": {"ok": False, "error": "boom"}}}
        self.assertFalse(speedup_intervals(report)["available"])


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestLossAgreementPolicy(unittest.TestCase):
    def test_deltas_stay_visible(self):
        report = {
            "providers": {
                "eager": {"ok": True, "losses": [1.0, 0.9]},
                "candidate": {"ok": True, "losses": [1.0, 0.6]},
            }
        }
        agreement = loss_agreement(report)
        self.assertAlmostEqual(agreement["max_abs_delta"]["candidate"], 0.3)
        self.assertFalse(agreement["gated"])
        self.assertIsNone(agreement["threshold"])

    def test_a_declared_threshold_is_applied(self):
        report = {
            "providers": {
                "eager": {"ok": True, "losses": [1.0]},
                "candidate": {"ok": True, "losses": [1.4]},
            },
            "verification_policy": {"loss_delta_threshold": 0.1},
        }
        agreement = loss_agreement(report)
        self.assertTrue(agreement["gated"])
        self.assertFalse(agreement["within_threshold"]["candidate"])

    def test_the_runner_carries_the_workloads_threshold_into_the_report(self):
        report = run_tier3(
            _ToyWorkload(threshold=0.5), {"eager": KernelSet()}, **_toy_options()
        )
        self.assertEqual(report["verification_policy"]["loss_delta_threshold"], 0.5)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestPatchProvenance(unittest.TestCase):
    """What a provider replaced belongs to that provider, not to the workload."""

    def _module_workload(self, *, matcher=None):
        class Leaf(nn.Module):
            def __init__(self, tag):
                super().__init__()
                self.tag = tag
                self.weight = nn.Parameter(torch.ones(1))

        def factory():
            root = nn.Module()
            root.a = Leaf("norm")
            root.b = Leaf("norm")
            root.c = Leaf("other")
            return root

        return ModuleWorkload(
            name="toy",
            factory=factory,
            make_batch=lambda seed: seed,
            compute_loss=lambda model, batch: torch.tensor(0.0),
            units=1,
            patches=(
                ModulePatch(
                    "rms_norm",
                    matcher or (lambda m: getattr(m, "tag", None) == "norm"),
                    lambda original, kernel: Leaf("patched"),
                ),
            ),
        )

    def test_each_provider_reports_its_own_sites(self):
        workload = self._module_workload()
        eager = build_with_provenance(workload, KernelSet())[1]
        patched = build_with_provenance(
            workload, patch(KernelSet(), "rms_norm", lambda *a: None)
        )[1]
        self.assertEqual(eager.actual_sites, ())
        self.assertEqual(patched.paths, {"rms_norm": ("a", "b")})
        # Building the patched provider must not retroactively describe the
        # eager one, which is what workload-wide state did.
        self.assertEqual(eager.actual_sites, ())

    def test_the_counts_reach_the_report(self):
        workload = self._module_workload()
        _model, provenance = build_with_provenance(
            workload, patch(KernelSet(), "rms_norm", lambda *a: None)
        )
        encoded = provenance.to_dict()
        self.assertEqual(encoded["method"], "module_surgery")
        self.assertEqual(encoded["requested_sites"], ["rms_norm"])
        self.assertEqual(encoded["counts"], {"rms_norm": 2})
        self.assertEqual(encoded["modules_replaced"], 2)

    def test_a_site_that_matches_nothing_fails_the_build(self):
        workload = self._module_workload(matcher=lambda m: False)
        with self.assertRaises(ValueError):
            build_with_provenance(
                workload, patch(KernelSet(), "rms_norm", lambda *a: None)
            )

    def test_a_by_construction_workload_reports_the_kernel_sets_sites(self):
        kernels = patch(KernelSet(), "swiglu", lambda g, u: None)
        provenance = by_construction_provenance(kernels)
        self.assertEqual(provenance.method, "by_construction")
        self.assertEqual(provenance.actual_sites, ("swiglu",))
        self.assertEqual(provenance.paths, {})

    def test_each_measured_provider_carries_its_provenance(self):
        report = run_tier3(
            _ToyWorkload(),
            {
                "eager": KernelSet(),
                "control": identity_control_kernels(OPS, ("rms_norm",)),
            },
            **_toy_options(),
        )
        providers = report["providers"]
        self.assertEqual(providers["eager"]["patch_provenance"]["actual_sites"], [])
        self.assertEqual(
            providers["control"]["patch_provenance"]["actual_sites"], ["rms_norm"]
        )
        self.assertEqual(
            [s["origin"] for s in providers["control"]["kernel_sources"]],
            ["identity_control"],
        )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestIsolation(unittest.TestCase):
    """A provider that hangs or dies must cost its own row and nothing else."""

    def test_a_timeout_kills_the_child_and_records_it(self):
        import subprocess
        from unittest import mock

        from evograd.bench.tier3_cli import _run_isolated

        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="tier3", timeout=5),
        ):
            entry = _run_isolated([], "candidate", timeout=5)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["provider"], "candidate")
        self.assertEqual(entry["failed_at"], "timeout")
        self.assertIn("5s", entry["error"])

    def test_a_child_that_dies_is_recorded_with_its_stderr(self):
        from evograd.bench.tier3_cli import _run_isolated

        # An unparsable flag makes argparse exit(2) before writing a result.
        entry = _run_isolated(["--not-a-flag"], "eager", timeout=120)
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["failed_at"], "subprocess")
        self.assertIn("without a result", entry["error"])
        self.assertIn("stderr_tail", entry)

    def test_a_failing_provider_does_not_remove_the_others(self):
        report = run_tier3(
            _ToyWorkload(),
            {"eager": KernelSet(), "broken": patch(KernelSet(), "rms_norm", None)},
            **_toy_options(),
        )
        self.assertTrue(report["providers"]["eager"]["ok"])
        self.assertFalse(report["providers"]["broken"]["ok"])
        self.assertEqual(len(report["providers"]), 2)

    def test_the_report_states_how_providers_were_executed(self):
        report = run_tier3(_ToyWorkload(), {"eager": KernelSet()}, **_toy_options())
        self.assertIn("in-process", report["isolation"])

    def test_the_cli_defaults_to_isolation_with_a_budget(self):
        from evograd.bench.tier3_cli import _parser

        args = _parser().parse_args([])
        self.assertFalse(args.no_isolate)
        self.assertGreater(args.timeout, 0)

    def test_the_module_workload_documents_that_it_cannot_be_isolated(self):
        self.assertIn("not picklable", ModuleWorkload.__doc__)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestIdentityControlIsAnUpperBound(unittest.TestCase):
    """Its backward recomputes the forward; no candidate pays that."""

    def test_the_control_says_upper_bound_not_harness_tax(self):
        from evograd.bench import tier3_patch

        text = tier3_patch.eager_pair_for.__doc__
        self.assertIn("upper bound", text)
        self.assertNotIn("Whatever that costs is the harness tax", text)

    def test_the_cli_help_says_upper_bound(self):
        from evograd.bench.tier3_cli import _parser

        help_text = _parser().format_help()
        self.assertIn("UPPER BOUND", help_text)

    def test_the_control_recomputes_the_forward_in_its_backward(self):
        # This is the whole reason the control is a ceiling rather than a
        # measurement: it evaluates the reference twice per call, and no
        # candidate does. Counted rather than asserted from the source.
        from unittest import mock

        from evograd.opdecl.oracle import resolve_runtime_forward

        op = get_op("rmsnorm")
        reference = resolve_runtime_forward(op)
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return reference(*args, **kwargs)

        with mock.patch(
            "evograd.bench.tier3_patch.resolve_runtime_forward",
            return_value=counting,
        ):
            control = eager_pair_for(op)

        x = torch.randn(4, 8, requires_grad=True)
        weight = torch.ones(8, requires_grad=True)
        y, saved = control.forward_with_saved(x, weight, 1e-5)
        self.assertFalse(y.requires_grad)   # forward ran under no_grad
        self.assertEqual(len(calls), 1)
        control.backward_from_saved(torch.ones_like(y), saved)
        self.assertEqual(len(calls), 2)     # the backward ran it again

    def test_the_controls_gradients_still_match_autograd(self):
        # A ceiling on cost, not on correctness: the control must produce the
        # same gradients eager does, or the provider it defines is meaningless.
        op = get_op("rmsnorm")
        control = eager_pair_for(op)
        from evograd.opdecl.oracle import resolve_runtime_forward

        reference = resolve_runtime_forward(op)
        x = torch.randn(4, 8, dtype=torch.float64)
        weight = torch.randn(8, dtype=torch.float64)
        dy = torch.randn(4, 8, dtype=torch.float64)

        leaves = [x.clone().requires_grad_(True), weight.clone().requires_grad_(True)]
        expected = torch.autograd.grad(reference(*leaves, 1e-5), leaves, dy)

        _y, saved = control.forward_with_saved(x, weight, 1e-5)
        got = control.backward_from_saved(dy, saved)
        for name, a, b in zip(op.grad_names(), got, expected):
            with self.subTest(grad=name):
                self.assertTrue(torch.allclose(a, b))


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestMultiOutputFixturesAreNotPatchSites(unittest.TestCase):
    """They exercise the adapters; they are not part of the model surface yet."""

    def test_no_multi_output_operator_is_a_patch_site(self):
        for op_name in LLAMA_SITE_OPS.values():
            with self.subTest(op=op_name):
                self.assertFalse(get_op(op_name).is_multi_output)

    def test_the_fixtures_really_do_have_several_outputs(self):
        for name in MULTI_OUTPUT:
            with self.subTest(op=name):
                self.assertTrue(get_op(name).is_multi_output)

    def test_a_multi_output_kernel_set_can_still_be_described(self):
        # `KernelSource` does not care how many outputs a site's op has, so a
        # future site is a registry entry rather than an adapter change.
        source = KernelSource(
            site="rms_norm", op_name="fused_add_rms_norm", module=object()
        )
        self.assertTrue(source.verifiable)
        self.assertEqual(source.to_dict()["op"], "fused_add_rms_norm")


if __name__ == "__main__":
    unittest.main()
