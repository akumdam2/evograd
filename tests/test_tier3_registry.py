"""Tier 3's patch sites belong to a workload, not to the module.

There was one module-level ``SITE_OPS`` dict, and it was correct exactly as long
as there was one model. With two it is wrong in both directions: one model's
identity control would patch a site belonging to another architecture, and a
candidate would be accepted for ``rms_norm`` because that name happens to exist
somewhere. These pin the registry that replaced it -- two independent ones side
by side, and every tier-3 operation reading the one the workload owns.

The registries here are *fixtures* (``tests/_registry_fixture.py``). A real one
belongs to a workload package; the patcher ships none, which is the property
these tests exist to keep.
"""

from __future__ import annotations

import unittest

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

if HAVE_TORCH:
    from torch import nn

    from evograd.evaluation.tier3 import (
        KernelSet,
        KernelSource,
        ModulePatch,
        ModuleWorkload,
        Site,
        SiteRegistry,
        eager_pair_for,
        identity_control_kernels,
        kernel_from_pair,
        measure_one,
        patch,
        patched_kernels,
        restrict,
        run_tier3,
        site_registry_for,
    )
    from evograd.benchmark import TASKS, get_task

    from tests._registry_fixture import (
        SAMPLE_SITE_OPS,
        SAMPLE_SITES,
    )


def _qwen_like_registry() -> "SiteRegistry":
    """A second registry, deliberately sharing no site name with the first.

    Its sites are the Qwen3 Level-2 boundaries, built here rather than imported
    so this file exercises the machinery without depending on a workload.
    """
    return SiteRegistry(
        name="qwen3_0_6b",
        sites=(
            Site("qkv_norm_rope", "qwen3_qkv_norm_rope", lambda *a, **k: None),
            Site("attention", "qwen3_attention", lambda *a, **k: None),
            Site("swiglu_mlp", "qwen3_swiglu_mlp", lambda *a, **k: None),
            Site("residual_rmsnorm", "fused_add_rms_norm", lambda *a, **k: None),
        ),
    )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestTwoIndependentRegistries(unittest.TestCase):
    def test_the_two_share_no_site_names(self):
        qwen = _qwen_like_registry()
        self.assertEqual(set(SAMPLE_SITES.names) & set(qwen.names), set())

    def test_a_site_of_one_is_unknown_to_the_other(self):
        qwen = _qwen_like_registry()
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(registry=SAMPLE_SITES), "swiglu_mlp", lambda *a: None)
        # The error has to name the workload, or a two-model repository cannot
        # tell "wrong site" from "wrong model".
        self.assertIn("sample_decoder", str(caught.exception))
        self.assertIn("swiglu_mlp", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(registry=qwen), "rms_norm", lambda *a: None)
        self.assertIn("qwen3_0_6b", str(caught.exception))

    def test_each_registry_maps_its_sites_to_declared_operators(self):
        for registry in (SAMPLE_SITES, _qwen_like_registry()):
            with self.subTest(registry=registry.name):
                for site, op_name in registry.site_ops.items():
                    self.assertIn(op_name, TASKS, f"{registry.name}.{site}")

    def test_the_site_ops_alias_matches_the_registry(self):
        self.assertEqual(SAMPLE_SITE_OPS, SAMPLE_SITES.site_ops)
        self.assertEqual(
            SAMPLE_SITE_OPS,
            {
                "rms_norm": "rmsnorm",
                "swiglu": "swiglu",
                "cross_entropy": "cross_entropy",
            },
        )

    def test_duplicate_site_names_are_rejected(self):
        with self.assertRaises(ValueError):
            SiteRegistry(
                name="broken",
                sites=(Site("a", "rmsnorm", print), Site("a", "swiglu", print)),
            )

    def test_a_kernel_set_carries_its_registry_through_every_operation(self):
        qwen = _qwen_like_registry()
        kernels = patch(KernelSet(registry=qwen), "attention", lambda *a: None)
        self.assertIs(kernels.registry, qwen)
        self.assertIs(restrict(kernels, ("attention",)).registry, qwen)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestIdentityControlStaysInItsRegistry(unittest.TestCase):
    def test_a_control_claims_only_its_own_registrys_sites(self):
        kernels = identity_control_kernels(TASKS, registry=SAMPLE_SITES)
        self.assertEqual(set(kernels.patched), set(SAMPLE_SITES.names))
        self.assertNotIn("swiglu_mlp", kernels.patched)

    def test_a_second_registry_gets_its_own_control(self):
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(TASKS, registry=qwen)
        self.assertEqual(set(kernels.patched), set(qwen.names))
        self.assertEqual(
            {s.op_name for s in kernels.sources},
            set(qwen.site_ops.values()),
        )

    def test_the_control_can_be_restricted_within_its_registry(self):
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(TASKS, ("swiglu_mlp",), registry=qwen)
        self.assertEqual(kernels.patched, ("swiglu_mlp",))

    def test_a_multi_output_site_gets_a_working_control(self):
        # qwen3_qkv_norm_rope returns three tensors. The control is built the
        # same way whatever the arity, which is what makes a future site a
        # registry entry rather than an adapter change.
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(TASKS, ("qkv_norm_rope",), registry=qwen)
        self.assertTrue(callable(kernels.kernel_for("qkv_norm_rope")))
        self.assertTrue(get_task("qwen3_qkv_norm_rope").is_multi_output)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestBaselineAndCandidateSelection(unittest.TestCase):
    def test_candidates_are_bound_against_the_named_registry(self):
        qwen = _qwen_like_registry()
        op = get_task("qwen3_swiglu_mlp")
        kernels = patched_kernels(
            {"swiglu_mlp": eager_pair_for(op)}, TASKS, registry=qwen
        )
        self.assertEqual(kernels.patched, ("swiglu_mlp",))
        self.assertEqual(kernels.source_for("swiglu_mlp").op_name, "qwen3_swiglu_mlp")

    def test_a_candidate_for_another_models_site_is_refused(self):
        with self.assertRaises(ValueError):
            patched_kernels(
                {"swiglu_mlp": object()}, TASKS, registry=SAMPLE_SITES
            )

@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestWorkloadOwnsItsRegistry(unittest.TestCase):
    def test_a_workload_declares_its_own_registry(self):
        """The live example. The registry a workload is measured against comes
        from the workload, never from a module-level default."""
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload

        workload = Qwen3Workload.from_config({"device": "cpu"})
        self.assertIs(workload.site_registry, qwen3_sites())

    def test_the_patcher_ships_no_registry_of_its_own(self):
        """The property the deleted built-in violated: a default registry means
        a kernel set for one model silently claims another's sites."""
        from evograd.evaluation.tier3.patch import NO_SITES

        self.assertEqual(NO_SITES.sites, ())
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(), "rms_norm", lambda *a: None)
        self.assertIn("unspecified", str(caught.exception))

    def test_a_workload_without_one_is_refused_rather_than_guessed(self):
        class _Bare:
            name = "bare"

        with self.assertRaises(ValueError) as caught:
            site_registry_for(_Bare())
        self.assertIn("site_registry", str(caught.exception))

    def test_a_module_workload_derives_a_namespace_from_its_patches(self):
        workload = ModuleWorkload(
            name="hf-thing",
            factory=nn.Module,
            make_batch=lambda seed: seed,
            compute_loss=lambda m, b: torch.tensor(0.0),
            units=1,
            patches=(ModulePatch("norm", lambda m: False, lambda o, k: o),),
        )
        registry = site_registry_for(workload)
        self.assertEqual(registry.name, "hf-thing")
        self.assertEqual(registry.names, ("norm",))
        self.assertNotIn("rms_norm", registry)

    def test_an_explicit_registry_wins(self):
        qwen = _qwen_like_registry()
        workload = ModuleWorkload(
            name="explicit",
            factory=nn.Module,
            make_batch=lambda seed: seed,
            compute_loss=lambda m, b: torch.tensor(0.0),
            units=1,
            registry=qwen,
        )
        self.assertIs(site_registry_for(workload), qwen)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestProvenanceAndReporting(unittest.TestCase):
    class _Toy:
        """A tiny by-construction workload with a registry of its own."""

        unit_name = "rows"

        def __init__(self, registry):
            self.name = "toy"
            self.site_registry = registry

        def units_per_step(self):
            return 4

        def build(self, kernels):
            registry = kernels.registry

            class _Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(4))

                def forward(self, x):
                    return (kernels.kernel_for("rms_norm")(x, self.weight, 1e-5) ** 2).mean()

            torch.manual_seed(0)
            self.registry_seen = registry
            return _Model()

        def batch_for(self, *, seed: int):
            return torch.randn(4, 4, generator=torch.Generator().manual_seed(seed))

        def loss(self, model, batch):
            return model(batch)

        def describe(self):
            return {"workload": "toy", "name": self.name}

    def test_the_report_serializes_the_exact_site_to_operator_mapping(self):
        from tests._registry_fixture import fused_rms_norm as _rms_norm_fused

        registry = SiteRegistry(
            name="toy_family",
            sites=(Site("rms_norm", "rmsnorm", _rms_norm_fused),),
        )
        workload = self._Toy(registry)
        report = run_tier3(
            workload,
            {"eager": KernelSet(registry=registry)},
            warmup=1, steps=2, blocks=2, loss_steps=2, seed=0,
            device="cpu",
        )
        self.assertEqual(
            report["site_registry"],
            {
                "workload_family": "toy_family",
                "site_ops": {"rms_norm": "rmsnorm"},
                "preflight_workloads": {"rms_norm": 0},
            },
        )

    def test_provenance_still_travels_per_provider(self):
        from tests._registry_fixture import fused_rms_norm as _rms_norm_fused

        registry = SiteRegistry(
            name="toy_family",
            sites=(Site("rms_norm", "rmsnorm", _rms_norm_fused),),
        )
        workload = self._Toy(registry)
        control = identity_control_kernels(TASKS, ("rms_norm",), registry=registry)
        report = run_tier3(
            workload,
            {"eager": KernelSet(registry=registry), "control": control},
            warmup=1, steps=2, blocks=2, loss_steps=2, seed=0,
            device="cpu",
        )
        providers = report["providers"]
        self.assertEqual(providers["eager"]["patch_provenance"]["actual_sites"], [])
        self.assertEqual(
            providers["control"]["patch_provenance"]["actual_sites"], ["rms_norm"]
        )

    def test_a_kernel_source_from_another_registry_is_refused(self):
        with self.assertRaises(ValueError):
            patch(
                KernelSet(registry=SAMPLE_SITES),
                "rms_norm",
                lambda *a: None,
                source=KernelSource(site="swiglu"),
            )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestSampleRegistryBehaviourUnchanged(unittest.TestCase):
    """The three sample sites and their defaults."""

    def test_the_defaults_are_still_the_declared_spellings(self):
        from tests import _registry_fixture

        kernels = KernelSet(registry=SAMPLE_SITES)
        self.assertIs(kernels.rms_norm, _registry_fixture.fused_rms_norm)
        self.assertIs(kernels.swiglu, _registry_fixture.float32_swiglu)

    def test_attribute_access_still_reaches_a_patched_site(self):
        kernels = patch(KernelSet(registry=SAMPLE_SITES), "swiglu", lambda a, b: "patched")
        self.assertEqual(kernels.swiglu(1, 2), "patched")
        self.assertIs(kernels.rms_norm, KernelSet(registry=SAMPLE_SITES).rms_norm)

    def test_an_unknown_attribute_is_still_an_attribute_error(self):
        with self.assertRaises(AttributeError):
            KernelSet(registry=SAMPLE_SITES).not_a_site


if __name__ == "__main__":
    unittest.main()
