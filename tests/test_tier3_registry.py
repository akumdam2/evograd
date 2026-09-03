"""Tier 3's patch sites belong to a workload, not to the module.

There was one module-level ``SITE_OPS`` dict, and it was correct exactly as long
as there was one model. With two it is wrong in both directions: Llama's
identity control would patch a site belonging to another architecture, and a
candidate would be accepted for ``rms_norm`` because that name happens to exist
somewhere. These pin the registry that replaced it -- two independent ones side
by side, and every tier-3 operation reading the one the workload owns.

Also here: the hook by which a workload adds its own model-derived shapes to a
site's preflight. That is what decides whether an operator's *observed*
configuration can block tier-3 timing, rather than only its small declared grid.
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

    from evograd.bench.tier3 import (
        LLAMA_SITE_OPS,
        LLAMA_SITES,
        KernelSet,
        KernelSource,
        ModulePatch,
        ModuleWorkload,
        PreflightFailure,
        Site,
        SiteRegistry,
        eager_pair_for,
        identity_control_kernels,
        kernel_from_pair,
        measure_one,
        patch,
        patched_kernels,
        preflight,
        restrict,
        run_tier3,
        site_registry_for,
    )
    from evograd.opdecl.activity import Workload
    from evograd.ops import OPS, get_op


def _qwen_like_registry() -> "SiteRegistry":
    """A second registry, deliberately sharing no site name with Llama's.

    Its sites are the Qwen Level-2 boundaries. It is a *fixture*: no Qwen
    workload exists yet, and the point is that a registry is all it takes to
    describe one.
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
        self.assertEqual(set(LLAMA_SITES.names) & set(qwen.names), set())

    def test_a_site_of_one_is_unknown_to_the_other(self):
        qwen = _qwen_like_registry()
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(registry=LLAMA_SITES), "swiglu_mlp", lambda *a: None)
        # The error has to name the workload, or a two-model repository cannot
        # tell "wrong site" from "wrong model".
        self.assertIn("llama_3", str(caught.exception))
        self.assertIn("swiglu_mlp", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            patch(KernelSet(registry=qwen), "rms_norm", lambda *a: None)
        self.assertIn("qwen3_0_6b", str(caught.exception))

    def test_each_registry_maps_its_sites_to_declared_operators(self):
        for registry in (LLAMA_SITES, _qwen_like_registry()):
            with self.subTest(registry=registry.name):
                for site, op_name in registry.site_ops.items():
                    self.assertIn(op_name, OPS, f"{registry.name}.{site}")

    def test_the_llama_alias_is_the_llama_registry(self):
        self.assertEqual(LLAMA_SITE_OPS, LLAMA_SITES.site_ops)
        self.assertEqual(
            LLAMA_SITE_OPS,
            {
                "rms_norm": "rmsnorm",
                "swiglu": "swiglu",
                "cross_entropy": "fused_linear_cross_entropy",
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
    def test_llama_control_claims_only_llama_sites(self):
        kernels = identity_control_kernels(OPS, registry=LLAMA_SITES)
        self.assertEqual(set(kernels.patched), set(LLAMA_SITES.names))
        self.assertNotIn("swiglu_mlp", kernels.patched)

    def test_a_second_registry_gets_its_own_control(self):
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(OPS, registry=qwen)
        self.assertEqual(set(kernels.patched), set(qwen.names))
        self.assertEqual(
            {s.op_name for s in kernels.sources},
            set(qwen.site_ops.values()),
        )

    def test_the_control_can_be_restricted_within_its_registry(self):
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(OPS, ("swiglu_mlp",), registry=qwen)
        self.assertEqual(kernels.patched, ("swiglu_mlp",))

    def test_a_multi_output_site_gets_a_working_control(self):
        # qwen3_qkv_norm_rope returns three tensors. The control is built the
        # same way whatever the arity, which is what makes a future site a
        # registry entry rather than an adapter change.
        qwen = _qwen_like_registry()
        kernels = identity_control_kernels(OPS, ("qkv_norm_rope",), registry=qwen)
        self.assertTrue(callable(kernels.kernel_for("qkv_norm_rope")))
        self.assertTrue(get_op("qwen3_qkv_norm_rope").is_multi_output)


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestBaselineAndCandidateSelection(unittest.TestCase):
    def test_candidates_are_bound_against_the_named_registry(self):
        qwen = _qwen_like_registry()
        op = get_op("qwen3_swiglu_mlp")
        kernels = patched_kernels(
            {"swiglu_mlp": eager_pair_for(op)}, OPS, registry=qwen
        )
        self.assertEqual(kernels.patched, ("swiglu_mlp",))
        self.assertEqual(kernels.source_for("swiglu_mlp").op_name, "qwen3_swiglu_mlp")

    def test_a_candidate_for_another_models_site_is_refused(self):
        with self.assertRaises(ValueError):
            patched_kernels(
                {"swiglu_mlp": object()}, OPS, registry=LLAMA_SITES
            )

    def test_baseline_discovery_walks_the_workloads_registry(self):
        from evograd.bench.tier3_cli import _baseline_kernels

        kernels, covered = _baseline_kernels("liger", OPS, LLAMA_SITES)
        self.assertIsNotNone(kernels)
        self.assertTrue(set(covered) <= set(LLAMA_SITES.names))

    def test_a_registry_whose_ops_have_no_such_baseline_finds_nothing(self):
        from evograd.bench.tier3_cli import _baseline_kernels

        registry = SiteRegistry(
            name="toy", sites=(Site("only", "qwen3_attention", print),)
        )
        kernels, covered = _baseline_kernels("liger", OPS, registry)
        self.assertIsNone(kernels)
        self.assertEqual(covered, [])


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestWorkloadOwnsItsRegistry(unittest.TestCase):
    def test_the_llama_workload_declares_llama_sites(self):
        from evograd.bench.tier3_llama import LlamaWorkload

        self.assertIs(LlamaWorkload.site_registry, LLAMA_SITES)

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
class TestWorkloadSuppliedPreflightShapes(unittest.TestCase):
    """A site can demand shapes the operator's own grid does not contain.

    This is what decides whether an *observed* configuration blocks timing. The
    declared grid for `rmsnorm` tops out at 128 rows; a model presents 4096, and
    a kernel can be right at one and wrong at the other.
    """

    def _wrong_beyond(self, op, threshold: int):
        """A pair that is exact below ``threshold`` rows and wrong at or above it."""
        honest = eager_pair_for(op)

        class _Sneaky:
            @staticmethod
            def rmsnorm_forward_with_saved(x, weight, eps=1e-5):
                y, saved = honest.forward_with_saved(x, weight, eps)
                if x.shape[0] >= threshold:
                    y = y * 1.5
                return y, saved

            rmsnorm_backward_from_saved = staticmethod(honest.backward_from_saved)

        return _Sneaky

    def _registry(self, extra):
        from evograd.bench.tier3_patch import _rms_norm_fused

        return SiteRegistry(
            name="fixture",
            sites=(Site("rms_norm", "rmsnorm", _rms_norm_fused, preflight=extra),),
        )

    def test_the_default_grid_alone_lets_the_sneaky_pair_through(self):
        op = get_op("rmsnorm")
        biggest = max(w.dims["rows"] for w in op.correctness)
        kernels = patched_kernels(
            {"rms_norm": self._wrong_beyond(op, biggest + 1)},
            OPS,
            registry=self._registry(()),
        )
        report = preflight(kernels, OPS, device="cpu")
        self.assertTrue(all(c["ok"] for c in report["checked"]))
        self.assertEqual(report["checked"][0]["workload_supplied_cases"], 0)

    def test_a_workload_supplied_shape_catches_it(self):
        op = get_op("rmsnorm")
        biggest = max(w.dims["rows"] for w in op.correctness)
        observed = Workload(
            dims={"rows": biggest * 4, "hidden": 64}, dtype="float32"
        )
        kernels = patched_kernels(
            {"rms_norm": self._wrong_beyond(op, biggest + 1)},
            OPS,
            registry=self._registry((observed,)),
        )
        with self.assertRaises(PreflightFailure) as caught:
            preflight(kernels, OPS, device="cpu")
        self.assertIn("rms_norm", str(caught.exception))
        self.assertIn("fixture", str(caught.exception))

    def test_the_supplied_shape_is_really_run_and_reported(self):
        op = get_op("rmsnorm")
        observed = Workload(dims={"rows": 512, "hidden": 64}, dtype="float32")
        kernels = patched_kernels(
            {"rms_norm": eager_pair_for(op)}, OPS, registry=self._registry((observed,))
        )
        report = preflight(kernels, OPS, device="cpu")
        entry = report["checked"][0]
        self.assertEqual(entry["declared_cases"], len(op.correctness))
        self.assertEqual(entry["workload_supplied_cases"], 1)
        self.assertEqual(entry["cases"], len(op.correctness) + 1)
        supplied = [
            c for c in entry["checked_configs"] if c["source"] == "workload_supplied"
        ]
        self.assertEqual(len(supplied), 1)
        self.assertEqual(supplied[0]["dims"], {"rows": 512, "hidden": 64})
        self.assertIn("rows=512", supplied[0]["id"])

    def test_llama_supplies_none_so_its_behaviour_is_unchanged(self):
        for site in LLAMA_SITES.sites:
            with self.subTest(site=site.name):
                self.assertEqual(site.preflight, ())

    def test_the_report_names_the_family_and_the_mapping(self):
        kernels = identity_control_kernels(OPS, ("rms_norm",), registry=LLAMA_SITES)
        report = preflight(kernels, OPS, device="cpu")
        self.assertEqual(report["workload_family"], "llama_3")
        self.assertEqual(report["site_ops"], LLAMA_SITE_OPS)


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
        from evograd.bench.tier3_patch import _rms_norm_fused

        registry = SiteRegistry(
            name="toy_family",
            sites=(Site("rms_norm", "rmsnorm", _rms_norm_fused),),
        )
        workload = self._Toy(registry)
        report = run_tier3(
            workload,
            {"eager": KernelSet(registry=registry)},
            warmup=1, steps=2, blocks=2, loss_steps=2, seed=0,
            device="cpu", ops=OPS,
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
        from evograd.bench.tier3_patch import _rms_norm_fused

        registry = SiteRegistry(
            name="toy_family",
            sites=(Site("rms_norm", "rmsnorm", _rms_norm_fused),),
        )
        workload = self._Toy(registry)
        control = identity_control_kernels(OPS, ("rms_norm",), registry=registry)
        report = run_tier3(
            workload,
            {"eager": KernelSet(registry=registry), "control": control},
            warmup=1, steps=2, blocks=2, loss_steps=2, seed=0,
            device="cpu", ops=OPS,
        )
        providers = report["providers"]
        self.assertEqual(providers["eager"]["patch_provenance"]["actual_sites"], [])
        self.assertEqual(
            providers["control"]["patch_provenance"]["actual_sites"], ["rms_norm"]
        )
        self.assertEqual(
            providers["control"]["verification"]["workload_family"], "toy_family"
        )

    def test_a_kernel_source_from_another_registry_is_refused(self):
        with self.assertRaises(ValueError):
            patch(
                KernelSet(registry=LLAMA_SITES),
                "rms_norm",
                lambda *a: None,
                source=KernelSource(site="swiglu"),
            )


@unittest.skipUnless(HAVE_TORCH, "torch not installed on this machine")
class TestLlamaBehaviourUnchanged(unittest.TestCase):
    """The three sites, their defaults, and the CLI's validation."""

    def test_the_defaults_are_still_the_declared_spellings(self):
        from evograd.ops.level3.llama3_decoder_layer import forward_ref

        kernels = KernelSet()
        self.assertIs(kernels.rms_norm, forward_ref._rms_norm_fused)
        self.assertIs(kernels.swiglu, forward_ref._default_swiglu)

    def test_attribute_access_still_reaches_a_patched_site(self):
        kernels = patch(KernelSet(), "swiglu", lambda a, b: "patched")
        self.assertEqual(kernels.swiglu(1, 2), "patched")
        self.assertIs(kernels.rms_norm, KernelSet().rms_norm)

    def test_an_unknown_attribute_is_still_an_attribute_error(self):
        with self.assertRaises(AttributeError):
            KernelSet().not_a_site

    def test_the_cli_rejects_an_unknown_site_naming_the_workload(self):
        from evograd.bench.tier3_cli import _parser, _sites

        import contextlib, io

        args = _parser().parse_args(["--sites", "swiglu_mlp"])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            _sites(args, LLAMA_SITES)
        self.assertIn("llama_3", stderr.getvalue())

    def test_the_cli_default_providers_are_unchanged(self):
        from evograd.bench.tier3_cli import _parser, build_providers

        args = _parser().parse_args(["--identity-control", "--baseline", "liger"])
        providers = build_providers(args, quiet=True)
        self.assertEqual(
            sorted(providers), ["eager", "eager_through_bind", "liger"]
        )
        for kernels in providers.values():
            self.assertIs(kernels.registry, LLAMA_SITES)


class TestTier3WorkloadRegistry(unittest.TestCase):
    """The CLI reaches a workload by name; it names no architecture itself.

    ``tier3_cli`` used to hold three ``if args.model == "qwen3_0_6b"`` branches:
    one to build the workload, one to add its structural-identity provider, and
    one to reject that flag for anything else. A third workload was therefore an
    edit in three places, and -- worse -- a flag that meant nothing for the
    selected model was silently ignored rather than refused.
    """

    def test_the_cli_model_choices_come_from_the_registry(self):
        from evograd.bench.tier3_cli import MODELS
        from evograd.bench.workloads import TIER3_ADAPTERS

        self.assertEqual(set(MODELS), set(TIER3_ADAPTERS))

    def test_the_cli_source_names_no_architecture(self):
        """The load-bearing one: evaluation must not know what a Qwen is."""
        import pathlib

        import evograd.bench.tier3_cli as cli

        source = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
        for architecture in ("qwen3", "Qwen3", "Qwen"):
            self.assertNotIn(architecture, source, architecture)

    def test_every_registered_adapter_resolves_and_is_well_formed(self):
        from evograd.bench.workloads import (
            Tier3Adapter, TIER3_ADAPTERS, tier3_adapter,
        )

        for name in TIER3_ADAPTERS:
            with self.subTest(workload=name):
                adapter = tier3_adapter(name)
                self.assertIsInstance(adapter, Tier3Adapter)
                self.assertEqual(adapter.name, name)
                self.assertTrue(callable(adapter.build))
                self.assertTrue(adapter.summary)
                if adapter.providers is not None:
                    self.assertTrue(callable(adapter.providers))

    def test_an_unknown_workload_names_the_ones_that_exist(self):
        from evograd.bench.workloads import UnknownWorkload, tier3_adapter

        with self.assertRaises(UnknownWorkload) as caught:
            tier3_adapter("gpt_9")
        self.assertIn("gpt_9", str(caught.exception))
        self.assertIn("llama_3_8b", str(caught.exception))

    def test_the_registry_imports_no_adapter_until_one_is_asked_for(self):
        """``ops`` imports this module at declaration time; an eager import of
        the adapters would drag torch and Transformers in behind it."""
        import ast
        import pathlib

        import evograd.bench.workloads as registry

        tree = ast.parse(pathlib.Path(registry.__file__).read_text(encoding="utf-8"))
        toplevel = {
            node.module.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertEqual(
            toplevel, {"__future__", "importlib", "dataclasses", "pathlib", "typing"}
        )

    def test_a_workload_only_gets_the_providers_it_declares(self):
        from evograd.bench.tier3_cli import _parser, build_providers

        llama = _parser().parse_args(["--model", "llama_3_8b_4l", "--baseline", "none"])
        self.assertEqual(sorted(build_providers(llama, quiet=True)), ["eager"])

        qwen = _parser().parse_args([
            "--model", "qwen3_0_6b", "--baseline", "none",
            "--structural-identity", "--device", "cpu", "--layers", "1",
        ])
        self.assertEqual(
            sorted(build_providers(qwen, quiet=True)),
            ["eager", "structural_identity"],
        )

    def test_a_flag_the_workload_does_not_declare_is_refused_by_name(self):
        """Previously ``--layers 2`` on Llama was accepted and ignored, which is
        the failure that reports a number for a run that did not happen."""
        import contextlib, io

        from evograd.bench.tier3_cli import _parser, check_options

        for flag, value in (("--structural-identity", None),
                            ("--layers", "2"),
                            ("--data-seed", "7")):
            with self.subTest(flag=flag):
                argv = ["--model", "llama_3_8b_4l", flag]
                if value is not None:
                    argv.append(value)
                args = _parser().parse_args(argv)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                    check_options(args)
                message = stderr.getvalue()
                self.assertIn(flag, message)
                self.assertIn("llama_3_8b_4l", message)

    def test_the_declaring_workload_accepts_those_same_flags(self):
        from evograd.bench.tier3_cli import _parser, check_options

        args = _parser().parse_args([
            "--model", "qwen3_0_6b", "--structural-identity",
            "--layers", "2", "--data-seed", "7",
        ])
        check_options(args)  # must not raise

    def test_unset_optional_flags_are_never_refused(self):
        from evograd.bench.tier3_cli import _parser, check_options

        for model in ("llama_3_8b_4l", "llama_3_8b", "qwen3_0_6b"):
            with self.subTest(model=model):
                check_options(_parser().parse_args(["--model", model]))


if __name__ == "__main__":
    unittest.main()
