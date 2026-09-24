"""The MegaFold level-4 workload.

Two halves, and the split is deliberate.

The **declaration half** always runs. It pins what the suite claims about this
task with no model in sight: that the modules import without MegaFold, that a
missing MegaFold is reported as a sentence someone can act on, that the
declaration is registered and validates, that both adapters resolve, and that
the exclusions actually state every restriction the task operates under. That
last one is the point of having exclusions at all: a level-4 number read
without them overstates what was measured.

The **model half** skips unless MegaFold can be built. It needs more than the
package: ``megafold.model.megafold`` constructs fused Triton modules at import
time and resolves a Triton driver while doing it, so on a login or CPU-only
node the import fails even though every dependency is installed. The guard
below therefore tries the real import rather than checking for the package.

What the model half pins is the claim the three providers rest on: that the
eager tree and MegaFold's own tree compute the same loss. If they do not, every
speedup measured between them is unattributable.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import unittest

import torch

from evograd.benchmark import TASKS, WORKLOADS, get_workload
from evograd.evaluation.tier3.patch import KernelSet, identity_control_kernels
from evograd.opdecl.activity import Workload
from evograd.opdecl.models import ALPHAFOLD3_2L


def _megafold_root() -> pathlib.Path | None:
    """Where the MegaFold checkout is, without hardcoding anyone's home.

    ``MEGAFOLD_ROOT`` wins; otherwise the checkout is wherever the importable
    package came from, which is the same directory ``PYTHONPATH`` names.
    """
    override = os.environ.get("MEGAFOLD_ROOT")
    if override:
        return pathlib.Path(override)
    spec = importlib.util.find_spec("megafold")
    if spec is None or not spec.origin:
        return None
    return pathlib.Path(spec.origin).resolve().parent.parent


#: MegaFold's own 1x1 training configuration, which the workload's defaults
#: must reproduce. Parsed in the tests rather than restated, so a default that
#: drifts away from the file is a failure rather than a matching pair of
#: transcriptions.
def _megafold_config() -> pathlib.Path | None:
    root = _megafold_root()
    return None if root is None else root / "configs" / "megafold_1x1.yaml"


#: Two separate conditions, because they fail separately and the tests that
#: depend on them are different tests. The *package* imports anywhere it is on
#: PYTHONPATH; the *model module* additionally builds fused Triton modules at
#: import time and resolves a Triton driver while doing it, so it needs a
#: visible GPU and fails on a login or CPU-only node with everything installed.
HAVE_MEGAFOLD_PACKAGE = importlib.util.find_spec("megafold") is not None

try:
    from megafold.model import megafold as _model_module  # noqa: F401

    HAVE_MEGAFOLD = True
    WHY_NOT = ""
except Exception as exc:  # noqa: BLE001 - a Triton driver error is not an ImportError
    HAVE_MEGAFOLD = False
    WHY_NOT = f"{type(exc).__name__}: {exc}"

needs_megafold = unittest.skipUnless(
    HAVE_MEGAFOLD, f"MegaFold model module not importable ({WHY_NOT})"
)
needs_checkout = unittest.skipIf(
    _megafold_config() is None or not _megafold_config().exists(),
    "no MegaFold checkout found (set MEGAFOLD_ROOT or put it on PYTHONPATH)",
)


def _tiny_case(residues: int = 16) -> Workload:
    return Workload(
        dims=ALPHAFOLD3_2L.train_step_dims(batch=1, residues=residues),
        dtype="float32",
    )


def _yaml_model_section() -> dict:
    """MegaFold's ``model:`` block, without taking a YAML dependency.

    The file is flat enough at the depth these tests read that a two-level
    indent scan is honest, and pulling in a parser for five scalars would be a
    new dependency in the test suite for no more certainty.
    """
    config = _megafold_config()
    if config is None or not config.exists():
        raise unittest.SkipTest("no MegaFold checkout found")
    values: dict[str, str] = {}
    in_model = False
    for line in config.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            in_model = line.startswith("model:")
            continue
        if not in_model or line.startswith("    "):
            continue
        key, _, value = line.strip().partition(":")
        values[key] = value.split("#")[0].strip()
    return values


# -- the declaration half, which always runs ---------------------------------


class TestModulesImportWithoutMegaFold(unittest.TestCase):
    def test_the_workload_module_imports(self):
        from evograd.evaluation.tier3.workloads.megafold import workload

        self.assertTrue(callable(workload.make_workload))

    def test_the_adapter_module_imports(self):
        from evograd.evaluation.tier3.workloads.megafold import adapter

        self.assertEqual(adapter.ADAPTER.name, "megafold")

    @unittest.skipIf(
        HAVE_MEGAFOLD_PACKAGE, "MegaFold is on PYTHONPATH in this environment"
    )
    def test_make_workload_says_how_to_get_megafold(self):
        """The absent case is an instruction, not a traceback.

        MegaFold ships no packaging metadata, so 'install the extra' would be
        wrong advice; the message has to name PYTHONPATH.
        """
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

        with self.assertRaises(ImportError) as caught:
            make_workload(_tiny_case(), device="cpu", seed=0)
        message = str(caught.exception)
        self.assertIn("PYTHONPATH", message)
        self.assertIn("MegaFold", message)

    @unittest.skipUnless(
        HAVE_MEGAFOLD_PACKAGE and not HAVE_MEGAFOLD,
        "MegaFold either absent or fully importable here",
    )
    def test_a_gpuless_node_is_reported_as_one(self):
        """The third failure mode, and the one a login node actually hits.

        MegaFold reaches its fused Triton modules from its own ``__init__``,
        and those resolve a Triton driver at import time, so a complete
        checkout with every dependency installed still fails on a machine with
        no GPU -- with a ``RuntimeError``, not an ``ImportError``. Reported as
        "install something" it would send someone to fix a working install.
        """
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

        with self.assertRaises(ImportError) as caught:
            make_workload(_tiny_case(), device="cpu", seed=0)
        message = str(caught.exception)
        self.assertIn("GPU", message)
        self.assertNotIn("clone", message)


class TestTheDeclaration(unittest.TestCase):
    def test_it_is_the_registered_level_four_workload(self):
        self.assertIn("megafold", WORKLOADS)
        self.assertNotIn("alphafold3", WORKLOADS)
        self.assertEqual(get_workload("megafold").level, 4)

    def test_every_site_names_a_declared_operator(self):
        for site, op_name in get_workload("megafold").sites.items():
            with self.subTest(site=site):
                self.assertIn(op_name, TASKS)

    def test_the_five_sites_are_the_ones_megafold_fuses(self):
        self.assertEqual(
            get_workload("megafold").sites,
            {
                "layer_norm": "layernorm",
                "layernorm_linear": "layernorm_linear",
                "transition": "swiglu",
                "pair_bias_attention": "evoattention",
                "triangle_attention": "evoattention",
            },
        )

    def test_the_registry_agrees_with_the_declaration(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import MEGAFOLD_SITES

        self.assertEqual(MEGAFOLD_SITES.site_ops, get_workload("megafold").sites)

    def test_surgery_defaults_refuse_to_compute(self):
        """An unpatched surgery site is the original module in the tree, so a
        call through the kernel set for it is a wiring bug and must say so."""
        from evograd.evaluation.tier3.workloads.megafold.workload import MEGAFOLD_SITES

        kernels = KernelSet(registry=MEGAFOLD_SITES)
        with self.assertRaises(RuntimeError):
            kernels.layer_norm(None, None, None, 1e-5)

    def test_it_requires_megafold_by_name(self):
        self.assertEqual(get_workload("megafold").requires, "megafold")

    def test_the_notes_say_the_shapes_are_declared_not_harvested(self):
        notes = get_workload("megafold").notes
        self.assertIn("declared from MegaFold's configuration", notes)
        self.assertIn("not harvested", notes)


class TestTheExclusionsStateEveryRestriction(unittest.TestCase):
    """Each restriction the task operates under, named in the declaration.

    Not a spelling test. A level-4 speedup read without these overstates what
    was measured, so every one of them is a claim the declaration has to make
    out loud rather than a fact discoverable only by reading the workload.
    """

    def setUp(self):
        self.exclusions = get_workload("megafold").exclusions

    def _assert_states(self, *phrases: str):
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.exclusions)

    def test_the_unpatched_internals_are_named(self):
        self._assert_states(
            "triangle multiplication",
            "outer product mean",
            "MSA pair-weighted averaging",
            "diffusion module internals",
            "confidence head internals",
        )

    def test_attention_dropout_is_declared_off_for_every_provider(self):
        self._assert_states("Attention-weight dropout is disabled for every provider")

    def test_the_data_pipeline_is_declared_synthetic(self):
        self._assert_states(
            "MSA and template data pipeline is replaced by seeded synthetic "
            "features at a fixed crop",
            "constraint channels are present but uniformly masked",
        )

    def test_the_excluded_training_machinery_is_named(self):
        self._assert_states(
            "EMA", "gradient clipping", "DeepSpeed", "sequence parallelism"
        )

    def test_the_pinned_stochastic_choices_are_named(self):
        self._assert_states(
            "recycling count and whether the confidence head is called are "
            "pinned rather than sampled"
        )

    def test_random_weights_and_incomparable_timings_are_declared(self):
        self._assert_states(
            "Weights are randomly initialized",
            "not comparable to the MegaFold",
        )


class TestTheRunnerPassesTheProviderName(unittest.TestCase):
    """Which provider is being built is an argument, not kernel-set cargo.

    MegaFold's three providers are different *trees*, so an identical kernel
    set cannot distinguish them. The previous spelling smuggled a build mode
    into ``KernelSet.overrides`` under a key chosen not to collide with a site
    name, which put one workload's private convention inside a shared type.
    """

    def test_megafold_declares_the_provider_parameter(self):
        from evograd.evaluation.tier3.model import accepts_provider
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MegaFoldWorkload,
        )

        self.assertTrue(accepts_provider(MegaFoldWorkload.build_patched))

    def test_a_workload_that_does_not_want_it_is_not_given_it(self):
        """The compatibility property: every other workload is untouched.

        A builder that never declared a ``provider`` parameter must not start
        receiving one, or adding this hook would be a breaking change to every
        workload in the tree rather than an opt-in for one.
        """
        from evograd.evaluation.tier3.model import accepts_provider
        from evograd.evaluation.tier3.model import ModuleWorkload

        self.assertFalse(accepts_provider(ModuleWorkload.build_patched))

    def test_a_builder_taking_kwargs_is_given_it(self):
        from evograd.evaluation.tier3.model import accepts_provider

        def builder(kernels, **kwargs):
            return None

        self.assertTrue(accepts_provider(builder))

    def test_the_name_reaches_build_patched(self):
        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.patch import PatchProvenance

        seen = {}

        class _Workload:
            name = "spy"

            def build_patched(self, kernels, *, provider=None):
                seen["provider"] = provider
                return object(), PatchProvenance(method="module_surgery")

        build_with_provenance(_Workload(), KernelSet(), provider="megafold")
        self.assertEqual(seen["provider"], "megafold")

    def test_a_builder_without_it_still_builds(self):
        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.patch import PatchProvenance

        class _Workload:
            name = "old-style"

            def build_patched(self, kernels):
                return object(), PatchProvenance(method="module_surgery")

        _model, provenance = build_with_provenance(
            _Workload(), KernelSet(), provider="megafold"
        )
        self.assertEqual(provenance.method, "module_surgery")


class TestTheProviderKernelSetsCarryNothing(unittest.TestCase):
    """Three providers, three identical empty kernel sets, three trees."""

    def _providers(self):
        import argparse

        from evograd.evaluation.tier3.workloads.megafold.adapter import providers
        from evograd.evaluation.tier3.workloads.megafold.workload import MEGAFOLD_SITES

        return providers(argparse.Namespace(), MEGAFOLD_SITES)

    def test_the_four_providers_are_named(self):
        self.assertEqual(
            sorted(self._providers()),
            ["compile", "compile_clean", "eager", "megafold"],
        )

    def test_the_scaffolding_module_imports_without_megafold(self):
        """Its patches reach MegaFold only inside install()."""
        from evograd.evaluation.tier3.workloads.megafold import scaffolding

        self.assertFalse(scaffolding.installed())
        self.assertEqual(scaffolding.APPLIED, [])

    def test_none_of_them_carries_an_override(self):
        for name, kernels in self._providers().items():
            with self.subTest(provider=name):
                self.assertEqual(kernels.overrides, {})
                self.assertEqual(kernels.patched, ())


class TestTheAdapters(unittest.TestCase):
    def test_both_names_resolve(self):
        from evograd.evaluation.tier3.workloads import Tier3Adapter, tier3_adapter

        for name in ("megafold", "megafold_2l"):
            with self.subTest(model=name):
                adapter = tier3_adapter(name)
                self.assertIsInstance(adapter, Tier3Adapter)
                self.assertEqual(adapter.name, name)
                self.assertTrue(adapter.summary)

    def test_the_cli_offers_both(self):
        from evograd.evaluation.tier3.cli import MODELS

        self.assertIn("megafold", MODELS)
        self.assertIn("megafold_2l", MODELS)

    def test_the_workload_specific_flags_are_accepted_here(self):
        from evograd.evaluation.tier3.cli import _parser, check_options

        args = _parser().parse_args(
            ["--model", "megafold_2l", "--device", "cpu", "--residues", "32",
             "--evo", "none", "--checkpointing", "off", "--recycling", "2",
             "--confidence-head", "--baseline", "none"]
        )
        check_options(args)  # must not raise

    def test_those_flags_are_refused_for_a_workload_without_them(self):
        """A flag the selected workload does not understand is an error, not a
        silently ignored request."""
        import contextlib
        import io

        from evograd.evaluation.tier3.cli import _parser, check_options

        args = _parser().parse_args(
            ["--model", "qwen3_0_6b", "--device", "cpu", "--evo", "triton"]
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            check_options(args)
        self.assertIn("--evo", stderr.getvalue())

    def test_the_canonical_dtype_is_float32(self):
        from evograd.evaluation.tier3.workloads.megafold.adapter import DEFAULT_DTYPE

        self.assertEqual(DEFAULT_DTYPE, "float32")


# -- the model half, which needs a GPU node ----------------------------------


@needs_checkout
class TestTheDefaultsReproduceMegaFoldsConfig(unittest.TestCase):
    """The workload's defaults, checked against the yaml rather than restated.

    Gated on the checkout rather than on a buildable model: these read the
    workload's own constants against the file, which needs no GPU and no
    import of MegaFold's model module. It is the part of the model half that
    a CPU node can still hold to account.
    """

    def setUp(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MegaFoldWorkload,
        )

        self.yaml = _yaml_model_section()
        self.workload = MegaFoldWorkload(
            ALPHAFOLD3_2L, batch=1, residues=16, device="cpu"
        )

    def test_the_checkpoint_flags_match_the_yaml(self):
        described = self.workload.describe()
        for key in (
            "checkpoint_input_embedding",
            "checkpoint_trunk_pairformer",
            "checkpoint_diffusion_module",
            "checkpoint_distogram_head",
            "checkpoint_confidence_head",
        ):
            with self.subTest(flag=key):
                self.assertEqual(described[key], self.yaml[key] == "true")

    def test_the_yaml_model_kwargs_are_the_files_own(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            YAML_MODEL_KWARGS,
        )

        for key, value in YAML_MODEL_KWARGS.items():
            if key == "constraints":
                continue
            with self.subTest(key=key):
                self.assertIn(key, self.yaml)
                self.assertEqual(str(value).lower(), self.yaml[key].lower())

    def test_the_augmentation_count_is_the_trainers(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            DIFFUSION_NUM_AUGMENTATIONS,
        )

        text = _megafold_config().read_text(encoding="utf-8")
        self.assertIn(
            f"diffusion_num_augmentations: {DIFFUSION_NUM_AUGMENTATIONS}", text
        )


@needs_megafold
class TestTheModelBuilds(unittest.TestCase):
    def setUp(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MEGAFOLD_SITES,
            make_workload,
        )

        self.sites = MEGAFOLD_SITES
        self.workload = make_workload(_tiny_case(), device="cpu", seed=0)

    def test_build_returns_a_module(self):
        from torch import nn

        model = self.workload.build(KernelSet(registry=self.sites))
        self.assertIsInstance(model, nn.Module)

    def test_the_loss_is_finite(self):
        model = self.workload.build(KernelSet(registry=self.sites))
        loss = self.workload.loss(model, self.workload.batch_for(seed=1))
        self.assertTrue(torch.isfinite(loss).all())

    def test_describe_reports_every_pinned_choice(self):
        described = self.workload.describe()
        for key in (
            "residues", "batch", "seed", "dtype", "autocast_dtype",
            "use_optimized_evo", "num_recycling_steps",
            "diffusion_num_augmentations", "diffusion_chunk_size",
            "include_confidence_head",
            "checkpoint_input_embedding", "checkpoint_trunk_pairformer",
            "checkpoint_diffusion_module", "checkpoint_distogram_head",
            "checkpoint_confidence_head",
        ):
            with self.subTest(key=key):
                self.assertIn(key, described)

    def test_batches_are_identical_for_identical_seeds(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

        first = make_workload(_tiny_case(), device="cpu", seed=0).batch_for(seed=5)
        second = make_workload(_tiny_case(), device="cpu", seed=0).batch_for(seed=5)
        for key, value in first.items():
            with self.subTest(feature=key):
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, second[key]))

    def test_the_crop_length_is_exactly_as_declared(self):
        batch = self.workload.batch_for(seed=1)
        self.assertEqual(
            batch["atom_inputs"].shape[1], 16 * ALPHAFOLD3_2L.atoms_per_window
        )


@needs_megafold
class TestEverySiteMatchesSomething(unittest.TestCase):
    """A matcher that finds nothing would label a provider patched and be eager.

    ``patch_modules`` raises in that case, which is the protection; this checks
    it site by site so a failure names which matcher went stale rather than
    only that one did.
    """

    def test_each_site_finds_at_least_one_module(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MEGAFOLD_SITES,
            _module_patches,
            make_workload,
        )

        workload = make_workload(_tiny_case(), device="cpu", seed=0)
        model = workload.build(KernelSet(registry=MEGAFOLD_SITES))
        for spec in _module_patches():
            with self.subTest(site=spec.site):
                matched = [
                    path
                    for path, module in model.named_modules()
                    if path and spec.match(module)
                ]
                self.assertTrue(matched, f"{spec.site} matched nothing")


@needs_megafold
class TestTheProvidersAgree(unittest.TestCase):
    """The claim every level-4 number rests on.

    ``eager`` and ``megafold`` are two implementations of one architecture on
    identical weights and identical batches. If their losses disagree, the
    speedup between them is not attributable to the kernels.
    """

    def _providers(self):
        from evograd.evaluation.tier3.workloads.megafold.adapter import providers
        from evograd.evaluation.tier3.workloads.megafold.workload import MEGAFOLD_SITES

        import argparse

        args = argparse.Namespace()
        return providers(args, MEGAFOLD_SITES)

    def _loss_per_provider(self) -> dict[str, float]:
        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

        losses = {}
        for name, kernels in self._providers().items():
            workload = make_workload(_tiny_case(), device="cpu", seed=0)
            model, _ = build_with_provenance(workload, kernels, provider=name)
            losses[name] = float(
                workload.loss(model, workload.batch_for(seed=1)).detach()
            )
        return losses

    def test_every_provider_builds(self):
        from torch import nn

        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.workloads.megafold.workload import make_workload

        for name, kernels in self._providers().items():
            with self.subTest(provider=name):
                workload = make_workload(_tiny_case(), device="cpu", seed=0)
                model, provenance = build_with_provenance(
                    workload, kernels, provider=name
                )
                self.assertIsInstance(model, nn.Module)
                # The provider that built the tree is recorded, because three
                # providers request the same sites and still differ.
                self.assertEqual(provenance.method, f"module_surgery:{name}")

    def test_an_unrecognized_provider_gets_the_eager_tree(self):
        """``candidate`` and the identity control are eager plus their kernels.

        That is what makes an evolved provider comparable to the reference: it
        differs in its kernels and in nothing else.
        """
        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MEGAFOLD_SITES,
            make_workload,
        )

        workload = make_workload(_tiny_case(), device="cpu", seed=0)
        _model, provenance = build_with_provenance(
            workload, KernelSet(registry=MEGAFOLD_SITES), provider="candidate"
        )
        self.assertEqual(provenance.method, "module_surgery:eager")

    def test_eager_and_megafold_agree_on_one_batch(self):
        losses = self._loss_per_provider()
        self.assertLess(abs(losses["eager"] - losses["megafold"]), 5e-3, losses)

    def test_compile_agrees_with_eager(self):
        """``torch.compile`` must not change the answer.

        It is the same module tree behind a compiler, so any disagreement is
        the compiler's and would make its speedup unattributable.
        """
        losses = self._loss_per_provider()
        self.assertLess(abs(losses["eager"] - losses["compile"]), 5e-3, losses)

    def test_the_compile_provider_does_no_surgery(self):
        """It wraps the eager model; it replaces no module.

        If it patched sites as well, its number would blend the compiler with
        whatever was patched, which is exactly what it exists to separate.
        """
        from evograd.evaluation.tier3.model import build_with_provenance
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MEGAFOLD_SITES,
            make_workload,
        )

        workload = make_workload(_tiny_case(), device="cpu", seed=0)
        model, provenance = build_with_provenance(
            workload, KernelSet(registry=MEGAFOLD_SITES), provider="compile"
        )
        self.assertEqual(provenance.actual_sites, ())
        self.assertEqual(provenance.method, "module_surgery:compile")
        # torch.compile hands back a wrapper around the original module.
        self.assertTrue(hasattr(model, "_orig_mod"))


@needs_megafold
class TestIdentityControl(unittest.TestCase):
    """Same mathematics, all of the patching machinery, same loss.

    Patching every site with the eager mathematics routed through ``bind`` and
    the adapters -- the exact plumbing an evolved kernel uses -- must not change
    the loss. A patch layer that alters the answer would make every level-4
    speedup unattributable.
    """

    def test_patched_and_unpatched_losses_agree(self):
        from evograd.evaluation.tier3.workloads.megafold.workload import (
            MEGAFOLD_SITES,
            make_workload,
        )

        workload = make_workload(_tiny_case(), device="cpu", seed=0)
        model = workload.build(KernelSet(registry=MEGAFOLD_SITES))
        loss_eager = workload.loss(model, workload.batch_for(seed=1))

        patched = make_workload(_tiny_case(), device="cpu", seed=0)
        control = identity_control_kernels(TASKS, registry=MEGAFOLD_SITES)
        patched_model, provenance = patched.build_patched(control)
        self.assertEqual(set(provenance.actual_sites), set(MEGAFOLD_SITES.names))
        self.assertTrue(any(provenance.paths.values()))
        loss_patched = patched.loss(patched_model, patched.batch_for(seed=1))

        self.assertLess(
            abs(float(loss_eager.detach()) - float(loss_patched.detach())), 5e-3
        )

        loss_patched.backward()
        grads = sum(1 for p in patched_model.parameters() if p.grad is not None)
        self.assertGreater(grads, 0)


if __name__ == "__main__":
    unittest.main()
