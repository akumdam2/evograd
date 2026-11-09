"""Nested sites are declared, and a model never holds two of them at once."""

from __future__ import annotations

import unittest

from torch import nn

from evograd.evaluation.tier3.patch import (
    KernelSet,
    ModulePatch,
    OverlappingSites,
    Site,
    SiteRegistry,
    patch,
    patch_modules,
)


def refuse(*_args):
    raise AssertionError("no kernel runs here")


def registry(overlaps=()):
    return SiteRegistry("nested", (Site("outer", "op", refuse), Site("inner", "op", refuse),
                                   Site("beside", "op", refuse)), overlaps=overlaps)


class Inner(nn.Module):
    pass


class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Inner()


class Beside(nn.Module):
    pass


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.outer = Outer()
        self.beside = Beside()


PATCHES = (ModulePatch("outer", lambda m: type(m) is Outer, lambda m, _k: nn.Identity()),
           ModulePatch("inner", lambda m: type(m) is Inner, lambda m, _k: nn.Identity()),
           ModulePatch("beside", lambda m: type(m) is Beside, lambda m, _k: nn.Identity()))


def kernels(registered, *sites):
    out = KernelSet(registry=registered)
    for site in sites:
        out = patch(out, site, refuse)
    return out


class OverlapsAreDeclared(unittest.TestCase):
    def test_a_pair_of_unknown_sites_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown sites"):
            registry(overlaps=(("outer", "nowhere"),))

    def test_conflicts_names_only_pairs_both_present(self):
        declared = registry(overlaps=(("outer", "inner"),))
        self.assertEqual(declared.conflicts(("outer", "inner", "beside")), (("outer", "inner"),))
        self.assertEqual(declared.conflicts(("outer", "beside")), ())

    def test_groups_cover_every_site_and_hold_no_nested_pair(self):
        declared = registry(overlaps=(("outer", "inner"),))
        groups = declared.groups()
        self.assertEqual(groups, (("outer", "beside"), ("inner",)))
        self.assertEqual(sorted(s for g in groups for s in g), sorted(declared.names))
        self.assertFalse(any(declared.conflicts(g) for g in groups))

    def test_a_registry_without_overlaps_is_one_group(self):
        self.assertEqual(registry().groups(), (("outer", "inner", "beside"),))


class PatchModulesRefusesThem(unittest.TestCase):
    def test_a_nested_pair_is_refused_before_the_tree_is_touched(self):
        model, declared = Model(), registry(overlaps=(("outer", "inner"),))
        with self.assertRaisesRegex(OverlappingSites, "outer"):
            patch_modules(model, PATCHES, kernels(declared, "outer", "inner"))
        self.assertIsInstance(model.outer, Outer)
        self.assertIsInstance(model.outer.inner, Inner)

    def test_the_error_is_a_value_error(self):
        self.assertTrue(issubclass(OverlappingSites, ValueError))

    def test_either_site_alone_and_siblings_install(self):
        declared = registry(overlaps=(("outer", "inner"),))
        for sites in (("outer",), ("inner",), ("outer", "beside"), ("inner", "beside")):
            with self.subTest(sites=sites):
                provenance = patch_modules(Model(), PATCHES, kernels(declared, *sites))
                self.assertEqual(sorted(provenance.actual_sites), sorted(sites))

    def test_undeclared_sites_behave_as_before(self):
        provenance = patch_modules(Model(), PATCHES[1:], kernels(registry(), "inner", "beside"))
        self.assertEqual(sorted(provenance.actual_sites), ["beside", "inner"])


class TheIdentityControl(unittest.TestCase):
    def ops(self):
        from types import SimpleNamespace

        from evograd.benchmark import get_task

        return {"op": get_task("swiglu"), **{"unused": SimpleNamespace()}}

    def test_by_default_it_patches_what_can_be_installed_together(self):
        from evograd.evaluation.tier3.patch import identity_control_kernels

        declared = registry(overlaps=(("outer", "inner"),))
        kernels = identity_control_kernels(self.ops(), registry=declared)
        self.assertEqual(sorted(kernels.patched), ["beside", "outer"])
        self.assertEqual(declared.conflicts(kernels.patched), ())

    def test_a_registry_without_overlaps_gets_every_site_as_before(self):
        from evograd.evaluation.tier3.patch import identity_control_kernels

        self.assertEqual(sorted(identity_control_kernels(self.ops(), registry=registry()).patched),
                         ["beside", "inner", "outer"])

    def test_sites_named_are_taken_as_named(self):
        from evograd.evaluation.tier3.patch import identity_control_kernels

        kernels = identity_control_kernels(self.ops(), ("inner",), registry=registry())
        self.assertEqual(kernels.patched, ("inner",))


if __name__ == "__main__":
    unittest.main()
