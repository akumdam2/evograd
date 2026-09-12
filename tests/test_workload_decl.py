"""Level-4 workload declarations: validation rules and the registry.

What these pin is the declaration contract — what a whole-model task must
state before it can claim a place in the suite — and that the AlphaFold3
declaration's shapes re-derive from the model configuration it cites, the same
provenance discipline every level-1..3 benchmark case is held to.
"""

from __future__ import annotations

import unittest

from evograd.opdecl import Provenance, Workload, declare_workload
from evograd.opdecl.models import ALPHAFOLD3, rederive_dims
from evograd.benchmark import WORKLOADS, get_workload
from evograd.benchmark import TASKS


def _case(dims=None, model="alphafold3", component="train_step", free=None):
    return Workload(
        dims=dims or ALPHAFOLD3.train_step_dims(batch=1, residues=128),
        dtype="float32",
        provenance=Provenance(
            model=model, component=component, free=free or {"batch": 1, "residues": 128}
        ),
    )


def _declare(**overrides):
    kwargs = dict(
        name="alphafold3",
        factory="evograd.evaluation.tier3.workloads.alphafold3.workload:make_workload",
        family="protein",
        model="alphafold3",
        sites={"layer_norm": "layernorm"},
        benchmark=(_case(),),
        exclusions="stated",
    )
    kwargs.update(overrides)
    return declare_workload(**kwargs)


class TestValidation(unittest.TestCase):
    def test_a_complete_declaration_validates(self):
        self.assertEqual(_declare().level, 4)

    def test_level_four_is_the_only_level(self):
        with self.assertRaises(ValueError):
            _declare(level=3)

    def test_sites_are_required(self):
        with self.assertRaises(ValueError):
            _declare(sites={})

    def test_exclusions_must_be_stated(self):
        with self.assertRaises(ValueError):
            _declare(exclusions="")

    def test_benchmark_cases_need_provenance(self):
        bare = Workload(
            dims=ALPHAFOLD3.train_step_dims(batch=1, residues=128), dtype="float32"
        )
        with self.assertRaises(ValueError):
            _declare(benchmark=(bare,))

    def test_provenance_must_cite_the_declared_model(self):
        with self.assertRaises(ValueError):
            _declare(benchmark=(_case(model="llama_3_2_1b", component="rmsnorm"),))

    def test_factory_must_be_a_module_reference(self):
        with self.assertRaises(ValueError):
            _declare(factory="not_a_reference")


class TestTheAlphafold3Declaration(unittest.TestCase):
    def test_it_is_registered(self):
        self.assertIn("alphafold3", WORKLOADS)
        self.assertEqual(get_workload("alphafold3").level, 4)

    def test_every_site_names_a_declared_operator(self):
        for site, op_name in get_workload("alphafold3").sites.items():
            with self.subTest(site=site):
                self.assertIn(op_name, TASKS)

    def test_every_benchmark_case_rederives_from_the_config(self):
        """The provenance assertion, extended to level 4: the stored dims and
        the model configuration must agree, or one of them is wrong."""
        for case in get_workload("alphafold3").benchmark:
            with self.subTest(dims=case.dims):
                self.assertEqual(rederive_dims(case.provenance), case.dims)

    def test_the_factory_resolves(self):
        factory = get_workload("alphafold3").resolve_factory()
        self.assertTrue(callable(factory))

    def test_unknown_workload_names_fail_loudly(self):
        with self.assertRaises(KeyError):
            get_workload("alphafold4")


if __name__ == "__main__":
    unittest.main()


class TestArchIsNotMutableByTheLibrary(unittest.TestCase):
    """``spec.arch`` is handed to ``AutoConfig``, which rewrites it in place.

    Transformers normalises RoPE by writing ``rope_theta`` into ``rope_scaling``.
    With a shallow copy that reached the module-level declaration, so one
    ``build_model`` changed ``config_hash`` and ``workload_id`` for the rest of
    the process -- a harvest then recorded an id the capture refused. Only a
    model with a non-``None`` nested config value can show it.
    """

    def test_mutating_the_returned_arch_leaves_the_declaration_alone(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level4.spec import (
            CANONICAL, LLAMA_3_2_1B,
        )

        before = CANONICAL.workload_id
        arch = CANONICAL.arch
        arch["rope_scaling"]["rope_theta"] = 500000.0   # what transformers does
        arch["num_hidden_layers"] = 999

        self.assertNotIn("rope_theta", LLAMA_3_2_1B["rope_scaling"])
        self.assertEqual(LLAMA_3_2_1B["num_hidden_layers"], 16)
        self.assertEqual(CANONICAL.workload_id, before)

    def test_two_reads_are_independent(self):
        from evograd.benchmark.topdown.llama3_2_1b.levels.level4.spec import CANONICAL

        first = CANONICAL.arch
        first["rope_scaling"]["injected"] = True
        self.assertNotIn("injected", CANONICAL.arch["rope_scaling"])
