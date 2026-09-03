"""The AlphaFold3 level-4 workload, on CPU.

The expensive claim these tests pin is the identity control: patching every
AF3 site with the eager mathematics routed through ``bind`` and the adapters —
the exact plumbing an evolved kernel uses — must not change the loss. A patch
layer that alters the answer would make every level-4 speedup unattributable.

Tests needing the model implementation skip when ``alphafold3-pytorch`` is not
installed (the ``af3`` extra); the declaration-side tests always run.
"""

from __future__ import annotations

import unittest

import torch

from evograd.bench.suite import task_from_tier3_report
from evograd.bench.tier3_patch import KernelSet, identity_control_kernels
from evograd.opdecl.activity import Workload
from evograd.opdecl.models import ALPHAFOLD3_2L
from evograd.ops import OPS, get_workload

try:
    import alphafold3_pytorch  # noqa: F401

    HAVE_AF3 = True
except ImportError:
    HAVE_AF3 = False

needs_af3 = unittest.skipUnless(HAVE_AF3, "alphafold3-pytorch not installed")


def _tiny_case() -> Workload:
    return Workload(
        dims=ALPHAFOLD3_2L.train_step_dims(batch=1, residues=4), dtype="float32"
    )


class TestRegistryAgreesWithTheDeclaration(unittest.TestCase):
    def test_sites_match(self):
        from evograd.ops.level4.alphafold3.workload import AF3_SITES

        self.assertEqual(AF3_SITES.site_ops, get_workload("alphafold3").sites)

    def test_surgery_defaults_refuse_to_compute(self):
        """An unpatched surgery site is the original module in the tree, so a
        call through the kernel set for it is a wiring bug and must say so."""
        from evograd.ops.level4.alphafold3.workload import AF3_SITES

        kernels = KernelSet(registry=AF3_SITES)
        with self.assertRaises(RuntimeError):
            kernels.layer_norm(None, None, None, 1e-5)


class TestSuiteReader(unittest.TestCase):
    def _report(self, **providers):
        return {"protocol": "evograd-tier3-model-v1", "providers": providers}

    def test_full_step_speedup_from_step_latency(self):
        task = task_from_tier3_report(
            "alphafold3",
            "protein",
            self._report(
                eager={"ok": True, "step_ms": 30.0},
                candidate={"ok": True, "step_ms": 20.0},
            ),
        )
        self.assertEqual(task.level, 4)
        self.assertEqual(task.tier, "model")
        self.assertAlmostEqual(task.speedup, 1.5)
        self.assertTrue(task.ok)

    def test_a_failed_candidate_is_a_case_error_not_a_speedup(self):
        task = task_from_tier3_report(
            "alphafold3",
            "protein",
            self._report(
                eager={"ok": True, "step_ms": 30.0},
                candidate={"ok": False, "error": "OOM"},
            ),
        )
        self.assertEqual(task.speedups, ())
        self.assertFalse(task.ok)
        self.assertIn("OOM", task.case_errors[0])

    def test_a_missing_provider_is_an_error(self):
        task = task_from_tier3_report(
            "alphafold3", "protein", self._report(eager={"ok": True, "step_ms": 30.0})
        )
        self.assertIsNotNone(task.error)


@needs_af3
class TestWorkloadDeterminism(unittest.TestCase):
    def test_batches_are_identical_for_identical_seeds(self):
        factory = get_workload("alphafold3").resolve_factory()
        first = factory(_tiny_case(), device="cpu", seed=0).batch_for(seed=5)
        second = factory(_tiny_case(), device="cpu", seed=0).batch_for(seed=5)
        for key, value in first.items():
            with self.subTest(feature=key):
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, second[key]))

    def test_the_crop_length_is_exactly_as_declared(self):
        factory = get_workload("alphafold3").resolve_factory()
        batch = factory(_tiny_case(), device="cpu", seed=0).batch_for(seed=1)
        self.assertEqual(
            batch["atom_inputs"].shape[1], 4 * ALPHAFOLD3_2L.atoms_per_window
        )


@needs_af3
class TestIdentityControl(unittest.TestCase):
    """Same mathematics, all of the patching machinery, same loss."""

    def test_patched_and_unpatched_losses_agree(self):
        from evograd.ops.level4.alphafold3.workload import AF3_SITES, make_workload

        case = _tiny_case()

        workload = make_workload(case, device="cpu", seed=0)
        model = workload.build(KernelSet(registry=AF3_SITES))
        batch = workload.batch_for(seed=1)
        loss_eager = workload.loss(model, batch)

        patched = make_workload(case, device="cpu", seed=0)
        control = identity_control_kernels(OPS, registry=AF3_SITES)
        patched_model, provenance = patched.build_patched(control)
        self.assertEqual(provenance.method, "module_surgery")
        self.assertEqual(
            set(provenance.actual_sites), set(AF3_SITES.names)
        )
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
