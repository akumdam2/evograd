"""Llama-3's tier-3 half: the sites, the workload, and the two identity controls.

Everything here runs on a two-layer CPU model with reduced widths. That is not a
compromise -- the properties being checked are structural (does the adapter call
the same submodules in the same order, does the residual carrier hand the right
tensor to the right norm, does every site fire the declared number of times) and
none of them depends on the model being 8B or on the device being a GPU.

The load-bearing assertion is ``structural identity``: patching every site with
the production spelling changes the module structure and no arithmetic, so the
loss and every gradient must be **bitwise** identical to the unpatched model.
Anything else is a defect in the restructure rather than a tolerance question.
"""

from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover - exercised on machines without them
    torch = None


#: A two-layer Llama with 4:1 grouped-query attention and Llama's own
#: ``n_heads * head_dim == hidden`` relation preserved, so the adapters meet the
#: shape family the real model presents rather than a rounder one.
SMALL = {
    "device": "cpu",
    "dtype": "float32",
    "batch_size": 1,
    "seq_len": 32,
    "arch_overrides": {
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 128,
    },
}


def _workload():
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    return Llama3Workload.from_config(SMALL)


def _step(workload, kernels):
    """One forward/backward, returning the loss and every parameter gradient."""
    model, provenance = workload.build_patched(kernels)
    loss = workload.loss(model, workload.batch_for(seed=0))
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return float(loss.detach()), grads, provenance, workload.last_build


class TestSiteRegistry(unittest.TestCase):
    def setUp(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import sites

        self.sites = sites

    def test_the_four_site_names_do_not_collide_with_qwen3s(self):
        """A shared name would let one model's kernel set be restricted onto the
        other's model without complaint. ``qkv_rope`` is a different operator
        from ``qkv_norm_rope`` and says so."""
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import sites as qwen

        mine = {s.name for s in self.sites.llama3_sites().sites}
        theirs = {s.name for s in qwen.qwen3_sites().sites}
        self.assertEqual(mine & theirs, {"swiglu_mlp", "residual_rmsnorm"})
        self.assertIn("qkv_rope", mine)
        self.assertNotIn("qkv_rope", theirs)

    def test_the_projection_site_is_gated_against_llamas_own_declaration(self):
        registry = self.sites.llama3_sites()
        self.assertEqual(registry.require("qkv_rope").op, "llama3_qkv_rope")

    def test_patching_either_attention_site_makes_both_live(self):
        """One ``LlamaAttention`` adapter holds two switches, so a count read
        against the requested sites alone would be wrong."""
        self.assertEqual(self.sites.live_sites(("qkv_rope",)),
                         ("attention", "qkv_rope"))
        self.assertEqual(self.sites.supporting_sites(("qkv_rope",)), ("attention",))

    def test_expected_counts_follow_the_layer_count(self):
        """``2 * layers`` fusions, not ``2 * layers + 1``: layer 0's
        ``input_layernorm`` has no preceding decoder add."""
        self.assertEqual(
            self.sites.expected_counts(32),
            {"qkv_rope": 32, "attention": 32, "swiglu_mlp": 32,
             "residual_rmsnorm": 64},
        )

    def test_production_attention_does_not_pass_sliding_window(self):
        """``LlamaAttention`` has no such attribute and does not pass one.
        Copying Qwen3's call would be a different call, not the same one twice."""
        import inspect

        source = inspect.getsource(self.sites.production_attention)
        self.assertNotIn("sliding_window", source.split('"""')[2])


@unittest.skipIf(torch is None, "torch and transformers are required")
class TestIdentityControls(unittest.TestCase):
    """Both ways of reaching a kernel, on a real two-layer model."""

    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.patch import KernelSet

        cls.workload = _workload()
        cls.eager_loss, cls.eager_grads, _p, _b = _step(
            cls.workload, KernelSet(registry=cls.workload.site_registry)
        )

    def test_structural_identity_is_bitwise(self):
        """Every site patched with the spelling it already had. The module
        structure changes and the arithmetic does not, so this is not a
        tolerance question -- it is equality."""
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import (
            structural_identity_kernels,
        )

        loss, grads, provenance, built = _step(
            self.workload, structural_identity_kernels(self.workload.site_registry)
        )
        self.assertEqual(loss, self.eager_loss)
        for name, grad in grads.items():
            with self.subTest(parameter=name):
                self.assertTrue(bool(torch.equal(grad, self.eager_grads[name])))
        self.assertEqual(provenance.method, "module_surgery")
        self.assertEqual(built.count_problems(), [])

    def test_structural_identity_leaves_the_state_dict_untouched(self):
        """The adapters rebind ``forward``; they do not build a lookalike and
        copy weights. So the parameters are the same objects, and key order,
        dtype and ``requires_grad`` are identical by construction."""
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import (
            structural_identity_kernels,
        )

        plain, _ = self.workload.build_patched(
            KernelSet(registry=self.workload.site_registry)
        )
        patched, _ = self.workload.build_patched(
            structural_identity_kernels(self.workload.site_registry)
        )
        self.assertEqual(list(plain.state_dict()), list(patched.state_dict()))

    def test_every_site_runs_the_declared_number_of_times(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import (
            structural_identity_kernels,
        )

        _loss, _grads, _prov, built = _step(
            self.workload, structural_identity_kernels(self.workload.site_registry)
        )
        self.assertEqual(
            built.observed(),
            {"qkv_rope": 2, "attention": 2, "swiglu_mlp": 2, "residual_rmsnorm": 4},
        )

    def test_the_bound_pair_reaches_every_site_through_bind(self):
        """The path an evolved kernel takes: the declared operator wrapped in a
        ``torch.autograd.Function``. Gated by declared tolerances rather than by
        equality, because the reference and the production spelling are
        different computations -- but in float32 on these widths they agree."""
        from evograd.evaluation.tier3.patch import restrict
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import (
            bound_pair_identity_kernels,
        )
        from evograd.benchmark import TASKS

        full = bound_pair_identity_kernels(TASKS, None, self.workload.site_registry)
        self.assertEqual(
            sorted(full.patched),
            ["attention", "qkv_rope", "residual_rmsnorm", "swiglu_mlp"],
        )
        for site in sorted(full.patched):
            with self.subTest(site=site):
                loss, grads, _p, _b = _step(self.workload, restrict(full, (site,)))
                self.assertAlmostEqual(loss, self.eager_loss, places=5)
                worst = max(
                    float((grads[n] - self.eager_grads[n]).abs().max())
                    for n in self.eager_grads
                )
                self.assertLess(worst, 1e-5)


if __name__ == "__main__":
    unittest.main()
