"""Qwen3's whole-model ``torch.compile`` provider: patches nothing, combines with nothing."""

from __future__ import annotations

import unittest


class TestWholeModelCompileProvider(unittest.TestCase):
    def test_patches_nothing_and_is_recognised(self):
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites

        kernels = W.whole_model_compile_kernels(qwen3_sites())
        self.assertEqual(kernels.patched, ())
        self.assertTrue(W.whole_model_compile_requested(kernels))
        self.assertFalse(W.whole_model_compile_requested(KernelSet(registry=qwen3_sites())))

    def test_cannot_be_combined_with_a_site_patch(self):
        from evograd.evaluation.tier3.patch import KernelSource, patch
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import workload as W
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.sites import qwen3_sites

        registry = qwen3_sites()
        kernels = patch(W.whole_model_compile_kernels(registry), "swiglu_mlp", lambda *a: None,
                        source=KernelSource(site="swiglu_mlp", op_name="qwen3_swiglu_mlp",
                                            module=None, origin="callable"))
        wl = W.Qwen3Workload(device="cpu", arch_overrides={"num_hidden_layers": 1})
        with self.assertRaises(ValueError):
            wl.build_patched(kernels)


if __name__ == "__main__":
    unittest.main()
