"""Llama's unpatched whole-model ``torch.compile`` provider, and its synthetic batches."""

from __future__ import annotations

import unittest

import torch

SMALL = {
    "device": "cpu", "dtype": "float32", "batch_size": 1, "seq_len": 32,
    "arch_overrides": {"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128,
                       "num_attention_heads": 4, "num_key_value_heads": 1,
                       "head_dim": 16, "vocab_size": 128},
}


def _workload():
    from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload

    return Llama3Workload.from_config(SMALL)


class TestWholeModelCompileProvider(unittest.TestCase):
    def test_it_patches_nothing_and_is_recognised(self):
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.llama3_2_1b.sites import llama3_sites
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import (
            whole_model_compile_kernels, whole_model_compile_requested,
        )

        registry = llama3_sites()
        kernels = whole_model_compile_kernels(registry)
        self.assertEqual(kernels.patched, ())
        self.assertTrue(whole_model_compile_requested(kernels))
        self.assertFalse(whole_model_compile_requested(KernelSet(registry=registry)))

    def test_it_cannot_be_combined_with_a_site_patch(self):
        from evograd.evaluation.tier3.patch import KernelSource, patch
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import whole_model_compile_kernels

        workload = _workload()
        both = patch(whole_model_compile_kernels(workload.site_registry), "swiglu_mlp",
                     lambda *a: None,
                     source=KernelSource("swiglu_mlp", "llama3_swiglu_mlp", None, "candidate"))
        with self.assertRaises(ValueError) as caught:
            workload.build_patched(both)
        self.assertIn("cannot be combined", str(caught.exception))


class TestSyntheticBatches(unittest.TestCase):
    def test_training_and_validation_streams_are_disjoint_and_indexed(self):
        from evograd.evaluation.tier3.workloads.llama3_2_1b import training as T

        workload = _workload()
        train = T.train_batches(workload, 4)
        validation = T.validation_batches(workload, 2)
        self.assertNotEqual(train.offset, validation.offset)
        self.assertGreater(abs(train.offset - validation.offset), 1000)
        self.assertFalse(torch.equal(train.batch(0)[0], validation.batch(0)[0]))
        # positional, so re-reading gives the same tokens and validation cannot
        # advance the training stream
        self.assertTrue(torch.equal(validation.batch(1)[0], validation.batch(1)[0]))
        self.assertTrue(torch.equal(train.batch(2)[0], train.batch(2)[0]))
        self.assertIn("synthetic", validation.describe()["source"])
        self.assertIn("not", validation.describe()["note"])   # the label travels


if __name__ == "__main__":
    unittest.main()
