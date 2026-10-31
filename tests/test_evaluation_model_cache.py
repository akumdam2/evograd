"""Pristine snapshots never share trained weights, gradients or patches."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch

from evograd.evaluation.model_cache import PristineModelCache

BUILDS = []


@dataclass(frozen=True)
class Spec:
    device: str = "cpu"
    weights: str = "pretrained"
    seed: int = 3


def build(spec):
    BUILDS.append(spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        model = torch.nn.Linear(4, 4)
    model.tied_weight = model.weight
    model.register_buffer("extra", torch.ones(1), persistent=False)
    return model.to(spec.device).train()


def changes_rng(spec):
    return torch.nn.Linear(4, 4)


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        BUILDS.clear()

    def test_load_once_and_reset_weights_gradients_buffers_and_patches(self):
        cache = PristineModelCache(self.directory)
        before = torch.random.get_rng_state().clone()
        first = cache.build(build, Spec())
        expected = first.weight.detach().clone()
        first(torch.ones(2, 4)).sum().backward()
        with torch.no_grad():
            first.weight.add_(5)
            first.extra.add_(3)
        first.patch = "agent-a"
        second = cache.build(build, Spec())
        self.assertEqual(len(BUILDS), 1)
        self.assertTrue(torch.equal(second.weight, expected))
        self.assertIsNone(second.weight.grad)
        self.assertFalse(hasattr(second, "patch"))
        self.assertTrue(torch.equal(second.extra, torch.ones(1)))
        self.assertIs(second.weight, second.tied_weight)
        self.assertNotEqual(first.weight.data_ptr(), second.weight.data_ptr())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_independent_cache_can_read_snapshot_without_checkpoint_builder(self):
        first = PristineModelCache(self.directory).build(build, Spec())
        cache = PristineModelCache(self.directory)
        second = cache.build(build, Spec())
        self.assertEqual(len(BUILDS), 1)
        self.assertEqual(cache.loads, 0)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertIs(second.weight, second.tied_weight)
        self.assertNotEqual(first.weight.data_ptr(), second.weight.data_ptr())

    def test_spawned_python_reads_snapshot_without_reloading_checkpoint(self):
        PristineModelCache(self.directory).build(build, Spec())
        code = """
import importlib, sys
from evograd.evaluation.model_cache import PristineModelCache
module = importlib.import_module(sys.argv[2])
cache = PristineModelCache(sys.argv[1])
model = cache.build(module.build, module.Spec())
assert cache.loads == 0
assert module.BUILDS == []
assert model.weight is model.tied_weight
"""
        child = subprocess.run([sys.executable, "-c", code, str(self.directory), build.__module__],
                               capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_different_specs_do_not_share_snapshot(self):
        cache = PristineModelCache(self.directory)
        first = cache.build(build, Spec(seed=3))
        second = cache.build(build, Spec(seed=4))
        self.assertEqual(len(BUILDS), 2)
        self.assertFalse(torch.equal(first.weight, second.weight))

    def test_random_models_are_built_normally(self):
        cache = PristineModelCache(self.directory)
        cache.build(build, Spec(weights="random"))
        cache.build(build, Spec(weights="random"))
        self.assertEqual(len(BUILDS), 2)
        self.assertFalse(list(self.directory.glob("*.pt")))

    def test_rng_changing_loader_is_not_silently_cached(self):
        cache = PristineModelCache(self.directory)
        with self.assertRaisesRegex(RuntimeError, "changes RNG"):
            cache.build(changes_rng, Spec())
        self.assertFalse(list(self.directory.glob("*.pt")))


if __name__ == "__main__":
    unittest.main()
