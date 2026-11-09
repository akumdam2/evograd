"""The recorder on a small eager tree (CPU, no MegaFold import).

It changes no arithmetic, sees nested boundaries and the primitives inside them,
ignores calls that are not training calls, and counts everything in the census.
"""

from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from evograd.evaluation.tier3.patch import ModulePatch
from evograd.evaluation.tier3.workloads.megafold import sites as S
from evograd.evaluation.tier3.workloads.megafold import workload as W
from evograd.verification.integrations.megafold.capture import MegaFoldCapture, _dims, module_role


def layernorm_linear(k=8, n=16):
    return W._EagerLayernormLinear(SimpleNamespace(
        WEIGHT=nn.Parameter(1 + 0.1 * torch.randn(k)), BIAS=nn.Parameter(0.1 * torch.randn(k)),
        linear_weight=nn.Parameter(torch.randn(k, n) / k ** 0.5), linear_bias=None, K=k, N=n,
        has_layernorm_bias=True, has_linear_bias=False))


class Tree(nn.Module):
    """A transition holding a LayerNorm+Linear, a LayerNorm+Linear beside it, checkpointed or not."""

    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.block = W._EagerTransition(SimpleNamespace(ff=(
            layernorm_linear(8, 16), None, nn.Linear(8, 8, bias=False))))
        self.head = layernorm_linear(8, 4)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        y = checkpoint(self.block, x, use_reentrant=False) if self.use_checkpoint else self.block(x)
        return self.head(y + x)


def patches(_model):
    return (ModulePatch("layernorm_linear", lambda m: type(m) is W._EagerLayernormLinear, None),
            ModulePatch("transition", lambda m: type(m) is W._EagerTransition, None))


class Recording(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        patcher = mock.patch.object(W, "_module_patches", patches)
        patcher.start()
        self.addCleanup(patcher.stop)

    def step(self, tree, x):
        tree.zero_grad()
        loss = tree(x).pow(2).sum()
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in tree.parameters()]

    def test_the_step_is_unchanged_bit_for_bit(self):
        tree, x = Tree(), torch.randn(3, 5, 8)
        plain_loss, plain_grads = self.step(tree, x)
        capture = MegaFoldCapture(tree).install()
        try:
            loss, grads = self.step(tree, x)
        finally:
            capture.remove()
        self.assertTrue(torch.equal(loss, plain_loss))
        for mine, theirs in zip(grads, plain_grads):
            self.assertTrue(torch.equal(mine, theirs))

    def test_every_level_of_a_nested_boundary_is_seen(self):
        capture = MegaFoldCapture(Tree()).install()
        try:
            self.step(capture.model, torch.randn(3, 5, 8))
        finally:
            capture.remove()
        tasks = sorted(c["task"] for c in capture.calls)
        self.assertEqual(tasks, ["layernorm", "layernorm", "linear_no_bias", "linear_no_bias",
                                 "linear_no_bias", "megafold_layernorm_linear",
                                 "megafold_layernorm_linear", "swiglu"])
        by_path = {c["path"]: c for c in capture.calls if c["task"] == "megafold_layernorm_linear"}
        self.assertEqual(by_path["block.ff.0"]["dims"], dict(M=15, K=8, N=16))
        self.assertEqual(by_path["head"]["dims"], dict(M=15, K=8, N=4))

    def test_the_census_counts_what_no_task_binds(self):
        capture = MegaFoldCapture(Tree()).install()
        try:
            self.step(capture.model, torch.randn(3, 5, 8))
        finally:
            capture.remove()
        self.assertEqual(capture.census["layer_norm"], 2)
        self.assertEqual(capture.census["silu"], 1)
        self.assertEqual(capture.census["linear"], 3)

    def test_the_recorders_own_work_is_not_counted(self):
        capture = MegaFoldCapture(Tree()).install()
        try:
            self.step(capture.model, torch.randn(3, 5, 8))
        finally:
            capture.remove()
        # chunk runs once in the model; the swiglu recorder chunks again, quietly.
        self.assertEqual(capture.census["chunk"], 1)

    def test_a_checkpoint_recompute_is_not_a_training_call(self):
        capture = MegaFoldCapture(Tree(use_checkpoint=True)).install()
        try:
            self.step(capture.model, torch.randn(3, 5, 8))
        finally:
            capture.remove()
        self.assertEqual(len(capture.calls), 8)

    def test_no_grad_calls_are_not_training_calls(self):
        capture = MegaFoldCapture(Tree()).install()
        try:
            with torch.no_grad():
                capture.model(torch.randn(3, 5, 8))
        finally:
            capture.remove()
        self.assertEqual(capture.calls, [])

    def test_remove_leaves_nothing_behind(self):
        tree = Tree()
        capture = MegaFoldCapture(tree).install()
        capture.remove()
        self.step(tree, torch.randn(3, 5, 8))
        self.assertEqual(capture.calls, [])
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in tree.modules()))

    def test_an_old_snapshot_cannot_record_a_corpus(self):
        with mock.patch("evograd.benchmark.topdown.load_snapshot",
                        lambda _n: {"schema_version": "evograd-megafold-snapshot/1"}):
            with self.assertRaisesRegex(RuntimeError, "regenerate"):
                MegaFoldCapture(Tree(), cases=()).install()


class Primitives(unittest.TestCase):
    """Each call is filed under the task its parameters make it, with autocast's casts."""

    def bound(self, func, *args, **kwargs):
        from evograd.verification.integrations.megafold.capture import PRIMITIVES

        return PRIMITIVES[func](args, kwargs)

    def test_a_layernorm_by_its_parameters(self):
        x, w, b = torch.randn(2, 3, 8), torch.ones(8), torch.zeros(8)
        self.assertEqual(self.bound("layer_norm", x, (8,), w, b)[0], "layernorm")
        self.assertEqual(self.bound("layer_norm", x, (8,), w, None)[0], "layernorm_no_bias")
        self.assertEqual(self.bound("layer_norm", x, (8,), None, None)[0], "layernorm_no_affine")
        self.assertIsNone(self.bound("layer_norm", x, (3, 8), None, None))

    def test_autocast_casts_are_made(self):
        x, w = torch.randn(4, 8), torch.randn(5, 8)
        with mock.patch.object(S, "_autocast_dtype", lambda _t: torch.bfloat16):
            task, values = self.bound("linear", x, w)
            self.assertEqual((task, values["x"].dtype, values["weight"].dtype),
                             ("linear_no_bias", torch.bfloat16, torch.bfloat16))
            _, values = self.bound("layer_norm", x.bfloat16(), (8,), torch.ones(8), torch.zeros(8))
            self.assertEqual(values["x"].dtype, torch.float32)
            _, values = self.bound("softmax", x.bfloat16(), -1)
            self.assertEqual(values["x"].dtype, torch.float32)

    def test_a_biased_gemm_that_leaves_the_fused_path_stays_in_the_inventory(self):
        w, b = torch.randn(5, 8), torch.randn(5)
        self.assertEqual(self.bound("linear", torch.randn(4, 8), w, b)[0], "linear")
        self.assertEqual(self.bound("linear", torch.randn(2, 4, 8), w, b)[0], "linear")
        self.assertIsNone(self.bound("linear", torch.randn(1, 2, 4, 8), w, b))
        self.assertIsNone(self.bound("linear", torch.randn(8, 2, 4).permute(2, 1, 0), w, b))

    def test_only_the_last_axis_is_a_task(self):
        self.assertIsNone(self.bound("softmax", torch.randn(3, 4), 0))
        self.assertIsNone(self.bound("glu", torch.randn(4, 6), 0))
        self.assertIsNone(self.bound("sigmoid", torch.tensor(1.0)))


class TheBindings(unittest.TestCase):
    def test_a_task_the_snapshot_lacks_contributes_nothing(self):
        from evograd.benchmark import get_task
        from evograd.benchmark.topdown.megafold.levels.level1.manifest import OBSERVED_BINDINGS

        by_task = {b.task: b for b in OBSERVED_BINDINGS}
        with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda _n: {"level1": {}}):
            self.assertEqual(by_task["sigmoid"].workloads(get_task("sigmoid")), ())

    def test_the_harvest_expects_every_binding(self):
        from evograd.benchmark.topdown.megafold.harvest.harvest import LEVEL1
        from evograd.benchmark.topdown.megafold.levels.level1.manifest import OPERATORS

        self.assertEqual(set(OPERATORS), LEVEL1)


class Roles(unittest.TestCase):
    def test_a_role_ignores_indices_and_stacks(self):
        self.assertEqual(module_role("pairformer.layers.3.1.fn.attn.to_q"), "attn.to_q")
        self.assertEqual(module_role("msa_module.layers.0.3.tri_mult_outgoing.to_out.0"),
                         "tri_mult_outgoing.to_out")
        self.assertEqual(module_role(""), "")


class TheCensus(unittest.TestCase):
    def test_an_unknown_name_fails_the_snapshot(self):
        from evograd.benchmark.topdown.megafold.harvest.harvest import classify

        with self.assertRaisesRegex(ValueError, "nearly_a_primitive"):
            classify({"nearly_a_primitive": 3}, set())

    def test_every_name_lands_in_one_place(self):
        from evograd.benchmark.topdown.megafold.harvest.harvest import classify

        out = classify({"layer_norm": 2, "einsum": 9, "reshape": 4, "__add__": 7}, {"layer_norm"})
        self.assertEqual(out["bound"], {"layer_norm": 2})
        self.assertEqual(out["unbound"]["einsum"]["calls"], 9)
        self.assertEqual(out["scaffolding"], {"__add__": 7, "reshape": 4})

    def test_a_configuration_missing_a_dim_fails_before_the_snapshot_is_written(self):
        from evograd.benchmark.topdown.megafold.harvest.harvest import _require_declared_dims

        with self.assertRaisesRegex(ValueError, "glu declares"):
            _require_declared_dims({"glu": {"configurations": [{"dims": {"rows": 3, "width": 8}}]}})
        _require_declared_dims({"glu": {"configurations": [{"dims": {"rows": 3, "width": 8, "cols": 4}}]}})


class Dims(unittest.TestCase):
    def test_glu_reads_the_dim_no_argument_carries(self):
        from evograd.benchmark import get_task

        dims, dtype = _dims(get_task("glu"), {"x": torch.zeros(3, 8, dtype=torch.bfloat16)})
        self.assertEqual(dims, {"rows": 3, "width": 8, "cols": 4})
        self.assertEqual(dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
