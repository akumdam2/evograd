"""Which harvested invocation the Level-2 calibration builds, for every model.

The dispatch used to be a lookup by Qwen3-0.6B's task names (plus "the first
task is q/k/v"), so Llama's attention and MLP harvested cases -- and any other
model's -- were silently skipped. Each model now states, per task, which
captured boundary its invocation is rebuilt from; a model without a captured
layer names a hook instead.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from evograd.evaluation.workloads.common import level2_calibrate as L2
from evograd.evaluation.workloads.llama3_2_1b.descriptor import WORKLOAD as LLAMA
from evograd.evaluation.workloads.qwen3_0_6b.descriptor import WORKLOAD as QWEN3


class _Op:
    def __init__(self, name):
        self.name = name


class TestHarvestedDispatch(unittest.TestCase):
    def test_every_task_names_its_boundary(self):
        for descriptor in (QWEN3, LLAMA):
            with self.subTest(model=descriptor.name):
                self.assertEqual(set(descriptor.harvested_tasks), set(descriptor.level2_tasks))
                self.assertEqual(sorted(descriptor.harvested_tasks.values()),
                                 ["attention", "mlp", "qkv"])

    def test_each_task_reaches_its_builder(self):
        for descriptor in (QWEN3, LLAMA):
            for task, role in descriptor.harvested_tasks.items():
                with self.subTest(task=task):
                    reached = []

                    def fake(op, artifact, *, device, descriptor, _role=role):
                        reached.append(_role)
                        raise FileNotFoundError("no capture in this test")

                    with mock.patch.object(L2, f"_harvested_{role}", fake):
                        record = L2._harvested_case(_Op(task), Path("missing.pt"), device="cpu",
                                                    repeats=1, descriptor=descriptor)
                    self.assertEqual(reached, [role])
                    self.assertIn("unavailable", record["label"])

    def test_a_task_without_a_boundary_has_no_case(self):
        self.assertIsNone(L2._harvested_case(_Op("not_a_task"), Path("x.pt"), device="cpu",
                                             repeats=1, descriptor=LLAMA))

    def test_a_hook_replaces_the_artifact(self):
        from dataclasses import replace

        import torch

        from evograd.opdecl import Active, Workload, declare_op

        op = declare_op(name="toy_scale", forward="tests.test_level2_calibrate_harvested:_scale",
                        dims=("N",), args=(Active("x", "[N]"),), output=Active("y", "[N]"),
                        forward_semantics="", backward_semantics="",
                        correctness=(Workload(dims={"N": 4}, dtype="float32"),),
                        tolerances={"float32": (1e-6, 1e-6)})
        global _HOOK_RESULT
        workload = op.correctness[0]
        _HOOK_RESULT = ({"x": torch.ones(4), "dy": torch.ones(4)}, workload, "derived layer-0")
        descriptor = replace(LLAMA, harvested_invocation=f"{__name__}:_hook")
        record = L2._harvested_case(op, None, device="cpu", repeats=1, descriptor=descriptor)
        self.assertEqual(record["label"], "derived layer-0")
        self.assertTrue(record["ok"])
        self.assertEqual(record["source"], f"{__name__}:_hook")


_HOOK_RESULT = None


def _hook(op, device):
    return _HOOK_RESULT


def _scale(x):
    return x * 2.0


if __name__ == "__main__":
    unittest.main()
