"""A toy model integration that is not a language model, for CPU tests.

A two-matrix classifier whose activation is a ``swiglu`` model site, wired as
a :class:`~evograd.verification.integrations.base.Model` with the
:class:`~evograd.verification.integrations.objectives.PrimaryOutput`
objective: capture records every site call as a payload, and the gates run
exactly as for a real model. The one case is treated as Level 2 so that the
model-scope gates apply.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest import mock

import torch

from tests.verification.test_objective_workloads import ROWS, COLS, ToyWorkload as _Base

CASE_ID = "swiglu/act"

CANDIDATE = '''
import torch

SCALE = {scale!r}


def forward_with_saved(a, b):
    return torch.nn.functional.silu(a) * b, (a, b)


def backward_from_saved(dc, saved):
    a, b = saved
    s = torch.sigmoid(a)
    return dc * b * (s * (1 + a * (1 - s))) * SCALE, dc * torch.nn.functional.silu(a)
'''


class Batches:
    def __init__(self, workload, order):
        self.workload, self.order = workload, int(order)

    def batch(self, index):
        return self.workload.batch_for_order(self.order, int(index))

    def describe(self):
        return {"source": "toy", "order": self.order}


class ToyWorkload(_Base):
    """The tier-3 toy plus the real-data verification interface."""

    def batch_for_order(self, order: int, index: int):
        g = torch.Generator().manual_seed(order * 1000 + index)
        return {"x": torch.randn(ROWS, 12, generator=g),
                "labels": torch.randint(0, 5, (ROWS,), generator=g)}

    def train_batches(self, order):
        return Batches(self, order)

    def batch_for(self, *, seed):
        return self.batch_for_order(int(seed), 0)

    def data_identity(self):
        return {"source": "toy generator"}

    def data_recipe(self):
        return {"rows": ROWS}


class ToyCapture:
    """Records every call of the model's ``act`` site (a pass-through)."""

    def __init__(self, model):
        self.model, self.calls, self._original = model, [], None

    def install(self):
        self._original = original = self.model.act

        def act(a, b):
            c = original(a, b)
            record = {"a": a.detach(), "b": b.detach(), "c": c.detach()}
            if c.requires_grad:
                c.register_hook(lambda g, r=record: r.__setitem__("dc", g.detach().clone()))
            self.calls.append(record)
            return c

        self.model.act = act
        return self

    def remove(self):
        self.model.act = self._original


def invocations(recorder, model, cases):
    from evograd.verification.integrations.base import invocation

    for i, r in enumerate(recorder.calls):
        yield invocation(cases, "swiglu", 2, {"path": f"act#{i}", "layer": None, "role": "act",
                                              "kind": "site:act"},
                         {"a": r["a"], "b": r["b"]}, {"dc": r["dc"]}, {"c": r["c"]})


def train_batches(workload, orders):
    from evograd.verification.integrations.objectives import batch_digest

    batches = {order: workload.train_batches(order).batch(0) for order in orders}
    return batches, {"batches": {str(o): {"batch_sha256": batch_digest(b)} for o, b in batches.items()}}


def cases():
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    return (Case(CASE_ID, "swiglu", 2, Workload(dims={"rows": ROWS, "cols": COLS},
                                                 dtype="float32"), 1),)


def model(gate3_steps: int = 4):
    from evograd.benchmark import get_task
    from evograd.verification.integrations.base import Model
    from evograd.verification.integrations.objectives import PrimaryOutput

    workload = ToyWorkload()
    return Model(name="toy", suite="toy_observed", workload_config={"weights": "toy"},
                 load_workload=lambda: workload, train_batches=train_batches,
                 live_capture=ToyCapture, invocations=invocations, reference_op=get_task,
                 contract="toy-reference/1", source_identity=lambda: {"toy": 1},
                 objective=PrimaryOutput(ToyWorkload.primary_output_name,
                                         ToyWorkload.objective_definition),
                 gate3_steps=gate3_steps)


@contextlib.contextmanager
def toy_environment():
    """The toy's cases and snapshot in place of a registry suite, and the
    deterministic settings ``configure()`` sets, restored afterwards."""
    previous = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "precision": torch.get_float32_matmul_precision(),
                "threads": torch.get_num_threads()}
    env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        with mock.patch("evograd.verification.integrations.base.observed_cases",
                        lambda m: cases()), \
                mock.patch("evograd.benchmark.topdown.load_snapshot",
                           lambda name: {"snapshot_hash": "toy"}):
            yield SimpleNamespace(cases=cases())
    finally:
        torch.use_deterministic_algorithms(previous["deterministic"], warn_only=previous["warn_only"])
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous["tf32"]
        torch.set_float32_matmul_precision(previous["precision"])
        torch.set_num_threads(previous["threads"])
        if env is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = env
