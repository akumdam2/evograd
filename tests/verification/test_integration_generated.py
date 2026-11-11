"""DiT-S/2 and DLRM real-payload capture, on tiny generated models on CPU.

The cases come from the harvest's own snapshot grouping of the tiny model (one
role per configuration), read the way the task declarations read a tracked
snapshot. Instrumentation does not change the step (loss, primary output,
every gradient), every call maps to exactly one case, payloads round-trip in
each task's ABI, each task's forward reproduces every live output bit for bit,
DiT's residual cotangents are the gradients from outside each call, and Gate 1
and operator Gate 2 judge a linear candidate on the recorded payloads. The
tier-3 workload installs every DiT site at its calls per step and routes the
two residual configurations apart.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

TINY = {
    "torchrec_dlrm": dict(batch_size=4, arch_overrides=dict(
        embedding_dim=8, num_embeddings_per_feature=[50] * 26, dense_arch_layer_sizes=[16, 8],
        over_arch_layer_sizes=[32, 16, 1])),
    "dit_s_2": dict(batch_size=2, arch_overrides=dict(
        input_size=8, hidden_size=48, depth=2, num_heads=2, num_classes=10, diffusion_steps=50)),
}


def linear_pair(scale: float):
    """A biased linear pair whose weight gradient is ``scale`` times the true one."""
    import types

    def forward_with_saved(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias), (x, weight)

    def backward_from_saved(dy, saved):
        x, weight = saved
        return dy @ weight, (dy.t() @ x) * scale, dy.sum(0)

    return types.SimpleNamespace(__name__=f"linear_x{scale}", forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved)


@contextlib.contextmanager
def tracked(payload):
    """``payload`` as the tracked snapshot of its model, wherever one is read."""
    from unittest import mock

    patches = [mock.patch(f"{where}.{name}", value)
               for where in ("evograd.benchmark.topdown",
                             "evograd.benchmark.topdown.common.generated.contracts")
               for name, value in (("has_snapshot", lambda model: True),
                                   ("load_snapshot", lambda model: payload))]
    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield


def tiny_cases(spec):
    """The tiny model's cases: Level-2 workloads as its declarations read them
    (recorded strides and static parameters), with their module paths."""
    from evograd.benchmark.topdown.common.generated.contracts import observed
    from evograd.benchmark.topdown.common.generated.harvest import collect, snapshot
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    payload = snapshot(collect(spec))
    cases = []
    for key, level in (("level1", 1), ("tasks", 2)):
        for task, entry in payload[key].items():
            with tracked(payload):
                workloads = (observed(spec.model, task) if level == 2 else
                             [Workload(dims=c["dims"], dtype=c["dtype"]) for c in entry["configurations"]])
            for c, w in zip(entry["configurations"], workloads):
                cases.append(Case(f"{task}/{c['provenance']['component']}", task, level, w,
                                  c["frequency"], tuple(c["module_paths"])))
                assert get_task(task)
    return tuple(cases), payload


def warm_start(spec, path, steps=3):
    """``path``: the tiny model's state after ``steps`` AdamW updates from its
    initialization, in the warm-start format (adaLN-Zero gates no longer zero)."""
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
    from evograd.verification import gate3

    cold = GeneratedWorkload(spec)
    model = cold.build(KernelSet(registry=cold.site_registry))
    optimizer = gate3.adamw(lr=1e-2).build(model)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        cold.loss(model, cold.batch_for(seed=step)).backward()
        optimizer.step()
    torch.save({"schema": "evograd-generated-init-state/1", "model": spec.model,
                "recipe": {"steps": steps}, "state_dict": model.state_dict()}, path)
    return str(path)


class _Tiny:
    name = ""
    #: Whether the model starts from a few updates past its initialization.
    warm = False

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.common.generated.spec import canonical
        from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations import load
        from evograd.verification.integrations.objectives import step_record
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.MODEL = load(cls.name)
        spec = canonical(cls.name).replace(device="cpu", **TINY[cls.name])
        cls.spec = spec
        cls.tmp = tempfile.TemporaryDirectory()
        cls.workload = GeneratedWorkload(spec, init_checkpoint=(
            warm_start(spec, Path(cls.tmp.name) / "warm.pt") if cls.warm else None))
        cls.cases, cls.snapshot = tiny_cases(spec)
        objective = cls.MODEL.objective.bind(cls.workload)
        primary = objective.primary
        batch = cls.workload.batch_for(seed=1)
        plain = Q.build_model(cls.workload)
        cls.plain = step_record(plain, *objective.capture_step(plain, batch), primary)
        writer = CorpusWriter(Path(cls.tmp.name) / "corpus")
        for order, seed in ((101, 1), (303, 2)):
            model = Q.build_model(cls.workload)
            recorder = cls.MODEL.live_capture(model).install()
            try:
                loss, output = objective.capture_step(model, cls.workload.batch_for(seed=seed))
            finally:
                recorder.remove()
            if order == 101:
                cls.instrumented = step_record(model, loss, output, primary)
            Q.record_step(writer, cls.MODEL.invocations(recorder, model, cls.cases),
                          {"order": order, "sha256": str(order)})
        cls.coverage = Q.coverage(cls.cases, writer.payloads)
        cls.corpus = Corpus.open(writer.finish(identity={"name": "tiny"}, splits=Q.splits(writer.payloads),
                                               coverage=cls.coverage))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_instrumentation_changes_nothing(self):
        self.assertEqual(self.plain, self.instrumented)

    def test_every_call_is_one_payload_of_its_case(self):
        self.assertTrue(self.coverage["complete"], self.coverage["problems"])
        self.assertEqual(self.coverage["orders"], ["101", "303"])

    def test_payloads_match_each_task_interface(self):
        from evograd.benchmark import get_task
        from evograd.verification.payloads.gate1 import check_payload_interface

        for record in self.corpus.payloads():
            self.assertEqual(check_payload_interface(get_task(record["task"]), record), [],
                             record["payload_id"])

    def test_model_reference_reproduces_every_live_output(self):
        from evograd.opdecl.inputs import as_output_tuple
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.tensors import tensor_digest

        for record in self.corpus.payloads():
            op = self.MODEL.reference_op(record["task"])
            values = self.corpus.load(record, device="cpu")
            with torch.no_grad():
                out = resolve_runtime_forward(op)(*[values.get(a.name, getattr(a, "default", None))
                                                    for a in op.args])
            outs = dict(zip(op.output_names, as_output_tuple(op, out)))
            for name, live in record["live_outputs"].items():
                self.assertEqual(tensor_digest(outs[name]), live["digest"],
                                 f"{record['payload_id']}:{name}")

    #: A linear case whose recorded cotangents are not all zero at this state.
    judged_linear = ""

    def test_gate1_and_operator_gate2_judge_a_linear_candidate(self):
        from evograd.verification.integrations.gates import judge_case

        case = next(c for c in self.cases if c.case_id == self.judged_linear)
        for scale, expected in ((1.0, "passed"), (1.05, "failed")):
            reports = judge_case(self.MODEL, case, linear_pair(scale), self.corpus, device="cpu",
                                 compile_backend="aot_eager")
            status = {name: r.status for name, r in reports.items()}
            self.assertEqual(status, {"gate1_payloads": expected,
                                      "gate2_operator_calibration": "calibrated",
                                      "gate2_operator": expected}, (case.case_id, scale))


class TinyDLRM(_Tiny, unittest.TestCase):
    name = "torchrec_dlrm"
    judged_linear = "linear/dense_layers"

    def test_shared_shape_linear_layers_are_one_case(self):
        linear = {c.case_id: c.frequency for c in self.cases if c.task == "linear"}
        # dense 13->16->8, over (8+26*27/2)->32->16->1: every layer a distinct shape here.
        self.assertEqual(sum(linear.values()), 5)
        bags = [c for c in self.cases if c.task == "embedding_bag"]
        self.assertEqual([c.frequency for c in bags], [26])


class TinyDiT(_Tiny, unittest.TestCase):
    name = "dit_s_2"
    judged_linear = "linear/final_linear"

    def test_level2_sites_are_recorded_per_block(self):
        # Per block one modulation, attention and MLP and two residual adds. The
        # first block's attention input has no residual add before it; the last
        # block's MLP residual feeds the final layer, whose shift and scale are
        # halves of a [B, 2H] projection (row stride 2H, not 6H): its own case.
        depth = self.spec.arch["depth"]
        level2 = {c.case_id: c.frequency for c in self.cases if c.level == 2}
        self.assertEqual(level2, {"dit_s_2_dit_modulation/blocks.modulation": depth,
                                  "dit_s_2_dit_attention/blocks.attention": depth,
                                  "dit_s_2_dit_adaln_residual/blocks.residual": 2 * depth - 1,
                                  "dit_s_2_dit_adaln_residual/final_residual": 1,
                                  "dit_s_2_dit_mlp/blocks.mlp": depth})
        h = self.spec.arch["hidden_size"]
        strides = {c.case_id.split("/")[1]: dict(c.workload.provenance.strides)
                   for c in self.cases if c.task == "dit_s_2_dit_adaln_residual"}
        self.assertEqual({k: (v["shift"], v["scale"], v["gate"]) for k, v in strides.items()},
                         {"blocks.residual": ((6 * h, 1),) * 3,
                          "final_residual": ((2 * h, 1), (2 * h, 1), (6 * h, 1))})
        # The residual stream keeps the patch embedding's channel-first layout.
        tokens = (self.spec.arch["input_size"] // self.spec.arch["patch_size"]) ** 2
        self.assertEqual({v["x"] for v in strides.values()}, {(tokens * h, 1, tokens)})
        attention = next(c for c in self.cases if c.task == "dit_s_2_dit_attention")
        self.assertEqual(attention.workload.provenance.free, {"heads": self.spec.arch["num_heads"]})

    def test_a_residual_candidate_takes_only_its_configuration(self):
        # With the workload's own snapshot the residual site has both observed
        # configurations, told apart by their strides; a candidate for the block
        # case gets those calls and the final layer's call runs the reference.
        from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
        from evograd.verification.integrations.gates import by_configuration

        with tracked(self.snapshot):
            registry = GeneratedWorkload(self.spec).site_registry
        self.assertEqual(len(registry.require("dit_adaln_residual").preflight), 2)
        block = next(c for c in self.cases if c.case_id.endswith("/blocks.residual"))
        seen = []
        site = by_configuration(self.MODEL, registry, "dit_adaln_residual",
                                [(block, lambda *args: seen.append(args) or "candidate")])
        op = self.MODEL.reference_op(block.task)
        for record in self.corpus.payloads():
            if record["task"] == block.task and record["batch"]["order"] == 101:
                out = site(*[self.corpus.load(record, device="cpu")[a.name] for a in op.args])
                self.assertEqual(out == "candidate", record["case_id"] == block.case_id)
        self.assertEqual(len(seen), block.frequency)

    def test_adaln_zero_initial_state_leaves_gated_cotangents_zero(self):
        # adaLN-Zero: every block's gates and the final layer start at zero, so at
        # the initial state no gradient reaches the blocks or the conditioning
        # path; only final_linear's output cotangent is nonzero. Payloads recorded
        # there cannot show a gradient fault in those calls.
        nonzero = {r["case_id"] for r in self.corpus.payloads()
                   if any(not s["all_zero"] for s in r["cotangent_stats"].values())}
        self.assertEqual(nonzero, {"linear/final_linear"})


class TinyWarmDiT(_Tiny, unittest.TestCase):
    """DiT a few updates past adaLN-Zero, where its blocks' cotangents are not zero."""

    name = "dit_s_2"
    warm = True
    judged_linear = "linear/final_linear"

    def test_residual_cotangents_are_the_gradient_from_outside_each_call(self):
        # Each residual call's x is the previous call's summed, so the model
        # reference's dx at a call is the previous call's recorded dsummed; the
        # last call's sum has no consumer and records zeros.
        from evograd.opdecl.oracle import oracle

        op = self.MODEL.reference_op("dit_s_2_dit_adaln_residual")
        calls = sorted((r for r in self.corpus.payloads() if r["task"] == op.name
                        and r["batch"]["order"] == 101), key=lambda r: r["call"]["index_in_step"])
        self.assertEqual(len(calls), 2 * self.spec.arch["depth"])
        values = [self.corpus.load(r, device="cpu") for r in calls]
        for before, after in zip(values, values[1:]):
            _, grads = oracle(op, after, (after["dmodulated"], after["dsummed"]))
            self.assertTrue(torch.equal(grads["dx"], before["dsummed"]))
            self.assertTrue(bool(before["dsummed"].abs().sum() > 0))
        self.assertEqual(calls[-1]["case_id"], "dit_s_2_dit_adaln_residual/final_residual")
        self.assertEqual(calls[-1]["notes"]["absent_cotangents"], ["dsummed"])
        self.assertFalse(bool(values[-1]["dsummed"].any()))

    def test_tier3_installs_every_site_at_its_calls_per_step(self):
        from evograd.evaluation.tier3.patch import KernelSet, patch
        from evograd.evaluation.tier3.workloads.common.generated.workload import DIT_SITES

        calls = {site: 0 for site in DIT_SITES}

        def counted(site, fn):
            def call(*args):
                calls[site] += 1
                return fn(*args)
            return call

        kernels = KernelSet(registry=self.workload.site_registry)
        for site, fn in DIT_SITES.items():
            kernels = patch(kernels, site, counted(site, fn))
        model = self.workload.build(kernels)
        batch = self.workload.batch_for(seed=1)
        self.workload.loss(model, batch).backward()
        self.assertEqual(calls, self.workload.expected_site_calls())
        native = self.workload.build(KernelSet(registry=self.workload.site_registry))
        self.workload.loss(native, batch).backward()
        for (name, a), b in zip(model.named_parameters(), native.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad), name)


class WarmStart(unittest.TestCase):
    """A frozen initial state replaces the initialization exactly, and after a
    few updates DiT's blocks receive gradient (they do not at the init)."""

    def test_initial_state_is_loaded_and_identified(self):
        from evograd.benchmark.topdown.common.generated.spec import canonical
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload

        spec = canonical("dit_s_2").replace(device="cpu", **TINY["dit_s_2"])
        with tempfile.TemporaryDirectory() as tmp:
            path = warm_start(spec, Path(tmp) / "warm.pt")
            model = torch.load(path, weights_only=False)["state_dict"]
            warm = GeneratedWorkload(spec, init_checkpoint=path)
            again = warm.build(KernelSet(registry=warm.site_registry))
            for (name, a), b in zip(model.items(), again.state_dict().values()):
                self.assertTrue(torch.equal(a, b), name)
            identity = warm.describe()["init_checkpoint"]
            self.assertEqual(identity["recipe"], {"steps": 3})
            self.assertEqual(len(identity["sha256"]), 64)
            again.zero_grad(set_to_none=True)
            warm.loss(again, warm.batch_for(seed=9)).backward()
            self.assertTrue(bool(again.blocks[0].fc1.weight.grad.abs().sum() > 0))


if __name__ == "__main__":
    unittest.main()
