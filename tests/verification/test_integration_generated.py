"""DiT-S/2 and DLRM real-payload capture, on tiny generated models on CPU.

The cases come from the harvest's own snapshot grouping of the tiny model (one
role per configuration). Instrumentation does not change the step (loss,
primary output, every gradient), every call maps to exactly one case,
payloads round-trip in each task's ABI, each task's forward reproduces every
live output bit for bit, and Gate 1 and operator Gate 2 judge a linear
candidate on the recorded payloads.
"""

from __future__ import annotations

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


def tiny_cases(spec):
    from evograd.benchmark.topdown.common.generated.harvest import collect, snapshot
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    payload = snapshot(collect(spec))
    cases = []
    for key, level in (("level1", 1), ("tasks", 2)):
        for task, entry in payload[key].items():
            for c in entry["configurations"]:
                cases.append(Case(f"{task}/{c['provenance']['component']}", task, level,
                                  Workload(dims=c["dims"], dtype=c["dtype"]), c["frequency"]))
                assert get_task(task)
    return tuple(cases)


class _Tiny:
    name = ""

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
        cls.workload = GeneratedWorkload(spec)
        cls.cases = tiny_cases(spec)
        objective = cls.MODEL.objective.bind(cls.workload)
        primary = objective.primary
        batch = cls.workload.batch_for(seed=1)
        plain = Q.build_model(cls.workload)
        cls.plain = step_record(plain, *objective.capture_step(plain, batch), primary)
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
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
        level2 = {c.case_id: c.frequency for c in self.cases if c.level == 2}
        self.assertEqual(sorted(level2.values()), [2, 2])

    def test_adaln_zero_initial_state_leaves_gated_cotangents_zero(self):
        # adaLN-Zero: every block's gates and the final layer start at zero, so at
        # the initial state no gradient reaches the blocks or the conditioning
        # path; only final_linear's output cotangent is nonzero. Payloads recorded
        # there cannot show a gradient fault in those calls.
        nonzero = {r["case_id"] for r in self.corpus.payloads()
                   if any(not s["all_zero"] for s in r["cotangent_stats"].values())}
        self.assertEqual(nonzero, {"linear/final_linear"})


if __name__ == "__main__":
    unittest.main()


class WarmStart(unittest.TestCase):
    """A frozen initial state replaces the initialization exactly, and after a
    few updates DiT's blocks receive gradient (they do not at the init)."""

    def test_initial_state_is_loaded_and_identified(self):
        from evograd.benchmark.topdown.common.generated.spec import canonical
        from evograd.evaluation.tier3.patch import KernelSet
        from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
        from evograd.verification import gate3

        spec = canonical("dit_s_2").replace(device="cpu", **TINY["dit_s_2"])
        cold = GeneratedWorkload(spec)
        model = cold.build(KernelSet(registry=cold.site_registry))
        optimizer = gate3.adamw(lr=1e-2).build(model)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            cold.loss(model, cold.batch_for(seed=step)).backward()
            optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warm.pt"
            torch.save({"schema": "evograd-generated-init-state/1", "model": "dit_s_2",
                        "recipe": {"steps": 3}, "state_dict": model.state_dict()}, path)
            warm = GeneratedWorkload(spec, init_checkpoint=str(path))
            again = warm.build(KernelSet(registry=warm.site_registry))
            for (name, a), b in zip(model.state_dict().items(), again.state_dict().values()):
                self.assertTrue(torch.equal(a, b), name)
            identity = warm.describe()["init_checkpoint"]
            self.assertEqual(identity["recipe"], {"steps": 3})
            self.assertEqual(len(identity["sha256"]), 64)
            again.zero_grad(set_to_none=True)
            warm.loss(again, warm.batch_for(seed=9)).backward()
            self.assertTrue(bool(again.blocks[0].fc1.weight.grad.abs().sum() > 0))
