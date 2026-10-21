"""ConvNeXt V2 and Qwen2-VL real-payload capture, on tiny random models on CPU.

The cases come from the harvest's own snapshot grouping of the tiny model.
Instrumentation does not change the step (loss, logits, every gradient),
every call maps to exactly one case, payloads round-trip in each task's ABI,
the model reference reproduces every live output bit for bit, and Gate 1 and
operator Gate 2 judge a linear candidate on the recorded payloads.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


def small_spec(model):
    """The vision test suite's tiny configurations (``tests/vision_workloads``)."""
    from evograd.benchmark.topdown.common.vision.spec import canonical

    if model == "convnextv2_tiny":
        overrides = {"depths": [1, 1, 1, 1], "hidden_sizes": [8, 16, 24, 32], "num_labels": 7}
        image, tokens = 32, 0
    else:
        overrides = {
            "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 128,
            "vision_start_token_id": 120, "vision_end_token_id": 121, "image_token_id": 122,
            "video_token_id": 123, "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
            "vision_config": {"depth": 1, "embed_dim": 32, "num_heads": 4, "hidden_size": 32},
        }
        image, tokens = 28, 8
    return canonical(model).replace(device="cpu", dtype="float32", batch_size=1, image_size=image,
                                    seq_len=tokens, arch_overrides=overrides)


def linear_pair(scale: float):
    import types

    def forward_with_saved(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias), (x, weight)

    def backward_from_saved(dy, saved):
        x, weight = saved
        return dy @ weight, (dy.t() @ x) * scale, dy.sum(0)

    return types.SimpleNamespace(__name__=f"linear_x{scale}", forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved)


def tiny_cases(spec):
    from evograd.benchmark.topdown.common.vision.harvest import collect, snapshot
    from evograd.opdecl.activity import Provenance, Workload
    from evograd.verification.integrations.base import Case

    payload = snapshot(collect(spec))
    cases = []
    for key, level in (("level1", 1), ("tasks", 2)):
        for task, entry in payload[key].items():
            for c in entry["configurations"]:
                # Like the observed suites: the strides of non-contiguous arguments.
                strides = tuple((name, tuple(stride)) for name, stride, meta in
                                zip(c["strides"], c["strides"].values(), c["inputs"])
                                if tuple(stride) != tuple(_contiguous(meta["shape"])))
                cases.append(Case(f"{task}/{c['provenance']['component']}", task, level,
                                  Workload(dims=c["dims"], dtype=c["dtype"],
                                           provenance=Provenance(model="tiny",
                                                                 component=c["provenance"]["component"],
                                                                 free=c["provenance"]["free"],
                                                                 strides=strides)),
                                  c["frequency"], tuple(c["module_paths"])))
    return tuple(cases)


def _contiguous(shape):
    return torch.empty(tuple(shape), device="meta").stride()


class _Tiny:
    name = ""

    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.workloads.common.vision.workload import VisionWorkload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations import load
        from evograd.verification.integrations.objectives import step_record
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.MODEL = load(cls.name)
        spec = small_spec(cls.name)
        cls.workload = VisionWorkload(spec)
        cls.cases = tiny_cases(spec)
        objective = cls.MODEL.objective.bind(cls.workload)

        def build():
            model = Q.build_model(cls.workload)
            g = torch.Generator().manual_seed(11)     # move zero-initialized parameters off zero
            with torch.no_grad():
                for _, p in sorted(model.named_parameters()):
                    if not p.any():
                        p.copy_(0.1 * torch.randn(p.shape, generator=g))
            return model

        plain = build()
        cls.plain = step_record(plain, *objective.capture_step(plain, cls.workload.batch_for(seed=1)),
                                objective.primary)
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        for order, seed in ((101, 1), (303, 2)):
            model = build()
            recorder = cls.MODEL.live_capture(model).install()
            try:
                loss, output = objective.capture_step(model, cls.workload.batch_for(seed=seed))
            finally:
                recorder.remove()
            if order == 101:
                cls.instrumented = step_record(model, loss, output, objective.primary)
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

    def test_gate1_and_operator_gate2_judge_a_linear_candidate(self):
        from evograd.verification.integrations.gates import judge_case

        case = max((c for c in self.cases if c.task == "linear"), key=lambda c: c.frequency)
        for scale, expected in ((1.0, "passed"), (1.05, "failed")):
            reports = judge_case(self.MODEL, case, linear_pair(scale), self.corpus, device="cpu",
                                 compile_backend="aot_eager")
            status = {name: r.status for name, r in reports.items()}
            self.assertEqual(status, {"gate1_generated": expected, "gate1_payloads": expected,
                                      "gate2_operator_calibration": "calibrated",
                                      "gate2_operator": expected}, (case.case_id, scale))


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyConvNeXt(_Tiny, unittest.TestCase):
    name = "convnextv2_tiny"

    def test_stage_sites_are_four_cases_of_one_task(self):
        level2 = [c for c in self.cases if c.level == 2]
        self.assertEqual(len(level2), 4)
        self.assertEqual({c.task for c in level2}, {"convnextv2_tiny_convnext_mlp"})

    def test_each_stage_case_has_its_own_site(self):
        from evograd.verification.integrations.gates import site_for_case

        sites = {site_for_case(self.workload.site_registry, c) for c in self.cases if c.level == 2}
        self.assertEqual(sites, {f"convnext_mlp_stage{i}" for i in range(4)})


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyQwen2VL(_Tiny, unittest.TestCase):
    name = "qwen2_vl_2b"


if __name__ == "__main__":
    unittest.main()
