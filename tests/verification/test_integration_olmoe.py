"""OLMoE-1B-7B real-payload capture, on a tiny random OLMoE on CPU.

Instrumentation does not change the step (the HF loss with the router's
load-balancing term), every call maps to exactly one case -- the q and k rope
calls have identical shapes here, as in the real model, and are told apart by
role -- payloads round-trip in each task's ABI (the experts' routing indices
and weights included), the model reference reproduces every live output bit
for bit, and the integration runs Gate 1 and operator Gate 2 only.
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

B, T, H, I, HEADS, D, E, K, V, LAYERS = 2, 16, 64, 32, 4, 16, 4, 2, 512, 2


def tiny_cases():
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Provenance, Workload
    from evograd.verification.integrations.base import Case

    n, bf = B * T, "bfloat16"
    gapped = (("a", (2 * I, 1)), ("b", (2 * I, 1)))
    rows = [
        ("linear_no_bias", "qkvo", {"M": n, "K": H, "N": H}, bf, 4 * LAYERS, ()),
        ("linear_no_bias", "lm_head", {"M": n, "K": H, "N": V}, bf, 1, ()),
        ("rmsnorm", "residual_norm", {"rows": n, "hidden": H}, bf, 4 * LAYERS + 1, ()),
        ("rope", "rope", {"B": B, "n_heads": HEADS, "T": T, "head_dim": D}, bf, LAYERS, ()),
        ("rope", "rope_kv", {"B": B, "n_heads": HEADS, "T": T, "head_dim": D}, bf, LAYERS, ()),
        ("swiglu", "moe_activation", {"rows": n * K, "cols": I}, bf, LAYERS, gapped),
        ("causal_gqa_attention", "sdpa", {"B": B, "HQ": HEADS, "HK": HEADS, "T": T, "D": D}, bf,
         LAYERS, ()),
        ("cross_entropy", "logits", {"rows": n, "cols": V}, "float32", 1, ()),
        ("olmoe_qkv_norm_rope", "qkv", {"B": B, "T": T, "H": H, "HQ": HEADS, "HK": HEADS, "D": D,
                                        "QO": H, "KVO": H}, bf, LAYERS, ()),
        ("olmoe_attention", "attn", {"B": B, "T": T, "HQ": HEADS, "HK": HEADS, "D": D, "QO": H,
                                     "H": H}, bf, LAYERS, ()),
        ("olmoe_moe_experts", "experts", {"T": n, "H": H, "E": E, "G": 2 * I, "I": I, "K": K}, bf,
         LAYERS, ()),
        ("olmoe_residual_rmsnorm", "residual", {"B": B, "T": T, "H": H}, bf, LAYERS, ()),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level,
                      Workload(dims=dims, dtype=dtype,
                               provenance=Provenance(model="tiny", component=slug, strides=strides)),
                      freq) for task, slug, dims, dtype, freq, strides in rows)


class RealCaseTable(unittest.TestCase):
    def test_observed_cases_and_model_gates(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.olmoe_1b_7b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.level == 1 for c in cases.values()), 8)
        self.assertEqual(sum(c.level == 2 for c in cases.values()), 4)
        self.assertEqual(cases["rmsnorm/residual_norm"].frequency, 65)
        self.assertFalse(MODEL.model_gates)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.olmoe_1b_7b.architecture import OLMOE_1B_7B, with_layers
        from evograd.evaluation.tier3.workloads.olmoe_1b_7b.workload import OlmoeWorkload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.olmoe_1b_7b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.Q, cls.C = Q, C
        arch = {**with_layers(OLMOE_1B_7B, LAYERS), "hidden_size": H, "intermediate_size": I,
                "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "num_experts": E,
                "num_experts_per_tok": K, "vocab_size": V, "eos_token_id": None, "pad_token_id": 1,
                "max_position_embeddings": 512}
        workload = OlmoeWorkload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                 arch_overrides=arch)

        def build():
            model = Q.build_model(workload)
            g = torch.Generator().manual_seed(11)
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if name.endswith("norm.weight"):
                        p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            return model

        def step(ids, order, writer):
            model = build()
            recorder = C.LiveCapture(model).install()
            try:
                output = Q.run_step(model, ids, ids.clone())
            finally:
                recorder.remove()
            Q.record_step(writer, C.invocations(recorder, model, cls.cases),
                          {"order": order, "sha256": str(order)})
            return model, output

        cls.cases = tiny_cases()
        cls.ids = torch.randint(2, V, (B, T), generator=torch.Generator().manual_seed(7))
        plain = build()
        cls.plain = Q.step_record(plain, Q.run_step(plain, cls.ids, cls.ids.clone()))
        del plain
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        cls.model, output = step(cls.ids, 101, writer)
        cls.instrumented = Q.step_record(cls.model, output)
        step(torch.randint(2, V, (B, T), generator=torch.Generator().manual_seed(8)), 303, writer)
        cls.coverage = Q.coverage(cls.cases, writer.payloads)
        cls.corpus = Corpus.open(writer.finish(identity={"name": "tiny"}, splits=Q.splits(writer.payloads),
                                               coverage=cls.coverage))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_instrumentation_changes_nothing(self):
        self.assertEqual(self.plain["loss_digest"], self.instrumented["loss_digest"])
        self.assertEqual(self.plain["logits_digest"], self.instrumented["logits_digest"])
        self.assertEqual(self.plain["grads"], self.instrumented["grads"])

    def test_every_call_is_one_payload_of_its_case(self):
        self.assertTrue(self.coverage["complete"], self.coverage["problems"])
        roles = {r["case_id"] for r in self.corpus.payloads() if r["task"] == "rope"}
        self.assertEqual(roles, {"rope/rope", "rope/rope_kv"})

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
        from evograd.verification.integrations.olmoe_1b_7b.reference import reference_op

        for record in self.corpus.payloads():
            op = reference_op(record["task"])
            values = self.corpus.load(record, device="cpu")
            with torch.no_grad():
                out = resolve_runtime_forward(op)(*[values.get(a.name, getattr(a, "default", None))
                                                    for a in op.args])
            outs = dict(zip(op.output_names, as_output_tuple(op, out)))
            for name, live in record["live_outputs"].items():
                self.assertEqual(tensor_digest(outs[name]), live["digest"],
                                 f"{record['payload_id']}:{name}")


if __name__ == "__main__":
    unittest.main()
