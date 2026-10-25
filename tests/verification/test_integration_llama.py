"""Llama-3.2-1B real-payload capture, on a tiny random Llama on CPU.

The GPU capture records the pinned 1B model on real TRAIN batches. This pins
the machinery on a 2-layer model: instrumentation does not change the step,
every call maps to exactly one case by shape, payloads round-trip in each
task's ABI, the model reference reproduces every live output bit for bit, and
Gate 1 and operator Gate 2 judge a candidate on the recorded payloads. The
real case table is checked against the registry and the snapshot.
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

B, T, H, I, HQ, HK, D, V, LAYERS = 2, 16, 64, 128, 4, 2, 16, 80, 2
ARCH = {"num_hidden_layers": LAYERS, "hidden_size": H, "intermediate_size": I,
        "num_attention_heads": HQ, "num_key_value_heads": HK, "head_dim": D, "vocab_size": V,
        "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1}


def tiny_cases():
    """The tiny model's cases; q_proj and o_proj share one, as in the 1B model."""
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    n, bf = B * T, "bfloat16"
    rows = [
        ("linear_no_bias", "q_o", {"M": n, "K": H, "N": HQ * D}, bf, 2 * LAYERS),
        ("linear_no_bias", "kv", {"M": n, "K": H, "N": HK * D}, bf, 2 * LAYERS),
        ("linear_no_bias", "gate_up", {"M": n, "K": H, "N": I}, bf, 2 * LAYERS),
        ("linear_no_bias", "down", {"M": n, "K": I, "N": H}, bf, LAYERS),
        ("linear_no_bias", "lm_head", {"M": n, "K": H, "N": V}, bf, 1),
        ("rmsnorm", "hidden", {"rows": n, "hidden": H}, bf, 2 * LAYERS + 1),
        ("rope", "q", {"B": B, "n_heads": HQ, "T": T, "head_dim": D}, bf, LAYERS),
        ("rope", "k", {"B": B, "n_heads": HK, "T": T, "head_dim": D}, bf, LAYERS),
        ("swiglu", "act", {"rows": n, "cols": I}, bf, LAYERS),
        ("causal_gqa_attention", "sdpa", {"B": B, "HQ": HQ, "HK": HK, "T": T, "D": D}, bf, LAYERS),
        ("cross_entropy", "loss", {"rows": n, "cols": V}, "float32", 1),
        ("llama3_qkv_rope", "qkv", {"B": B, "T": T, "H": H, "HQ": HQ, "HK": HK, "D": D,
                                    "QO": HQ * D, "KVO": HK * D}, bf, LAYERS),
        ("llama3_attention", "attn", {"B": B, "T": T, "HQ": HQ, "HK": HK, "D": D, "QO": HQ * D,
                                      "H": H}, bf, LAYERS),
        ("llama3_swiglu_mlp", "mlp", {"B": B, "T": T, "H": H, "I": I}, bf, LAYERS),
        ("llama3_residual_rmsnorm", "residual", {"rows": n, "cols": H}, bf, 2 * LAYERS),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level, Workload(dims=dims, dtype=dtype),
                        freq) for task, slug, dims, dtype, freq in rows)


def linear_pair(scale: float):
    """A linear_no_bias pair whose weight gradient is ``scale`` times the true one."""
    import types

    def forward_with_saved(x, weight):
        return torch.nn.functional.linear(x, weight), (x, weight)

    def backward_from_saved(dy, saved):
        x, weight = saved
        return dy @ weight, (dy.t() @ x) * scale

    return types.SimpleNamespace(__name__=f"linear_x{scale}", forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved)


class RealCaseTable(unittest.TestCase):
    def test_observed_cases_follow_registry_and_snapshot(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.llama3_2_1b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.level == 1 for c in cases.values()), 11)
        self.assertEqual(sum(c.level == 2 for c in cases.values()), 4)
        self.assertEqual(cases["rmsnorm/rmsnorm"].frequency, 33)
        self.assertEqual(cases["llama3_residual_rmsnorm/residual_rmsnorm"].frequency, 32)
        self.assertEqual(cases["linear_no_bias/attn_out_proj"].frequency, 32)

    def test_every_case_task_is_redeclared_under_the_contract(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.llama3_2_1b import MODEL, reference

        for task in sorted({c.task for c in observed_cases(MODEL)}):
            op = reference.reference_op(task)
            self.assertEqual(op.reference_contract, reference.CONTRACT)
            self.assertIsNone(op.runtime_forward)

    def test_registered_under_both_spellings(self):
        from evograd.verification import integrations

        self.assertIs(integrations.load("llama3_2_1b"), integrations.load("llama_3_2_1b"))

    @unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
    def test_rms_norm_is_llamarmsnorm_bit_for_bit(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        from evograd.verification.integrations.llama3_2_1b import reference

        g = torch.Generator().manual_seed(3)
        norm = LlamaRMSNorm(64, eps=reference.RMS_NORM_EPS).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(64, generator=g))
        x = torch.randn(32, 64, generator=g).to(torch.bfloat16)
        r = torch.randn(32, 64, generator=g).to(torch.bfloat16)
        self.assertTrue(torch.equal(reference.rms_norm(x, norm.weight), norm(x)))
        out, summed = reference.residual_rms_norm(x, r, norm.weight)
        self.assertTrue(torch.equal(out, norm(r + x)))


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.workloads.llama3_2_1b.workload import Llama3Workload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.llama3_2_1b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.Q, cls.C = Q, C
        workload = Llama3Workload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                  arch_overrides=ARCH)

        def build():
            model = Q.build_model(workload)
            g = torch.Generator().manual_seed(11)       # a trained spread, not the unit init
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if name.endswith(("norm.weight", "layernorm.weight")):
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
            return model, recorder, output

        cls.cases = tiny_cases()
        cls.ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(7))
        plain = build()
        cls.plain = Q.step_record(plain, Q.run_step(plain, cls.ids, cls.ids.clone()))
        del plain
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        cls.model, cls.recorder, output = step(cls.ids, 101, writer)
        cls.instrumented = Q.step_record(cls.model, output)
        cls.invocations = list(C.invocations(cls.recorder, cls.model, cls.cases))
        held_out = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(8))
        step(held_out, 303, writer)
        cls.coverage = Q.coverage(cls.cases, writer.payloads)
        cls.corpus = Corpus.open(writer.finish(identity={"name": "tiny"}, splits=Q.splits(writer.payloads),
                                               coverage=cls.coverage))

    @classmethod
    def tearDownClass(cls):
        import gc

        cls.tmp.cleanup()
        for name in ("model", "recorder", "invocations", "corpus"):
            if hasattr(cls, name):
                delattr(cls, name)
        gc.collect()

    def test_instrumentation_changes_nothing(self):
        self.assertEqual(self.plain["loss_digest"], self.instrumented["loss_digest"])
        self.assertEqual(self.plain["logits_digest"], self.instrumented["logits_digest"])
        self.assertEqual(self.plain["grads"], self.instrumented["grads"])

    def test_every_call_is_one_payload_of_its_case(self):
        self.assertTrue(self.coverage["complete"], self.coverage["problems"])
        self.assertEqual(self.coverage["orders"], ["101", "303"])
        residual = [i for i in self.invocations if i["task"] == "llama3_residual_rmsnorm"]
        self.assertEqual(sorted(i["call"]["role"] for i in residual).count("final_model_norm"), 1)

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
        from evograd.verification.integrations.llama3_2_1b.reference import reference_op

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

    def test_gate1_and_operator_gate2_judge_a_candidate_on_payloads(self):
        from evograd.verification.integrations.gates import judge_case
        from evograd.verification.integrations.llama3_2_1b import MODEL

        case = next(c for c in self.cases if c.case_id == "linear_no_bias/q_o")
        for scale, expected in ((1.0, "passed"), (1.05, "failed")):
            reports = judge_case(MODEL, case, linear_pair(scale), self.corpus, device="cpu",
                                 compile_backend="aot_eager")
            status = {name: r.status for name, r in reports.items()}
            self.assertEqual(status, {"gate1_payloads": expected,
                                      "gate2_operator_calibration": "calibrated",
                                      "gate2_operator": expected}, scale)

    def test_control_candidates_are_accepted_and_faults_rejected(self):
        from evograd.verification.integrations.candidates import native_pair
        from evograd.verification.integrations.gates import judge_case
        from evograd.verification.integrations.llama3_2_1b import MODEL

        for case_id, faulty in (("linear_no_bias/q_o", 1.05), ("llama3_swiglu_mlp/mlp", 1.05)):
            case = next(c for c in self.cases if c.case_id == case_id)
            for scale, expected in ((1.0, "passed"), (faulty, "failed")):
                reports = judge_case(MODEL, case, native_pair("llama_3_2_1b", case.task, scale=scale),
                                     self.corpus, device="cpu", compile_backend="aot_eager")
                self.assertEqual(reports["gate2_operator"].status, expected, (case_id, scale))
                if scale == 1.0:
                    self.assertEqual(reports["gate1_payloads"].status, "passed", case_id)


if __name__ == "__main__":
    unittest.main()
