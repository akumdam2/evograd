"""Qwen3.5-4B real-payload capture, on a tiny random Qwen3.5 on CPU.

Eight layers in the published pattern (six Gated DeltaNet, two gated
attention), two data orders. Pins: instrumentation changes nothing, every
call maps to exactly one case by shape, payloads follow each task's ABI, and
the model reference reproduces every live output bit for bit. The real case
table is checked against the registry and the snapshot.
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

T, H, I, HQ, HK, D, R, V = 40, 64, 96, 4, 2, 32, 8, 512      # attention: head_dim 32, rotary 8
HV, DK, C, VD = 4, 16, 128, 64                                # GDN: 4 value heads of 16, conv 128
GDN_LAYERS, ATTN_LAYERS = 6, 2


def tiny_cases():
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    bf, layers = "bfloat16", GDN_LAYERS + ATTN_LAYERS
    rows = [
        ("linear_no_bias", "gate_up", {"M": T, "K": H, "N": I}, bf, 2 * layers),
        ("linear_no_bias", "down", {"M": T, "K": I, "N": H}, bf, layers),
        ("linear_no_bias", "in_qkv", {"M": T, "K": H, "N": C}, bf, GDN_LAYERS),
        ("linear_no_bias", "q_proj", {"M": T, "K": H, "N": 2 * HQ * D}, bf, ATTN_LAYERS),
        # in_proj_z, out_proj, k_proj and v_proj share one configuration here
        ("linear_no_bias", "square", {"M": T, "K": H, "N": H}, bf, 2 * GDN_LAYERS + 2 * ATTN_LAYERS),
        ("linear_no_bias", "in_ba", {"M": T, "K": H, "N": HV}, bf, 2 * GDN_LAYERS),
        ("linear_no_bias", "o_proj", {"M": T, "K": HQ * D, "N": H}, bf, ATTN_LAYERS),
        ("linear_no_bias", "lm_head", {"M": T, "K": H, "N": V}, bf, 1),
        ("gemma_rmsnorm", "residual", {"rows": T, "hidden": H}, bf, 2 * layers + 1),
        ("gemma_rmsnorm", "q_norm", {"rows": T * HQ, "hidden": D}, bf, ATTN_LAYERS),
        ("gemma_rmsnorm", "k_norm", {"rows": T * HK, "hidden": D}, bf, ATTN_LAYERS),
        ("gated_rmsnorm", "norm", {"rows": T * HV, "hidden": DK}, bf, GDN_LAYERS),
        ("causal_conv1d", "conv", {"B": 1, "C": C, "T": T, "K": 4}, bf, GDN_LAYERS),
        ("l2norm", "qk", {"rows": T * HV, "cols": DK}, bf, 2 * GDN_LAYERS),
        ("gdn_gating", "gating", {"rows": T, "heads": HV}, bf, GDN_LAYERS),
        ("partial_rope", "q", {"B": 1, "H": HQ, "T": T, "D": D, "R": R}, bf, ATTN_LAYERS),
        ("partial_rope", "k", {"B": 1, "H": HK, "T": T, "D": D, "R": R}, bf, ATTN_LAYERS),
        ("causal_gqa_attention", "sdpa", {"B": 1, "HQ": HQ, "HK": HK, "T": T, "D": D}, bf, ATTN_LAYERS),
        ("swiglu", "act", {"rows": T, "cols": I}, bf, layers),
        ("cross_entropy", "loss", {"rows": T, "cols": V}, "float32", 1),
        ("qwen3_5_gdn_in_proj", "in_proj", {"B": 1, "T": T, "H": H, "C": C, "VD": VD, "HV": HV,
                                            "K": 4, "D": DK}, bf, GDN_LAYERS),
        ("qwen3_5_gated_delta_rule", "delta", {"B": 1, "T": T, "HV": HV, "D": DK}, bf, GDN_LAYERS),
        ("qwen3_5_gdn_norm_out", "norm_out", {"B": 1, "T": T, "HV": HV, "D": DK, "VD": VD, "H": H},
         bf, GDN_LAYERS),
        ("qwen3_5_qkv_norm_partial_rope", "qkv", {"B": 1, "T": T, "H": H, "HQ": HQ, "HK": HK, "D": D,
                                                  "R": R, "QG": 2 * HQ * D, "KVO": HK * D,
                                                  "QO": HQ * D}, bf, ATTN_LAYERS),
        ("qwen3_5_gated_attention", "attn", {"B": 1, "T": T, "HQ": HQ, "HK": HK, "D": D, "QO": HQ * D,
                                             "H": H}, bf, ATTN_LAYERS),
        ("qwen3_5_residual_rmsnorm", "residual", {"B": 1, "T": T, "H": H}, bf, layers),
        ("qwen3_5_swiglu_mlp", "mlp", {"B": 1, "T": T, "H": H, "I": I}, bf, layers),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level, Workload(dims=dims, dtype=dtype),
                      freq) for task, slug, dims, dtype, freq in rows)


class RealCaseTable(unittest.TestCase):
    def test_observed_cases_follow_registry_and_snapshot(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_5_4b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.frequency for c in cases.values() if c.level == 1), 507)
        self.assertEqual(sum(c.frequency for c in cases.values() if c.level == 2), 152)
        # The residual site's own calls, not its anchor norm's 65.
        self.assertEqual(cases["qwen3_5_residual_rmsnorm/residual_rmsnorm"].frequency, 32)

    def test_gate3_horizon_is_per_model(self):
        from evograd.verification.integrations.qwen3_0_6b import MODEL as QWEN3
        from evograd.verification.integrations.qwen3_5_4b import MODEL

        self.assertEqual((QWEN3.gate3_steps, MODEL.gate3_steps), (1000, 200))

    def test_every_case_task_is_redeclared_under_the_contract(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_5_4b import MODEL, reference

        for task in sorted({c.task for c in observed_cases(MODEL)}):
            op = reference.reference_op(task)
            self.assertEqual(op.reference_contract, reference.CONTRACT)
            self.assertIsNone(op.runtime_forward)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
        from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.qwen3_5_4b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter
        from tests.qwen3_5_4b._tiny import TINY_OVERRIDES

        cls.Q, cls.C = Q, C
        workload = Qwen3_5Workload(device="cpu", dtype="bfloat16", batch_size=1, seq_len=T,
                                   arch_overrides={**with_layers(QWEN3_5_4B, 8), **TINY_OVERRIDES})

        def build():
            model = Q.build_model(workload)
            g = torch.Generator().manual_seed(11)       # a trained spread, not the zero init
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if name.endswith("norm.weight"):
                        p.copy_(0.1 * torch.randn(p.shape, generator=g))
            return model

        def step(ids, order, writer=None):
            model = build()
            recorder = C.LiveCapture(model).install()
            try:
                output = Q.run_step(model, ids, ids.clone())
            finally:
                recorder.remove()
            if writer is not None:
                Q.record_step(writer, C.invocations(recorder, model, cls.cases),
                              {"order": order, "sha256": str(order)})
            return model, recorder, output

        cls.cases = tiny_cases()
        cls.ids = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(7))
        plain = build()
        cls.plain = Q.step_record(plain, Q.run_step(plain, cls.ids, cls.ids.clone()))
        del plain
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        cls.model, cls.recorder, output = step(cls.ids, 101, writer)
        cls.instrumented = Q.step_record(cls.model, output)
        cls.invocations = list(C.invocations(cls.recorder, cls.model, cls.cases))
        held_out = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(8))
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
        self.assertEqual(len(self.invocations), sum(c.frequency for c in self.cases))

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
        from evograd.verification.integrations.qwen3_5_4b.reference import reference_op

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

    def test_gate_and_residual_conventions(self):
        attn = next(r for r in self.corpus.payloads() if r["task"] == "qwen3_5_gated_attention")
        qkv = next(r for r in self.corpus.payloads()
                   if r["task"] == "qwen3_5_qkv_norm_partial_rope" and r["call"]["layer"] == attn["call"]["layer"]
                   and r["batch"] == attn["batch"])
        from evograd.verification.common.tensors import tensor_digest

        loaded_attn = self.corpus.load(attn, device="cpu")
        loaded_qkv = self.corpus.load(qkv, device="cpu")
        self.assertEqual(attn["args"]["gate"]["shape"], [1, T, HQ * D])
        # The gate the attention consumes is the gate the QKV boundary produces.
        self.assertEqual(tensor_digest(loaded_attn["gate"]), qkv["live_outputs"]["gate"]["digest"])
        self.assertTrue(loaded_qkv["dgate"].is_contiguous())
        residual = [r for r in self.corpus.payloads() if r["task"] == "qwen3_5_residual_rmsnorm"]
        self.assertEqual(len(residual), 2 * (GDN_LAYERS + ATTN_LAYERS))
        self.assertTrue(all(not r["cotangent_stats"]["dsummed"].get("all_zero") for r in residual))


if __name__ == "__main__":
    unittest.main()
