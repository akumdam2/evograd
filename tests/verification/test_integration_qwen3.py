"""Qwen3-0.6B real-payload capture, on a tiny random Qwen3 on CPU.

The GPU capture records the pinned 0.6B model on real TRAIN batches. This pins
the machinery on a 2-layer model: instrumentation does not change the step,
every call maps to exactly one case by shape, payloads round-trip with their
layouts in each task's ABI, local VJPs come from the payload (not from
model-level ``.grad``), the residual / CE / tied-head conventions hold, the
model reference reproduces every live output bit for bit, and Gate 1 and
operator Gate 2 judge a candidate on the recorded payloads. The real case
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

B, T, H, I, HQ, HK, D, V, LAYERS = 2, 16, 64, 128, 6, 2, 16, 80, 2
ARCH = {"num_hidden_layers": LAYERS, "hidden_size": H, "intermediate_size": I,
        "num_attention_heads": HQ, "num_key_value_heads": HK, "head_dim": D, "vocab_size": V,
        "max_position_embeddings": 64, "bos_token_id": 0, "eos_token_id": 1}


def tiny_cases():
    """The tiny model's cases, every dimension distinct so each call has one."""
    from evograd.benchmark import get_task
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.qwen3_0_6b import capture as Q

    n, bf = B * T, "bfloat16"
    rows = [
        ("linear_no_bias", "q", {"M": n, "K": H, "N": HQ * D}, bf, LAYERS),
        ("linear_no_bias", "kv", {"M": n, "K": H, "N": HK * D}, bf, 2 * LAYERS),
        ("linear_no_bias", "o", {"M": n, "K": HQ * D, "N": H}, bf, LAYERS),
        ("linear_no_bias", "gate_up", {"M": n, "K": H, "N": I}, bf, 2 * LAYERS),
        ("linear_no_bias", "down", {"M": n, "K": I, "N": H}, bf, LAYERS),
        ("linear_no_bias", "lm_head", {"M": n, "K": H, "N": V}, bf, 1),
        ("rmsnorm", "hidden", {"rows": n, "hidden": H}, bf, 2 * LAYERS + 1),
        ("rmsnorm", "q_norm", {"rows": n * HQ, "hidden": D}, bf, LAYERS),
        ("rmsnorm", "k_norm", {"rows": n * HK, "hidden": D}, bf, LAYERS),
        ("rope", "q", {"B": B, "n_heads": HQ, "T": T, "head_dim": D}, bf, LAYERS),
        ("rope", "k", {"B": B, "n_heads": HK, "T": T, "head_dim": D}, bf, LAYERS),
        ("swiglu", "act", {"rows": n, "cols": I}, bf, LAYERS),
        ("causal_gqa_attention", "sdpa", {"B": B, "HQ": HQ, "HK": HK, "T": T, "D": D}, bf, LAYERS),
        ("cross_entropy", "loss", {"rows": n, "cols": V}, "float32", 1),
        ("qwen3_qkv_norm_rope", "qkv", {"B": B, "T": T, "H": H, "HQ": HQ, "HK": HK, "D": D,
                                        "QO": HQ * D, "KVO": HK * D}, bf, LAYERS),
        ("qwen3_attention", "attn", {"B": B, "T": T, "HQ": HQ, "HK": HK, "D": D, "QO": HQ * D,
                                     "H": H}, bf, LAYERS),
        ("qwen3_swiglu_mlp", "mlp", {"B": B, "T": T, "H": H, "I": I}, bf, LAYERS),
        ("fused_add_rms_norm", "residual", {"rows": n, "cols": H}, bf, 2 * LAYERS),
    ]
    return tuple(Q.Case(f"{task}/{slug}", task, get_task(task).level, Workload(dims=dims, dtype=dtype),
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


class ModelReference(unittest.TestCase):
    def test_every_case_task_is_redeclared_under_the_contract(self):
        from evograd.verification.integrations.qwen3_0_6b import capture, reference

        for task in sorted({c.task for c in capture.observed_cases()}):
            op = reference.reference_op(task)
            self.assertEqual(op.reference_contract, reference.CONTRACT)
            self.assertIsNone(op.runtime_forward)
            self.assertIsNone(op.reference_dtype)

    @unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
    def test_rms_norm_is_qwen3rmsnorm_bit_for_bit(self):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        from evograd.verification.integrations.qwen3_0_6b import reference

        g = torch.Generator().manual_seed(3)
        norm = Qwen3RMSNorm(64, eps=reference.RMS_NORM_EPS).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn(64, generator=g))
        x = torch.randn(32, 64, generator=g).to(torch.bfloat16)
        r = torch.randn(32, 64, generator=g).to(torch.bfloat16)
        self.assertTrue(torch.equal(reference.rms_norm(x, norm.weight), norm(x)))
        out, summed = reference.residual_rms_norm(x, r, norm.weight)
        self.assertTrue(torch.equal(summed, r + x))
        self.assertTrue(torch.equal(out, norm(r + x)))

    def test_rmsnorm_generated_inputs_use_the_model_eps(self):
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.verification.integrations.qwen3_0_6b import capture, reference

        case = next(c for c in capture.observed_cases() if c.case_id == "rmsnorm/k_head_norm")
        op = reference.reference_op("rmsnorm")
        small = type(case.workload)(dims={"rows": 4, "hidden": 8}, dtype="float32",
                                    provenance=case.workload.provenance)
        self.assertEqual(make_case_inputs(op, small, device="cpu")["eps"], reference.RMS_NORM_EPS)


class RealCaseTable(unittest.TestCase):
    def test_observed_cases_follow_registry_and_snapshot(self):
        from evograd.verification.integrations.qwen3_0_6b import capture as Q

        cases = {c.case_id: c for c in Q.observed_cases()}
        self.assertEqual(sum(c.level == 1 for c in cases.values()), 14)
        self.assertEqual(sum(c.level == 2 for c in cases.values()), 4)
        self.assertEqual(cases["rmsnorm/rmsnorm"].frequency, 57)
        self.assertEqual(cases["fused_add_rms_norm/residual_rmsnorm"].frequency, 56)
        self.assertEqual(cases["rope/rope_kv"].workload.dims["n_heads"], 8)
        self.assertEqual(cases["linear_no_bias/lm_head"].frequency, 1)

    def test_pins_match_the_tier3_real_text_workload(self):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import protocol4_cli
        from evograd.verification.integrations.qwen3_0_6b import capture as Q

        self.assertEqual((Q.PRETRAINED, Q.REAL_TEXT), (protocol4_cli.PRETRAINED, protocol4_cli.REAL_TEXT))


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        from evograd.verification.integrations.qwen3_0_6b import capture as Q
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.Q = Q
        workload = Qwen3Workload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                 arch_overrides=ARCH)
        cls.ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(7))
        held_out = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(8))

        def build():
            model = Q.build_model(workload)
            # A fresh norm weight is exactly 1; give it a trained spread.
            g = torch.Generator().manual_seed(11)
            with torch.no_grad():
                for name, p in sorted(model.named_parameters()):
                    if name.endswith(("norm.weight", "layernorm.weight")):
                        p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            return model

        plain_model = build()
        cls.plain = Q.step_record(plain_model, Q.run_step(plain_model, cls.ids, cls.ids.clone()))
        del plain_model
        cls.model = build()
        cls.recorder = Q.LiveCapture(cls.model).install()
        try:
            output = Q.run_step(cls.model, cls.ids, cls.ids.clone())
        finally:
            cls.recorder.remove()
        cls.instrumented = Q.step_record(cls.model, output)
        cls.cases = tiny_cases()
        cls.invocations = list(Q.invocations(cls.recorder, cls.model, cls.cases))
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        Q.record_step(writer, cls.recorder, cls.model, cls.cases, {"order": 101, "sha256": "b"})
        # A second data order, so payloads split into calibration (101) and acceptance (303).
        other = build()
        other_recorder = Q.LiveCapture(other).install()
        try:
            Q.run_step(other, held_out, held_out.clone())
        finally:
            other_recorder.remove()
        Q.record_step(writer, other_recorder, other, cls.cases, {"order": 303, "sha256": "c"})
        del other, other_recorder
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
        self.assertEqual(sum(i["level"] == 1 for i in self.invocations), 15 * LAYERS + 3)
        residual = [i for i in self.invocations if i["task"] == "fused_add_rms_norm"]
        self.assertEqual(sorted(i["call"]["role"] for i in residual).count("final_model_norm"), 1)

    def test_payloads_match_each_task_interface(self):
        from evograd.benchmark import get_task
        from evograd.verification.payloads.gate1 import check_payload_interface

        for record in self.corpus.payloads():
            self.assertEqual(check_payload_interface(get_task(record["task"]), record), [],
                             record["payload_id"])

    def test_unmapped_call_is_an_error(self):
        cases = tuple(c for c in self.cases if c.case_id != "linear_no_bias/lm_head")
        with self.assertRaises(KeyError):
            list(self.Q.invocations(self.recorder, self.model, cases))

    def test_cotangent_layouts_survive(self):
        qkv = next(r for r in self.corpus.payloads()
                   if r["task"] == "qwen3_qkv_norm_rope" and r["call"]["layer"] == 0)
        sdpa = self.recorder.sdpa[0]
        loaded = self.corpus.load(qkv, device="cpu")
        for name in ("dq", "dk", "dv"):
            self.assertEqual(qkv["cotangents"][name]["stride"], list(sdpa[name].stride()), name)
            self.assertEqual(loaded[name].stride(), sdpa[name].stride(), name)

    def test_residual_final_cotangent_is_recorded_as_absent(self):
        final = next(r for r in self.corpus.payloads() if r["call"].get("role") == "final_model_norm")
        self.assertTrue(final["cotangent_stats"]["dsummed"]["all_zero"])
        self.assertEqual(final["cotangent_stats"]["dsummed"]["origin"], "absent_in_model_zero_supplied")
        inner = next(r for r in self.corpus.payloads() if r["call"].get("role") == "within_layer")
        self.assertFalse(inner["cotangent_stats"]["dsummed"]["all_zero"])

    def test_model_eps_is_recorded(self):
        norm = next(r for r in self.corpus.payloads() if r["task"] == "rmsnorm")
        self.assertEqual(norm["args"]["eps"], {"scalar": self.model.model.norm.variance_epsilon})

    def test_ce_target_is_shifted_once_and_lm_head_is_tied(self):
        ce = next(r for r in self.corpus.payloads() if r["task"] == "cross_entropy")
        target = self.corpus.load(ce, device="cpu")["target"]
        expected = torch.cat([self.ids[:, 1:], torch.full((B, 1), -100)], dim=1).reshape(-1)
        self.assertTrue(torch.equal(target, expected))
        head = next(r for r in self.corpus.payloads() if r["call"]["path"] == "lm_head")
        self.assertEqual(head["notes"].get("tied_weight"), "model.embed_tokens.weight")

    def test_local_vjp_comes_from_the_payload_not_model_grad(self):
        from evograd.benchmark import get_task
        from evograd.opdecl.oracle import oracle

        head = next(r for r in self.corpus.payloads() if r["call"]["path"] == "lm_head")
        _y, grads = oracle(get_task("linear_no_bias"), self.corpus.load(head, device="cpu"))
        # The tied weight's model-level gradient also holds the embedding's contribution.
        self.assertFalse(torch.equal(grads["dweight"].to(self.model.lm_head.weight.grad.dtype),
                                     self.model.lm_head.weight.grad))
        residual = next(r for r in self.corpus.payloads() if r["call"].get("role") == "within_layer")
        _y, grads = oracle(get_task("fused_add_rms_norm"), self.corpus.load(residual, device="cpu"))
        self.assertEqual(sorted(grads), ["dr", "dweight", "dx"])
        self.assertTrue(torch.equal(grads["dx"], grads["dr"]))       # summed = x + r

    def test_model_reference_reproduces_every_live_output(self):
        from evograd.opdecl.inputs import as_output_tuple
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.common.tensors import tensor_digest
        from evograd.verification.integrations.qwen3_0_6b.reference import reference_op

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
        from evograd.verification.integrations.qwen3_0_6b.gates import judge_case

        case = next(c for c in self.cases if c.case_id == "linear_no_bias/q")
        for scale, expected in ((1.0, "passed"), (1.05, "failed")):
            reports = judge_case(case, linear_pair(scale), self.corpus, device="cpu",
                                 compile_backend="aot_eager")
            status = {name: r.status for name, r in reports.items()}
            self.assertEqual(status, {"gate1_generated": expected, "gate1_payloads": expected,
                                      "gate2_operator_calibration": "calibrated",
                                      "gate2_operator": expected}, scale)

    def test_development_subset_is_deterministic_and_covers_every_role(self):
        again = self.Q.splits(self.corpus.payloads())
        self.assertEqual(again["development"], self.corpus.split("development"))
        roles = {(r["case_id"], r["call"]["role"]) for r in self.corpus.payloads()}
        dev = {(r["case_id"], r["call"]["role"]) for r in self.corpus.payloads(split="development")}
        self.assertEqual(roles, dev)


if __name__ == "__main__":
    unittest.main()
