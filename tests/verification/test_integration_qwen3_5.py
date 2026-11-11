"""Qwen3.5-4B real-payload capture, on a tiny random Qwen3.5 on CPU.

Eight layers in the published pattern (six Gated DeltaNet, two gated
attention), two data orders, in bfloat16 and float32. Pins: instrumentation
changes nothing, every call maps to exactly one case by shape, each Level-2
task gets the calls per step the harvest counts (``manifest.FREQUENCY``),
payloads follow each task's ABI, and the model reference reproduces every live
output bit for bit -- the attention block and the 2-per-layer residual fusion
included. The real case table is checked against the registry and the
snapshot once the tracked snapshot is regenerated.
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


def tiny_cases(dtype: str = "bfloat16"):
    """The tiny model's case table; every Level-2 task's calls per step are the
    manifest's (what the harvest snapshot counts, ``test_harvest``)."""
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
    from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    bf, layers = dtype, GDN_LAYERS + ATTN_LAYERS
    calls = {manifest.task_key(site):
             manifest.expected_model_frequency(site, with_layers(QWEN3_5_4B, layers))
             for site in manifest.SITES}
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
                                            "K": 4, "D": DK}, bf, calls["qwen3_5_gdn_in_proj"]),
        ("qwen3_5_gated_delta_rule", "delta", {"B": 1, "T": T, "HV": HV, "D": DK}, bf,
         calls["qwen3_5_gated_delta_rule"]),
        ("qwen3_5_gdn_norm_out", "norm_out", {"B": 1, "T": T, "HV": HV, "D": DK, "VD": VD, "H": H},
         bf, calls["qwen3_5_gdn_norm_out"]),
        ("qwen3_5_attention_block", "attention_block",
         {"B": 1, "T": T, "H": H, "HQ": HQ, "HK": HK, "D": D, "R": R, "QG": 2 * HQ * D,
          "KVO": HK * D, "QO": HQ * D}, bf, calls["qwen3_5_attention_block"]),
        ("qwen3_5_residual_rmsnorm", "residual", {"B": 1, "T": T, "H": H}, bf,
         calls["qwen3_5_residual_rmsnorm"]),
        ("qwen3_5_swiglu_mlp", "mlp", {"B": 1, "T": T, "H": H, "I": I}, bf,
         calls["qwen3_5_swiglu_mlp"]),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level, Workload(dims=dims, dtype=dtype),
                      freq) for task, slug, dims, dtype, freq in rows)


def snapshot_is_current() -> bool:
    """Whether the tracked snapshot holds the Level-2 tasks the code declares; it
    predates them until the GPU harvest regenerates it."""
    from evograd.benchmark.topdown import load_snapshot
    from evograd.benchmark.topdown.qwen3_5_4b.harvest.snapshot import TASK_SOURCES

    return set(load_snapshot("qwen3_5_4b")["tasks"]) == set(TASK_SOURCES)


STALE_SNAPSHOT = ("the tracked Qwen3.5 snapshot predates its Level-2 tasks "
                  "(qwen3_5_attention_block, the 64-call residual fusion); regenerate it on a "
                  "GPU node")


class RealCaseTable(unittest.TestCase):
    @unittest.skipUnless(snapshot_is_current(), STALE_SNAPSHOT)
    def test_observed_cases_follow_registry_and_snapshot(self):
        from evograd.benchmark.topdown.qwen3_5_4b.levels.level2 import manifest
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_5_4b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.frequency for c in cases.values() if c.level == 1), 507)
        self.assertEqual({c.task: c.frequency for c in cases.values() if c.level == 2},
                         {manifest.task_key(site): manifest.expected_model_frequency(site)
                          for site in manifest.SITES})
        # The residual site's own calls (two per layer), not its anchor norm's 65.
        self.assertEqual(cases["qwen3_5_residual_rmsnorm/residual_rmsnorm"].frequency, 64)
        self.assertEqual(cases["qwen3_5_attention_block/attention_block"].frequency, 8)

    def test_gate3_horizon_is_per_model(self):
        from evograd.verification.integrations.convnextv2_tiny import MODEL as CONVNEXT
        from evograd.verification.integrations.qwen2_vl_2b import MODEL as QWEN2_VL
        from evograd.verification.integrations.qwen3_0_6b import MODEL as QWEN3
        from evograd.verification.integrations.qwen3_5_4b import MODEL

        self.assertEqual((QWEN3.gate3_steps, MODEL.gate3_steps), (100, 100))
        self.assertEqual((CONVNEXT.gate3_steps, QWEN2_VL.gate3_steps), (50, 10))
        self.assertEqual(MODEL.gate3_optimizer, {"warmup_steps": 100, "clip_grad_norm": 1.0,
                                                 "lr": 3e-5})

    @unittest.skipUnless(snapshot_is_current(), STALE_SNAPSHOT)
    def test_every_case_task_is_redeclared_under_the_contract(self):
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_5_4b import MODEL, reference

        for task in sorted({c.task for c in observed_cases(MODEL)}):
            op = reference.reference_op(task)
            self.assertEqual(op.reference_contract, reference.CONTRACT)
            self.assertIsNone(op.runtime_forward)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    DTYPE = "bfloat16"

    @classmethod
    def setUpClass(cls):
        from evograd.benchmark.topdown.qwen3_5_4b.architecture import QWEN3_5_4B, with_layers
        from evograd.evaluation.tier3.workloads.qwen3_5_4b.workload import Qwen3_5Workload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.qwen3_5_4b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter
        from tests.qwen3_5_4b._tiny import TINY_OVERRIDES

        cls.Q, cls.C = Q, C
        workload = Qwen3_5Workload(device="cpu", dtype=cls.DTYPE, batch_size=1, seq_len=T,
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

        cls.cases = tiny_cases(cls.DTYPE)
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

    def _by_order(self, task):
        out: dict[int, list] = {}
        for record in self.corpus.payloads():
            if record["task"] == task:
                out.setdefault(record["batch"]["order"], []).append(record)
        return out

    def test_the_attention_block_is_the_whole_module(self):
        by_order = self._by_order("qwen3_5_attention_block")
        self.assertEqual(sorted(by_order), [101, 303])
        for order, records in by_order.items():
            self.assertEqual(sorted(r["call"]["layer"] for r in records), [3, 7], order)
            for record in records:
                self.assertEqual(record["args"]["cos"]["shape"], [1, T, R])
                self.assertEqual(record["args"]["o_weight"]["shape"], [H, HQ * D])
                self.assertEqual(sorted(record["cotangents"]), ["dy"])

    def _dresidual(self, record):
        """The residual input's gradient, by autograd of the model reference on the
        payload's own cotangents: what the fusion hands back to the sum before it."""
        from evograd.opdecl.oracle import resolve_runtime_forward
        from evograd.verification.integrations.qwen3_5_4b.reference import reference_op

        op = reference_op(record["task"])
        values = self.corpus.load(record, device="cpu")
        branch, residual = (values[n].clone().requires_grad_(True) for n in ("branch", "residual"))
        normed, summed = resolve_runtime_forward(op)(branch, residual, values["weight"],
                                                     values.get("eps", op.args[-1].default))
        return torch.autograd.grad((normed, summed), (branch, residual),
                                   (values["dnormed"], values["dsummed"]))[1]

    def test_two_residual_fusions_per_layer_and_the_external_dsummed(self):
        """``dsummed`` is what the next residual add delivers, so replaying each
        fusion's backward on its payload gives exactly the ``dsummed`` of the
        fusion before it: the chain closes bit for bit. A ``dsummed`` that
        included the fused norm's own path (the old Qwen3.5 recording) breaks it."""
        layers = GDN_LAYERS + ATTN_LAYERS
        for order, records in self._by_order("qwen3_5_residual_rmsnorm").items():
            roles: dict[str, dict[int, dict]] = {}
            for record in records:
                roles.setdefault(record["call"]["role"], {})[record["call"]["layer"]] = record
            self.assertEqual({role: sorted(found) for role, found in roles.items()},
                             {"within_layer": list(range(layers)),
                              "cross_layer": list(range(layers - 1)),
                              "final_model_norm": [layers - 1]})
            final = roles["final_model_norm"][layers - 1]
            self.assertEqual(final["call"]["norm"], "model.norm")
            self.assertTrue(final["cotangent_stats"]["dsummed"]["all_zero"])
            self.assertEqual(final["notes"]["absent_cotangents"], ["dsummed"])
            closing = {**roles["cross_layer"], layers - 1: final}
            for layer in range(layers):
                within = roles["within_layer"][layer]
                self.assertFalse(within["cotangent_stats"]["dsummed"].get("all_zero"))
                with self.subTest(order=order, layer=layer, fusion="mlp add"):
                    self.assertTrue(torch.equal(
                        self._dresidual(closing[layer]),
                        self.corpus.load(within, device="cpu")["dsummed"]))
                if layer:
                    with self.subTest(order=order, layer=layer, fusion="attention add"):
                        self.assertTrue(torch.equal(
                            self._dresidual(within),
                            self.corpus.load(roles["cross_layer"][layer - 1],
                                             device="cpu")["dsummed"]))


class TinyCaptureFloat32Tests(TinyCaptureTests):
    """The same capture of a float32 model: the references reproduce it bitwise too."""

    DTYPE = "float32"


if __name__ == "__main__":
    unittest.main()
