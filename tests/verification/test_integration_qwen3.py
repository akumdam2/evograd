"""Qwen3-0.6B real-payload capture, on a tiny random Qwen3 on CPU.

The GPU capture records the pinned 0.6B model on real TRAIN batches. This pins
the machinery on a 2-layer model: instrumentation does not change the step,
every call maps to exactly one case by shape, payloads round-trip with their
layouts in each task's ABI, local VJPs come from the payload (not from
model-level ``.grad``), the residual / CE / tied-head conventions hold, the
model reference reproduces every live output bit for bit, and Gate 1 and
operator Gate 2 judge a candidate on the recorded payloads. The Level-2
references are the modules' own arithmetic, bit for bit in float32 and BF16.
The real case table is checked against the registry and the snapshot.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from tests.qwen3 import QWEN3_SNAPSHOT_STALE, qwen3_snapshot_current

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
    from evograd.verification.integrations.base import Case

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
        ("qwen3_attention_block", "attention_block", {"B": B, "T": T, "H": H, "HQ": HQ, "HK": HK,
                                                      "D": D, "QO": HQ * D, "KVO": HK * D},
         bf, LAYERS),
        ("qwen3_swiglu_mlp", "mlp", {"B": B, "T": T, "H": H, "I": I}, bf, LAYERS),
        ("fused_add_rms_norm", "residual", {"rows": n, "cols": H}, bf, 2 * LAYERS),
        ("qwen3_lm_head_ce", "flce", {"rows": n, "hidden": H, "vocab": V}, bf, 1),
    ]
    return tuple(Case(f"{task}/{slug}", task, get_task(task).level, Workload(dims=dims, dtype=dtype),
                        freq) for task, slug, dims, dtype, freq in rows)


def flce_pair(scale: float):
    """A qwen3_lm_head_ce pair whose weight gradient is ``scale`` times the true one."""
    import types

    import torch.nn.functional as F

    def loss_of(x, weight, target):
        return F.cross_entropy(F.linear(x, weight).float(), target, ignore_index=-100)

    def forward_with_saved(x, weight, target):
        return loss_of(x, weight, target), (x, weight, target)

    def backward_from_saved(dloss, saved):
        x, weight, target = (t.detach() for t in saved)
        with torch.enable_grad():
            x, weight = x.requires_grad_(True), weight.requires_grad_(True)
            dx, dweight = torch.autograd.grad(loss_of(x, weight, target), (x, weight), dloss)
        return dx, dweight * scale

    return types.SimpleNamespace(__name__=f"flce_x{scale}", forward_with_saved=forward_with_saved,
                                 backward_from_saved=backward_from_saved)


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
        from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import TASK_SOURCES, load
        from evograd.verification.integrations.qwen3_0_6b import reference

        for task in sorted({*load()["level1"], *TASK_SOURCES}):
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
        from evograd.benchmark import get_task
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.verification.integrations.qwen3_0_6b import reference

        workload = next(w for w in get_task("rmsnorm").benchmark_suites["qwen3_0_6b_observed"]
                        if w.provenance.component == "k_head_norm")
        op = reference.reference_op("rmsnorm")
        small = type(workload)(dims={"rows": 4, "hidden": 8}, dtype="float32",
                               provenance=workload.provenance)
        self.assertEqual(make_case_inputs(op, small, device="cpu")["eps"], reference.RMS_NORM_EPS)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class LevelTwoReferencesAreTheModulesArithmetic(unittest.TestCase):
    """The attention block and the LM head with the loss, against the live modules.

    On the tiny model in float32 and BF16: the reference's output and its
    gradients (the payload VJPs every gate recomputes) equal the module's own,
    bit for bit, at the module's own inputs. ``fidelity`` checks the forward
    again on every recorded call of the pinned model on GPU.
    """

    def _model(self, dtype):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        from evograd.verification.integrations import base as Q

        workload = Qwen3Workload(device="cpu", dtype=dtype, batch_size=B, seq_len=T,
                                 arch_overrides=ARCH)
        model = Q.build_model(workload)
        g = torch.Generator().manual_seed(11)
        with torch.no_grad():
            for name, p in sorted(model.named_parameters()):
                if name.endswith("norm.weight"):
                    p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
        return model

    @staticmethod
    def _grads(output, leaves, cotangent):
        return torch.autograd.grad(output, leaves, cotangent)

    def test_attention_block(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.attention_block.reference import (
            qwen3_attention_block_forward as reference)

        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                model = self._model(dtype)
                attn, seen = model.model.layers[1].self_attn, {}
                handle = attn.register_forward_pre_hook(
                    lambda m, a, k: seen.update(k), with_kwargs=True)
                ids = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(7))
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                handle.remove()
                self.assertIsNone(seen["attention_mask"])
                cos, sin = seen["position_embeddings"]
                x = seen["hidden_states"].detach().requires_grad_(True)
                dout = torch.randn(B, T, H, generator=torch.Generator().manual_seed(3)).to(x.dtype)
                weights = [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight,
                           attn.o_proj.weight, attn.q_norm.weight, attn.k_norm.weight]
                live, _ = attn(hidden_states=x, position_embeddings=(cos, sin), attention_mask=None)
                ours = reference(x, *weights, cos[:1], sin[:1], attn.q_norm.variance_epsilon)
                self.assertTrue(torch.equal(ours, live))
                for a, b in zip(self._grads(live, [x, *weights], dout),
                                self._grads(ours, [x, *weights], dout)):
                    self.assertTrue(torch.equal(a, b))

    def test_lm_head_ce(self):
        import torch.nn.functional as F

        from evograd.benchmark.topdown.qwen3_0_6b.levels.level2.lm_head_ce.reference import (
            qwen3_lm_head_ce_forward as reference)

        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                model = self._model(dtype)
                labels = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(5))
                hidden = torch.randn(B, T, H, generator=torch.Generator().manual_seed(6)).to(
                    model.lm_head.weight.dtype).requires_grad_(True)
                weight = model.lm_head.weight
                live = model.loss_function(logits=model.lm_head(hidden), labels=labels,
                                           vocab_size=V)
                target = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
                ours = reference(hidden.reshape(-1, H), weight, target)
                self.assertEqual(ours.dtype, torch.float32)
                self.assertTrue(torch.equal(ours, live))
                for a, b in zip(self._grads(live, [hidden, weight], torch.ones(())),
                                self._grads(ours, [hidden, weight], torch.ones(()))):
                    self.assertTrue(torch.equal(a, b))


class RealCaseTable(unittest.TestCase):
    @unittest.skipUnless(qwen3_snapshot_current(), QWEN3_SNAPSHOT_STALE)
    def test_observed_cases_follow_registry_and_snapshot(self):
        from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import TASK_SOURCES
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_0_6b import MODEL

        cases = {c.case_id: c for c in observed_cases(MODEL)}
        self.assertEqual(sum(c.level == 1 for c in cases.values()), 14)
        self.assertEqual(sorted(c.task for c in cases.values() if c.level == 2),
                         sorted(TASK_SOURCES))
        self.assertEqual(cases["rmsnorm/rmsnorm"].frequency, 57)
        self.assertEqual(cases["fused_add_rms_norm/residual_rmsnorm"].frequency, 56)
        self.assertEqual(cases["qwen3_attention_block/attention_block"].frequency, 28)
        self.assertEqual(cases["qwen3_swiglu_mlp/swiglu_mlp"].frequency, 28)
        self.assertEqual(cases["qwen3_lm_head_ce/flce"].frequency, 1)
        self.assertEqual(cases["rope/rope_kv"].workload.dims["n_heads"], 8)
        self.assertEqual(cases["linear_no_bias/lm_head"].frequency, 1)

    def test_a_snapshot_entry_of_a_removed_task_is_no_case(self):
        """Until the next harvest, a merged task's old entries are dropped, and the
        tasks that replace them are no cases yet."""
        from unittest import mock

        from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import load
        from evograd.verification.integrations.base import observed_cases
        from evograd.verification.integrations.qwen3_0_6b import MODEL

        snapshot = load()
        before = {c.case_id for c in observed_cases(MODEL)}
        snapshot["tasks"] = {**snapshot["tasks"], "qwen3_removed_task": {"frequency": 1}}
        with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: snapshot):
            after = {c.case_id for c in observed_cases(MODEL)}
        self.assertEqual(after, before)
        self.assertFalse(any(c.startswith("qwen3_removed_task/") for c in after))

    def test_pins_match_the_tier3_real_text_workload(self):
        from evograd.benchmark.topdown.qwen3_0_6b.levels.level3.spec import CANONICAL
        from evograd.evaluation.tier3.workloads.qwen3_0_6b import textdata
        from evograd.verification.integrations.qwen3_0_6b import capture as Q

        spec = CANONICAL.replace(weights=Q.PRETRAINED, data=Q.REAL_TEXT)
        self.assertTrue(spec.pretrained and spec.real_text)
        self.assertEqual(spec.pretrained_repo[0], "Qwen/Qwen3-0.6B")
        self.assertEqual(Q.REAL_TEXT, f"wikitext-2-raw:{textdata.DATASET_REVISION}")


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class RealRunHarvest(unittest.TestCase):
    """The harvest observes the capture's own first step; the capture checks that it did.

    The integration's model with a tiny random CPU workload and seeded batches
    in place of the pinned checkpoint and WikiText: the harvest builds through
    ``base.real_run`` exactly as on a GPU node, its snapshot records the run,
    and a capture's own identity of the same run is accepted against it while
    another run's is refused.
    """

    @classmethod
    def setUpClass(cls):
        import dataclasses
        import json
        from unittest import mock

        from evograd.benchmark.topdown.qwen3_0_6b.harvest import harvest, snapshot
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        from evograd.verification.common.tensors import tensor_digest
        from evograd.verification.integrations import base
        from evograd.verification.integrations.qwen3_0_6b import MODEL
        from tests.verification._toy_integration import configured

        workload = Qwen3Workload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                 arch_overrides=ARCH)

        def train_batches(_workload, orders, offset=0):
            seeded = {o: torch.Generator().manual_seed(o + offset) for o in orders}
            ids = {o: torch.randint(0, V, (B, T), generator=g) for o, g in seeded.items()}
            batches = {str(o): {"order": o, "input_ids_sha256": tensor_digest(i)}
                       for o, i in ids.items()}
            identity = {"dataset": "tiny", "batches": batches}
            return {o: (i, i.clone()) for o, i in ids.items()}, identity

        cls.base, cls.snapshot_module, cls.workload = base, snapshot, workload
        cls.train_batches = staticmethod(train_batches)
        cls.model = dataclasses.replace(
            MODEL, workload_config={"model": "qwen3_0_6b", "device": "cpu", "arch_overrides": ARCH},
            load_workload=lambda: workload, train_batches=train_batches)
        with configured(), \
                mock.patch("evograd.verification.integrations.qwen3_0_6b.MODEL", cls.model):
            cls.manifest = json.loads(json.dumps(harvest.harvest_real_run(device="cpu")))
        cls.snapshot = snapshot.extract(cls.manifest, layer_index=1)

    def capture_identity(self, *, offset=0, perturb=False):
        """What a capture of ``orders`` computes for the run check."""
        model = self.base.build_model(self.workload)
        if perturb:
            with torch.no_grad():
                model.model.norm.weight.add_(1)
        state = self.base.objective_of(self.model).parameter_state(model)
        batches = self.train_batches(self.workload, self.base.ORDERS, offset=offset)[1]
        return self.base.run_identity(self.model, state, batches)

    def require(self, snapshot, identity):
        from unittest import mock

        with mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: snapshot):
            self.base.require_harvested_run(self.model, identity)

    def test_the_harvest_observes_the_order_101_batch_of_the_built_model(self):
        run = self.manifest["run"]
        ids = self.train_batches(self.workload, (101,))[0][101][0]
        self.assertEqual(run["order"], 101)
        effective = self.manifest["validation"]["effective"]
        self.assertEqual(effective["input_ids_checksum"], int(ids.sum()))
        self.assertEqual(run["parameter_state"],
                         self.base.parameter_state(self.base.build_model(self.workload)))
        self.assertTrue(run["execution"]["deterministic_algorithms"])
        self.assertEqual(self.manifest["workload_id"], self.workload.spec.workload_id)
        self.assertEqual(self.manifest["workload"]["weights"], "random")

    def test_the_snapshot_records_the_run(self):
        self.assertEqual(self.snapshot["schema_version"], self.snapshot_module.SCHEMA_VERSION)
        self.assertEqual(self.snapshot["run"], self.manifest["run"])
        self.assertEqual(self.snapshot["snapshot_hash"],
                         self.snapshot_module.snapshot_hash(self.snapshot))

    def test_a_capture_of_the_same_run_is_accepted_and_of_another_refused(self):
        self.require(self.snapshot, self.capture_identity())
        for label, identity in (("batch", self.capture_identity(offset=1)),
                                ("weights", self.capture_identity(perturb=True))):
            with self.subTest(label), self.assertRaisesRegex(RuntimeError, "another run"):
                self.require(self.snapshot, identity)

    def test_a_synthetic_harvest_backs_no_capture(self):
        from evograd.benchmark.topdown.qwen3_0_6b.harvest.harvest import run_harvest

        synthetic = self.snapshot_module.extract(run_harvest(self.workload.spec), layer_index=1)
        self.assertIsNone(synthetic["run"])
        with self.assertRaisesRegex(RuntimeError, "records no harvested run"):
            self.require(synthetic, self.capture_identity())

    def test_the_snapshot_holds_every_level2_task_at_the_captures_calls_per_step(self):
        """The harvested calls per step are what the capture emits per order."""
        tasks = self.snapshot["tasks"]
        self.assertEqual(set(tasks), set(self.snapshot_module.TASK_SOURCES))
        per_step = {task: (entry.get("fusion_sites") or {}).get("total") or entry["frequency"]
                    for task, entry in tasks.items()}
        self.assertEqual(per_step, {c.task: c.frequency for c in tiny_cases() if c.level == 2})
        self.assertEqual(tasks["qwen3_attention_block"]["module_paths"],
                         [f"model.layers.{i}.self_attn" for i in range(LAYERS)])
        self.assertEqual(sorted(tasks["qwen3_lm_head_ce"]["supporting"]),
                         ["flat_cross_entropy", "lm_head"])

    def test_the_tracked_snapshot_without_a_run_still_loads(self):
        from evograd.benchmark.topdown.qwen3_0_6b.harvest.snapshot import READABLE_SCHEMAS, load

        self.assertIn(load()["schema_version"], READABLE_SCHEMAS)


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class TinyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        from evograd.verification.integrations import base as Q
        from evograd.verification.integrations.qwen3_0_6b import capture as C
        from evograd.verification.payloads import Corpus, CorpusWriter

        cls.Q, cls.C = Q, C
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
        cls.recorder = C.LiveCapture(cls.model).install()
        try:
            output = Q.run_step(cls.model, cls.ids, cls.ids.clone())
        finally:
            cls.recorder.remove()
        cls.instrumented = Q.step_record(cls.model, output)
        cls.cases = tiny_cases()
        cls.invocations = list(C.invocations(cls.recorder, cls.model, cls.cases))
        cls.tmp = tempfile.TemporaryDirectory()
        writer = CorpusWriter(Path(cls.tmp.name))
        Q.record_step(writer, C.invocations(cls.recorder, cls.model, cls.cases),
                      {"order": 101, "sha256": "b"})
        # A second data order, so payloads split into calibration (101) and acceptance (303).
        other = build()
        other_recorder = C.LiveCapture(other).install()
        try:
            Q.run_step(other, held_out, held_out.clone())
        finally:
            other_recorder.remove()
        Q.record_step(writer, C.invocations(other_recorder, other, cls.cases),
                      {"order": 303, "sha256": "c"})
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
        blocks = [i["call"]["path"] for i in self.invocations if i["task"] == "qwen3_attention_block"]
        self.assertEqual(blocks, [f"model.layers.{i}.self_attn#attention_block" for i in range(LAYERS)])
        self.assertEqual([i["call"]["path"] for i in self.invocations
                          if i["task"] == "qwen3_lm_head_ce"], ["lm_head#lm_head_ce"])

    def test_payloads_match_each_task_interface(self):
        from evograd.benchmark import get_task
        from evograd.verification.payloads.gate1 import check_payload_interface

        for record in self.corpus.payloads():
            self.assertEqual(check_payload_interface(get_task(record["task"]), record), [],
                             record["payload_id"])

    def test_unmapped_call_is_an_error(self):
        cases = tuple(c for c in self.cases if c.case_id != "linear_no_bias/lm_head")
        with self.assertRaises(KeyError):
            list(self.C.invocations(self.recorder, self.model, cases))

    def test_cotangent_layouts_survive(self):
        sdpa = next(r for r in self.corpus.payloads()
                    if r["task"] == "causal_gqa_attention" and r["call"]["layer"] == 0)
        loaded = self.corpus.load(sdpa, device="cpu")
        for name, recorded in (("q", self.recorder.sdpa[0]["q"]), ("do", self.recorder.sdpa[0]["do"])):
            entry = sdpa["args"].get(name) or sdpa["cotangents"][name]
            self.assertEqual(entry["stride"], list(recorded.stride()), name)
            self.assertEqual(loaded[name].stride(), recorded.stride(), name)

    def test_the_attention_block_payload_is_the_modules_call(self):
        block = next(r for r in self.corpus.payloads()
                     if r["task"] == "qwen3_attention_block" and r["call"]["layer"] == 1)
        loaded = self.corpus.load(block, device="cpu")
        attn = self.model.model.layers[1].self_attn
        record = self.recorder.attn[1]
        self.assertTrue(torch.equal(loaded["x"], record["x"]))
        self.assertTrue(torch.equal(loaded["dout"], record["dy"]))
        self.assertTrue(torch.equal(loaded["o_weight"], attn.o_proj.weight))
        self.assertTrue(torch.equal(loaded["k_norm_weight"], attn.k_norm.weight))
        self.assertEqual(tuple(loaded["cos"].shape), (1, T, D))
        self.assertEqual(block["args"]["eps"], {"scalar": attn.q_norm.variance_epsilon})

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
        flce = next(r for r in self.corpus.payloads() if r["task"] == "qwen3_lm_head_ce")
        loaded = self.corpus.load(flce, device="cpu")
        self.assertTrue(torch.equal(loaded["target"], expected))
        self.assertTrue(torch.equal(loaded["x"], self.corpus.load(head, device="cpu")["x"]))
        self.assertEqual(flce["notes"].get("tied_weight"), "model.embed_tokens.weight")
        self.assertEqual(flce["notes"]["ignored_rows"], B)
        # The same tensors as the lm_head and CE payloads: stored once.
        self.assertEqual(flce["args"]["x"]["storage"], head["args"]["x"]["storage"])
        self.assertEqual(flce["args"]["target"]["storage"], ce["args"]["target"]["storage"])

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
        from evograd.verification.integrations.gates import judge_case
        from evograd.verification.integrations.qwen3_0_6b import MODEL

        for case_id, pair in (("linear_no_bias/q", linear_pair), ("qwen3_lm_head_ce/flce", flce_pair)):
            case = next(c for c in self.cases if c.case_id == case_id)
            for scale, expected in ((1.0, "passed"), (1.05, "failed")):
                reports = judge_case(MODEL, case, pair(scale), self.corpus, device="cpu",
                                     compile_backend="aot_eager")
                status = {name: r.status for name, r in reports.items()}
                self.assertEqual(status, {"gate1_payloads": expected,
                                          "gate2_operator_calibration": "calibrated",
                                          "gate2_operator": expected}, (case_id, scale))

    def test_development_subset_is_deterministic_and_covers_every_role(self):
        again = self.Q.splits(self.corpus.payloads())
        self.assertEqual(again["development"], self.corpus.split("development"))
        roles = {(r["case_id"], r["call"]["role"]) for r in self.corpus.payloads()}
        dev = {(r["case_id"], r["call"]["role"]) for r in self.corpus.payloads(split="development")}
        self.assertEqual(roles, dev)


if __name__ == "__main__":
    unittest.main()
