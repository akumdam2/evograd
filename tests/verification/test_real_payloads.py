"""Real training payloads as Gate 1 / Gate 2 operator inputs (CPU, no GPU, no network).

The corpus here is written from tensors built in the test (a stand-in for a
captured step); what is pinned is the machinery: exact round trips, refusal
of corrupt or overlapping data, payload-only Gate 1 cases, the corpus-backed
operator scope, and separation from declaration-generated inputs.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from evograd.opdecl.activity import Workload
from evograd.verification import gate1, gate2
from evograd.verification.common.execution import Deployment
from evograd.verification.payloads import Corpus, CorpusError, CorpusWriter, UnsupportedLayout, payload_cases
from evograd.verification.payloads.corpus import cotangent_stats

from tests.verification import _fixtures as F

FAST = "aot_eager"


def _write_corpus(root: Path, payloads: list[tuple[str, int, dict, dict]], *, case_id="case/x",
                  task="vt_scale_shift", name="test") -> Path:
    """``payloads`` = [(call, order, args, cotangents)]."""
    writer = CorpusWriter(root)
    for call, order, args, cots in payloads:
        writer.new_step()
        writer.add_payload({
            "payload_id": f"o{order}/{case_id}/{call}", "case_id": case_id, "task": task, "level": 1,
            "batch": {"order": order, "sha256": f"batch{order}"},
            "call": {"path": call, "layer": 0, "role": "r", "index_in_step": ord(call[0]) - 97},
            "args": {k: (writer.put(v) if torch.is_tensor(v) else {"scalar": v}) for k, v in args.items()},
            "cotangents": {k: writer.put(v) for k, v in cots.items()},
            "cotangent_stats": {k: cotangent_stats(v) for k, v in cots.items()},
            "live_outputs": {}, "aliases": []})
    ids = [f"o{o}/{case_id}/{c}" for c, o, _a, _c in payloads]
    splits = {"calibration": [i for i in ids if i.startswith(("o101", "o202"))],
              "acceptance": [i for i in ids if i.startswith(("o303", "o404"))],
              "development": [i for i in ids if i.startswith("o101")]}
    return writer.finish(identity={"name": name, "reference": {"contract": "test"}}, splits=splits,
                         coverage={"complete": True})


def _scale_shift_payloads(R=4, C=8, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for order in (101, 202, 303, 404):
        for call in ("a", "b"):
            x = torch.randn(R, C, generator=g)
            w = torch.randn(C, generator=g)
            b = torch.randn(C, generator=g)
            dy = torch.randn(R, C, generator=g)
            out.append((call, order, {"x": x, "w": w, "b": b}, {"dy": dy}))
    return out


class CorpusRoundTripTests(unittest.TestCase):
    def test_values_dtypes_strides_offsets_scalars_and_aliases_survive(self):
        base = torch.randn(2, 6, 4, 8, dtype=torch.bfloat16)              # [B, T, H, D]
        head_major = base.permute(0, 2, 1, 3)                             # [B, H, T, D] view
        offset_view = base.reshape(-1)[16:16 + 32].view(4, 8)            # nonzero storage offset
        target = torch.tensor([3, -100, 1], dtype=torch.int64)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_corpus(Path(tmp), [("c", 101, {"x": head_major, "y": offset_view,
                                                             "base": base, "eps": 1e-6, "t": target},
                                                  {"dy": head_major.transpose(2, 3)})])
            corpus = Corpus.open(manifest)
            record = corpus.payloads()[0]
            loaded = corpus.load(record, device="cpu")
            for name, original in (("x", head_major), ("y", offset_view), ("base", base), ("t", target)):
                self.assertEqual(loaded[name].dtype, original.dtype, name)
                self.assertEqual(loaded[name].stride(), original.stride(), name)
                self.assertEqual(loaded[name].storage_offset(), original.storage_offset(), name)
                self.assertTrue(torch.equal(loaded[name], original), name)
            self.assertEqual(loaded["eps"], 1e-6)
            self.assertEqual(loaded["dy"].stride(), head_major.transpose(2, 3).stride())
            # x, y, base and dy shared one storage at capture and share one after loading.
            ptrs = {loaded[n].untyped_storage().data_ptr() for n in ("x", "y", "base", "dy")}
            self.assertEqual(len(ptrs), 1)
            self.assertEqual(len(corpus.data["storages"]), 2)             # the bf16 buffer + the target

    def test_overlapping_layout_is_refused_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = CorpusWriter(Path(tmp))
            with self.assertRaises(UnsupportedLayout):
                writer.put(torch.randn(1, 4).expand(3, 4))

    def test_corrupt_shard_and_tampered_manifest_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_corpus(Path(tmp), _scale_shift_payloads())
            corpus = Corpus.open(manifest)
            shard = corpus.root / corpus.data["shards"][0]["file"]
            data = bytearray(shard.read_bytes())
            data[10] ^= 0xFF
            shard.write_bytes(bytes(data))
            with self.assertRaises(CorpusError):
                corpus.load(corpus.payloads()[0], device="cpu")
            os.chmod(manifest, 0o640)
            body = json.loads(manifest.read_text())
            body["payloads"][0]["call"]["layer"] = 7
            manifest.write_text(json.dumps(body))
            with self.assertRaises(CorpusError):
                Corpus.open(manifest)

    def test_corpus_is_immutable_and_splits_must_name_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_corpus(Path(tmp), _scale_shift_payloads())
            with self.assertRaises(CorpusError):
                CorpusWriter(Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            writer = CorpusWriter(Path(tmp))
            with self.assertRaises(CorpusError):
                writer.finish(identity={"name": "x"}, splits={"acceptance": ["nope"]}, coverage={})

    def test_zero_cotangent_is_reported_as_not_exercising_backward(self):
        self.assertTrue(cotangent_stats(torch.zeros(3))["all_zero"])
        self.assertFalse(cotangent_stats(torch.ones(3))["all_zero"])
        self.assertEqual(cotangent_stats(None), {"present": False})


class Gate1PayloadCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = _write_corpus(Path(self.tmp.name), _scale_shift_payloads())
        self.corpus = Corpus.open(self.manifest)
        self.op = replace(F.SCALE_SHIFT, correctness=(Workload({"R": 4, "C": 8}, "float32"),))

    def tearDown(self):
        self.tmp.cleanup()

    def cases(self, split="acceptance"):
        return payload_cases(self.op, self.op.correctness[0], self.corpus, case_id="case/x",
                             split=split, device="cpu")

    def test_cases_are_exactly_the_split_payloads_and_bypass_generation(self):
        with mock.patch("evograd.verification.gate1.adapters.declared_values",
                        side_effect=AssertionError("generated inputs used")), \
             mock.patch("evograd.verification.gate1.runner.make_case_inputs",
                        side_effect=AssertionError("generated inputs used")):
            cases = self.cases()
            report = gate1.verify_pair(self.op, F.scale_shift_pair(), cases=cases, device="cpu")
        self.assertEqual(sorted(c.case_id for c in cases), sorted(self.corpus.split("acceptance")))
        self.assertEqual(report.status, "passed")
        self.assertTrue(all(c.evidence["inputs"] == "real_training_payload" for c in report.cases))
        self.assertEqual({c.evidence["provenance"]["batch"]["order"] for c in report.cases}, {303, 404})

    def test_wrong_candidate_fails_and_only_args_and_cotangents_reach_it(self):
        seen = []

        def spy(x, w, b):
            seen.append(sorted(k for k in ("x", "w", "b")))
            return x * w + b + 1e-3, (x, w)

        def bwd(dy, saved):
            x, w = saved
            return dy * w, (dy * x).sum(0), dy.sum(0)

        candidate = type("M", (), {"forward_with_saved": staticmethod(spy),
                                   "backward_from_saved": staticmethod(bwd)})
        report = gate1.verify_pair(self.op, candidate, cases=self.cases(), device="cpu")
        self.assertEqual(report.status, "failed")
        values = self.cases()[0].values.materialize()
        self.assertEqual(sorted(values), ["b", "dy", "w", "x"])      # no captured output, no saved state

    def test_mutating_candidate_cannot_contaminate_the_corpus(self):
        record = self.corpus.payloads(split="acceptance")[0]
        before = self.corpus.load(record, device="cpu")
        report = gate1.verify_pair(self.op, F.scale_shift_pair(mutate_input=True), cases=self.cases(),
                                   device="cpu")
        self.assertEqual(report.status, "failed")
        after = self.corpus.load(record, device="cpu")
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))

    def test_missing_case_or_interface_mismatch_is_a_setup_error(self):
        with self.assertRaises(CorpusError):
            payload_cases(self.op, self.op.correctness[0], self.corpus, case_id="case/none",
                          split="acceptance", device="cpu")
        with self.assertRaises(CorpusError):
            payload_cases(F.MUL, F.MUL.correctness[0], self.corpus, case_id="case/x",
                          split="acceptance", device="cpu")


class PayloadOperatorScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evograd.verification.gate2.adapters.payload import PayloadOperatorScope
        from evograd.verification.gate2.adapters.operator import OperatorScope

        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest = _write_corpus(Path(cls.tmp.name), _scale_shift_payloads())
        cls.corpus = Corpus.open(cls.manifest)
        cls.op = replace(F.SCALE_SHIFT, correctness=(Workload({"R": 4, "C": 8}, "float32"),))
        cls.scope = PayloadOperatorScope(cls.op, cls.op.correctness[0], cls.corpus, "case/x", device="cpu")
        cls.calibration = gate2.calibrate(cls.scope, patch_set=("operator",),
                                          samples=cls.scope.samples("calibration"), compile_backend=FAST)
        cls.generated_scope = OperatorScope(cls.op, cls.op.correctness[0], device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_calibrated_on_calibration_payloads_judged_on_acceptance_payloads(self):
        self.assertEqual(self.calibration.status, "calibrated", self.calibration.to_dict())
        policy = self.calibration.policy
        samples = self.scope.samples("acceptance")
        self.assertEqual({s.seed // 10**6 for s in samples}, {303, 404})
        good = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair()),
                                        policy=policy, samples=samples)
        bad = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair(dw_scale=1.5)),
                                       policy=policy, samples=samples)
        self.assertEqual(good.status, "passed")
        self.assertEqual(bad.status, "failed")
        self.assertTrue(all(s.sample_id.startswith(self.corpus.corpus_id) for s in samples))

    def test_calibration_payloads_cannot_be_judged(self):
        report = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair()),
                                          policy=self.calibration.policy,
                                          samples=self.scope.samples("calibration"))
        self.assertEqual(report.status, "incomplete")

    def test_generated_input_policy_is_refused_and_another_corpus_too(self):
        from evograd.verification.gate2.adapters.payload import PayloadOperatorScope

        synthetic = gate2.calibrate(self.generated_scope, patch_set=("operator",),
                                    samples=self.generated_scope.samples([1, 2]), compile_backend=FAST).policy
        report = gate2.verify_integration(self.scope, self.scope.candidate(F.scale_shift_pair()),
                                          policy=synthetic, samples=self.scope.samples("acceptance"))
        self.assertEqual(report.status, "incomplete")
        self.assertIn("policy_case_mismatch", {f.code for f in report.all_findings()})
        with tempfile.TemporaryDirectory() as tmp:
            other = Corpus.open(_write_corpus(Path(tmp), _scale_shift_payloads(seed=5)))
            scope2 = PayloadOperatorScope(self.op, self.op.correctness[0], other, "case/x", device="cpu")
            self.assertNotEqual(scope2.identity(), self.scope.identity())
            report = gate2.verify_integration(scope2, scope2.candidate(F.scale_shift_pair()),
                                              policy=self.calibration.policy,
                                              samples=scope2.samples("acceptance"))
            self.assertEqual(report.status, "incomplete")

    def test_native_and_compile_controls_qualify_on_held_out_payloads(self):
        samples = self.scope.samples("acceptance")
        control, _graphs = gate2.compile_control(self.scope, ("operator",), backend=FAST)
        report = gate2.verify_integration(self.scope, control, policy=self.calibration.policy, samples=samples)
        self.assertEqual(report.status, "passed")
        native = gate2.execute(self.scope, Deployment.native(), samples[0])
        self.assertIsNone(native.error)


if __name__ == "__main__":
    unittest.main()
