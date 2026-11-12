"""A capture records the benchmark rows only, and every reader works on such a corpus (CPU).

On the tiny random Qwen3 of ``test_integration_qwen3`` (its 18 cases), with a
search set that makes three of them benchmark rows (two searched, one
``identical_config_of``):

* ``capture``'s default selection records exactly those cases, every call of
  each in every order, as the very records a ``--cases all`` capture holds for
  them; the identity states the rule, the search set and the cases left out,
  so the two corpora have different identities;
* replay fidelity, Gate 1 and operator Gate 2, the task pack's search
  payloads, Tier 2's recorded calls and ``open_run`` work on it;
* a case it does not record is refused, naming the rule, by every reader;
* a selection that cannot be applied is refused before anything is written;
* a corpus recorded under the old rule (its identity lists every case and no
  ``case_selection``: MegaFold's) keeps serving its cases.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False

SEARCHED = ("linear_no_bias/q", "qwen3_swiglu_mlp/mlp")
IDENTICAL = "rope/k"
ROWS = {*SEARCHED, IDENTICAL}
LEFT_OUT = "linear_no_bias/kv"
LINEAR = '''
import torch


def forward_with_saved(x, weight):
    return torch.nn.functional.linear(x, weight), (x, weight)


def backward_from_saved(dy, saved):
    x, weight = saved
    return dy @ weight, dy.t() @ x
'''


def search_set_file(path: Path, *, model="qwen3_0_6b", snapshot_hash="tiny", extra=(),
                    empty=False) -> Path:
    """A search set in which ``ROWS`` and ``extra`` (searched) are the tiny model's benchmark rows."""
    from evograd.benchmark import search_set
    from tests.verification.test_integration_qwen3 import tiny_cases

    def row(case_id):
        if empty:
            return {"searched": False, "reason": "not_time_max_config", "of": None}
        if case_id == IDENTICAL:
            return {"searched": False, "reason": "identical_config_of", "of": "llama_3_2_1b:rope/x"}
        if case_id in SEARCHED or case_id in extra:
            return {"searched": True, "reason": None, "of": None}
        return {"searched": False, "reason": "not_time_max_config", "of": None}

    cases = {cid: row(cid) for cid in [c.case_id for c in tiny_cases()] + list(extra)}
    path.write_text(json.dumps({"schema": search_set.SCHEMA, "models": {
        model: {"snapshot_hash": snapshot_hash, "cases": cases}}}))
    return path


@unittest.skipUnless(HAVE_TRANSFORMERS, "transformers is not installed")
class BenchmarkCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses

        from evograd.evaluation.tier3.workloads.qwen3_0_6b.workload import Qwen3Workload
        from evograd.verification.common.tensors import tensor_digest
        from evograd.verification.integrations import base
        from evograd.verification.integrations.qwen3_0_6b import MODEL
        from evograd.verification.payloads import Corpus
        from tests.verification._toy_integration import configured
        from tests.verification.test_integration_qwen3 import ARCH, B, T, V

        workload = Qwen3Workload(device="cpu", dtype="bfloat16", batch_size=B, seq_len=T,
                                 arch_overrides=ARCH)

        def train_batches(_workload, orders):
            ids = {o: torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(o))
                   for o in orders}
            return ({o: (i, i.clone()) for o, i in ids.items()},
                    {"dataset": "tiny", "batches": {
                        str(o): {"order": o, "input_ids_sha256": tensor_digest(i)}
                        for o, i in ids.items()}})

        cls.base = base
        cls.model = dataclasses.replace(
            MODEL, workload_config={"model": "qwen3_0_6b", "device": "cpu", "arch_overrides": ARCH},
            load_workload=lambda: workload, train_batches=train_batches)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.search_set = search_set_file(cls.root / "search_set.json")
        with configured():
            cls.snapshot = {"snapshot_hash": "tiny",
                            "run": base.real_run(cls.model, device="cpu")[3]}
        with cls.environment():
            cls.benchmark = base.capture(cls.model, cls.root / "benchmark", device="cpu",
                                         log=lambda *a: None)
            cls.full = base.capture(cls.model, cls.root / "all", selection="all", device="cpu",
                                    log=lambda *a: None)
        cls.corpus = Corpus.open(cls.benchmark)
        cls.program = cls.root / "linear.py"
        cls.program.write_text(LINEAR)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    @contextlib.contextmanager
    def environment(cls, search_set=None):
        """The tiny cases, a snapshot of the tiny run and the test's search set in place."""
        from evograd.benchmark import search_set as module
        from tests.verification._toy_integration import configured
        from tests.verification.test_integration_qwen3 import tiny_cases

        with configured(), \
                mock.patch("evograd.verification.integrations.base.observed_cases",
                           lambda model: tiny_cases()), \
                mock.patch("evograd.benchmark.topdown.load_snapshot", lambda name: cls.snapshot), \
                mock.patch.object(module, "PATH", search_set or cls.search_set):
            yield

    def case(self, case_id):
        from tests.verification.test_integration_qwen3 import tiny_cases

        return next(c for c in tiny_cases() if c.case_id == case_id)

    def test_the_capture_records_exactly_the_benchmark_rows(self):
        from evograd.benchmark import search_set
        from tests.verification.test_integration_qwen3 import tiny_cases

        manifest = json.loads(Path(self.benchmark).read_text())
        identity = manifest["identity"]
        self.assertEqual(set(identity["cases"]), ROWS)
        self.assertTrue(manifest["coverage"]["complete"], manifest["coverage"])
        self.assertEqual(manifest["coverage"]["expected_per_order"],
                         {cid: self.case(cid).frequency for cid in ROWS})
        self.assertEqual({p["case_id"] for p in manifest["payloads"]}, ROWS)
        self.assertEqual(len(manifest["payloads"]), 4 * sum(self.case(c).frequency for c in ROWS))
        self.assertEqual(identity["case_selection"], {
            "rule": "benchmark", "definition": self.base.BENCHMARK_CASES,
            "search_set": {"schema": search_set.SCHEMA,
                           "sha256": self.base.sha_file(self.search_set), "snapshot_hash": "tiny"},
            "not_recorded": sorted({c.case_id for c in tiny_cases()} - ROWS)})
        self.assertEqual(set(identity["references"]), {"linear_no_bias", "qwen3_swiglu_mlp", "rope"})

    def test_it_holds_the_full_corpus_records_of_its_cases_under_another_identity(self):
        from tests.verification.test_integration_qwen3 import tiny_cases

        bench = json.loads(Path(self.benchmark).read_text())
        full = json.loads(Path(self.full).read_text())
        self.assertEqual(full["identity"]["case_selection"],
                         {"rule": "all", "definition": "every observed case"})
        self.assertEqual(set(full["identity"]["cases"]), {c.case_id for c in tiny_cases()})
        self.assertTrue(full["coverage"]["complete"])
        # The same calls, call indices and stored bytes: only the other cases are left out.
        self.assertEqual(bench["payloads"], [p for p in full["payloads"] if p["case_id"] in ROWS])
        kept = {p["payload_id"] for p in bench["payloads"]}
        self.assertEqual(bench["splits"], {name: [i for i in ids if i in kept]
                                           for name, ids in full["splits"].items()})
        self.assertNotEqual(bench["corpus_id"], full["corpus_id"])

    def test_every_reader_works_on_it(self):
        from evograd.evaluation.protocol import TIER2_INPUTS, tier2_payloads
        from evograd.verification.integrations.gates import judge_case, open_run
        from evograd.verification.payloads.gate1 import search_payload_ids
        from tests.verification.test_integration_qwen3 import linear_pair

        q = self.case("linear_no_bias/q")
        op = self.model.reference_op(q.task)
        with self.environment():
            fidelity = self.base.replay_fidelity(self.model, self.benchmark, device="cpu",
                                                 log=lambda *a: None)
            self.assertEqual(fidelity["payloads"], len(self.corpus.payloads()))
            self.assertTrue(fidelity["layouts_ok"] and fidelity["outputs_bitwise"])
            for scale, expected in ((1.0, "passed"), (1.05, "failed")):
                reports = judge_case(self.model, q, linear_pair(scale), self.corpus, device="cpu",
                                     compile_backend="aot_eager")
                self.assertEqual({name: r.status for name, r in reports.items()},
                                 {"gate1_payloads": expected,
                                  "gate2_operator_calibration": "calibrated",
                                  "gate2_operator": expected}, scale)
            ids = search_payload_ids(self.corpus, q.case_id)
            self.assertEqual(len(ids), 4)
            self.assertTrue(set(ids) <= set(self.corpus.split("calibration")))
            timed = tier2_payloads(self.corpus, op, q)
            self.assertEqual([r["batch"]["order"] for r in timed], [TIER2_INPUTS["order"]] * 2)
            run = open_run(self.model, {q.case_id: str(self.program)}, self.benchmark,
                           self.root / "run", device="cpu")
            self.assertEqual(run.corpus.corpus_id, self.corpus.corpus_id)
            pack = self.pack("linear_no_bias/q")
        self.assertEqual(pack.spec["search_correctness"]["payload_ids"], ids)
        self.assertEqual(pack.spec["search_correctness"]["corpus"]["corpus_id"],
                         self.corpus.corpus_id)

    def pack(self, case_id):
        """A task pack from the corpus; only the envelope's calibration is stubbed."""
        from types import SimpleNamespace

        from evograd.adbench_tools.task import build_task
        from evograd.verification.integrations import gates

        def calibration(model, case, corpus, **_):
            policy = SimpleNamespace(gamma=4.0, thresholds={})  # a calibration with no envelope
            return SimpleNamespace(op=model.reference_op(case.task)), SimpleNamespace(policy=policy)

        with mock.patch.object(gates, "operator_calibration", calibration):
            return build_task(self.root / "packs" / case_id.replace("/", "__"), model="qwen3_0_6b",
                              case_id=case_id, corpus=self.benchmark, device="cpu")

    def test_a_case_it_does_not_record_is_refused_by_every_reader(self):
        from evograd.evaluation.protocol import tier2_payloads
        from evograd.verification.integrations.gates import judge_case, open_run
        from evograd.verification.payloads import Corpus, CorpusError, payload_cases
        from evograd.verification.payloads.gate1 import search_payload_ids
        from tests.verification.test_integration_qwen3 import linear_pair

        kv = self.case(LEFT_OUT)
        op = self.model.reference_op(kv.task)
        refused = rf"records no payload of '{LEFT_OUT}': it records 3 cases, the 'benchmark' selection"
        attempts = {
            "payloads": lambda: self.corpus.payloads(case_id=LEFT_OUT),
            "gate1": lambda: payload_cases(op, kv.workload, self.corpus, case_id=LEFT_OUT,
                                           split="acceptance", device="cpu"),
            "gate2": lambda: judge_case(self.model, kv, linear_pair(1.0), self.corpus,
                                        device="cpu", compile_backend="aot_eager"),
            "search payloads": lambda: search_payload_ids(self.corpus, LEFT_OUT),
            "tier2": lambda: tier2_payloads(self.corpus, op, kv),
            "open_run": lambda: open_run(self.model, {LEFT_OUT: str(self.program)},
                                         self.benchmark, self.root / "refused", device="cpu"),
            "pack": lambda: self.pack(LEFT_OUT),
        }
        with self.environment():
            for name, attempt in attempts.items():
                with self.subTest(name), self.assertRaisesRegex(CorpusError, refused):
                    attempt()
        self.assertFalse((self.root / "packs" / LEFT_OUT.replace("/", "__")).exists())
        self.assertEqual(len(Corpus.open(self.full).payloads(case_id=LEFT_OUT)),
                         4 * kv.frequency)

    def test_a_selection_that_cannot_be_applied_is_refused_before_anything_is_written(self):
        files = {
            "stale": (search_set_file(self.root / "stale.json", snapshot_hash="older"),
                      ValueError, "generated from snapshot_hash older, not the tracked tiny"),
            "no entry": (search_set_file(self.root / "other.json", model="llama_3_2_1b"),
                         KeyError, "no entry in the search set"),
            "unobserved": (search_set_file(self.root / "unobserved.json", extra=["rope/gone"]),
                           ValueError, "not observed cases"),
            "empty": (search_set_file(self.root / "empty.json", empty=True),
                      ValueError, "has no benchmark row"),
        }
        for label, (path, error, message) in files.items():
            with self.subTest(label), self.environment(search_set=path), \
                    self.assertRaisesRegex(error, message):
                self.base.capture(self.model, self.root / label, device="cpu",
                                  log=lambda *a: None)
            self.assertFalse((self.root / label).exists())
        with self.environment(), self.assertRaisesRegex(ValueError, "not one of"):
            self.base.capture(self.model, self.root / "some", selection="some", device="cpu")
        self.assertFalse((self.root / "some").exists())

    def test_the_command_line_records_the_benchmark_rows_unless_told_otherwise(self):
        from evograd.verification.integrations import cli

        calls = []

        def capture(model, out, **options):
            calls.append(options["selection"])
            return Path(out) / "manifest.json"

        with mock.patch.object(self.base, "capture", capture), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(self.model, ["capture", "--out", "x"]), 0)
            self.assertEqual(cli.main(self.model, ["capture", "--out", "x", "--cases", "all"]), 0)
            with self.assertRaises(SystemExit):
                cli.main(self.model, ["capture", "--out", "x", "--cases", "some"])
        self.assertEqual(calls, ["benchmark", "all"])

    def test_verify_fails_when_its_combination_is_not_accepted(self):
        from evograd.verification.integrations import cli, gates

        argv = ["verify", "--corpus", "c.json", "--out", "o", "--candidate", "t/1=p.py"]
        for combination, code in ((None, 0), ({"accepted": True}, 0), ({"accepted": False}, 1)):
            verdict = {"candidates": {"t/1": {"accepted": True}}, "combination": combination}
            with self.subTest(combination=combination), \
                    mock.patch.object(gates, "verify", lambda *a, **k: verdict), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(self.model, argv), code)

    def test_verify_takes_the_native_reuse_options(self):
        from evograd.verification import reuse
        from evograd.verification.integrations import cli, gates

        argv = ["verify", "--corpus", "c.json", "--out", "o", "--candidate", "t/1=p.py"]
        seen = []

        def verify(*args, **kwargs):
            seen.append(reuse.settings()["share_gate3_native"])
            return {"candidates": {"t/1": {"accepted": True}}, "combination": None}

        with mock.patch.object(gates, "verify", verify), contextlib.redirect_stdout(io.StringIO()):
            cli.main(self.model, argv)
            cli.main(self.model, argv + ["--share-gate3-native"])
        self.assertEqual(seen, [False, True])

    def test_verify_runs_with_expandable_segments_unless_told_otherwise(self):
        from evograd.verification.integrations import cli, gates
        from evograd.verification.integrations.base import CUDA_ALLOC_CONF

        argv = ["verify", "--corpus", "c.json", "--out", "o", "--candidate", "t/1=p.py"]
        verdict = {"candidates": {"t/1": {"accepted": True}}, "combination": None}
        for preset, expected in ((None, CUDA_ALLOC_CONF), ("max_split_size_mb:512", "max_split_size_mb:512")):
            env = {} if preset is None else {"PYTORCH_CUDA_ALLOC_CONF": preset}
            with self.subTest(preset=preset), mock.patch.dict(os.environ, env), \
                    mock.patch.object(gates, "verify", lambda *a, **k: verdict), \
                    contextlib.redirect_stdout(io.StringIO()):
                if preset is None:
                    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                cli.main(self.model, argv)
                self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], expected)


class OldRuleCorpus(unittest.TestCase):
    """A corpus whose identity lists every case it recorded and no ``case_selection``
    (MegaFold's, recorded before the rule) serves those cases and refuses others;
    one whose identity lists no cases is not checked."""

    def write(self, root: Path, identity: dict):
        from evograd.verification.payloads import CorpusWriter
        from evograd.verification.payloads.corpus import Corpus

        writer = CorpusWriter(root)
        writer.add_payload({"payload_id": "o101/t/a/p", "case_id": "t/a", "task": "t", "level": 1,
                            "batch": {"order": 101}, "call": {"path": "p", "index_in_step": 0},
                            "args": {"x": writer.put(torch.ones(2))}, "cotangents": {},
                            "cotangent_stats": {}, "live_outputs": {}, "aliases": []})
        return Corpus.open(writer.finish(identity=identity, splits={"calibration": ["o101/t/a/p"]},
                                         coverage={"complete": True}))

    def test_old_rule(self):
        from evograd.verification.payloads import CorpusError

        with tempfile.TemporaryDirectory() as tmp:
            old = self.write(Path(tmp) / "old", {"name": "old", "cases": {"t/a": {"recorded": 1}}})
            self.assertEqual(len(old.payloads(case_id="t/a", split="calibration")), 1)
            self.assertTrue(old.records_case("t/a") and not old.records_case("t/b"))
            with self.assertRaisesRegex(CorpusError, "records 1 cases, every case observed at its "
                                                     "capture"):
                old.payloads(case_id="t/b")
            bare = self.write(Path(tmp) / "bare", {"name": "bare"})
            self.assertEqual(bare.payloads(case_id="t/b"), [])


if __name__ == "__main__":
    unittest.main()
