"""Tier 2 times recorded training calls, never generated inputs (CPU).

* the calls are the case's every call in the training step of one fixed data
  order, in call order, whatever the candidate; a missing or incomplete step is
  an error, not a cue to generate;
* the fair timer gives each provider its own copy of every call with the
  recorded strides, storage offsets and aliasing, loaded before anything is
  timed;
* the speedup is the ratio of per-call medians summed over the step;
* every baseline is timed on the same calls, and a declared library baseline is
  checked on a recorded call, not on generated inputs, at the scale of the
  recorded gradients;
* an unreadable recorded call stops Tier 2 instead of becoming a baseline's error.

CUDA events are replaced by a fixed clock: what is under test is which values
reach the providers, not how long they take.
"""

from __future__ import annotations

import contextlib
import tempfile
import unittest
import warnings
from functools import partial
from pathlib import Path
from unittest import mock

warnings.filterwarnings("ignore")

CASE_ID = "swiglu/act"
ROWS, COLS = 8, 16


def write_corpus(root, steps, *, cotangent_scale=1.0):
    """One recorded ``swiglu`` call per ``(order, index_in_step)``, with the
    layouts a model hands over: ``a`` and ``b`` interleaved views of one buffer
    (offset, gapped strides, aliasing) and ``dc`` a transposed view, scaled as
    small as training's output gradients when asked."""
    import torch

    from evograd.verification.payloads import Corpus, CorpusWriter

    writer = CorpusWriter(root)
    members: dict[str, list[str]] = {"calibration": [], "acceptance": []}
    for order, index in steps:
        writer.new_step()
        g = torch.Generator().manual_seed(order * 100 + index)
        base = torch.randn(ROWS + 1, 2 * COLS, generator=g)
        a, b = base[1:, ::2], base[1:, 1::2]
        dc = (torch.randn(COLS, ROWS, generator=g) * cotangent_scale).t()
        payload_id = f"o{order}/{CASE_ID}/act#{index}"
        writer.add_payload({
            "payload_id": payload_id, "case_id": CASE_ID, "task": "swiglu", "level": 2,
            "batch": {"order": order},
            "call": {"index_in_step": index, "role": "act", "layer": None, "path": f"act#{index}"},
            "args": {"a": writer.put(a), "b": writer.put(b)}, "cotangents": {"dc": writer.put(dc)},
            "aliases": [["a", "b"]]})
        members["acceptance" if order in (303, 404) else "calibration"].append(payload_id)
    return Corpus.open(writer.finish(identity={"name": "tier2-test"}, splits=members, coverage={}))


def case(frequency):
    from evograd.opdecl.activity import Workload
    from evograd.verification.integrations.base import Case

    return Case(CASE_ID, "swiglu", 2, Workload(dims={"rows": ROWS, "cols": COLS}, dtype="float32"),
                frequency)


def layout(t):
    return (tuple(t.shape), tuple(t.stride()), t.storage_offset(), t.dtype)


@contextlib.contextmanager
def fixed_clock():
    """The fair timer on CPU: every event pair reads one millisecond."""
    import torch

    class Event:
        def __init__(self, enable_timing=False):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    class NoCache:
        def clear(self):
            pass

    with mock.patch("evograd.evaluation.tier1.fair.L2Cache", NoCache), \
            mock.patch.object(torch.cuda, "Event", Event), \
            mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
            mock.patch.object(torch.cuda, "empty_cache", lambda *a, **k: None), \
            mock.patch.object(torch.cuda, "get_device_name", lambda *a, **k: "cpu"):
        yield


def forbidden(*args, **kwargs):
    raise AssertionError("Tier 2 generated inputs")


def swiglu():
    """The ``swiglu`` declaration with its own input hook forbidden too."""
    from dataclasses import replace

    from evograd.benchmark import get_task

    return replace(get_task("swiglu"), make_inputs=forbidden)


@contextlib.contextmanager
def no_generator():
    """``make_case_inputs`` raises, under every name a loaded module holds it by,
    including modules first imported inside the block."""
    import sys

    from evograd.opdecl import inputs

    def holding(value):
        # Read each namespace directly: a lazy module's __getattr__ would import.
        return [m for m in list(sys.modules.values())
                if (getattr(m, "__dict__", None) or {}).get("make_case_inputs") is value]

    real = inputs.make_case_inputs
    for module in holding(real):
        module.make_case_inputs = forbidden
    try:
        yield
    finally:
        for module in holding(forbidden):
            module.make_case_inputs = real


class Selection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        # Recorded out of call order, in two acceptance orders and a calibration
        # one; the call indices do not sort as text the way they sort as numbers.
        cls.corpus = write_corpus(root / "full", [(303, 10), (303, 2), (404, 0), (101, 0),
                                                  (303, 1), (404, 1), (404, 2)])
        cls.no_303 = write_corpus(root / "no303", [(404, 0), (404, 1), (404, 2)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_call_of_the_fixed_step_in_call_order(self):
        from evograd.evaluation.protocol import TIER2_INPUTS, tier2_binding, tier2_payloads

        records = tier2_payloads(self.corpus, swiglu(), case(3))
        self.assertEqual([r["payload_id"] for r in records],
                         [f"o303/{CASE_ID}/act#{i}" for i in (1, 2, 10)])
        binding = tier2_binding(self.corpus, records)
        self.assertEqual(binding["payload_ids"], [r["payload_id"] for r in records])
        self.assertEqual(binding["corpus"], self.corpus.binding())
        self.assertEqual({k: binding[k] for k in TIER2_INPUTS}, TIER2_INPUTS)

    def test_a_missing_or_incomplete_step_is_an_error(self):
        from evograd.evaluation.protocol import tier2_payloads
        from evograd.verification.payloads import CorpusError

        op = swiglu()
        with self.assertRaisesRegex(CorpusError, "records 0 calls"):
            tier2_payloads(self.no_303, op, case(3))
        with self.assertRaisesRegex(CorpusError, "records 3 calls .* makes 4 per step"):
            tier2_payloads(self.corpus, op, case(4))

    def test_a_call_that_does_not_match_the_declaration_is_an_error(self):
        from dataclasses import replace

        from evograd.evaluation.protocol import tier2_payloads
        from evograd.verification.payloads import CorpusError

        with self.assertRaisesRegex(CorpusError, "is for task 'swiglu'"):
            tier2_payloads(self.corpus, replace(swiglu(), name="other"), case(3))


class Timer(unittest.TestCase):
    """The fair timer itself, on the recorded calls."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus = write_corpus(Path(cls.tmp.name) / "c", [(303, 0), (303, 1), (303, 2)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_provider_times_its_own_layout_exact_copy_of_every_call(self):
        from dataclasses import replace

        import torch

        from evograd.evaluation.common.provider import pytorch_autograd_provider, renamed_provider
        from evograd.evaluation.protocol import tier2_payloads
        from evograd.evaluation.tier1.fair import run_fair_benchmarks

        op = swiglu()
        records = tier2_payloads(self.corpus, op, case(3))
        events = []

        def load(record):
            events.append(("load", record["payload_id"]))
            return self.corpus.load(record, device="cpu")

        def recording(name):
            provider = renamed_provider(pytorch_autograd_provider(op), name)

            def forward(values):
                events.append(("forward", name, values))
                return provider.forward(values)

            return replace(provider, forward=forward)

        inputs = [(case(3).workload, partial(load, r)) for r in records]
        with fixed_clock(), no_generator():
            report = run_fair_benchmarks(op, recording("candidate"), recording("eager"),
                                         inputs=inputs, warmup=1, reps=2, blocks=2, seed=0)
        self.assertEqual(len(report["cases"]), 3)
        self.assertEqual(report["aggregate"]["speedup_pair_full"], 1.0)

        # One load per call, each before any provider runs on that call.
        loads = [i for i, e in enumerate(events) if e[0] == "load"]
        self.assertEqual([events[i][1] for i in loads], [r["payload_id"] for r in records])
        for position, record in zip(loads, records):
            expected = self.corpus.load(record, device="cpu")
            upto = next((i for i, e in enumerate(events[position + 1:]) if e[0] == "load"),
                        len(events))
            following = [e for e in events[position + 1:position + 1 + upto] if e[0] == "forward"]
            by_provider = {}
            for _kind, name, values in following:
                by_provider.setdefault(name, set()).add(id(values))
                for arg, want in expected.items():
                    self.assertEqual(layout(values[arg]), layout(want), (record["payload_id"], arg))
                    self.assertTrue(torch.equal(values[arg].detach(), want))
                self.assertEqual(values["a"].untyped_storage().data_ptr(),
                                 values["b"].untyped_storage().data_ptr(), "aliasing lost")
            # One copy per provider for the whole call, not shared between them.
            self.assertEqual({k: len(v) for k, v in by_provider.items()},
                             {"candidate": 1, "eager": 1})
            copies = {name: next(e[2] for e in following if e[1] == name) for name in by_provider}
            self.assertNotEqual(copies["candidate"]["a"].untyped_storage().data_ptr(),
                                copies["eager"]["a"].untyped_storage().data_ptr())

    def test_the_speedup_is_the_summed_baseline_over_the_summed_candidate(self):
        from evograd.evaluation.common.provider import pytorch_autograd_provider, renamed_provider
        from evograd.evaluation.tier1.fair import run_fair_benchmarks

        def call(candidate_ms, baseline_ms):
            """One call's fair result: two paired blocks of constant samples."""
            def blocks(ms):
                return [{"block_index": i, "pair_index": i,
                         "samples_ms": {m: [ms, ms] for m in ("forward", "backward", "full")}}
                        for i in range(2)]

            def summary(ms):
                return {m: {"median_ms": ms} for m in ("forward", "backward", "pair_full")}

            return {"providers": {"candidate": summary(candidate_ms), "eager": summary(baseline_ms)},
                    "raw_blocks": {"candidate": blocks(candidate_ms), "eager": blocks(baseline_ms)}}

        op = swiglu()
        eager = pytorch_autograd_provider(op)
        with fixed_clock(), mock.patch("evograd.evaluation.tier1.fair.benchmark_case_fair",
                                       side_effect=[call(1.0, 3.0), call(4.0, 4.0)]):
            aggregate = run_fair_benchmarks(
                op, renamed_provider(eager, "candidate"), renamed_provider(eager, "eager"),
                inputs=[(case(2).workload, dict)] * 2)["aggregate"]
        # (3 + 4) / (1 + 4), not the mean of the per-call ratios (3 and 1).
        for metric in ("pair_full", "forward", "backward"):
            self.assertAlmostEqual(aggregate[f"speedup_{metric}"], 7 / 5)
            interval = aggregate[f"speedup_{metric}_ci95"]
            self.assertAlmostEqual(interval["low"], 7 / 5)
            self.assertAlmostEqual(interval["high"], 7 / 5)


class TimeCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        # Output gradients as small as training's, far below any absolute tolerance.
        cls.corpus = write_corpus(Path(cls.tmp.name) / "c", [(303, 0), (303, 1)],
                                  cotangent_scale=1e-4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def library(self, scale):
        """``swiglu`` with a library baseline that is the candidate pair scaled."""
        from dataclasses import replace
        import types

        from tests.verification import _toy_integration as T

        op = swiglu()
        module = types.ModuleType(f"library_{scale}")
        exec(T.CANDIDATE.format(scale=scale), module.__dict__)
        hook = types.SimpleNamespace(
            pair_factory=lambda: (module.forward_with_saved, module.backward_from_saved),
            forward_args=("a", "b"), backward_extras=())
        return replace(op, performance_baselines={"library": hook})

    def inputs(self, op):
        from evograd.evaluation.protocol import tier2_payloads

        records = tier2_payloads(self.corpus, op, case(2))
        return [(case(2).workload, partial(self.corpus.load, r, device="cpu")) for r in records]

    def time(self, op, inputs, *, timer=True):
        """``time_case`` with the fair timer standing in (``timer``) or real on a
        fixed clock; torch.compile is eager, so CPU needs no compiler."""
        import types

        from evograd.evaluation.common.provider import pytorch_autograd_provider
        from evograd.evaluation.protocol import time_case
        from tests.verification import _toy_integration as T

        candidate = types.ModuleType("candidate")
        exec(T.CANDIDATE.format(scale=1.0), candidate.__dict__)
        seen = []

        def measure(op, candidate, baseline, *, inputs, **options):
            seen.append((baseline.name, inputs))
            return {"aggregate": {
                "speedup_pair_full": 1.2, "speedup_pair_full_ci95": {"low": 1.1, "high": 1.3},
                "candidate_pair_full_ms": 1.0, "baseline_pair_full_ms": 1.2,
                "speedup_forward": 1.1, "speedup_backward": 1.3}}

        timing = (mock.patch("evograd.evaluation.tier1.fair.run_fair_benchmarks", measure)
                  if timer else fixed_clock())
        with timing, \
                mock.patch("evograd.evaluation.common.provider.torch_compile_provider",
                           side_effect=lambda op, **kwargs: pytorch_autograd_provider(op)), \
                mock.patch("evograd.opdecl.baselines.verify_performance_baseline", forbidden), \
                no_generator():
            options = {"warmup": 1, "reps": 2, "blocks": 1, "seed": 0}
            return time_case(op, candidate, inputs, options=options), seen

    def test_every_baseline_is_timed_on_the_same_recorded_calls(self):
        op = self.library(1.0)
        inputs = self.inputs(op)
        result, seen = self.time(op, inputs)
        self.assertEqual(set(result), {"eager", "torch_compile", "library"})
        self.assertEqual(len(seen), 3)
        self.assertIn("library", [name for name, _ in seen])
        for _name, given in seen:
            self.assertIs(given, inputs)
        self.assertEqual(result["library"]["speedup"], 1.2)

    def test_a_library_baseline_that_differs_from_eager_on_a_recorded_call_is_not_timed(self):
        # A 5% gradient error, and gradients of zero: both far below the declared
        # absolute tolerance at the recorded scale, both rejected.
        for scale in (1.05, 0.0):
            with self.subTest(scale=scale):
                op = self.library(scale)
                result, seen = self.time(op, self.inputs(op))
                self.assertIn("differs from eager in 'da'", result["library"]["error"])
                self.assertEqual(len(seen), 2)
                self.assertNotIn("library", [name for name, _ in seen])
                self.assertEqual(result["eager"]["speedup"], 1.2)

    def test_an_unreadable_recorded_call_stops_tier2(self):
        from evograd.verification.payloads import CorpusError

        def unreadable():
            raise CorpusError("storage 0123: content does not match its hash")

        op = swiglu()
        inputs = self.inputs(op)
        inputs[1] = (inputs[1][0], unreadable)
        with self.assertRaisesRegex(CorpusError, "does not match its hash"):
            self.time(op, inputs, timer=False)


class Executable(unittest.TestCase):
    def test_wrong_output_shape_on_a_recorded_call_is_not_executable(self):
        import types

        from evograd.evaluation.protocol import NotExecutable, check_executable

        module = types.ModuleType("wrong_shape")
        exec('''
import torch
def forward_with_saved(a, b):
    return (torch.nn.functional.silu(a) * b)[:1], (a, b)
def backward_from_saved(dc, saved):
    a, b = saved
    return torch.zeros_like(a), torch.zeros_like(b)
''', module.__dict__)
        with tempfile.TemporaryDirectory() as tmp:
            corpus = write_corpus(Path(tmp) / "c", [(303, 0)])
            values = corpus.load(corpus.payloads()[0], device="cpu")
            with no_generator(), self.assertRaises(NotExecutable):
                check_executable(swiglu(), module, values)


if __name__ == "__main__":
    unittest.main()
