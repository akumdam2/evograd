"""T3-G: the Tier 3 step replayed from one CUDA graph, beside the eager step.

The report wiring runs on CPU; capture and the replay check need a GPU.
"""
import pytest
import torch
from torch import nn

from evograd.evaluation import protocol as P
from evograd.evaluation.tier3 import KernelSet, measure_one
from evograd.evaluation.tier3.runner import (CUDA_GRAPH, assemble_report, measure_cuda_graph,
                                             speedup_intervals)
from tests._registry_fixture import SAMPLE_SITES
from tests.test_tier3_hardening import _toy_options, _ToyWorkload

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _row(ms, graph_ms=None, graph_ok=True, **extra):
    row = {"ok": True, "step_ms": ms, "per_block_ms": [ms] * 3, "peak_memory_bytes": 100, **extra}
    if graph_ms is not None:
        row[CUDA_GRAPH] = {"ok": graph_ok, "step_ms": graph_ms, "per_block_ms": [graph_ms] * 3,
                           "capture_s": 0.2, "check": {"agrees": graph_ok}}
    return row


def _report(raw):
    return assemble_report(_ToyWorkload(), raw, list(raw), verify=False, isolation="test",
                           **P.TIER3_OPTIONS)


def test_a_failed_capture_leaves_the_eager_row_standing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    entry = measure_one(_ToyWorkload(), "eager", KernelSet(registry=SAMPLE_SITES),
                        cuda_graph=True, **_toy_options())
    assert entry["ok"] and entry["step_ms"] > 0
    assert entry[CUDA_GRAPH] == {"ok": False, "failed_at": "capture",
                                 "error": "CUDA graphs need a CUDA device"}


def test_graph_rows_are_compared_with_the_graph_reference():
    report = _report({"eager": _row(100, 80), "torch_compile": _row(60, 50),
                      "a": _row(80, 55), "b": _row(90, 70, graph_ok=False)})
    assert CUDA_GRAPH in report["timing_protocol"]
    eager = speedup_intervals(report)["vs_reference"]
    graph = report["cuda_graph_speedup_intervals"]["vs_reference"]
    assert eager["a"]["step_ms_ratio"] == pytest.approx(100 / 80)
    assert graph["a"]["step_ms_ratio"] == pytest.approx(80 / 55)
    assert "b" not in graph and "b" in eager
    row = P._tier3_row(report, "a")
    assert row["speedup"] == pytest.approx(100 / 80)
    assert row[CUDA_GRAPH]["speedup"] == pytest.approx(80 / 55)
    assert row[CUDA_GRAPH]["capture_s"] == 0.2
    assert P._tier3_row(report, "b")[CUDA_GRAPH] == {"ok": False, "step_ms": 70, "capture_s": 0.2,
                                                      "check": {"agrees": False}}


def test_speedups_against_compile_in_each_execution():
    compiled = P._tier3_row(_report({"eager": _row(100, 80),
                                     "torch_compile": _row(60, 50)}),
                            "torch_compile")
    own = {"ok": True, "step_ms": 75, CUDA_GRAPH: {"ok": True, "step_ms": 55}}
    assert P.speedups_vs_compile(own, compiled) == pytest.approx(
        {"speedup_vs_torch_compile": 60 / 75, "cuda_graph_speedup_vs_torch_compile": 50 / 55})
    own[CUDA_GRAPH]["ok"] = False
    assert P.speedups_vs_compile(own, compiled) == pytest.approx({"speedup_vs_torch_compile": 60 / 75})
    assert P.speedups_vs_compile(own, {**compiled, "ok": False}) == {}


def test_without_cuda_graph_the_report_is_unchanged():
    options = {**P.TIER3_OPTIONS, "cuda_graph": False}
    report = assemble_report(_ToyWorkload(), {"eager": _row(100)}, ["eager"], verify=False,
                             isolation="test", **options)
    assert CUDA_GRAPH not in report["timing_protocol"]
    assert "cuda_graph_speedup_intervals" not in report
    assert CUDA_GRAPH not in P._tier3_row(report, "eager")


# ── on a GPU ─────────────────────────────────────────────────────────────────


class _Workload:
    """``loss`` of a small MLP; ``capture_shift`` changes it only while capturing."""

    def __init__(self, capture_shift=0.0, host_sync=False):
        self.capture_shift, self.host_sync = capture_shift, host_sync

    def loss(self, model, batch):
        loss = model(batch).square().mean()
        if self.host_sync:
            loss = loss * (1.0 + 0.0 * loss.item())
        if torch.cuda.is_current_stream_capturing():
            loss = loss + self.capture_shift
        return loss


def _setup():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer, torch.randn(32, 64, device="cuda")


@cuda
def test_replay_reproduces_eager_and_is_timed():
    model, optimizer, batch = _setup()
    before = [p.detach().clone() for p in model.parameters()]
    row = measure_cuda_graph(_Workload(), model, optimizer, batch, warmup=2, steps=3, blocks=2)
    assert row["ok"], row
    assert row["check"]["agrees"] and row["check"]["replay"] == {"loss": 0.0, "grad": 0.0}
    assert len(row["check"]["eager_repeats"]) == 2
    assert row["step_ms"] > 0 and len(row["per_block_ms"]) == 2
    # timing trains: the replayed gradients reach the optimizer
    assert any(not torch.equal(p, b) for p, b in zip(model.parameters(), before))
    assert all(p.grad is None for p in model.parameters())


@cuda
def test_a_step_that_changes_under_capture_is_not_timed():
    model, optimizer, batch = _setup()
    row = measure_cuda_graph(_Workload(capture_shift=1.0), model, optimizer, batch,
                             warmup=2, steps=3, blocks=2)
    assert not row["ok"] and row["failed_at"] == "replay_check"
    assert row["check"]["replay"]["loss"] == pytest.approx(1.0)
    assert "step_ms" not in row


@cuda
def test_a_host_sync_fails_the_capture_only():
    model, optimizer, batch = _setup()
    row = measure_cuda_graph(_Workload(host_sync=True), model, optimizer, batch,
                             warmup=2, steps=3, blocks=2)
    assert not row["ok"] and row["failed_at"] == "capture", row
