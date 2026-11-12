"""candidates/bench.py reads the search set: ``tasks --searched``, ``run``'s default
cases and ``evaluate``'s identical_config_of reuse, without a GPU or an agent."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evograd.adbench_tools import task as task_module
from evograd.benchmark import search_set

REPO = Path(__file__).resolve().parents[2]
SEARCHED = {"searched": True, "reason": None, "of": None}
CASES = {"t/1": SEARCHED, "t/2": {"searched": False, "reason": "not_time_max_config", "of": None},
         "u/1": SEARCHED, "v/1": {"searched": False, "reason": "identical_config_of", "of": "b:v/9"}}


@pytest.fixture
def bench(tmp_path, monkeypatch):
    path = tmp_path / "search_set.json"
    path.write_text(json.dumps({"schema": search_set.SCHEMA, "models": {
        "a": {"snapshot_hash": "h", "cases": CASES},
        "b": {"snapshot_hash": "h", "cases": {"v/9": SEARCHED}}}}))
    monkeypatch.setattr(search_set, "PATH", path)
    spec = importlib.util.spec_from_file_location("bench_under_test", REPO / "candidates" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tasks_searched_builds_a_pack_for_each_searched_case(bench, tmp_path, monkeypatch):
    built = []

    def build_task(directory, *, model, case_id, library, corpus, state):
        built.append((model, case_id))
        return SimpleNamespace(task_id=case_id, allowed_primitives=(),
                               spec={"level": 1, "search_correctness": {"rule": "gate1"}})

    monkeypatch.setattr(task_module, "build_task", build_task)
    monkeypatch.setattr(bench, "model_cases", lambda model: dict.fromkeys(CASES))
    assert bench.main(["tasks", "--model", "a", "--searched", "--root", str(tmp_path)]) == 0
    assert built == [("a", "t/1"), ("a", "u/1")]


def test_run_defaults_to_the_searched_cases(bench, tmp_path, monkeypatch):
    root = tmp_path / "a"
    for cid in CASES:
        (root / "tasks" / bench.case_dir(cid)).mkdir(parents=True)
        (root / "tasks" / bench.case_dir(cid) / "task.json").write_text("{}")
    launched = []

    def run(command, **kwargs):
        run_dir = Path(command[command.index("--run-dir") + 1])
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"status": "completed"}))
        launched.append(run_dir.name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(task_module, "load", lambda directory: SimpleNamespace(root=directory))
    monkeypatch.setattr(bench.subprocess, "run", run)
    monkeypatch.setattr(bench, "submit", lambda task, run_dir, agent: {
        "submission": str(run_dir / "submission.py"), "status": "completed"})
    assert bench.main(["run", "--model", "a", "--agent", "seed", "--root", str(tmp_path)]) == 0
    assert launched == ["t__1", "u__1"]


def _record(root, model, cid, submission):
    run_dir = root / model / "runs" / "seed" / cid.replace("/", "__")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "submission.json"
    path.write_text(json.dumps({"agent": "seed", "case_id": cid, "status": "completed",
                                "submission": submission and str(run_dir / "submission.py")}))
    return path


def test_evaluate_reuses_the_source_submission_and_records_a_search_failure(
        bench, tmp_path, monkeypatch):
    launched = []
    monkeypatch.setattr(bench.subprocess, "run",
                        lambda command, **kwargs: launched.append(command) or
                        SimpleNamespace(returncode=0))

    def evaluate(*extra):
        launched.clear()
        assert bench.main(["evaluate", "--model", "a", "--agent", "seed", "--corpus",
                           str(tmp_path / "manifest.json"), "--root", str(tmp_path), *extra]) == 0
        (command,) = launched
        candidates = dict(command[i + 1].split("=", 1) for i, flag in enumerate(command)
                          if flag == "--candidate")
        selection = json.loads((tmp_path / "a" / "evaluation" / "seed" / "selection.json").read_text())
        assert selection["candidates"] == candidates
        return candidates, selection

    own = _record(tmp_path, "a", "t/1", True).with_name("submission.py")
    _record(tmp_path, "a", "u/1", False)
    source = _record(tmp_path, "b", "v/9", True)
    candidates, selection = evaluate()
    assert candidates == {"t/1": str(own), "v/1": str(source.with_name("submission.py"))}
    assert selection["reused_from"] == {"v/1": "b:v/9"}
    assert set(selection["search_failures"]) == {"u/1"}
    # The source search finished without a submission: a search failure here too.
    _record(tmp_path, "b", "v/9", False)
    candidates, selection = evaluate()
    assert candidates == {"t/1": str(own)}
    assert Path(selection["search_failures"]["v/1"]["record"]).resolve() == source.resolve()
    # No finished source run: refused, unless the case is left out.
    source.unlink()
    with pytest.raises(SystemExit, match="no finished run"):
        evaluate()
    assert evaluate("--case-id", "t/1")[0] == {"t/1": str(own)}


def test_tasks_fails_on_a_case_without_an_operator_gate2_calibration(bench, tmp_path, monkeypatch,
                                                                     capsys):
    # A failed calibration is reported with its reason after the other packs are built,
    # and a pack kept from before without one is refused as well.
    built = []

    def build_task(directory, *, model, case_id, library, corpus, state):
        if case_id == "u/1":
            raise task_module.CalibrationFailed("u/1: operator Gate 2 calibration is incomplete "
                                                "(insufficient_calibration: out of memory)")
        built.append(case_id)
        return SimpleNamespace(task_id=case_id, allowed_primitives=(),
                               spec={"level": 1, "search_correctness": {"rule": "gate1"}})

    monkeypatch.setattr(task_module, "build_task", build_task)
    monkeypatch.setattr(bench, "model_cases", lambda model: dict.fromkeys(CASES))
    assert bench.main(["tasks", "--model", "a", "--searched", "--root", str(tmp_path)]) == 1
    assert built == ["t/1"]
    assert "insufficient_calibration: out of memory" in capsys.readouterr().err

    kept = bench.task_dirs(bench.root_for("a", str(tmp_path)), ["t/1"])["t/1"]
    kept.mkdir(parents=True)
    (kept / "task.json").write_text("{}")
    monkeypatch.setattr(task_module, "load", lambda directory: SimpleNamespace(
        task_id="t/1", spec={"case_id": "t/1", "model": "a",
                             "search_correctness": {"calibrated": False}}))
    assert bench.main(["tasks", "--model", "a", "--case-id", "t/1", "--root", str(tmp_path)]) == 1
    assert "has no operator Gate 2 calibration" in capsys.readouterr().err
