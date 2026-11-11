"""The committed search set: pinned to the tracked snapshots, well formed, and its reader.

``src/evograd/benchmark/search_set.json`` is generated from ``observed_cases`` and
the measurements under ``scripts/adbench_selection/``. A snapshot that changes
without regenerating it fails here; a snapshot the GPU harvest has yet to
regenerate (``tests/_snapshots.py``) skips its model until it lands.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from evograd.benchmark import search_set
from evograd.verification.integrations import PACKAGES
from tests._snapshots import STALE_SNAPSHOT, snapshot_is_current

REPO = Path(__file__).resolve().parents[1]
REGENERATE = "PYTHONPATH=src:. python scripts/adbench_selection/write_search_set.py"
_spec = importlib.util.spec_from_file_location(
    "write_search_set", REPO / "scripts" / "adbench_selection" / "write_search_set.py")
writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(writer)
DATA = search_set.load()


def _cases():
    for name, entry in DATA["models"].items():
        for cid, case in entry["cases"].items():
            yield name, cid, case


def test_every_model_and_no_other():
    assert DATA["schema"] == search_set.SCHEMA
    assert set(DATA["models"]) == set(PACKAGES)


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_pinned_to_the_tracked_snapshot(name):
    if not snapshot_is_current(name):
        pytest.skip(f"{name}: {STALE_SNAPSHOT}; then regenerate the search set: {REGENERATE}")
    from evograd.benchmark.topdown import load_snapshot
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import observed_cases

    model = load(name)
    entry = search_set.entry(name)
    assert entry["snapshot_hash"] == load_snapshot(model.name)["snapshot_hash"], (
        f"{name}'s snapshot changed since search_set.json was generated: {REGENERATE}")
    assert list(entry["cases"]) == [c.case_id for c in observed_cases(model)], REGENERATE


def test_the_generator_reproduces_the_file():
    stale = sorted(name for name in PACKAGES if not snapshot_is_current(name))
    if stale:
        pytest.skip(f"{', '.join(stale)}: {STALE_SNAPSHOT}; then regenerate: {REGENERATE}")
    assert writer.render(writer.select()) == search_set.PATH.read_text(encoding="utf-8"), REGENERATE


def test_each_case_is_searched_or_has_one_reason():
    for name, cid, case in _cases():
        where = f"{name} {cid}"
        if case["searched"]:
            assert case["reason"] is None and case["of"] is None, where
        else:
            assert case["reason"] in search_set.REASONS, where
        assert (case["task"] in writer.VENDOR) == (case["reason"] == "vendor"), where
        assert (case["task"] in writer.REMOVED) == (case["reason"] == "removed"), where
        if case["reason"] == "removed":
            assert case["note"], where
        if case["reason"] not in ("identical_config_of", "nested_in"):
            assert case["of"] is None, where


def test_targets_exist_and_are_searched():
    for name, cid, case in _cases():
        where = f"{name} {cid}"
        if case["reason"] == "identical_config_of":
            model, source = case["of"].split(":", 1)
            target = DATA["models"][model]["cases"][source]
            assert model != name and target["searched"], where
            assert (target["task"], target["level"]) == (case["task"], case["level"]) == (case["task"], 1)
            assert target["evidence"]["calls"] >= case["evidence"]["calls"], where
        if case["reason"] == "nested_in":
            cases = DATA["models"][name]["cases"]
            target = cases[case["of"]]
            assert target["level"] == case["level"] == 2, where
            # The containing configuration, or its task's searched one when it is not.
            assert any(c["searched"] for c in cases.values() if c["task"] == target["task"]), where


def test_one_benchmark_case_per_task_the_time_max():
    for name, entry in DATA["models"].items():
        tasks = {}
        for cid, case in entry["cases"].items():
            if case["reason"] not in ("vendor", "removed", "nested_in"):
                tasks.setdefault(case["task"], []).append(case)
        for task, group in tasks.items():
            main = [c for c in group if c["reason"] != "not_time_max_config"]
            assert len(main) == 1, f"{name} {task}"
            key = "ms_x_calls" if main[0]["level"] == 1 else "wall_x_calls"
            if len(group) > 1:
                assert main[0]["evidence"][key] == max(c["evidence"][key] for c in group), f"{name} {task}"


def test_level2_share_threshold_and_exceptions():
    threshold = DATA["inputs"]["l2_share"]["threshold_pct"]
    for name, cid, case in _cases():
        if case["level"] != 2 or case["reason"] not in (None, "optimizable_share_below_1pct"):
            continue
        share = case["evidence"]["optimizable_share_pct"]
        if case["reason"] == "optimizable_share_below_1pct":
            assert share < threshold, f"{name} {cid}"
        elif share < threshold:
            assert case["task"] in writer.EXCEPTIONS and case["note"].startswith("exception"), cid


def test_reader(tmp_path):
    path = tmp_path / "search_set.json"
    path.write_text(json.dumps({"schema": search_set.SCHEMA, "models": {"a": {
        "snapshot_hash": "h", "cases": {
            "t/1": {"searched": True, "reason": None, "of": None},
            "t/2": {"searched": False, "reason": "not_time_max_config", "of": None},
            "u/1": {"searched": False, "reason": "identical_config_of", "of": "b:u/9"}}}}}))
    assert search_set.searched("a", path) == ["t/1"]
    assert search_set.evaluated("a", path) == {"t/1": None, "u/1": ("b", "u/9")}
    assert search_set.row_tag("a", "u/1", path) == {
        "searched": False, "reason": "identical_config_of", "of": "b:u/9"}
    assert search_set.row_tag("toy", "t/1", path) == {}
    with pytest.raises(ValueError, match="predates the snapshot"):
        search_set.row_tag("a", "t/3", path)
    with pytest.raises(KeyError):
        search_set.searched("toy", path)
    assert search_set.benchmark_row({"searched": False, "reason": "identical_config_of"})
    assert not search_set.benchmark_row({"searched": False, "reason": "vendor"})
    assert search_set.benchmark_row({"task": "t"})
