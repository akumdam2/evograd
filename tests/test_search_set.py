"""The committed search set: pinned to the tracked snapshots, well formed, the user's
decisions, its generator's refusals and its reader.

``src/evograd/benchmark/search_set.json`` is generated from ``observed_cases`` and
the measurements under ``scripts/adbench_selection/``. A snapshot that changes
without regenerating it fails here; a snapshot the GPU harvest has yet to
regenerate (``tests/_snapshots.py``) skips its model until it lands. The
decided outcomes are spelled out below, not read from the generator.
"""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from evograd.benchmark import search_set
from evograd.verification.integrations import PACKAGES
from tests._snapshots import STALE_SNAPSHOT, snapshot_is_current

REPO = Path(__file__).resolve().parents[1]
REGENERATE = ("regenerate it: scripts/adbench_selection/write_search_set.py "
              "(the command is in src/evograd/benchmark/README.md)")
_spec = importlib.util.spec_from_file_location(
    "write_search_set", REPO / "scripts" / "adbench_selection" / "write_search_set.py")
writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(writer)
DATA = search_set.load()

#: Decided 2026-11-10: vendor primitives (GEMM, SDPA) are not searched.
VENDOR_TASKS = {"causal_gqa_attention", "linear", "linear_no_bias"}
#: (model, task) searched whatever their optimizable share, and what their note says.
ABLATION = {("qwen3_0_6b", "qwen3_swiglu_mlp"), ("llama_3_2_1b", "llama3_swiglu_mlp"),
            ("qwen3_5_4b", "qwen3_5_swiglu_mlp"), ("qwen2_vl_2b", "qwen2_vl_2b_text_mlp")}
DLRM_INTERACTION = ("torchrec_dlrm", "torchrec_dlrm_dlrm_interaction")


def _cases():
    for name, entry in DATA["models"].items():
        for cid, case in entry["cases"].items():
            yield name, cid, case


def _task(model, task):
    return {cid: case for cid, case in DATA["models"][model]["cases"].items()
            if case["task"] == task}


def _target(of):
    model, cid = of.split(":", 1)
    return DATA["models"][model]["cases"][cid]


def test_every_model_and_no_other():
    assert DATA["schema"] == search_set.SCHEMA
    assert set(DATA["models"]) == set(PACKAGES)


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_pinned_to_the_tracked_snapshot(name):
    if not snapshot_is_current(name):
        pytest.skip(f"{name}: {STALE_SNAPSHOT}; then {REGENERATE}")
    from evograd.benchmark.topdown import load_snapshot
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import observed_cases

    model = load(name)
    entry = search_set.entry(name)
    assert entry["snapshot_hash"] == load_snapshot(model.name)["snapshot_hash"], (
        f"{name}'s snapshot changed since search_set.json was generated: {REGENERATE}")
    assert list(entry["cases"]) == [c.case_id for c in observed_cases(model)], REGENERATE


def test_the_generator_reproduces_the_file():
    """``--check`` as a test. A model whose tracked snapshot changed since the file
    was generated is left out until it is regenerated; every other model must match."""
    from evograd.benchmark.topdown import load_snapshot
    from evograd.verification.integrations import load

    changed = sorted(name for name in PACKAGES if DATA["models"][name]["snapshot_hash"]
                     != load_snapshot(load(name).name)["snapshot_hash"])
    try:
        fresh = writer.select()
    except SystemExit as exc:
        if changed:
            pytest.skip(f"{', '.join(changed)}: the snapshot changed and the generator stops "
                        f"({exc}); {REGENERATE}")
        raise
    if not changed:
        assert writer.render(fresh) == search_set.PATH.read_text(encoding="utf-8"), REGENERATE
        return
    head = {k: v for k, v in fresh.items() if k != "models"}
    assert head == {k: v for k, v in DATA.items() if k != "models"}, REGENERATE
    for name in sorted(set(PACKAGES) - set(changed)):
        assert fresh["models"][name] == DATA["models"][name], f"{name}: {REGENERATE}"
    pytest.skip(f"{', '.join(changed)}: the snapshot changed since the file was generated; "
                f"{REGENERATE}")


def test_each_case_is_searched_or_has_one_reason():
    for name, cid, case in _cases():
        where = f"{name} {cid}"
        if case["searched"]:
            assert case["reason"] is None and case["of"] is None, where
        else:
            assert case["reason"] in search_set.REASONS, where
        assert (case["task"] in VENDOR_TASKS) == (case["reason"] == "vendor"), where
        assert (case["task"] == "megafold_layernorm_linear") == (case["reason"] == "removed"), where
        if case["reason"] == "removed":
            assert case["note"], where
        if case["reason"] not in ("identical_config_of", "nested_in"):
            assert case["of"] is None, where


def test_targets_exist_and_are_searched():
    for name, cid, case in _cases():
        where = f"{name} {cid}"
        if case["reason"] == "identical_config_of":
            target = _target(case["of"])
            assert case["of"].split(":")[0] != name and target["searched"], where
            assert (target["task"], target["level"]) == (case["task"], case["level"]) == (case["task"], 1)
            assert target["evidence"]["calls"] >= case["evidence"]["calls"], where
        if case["reason"] == "nested_in":
            target = DATA["models"][name]["cases"][case["of"]]
            assert target["level"] == case["level"] == 2 and target["searched"], where
            assert target["task"] != case["task"], where


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
    assert threshold == 1.0
    for name, cid, case in _cases():
        if case["level"] != 2 or case["reason"] not in (None, "optimizable_share_below_1pct"):
            continue
        share = case["evidence"]["optimizable_share_pct"]
        if case["reason"] == "optimizable_share_below_1pct":
            assert share < threshold, f"{name} {cid}"
        elif share < threshold:
            assert (name, case["task"]) in ABLATION | {DLRM_INTERACTION}, f"{name} {cid}"
            assert case["note"].startswith("exception"), f"{name} {cid}"


# ── the user's decisions (2026-11-10), by content ────────────────────────────


def test_the_two_identical_configurations():
    pairs = {(name, case["task"], case["of"].split(":")[0], _target(case["of"])["task"])
             for name, _cid, case in _cases() if case["reason"] == "identical_config_of"}
    assert pairs == {("llama_3_2_1b", "rmsnorm", "olmoe_1b_7b", "rmsnorm"),
                     ("olmoe_1b_7b", "rope", "qwen3_0_6b", "rope")}
    for model, task in (("llama_3_2_1b", "rmsnorm"), ("olmoe_1b_7b", "rope")):
        assert not any(c["searched"] for c in _task(model, task).values()), (model, task)


def test_below_one_percent():
    below = {(name, case["task"]) for name, _cid, case in _cases()
             if case["reason"] == "optimizable_share_below_1pct"}
    assert below == {("dit_s_2", "dit_s_2_dit_modulation"),
                     ("megafold", "megafold_outer_product_mean")}


def test_the_swiglu_mlp_ablation_group_and_dlrm_interaction_are_searched():
    named = {(name, case["task"]) for name, _cid, case in _cases()
             if case["level"] == 2 and case["task"].endswith("swiglu_mlp")}
    assert named <= ABLATION
    for model, task in sorted(ABLATION | {DLRM_INTERACTION}):
        (case,) = [c for c in _task(model, task).values() if c["searched"]]
        assert ("ablation" if task != DLRM_INTERACTION[1] else "DLRM") in case["note"], task
    (interaction,) = _task(*DLRM_INTERACTION).values()
    assert interaction["evidence"]["optimizable_share_pct"] < 1.0
    assert interaction["note"].startswith("exception")


def test_megafold_layernorm_linear_is_removed():
    removed = _task("megafold", "megafold_layernorm_linear")
    assert removed and all(c["reason"] == "removed" for c in removed.values())
    assert {name for name, _cid, case in _cases() if case["reason"] == "removed"} == {"megafold"}


def test_evoattention_searches_the_diffusion_token_transformer_and_nests_only_in_searched_cases():
    rows = _task("megafold", "megafold_evoattention")
    (searched,) = [cid for cid, c in rows.items() if c["searched"]]
    assert "diffusion_module.token_transformer" in searched
    msa, confidence = (next(c for cid, c in rows.items() if cid.startswith(f"megafold_evoattention/{part}")
                            and "tri_attn" in cid) for part in ("msa_module", "confidence_head"))
    assert msa["reason"] == "nested_in"
    assert msa["of"].startswith("megafold_triangle_attention_starting/msa_module")
    assert "megafold_triangle_attention_ending/msa_module" in msa["note"]
    # Its containing confidence-head triangle cases are not searched: no co-install conflict.
    assert confidence["reason"] == "not_time_max_config"


# ── the generator refuses a measurement that does not match ──────────────────


def _case(cid, task, dims, dtype="bfloat16", frequency=1):
    return SimpleNamespace(case_id=cid, task=task, frequency=frequency, paths=(cid,),
                           workload=SimpleNamespace(dims=dims, dtype=dtype))


def test_l1_timing_is_found_by_content_with_the_case_id_as_a_cross_check():
    rows = [{"model": "m", "task": "t", "case": "t/a", "dims": {"n": 1}, "dtype": "bfloat16", "ms": 1.0},
            {"model": "m", "task": "t", "case": "t/b", "dims": {"n": 2}, "dtype": "bfloat16", "ms": 3.0},
            {"model": "m", "task": "u", "case": "u/a", "dims": {"n": 1}, "dtype": "float32", "ms": 1.0},
            {"model": "m", "task": "u", "case": "u/b", "dims": {"n": 1}, "dtype": "float32", "ms": 2.0}]
    timing = writer.timing_index({"rows": rows})
    assert writer.timed_l1("m", _case("t/a", "t", {"n": 1}), timing)["ms"] == 1.0
    # A harvest that renames the id but keeps the configuration still resolves.
    renamed, _ = writer.level1("m", [_case("t/x", "t", {"n": 1}, frequency=4),
                                     _case("t/y", "t", {"n": 2})], timing)
    assert renamed["t/x"]["evidence"]["timed_as"] == "t/a" and renamed["t/x"]["searched"]
    # Two layouts of one content: the case id decides.
    assert writer.timed_l1("m", _case("u/b", "u", {"n": 1}, "float32"), timing)["ms"] == 2.0
    for case, match in ((_case("t/a", "t", {"n": 3}), "no eager timing.*its case id was timed"),
                        (_case("t/a", "t", {"n": 1}, "float32"), "no eager timing"),
                        (_case("t/b", "t", {"n": 1}), "re-time both"),
                        (_case("u/x", "u", {"n": 1}, "float32"), "none its case id")):
        with pytest.raises(SystemExit, match=match):
            writer.timed_l1("m", case, timing)


def test_l2_measurements_must_match_calls_and_dims():
    def report(**changes):
        base = {"model": "m", "task": "t", "config": "t/a", "kind": "corpus", "calls": 16,
                "dims": {"n": 1}, "dtype": "bfloat16", "wall_ms": 2.0, "share_pct": 1.5}
        return writer.report_rows({"rows": [{**base, **changes}]})

    assert writer.measured_l2("m", _case("t/a", "t", {"n": 1}, frequency=16), report())[
        "optimizable_share_pct"] == 1.5
    with pytest.raises(SystemExit, match="measured 16 calls per step, the snapshot has 32"):
        writer.measured_l2("m", _case("t/a", "t", {"n": 1}, frequency=32), report())
    with pytest.raises(SystemExit, match=re.escape("measured {'n': 1} bfloat16, the snapshot has")):
        writer.measured_l2("m", _case("t/a", "t", {"n": 2}, frequency=16), report())
    # The documented call-count changes, and only those, scale the share by calls.
    rescaled = writer.measured_l2("olmoe_1b_7b", _case(
        "olmoe_residual_rmsnorm/residual_rmsnorm", "olmoe_residual_rmsnorm", {"n": 1}, frequency=32),
        report(model="olmoe_1b_7b", task="olmoe_residual_rmsnorm",
               config="olmoe_residual_rmsnorm/residual_rmsnorm"))
    assert rescaled["optimizable_share_pct"] == 3.0
    with pytest.raises(SystemExit, match="RESCALED"):
        writer.measured_l2("olmoe_1b_7b", _case(
            "olmoe_residual_rmsnorm/residual_rmsnorm", "olmoe_residual_rmsnorm", {"n": 1}, frequency=48),
            report(model="olmoe_1b_7b", task="olmoe_residual_rmsnorm",
                   config="olmoe_residual_rmsnorm/residual_rmsnorm"))
    # A synthetic block names its dims and dtype in its label.
    synthetic = report(model="dit_s_2", task="dit_adaln_residual", config="x[2,4,8] fp32",
                       kind="synthetic", calls=24, dims=None, dtype=None)
    dims = {"B": 2, "T": 4, "H": 8}
    final = writer.measured_l2("dit_s_2", _case(
        "dit_s_2_dit_adaln_residual/final", "dit_s_2_dit_adaln_residual", dims, "float32", 1), synthetic)
    assert final["calls"] == 1 and final["optimizable_share_pct"] == 1.5 / 24
    with pytest.raises(SystemExit, match="no measurement"):
        writer.measured_l2("dit_s_2", _case(
            "dit_s_2_dit_adaln_residual/final", "dit_s_2_dit_adaln_residual", dims, "bfloat16"), synthetic)


# ── the reader ───────────────────────────────────────────────────────────────


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
