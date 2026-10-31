"""Baseline-only runs preserve candidates and ignore stale group reuse flags."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evograd.evaluation import protocol as P

path = Path(__file__).resolve().parents[2] / "candidates/evaluate_baselines.py"
spec = importlib.util.spec_from_file_location("evaluate_baselines", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("name", ["qwen2_vl_2b", "dit_s_2", "convnextv2_tiny", "torchrec_dlrm"])
def test_only_baselines_run_and_original_results_stay_unchanged(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    original = root / "evaluation/gepa/evaluation.json"
    original.parent.mkdir(parents=True)
    config = {"model": name, "stream": True}
    original.write_text(json.dumps({"model": name, "workload": config,
                                    "settings": {"tier3": P.TIER3_OPTIONS},
                                    "cases": {"keep": "unchanged"}}))
    before = original.read_bytes()
    plan = {"model": name, "calibration_reuse": {"share_gate3_native": True},
            "jobs": [{"out": str(original.parent), "selection": {"corpus": "saved-corpus"}}]}
    (root / "all_agents.json").write_text(json.dumps(plan))
    run = SimpleNamespace(workload=object(), corpus=SimpleNamespace(binding=lambda: {"sha": "same"}))
    measured = {"providers": {"eager": {"ok": True},
                               "torch_compile": {"ok": True, "correctness": {"status": "passed"}}}}
    with patch("evograd.verification.integrations.load", return_value=SimpleNamespace(workload_config=config)), \
            patch("evograd.verification.integrations.base.configure"), \
            patch("evograd.verification.integrations.gates.open_run", return_value=run) as opened, \
            patch.object(P, "baseline_implementation", return_value={"revision": "test"}), \
            patch.object(P, "_environment", return_value={"device": "test"}), \
            patch.object(P, "run_tier3", return_value=measured) as measure:
        assert runner.measure(name, tmp_path, device="cpu") == 0
        assert opened.call_args.args[1] == {}
        assert measure.call_args.args[1] == {"eager": "eager", "torch_compile": "torch_compile"}
        assert measure.call_args.kwargs["options"] == P.TIER3_OPTIONS
        assert runner.measure(name, tmp_path, device="cpu") == 0
        assert measure.call_count == 1
        assert original.read_bytes() == before
        # Compiler changes invalidate only this separate baseline report.
        with patch.object(P, "baseline_implementation", return_value={"revision": "new"}):
            assert runner.measure(name, tmp_path, device="cpu") == 0
        assert measure.call_count == 2
    result = json.loads((root / "shared/tier3/baselines_only.json").read_text())
    assert result["baseline_only"] and result["baseline_complete"]


def test_model_mismatch_stops_before_measurement(tmp_path):
    root = tmp_path / "dit_s_2"
    root.mkdir()
    (root / "all_agents.json").write_text(json.dumps({"model": "torchrec_dlrm"}))
    with patch.object(P, "run_tier3") as measure, pytest.raises(ValueError, match="another model"):
        runner.measure("dit_s_2", tmp_path)
    measure.assert_not_called()
