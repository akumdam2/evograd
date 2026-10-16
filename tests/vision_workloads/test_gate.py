import json
import pytest
import torch
from .test_models import small_spec
from evograd.evaluation.tier3.workloads.common.vision.workload import (
    VisionWorkload,
    structural_identity_kernels,
)
from evograd.evaluation.tier3.workloads.common.vision.gate import (
    calibrate,
    evaluate,
    metrics,
    trajectory,
)
from evograd.evaluation.tier3.patch import patch


def test_metrics_keep_spikes_and_reject_nan():
    assert trajectory([1, 1, 2], [1, 1, 1])["max_abs"] == 1
    assert trajectory([2, 0], [1, 1])["rms"] == 1
    with pytest.raises(ValueError):
        metrics(torch.tensor([float("nan")]), torch.ones(1))


@pytest.mark.parametrize("model", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_calibration_and_holdout(tmp_path, model):
    w = VisionWorkload(small_spec(model))
    kernels = structural_identity_kernels(w.site_registry)
    assert not evaluate(w, kernels)["ok"]
    artifact = calibrate(w, list(w.site_registry), steps=2, backend="aot_eager")
    assert artifact["data_mode"] == "synthetic"
    assert set(artifact["calibration_indices"]).isdisjoint(artifact["evaluation_indices"])
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))
    w.calibration = str(path)
    result = evaluate(w, kernels)
    assert result["ok"], result
    site = w.site_registry.names[0]
    default = w.site_registry.require(site).default
    bad = patch(kernels, site, lambda *args: default(*args) * 0)
    assert not evaluate(w, bad)["ok"]
    corrupt = dict(artifact)
    corrupt["steps"] = 99
    path.write_text(json.dumps(corrupt))
    assert "hash mismatch" in evaluate(w, kernels)["reason"]


@pytest.mark.parametrize("model", ["convnextv2_tiny", "qwen2_vl_2b"])
def test_real_input_selects_pinned_checkpoint_and_data(monkeypatch, model):
    from evograd.evaluation.tier3.cli import _parser, build_workload, check_options
    from evograd.benchmark.topdown.common.vision import data

    seen = []
    def cached(spec, path=None, **kwargs):
        seen.append((spec.weights, spec.model, path))
        return "/tmp/pinned-vision-data.pt"
    monkeypatch.setattr(data, "ensure_pinned_data", cached)
    args = _parser().parse_args(["--model", model, "--real-input"])
    check_options(args)
    workload = build_workload(args)
    assert seen == [("pretrained", model, None)]
    assert workload.spec.weights == "pretrained"
    assert workload.data_path == "/tmp/pinned-vision-data.pt"
    assert build_workload(_parser().parse_args(["--model", model])).spec.weights == "random"
