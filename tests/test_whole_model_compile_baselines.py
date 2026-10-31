"""Real workload discovery, compiled builds and independent baseline evidence."""
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from evograd.evaluation import protocol
from evograd.evaluation.tier3 import compile_baseline as B
from evograd.evaluation.tier3.patch import KernelSet, SiteRegistry, Site, patch as patch_kernel


@pytest.mark.parametrize("module,cls,model", [
    ("qwen3_0_6b", "Qwen3Workload", "qwen3_0_6b"),
    ("qwen3_5_4b", "Qwen35Workload", "qwen3_5_4b"),
    ("llama3_2_1b", "LlamaWorkload", "llama_3_2_1b"),
    ("olmoe_1b_7b", "OlmoeWorkload", "olmoe_1b_7b"),
    ("common.vision", "VisionWorkload", "qwen2_vl_2b"),
    ("common.vision", "VisionWorkload", "convnextv2_tiny"),
    ("common.generated", "GeneratedWorkload", "dit_s_2"),
    ("common.generated", "GeneratedWorkload", "torchrec_dlrm"),
])
def test_registered_workload_compile_discovery(module, cls, model):
    module = importlib.import_module(f"evograd.evaluation.tier3.workloads.{module}.workload")
    # No weights, datasets or CUDA allocations are needed to discover support.
    classes = [c for c in vars(module).values() if isinstance(c, type)
               and c.__module__ == module.__name__ and hasattr(c, "build_patched")]
    workload = classes[0].__new__(classes[0])
    workload.site_registry = SiteRegistry(model, ())
    kernels = protocol.whole_model_compile(workload)
    assert not kernels.patched
    assert B.whole_model_compile_requested(kernels)


@pytest.mark.parametrize("family,model", [
    ("vision", "qwen2_vl_2b"), ("vision", "convnextv2_tiny"),
    ("generated", "dit_s_2"), ("generated", "torchrec_dlrm"),
])
def test_missing_workloads_build_compiled_model(family, model):
    module = importlib.import_module(f"evograd.evaluation.tier3.workloads.common.{family}.workload")
    cls = module.VisionWorkload if family == "vision" else module.GeneratedWorkload
    workload = cls.__new__(cls)
    workload.spec = SimpleNamespace(model=model)
    workload.init_checkpoint = None
    kernels = B.whole_model_compile_kernels(SiteRegistry(model, ()))
    original = torch.nn.Linear(2, 2)
    with patch.object(module, "build_model", return_value=original), patch.object(torch, "compile", return_value=original) as compile:
        result, provenance = workload.build_patched(kernels)
    compile.assert_called_once_with(original, **B.COMPILE_SETTINGS)
    assert result is original
    assert provenance.method == B.ORIGIN


def test_compiled_baseline_rejects_site_patches():
    registry = SiteRegistry("toy", (Site("x", "x", lambda x: x),))
    kernels = patch_kernel(B.whole_model_compile_kernels(registry), "x", lambda x: x)
    with pytest.raises(ValueError, match="unpatched"):
        B.compile_model(torch.nn.Linear(1, 1), kernels)


def test_no_graphs_is_reported_as_failed_compile():
    row = {"ok": True, "dynamo": {"unique_graphs": 0}, "step_ms": 1}
    B.require_compiled_graphs(row)
    assert not row["ok"]
    assert row["failed_at"] == "compile_coverage"


def execution(value, prefix=""):
    return SimpleNamespace(error=None, mutated=[], input_digest="same",
                           initial_state={prefix + "weight": "same"},
                           results={"grad:" + prefix + "weight": value})


def test_correctness_normalizes_wrapper_names_and_rejects_bad_gradients():
    reference = execution(torch.ones(2))
    assert not B.compare(reference, execution(torch.ones(2), "_orig_mod."))
    assert B.compare(reference, execution(torch.zeros(2), "_orig_mod."))
    assert B.compare(reference, execution(torch.full((2,), float("nan"))))


def test_compile_failure_preserves_correctness_evidence():
    run = SimpleNamespace(model=None, workload=None, device="cpu")
    with patch.object(B, "check_correctness", return_value={"status": "failed"}), \
         patch.object(protocol, "provider_kernels", side_effect=RuntimeError("compile unavailable")):
        row = protocol._measure_tier3_provider(run, "torch_compile", "torch_compile", {})
    assert not row["ok"]
    assert row["correctness"]["status"] == "failed"
    assert "compile unavailable" in row["error"]


def test_correctness_executes_real_compiled_forward_and_backward_on_cpu():
    from evograd.verification.gate2.adapters.model import ObjectiveModelScope
    from tests.verification import _toy_integration as T

    class Workload:
        name = "toy"
        spec = SimpleNamespace(device="cpu", dtype="float32")
        site_registry = SiteRegistry("toy", ())
        primary_output_name = "prediction"
        objective_definition = "squared prediction"

        def build_patched(self, kernels):
            model = torch.nn.Linear(2, 1)
            if B.whole_model_compile_requested(kernels):
                return B.compile_model(model, kernels)
            return model, None

        def expected_site_calls(self):
            return {}

        def batch_for(self, *, seed):
            return {"x": torch.ones(1, 2) * (seed / 100)}

        def primary_output(self, model, batch):
            prediction = model(batch["x"])
            return prediction.square().mean(), prediction

    workload = Workload()
    with T.toy_environment(), \
         patch("evograd.verification.integrations.gates.model_scope",
               side_effect=lambda model, workload: ObjectiveModelScope(workload)), \
         patch.object(protocol, "whole_model_compile", side_effect=lambda workload: B.whole_model_compile_kernels(workload.site_registry)), \
         patch.dict(B.COMPILE_SETTINGS, backend="aot_eager"):
        result = B.check_correctness(None, workload)
    assert result["status"] == "passed", result
    assert [row["seed"] for row in result["samples"]] == [101, 202, 303, 404]
    assert all(row["results_checked"] >= 4 for row in result["samples"])


def test_invalid_compile_correctness_cannot_supply_comparison_ratio():
    assert not protocol.compile_baseline_valid({"ok": True, "correctness": {"status": "failed"}})
    assert not protocol.compile_baseline_valid({"ok": True})
    assert protocol.compile_baseline_valid({"ok": True, "correctness": {"status": "passed"}})


def test_baseline_code_change_invalidates_binding():
    from tests.verification import _toy_integration as T
    workload = T.ToyWorkload()
    with patch("evograd.verification.integrations.base.sha_file", return_value="old"):
        before = protocol.baseline_implementation(workload)
    with patch("evograd.verification.integrations.base.sha_file", return_value="new"):
        after = protocol.baseline_implementation(workload)
    assert before != after


def test_legacy_evaluator_runs_and_resumes_baselines_without_candidates(tmp_path):
    from tests.verification import _toy_integration as T
    from evograd.verification.integrations.gates import Run
    from evograd.evaluation.tier3.runner import assemble_report

    model = T.model()
    workload = T.ToyWorkload()
    corpus = SimpleNamespace(binding=lambda: {"corpus": "toy"})
    run = Run(model, tmp_path, tmp_path / "state", corpus, tmp_path / "manifest.json",
              {}, {}, {}, {}, device="cpu", _workload=workload)
    raw = {name: {"ok": True, "step_ms": ms, "per_block_ms": [ms] * 3,
                   "peak_memory_bytes": 100, "correctness": {"status": "passed"}}
           for name, ms in (("eager", 2), ("torch_compile", 1))}
    report = assemble_report(workload, raw, list(raw), verify=False, isolation="test",
                             **protocol.TIER3_OPTIONS)
    with T.toy_environment(), \
         patch("evograd.verification.integrations.base.observed_cases", return_value=[]), \
         patch.object(protocol, "open_run", return_value=run), \
         patch.object(protocol, "verify_cases", return_value={"corpus": {}, "candidates": {}}), \
         patch.object(protocol, "run_tier3", return_value=report) as measure:
        for _ in range(2):
            result = protocol.evaluate(model, {}, "manifest.json", tmp_path, device="cpu",
                                       rules=False, check_imports=False, log=lambda *a: None)
            assert result["tier3"]["baselines"]["torch_compile"]["ok"]
            assert result["tier3"]["combination"]["cases"] == []
    assert measure.call_count == 1
    assert measure.call_args.args[1] == {"eager": "eager", "torch_compile": "torch_compile"}
