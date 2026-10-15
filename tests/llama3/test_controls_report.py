"""Control crashes must not become evidence that the model gate detected a fault."""

import json
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from evograd.evaluation.tier3.gate.faults import smallest_rejected

HAVE_TRITON = importlib.util.find_spec("triton") is not None
if HAVE_TRITON:
    from evograd.evaluation.tier3.workloads.llama3_2_1b import controls as llama_controls
    from evograd.evaluation.tier3.workloads.qwen3_0_6b import controls as qwen_controls
    CONTROL_MODULES = (llama_controls, qwen_controls)


def test_sensitivity_ignores_errored_records():
    fault = {"name": "grad_scale", "magnitude": 0.001}
    records = [
        {"fault": fault, "rejected": True, "errored": True, "error": "CUDA OOM"},
        {"fault": fault, "rejected": False},
    ]
    summary = smallest_rejected(records)["grad_scale"]
    assert summary["smallest_always_rejected"] is None
    assert summary["per_magnitude"]["0.001"] == {"rejected": 0, "of": 1}


@unittest.skipUnless(HAVE_TRITON, "Llama control imports require Triton")
def test_child_failure_is_incomplete_and_positive_controls_run_first():
    fault = SimpleNamespace(name="grad_scale", to_dict=lambda: {
        "name": "grad_scale", "magnitude": 0.001,
    })
    for controls in CONTROL_MODULES:
        calls = []

        def isolated(argv):
            calls.append(argv)
            if "--positive-only" in argv:
                return [{"provider": "structural_identity", "seed": 11, "admitted": True},
                        {"provider": "bound_pair_identity", "seed": 11, "admitted": True}]
            return {"error": "child exited rc=-9 without a result"}

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            with mock.patch.object(controls, "_catalogue", return_value=[fault]), \
                    mock.patch.object(controls, "state_catalogue", return_value=[]), \
                    mock.patch.object(controls, "diagnostic_catalogue", return_value=[]), \
                    mock.patch.object(controls, "_isolated", side_effect=isolated), \
                    mock.patch.object(controls, "load_policy", return_value=SimpleNamespace(
                        environment_hash="env", workload_id="model")):
                assert controls.main(["--seeds", "11", "--report", str(path)]) == 0
            report = json.loads(path.read_text())

        assert "--positive-only" in calls[0]
        assert "--fault-index" in calls[1]
        assert report["schema_version"].endswith("/3")
        assert report["coverage"]["complete"] is False
        assert report["coverage"]["judged"] == 0
        assert report["coverage"]["errored"] == 1
        assert report["coverage"]["resource_failures"] == 1
        assert report["all_rejected"] is False
        assert report["sensitivity"] is None
        assert report["positive_controls_ok"] is True
        assert report["errored_controls"][0]["rejected"] is False


@unittest.skipUnless(HAVE_TRITON, "Llama control imports require Triton")
def test_gate_exception_is_an_error_not_a_rejection():
    fault = SimpleNamespace(
        to_dict=lambda: {"name": "grad_scale", "magnitude": 0.001},
        apply=mock.Mock(side_effect=RuntimeError("CUDA out of memory")),
    )
    for controls, model in ((llama_controls, llama_controls.Llama3Workload),
                            (qwen_controls, qwen_controls.Qwen3Workload)):
        with mock.patch.object(model, "from_config", return_value=object()):
            record = controls.run_one(fault, seed=11, policy=None, workload_config={})
        assert record["errored"] is True
        assert record["rejected"] is False
