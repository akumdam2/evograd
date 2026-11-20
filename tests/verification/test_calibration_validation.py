"""Model Gate 2's calibration cache invalidation, and operator Gate 2's thresholds from a
trusted control that drifts on one sample."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evograd.verification import gate2
from evograd.verification.common.execution import CANDIDATE, CONTROL, Deployment, Sample
from evograd.verification.gate2 import calibration as C, operator
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.integrations import gates
from tests.verification import _fixtures as F


@pytest.fixture
def scope():
    return F.toy_module_scope()


def options(scope):
    return dict(samples=[Sample(f"cache/seed{k}", k, {"recipe": "fixture"}) for k in (1, 2)],
                compile_backend="aot_eager")


def deployment(scope):
    return Deployment("reference", CANDIDATE, {"norm": scope.native_site("norm")})


def test_control_drift_on_one_sample_sets_the_threshold_and_a_bad_candidate_still_fails():
    # The vision_mlp case of 2026-11-03: compile drifts more on one batch than on the
    # others. Pooled over the judged samples, its own drift sets the threshold.
    scope = OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.benchmark[0], device="cpu")
    samples = scope.samples([1, 2])
    original = operator.execute

    def drifting_control(adapter, deployed, sample, **kwargs):
        execution = original(adapter, deployed, sample, **kwargs)
        if deployed.role == CONTROL and sample.seed == 2:
            execution.results["out:y"] = execution.results["out:y"] * (1 + 1e-3)
        return execution

    with patch.object(operator, "execute", drifting_control):
        control, _graphs = gate2.compile_control(scope, ("operator",), backend="aot_eager")
        own = operator.verify_operator(scope, control, samples=samples, compile_backend="aot_eager")
        bad = operator.verify_operator(scope, scope.candidate(F.scale_shift_pair(dw_scale=1.05)),
                                       samples=samples, compile_backend="aot_eager")
    entry = own.subject["thresholds"]["out:y"]["E2"]
    assert (entry["from"], round(entry["threshold"], 6)) == ("compile", 2e-3)
    assert own.status == "passed"
    assert bad.status == "failed"


@pytest.fixture
def cached(scope, tmp_path):
    path = tmp_path / "calibration.json"
    result = gates._calibrate(scope, deployment(scope), options(scope)["samples"], path,
                              **options(scope))
    assert result.status == "calibrated", result.to_dict()
    return path


def test_unchanged_request_reuses_policy(scope, cached):
    with patch.object(gate2, "calibrate") as calculate:
        result = gates._calibrate(scope, deployment(scope), options(scope)["samples"], cached,
                                  **options(scope))
    calculate.assert_not_called()
    assert result.status == "calibrated"
    assert result.evidence["reused_policy"] == str(cached)


@pytest.mark.parametrize("change", [
    {"compile_backend": "inductor"}, {"compile_fullgraph": False},
    {"compile_options": {"max_autotune": True}}, {"native_repeats": 3},
    {"gamma": 3.0}, {"boundaries": True},
    {"floors": {"output": {"E2": .1, "Emax": .1},
                "gradient": {"E2": .1, "Emax": .1}}},
    {"samples": [3, 4]},
])
def test_changed_calibration_request_is_not_reused(scope, cached, change):
    changed = copy.deepcopy(change)
    if "samples" in changed:
        changed["samples"] = [Sample(f"cache/seed{k}", k, {"recipe": "fixture"})
                              for k in changed["samples"]]
    fresh = C.CalibrationResult(policy=None)
    with patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        result = gates._calibrate(scope, deployment(scope), options(scope)["samples"], cached,
                                  **{**options(scope), **changed})
    calculate.assert_called_once()
    assert result is fresh


def test_legacy_cache_is_not_reused(scope, cached):
    data = json.loads(cached.read_text())
    data.pop("calibration_request")
    cached.write_text(json.dumps(data))
    fresh = C.CalibrationResult(policy=None)
    with patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        assert gates._calibrate(scope, deployment(scope), options(scope)["samples"], cached,
                                **options(scope)) is fresh
    calculate.assert_called_once()


def test_changed_calibration_implementation_is_not_reused(scope, cached):
    fresh = C.CalibrationResult(policy=None)
    with patch.object(C, "CALIBRATION_PROCEDURE", "new procedure"), \
            patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        assert gates._calibrate(scope, deployment(scope), options(scope)["samples"], cached,
                                **options(scope)) is fresh
    calculate.assert_called_once()


def test_verdict_cache_is_invalidated_when_comparator_changes(tmp_path):
    report = SimpleNamespace(status="passed", to_dict=lambda: {"status": "passed"})
    judge = Mock(return_value={"check": report})
    path = tmp_path / "verdict.json"
    run = SimpleNamespace(deadline=None)
    with patch("evograd.verification.integrations.base.sha_file", return_value="old"):
        gates._judged_stage(run, path, {"candidate": "same"}, judge)
        gates._judged_stage(run, path, {"candidate": "same"}, judge)
    assert judge.call_count == 1
    with patch("evograd.verification.integrations.base.sha_file", return_value="new"):
        gates._judged_stage(run, path, {"candidate": "same"}, judge)
    assert judge.call_count == 2
