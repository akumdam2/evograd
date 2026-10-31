"""Cache invalidation and independent operator-control validation regressions."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evograd.verification import gate2
from evograd.verification.common.execution import CANDIDATE, CONTROL, Deployment
from evograd.verification.gate2 import calibration as C, runner
from evograd.verification.gate2.adapters.operator import OperatorScope
from evograd.verification.integrations import gates
from tests.verification import _fixtures as F


@pytest.fixture
def scope():
    return OperatorScope(F.SCALE_SHIFT, F.SCALE_SHIFT.correctness[0], device="cpu")


def options(scope):
    return dict(samples=scope.samples([1, 2]), compile_backend="aot_eager",
                validation_samples=scope.samples([11, 12]))


def deployment(scope):
    return Deployment("reference", CANDIDATE, {"operator": scope.native_site("operator")})


def test_operator_control_passes_holdout_and_a_bad_candidate_still_fails(scope):
    result = gate2.calibrate(scope, patch_set=("operator",), **options(scope))
    assert result.status == "calibrated", result.to_dict()
    assert result.evidence["holdout_validation"]["status"] == "passed"
    bad = gate2.verify_integration(scope, scope.candidate(F.scale_shift_pair(dw_scale=1.05)),
                                  policy=result.policy, samples=scope.samples([11, 12]))
    assert bad.status == "failed"


def test_compile_holdout_failure_withholds_policy_instead_of_fitting_the_holdout(scope):
    original = runner.execute

    def faulty_control(adapter, deployed, sample, **kwargs):
        execution = original(adapter, deployed, sample, **kwargs)
        if deployed.role == CONTROL and sample.seed == 11:
            execution.results["out:y"] = execution.results["out:y"] + 1
            execution.results_digest = runner.results_digest(execution.results)
        return execution

    with patch.object(runner, "execute", faulty_control):
        result = gate2.calibrate(scope, patch_set=("operator",), **options(scope))
    assert result.status == "incomplete"
    assert result.policy is None
    assert result.evidence["holdout_validation"]["status"] == "failed"
    assert result.evidence["unqualified_policy"]["floors"] == C.DEFAULT_FLOORS
    assert {f.code for f in result.findings} == {"control_holdout_disqualified"}
    report = gate2.verify_integration(scope, deployment(scope), policy=result.policy,
                                     samples=scope.samples([11]))
    assert report.status == "incomplete"


def test_holdout_cannot_overlap_calibration(scope):
    settings = options(scope)
    settings["validation_samples"] = settings["samples"]
    result = gate2.calibrate(scope, patch_set=("operator",), **settings)
    assert result.status == "incomplete"
    assert result.policy is None
    assert "sample_overlap" in str(result.evidence["holdout_validation"])


@pytest.fixture
def cached(scope, tmp_path):
    path = tmp_path / "calibration.json"
    result = gates._calibrate(scope, deployment(scope), scope.samples([11, 12]), path,
                              **options(scope))
    assert result.status == "calibrated", result.to_dict()
    return path


def test_unchanged_request_reuses_policy_and_keeps_holdout_evidence(scope, cached):
    with patch.object(gate2, "calibrate") as calculate:
        result = gates._calibrate(scope, deployment(scope), scope.samples([11, 12]), cached,
                                  **options(scope))
    calculate.assert_not_called()
    assert result.status == "calibrated"
    assert result.evidence["holdout_validation"]["status"] == "passed"


@pytest.mark.parametrize("change", [
    {"compile_backend": "inductor"}, {"compile_fullgraph": False},
    {"compile_options": {"max_autotune": True}}, {"native_repeats": 3},
    {"gamma": 3.0}, {"boundaries": True},
    {"floors": {"output": {"E2": .1, "Emax": .1},
                "gradient": {"E2": .1, "Emax": .1}}},
    {"samples": [3, 4]}, {"validation_samples": [21, 22]},
])
def test_changed_calibration_request_is_not_reused(scope, cached, change):
    changed = copy.deepcopy(change)
    for name in ("samples", "validation_samples"):
        if name in changed:
            changed[name] = scope.samples(changed[name])
    fresh = C.CalibrationResult(policy=None)
    with patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        result = gates._calibrate(scope, deployment(scope), scope.samples([11, 12]), cached,
                                  **{**options(scope), **changed})
    calculate.assert_called_once()
    assert result is fresh


def test_legacy_cache_is_not_reused(scope, cached):
    data = json.loads(cached.read_text())
    data.pop("calibration_request")
    cached.write_text(json.dumps(data))
    fresh = C.CalibrationResult(policy=None)
    with patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        assert gates._calibrate(scope, deployment(scope), scope.samples([11, 12]), cached,
                                **options(scope)) is fresh
    calculate.assert_called_once()


def test_changed_calibration_implementation_is_not_reused(scope, cached):
    fresh = C.CalibrationResult(policy=None)
    with patch.object(C, "CALIBRATION_PROCEDURE", "new procedure"), \
            patch.object(gate2, "calibrate", return_value=fresh) as calculate:
        assert gates._calibrate(scope, deployment(scope), scope.samples([11, 12]), cached,
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
