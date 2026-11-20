"""Operator Gate 2's threshold from a trusted control that drifts on one sample."""
from unittest.mock import patch

from evograd.verification import gate2
from evograd.verification.common.execution import CONTROL
from evograd.verification.gate2 import operator
from evograd.verification.gate2.adapters.operator import OperatorScope
from tests.verification import _fixtures as F


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
