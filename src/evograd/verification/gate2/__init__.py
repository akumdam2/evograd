"""Gate 2 -- framework-integration verification (one forward/backward, native reference).

Two scopes: ``operator`` (generated inputs, disjoint calibration/validation
samples) and ``model`` (fixed real samples, matched inputs: calibrated and
judged on the same content, per-sample thresholds).

    from evograd.verification import gate2
    from evograd.verification.gate2.adapters.operator import OperatorScope

    scope = OperatorScope(op, workload, device="cpu")
    calibration = gate2.calibrate(scope, patch_set=("operator",), samples=scope.samples([1, 2, 3]))
    report = gate2.verify_integration(scope, scope.candidate(artifact),
                                      policy=calibration.policy, samples=scope.samples([11, 12]))

    # model scope: the same samples on both sides
    cal = gate2.calibrate(model, patch_set=sites, samples=fixed, boundaries=True)
    report = gate2.verify_integration(model, deployment, policy=cal.policy, samples=fixed)
"""

from evograd.verification.common.execution import (
    CANDIDATE,
    CONTROL,
    NATIVE,
    Deployment,
    Invocation,
    Sample,
    candidate,
)
from evograd.verification.gate2.calibration import (
    BINDING_CODES,
    DEFAULT_FLOORS,
    DISJOINT,
    MATCHED,
    SAMPLE_PROTOCOLS,
    SAFETY_FACTOR_GAMMA,
    SCHEMA,
    CalibrationError,
    CalibrationResult,
    ControlMeasurement,
    Gate2Policy,
    calibrate,
    compile_control,
    derive_thresholds,
    freeze_policy,
    policy_findings,
)
from evograd.verification.gate2.runner import (
    IDENTITY_COMPONENTS,
    PROTOCOL,
    SCOPES,
    Execution,
    combine_scopes,
    compare_executions,
    execute,
    integration_contract,
    judge,
    verify_integration,
)

__all__ = [
    "BINDING_CODES", "CANDIDATE", "CONTROL", "DEFAULT_FLOORS", "DISJOINT", "IDENTITY_COMPONENTS",
    "MATCHED", "NATIVE", "PROTOCOL", "SAFETY_FACTOR_GAMMA", "SAMPLE_PROTOCOLS", "SCOPES",
    "SCHEMA", "CalibrationError", "CalibrationResult", "ControlMeasurement", "Deployment",
    "Execution", "Gate2Policy", "Invocation", "Sample", "calibrate", "candidate", "combine_scopes",
    "compare_executions", "compile_control", "derive_thresholds", "execute", "freeze_policy",
    "integration_contract", "judge", "policy_findings", "verify_integration",
]
