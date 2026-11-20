"""Gate 2 -- framework-integration verification (one forward/backward).

Two scopes. ``operator``: every recorded call of a case through the artifact's
deployment entry, against the float32 truth, within twice the error of native
and ``torch.compile`` or a floor of one rounding; nothing calibrated
(:mod:`.operator`)::

    from evograd.verification.gate2.operator import verify_operator

    report = verify_operator(scope, scope.candidate(artifact), samples=samples)

``model``: thresholds calibrated from trusted controls on fixed samples and
judged on fresh executions of the same samples (matched inputs)::

    cal = gate2.calibrate(model, patch_set=sites, samples=fixed)
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
from evograd.verification.gate2.operator import verify_operator
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
    "BINDING_CODES", "CANDIDATE", "CONTROL", "DEFAULT_FLOORS", "IDENTITY_COMPONENTS",
    "MATCHED", "NATIVE", "PROTOCOL", "SAFETY_FACTOR_GAMMA", "SAMPLE_PROTOCOLS", "SCOPES",
    "SCHEMA", "CalibrationError", "CalibrationResult", "ControlMeasurement", "Deployment",
    "Execution", "Gate2Policy", "Invocation", "Sample", "calibrate", "candidate", "combine_scopes",
    "compare_executions", "compile_control", "derive_thresholds", "execute", "freeze_policy",
    "integration_contract", "judge", "policy_findings", "verify_integration", "verify_operator",
]
