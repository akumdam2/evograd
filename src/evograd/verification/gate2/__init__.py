"""Gate 2 -- framework-integration verification (one forward/backward, no calibration).

Two scopes, one form: ``err(candidate, reference) <= max(floor, factor * err(trusted))``.

``operator`` (:mod:`.operator`): every recorded call of a case through the artifact's
deployment entry, against the float32 truth; trusted native and ``torch.compile``,
factor 2, E2 and the worst element::

    from evograd.verification.gate2.operator import verify_operator

    report = verify_operator(scope, scope.candidate(artifact), samples=samples)

``model`` (:mod:`.model`): the whole model with the candidate installed, against
native eager on the same batches; trusted ``torch.compile`` at the replaced sites,
factor gamma, E2::

    from evograd.verification.gate2.model import verify_model

    report = verify_model(model_scope, deployment, samples=batches, gamma=4.0)
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
from evograd.verification.gate2.controls import compile_control
from evograd.verification.gate2.model import verify_model
from evograd.verification.gate2.operator import verify_operator
from evograd.verification.gate2.runner import (
    SCOPES,
    Execution,
    execute,
    integration_contract,
    structure_checks,
)

__all__ = [
    "CANDIDATE", "CONTROL", "NATIVE", "SCOPES", "Deployment", "Execution", "Invocation", "Sample",
    "candidate", "compile_control", "execute", "integration_contract", "structure_checks",
    "verify_model", "verify_operator",
]
