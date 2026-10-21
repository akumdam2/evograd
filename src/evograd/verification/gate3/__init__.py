"""Gate 3 -- continuous training and calibrated whole-trajectory acceptance.

    from evograd.verification import gate3
    plan = gate3.TrainingPlan(steps=200, window=50, checkpoints=(0, 100, 200))
    native = gate3.measure_training(adapter, gate3.Deployment.native(), plan)
    cand = gate3.measure_training(adapter, candidate_deployment, plan)
    gate3.compare_curves(cand, native).to_dict()   # distances; status "incomplete"

``measure_training`` and ``compare_curves`` remain measurement APIs. The current
global-loss policy uses ``calibrate_global_loss`` on native/native/compile runs,
then ``verify_training`` on fresh executions of the same training sequence.

Historical stepwise rule (fixed workload, one limit per step): ``calibrate_step_loss`` on
native/native/compile runs of one exact batch sequence, then
``verify_training`` (dispatching by the policy schema) or ``verify_step_loss``.

Global-loss rule (current canonical Gate 3 for a fixed workload): whole-horizon
RMS and maximum of the same-step loss difference, ``calibrate_global_loss`` then
``verify_training`` (dispatching by schema) or ``verify_global_loss``.

``calibrate_window_loss`` retains the historical window protocol. Its policies
are explicitly versioned and are never reinterpreted as global-loss policies.
"""

from evograd.verification.common.execution import Deployment, candidate
from evograd.verification.gate3.adapters import (
    LanguageModelTraining,
    OptimizerSpec,
    SeedBatches,
    adamw,
    next_token_nll,
    WorkloadObjectiveTraining,
    site_model_training,
    workload_objective_training,
    workload_training,
)
from evograd.verification.gate3.metrics import (
    ACCEPTANCE_NOT_CONFIGURED,
    CurveComparison,
    TrainingPlan,
    compare_curves,
)
from evograd.verification.gate3.runner import (
    PROTOCOL,
    TrainingMeasurement,
    measure_training,
    optimizer_summary,
)
from evograd.verification.gate3.policy import (
    SAFETY_FACTOR_GAMMA,
    TrainingControl,
    TrainingVerdict,
    WindowCalibration,
    WindowLossPolicy,
    calibrate_window_loss,
    verify_training,
)
from evograd.verification.gate3.stepwise import (
    StepCalibration,
    StepLossPolicy,
    StepVerdict,
    calibrate_step_loss,
    verify_step_loss,
)
from evograd.verification.gate3.global_loss import (
    GlobalCalibration,
    GlobalLossPolicy,
    GlobalVerdict,
    calibrate_global_loss,
    loss_metrics,
    verify_global_loss,
)

__all__ = [
    "ACCEPTANCE_NOT_CONFIGURED", "CurveComparison", "Deployment", "LanguageModelTraining",
    "OptimizerSpec", "PROTOCOL", "SeedBatches", "TrainingMeasurement", "TrainingPlan", "adamw",
    "candidate", "compare_curves", "measure_training", "next_token_nll", "optimizer_summary",
    "site_model_training", "workload_training", "WorkloadObjectiveTraining",
    "workload_objective_training",
    "SAFETY_FACTOR_GAMMA", "TrainingControl", "TrainingVerdict", "WindowCalibration",
    "WindowLossPolicy", "calibrate_window_loss", "verify_training",
    "StepCalibration", "StepLossPolicy", "StepVerdict", "calibrate_step_loss", "verify_step_loss",
    "GlobalCalibration", "GlobalLossPolicy", "GlobalVerdict", "calibrate_global_loss",
    "loss_metrics", "verify_global_loss",
]
