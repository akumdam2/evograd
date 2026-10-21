"""Gate 3 (whole model): continuous training and its measurement. One core.

One run trains one deployment from the adapter's initial weights and optimizer
state over the adapter's batch sequence, and its parameters and optimizer state
evolve continuously -- nothing is copied from another run, nothing is reset
at a checkpoint. It records

* every step's loss sum and weight (tokens or samples) and the fixed-window
  weighted means of the plan;
* at each checkpoint, the validation loss of the *trained weights* measured by
  a trusted evaluator: a separately built native model (no replaced site) with
  the weights copied in, over the same fixed held-out batches for every run.
  The deployment's own forward never scores itself; a candidate site called
  during validation invalidates the measurement;
* optionally, at each checkpoint, digests of the parameters and a summary of
  the optimizer state, as evidence that state evolved continuously.

This function measures one run. ``gate3.verify_training`` subsequently compares
it to a native run under a frozen window-loss policy. A measurement alone
never constitutes a verification pass.
"""

from __future__ import annotations

import math
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from evograd.verification.common.contracts import environment_fingerprint, stable_hash
from evograd.verification.common.execution import NATIVE, Deployment, SiteHooks
from evograd.verification.common.reports import (
    Finding,
    failure,
    jsonable,
    missing,
    summarize,
)
from evograd.verification.common.tensors import isolated_rng, module_state_digest
from evograd.verification.gate3.metrics import (
    ACCEPTANCE_NOT_CONFIGURED,
    TrainingPlan,
    WindowAccumulator,
)

GATE = "gate3"
PROTOCOL = "evograd-verification-gate3-measurement/1"
#: What each step record's ``objective`` is (added for the stepwise rule; the
#: older ``loss_sum``/``weight``/``mean_loss`` fields are unchanged).
OBJECTIVE_DEFINITION = ("evograd-gate3-step-objective/1: float(loss_sum / weight), the scalar "
                        "tensor backward() differentiates, read before the optimizer update")


class TrainingAdapter(Protocol):
    """What a whole-model training measurement needs; see ``gate3.adapters``."""

    device: str

    def identity(self) -> Mapping[str, Any]: ...
    def data_identity(self) -> Mapping[str, Any]: ...
    def optimizer_config(self) -> Mapping[str, Any]: ...
    def build_model(self, deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module: ...
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer: ...
    def build_evaluator(self) -> torch.nn.Module: ...
    def train_batch(self, index: int) -> Any: ...
    def train_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, float]: ...
    def validation_batches(self) -> Sequence[Any]: ...
    def validation_loss(self, evaluator: torch.nn.Module, batch: Any) -> tuple[float, float]: ...


@dataclass
class TrainingMeasurement:
    deployment: dict[str, Any]
    identity: dict[str, Any]
    seed: int
    train_steps: list[dict[str, Any]] = field(default_factory=list)
    train_windows: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, dict[str, Any]] = field(default_factory=dict)
    state: dict[str, dict[str, Any]] = field(default_factory=dict)
    site_calls: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    completed: bool = False
    seconds: float = 0.0
    #: Distinguishes two executions even when their records are bitwise equal
    #: (deterministic repeats); provenance only, never part of the binding.
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Saved at its checkpoint's deadline; the same call continues it later.
    suspended: bool = False
    #: Provenance of a checkpointed run's processes; never part of the binding.
    segments: list[dict[str, Any]] = field(default_factory=list)

    def verdict(self) -> tuple[str, bool]:
        """Never ``passed``: acceptance is not configured in this phase."""
        return summarize([*self.findings, missing("acceptance_not_configured",
                                                  ACCEPTANCE_NOT_CONFIGURED)])

    def to_dict(self) -> dict[str, Any]:
        status, complete = self.verdict()
        return jsonable({
            "gate": GATE, "protocol": PROTOCOL, "objective": OBJECTIVE_DEFINITION,
            "run_id": self.run_id, "status": status, "complete": complete,
            "completed": self.completed,
            "acceptance": {"configured": False, "reason": ACCEPTANCE_NOT_CONFIGURED},
            "deployment": self.deployment, "identity": self.identity, "seed": self.seed,
            "train_steps": self.train_steps, "train_windows": self.train_windows,
            "validation": self.validation, "state": self.state, "site_calls": self.site_calls,
            "findings": [f.to_dict() for f in self.findings], "seconds": self.seconds,
            # Only a checkpointed run carries these; every other record is unchanged.
            **({"suspended": self.suspended, "segments": self.segments}
               if self.suspended or self.segments else {}),
        })


def _model_of(model: torch.nn.Module) -> torch.nn.Module:
    """The real module of a ``torch.compile``'d model."""
    return getattr(model, "_orig_mod", model)


def optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """How many parameters carry state, and the step counts they hold."""
    steps = set()
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps.add(int(step.item() if torch.is_tensor(step) else step))
    return {"parameters_with_state": len(optimizer.state), "step_counts": sorted(steps)}


#: ``on_step(record, model, optimizer, site_calls)`` after each optimizer update.
StepCallback = Callable[[dict[str, Any], torch.nn.Module, torch.optim.Optimizer,
                         Mapping[str, int]], None]


def measure_training(adapter: TrainingAdapter, deployment: Deployment, plan: TrainingPlan, *,
                     seed: int = 0, record_state: bool = False, acceptance: Any = None,
                     progress: Callable[[str], None] | None = None,
                     on_step: StepCallback | None = None,
                     protocol_context: Mapping[str, Any] | None = None,
                     checkpoint: Any = None) -> TrainingMeasurement:
    """Train ``deployment`` for ``plan.steps`` steps and measure both curves.

    ``seed`` fixes every random stream inside the run (dropout and the like);
    the caller's RNG is restored afterwards. ``record_state`` adds parameter
    digests and an optimizer summary at each checkpoint. A plan with
    ``validation=False`` builds no evaluator and reads no held-out data.

    ``on_step`` sees a copy of each step's record, the live model and
    optimizer and the cumulative site-call counts, after the update (e.g. to
    write the record out as the run goes). It must not change training state;
    an exception from it stops the run like any runtime error.

    ``protocol_context`` binds a later acceptance policy to the experiment:
    reference implementation revision, data recipe identity, patch set and
    numerical settings. It must be common to native/control/candidate, and
    exclude sample seeds, data order and candidate-specific implementation IDs.
    Missing context is allowed for measurement but cannot yield a policy/pass.

    ``checkpoint`` (:class:`~evograd.verification.gate3.checkpoint.RunCheckpoint`)
    makes the run resumable: its state is saved every ``checkpoint.every``
    steps, and when ``checkpoint.deadline`` passes the run is saved and
    returned ``suspended`` (not completed). Called again with the same
    checkpoint, it continues from the saved step; a checkpoint of another run
    (identity or deployment differ) is moved aside and the run starts over.
    The result is the run that never stopped: same records, identity, run id.
    """
    if acceptance is not None:
        raise NotImplementedError(ACCEPTANCE_NOT_CONFIGURED)
    say = progress or (lambda _message: None)
    native = deployment.role == NATIVE

    def problem(code: str, message: str, **details: Any) -> Finding:
        return (missing if native else failure)(code, message, subject=deployment.name, **details)

    run = TrainingMeasurement(deployment=deployment.describe(), identity={}, seed=int(seed))
    started = time.perf_counter()
    hooks = SiteHooks()
    stage = "build"
    try:
        with isolated_rng(seed, adapter.device):
            model = adapter.build_model(deployment, hooks)
            evaluator = None
            if plan.validation:
                stage = "evaluator"
                evaluator = adapter.build_evaluator()
                evaluator.eval()
            initial = module_state_digest(_model_of(model))
            run.identity = jsonable({
                "plan": plan.to_dict(), "seed": int(seed), "adapter": dict(adapter.identity()),
                "data": dict(adapter.data_identity()), "optimizer": dict(adapter.optimizer_config()),
                "environment": environment_fingerprint(adapter.device),
                "protocol_context": dict(protocol_context or {}),
                "initial_state": stable_hash(initial),
                "evaluator": None if evaluator is None else {
                    "kind": "native", "initial_state": stable_hash(module_state_digest(evaluator))},
            })
            if evaluator is not None:
                shared = ({id(p) for p in model.parameters()}
                          & {id(p) for p in evaluator.parameters()})
                if shared:
                    run.findings.append(missing("evaluator_shares_parameters",
                                                "the evaluator aliases the trained parameters"))
                    return run
            stage = "optimizer"
            optimizer = adapter.build_optimizer(model)
            batches = list(adapter.validation_batches()) if plan.validation else []
            if plan.validation and not batches:
                run.findings.append(missing("no_validation_data", "no held-out batch"))
                return run

            def validate(step: int) -> None:
                before = hooks.total
                evaluator.load_state_dict(_model_of(model).state_dict(), strict=True)
                loss_sum = weight = 0.0
                with torch.no_grad():
                    for batch in batches:
                        part, count = adapter.validation_loss(evaluator, batch)
                        loss_sum += float(part)
                        weight += float(count)
                record = {"loss_sum": loss_sum, "weight": weight, "batches": len(batches),
                          "mean_loss": loss_sum / weight if weight > 0 else None}
                run.validation[str(step)] = record
                if hooks.total != before:
                    run.findings.append(missing("evaluator_used_candidate",
                                                "replaced sites ran during validation; the "
                                                "evaluator is not native", calls=hooks.total - before))
                if record["mean_loss"] is None or not math.isfinite(record["mean_loss"]):
                    run.findings.append(problem("non_finite_validation_loss",
                                                f"validation loss at step {step} is not finite"))
                if record_state:
                    run.state[str(step)] = {"parameters": stable_hash(
                        module_state_digest(_model_of(model))), "optimizer": optimizer_summary(optimizer)}
                say(f"step {step} validation {record['mean_loss']}")

            windows = WindowAccumulator(plan)
            run.train_windows = windows.windows      # filled as the run goes
            first = 1
            if checkpoint is not None:
                stage = "resume"
                first = _resume(checkpoint, run, model, optimizer, windows, hooks, adapter.device)
            stage = "validation"
            if 0 in plan.checkpoints and first == 1:
                validate(0)
            for step in range(first, plan.steps + 1):
                stage = f"step {step}"
                batch = adapter.train_batch(step - 1)
                calls_before = dict(hooks.counts)
                optimizer.zero_grad(set_to_none=True)
                loss_sum, weight = adapter.train_loss(model, batch)
                value, weight = float(loss_sum.detach()), float(weight)
                # The objective differentiated below, and its scalar as recorded.
                objective = loss_sum / weight
                objective_value = float(objective.detach())
                if not (math.isfinite(value) and math.isfinite(objective_value)) or not weight > 0:
                    run.findings.append(problem("non_finite_training_loss",
                                                f"training loss sum {value} (weight {weight}) at "
                                                f"step {step}", step=step))
                    return run
                objective.backward()
                # A replaced site that did not run leaves the loss native, so a
                # matching loss would prove nothing: every site runs every step.
                idle = [site for site in deployment.patch_set
                        if hooks.counts.get(site, 0) == calls_before.get(site, 0)]
                if idle:
                    run.findings.append(problem("site_not_called",
                                                f"replaced sites {idle} did not run at step {step}",
                                                step=step, sites=idle))
                    return run
                optimizer.step()
                run.train_steps.append({"step": step, "loss_sum": value, "weight": weight,
                                        "mean_loss": value / weight,
                                        "objective": objective_value})
                windows.add(step, value, weight)
                if on_step is not None:
                    stage = f"on_step {step}"
                    on_step(dict(run.train_steps[-1]), model, optimizer, dict(hooks.counts))
                if step in plan.checkpoints:
                    stage = f"validation {step}"
                    validate(step)
                if checkpoint is not None and step < plan.steps and (
                        (checkpoint.every and step % checkpoint.every == 0) or checkpoint.expired()):
                    stage = f"checkpoint {step}"
                    _save(checkpoint, run, model, optimizer, hooks, adapter.device, step)
                    if checkpoint.expired():
                        run.suspended = True
                        run.segments[-1]["suspended_after_step"] = step
                        say(f"suspended after step {step}")
                        return run
            run.site_calls = {"total": dict(hooks.counts)}
            # Every step and checkpoint ran, and nothing invalidated a number.
            run.completed = not run.findings
            if checkpoint is not None and run.completed:
                run.segments[-1]["completed_at_step"] = plan.steps
                from evograd.verification.gate3 import checkpoint as ckpt

                ckpt.remove(checkpoint)
    except Exception as exc:
        run.findings.append(problem("training_runtime_error", f"{type(exc).__name__} at {stage}",
                                    trace=traceback.format_exc(limit=8)))
    finally:
        run.seconds = time.perf_counter() - started
    return run


# ── checkpointed runs ────────────────────────────────────────────────────────


def _save(checkpoint: Any, run: TrainingMeasurement, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, hooks: SiteHooks, device: str, step: int) -> None:
    from evograd.verification.gate3 import checkpoint as ckpt

    ckpt.save(checkpoint, {
        "identity_hash": stable_hash(run.identity), "deployment": run.deployment,
        "run_id": run.run_id, "step": step, "model": _model_of(model).state_dict(),
        "optimizer": optimizer.state_dict(), "rng": ckpt.rng_state(device),
        "train_steps": list(run.train_steps), "validation": dict(run.validation),
        "state": dict(run.state), "site_calls": dict(hooks.counts),
        "segments": list(run.segments)})


def _resume(checkpoint: Any, run: TrainingMeasurement, model: torch.nn.Module,
            optimizer: torch.optim.Optimizer, windows: WindowAccumulator, hooks: SiteHooks,
            device: str) -> int:
    """Restore a saved run in place; returns the next step to run."""
    from evograd.verification.gate3 import checkpoint as ckpt

    data = ckpt.load(checkpoint)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if data is not None and (data["identity_hash"] != stable_hash(run.identity)
                             or data["deployment"] != run.deployment):
        run.segments.append({"discarded_checkpoint": ckpt.discard(checkpoint, "another run")})
        data = None
    if data is None:
        run.segments.append({"started_at_step": 1, "time": stamp})
        return 1
    _model_of(model).load_state_dict(data["model"], strict=True)
    optimizer.load_state_dict(data["optimizer"])
    run.train_steps.extend(data["train_steps"])
    for record in run.train_steps:
        windows.add(record["step"], record["loss_sum"], record["weight"])
    run.validation.update(data["validation"])
    run.state.update(data["state"])
    hooks.counts.update(data["site_calls"])
    run.run_id = data["run_id"]
    run.segments = list(data["segments"]) + [{"resumed_at_step": data["step"] + 1, "time": stamp}]
    ckpt.restore_rng(data["rng"], device)
    return int(data["step"]) + 1
