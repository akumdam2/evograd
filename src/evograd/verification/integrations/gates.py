"""The three gates for a model's candidates, on real training data.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.<model> \\
        verify --corpus CORPUS/manifest.json --out DIR --candidate CASE_ID=PROGRAM.py [...]

Each candidate is an artifact for one observed case of the model
(:func:`.base.observed_cases`, e.g. ``qwen3_swiglu_mlp/swiglu_mlp``), judged
against the model reference (``Model.reference_op``):

* Gate 1 -- the pair on every real acceptance payload (data orders 303/404).
* Gate 2, operator scope -- thresholds from native repeats and
  ``torch.compile`` on the calibration payloads (orders 101/202), judged on the
  acceptance payloads through the artifact's deployment entry.
* Gate 2, model scope (Level 2 only) -- the candidate alone at its model site:
  one forward/backward of the pretrained model on the first real TRAIN batch of
  each order in :data:`MODEL_ORDERS`; thresholds pooled over those batches from
  native repeats and compile at the same site, with :data:`GATE2_MODEL_GAMMA`
  or the model's own ``Model.gate2_model_gamma`` (a whole model judges a
  thousand propagated reductions per batch, the operator scope a handful of
  results); coverage and input/parameter side
  effects checked, no per-call boundary replay (:data:`GATE2_MODEL_BOUNDARIES`).

* Gate 3, single site (Level 2 only) -- continuous training with the candidate
  alone at its model site, after all of its preceding gates pass.

A Level-1 candidate is accepted when Gate 1 and operator Gate 2 pass; a
Level-2 candidate also needs its single-site model Gate 2 and Gate 3.

The **combination** -- the Level-2 candidates deployed together -- is then
judged once more:

* Gate 2, model scope, on the combined patch set;
* Gate 3 -- ``Model.gate3_steps`` AdamW updates from the pretrained state on
  real TRAIN order :data:`TRAIN_ORDER` (the model's ``gate3_optimizer`` and
  ``gate3_accumulation`` where it sets them); thresholds from
  native/native/compile trajectories.

``verify`` combines every accepted Level-2 candidate; ``evaluate``
(:mod:`.evaluate`) combines those whose single-site training step is faster.
Gate 3 control runs and Gate 2 calibrations do not depend on the candidate and
are kept under ``--state`` (default ``DIR/controls``) for reuse; the
candidate's own runs never are. Every report is written to ``DIR``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

MODEL_ORDERS = (101, 202, 303, 404)
#: Gate 2 safety factor at model scope (the operator scope keeps the
#: calibration default, 2): the smallest integer at which every frontend seed
#: passes while a 1% fault at a site still fails. Qwen3-0.6B, 2026-10-23: the
#: residual_rmsnorm seed needs 3.3 (pooled budgets, one sample, two norm/proj
#: weight gradients); the x1.01 fault exceeds four times the budget on 401
#: comparisons in every data order.
GATE2_MODEL_GAMMA = 4.0
#: Model-scope Gate 2 judges the whole model's outputs and parameter gradients, plus
#: coverage and input/parameter side effects. Per-call boundary replay is off: its
#: local check duplicates operator Gate 2 (the same calls on the same orders) with
#: per-call budgets, and failed rounding-level candidates on single bf16 ULP flips.
GATE2_MODEL_BOUNDARIES = False
TRAIN_ORDER, TRAIN_SEED = 707, 0
GAMMA, FLOOR_RMS, FLOOR_MAX = 2.0, 1e-4, 1e-4
LR = 1e-4
#: Gate 3 runs save their state this often (and at the deadline).
CHECKPOINT_EVERY = 100


def model_gamma(model) -> float:
    """Model-scope Gate 2's safety factor for ``model`` (its own, else the protocol's)."""
    return GATE2_MODEL_GAMMA if model.gate2_model_gamma is None else float(model.gate2_model_gamma)


def gate3_gamma(model) -> float:
    """Gate 3's safety factor for ``model`` (its own, else the protocol's)."""
    return GAMMA if model.gate3_gamma is None else float(model.gate3_gamma)


def load_program(path: str | Path):
    spec = importlib.util.spec_from_file_location(f"qwen3_candidate_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: Any, *, attempts: int = 3) -> None:
    """Atomic, under a fresh temporary name per attempt (see ``gate3.checkpoint.save``)."""
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, default=str)
    for attempt in range(attempts):
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text)
            tmp.replace(path)
            return
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0 + attempt)


class Suspended(Exception):
    """A deadline passed between stages; the same command continues from there."""


def _check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.time() >= deadline:
        raise Suspended(f"deadline reached before {stage}")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _stage(path: Path, key: Mapping[str, Any]) -> dict[str, Any] | None:
    """A stage report an earlier run of exactly these inputs wrote, else ``None``."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("stage_key") == _jsonable(key) else None


def _calibration_request(scope, deployment, options):
    """Bind cached calibration to all requested settings and its implementation."""
    import inspect

    from evograd.verification.gate2 import calibration, runner
    from evograd.verification.common import contracts, tensors
    from .base import sha_file

    bound = inspect.signature(calibration.calibrate).bind(
        scope, patch_set=deployment.patch_set, **options)
    bound.apply_defaults()
    settings = dict(bound.arguments)
    settings.pop("adapter")
    for name in ("samples", "validation_samples"):
        if settings[name] is not None:
            settings[name] = [sample.to_dict() for sample in settings[name]]
    return _jsonable({"procedure": calibration.CALIBRATION_PROCEDURE,
                      "sources": {name: sha_file(module.__file__) for name, module in
                                  (("calibration", calibration), ("runner", runner),
                                   ("metrics", tensors), ("contracts", contracts))},
                      "settings": settings})


def _cached_policy(path: Path | None, scope, deployment, judged, request):
    """A Gate 2 calibration saved under ``--state``, if it still binds to this run.

    A policy whose binding (scope, case, reference, environment, numerics,
    patch set, samples, formula, gamma) does not match is recalibrated, never
    reinterpreted; ``verify_integration`` would withhold its thresholds anyway.
    """
    from evograd.verification.gate2.calibration import (
        BINDING_CODES,
        CalibrationError,
        CalibrationResult,
        Gate2Policy,
        policy_findings,
    )

    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("calibration_request") != request:
            return None
        policy = Gate2Policy.from_dict(data["policy"])
    except (CalibrationError, KeyError, ValueError):
        return None
    if float(policy.gamma) != float(request["settings"]["gamma"]):
        return None
    # An operator policy frozen before Gate 1's compile envelope existed carries none.
    if any("envelope" not in entry for entry in (policy.to_dict().get("thresholds") or {}).values()):
        return None
    if {f.code for f in policy_findings(policy, scope, deployment, judged)} & BINDING_CODES:
        return None
    evidence = {"reused_policy": str(path)}
    validation = data.get("calibration", {}).get("evidence", {}).get("holdout_validation")
    if request["settings"]["validation_samples"] is not None:
        if not validation or validation.get("status") != "passed":
            return None
        evidence["holdout_validation"] = validation
    return CalibrationResult(policy=policy, evidence=evidence)


def _calibrate(scope, deployment, judged, cache: Path | None, **options):
    """Gate 2 calibration, reused from ``cache`` when it binds to the ``judged``
    samples; saved there when made."""
    from evograd.verification import gate2
    request = _calibration_request(scope, deployment, options) if cache is not None else None
    reused = _cached_policy(cache, scope, deployment, judged, request)
    # A policy calibrated with boundary replay on (or off) does not judge a run without (with) it.
    if reused is not None and bool(reused.policy.boundaries) == bool(options.get("boundaries", False)):
        return reused
    calibration = gate2.calibrate(scope, patch_set=deployment.patch_set, **options)
    if cache is not None and calibration.policy is not None:
        _write(cache, {"calibration_request": request,
                       "policy": calibration.policy.to_dict(), "calibration": calibration.to_dict()})
    return calibration


# ── per case: Gate 1 and operator Gate 2 ─────────────────────────────────────


def gate1_envelope(op, policy) -> dict[str, float]:
    """Gate 1's elementwise envelope per result, from the operator Gate 2 policy.

    The policy records its own factor times compile's largest elementwise
    deviation (``envelope``, at ``policy.gamma``); Gate 1 applies its own,
    :data:`~evograd.verification.gate1.tolerances.ENVELOPE_GAMMA`, to the same
    deviation, so a stored calibration still binds.
    Gate 2 names a gradient by its argument (``grad:q_weight``, ``grad_in:x``);
    Gate 1 by the declared gradient name (``grad_in:dq_weight``, ``grad_in:dx``).
    An uncalibrated case (no policy) keeps assert_close's rule and Gate 1's rounding floor.
    """
    from evograd.verification.gate1.tolerances import ENVELOPE_GAMMA

    if policy is None:
        return {}
    grads = {a.name: a.grad_name for a in op.active_args()}
    envelope = {}
    for name, entry in (policy.thresholds or {}).items():
        if "envelope" not in entry:
            continue
        prefix, _, rest = name.partition(":")
        if prefix in ("grad", "grad_in") and rest in grads:
            name = f"grad_in:{grads[rest]}"
        envelope[name] = ENVELOPE_GAMMA * float(entry["envelope"]) / float(policy.gamma)
    return envelope


def operator_calibration(model, case, corpus, *, device: str = "cuda",
                         compile_backend: str = "inductor", state: Path | None = None):
    """The case's operator scope on ``corpus`` and its Gate 2 calibration, saved
    under ``state`` and reused.

    The calibration binds the patch set (the scope's one site), never a
    candidate, so the reference itself stands in for one: an ADBench task pack
    takes Gate 1's envelope from here before any candidate exists.
    """
    from evograd.verification.common.execution import CANDIDATE, Deployment
    from evograd.verification.gate2.adapters.payload import PayloadOperatorScope

    scope = PayloadOperatorScope(model.reference_op(case.task), case.workload, corpus,
                                 case.case_id, device=device)
    stand_in = Deployment("reference", CANDIDATE, {s: scope.native_site(s) for s in scope.sites()})
    cache = (None if state is None else state / "gate2" / "operator" / corpus.sha256[:16]
             / f"{case.case_id.replace('/', '__')}.json")
    calibration = _calibrate(scope, stand_in, scope.samples("acceptance"), cache,
                             samples=scope.samples("calibration"), compile_backend=compile_backend,
                             validation_samples=scope.samples("acceptance"))
    return scope, calibration


def judge_case(model, case, module, corpus, *, device: str = "cuda",
               compile_backend: str = "inductor", state: Path | None = None) -> dict[str, Any]:
    """Gate 1 on the case's real payloads and operator Gate 2. With ``state``,
    the operator calibration (which never depends on the candidate) is saved
    there and reused.

    Gate 1 takes the recorded training payloads only: generated inputs at the
    same shape add no coverage, and the fixed dtype tolerance judges them at a
    magnitude the model never trains at (user decision 2026-10-23)."""
    from evograd.verification import gate1, gate2
    from evograd.verification.payloads import payload_cases

    scope, calibration = operator_calibration(model, case, corpus, device=device,
                                              compile_backend=compile_backend, state=state)
    op = scope.op
    deployment = scope.candidate(module)
    envelope = gate1_envelope(op, calibration.policy)
    payloads = gate1.verify_pair(op, module, device=device, envelope=envelope, cases=payload_cases(
        op, case.workload, corpus, case_id=case.case_id, split="acceptance", device=device))
    operator = gate2.verify_integration(scope, deployment, policy=calibration.policy,
                                        samples=scope.samples("acceptance"))
    return {"gate1_payloads": payloads,
            "gate2_operator_calibration": calibration, "gate2_operator": operator}


# ── Level-2 candidates in the model: Gate 2 model scope and Gate 3 ───────────


def model_scope(model, workload):
    """The whole model, every site's native being its model reference."""
    from evograd.opdecl.oracle import resolve_runtime_forward

    from .base import objective_of

    registry = workload.site_registry
    return objective_of(model).model_scope(workload, {
        site: resolve_runtime_forward(model.reference_op(registry.require(site).op))
        for site in registry.names})


def site_for_case(registry, case) -> str:
    """The model site a Level-2 case's candidate is installed at.

    One site per task, except where one task serves several sites at different
    configurations (ConvNeXt's four stages): then the site whose declared
    preflight workloads include the case's dims and dtype.
    """
    sites = [s for s in registry.names if registry.require(s).op == case.task]
    if len(sites) > 1:
        def serves(site) -> bool:
            return any(dict(w.dims) == dict(case.workload.dims) and w.dtype == case.workload.dtype
                       for w in (registry.require(site).preflight or ()))

        sites = [s for s in sites if serves(s)]
    if len(sites) != 1:
        raise ValueError(f"{case.case_id}: {len(sites)} model sites serve this case")
    return sites[0]


def deployment_for(model, level2, workload):
    """Level-2 candidates at their model sites, each through its artifact's own entry."""
    from evograd.verification.common.execution import candidate
    from evograd.verification.gate2.adapters.operator import SITE, OperatorScope

    registry, sites, programs = workload.site_registry, {}, {}
    for case, module in level2:
        site = site_for_case(registry, case)
        if site in sites:
            raise ValueError(f"two candidates for model site {site}")
        single = OperatorScope(model.reference_op(case.task), case.workload).candidate(module)
        sites[site], programs[case.case_id] = single.sites[SITE], dict(single.identity)
    return candidate("candidate", sites, programs=programs)


def judge_model(model, deployment, workload, *, compile_backend: str = "inductor",
                state: Path | None = None) -> dict[str, Any]:
    from evograd.verification import gate2

    scope = model_scope(model, workload)
    samples = scope.samples(MODEL_ORDERS)
    cache = (None if state is None
             else state / "gate2" / "model" / f"{'+'.join(deployment.patch_set)}.json")
    calibration = _calibrate(scope, deployment, samples, cache, samples=samples,
                             boundaries=GATE2_MODEL_BOUNDARIES,
                             compile_backend=compile_backend, gamma=model_gamma(model))
    report = gate2.verify_integration(scope, deployment, policy=calibration.policy, samples=samples)
    return {"gate2_model_calibration": calibration, "gate2_model": report}


def _gate1_policy(op) -> dict[str, Any]:
    """What a cached Gate 1 verdict was judged under."""
    from evograd.verification.gate1.tolerances import policy_identity

    return policy_identity(op)


def _gate2_model_rule(model) -> dict[str, Any]:
    """What a cached model-scope Gate 2 verdict was judged under."""
    from evograd.verification.gate2.calibration import CALIBRATION_PROCEDURE, FORMULAS, MATCHED

    return {"gamma": model_gamma(model), "formula": FORMULAS[MATCHED],
            "calibration": CALIBRATION_PROCEDURE, "boundaries": GATE2_MODEL_BOUNDARIES}


def training_context(model, workload, patch_set) -> dict[str, Any]:
    """What every run of one Gate 3 group shares (bound into the policy)."""
    from .base import determinism_record, objective_of

    from evograd.verification.reuse import enabled
    from evograd.verification.gate3.global_loss import NATIVE_REUSE

    return {**({"native_reuse": NATIVE_REUSE} if enabled("share_gate3_native") else {}),
            "reference": {"contract": model.contract, **model.source_identity(),
                          "model": f"unpatched {model.name} at {model.workload_config.get('weights')}"},
            "data_recipe": {"train_order": TRAIN_ORDER, **objective_of(model).data_recipe(workload),
                            "data": workload.data_identity()},
            "patch_set": sorted(patch_set), "numerics": determinism_record()}


def train(model, path: Path, deployment, workload, patch_set, *, reuse: bool,
          log: Callable = print, deadline: float | None = None) -> dict[str, Any]:
    """One training run, written to ``path`` when it completes.

    A control (``reuse``) is read back from ``path`` when an earlier session
    completed it for the same workload description, optimizer, plan and
    protocol context; a candidate run only when it was also this very
    deployment. A run interrupted by ``deadline`` is checkpointed beside
    ``path`` and continued by the next call
    (:class:`~evograd.verification.gate3.checkpoint.RunCheckpoint`).
    """
    from evograd.verification import gate3
    from evograd.verification.common.reports import jsonable
    from evograd.verification.gate3.checkpoint import RunCheckpoint

    from .base import objective_of

    optimizer = gate3.adamw(**{"lr": LR, **dict(model.gate3_optimizer)})
    plan = gate3.TrainingPlan(steps=model.gate3_steps, window=1, checkpoints=(), validation=False,
                              accumulation=model.gate3_accumulation)
    from evograd.verification.reuse import enabled
    native_shared = enabled("share_gate3_native") and deployment.role == "native"
    context = training_context(model, workload, () if native_shared else patch_set)
    adapter = objective_of(model).training(workload, train_data=workload.train_batches(TRAIN_ORDER),
                                           optimizer=optimizer)
    if native_shared:
        from evograd.verification.common import contracts as C
        # Path independent of the experiment's patch set, but bound to all
        # observable execution settings. train() checks the stored identity too.
        identity_key = C.stable_hash({"context": context, "adapter": adapter.identity(),
                                     "optimizer": optimizer.config, "plan": plan.to_dict(),
                                     "seed": TRAIN_SEED, "data": adapter.data_identity(),
                                     "environment": C.environment_fingerprint(adapter.device)})
        path = path.parent / "shared_native" / identity_key / path.name
    if path.exists():
        done = json.loads(path.read_text())
        identity = done.get("identity") or {}
        # A run of another workload description (an older checkout's), optimizer,
        # plan or context is never reused: the verdict would refuse the pair as
        # differently bound.
        same_run = (identity.get("adapter") == _jsonable(jsonable(dict(adapter.identity())))
                    and identity.get("optimizer") == _jsonable(jsonable(dict(optimizer.config)))
                    and identity.get("plan") == _jsonable(jsonable(plan.to_dict()))
                    and identity.get("protocol_context") == _jsonable(jsonable(context)))
        if same_run and (reuse or (done.get("completed")
                                   and done.get("deployment") == _jsonable(deployment.describe()))):
            return done
    run = gate3.measure_training(adapter, deployment, plan, seed=TRAIN_SEED, progress=log,
                                 protocol_context=context,
                                 checkpoint=RunCheckpoint(path.with_suffix(".ckpt"), every=CHECKPOINT_EVERY,
                                                          deadline=deadline))
    data = run.to_dict()
    if run.completed:
        _write(path, data)
    if run.suspended:
        raise Suspended(f"{path.name}: saved after step {len(run.train_steps)} of {plan.steps}")
    return data


def judge_training(model, deployment, workload, out: Path, controls: Path, *,
                   compile_backend: str = "inductor", log: Callable = print,
                   deadline: float | None = None) -> dict[str, Any]:
    from evograd.verification import gate3
    from evograd.verification.common.execution import Deployment
    from evograd.verification.gate2.calibration import compile_control

    patch_set = deployment.patch_set
    group = controls / "gate3" / "+".join(patch_set)
    settings = {**dict(model.gate3_optimizer),
                **({"accumulation": model.gate3_accumulation} if model.gate3_accumulation != 1 else {})}
    if settings:
        # Controls of other training settings live beside, not over, the protocol's.
        group = group / "_".join(f"{key}={value}" for key, value in sorted(settings.items()))
    from evograd.verification.reuse import enabled
    native_group = controls / "gate3" if enabled("share_gate3_native") else group
    native_a = train(model, native_group / "native_a.json", Deployment.native(), workload, patch_set,
                     reuse=True, log=log, deadline=deadline)
    native_b = train(model, native_group / "native_b.json", Deployment.native(), workload, patch_set,
                     reuse=True, log=log, deadline=deadline)
    compiled, _graphs = compile_control(model_scope(model, workload), patch_set,
                                        backend=compile_backend)
    compile_run = train(model, group / "compile.json", compiled, workload, patch_set, reuse=True,
                        log=log, deadline=deadline)
    calibration = gate3.calibrate_global_loss([gate3.TrainingControl(native_a, native_b, compile_run)],
                                              floor_rms=FLOOR_RMS, floor_max=FLOOR_MAX,
                                              gamma=gate3_gamma(model),
                                              share_native=enabled("share_gate3_native"))
    candidate = train(model, out / "gate3_candidate.json", deployment, workload, patch_set,
                      reuse=False, log=log, deadline=deadline)
    verdict = gate3.verify_global_loss(candidate, native_a, policy=calibration.policy)
    return {"gate3_calibration": calibration, "gate3": verdict}


# ── all of it ────────────────────────────────────────────────────────────────


def _release_device_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _status(report) -> str:
    return getattr(report, "status", None) or report.to_dict()["status"]


@dataclass
class Run:
    """What every stage of one verify (or evaluate) run shares."""

    model: Any
    out: Path
    state: Path
    corpus: Any
    corpus_path: Path
    cases: dict[str, Any]
    programs: dict[str, str]
    paths: dict[str, str]
    modules: dict[str, Any]
    device: str = "cuda"
    compile_backend: str = "inductor"
    log: Callable = print
    deadline: float | None = None
    _workload: Any = None

    @property
    def workload(self):
        if self._workload is None:
            self._workload = self.model.load_workload()
        return self._workload


def open_run(model, candidates: Mapping[str, str], corpus_path: str | Path, out: str | Path, *,
             controls: str | Path | None = None, state: str | Path | None = None,
             device: str = "cuda", compile_backend: str = "inductor", log: Callable = print,
             deadline_min: float | None = None) -> Run:
    """Load the corpus and the candidate programs; refuse cases the model does not have."""
    from evograd.verification.payloads import Corpus

    from .base import observed_cases, sha_file

    out = Path(out)
    state = Path(state) if state is not None else (
        Path(controls) if controls is not None else out / "controls")
    from evograd.verification.reuse import FLAGS, settings as reuse_settings
    reuse = reuse_settings()
    if any(reuse[name] for name in FLAGS):
        state = state / "native_reuse_v1" / "-".join(str(int(reuse[name])) for name in FLAGS)
    cases = {c.case_id: c for c in observed_cases(model)}
    unknown = sorted(set(candidates) - set(cases))
    if unknown:
        raise ValueError(f"not observed cases: {unknown}; known: {sorted(cases)}")
    return Run(model=model, out=out, state=state, corpus=Corpus.open(corpus_path),
               corpus_path=Path(corpus_path), cases=cases,
               programs={cid: sha_file(path) for cid, path in candidates.items()},
               paths={cid: str(path) for cid, path in candidates.items()},
               modules={cid: load_program(path) for cid, path in sorted(candidates.items())},
               device=device, compile_backend=compile_backend, log=log,
               deadline=None if deadline_min is None else time.time() + 60.0 * float(deadline_min))


def _judged_stage(run: Run, path: Path, key: dict, judge: Callable[[], dict]) -> dict[str, str]:
    """Statuses of a stage: an earlier report of the same inputs, else ``judge()``'s."""
    from evograd.verification.reuse import settings as reuse_settings
    reuse = reuse_settings()
    if any(reuse[name] for name in ("share_gate3_native", "share_gate2_calibration_native", "share_gate2_verification_reference")):
        key = {**key, "calibration_reuse": reuse}
    # A stage verdict must not outlive the calibration/comparison code that
    # produced it, even if the candidate and corpus are unchanged.
    from evograd.verification.gate2 import calibration, runner
    from evograd.verification.common import contracts, tensors
    from .base import sha_file
    key = {**key, "verification_implementation": {
        "gates": sha_file(__file__), "calibration": sha_file(calibration.__file__),
        "comparison": sha_file(runner.__file__), "metrics": sha_file(tensors.__file__),
        "contracts": sha_file(contracts.__file__)},
        "verification_environment": contracts.environment_fingerprint(getattr(run, "device", None))}
    done = _stage(path, key)
    if done is not None:
        return dict(done["statuses"])
    _check_deadline(run.deadline, path.stem)
    reports = judge()
    statuses = {name: _status(r) for name, r in reports.items()}
    _write(path, {"stage_key": key, "statuses": statuses,
                  **{name: r.to_dict() for name, r in reports.items()}})
    del reports
    _release_device_memory()
    return statuses


def verify_cases(run: Run) -> dict[str, Any]:
    """Gate 1 and operator Gate 2 for every candidate; Level 2 also needs its
    single-site model Gate 2 and, after those pass, Gate 3."""
    from evograd.opdecl.oracle import reference_identity

    model = run.model
    verdict: dict[str, Any] = {"corpus": run.corpus.binding(), "candidates": {}}
    for case_id in sorted(run.modules):
        case, module = run.cases[case_id], run.modules[case_id]
        key = {"case": case_id, "program_sha256": run.programs[case_id],
               "corpus": run.corpus.binding(),
               "reference": reference_identity(model.reference_op(case.task)),
               "gate1_policy": _gate1_policy(model.reference_op(case.task)),
               "compile_backend": run.compile_backend}
        statuses = _judged_stage(
            run, run.out / "cases" / f"{case_id.replace('/', '__')}.json", key,
            lambda: judge_case(model, case, module, run.corpus, device=run.device,
                               compile_backend=run.compile_backend, state=run.state))
        entry = {"program": run.paths[case_id], "program_sha256": run.programs[case_id],
                 "level": case.level, **statuses}
        if case.level == 2 and model.model_gates:
            deployment = deployment_for(model, [(case, module)], run.workload)
            key = {"programs": {case_id: run.programs[case_id]},
                   "patch_set": list(deployment.patch_set), "compile_backend": run.compile_backend,
                   "model": model.name, "workload": model.workload_config,
                   "gate2_model": _gate2_model_rule(model)}
            entry.update(_judged_stage(
                run, run.out / "gate2_model" / f"{case_id.replace('/', '__')}.json", key,
                lambda: judge_model(model, deployment, run.workload,
                                    compile_backend=run.compile_backend, state=run.state)))
            if all(entry.get(gate) == "passed"
                   for gate in ("gate1_payloads", "gate2_operator", "gate2_model")):
                out = run.out / "gate3_single_site" / case_id.replace("/", "__")
                _check_deadline(run.deadline, f"{case_id} single-site Gate 3")
                training = judge_training(model, deployment, run.workload, out, run.state,
                                          compile_backend=run.compile_backend, log=run.log,
                                          deadline=run.deadline)
                _write(out / "gate3.json", {name: r.to_dict() for name, r in training.items()})
                entry.update({name: _status(r) for name, r in training.items()})
                del training
                _release_device_memory()
            else:
                entry["gate3"] = "skipped"
        entry["accepted"] = accepted(model, entry)
        verdict["candidates"][case_id] = entry
        run.log(case_id, {k: v for k, v in entry.items() if k not in ("program", "program_sha256")})
    return verdict


def accepted(model, entry: Mapping[str, Any]) -> bool:
    """Gate 1 on real payloads and operator Gate 2; for Level 2 also the
    single-site model Gate 2 and Gate 3, where the model has model-scope gates."""
    gates = [entry.get("gate1_payloads"), entry.get("gate2_operator")]
    if entry.get("level") == 2 and model.model_gates:
        gates.extend((entry.get("gate2_model"), entry.get("gate3")))
    return all(g == "passed" for g in gates)


def verify_combination(run: Run, case_ids) -> dict[str, Any]:
    """Model Gate 2 and Gate 3 of the given accepted Level-2 candidates deployed together."""
    model = run.model
    case_ids = sorted(case_ids)
    if not model.model_gates:
        return {"cases": case_ids, "accepted": False,
                "not_applicable": f"{model.name} has no model-scope gates yet; Level-2 candidates "
                                  "are judged by Gate 1 and operator Gate 2"}
    if not case_ids:
        return {"cases": [], "accepted": False, "empty": "no Level-2 candidate was selected"}
    level2 = [(run.cases[cid], run.modules[cid]) for cid in case_ids]
    out = run.out / "combination"
    deployment = deployment_for(model, level2, run.workload)
    key = {"programs": {cid: run.programs[cid] for cid in case_ids},
           "patch_set": list(deployment.patch_set), "compile_backend": run.compile_backend,
           "model": model.name, "workload": model.workload_config,
           "gate2_model": _gate2_model_rule(model)}
    statuses = _judged_stage(run, out / "gate2_model.json", key,
                             lambda: judge_model(model, deployment, run.workload,
                                                 compile_backend=run.compile_backend,
                                                 state=run.state))
    training = judge_training(model, deployment, run.workload, out, run.state,
                              compile_backend=run.compile_backend, log=run.log,
                              deadline=run.deadline)
    _write(out / "gate3.json", {name: r.to_dict() for name, r in training.items()})
    statuses.update({name: _status(r) for name, r in training.items()})
    result = {"cases": case_ids, "patch_set": list(deployment.patch_set), **statuses,
              "accepted": statuses.get("gate2_model") == "passed" and statuses.get("gate3") == "passed"}
    run.log("combination", result)
    return result


def verify(model, candidates: Mapping[str, str], corpus_path: str | Path, out: str | Path, *,
           controls: str | Path | None = None, state: str | Path | None = None,
           device: str = "cuda", compile_backend: str = "inductor", log: Callable = print,
           deadline_min: float | None = None) -> dict[str, Any]:
    """Every gate that applies to each candidate, then the combination of every
    accepted Level-2 candidate; writes ``DIR/verdict.json``.

    ``state`` (``controls`` is its older name; default ``DIR/controls``) keeps
    what does not depend on the candidate -- Gate 3 control runs and Gate 2
    calibrations -- for any later candidate of the same case or patch set.

    Rerunning the same command continues an interrupted one: stage reports
    of the same inputs (program hashes, corpus, reference) are reused, and a
    Gate 3 run resumes from its checkpoint. With ``deadline_min`` the run
    stops after that many minutes at the next stage boundary or Gate 3
    checkpoint and returns a partial verdict with ``suspended`` set.
    """
    from .base import configure

    settings = configure()
    run = open_run(model, candidates, corpus_path, out, controls=controls, state=state,
                   device=device, compile_backend=compile_backend, log=log,
                   deadline_min=deadline_min)
    verdict: dict[str, Any] = {"execution": settings, "candidates": {}, "combination": None}
    try:
        verdict.update(verify_cases(run))
        chosen = [cid for cid, entry in verdict["candidates"].items()
                  if entry["level"] == 2 and entry["accepted"]]
        verdict["combination"] = verify_combination(run, chosen)
    except Suspended as stop:
        verdict["suspended"] = str(stop)
        log("suspended:", stop, "-- rerun the same command to continue")
    _write(run.out / "verdict.json", verdict)
    return verdict
