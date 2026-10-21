"""The three gates for a model's candidates, on real training data.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.<model> \\
        verify --corpus CORPUS/manifest.json --out DIR --candidate CASE_ID=PROGRAM.py [...]

Each candidate is an artifact for one observed case of the model
(:func:`.base.observed_cases`, e.g. ``qwen3_swiglu_mlp/swiglu_mlp``), judged
against the model reference (``Model.reference_op``):

* Gate 1 -- the pair on the declaration's generated inputs at the case's
  configuration, and on every acceptance payload (data orders 303/404).
* Gate 2, operator scope -- thresholds from native repeats and
  ``torch.compile`` on the calibration payloads (orders 101/202), judged on the
  acceptance payloads through the artifact's deployment entry.

Level-2 candidates are then installed together at their model sites:

* Gate 2, model scope -- one forward/backward of the pretrained model on the
  first real TRAIN batch of each order in :data:`MODEL_ORDERS`; per-sample
  thresholds from native repeats and compile at the same sites; every site call
  replayed at its boundary.
* Gate 3 -- ``Model.gate3_steps`` AdamW updates from the pretrained state on
  real TRAIN order :data:`TRAIN_ORDER`; thresholds from native/native/compile
  trajectories.

Level-1 candidates get Gate 1 and operator Gate 2 only: the model's Level-3
run installs Level-2 sites. A candidate is accepted when every gate that
applies to it passed. Gate 3 control runs do not depend on the candidate and
are kept under ``DIR/controls`` (or ``--controls``) for reuse; the candidate's
own runs never are. Every report is written to ``DIR``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

MODEL_ORDERS = (101, 202, 303, 404)
TRAIN_ORDER, TRAIN_SEED = 707, 0
GAMMA, FLOOR_RMS, FLOOR_MAX = 2.0, 1e-4, 1e-4
LR = 1e-4
#: Gate 3 runs save their state this often (and at the deadline).
CHECKPOINT_EVERY = 100


def load_program(path: str | Path):
    spec = importlib.util.spec_from_file_location(f"qwen3_candidate_{abs(hash(str(path)))}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    tmp.replace(path)


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


def _cached_policy(path: Path | None, scope, deployment, judged):
    """A Gate 2 calibration saved under ``--state``, if it still binds to this run.

    A policy whose binding (scope, case, reference, environment, numerics,
    patch set, samples) does not match is recalibrated, never reinterpreted;
    ``verify_integration`` would withhold its thresholds anyway.
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
        policy = Gate2Policy.from_dict(json.loads(path.read_text())["policy"])
    except (CalibrationError, KeyError, ValueError):
        return None
    if {f.code for f in policy_findings(policy, scope, deployment, judged)} & BINDING_CODES:
        return None
    return CalibrationResult(policy=policy, evidence={"reused_policy": str(path)})


def _calibrate(scope, deployment, judged, cache: Path | None, **options):
    """Gate 2 calibration, reused from ``cache`` when it binds to the ``judged``
    samples; saved there when made."""
    from evograd.verification import gate2

    reused = _cached_policy(cache, scope, deployment, judged)
    if reused is not None:
        return reused
    calibration = gate2.calibrate(scope, patch_set=deployment.patch_set, **options)
    if cache is not None and calibration.policy is not None:
        _write(cache, {"policy": calibration.policy.to_dict(), "calibration": calibration.to_dict()})
    return calibration


# ── per case: Gate 1 and operator Gate 2 ─────────────────────────────────────


def judge_case(model, case, module, corpus, *, device: str = "cuda",
               compile_backend: str = "inductor", state: Path | None = None) -> dict[str, Any]:
    """Gate 1 and operator Gate 2. With ``state``, the operator calibration
    (which never depends on the candidate) is saved there and reused."""
    from evograd.verification import gate1, gate2
    from evograd.verification.gate2.adapters.payload import PayloadOperatorScope
    from evograd.verification.payloads import payload_cases

    op = model.reference_op(case.task)
    generated = gate1.verify_pair(op, module, workloads=[case.workload], device=device)
    payloads = gate1.verify_pair(op, module, device=device, cases=payload_cases(
        op, case.workload, corpus, case_id=case.case_id, split="acceptance", device=device))
    scope = PayloadOperatorScope(op, case.workload, corpus, case.case_id, device=device)
    deployment = scope.candidate(module)
    cache = (None if state is None else state / "gate2" / "operator" / corpus.sha256[:16]
             / f"{case.case_id.replace('/', '__')}.json")
    calibration = _calibrate(scope, deployment, scope.samples("acceptance"), cache,
                             samples=scope.samples("calibration"), compile_backend=compile_backend)
    operator = gate2.verify_integration(scope, deployment, policy=calibration.policy,
                                        samples=scope.samples("acceptance"))
    return {"gate1_generated": generated, "gate1_payloads": payloads,
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
    calibration = _calibrate(scope, deployment, samples, cache, samples=samples, boundaries=True,
                             compile_backend=compile_backend)
    report = gate2.verify_integration(scope, deployment, policy=calibration.policy, samples=samples)
    return {"gate2_model_calibration": calibration, "gate2_model": report}


def training_context(model, workload, patch_set) -> dict[str, Any]:
    """What every run of one Gate 3 group shares (bound into the policy)."""
    from .base import determinism_record, objective_of

    return {"reference": {"contract": model.contract, **model.source_identity(),
                          "model": f"unpatched {model.name} at {model.workload_config.get('weights')}"},
            "data_recipe": {"train_order": TRAIN_ORDER, **objective_of(model).data_recipe(workload),
                            "data": workload.data_identity()},
            "patch_set": sorted(patch_set), "numerics": determinism_record()}


def train(model, path: Path, deployment, workload, patch_set, *, reuse: bool,
          log: Callable = print, deadline: float | None = None) -> dict[str, Any]:
    """One training run, written to ``path`` when it completes.

    A control (``reuse``) is read back from ``path`` when an earlier session
    completed it; a candidate run only when it was this very deployment. A run
    interrupted by ``deadline`` is checkpointed beside ``path`` and continued
    by the next call (:class:`~evograd.verification.gate3.checkpoint.RunCheckpoint`).
    """
    from evograd.verification import gate3
    from evograd.verification.gate3.checkpoint import RunCheckpoint

    from .base import objective_of

    if path.exists():
        done = json.loads(path.read_text())
        if reuse or (done.get("completed")
                     and done.get("deployment") == _jsonable(deployment.describe())):
            return done
    adapter = objective_of(model).training(workload, train_data=workload.train_batches(TRAIN_ORDER),
                                           optimizer=gate3.adamw(lr=LR))
    plan = gate3.TrainingPlan(steps=model.gate3_steps, window=1, checkpoints=(), validation=False)
    run = gate3.measure_training(adapter, deployment, plan, seed=TRAIN_SEED, progress=log,
                                 protocol_context=training_context(model, workload, patch_set),
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
    native_a = train(model, group / "native_a.json", Deployment.native(), workload, patch_set,
                     reuse=True, log=log, deadline=deadline)
    native_b = train(model, group / "native_b.json", Deployment.native(), workload, patch_set,
                     reuse=True, log=log, deadline=deadline)
    compiled, _graphs = compile_control(model_scope(model, workload), patch_set,
                                        backend=compile_backend)
    compile_run = train(model, group / "compile.json", compiled, workload, patch_set, reuse=True,
                        log=log, deadline=deadline)
    calibration = gate3.calibrate_global_loss([gate3.TrainingControl(native_a, native_b, compile_run)],
                                              floor_rms=FLOOR_RMS, floor_max=FLOOR_MAX, gamma=GAMMA)
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


def verify(model, candidates: Mapping[str, str], corpus_path: str | Path, out: str | Path, *,
           controls: str | Path | None = None, state: str | Path | None = None,
           device: str = "cuda", compile_backend: str = "inductor", log: Callable = print,
           deadline_min: float | None = None) -> dict[str, Any]:
    """Every gate that applies to each candidate; writes ``DIR/verdict.json``.

    ``state`` (``controls`` is its older name; default ``DIR/controls``) keeps
    what does not depend on the candidate -- Gate 3 control runs and Gate 2
    calibrations -- for any later candidate of the same case or patch set.

    Rerunning the same command continues an interrupted one: stage reports
    of the same inputs (program hashes, corpus, reference) are reused, and a
    Gate 3 run resumes from its checkpoint. With ``deadline_min`` the run
    stops after that many minutes at the next stage boundary or Gate 3
    checkpoint and returns a partial verdict with ``suspended`` set.
    """
    from evograd.opdecl.oracle import reference_identity
    from evograd.verification.payloads import Corpus

    from .base import configure, observed_cases, sha_file

    out = Path(out)
    state = Path(state) if state is not None else (
        Path(controls) if controls is not None else out / "controls")
    deadline = None if deadline_min is None else time.time() + 60.0 * float(deadline_min)
    settings = configure()
    corpus = Corpus.open(corpus_path)
    cases = {c.case_id: c for c in observed_cases(model)}
    unknown = sorted(set(candidates) - set(cases))
    if unknown:
        raise ValueError(f"not observed cases: {unknown}; known: {sorted(cases)}")
    verdict: dict[str, Any] = {"corpus": corpus.binding(), "execution": settings,
                               "candidates": {}, "model": None}
    programs = {case_id: sha_file(path) for case_id, path in candidates.items()}
    try:
        level2 = []
        for case_id, path in sorted(candidates.items()):
            case, module = cases[case_id], load_program(path)
            key = {"case": case_id, "program_sha256": programs[case_id], "corpus": corpus.binding(),
                   "reference": reference_identity(model.reference_op(case.task)),
                   "compile_backend": compile_backend}
            report_path = out / "cases" / f"{case_id.replace('/', '__')}.json"
            done = _stage(report_path, key)
            if done is None:
                _check_deadline(deadline, case_id)
                reports = judge_case(model, case, module, corpus, device=device,
                                     compile_backend=compile_backend, state=state)
                statuses = {name: _status(r) for name, r in reports.items()}
                _write(report_path, {"stage_key": key, "statuses": statuses,
                                     **{name: r.to_dict() for name, r in reports.items()}})
                _release_device_memory()
            else:
                statuses = done["statuses"]
            verdict["candidates"][case_id] = {"program": str(path), "level": case.level, **statuses}
            log(case_id, statuses)
            if case.level == 2:
                level2.append((case, module))
        if level2 and not model.model_gates:
            verdict["model"] = {"not_applicable": f"{model.name} has no model-scope gates yet; "
                                                  "Level-2 candidates are judged by Gate 1 and "
                                                  "operator Gate 2"}
        elif level2:
            workload = model.load_workload()
            deployment = deployment_for(model, level2, workload)
            key = {"programs": {c.case_id: programs[c.case_id] for c, _ in level2},
                   "patch_set": list(deployment.patch_set), "compile_backend": compile_backend,
                   "model": model.name, "workload": model.workload_config}
            done = _stage(out / "gate2_model.json", key)
            if done is None:
                _check_deadline(deadline, "model-scope Gate 2")
                reports = judge_model(model, deployment, workload, compile_backend=compile_backend,
                                      state=state)
                statuses = {name: _status(r) for name, r in reports.items()}
                _write(out / "gate2_model.json", {"stage_key": key, "statuses": statuses,
                                                  **{name: r.to_dict() for name, r in reports.items()}})
                del reports
            else:
                statuses = dict(done["statuses"])
            _release_device_memory()
            training = judge_training(model, deployment, workload, out, state,
                                      compile_backend=compile_backend, log=log, deadline=deadline)
            _write(out / "gate3.json", {name: r.to_dict() for name, r in training.items()})
            statuses.update({name: _status(r) for name, r in training.items()})
            verdict["model"] = {"patch_set": list(deployment.patch_set), **statuses}
            log("model", verdict["model"])
    except Suspended as stop:
        verdict["suspended"] = str(stop)
        log("suspended:", stop, "-- rerun the same command to continue")
        _write(out / "verdict.json", verdict)
        return verdict
    for entry in verdict["candidates"].values():
        gates = [entry["gate1_generated"], entry["gate1_payloads"], entry["gate2_operator"]]
        if entry["level"] == 2 and model.model_gates:
            gates += [verdict["model"]["gate2_model"], verdict["model"]["gate3"]]
        entry["accepted"] = all(g == "passed" for g in gates)
    _write(out / "verdict.json", verdict)
    return verdict
