"""The benchmark's evaluation of one model's candidates: verification, then timing.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 evograd evaluate <model> \\
        --corpus CORPUS/manifest.json --out DIR --candidate CASE_ID=PROGRAM.py [...]
    evograd summarize DIR_A/evaluation.json DIR_B/evaluation.json [--out FILE]

Every submission follows the same steps, whoever generated it:

0. **Static rules** (:func:`check_rules`): the submission is a packaged
   artifact whose ``EVOLVE-BLOCK`` obeys the primitive rules
   (:func:`evograd.pipelines.shared.primitives.check_source`): no library
   contraction, normalization, attention or autograd call that the run did
   not grant. A *violation* is recorded and the candidate goes no further:
   neither verified nor timed.
1. **Verification** (:func:`evograd.verification.integrations.gates.verify_cases`): Gate 1 on the
   model's real training payloads and operator Gate 2 for every candidate, and
   the single-site model Gate 2 and Gate 3 for every Level-2 candidate (Gate 3
   where it applies to the model, ``Model.gate3_applies``). A candidate
   passing every gate that applies is *accepted*.
2. **Tier 2** for every *executable* candidate, accepted or not: the
   deployed forward and autograd backward on every recorded call of the case
   in one training step of the verification corpus (:data:`TIER2_INPUTS`),
   against eager PyTorch, ``torch.compile`` and any declared library baseline,
   all through the autograd engine and on the same calls. A candidate is
   executable when one forward and backward through the timed route on the
   first of those calls raises nothing, returns the declared number of outputs
   and gradients with the eager shapes and dtypes, and leaves its inputs
   unchanged; anything else is not timed.
3. **Tier 3, single site**: the training step with each executable Level-2
   candidate alone at its model site, against eager.
4. **Combination**: the accepted Level-2 candidates whose single-site speedup
   has a 95% interval above one (:data:`COMBINATION_RULE`), judged together by
   model Gate 2 and Gate 3 (where it applies), then timed at Tier 3 against
   eager and the whole-model ``torch.compile``.
5. **Summary** (:func:`summarize`): per level, the speedup over the accepted
   configurations, coverage and the verification funnel; peak memory at
   Level 3.

Verification runs under the deterministic settings
:func:`evograd.verification.integrations.base.configure` sets; timing runs with deterministic algorithms off, as training does. Every
stage's result is saved under ``DIR`` and reused when the same command is
rerun on the same programs.
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evograd.verification.integrations.base import ACCEPTANCE_ORDERS
from evograd.verification.integrations.gates import (
    Run,
    Suspended,
    _check_deadline,
    _jsonable,
    _release_device_memory,
    _write,
    deployment_for,
    open_run,
    verify_cases,
    verify_combination,
)

PROTOCOL = "evograd-evaluation-v3"
BASELINE_REVISION = "whole-model-baselines/1"
REFERENCE = "eager"
TIER2_OPTIONS = {"warmup": 10, "reps": 50, "blocks": 3, "seed": 0}
#: What Tier 2 times: recorded calls, fixed by the corpus and the same for every
#: agent and provider. Call k of the step is timed with :data:`TIER2_OPTIONS`
#: and seed ``seed + k`` for its block order.
TIER2_INPUTS = {"source": "real_training_payload", "split": "acceptance",
                "order": ACCEPTANCE_ORDERS[0],
                "calls": ("every call of the case the corpus records in that order's training step, "
                           "in call order: all of them, or the subset the model declares"),
                "speedup": "sum of the baseline's per-call medians over the candidate's"}
TIER3_OPTIONS = {"warmup": 3, "steps": 10, "blocks": 10, "loss_steps": 5,
                 "learning_rate": 1e-4, "seed": 0}
COMBINATION_RULE = ("every accepted Level-2 candidate whose single-site Tier 3 speedup over eager "
                    "has a 95% bootstrap interval with its lower bound above 1")
#: Baselines included in the per-level aggregate; every T2 baseline uses autograd.
ENGINE_BASELINES = ("eager", "torch_compile")


class NotExecutable(Exception):
    """The candidate cannot be run at its case; it is not timed."""


# ── timing settings ──────────────────────────────────────────────────────────


def timing_mode() -> dict[str, Any]:
    """Deterministic algorithms off for timing, as a training run has them."""
    import torch

    from evograd.verification.integrations.base import determinism_record

    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    return determinism_record()


# ── static rules ─────────────────────────────────────────────────────────────


def check_rules(op, module, path: str | Path) -> str | None:
    """The static rules on a submission's source; the violation, or ``None``.

    The granted primitives are the ones the artifact's own ``ARTIFACT_CONTRACT``
    records, as every later consumer of the artifact reads them. A direct
    deployment artifact also gets the deployment-layer checks
    (:func:`~evograd.pipelines.shared.artifact.validate_artifact`); a bare pair
    only its evolve block. A module without an ``EVOLVE-BLOCK`` is a violation:
    the rules have nothing to span.
    """
    from evograd.pipelines.shared.artifact import (
        ArtifactError,
        deployment_entry,
        validate_artifact,
    )
    from evograd.pipelines.shared.primitives import PrimitiveViolation, check_source, normalize

    source = Path(path).read_text(encoding="utf-8")
    declared = getattr(module, "ARTIFACT_CONTRACT", None)
    allowed = normalize((declared or {}).get("allowed_primitives", ())
                        if isinstance(declared, dict) else ())
    try:
        if deployment_entry(module) is not None:
            validate_artifact(op, module, source=source, allowed_primitives=allowed)
        else:
            check_source(source, allowed=allowed)
    except (ArtifactError, PrimitiveViolation, SyntaxError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def unimportable(candidates: Mapping[str, str], *, timeout: int = 600) -> dict[str, dict[str, Any]]:
    """Submissions whose module cannot be imported, each checked in a process of its own.

    Such a submission is neither verified nor timed; the rest of the agent's
    cases are judged as usual instead of the whole evaluation failing at import.
    """
    import subprocess
    import sys

    from evograd.verification.integrations.base import sha_file

    check = ("import importlib.util, sys\n"
             "spec = importlib.util.spec_from_file_location('submission_import_check', sys.argv[1])\n"
             "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n")
    failed = {}
    for case_id, path in sorted(candidates.items()):
        try:
            done = subprocess.run([sys.executable, "-c", check, str(path)], capture_output=True,
                                  text=True, timeout=timeout)
            error = None if done.returncode == 0 else (
                (done.stderr or done.stdout).strip().splitlines() or [f"exit {done.returncode}"])[-1]
        except subprocess.TimeoutExpired:
            error = f"import did not finish within {timeout}s"
        if error is not None:
            failed[case_id] = {"program": str(path), "program_sha256": sha_file(path),
                               "import_error": error}
    return failed


# ── Tier 2 ───────────────────────────────────────────────────────────────────


def tier2_payloads(corpus, op, case) -> list[dict[str, Any]]:
    """The recorded calls Tier 2 times for ``case`` (:data:`TIER2_INPUTS`), in call order.

    Raises :class:`~evograd.verification.payloads.CorpusError` when the corpus
    does not hold every call of the case in that step: nothing is generated in
    place of a missing call.
    """
    from evograd.verification.payloads import CorpusError
    from evograd.verification.payloads.gate1 import check_payload_interface

    order = TIER2_INPUTS["order"]
    records = sorted((r for r in corpus.payloads(case_id=case.case_id, split=TIER2_INPUTS["split"])
                      if (r.get("batch") or {}).get("order") == order),
                     key=lambda r: r["call"]["index_in_step"])
    if not records or len(records) != case.calls_recorded:
        raise CorpusError(f"corpus {corpus.corpus_id} records {len(records)} calls of "
                          f"{case.case_id!r} in the step of order {order}; the model records "
                          f"{case.calls_recorded} per step and Tier 2 times recorded calls only")
    problems = [p for r in records for p in check_payload_interface(op, r)]
    if problems:
        raise CorpusError("; ".join(problems[:5]))
    return records


def tier2_binding(corpus, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a Tier 2 result is bound to: the rule, the corpus and the very calls."""
    return {**TIER2_INPUTS, "corpus": corpus.binding(),
            "payload_ids": [r["payload_id"] for r in records]}


def _shape_dtype(values) -> list[tuple[tuple[int, ...], str] | None]:
    import torch

    return [(tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else None for v in values]


def _run_once(op, provider, values):
    """One forward and backward of ``provider`` on its own copy of ``values``;
    the outputs and gradients. Raises if it changes its inputs."""
    from evograd.evaluation.common.provider import assert_tensors_unchanged, snapshot_tensors
    from evograd.opdecl.inputs import as_output_tuple, upstream_grad_values
    from evograd.verification.common.tensors import isolate

    own = isolate(values)
    snapshot = snapshot_tensors(own, may_overwrite=tuple(op.backward_may_overwrite or ()))
    output, saved = provider.forward(own)
    grads = provider.backward(upstream_grad_values(op, own), saved, own)
    assert_tensors_unchanged(own, snapshot, provider=provider.name)
    return as_output_tuple(op, output), tuple(grads)


def check_executable(op, module, values) -> None:
    """One forward and backward through the route Tier 2 times, on ``values``
    (one recorded call); raises :class:`NotExecutable` with the reason."""
    from evograd.evaluation.common.provider import (
        candidate_autograd_provider,
        pytorch_autograd_provider,
    )
    from evograd.pipelines.shared.artifact import deployment_entry, validate_artifact

    try:
        if deployment_entry(module) is not None:
            validate_artifact(op, module)
        output, grads = _run_once(op, candidate_autograd_provider(op, module), values)
        ref_output, ref_grads = _run_once(op, pytorch_autograd_provider(op), values)
    except NotExecutable:
        raise
    except Exception as exc:
        raise NotExecutable(f"{type(exc).__name__}: {exc}") from exc
    got = _shape_dtype(output), _shape_dtype(grads)
    want = _shape_dtype(ref_output), _shape_dtype(ref_grads)
    if got[0] != want[0]:
        raise NotExecutable(f"outputs {got[0]}, the declaration's are {want[0]}")
    if got[1] != want[1]:
        raise NotExecutable(f"gradients {got[1]}, the declaration's are {want[1]}")


def check_baseline(op, provider, values, workload) -> None:
    """A declared library baseline must compute what eager does on ``values``
    (one recorded call) before it is timed.

    The declared tolerances are stated for unit-scale values, and recorded
    gradients are far below an absolute tolerance (a baseline returning zeros
    would pass), so each result is compared after dividing both sides by the
    largest magnitude of eager's.
    """
    import torch

    from evograd.evaluation.common.provider import pytorch_autograd_provider

    output, grads = _run_once(op, provider, values)
    ref_output, ref_grads = _run_once(op, pytorch_autograd_provider(op), values)
    names = [*op.output_names, *op.grad_names()]
    for name, got, ref in zip(names, (*output, *grads), (*ref_output, *ref_grads)):
        atol, rtol = op.tolerance_for(workload, name)
        scale = ref.detach().float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        if (got.shape != ref.shape or got.dtype != ref.dtype
                or not torch.allclose(got.float() / scale, ref.float() / scale,
                                      atol=atol, rtol=rtol)):
            raise RuntimeError(f"{op.name}: the {provider.name} baseline differs from eager in "
                               f"{name!r} on a recorded call")


def time_case(op, module, inputs, *, options: Mapping[str, Any] = TIER2_OPTIONS) -> dict[str, Any]:
    """Tier 2: deployed forward and autograd backward, one fair run per baseline.

    ``inputs`` are the case's recorded calls as ``(workload, load)`` pairs
    (:func:`~evograd.evaluation.tier1.fair.run_fair_benchmarks`); every
    baseline is timed against the candidate on the same calls. Every provider
    runs its differentiable deployment and ``torch.autograd.grad``; this
    operator scope includes neither an optimizer nor ``.grad`` accumulation.
    The shared fair timer does not determine correctness: real-payload gate
    verdicts are produced separately by :func:`verify_cases`.
    """
    from evograd.evaluation.common.provider import (
        candidate_autograd_provider,
        pytorch_autograd_provider,
        renamed_provider,
        torch_compile_provider,
    )
    from evograd.evaluation.tier1.fair import run_fair_benchmarks
    from evograd.opdecl.baselines import baseline_candidate_module
    from evograd.verification.payloads import CorpusError

    baselines: dict[str, Callable[[], Any]] = {
        "eager": lambda: pytorch_autograd_provider(op),
        "torch_compile": lambda: torch_compile_provider(op, name="torch_compile", dynamic=False),
    }
    for name in sorted(op.performance_baselines):
        def declared(name=name):
            provider = renamed_provider(
                candidate_autograd_provider(op, baseline_candidate_module(op, name)), name)
            workload, load = inputs[0]
            check_baseline(op, provider, load(), workload)
            return provider

        baselines[name] = declared
    results: dict[str, Any] = {}
    for name, make_baseline in baselines.items():
        try:
            baseline = make_baseline()
            candidate = candidate_autograd_provider(op, module)
            report = run_fair_benchmarks(op, candidate, baseline, inputs=inputs, **dict(options))
        except CorpusError:  # an unreadable recorded call is the corpus's fault, not the baseline's
            raise
        except Exception as exc:  # one baseline must not cost the others
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        aggregate = report["aggregate"]
        results[name] = {
            "route": "autograd",
            "speedup": aggregate["speedup_pair_full"],
            "ci95": aggregate["speedup_pair_full_ci95"],
            "candidate_ms": aggregate["candidate_pair_full_ms"],
            "baseline_ms": aggregate["baseline_pair_full_ms"],
            "speedup_forward": aggregate["speedup_forward"],
            "speedup_backward": aggregate["speedup_backward"],
        }
        _release_device_memory()
    return results


# ── Tier 3 ───────────────────────────────────────────────────────────────────


def deployment_kernels(run: Run, case_ids: Sequence[str]):
    """The Level-2 candidates at their sites, as the very callables verification installs."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    from evograd.verification.integrations.gates import site_for_case

    registry = run.workload.site_registry
    level2 = [(run.cases[cid], run.modules[cid]) for cid in case_ids]
    deployment = deployment_for(run.model, level2, run.workload)
    modules: dict[str, list] = {}
    for case, module in level2:
        modules.setdefault(site_for_case(registry, case), []).append(module)
    kernels = KernelSet(registry=registry)
    for site, call in deployment.sites.items():
        # Several candidates at one site (one per configuration) are no one module.
        module = modules[site][0] if len(modules[site]) == 1 else None
        kernels = patch(kernels, site, call, source=KernelSource(
            site=site, op_name=registry.require(site).op, module=module,
            origin="candidate:verified_deployment"))
    return kernels


def baseline_implementation(workload):
    import inspect
    from evograd.evaluation.tier3 import compile_baseline, runner
    from evograd.verification.integrations.base import sha_file
    paths = [__file__, compile_baseline.__file__, runner.__file__,
             inspect.getfile(type(workload))]
    return {"revision": BASELINE_REVISION,
            "sources": {str(path): sha_file(path) for path in paths}}


def compile_baseline_valid(row):
    return bool(row.get("ok")) and (row.get("correctness") or {}).get("status") == "passed"


def whole_model_compile(workload):
    """The workload's whole-model ``torch.compile`` provider, if it offers one."""
    import sys

    make = getattr(sys.modules.get(type(workload).__module__), "whole_model_compile_kernels", None)
    if make is None:
        raise ValueError(f"{type(workload).__name__}: whole-model compile is unsupported")
    return make(workload.site_registry)


def provider_kernels(run: Run, spec):
    """``"eager"``, ``"torch_compile"`` (whole model) or a tuple of Level-2 case ids."""
    from evograd.evaluation.tier3.patch import KernelSet

    if spec == REFERENCE:
        return KernelSet(registry=run.workload.site_registry)
    if spec == "torch_compile":
        return whole_model_compile(run.workload)
    return deployment_kernels(run, list(spec))


def _tier3_child(model_name: str, candidates: dict, corpus_path: str, name: str, spec,
                 options: dict, device: str) -> dict[str, Any]:
    """One Tier 3 provider, in a process of its own (spawned by :func:`run_tier3`)."""
    import tempfile

    from evograd.evaluation.model_cache import checkpoint_model_cache
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import configure

    # Spawned processes do not inherit the parent's PyTorch runtime settings.
    # Apply identical numerics and CPU threading before timing any provider.
    configure()
    if spec in ("eager", "torch_compile"):
        candidates = {}
    with checkpoint_model_cache(model_name), tempfile.TemporaryDirectory() as unused:
        run = open_run(load(model_name), candidates, corpus_path, unused, device=device)
        return _measure_tier3_provider(run, name, spec, options)


def _measure_tier3_provider(run, name, spec, options):
    from evograd.evaluation.tier3.runner import measure_one
    from evograd.evaluation.tier3.compile_baseline import check_correctness, require_compiled_graphs

    correctness = None
    try:
        if spec == "torch_compile":
            correctness = check_correctness(run.model, run.workload)
            # Correctness graphs must not count as evidence for the timed model.
            import torch._dynamo
            from torch._dynamo.utils import counters
            torch._dynamo.reset()
            counters.clear()
        timing_mode()
        row = measure_one(run.workload, name, provider_kernels(run, spec), verify=False,
                          device=run.device, **options)
    except Exception as exc:
        row = {"ok": False, "provider": name, "failed_at": "baseline" if spec == "torch_compile" else "build",
               "error": f"{type(exc).__name__}: {exc}"}
    if spec == "torch_compile":
        row["correctness"] = correctness or {"status": "error", "error": row.get("error", "not checked")}
        require_compiled_graphs(row)
    return row


def run_tier3(run: Run, providers: Mapping[str, Any], *,
              options: Mapping[str, Any] = TIER3_OPTIONS, isolate: bool = True) -> dict[str, Any]:
    """Tier 3 of these providers (``{name: spec}``, see :func:`provider_kernels`)
    on the model's pinned workload; the report carries each provider's interval
    against eager.

    With ``isolate`` each provider runs in a freshly spawned process, in a
    seeded random order: one provider's leftovers (a compiled graph, an
    optimizer state) never reach another's peak memory or step time, and a
    crash costs only its own row.
    """
    from evograd.evaluation.tier3.runner import assemble_report, provider_order

    options = dict(options)
    order = provider_order(providers, seed=int(options.get("seed", 0)))
    results: dict[str, Any] = {}
    for name in order:
        spec = providers[name]
        if not isolate:
            results[name] = _measure_tier3_provider(run, name, spec, options)
            continue
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _release_device_memory()
        try:
            with ProcessPoolExecutor(max_workers=1,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                results[name] = pool.submit(
                    _tier3_child, run.model.name, dict(run.paths), str(run.corpus_path), name,
                    spec, options, run.device).result()
        except Exception as exc:  # the child died: its row fails, the run goes on
            results[name] = {"ok": False, "provider": name, "failed_at": "process",
                             "error": f"{type(exc).__name__}: {exc}"}
    return assemble_report(run.workload, results, order, verify=False,
                           isolation=("one spawned process per provider" if isolate
                                      else "in-process"),
                           **{k: options[k] for k in ("warmup", "steps", "blocks", "loss_steps",
                                                      "learning_rate", "seed")})


def merge_baselines(run, baselines, measured, options):
    from evograd.evaluation.tier3.runner import assemble_report
    raw = {**baselines["providers"], **measured["providers"]}
    return assemble_report(run.workload, raw, list(raw), verify=False,
                           isolation=measured.get("isolation", "shared model baselines"), **dict(options))


def _tier3_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    """One provider's training-step result against eager."""
    providers = report.get("providers") or {}
    entry, base = providers.get(name) or {}, providers.get(REFERENCE) or {}
    if not entry.get("ok"):
        return {"ok": False, "error": entry.get("error", "not measured"),
                "failed_at": entry.get("failed_at"),
                **{field: entry[field] for field in ("correctness", "dynamo", "compile_settings")
                   if field in entry}}
    row = {"ok": True, "step_ms": entry["step_ms"], "peak_memory_bytes": entry.get("peak_memory_bytes"),
           "cpu_bound_fraction": entry.get("cpu_bound_fraction")}
    for field in ("correctness", "dynamo", "compile_settings"):
        if field in entry:
            row[field] = entry[field]
    intervals = report.get("speedup_intervals") or {}
    against = ((intervals.get("vs_reference") or {}).get(name)
               if intervals.get("reference") == REFERENCE else None)
    if against is not None:
        row["speedup"] = against["step_ms_ratio"]
        row["ci95"] = against["ci95"]
    if base.get("ok") and base.get("peak_memory_bytes") and entry.get("peak_memory_bytes"):
        row["peak_memory_ratio"] = entry["peak_memory_bytes"] / base["peak_memory_bytes"]
    return row


def _positive(case: Mapping[str, Any]) -> bool:
    """Accepted and faster than eager with the single-site 95% interval above one."""
    timed = case.get("tier3_single_site") or {}
    lower = (timed.get("ci95") or {}).get("low")
    return (bool(case.get("accepted")) and bool(timed.get("ok"))
            and isinstance(lower, (int, float)) and math.isfinite(lower) and lower > 1)


# ── stages ───────────────────────────────────────────────────────────────────


def _cached(path: Path, key: Mapping[str, Any], compute: Callable[[], Any], deadline) -> Any:
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("stage_key") == _jsonable(key):
            return data["result"]
    _check_deadline(deadline, path.stem)
    result = compute()
    _write(path, {"stage_key": _jsonable(key), "result": result})
    return result


def _environment() -> dict[str, Any]:
    import torch

    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def evaluate(model, candidates: Mapping[str, str], corpus_path: str | Path, out: str | Path, *,
             state: str | Path | None = None, device: str = "cuda",
             compile_backend: str = "inductor", log: Callable = print,
             deadline_min: float | None = None,
             tier2_options: Mapping[str, Any] = TIER2_OPTIONS,
             tier3_options: Mapping[str, Any] = TIER3_OPTIONS,
             time_tier2: Callable | None = None, isolate_tier3: bool = True,
             rules: bool = True, check_imports: bool = True) -> dict[str, Any]:
    """The whole protocol for one model; writes ``DIR/evaluation.json``.

    ``time_tier2(op, module, inputs)`` replaces :func:`time_case` (a CPU test
    has no CUDA events to time with); ``isolate_tier3=False`` runs the Tier 3
    providers in this process (a test model cannot be loaded by name);
    ``rules=False`` skips the static rules (a toy pair is not an artifact);
    ``check_imports=False`` skips importing each submission in a process of its own first.
    """
    from evograd.benchmark import get_task

    from evograd.verification.integrations.base import configure, observed_cases

    time_tier2 = time_tier2 or (lambda op, module, inputs: time_case(
        op, module, inputs, options=tier2_options))
    from evograd.verification.reuse import settings as reuse_settings, check_report_mode
    check_report_mode(Path(out) / "evaluation.json")
    verification_settings = configure()
    observed = {c.case_id for c in observed_cases(model)}
    unloadable = unimportable({cid: path for cid, path in candidates.items() if cid in observed}
                              ) if check_imports else {}
    candidates = {cid: path for cid, path in candidates.items() if cid not in unloadable}
    run = open_run(model, candidates, corpus_path, out, state=state, device=device,
                   compile_backend=compile_backend, log=log, deadline_min=deadline_min)
    # A submission's top-level code must not change the frozen verification settings.
    verification_settings = configure()
    result: dict[str, Any] = {
        "protocol": PROTOCOL, "model": model.name, "workload": dict(model.workload_config),
        "environment": _environment(), "reference": REFERENCE,
        "settings": {"verification": verification_settings, "calibration_reuse": reuse_settings(), "tier2": dict(tier2_options),
                     "tier2_inputs": dict(TIER2_INPUTS),
                     "tier3": dict(tier3_options), "combination_rule": COMBINATION_RULE},
        "cases": {c.case_id: {"task": c.task, "level": c.level, "family": get_task(c.task).family,
                              "frequency": c.frequency, "recorded": c.calls_recorded,
                              "submitted": c.case_id in candidates}
                  for c in observed_cases(model)},
    }
    import os
    if os.environ.get("EVOGRAD_EVALUATION_MODEL_CACHE"):
        result["settings"]["model_loading"] = "job-local pristine CPU snapshot; independent copies per build"
    for case_id, failure in unloadable.items():
        result["cases"][case_id].update({**failure, "submitted": True, "accepted": False,
                                         "executable": False})
        log(case_id, "import failed", failure["import_error"])
    try:
        # 0. Static rules: a violating submission is neither verified nor timed.
        for case_id in sorted(run.modules) if rules else ():
            violation = check_rules(model.reference_op(run.cases[case_id].task),
                                    run.modules[case_id], run.paths[case_id])
            if violation is None:
                continue
            result["cases"][case_id].update({
                "program": run.paths.pop(case_id), "program_sha256": run.programs.pop(case_id),
                "violation": violation, "accepted": False, "executable": False})
            del run.modules[case_id]
            log(case_id, "violation", violation)

        # Tier 2's recorded calls, checked before verification spends any time.
        tier2_calls = {cid: tier2_payloads(run.corpus, model.reference_op(run.cases[cid].task),
                                           run.cases[cid]) for cid in sorted(run.modules)}

        # 1. Verification.
        verdict = verify_cases(run)
        result["corpus"] = verdict["corpus"]
        for case_id, entry in verdict["candidates"].items():
            result["cases"][case_id].update(entry)

        # 2. Tier 2, every executable candidate.
        result["settings"]["timing"] = timing_mode()
        for case_id in sorted(run.modules):
            case, module = run.cases[case_id], run.modules[case_id]
            op = model.reference_op(case.task)
            binding = tier2_binding(run.corpus, tier2_calls[case_id])
            key = {"protocol": PROTOCOL, "case": case_id,
                   "program_sha256": run.programs[case_id], "inputs": binding,
                   "options": dict(tier2_options), "environment": result["environment"]}
            inputs = [(case.workload, partial(run.corpus.load, record, device=device))
                      for record in tier2_calls[case_id]]

            def tier2(op=op, module=module, inputs=inputs, binding=binding):
                try:
                    check_executable(op, module, inputs[0][1]())
                except NotExecutable as exc:
                    return {"executable": False, "error": str(exc), "tier2_inputs": binding}
                return {"executable": True, "tier2_inputs": binding,
                        "tier2": time_tier2(op, module, inputs)}

            timed = _cached(run.out / "tier2" / f"{case_id.replace('/', '__')}.json", key, tier2,
                            run.deadline)
            result["cases"][case_id].update(timed)
            log(case_id, "tier2", {k: v.get("speedup", v.get("error")) for k, v in
                                   (timed.get("tier2") or {}).items()} or timed.get("error"))

        # 3. Tier 3, each executable Level-2 candidate alone.
        level2 = [cid for cid in sorted(run.modules)
                  if run.cases[cid].level == 2 and result["cases"][cid].get("executable")]
        baseline_key = {"implementation": baseline_implementation(run.workload), "model": model.name,
                        "workload": model.workload_config, "corpus": run.corpus.binding(),
                        "options": dict(tier3_options), "environment": result["environment"],
                        "isolated": isolate_tier3}
        baseline_report = _cached(run.out / "tier3" / "baselines.json", baseline_key,
            lambda: run_tier3(run, {REFERENCE: REFERENCE, "torch_compile": "torch_compile"},
                             options=tier3_options, isolate=isolate_tier3), run.deadline)
        tier3: dict[str, Any] = {"baselines": {
            name: _tier3_row(baseline_report, name) for name in ENGINE_BASELINES}}
        result["tier3"] = tier3
        if level2:
            key = {"baseline": baseline_key, "programs": {cid: run.programs[cid] for cid in level2},
                   "options": dict(tier3_options), "environment": result["environment"],
                   "isolated": isolate_tier3}

            def single_site():
                providers = {cid: (cid,) for cid in level2}
                measured = run_tier3(run, providers, options=tier3_options, isolate=isolate_tier3)
                report = merge_baselines(run, baseline_report, measured, tier3_options)
                return {"report": report, "rows": {cid: _tier3_row(report, cid) for cid in level2}}

            single = _cached(run.out / "tier3" / "single_site.json", key, single_site, run.deadline)
            tier3["single_site"] = single["rows"]
            for cid, row in single["rows"].items():
                result["cases"][cid]["tier3_single_site"] = row
            log("tier3 single site", {cid: row.get("speedup", row.get("error"))
                                      for cid, row in single["rows"].items()})

        # 4. The combination: its gates, then its training step.
        chosen = [cid for cid in level2 if _positive(result["cases"][cid])]
        configure()
        combination = verify_combination(run, chosen)
        result["settings"]["timing"] = timing_mode()
        if chosen:
            key = {"baseline": baseline_key, "programs": {cid: run.programs[cid] for cid in chosen},
                   "options": dict(tier3_options), "environment": result["environment"],
                   "isolated": isolate_tier3}

            def combined():
                measured = run_tier3(run, {"combination": tuple(chosen)},
                                    options=tier3_options, isolate=isolate_tier3)
                report = merge_baselines(run, baseline_report, measured, tier3_options)
                rows = {name: _tier3_row(report, name) for name in ("combination", "torch_compile")}
                return {"report": report, "rows": rows}

            timed = _cached(run.out / "tier3" / "combination.json", key, combined, run.deadline)
            combination["tier3"] = timed["rows"]["combination"]
            if "torch_compile" in timed["rows"]:
                compiled, own = timed["rows"]["torch_compile"], timed["rows"]["combination"]
                combination["torch_compile_tier3"] = compiled
                if compile_baseline_valid(compiled) and own.get("ok"):
                    combination["speedup_vs_torch_compile"] = compiled["step_ms"] / own["step_ms"]
        tier3["combination"] = combination
        result["tier3"] = tier3
        log("combination", {k: v for k, v in combination.items() if k != "tier3"})
    except Suspended as stop:
        result["suspended"] = str(stop)
        log("suspended:", stop, "-- rerun the same command to continue")
        _write(run.out / "evaluation.json", result)
        return result
    result["summary"] = summarize([result])
    _write(run.out / "evaluation.json", result)
    return result


# ── summary ──────────────────────────────────────────────────────────────────


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _level_summary(rows: list[dict[str, Any]], level: int) -> dict[str, Any]:
    from evograd.benchmark.core.report import TaskResult, _pool_by_family

    rows = [r for r in rows if r["level"] == level]
    funnel = {"configurations": len(rows),
              "submitted": sum(bool(r.get("submitted")) for r in rows),
              "violation": sum("violation" in r for r in rows),
              "executable": sum(bool(r.get("executable")) for r in rows),
              "gate1": sum(bool(r.get("executable")) and r.get("gate1_payloads") == "passed"
                           for r in rows),
              "accepted": sum(bool(r.get("executable")) and bool(r.get("accepted")) for r in rows)}
    speedups: dict[str, dict[str, Any]] = {}
    for baseline in ENGINE_BASELINES:
        by_task: dict[str, list[float]] = {}
        families: dict[str, str] = {}
        totals: dict[str, int] = {}
        for r in rows:
            families[r["task"]] = r["family"]
            totals[r["task"]] = totals.get(r["task"], 0) + 1
            value = ((r.get("tier2") or {}).get(baseline) or {}).get("speedup")
            if r.get("executable") and r.get("accepted") and _finite(value):
                by_task.setdefault(r["task"], []).append(value)
        tasks = [TaskResult(op=task, level=level, family=families[task], baseline=baseline,
                            speedups=tuple(by_task.get(task, ())), cases_total=totals[task],
                            cases_ok=len(by_task.get(task, ()))) for task in sorted(totals)]
        covered = sum(t.cases_ok for t in tasks)
        speedups[baseline] = {"geomean": _pool_by_family(tasks) if covered else None,
                              "configurations_covered": covered}
    return {"funnel": funnel, "speedup": speedups,
            "coverage": funnel["accepted"] / len(rows) if rows else None}


def summarize(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per level: speedup over the accepted configurations (geometric mean over
    configurations, then tasks, then families), coverage and the verification
    funnel. Level 2 also retains each site's model-scope measurement without
    pooling it; Level 3 reports each model's combination, speedup and peak memory."""
    from evograd.benchmark.core.report import geomean

    for ev in evaluations:
        if ev.get("protocol") != PROTOCOL:
            raise ValueError(
                f"{ev.get('model', 'evaluation')}: protocol {ev.get('protocol')!r} is not "
                f"{PROTOCOL!r}; rerun evaluate to obtain formal Tier 2 results. Earlier "
                "protocols timed Tier 1 or generated inputs; they are not pooled with recorded calls.")
    rows = [dict(row) for ev in evaluations for row in ev["cases"].values()]
    models, single_sites = {}, {}
    for ev in evaluations:
        combination = (ev.get("tier3") or {}).get("combination") or {}
        timed = combination.get("tier3") or {}
        selected = set(combination.get("cases", ()))
        single_sites[ev["model"]] = {
            cid: {"task": row["task"], "submitted": bool(row.get("submitted")),
                  "executable": bool(row.get("executable")),
                  "accepted": bool(row.get("accepted")),
                  **{gate: row.get(gate) for gate in
                     ("gate1_payloads", "gate2_operator", "gate2_model", "gate3")},
                  "positive": _positive(row), "selected_for_combination": cid in selected,
                  "tier3": dict(row.get("tier3_single_site") or {})}
            for cid, row in sorted(ev["cases"].items()) if row["level"] == 2}
        models[ev["model"]] = {
            "baselines": (ev.get("tier3") or {}).get("baselines", {}),
            "cases": combination.get("cases", []),
            "accepted": bool(combination.get("accepted")),
            "gate2_model": combination.get("gate2_model"), "gate3": combination.get("gate3"),
            "speedup": timed.get("speedup"), "ci95": timed.get("ci95"),
            "speedup_vs_torch_compile": combination.get("speedup_vs_torch_compile"),
            "peak_memory_ratio": timed.get("peak_memory_ratio"),
            "cpu_bound_fraction": timed.get("cpu_bound_fraction")}
    counted = [m for m in models.values() if m["accepted"] and _finite(m["speedup"])]
    memory = [m["peak_memory_ratio"] for m in counted if _finite(m["peak_memory_ratio"])]
    return {
        "models": sorted(models),
        "level1": _level_summary(rows, 1),
        "level2": {**_level_summary(rows, 2), "tier3_single_site": single_sites},
        "level3": {"models": models,
                   "speedup": {"geomean": geomean([m["speedup"] for m in counted]) if counted else None,
                               "models_covered": len(counted)},
                   "coverage": len(counted) / len(models) if models else None,
                   "peak_memory_ratio_geomean": geomean(memory) if memory else None},
    }


def main(argv: Sequence[str] | None = None) -> int:
    """``evograd evaluate MODEL --corpus ... --out DIR --candidate CASE_ID=PROGRAM.py ...``.

    Exits 0 when the protocol finished and 3 when ``--deadline-min`` stopped
    it (rerun the same command to continue)."""
    import argparse
    import time

    from evograd.verification.integrations import PACKAGES, load

    parser = argparse.ArgumentParser(prog="evograd evaluate")
    parser.add_argument("model", help=f"one of {sorted(PACKAGES)}")
    parser.add_argument("--corpus", required=True, help="the model's real-payload corpus manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate", action="append", required=True, metavar="CASE_ID=PROGRAM.py")
    parser.add_argument("--state", default=None,
                        help="candidate-independent results to reuse: Gate 3 control runs and "
                             "Gate 2 calibrations (default DIR/controls)")
    parser.add_argument("--deadline-min", type=float, default=None,
                        help="stop after this many minutes at the next stage boundary or Gate 3 "
                             "checkpoint; exit 3; rerun the same command to continue")
    parser.add_argument("--tier3-blocks", type=int, default=TIER3_OPTIONS["blocks"],
                        help="timed blocks per Tier 3 provider; more narrow the interval the "
                             "combination rule reads")
    from evograd.verification.reuse import add_arguments, options, calibration_reuse
    add_arguments(parser)
    args = parser.parse_args(argv)
    model = load(args.model)

    def log(*parts):
        print(time.strftime("%H:%M:%S"), *parts, flush=True)

    with calibration_reuse(options(args), log=log):
        result = evaluate(model, dict(item.split("=", 1) for item in args.candidate), args.corpus,
                          args.out, state=args.state, log=log, deadline_min=args.deadline_min,
                          tier3_options={**TIER3_OPTIONS, "blocks": args.tier3_blocks})
    if result.get("suspended"):
        print(json.dumps({"suspended": result["suspended"]}))
        return 3
    print(json.dumps(result["summary"], indent=1))
    return 0


def summarize_main(argv: Sequence[str] | None = None) -> int:
    """``evograd summarize DIR/evaluation.json ... [--out FILE]``."""
    import argparse

    parser = argparse.ArgumentParser(prog="evograd summarize")
    parser.add_argument("evaluations", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    summary = summarize([json.loads(Path(p).read_text()) for p in args.evaluations])
    text = json.dumps(summary, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


__all__ = ["evaluate", "summarize", "check_executable", "check_baseline", "time_case",
           "tier2_payloads", "tier2_binding", "NotExecutable", "COMBINATION_RULE", "PROTOCOL",
           "TIER2_INPUTS", "main", "summarize_main"]


if __name__ == "__main__":
    import sys

    sys.exit(main())
