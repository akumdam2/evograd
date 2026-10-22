"""The benchmark's evaluation of one model's candidates: verification, then timing.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 evograd evaluate <model> \\
        --corpus CORPUS/manifest.json --out DIR --candidate CASE_ID=PROGRAM.py [...]
    evograd summarize DIR_A/evaluation.json DIR_B/evaluation.json [--out FILE]

Every submission follows the same steps, whoever generated it:

1. **Verification** (:func:`evograd.verification.integrations.gates.verify_cases`): Gate 1 and operator Gate 2
   for every candidate, and the single-site model Gate 2 for every Level-2
   candidate. A candidate passing every gate that applies is *accepted*.
2. **Tier 1** for every *executable* candidate, accepted or not: the
   forward--backward pair at the case's configuration, against eager PyTorch
   and ``torch.compile`` through autograd, and against any declared library
   pair called directly. A candidate is executable when one forward and
   backward through the timed route raises nothing, returns the declared
   number of outputs and gradients with the eager shapes and dtypes, and leaves
   its inputs unchanged; anything else is not timed.
3. **Tier 3, single site**: the training step with each executable Level-2
   candidate alone at its model site, against eager.
4. **Combination**: the accepted Level-2 candidates whose single-site speedup
   has a 95% interval above one (:data:`COMBINATION_RULE`), judged together by
   model Gate 2 and Gate 3, then timed at Tier 3 against eager and the
   whole-model ``torch.compile``.
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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

PROTOCOL = "evograd-evaluation-v1"
REFERENCE = "eager"
TIER1_OPTIONS = {"warmup": 10, "reps": 50, "blocks": 3, "seed": 0}
TIER3_OPTIONS = {"warmup": 3, "steps": 10, "blocks": 10, "loss_steps": 5,
                 "learning_rate": 1e-4, "seed": 0}
COMBINATION_RULE = ("every accepted Level-2 candidate whose single-site Tier 3 speedup over eager "
                    "has a 95% bootstrap interval with its lower bound above 1")
#: Engine-mediated baselines: the candidate is timed through autograd against them.
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


# ── Tier 1 ───────────────────────────────────────────────────────────────────


def _shape_dtype(values) -> list[tuple[tuple[int, ...], str] | None]:
    import torch

    return [(tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else None for v in values]


def check_executable(op, module, workload, *, device: str = "cuda") -> None:
    """One forward and backward through the route Tier 1 times; raises
    :class:`NotExecutable` with the reason."""
    from evograd.evaluation.common.provider import (
        assert_tensors_unchanged,
        candidate_autograd_provider,
        clone_values,
        pytorch_autograd_provider,
        snapshot_tensors,
    )
    from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
    from evograd.pipelines.shared.artifact import deployment_entry, validate_artifact

    try:
        if deployment_entry(module) is not None:
            validate_artifact(op, module)
        values = make_case_inputs(op, workload, device=device)
        reference_values = clone_values(values)
        snapshot = snapshot_tensors(values, may_overwrite=tuple(op.backward_may_overwrite or ()))
        candidate = candidate_autograd_provider(op, module)
        output, saved = candidate.forward(values)
        grads = candidate.backward(upstream_grad_values(op, values), saved, values)
        assert_tensors_unchanged(values, snapshot, provider="candidate")
        eager = pytorch_autograd_provider(op)
        ref_output, ref_saved = eager.forward(reference_values)
        ref_grads = eager.backward(upstream_grad_values(op, reference_values), ref_saved,
                                   reference_values)
    except NotExecutable:
        raise
    except Exception as exc:
        raise NotExecutable(f"{type(exc).__name__}: {exc}") from exc
    got = _shape_dtype(as_output_tuple(op, output)), _shape_dtype(grads)
    want = _shape_dtype(as_output_tuple(op, ref_output)), _shape_dtype(ref_grads)
    if got[0] != want[0]:
        raise NotExecutable(f"outputs {got[0]}, the declaration's are {want[0]}")
    if got[1] != want[1]:
        raise NotExecutable(f"gradients {got[1]}, the declaration's are {want[1]}")


def time_case(op, module, workload, *, device: str = "cuda",
              options: Mapping[str, Any] = TIER1_OPTIONS) -> dict[str, Any]:
    """Tier 1: the candidate against every baseline of its case, one fair run each."""
    from evograd.evaluation.common.provider import (
        candidate_autograd_provider,
        candidate_provider,
        declared_provider,
        pytorch_autograd_provider,
        torch_compile_provider,
    )
    from evograd.evaluation.tier1.fair import run_fair_benchmarks
    from evograd.opdecl.baselines import verify_performance_baseline

    baselines: dict[str, Callable[[], Any]] = {
        "eager": lambda: pytorch_autograd_provider(op),
        "torch_compile": lambda: torch_compile_provider(op, name="torch_compile", dynamic=False),
    }
    for name in sorted(op.performance_baselines):
        def declared(name=name):
            verify_performance_baseline(op, name, device=device)
            return declared_provider(op, name)

        baselines[name] = declared
    results: dict[str, Any] = {}
    for name, make_baseline in baselines.items():
        engine = name in ENGINE_BASELINES
        try:
            baseline = make_baseline()
            candidate = (candidate_autograd_provider(op, module) if engine
                         else candidate_provider(op, module))
            report = run_fair_benchmarks(op, candidate, baseline, workloads=(workload,),
                                         device=device, **dict(options))
        except Exception as exc:  # one baseline must not cost the others
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        aggregate = report["aggregate"]
        results[name] = {
            "route": "autograd" if engine else "pair",
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
    modules = {site_for_case(registry, case): module for case, module in level2}
    kernels = KernelSet(registry=registry)
    for site, call in deployment.sites.items():
        kernels = patch(kernels, site, call, source=KernelSource(
            site=site, op_name=registry.require(site).op, module=modules[site],
            origin="candidate:verified_deployment"))
    return kernels


def whole_model_compile(workload):
    """The workload's whole-model ``torch.compile`` provider, if it offers one."""
    import sys

    make = getattr(sys.modules.get(type(workload).__module__), "whole_model_compile_kernels", None)
    return None if make is None else make(workload.site_registry)


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

    from evograd.evaluation.tier3.runner import measure_one
    from evograd.verification.integrations import load

    with tempfile.TemporaryDirectory() as unused:
        run = open_run(load(model_name), candidates, corpus_path, unused, device=device)
        return measure_one(run.workload, name, provider_kernels(run, spec), verify=False,
                           device=device, **options)


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
    from evograd.evaluation.tier3.runner import assemble_report, measure_one, provider_order

    options = dict(options)
    order = provider_order(providers, seed=int(options.get("seed", 0)))
    results: dict[str, Any] = {}
    for name in order:
        spec = providers[name]
        if not isolate:
            results[name] = measure_one(run.workload, name, provider_kernels(run, spec),
                                        verify=False, device=run.device, **options)
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


def _tier3_row(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    """One provider's training-step result against eager."""
    providers = report.get("providers") or {}
    entry, base = providers.get(name) or {}, providers.get(REFERENCE) or {}
    if not entry.get("ok"):
        return {"ok": False, "error": entry.get("error", "not measured"),
                "failed_at": entry.get("failed_at")}
    row = {"ok": True, "step_ms": entry["step_ms"], "peak_memory_bytes": entry.get("peak_memory_bytes"),
           "cpu_bound_fraction": entry.get("cpu_bound_fraction")}
    intervals = report.get("speedup_intervals") or {}
    against = ((intervals.get("vs_reference") or {}).get(name)
               if intervals.get("reference") == REFERENCE else None)
    if against is not None:
        row["speedup"] = against["step_ms_ratio"]
        row["ci95"] = against["ci95"]
    if base.get("ok") and base.get("peak_memory_bytes") and entry.get("peak_memory_bytes"):
        row["peak_memory_ratio"] = entry["peak_memory_bytes"] / base["peak_memory_bytes"]
    return row


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
             tier1_options: Mapping[str, Any] = TIER1_OPTIONS,
             tier3_options: Mapping[str, Any] = TIER3_OPTIONS,
             time_tier1: Callable | None = None, isolate_tier3: bool = True) -> dict[str, Any]:
    """The whole protocol for one model; writes ``DIR/evaluation.json``.

    ``time_tier1(op, module, workload)`` replaces :func:`time_case` (a CPU test
    has no CUDA events to time with); ``isolate_tier3=False`` runs the Tier 3
    providers in this process (a test model cannot be loaded by name).
    """
    from evograd.benchmark import get_task

    from evograd.verification.integrations.base import configure, observed_cases

    time_tier1 = time_tier1 or (lambda op, module, workload: time_case(
        op, module, workload, device=device, options=tier1_options))
    verification_settings = configure()
    run = open_run(model, candidates, corpus_path, out, state=state, device=device,
                   compile_backend=compile_backend, log=log, deadline_min=deadline_min)
    result: dict[str, Any] = {
        "protocol": PROTOCOL, "model": model.name, "workload": dict(model.workload_config),
        "environment": _environment(), "reference": REFERENCE,
        "settings": {"verification": verification_settings, "tier1": dict(tier1_options),
                     "tier3": dict(tier3_options), "combination_rule": COMBINATION_RULE},
        "cases": {c.case_id: {"task": c.task, "level": c.level, "family": get_task(c.task).family,
                              "frequency": c.frequency, "submitted": c.case_id in candidates}
                  for c in observed_cases(model)},
    }
    try:
        # 1. Verification.
        verdict = verify_cases(run)
        result["corpus"] = verdict["corpus"]
        for case_id, entry in verdict["candidates"].items():
            result["cases"][case_id].update(entry)

        # 2. Tier 1, every executable candidate.
        result["settings"]["timing"] = timing_mode()
        for case_id in sorted(run.modules):
            case, module = run.cases[case_id], run.modules[case_id]
            op = model.reference_op(case.task)
            key = {"case": case_id, "program_sha256": run.programs[case_id],
                   "options": dict(tier1_options), "environment": result["environment"]}

            def tier1(op=op, module=module, case=case):
                try:
                    check_executable(op, module, case.workload, device=device)
                except NotExecutable as exc:
                    return {"executable": False, "error": str(exc)}
                return {"executable": True, "tier1": time_tier1(op, module, case.workload)}

            timed = _cached(run.out / "tier1" / f"{case_id.replace('/', '__')}.json", key, tier1,
                            run.deadline)
            result["cases"][case_id].update(timed)
            log(case_id, "tier1", {k: v.get("speedup", v.get("error")) for k, v in
                                   (timed.get("tier1") or {}).items()} or timed.get("error"))

        # 3. Tier 3, each executable Level-2 candidate alone.
        level2 = [cid for cid in sorted(run.modules)
                  if run.cases[cid].level == 2 and result["cases"][cid].get("executable")]
        tier3: dict[str, Any] = {}
        if level2:
            key = {"programs": {cid: run.programs[cid] for cid in level2},
                   "options": dict(tier3_options), "environment": result["environment"],
                   "isolated": isolate_tier3}

            def single_site():
                providers = {REFERENCE: REFERENCE, **{cid: (cid,) for cid in level2}}
                report = run_tier3(run, providers, options=tier3_options, isolate=isolate_tier3)
                return {"report": report, "rows": {cid: _tier3_row(report, cid) for cid in level2}}

            single = _cached(run.out / "tier3" / "single_site.json", key, single_site, run.deadline)
            tier3["single_site"] = single["rows"]
            for cid, row in single["rows"].items():
                result["cases"][cid]["tier3_single_site"] = row
            log("tier3 single site", {cid: row.get("speedup", row.get("error"))
                                      for cid, row in single["rows"].items()})

        # 4. The combination: its gates, then its training step.
        chosen = [cid for cid in level2 if result["cases"][cid].get("accepted")
                  and (tier3.get("single_site", {}).get(cid, {}).get("ci95") or {}).get("low", 0) > 1]
        configure()
        combination = verify_combination(run, chosen)
        result["settings"]["timing"] = timing_mode()
        if chosen:
            key = {"programs": {cid: run.programs[cid] for cid in chosen},
                   "options": dict(tier3_options), "environment": result["environment"],
                   "isolated": isolate_tier3}

            def combined():
                providers = {REFERENCE: REFERENCE, "combination": tuple(chosen)}
                if whole_model_compile(run.workload) is not None:
                    providers["torch_compile"] = "torch_compile"
                report = run_tier3(run, providers, options=tier3_options, isolate=isolate_tier3)
                rows = {name: _tier3_row(report, name) for name in providers if name != REFERENCE}
                return {"report": report, "rows": rows}

            timed = _cached(run.out / "tier3" / "combination.json", key, combined, run.deadline)
            combination["tier3"] = timed["rows"]["combination"]
            if "torch_compile" in timed["rows"]:
                compiled, own = timed["rows"]["torch_compile"], timed["rows"]["combination"]
                combination["torch_compile_tier3"] = compiled
                if compiled.get("ok") and own.get("ok"):
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
              "executable": sum(bool(r.get("executable")) for r in rows),
              "gate1": sum(bool(r.get("executable")) and r.get("gate1_generated") == "passed"
                           and r.get("gate1_payloads") == "passed" for r in rows),
              "accepted": sum(bool(r.get("executable")) and bool(r.get("accepted")) for r in rows)}
    speedups: dict[str, dict[str, Any]] = {}
    for baseline in ENGINE_BASELINES:
        by_task: dict[str, list[float]] = {}
        families: dict[str, str] = {}
        totals: dict[str, int] = {}
        for r in rows:
            families[r["task"]] = r["family"]
            totals[r["task"]] = totals.get(r["task"], 0) + 1
            value = ((r.get("tier1") or {}).get(baseline) or {}).get("speedup")
            if r.get("executable") and r.get("accepted") and _finite(value):
                by_task.setdefault(r["task"], []).append(value)
        tasks = [TaskResult(op=task, level=level, family=families[task], baseline=baseline,
                            speedups=tuple(by_task.get(task, ())), cases_total=totals[task],
                            cases_ok=len(by_task.get(task, ()))) for task in sorted(totals)]
        covered = sum(t.cases_ok for t in tasks)
        speedups[baseline] = {"geomean": _pool_by_family(tasks) if covered else None,
                              "configurations_covered": covered}
    covered = speedups[REFERENCE]["configurations_covered"]
    return {"funnel": funnel, "speedup": speedups,
            "coverage": covered / len(rows) if rows else None}


def summarize(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per level: speedup over the accepted configurations (geometric mean over
    configurations, then tasks, then families), coverage and the verification
    funnel; Level 3: each model's combination, its speedup and peak memory."""
    from evograd.benchmark.core.report import geomean

    rows = [dict(row) for ev in evaluations for row in ev["cases"].values()]
    models = {}
    for ev in evaluations:
        combination = (ev.get("tier3") or {}).get("combination") or {}
        timed = combination.get("tier3") or {}
        models[ev["model"]] = {
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
        "level2": _level_summary(rows, 2),
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
    args = parser.parse_args(argv)
    model = load(args.model)

    def log(*parts):
        print(time.strftime("%H:%M:%S"), *parts, flush=True)

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


__all__ = ["evaluate", "summarize", "check_executable", "time_case", "NotExecutable",
           "COMBINATION_RULE", "PROTOCOL", "main", "summarize_main"]


if __name__ == "__main__":
    import sys

    sys.exit(main())
