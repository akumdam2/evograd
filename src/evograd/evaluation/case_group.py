"""Case-major formal evaluation with shared T2 and model T3 baseline samples."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

from evograd.evaluation import protocol as P

GROUP = "case-major-shared-baselines/2"


def evaluate_group(plan, *, deadline, tier3_options=None, model=None, log=print,
                   time_tier2=None, time_provider=None, rules=True, check_imports=True, device="cuda",
                   compile_backend="inductor"):
    from evograd.verification.reuse import settings as reuse_settings, check_report_mode
    from evograd.benchmark import get_task
    from evograd.evaluation import group_timing
    from evograd.evaluation.payload_cache import case_payload_cache
    from evograd.evaluation.tier3.runner import assemble_report, provider_order
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import configure, observed_cases, sha_file
    from evograd.verification.integrations.gates import (
        Suspended, _jsonable, _write, open_run, verify_cases, verify_combination)

    model = model or load(plan["model"])
    t3 = {**P.TIER3_OPTIONS, **(tier3_options or {})}
    settings = configure()
    shared = Path(plan["state"]).parent / "shared"
    environment = P._environment()
    implementation = {"group": sha_file(__file__), "protocol": sha_file(P.__file__),
                      "timing": sha_file(group_timing.__file__)}
    reference = {"contract": model.contract, **model.source_identity()}
    runs, results = {}, {}
    observed = {case.case_id: case for case in observed_cases(model)}
    # Admit every agent before starting case-by-case verification/timing.
    for job in plan["jobs"]:
        check_report_mode(Path(job["out"]) / "evaluation.json")
    for job in plan["jobs"]:
        selection = job["selection"]
        agent = selection["agent"]
        paths = {cid: program["path"] for cid, program in selection["candidates"].items()}
        bad = P.unimportable({cid: path for cid, path in paths.items() if cid in observed}) if check_imports else {}
        paths = {cid: path for cid, path in paths.items() if cid not in bad}
        run = open_run(model, paths, selection["corpus"], job["out"], state=plan["state"], log=log,
                       device=device, compile_backend=compile_backend)
        run.deadline = deadline
        configure()
        runs[agent] = run
        result = {
            "protocol": P.PROTOCOL, "model": model.name, "agent": agent,
            "workload": dict(model.workload_config), "environment": environment,
            "reference": P.REFERENCE, "corpus": run.corpus.binding(),
            "settings": {"verification": settings, "calibration_reuse": reuse_settings(), "tier2": dict(P.TIER2_OPTIONS),
                         "tier2_inputs": dict(P.TIER2_INPUTS), "tier3": t3,
                         "combination_rule": P.COMBINATION_RULE, "group_execution": GROUP,
                         "weight_cache": plan.get("weight_cache", False),
                         "compile_backend": compile_backend,
                         "payload_loading": "one storage read per case; private device copies per execution",
                         "baseline_scope": "T2: shared per case; T3: shared per model"},
            "cases": {cid: {"task": case.task, "level": case.level,
                            "family": get_task(case.task).family, "frequency": case.frequency,
                            "submitted": cid in paths or cid in bad}
                      for cid, case in observed.items()},
        }
        for cid, failure in bad.items():
            result["cases"][cid].update({**failure, "submitted": True, "accepted": False, "executable": False})
        for cid in sorted(list(run.modules)) if rules else ():
            violation = P.check_rules(model.reference_op(run.cases[cid].task), run.modules[cid], run.paths[cid])
            if violation is not None:
                result["cases"][cid].update({"violation": violation, "accepted": False,
                                             "executable": False, "program": run.paths[cid],
                                             "program_sha256": run.programs[cid]})
                del run.modules[cid], run.paths[cid], run.programs[cid]
        results[agent] = result

    def save(suspended=None):
        for agent, result in results.items():
            if suspended:
                result["suspended"] = suspended
            else:
                result.pop("suspended", None)
                result["summary"] = P.summarize([result])
            _write(runs[agent].out / "evaluation.json", result)

    def key(**extra):
        return {"group": GROUP, "implementation": implementation, "model": model.name,
                "workload": _jsonable(model.workload_config), "environment": environment,
                "reference": reference, "compile_backend": compile_backend,
                "baseline_implementation": P.baseline_implementation(baseline_run.workload), **extra}

    def cache(path, binding, compute):
        return P._cached(path, binding, compute, deadline)

    baseline_run = next(iter(runs.values()))
    corpus_binding = baseline_run.corpus.binding()
    if any(run.corpus.binding() != corpus_binding for run in runs.values()):
        raise ValueError("All agents must use the same corpus")

    def measure(run, name, spec):
        if time_provider is not None:
            return time_provider(run, name, spec, t3)
        report = P.run_tier3(run, {name: spec}, options=t3)
        return report["providers"][name]

    def model_timing(name, run, spec, programs):
        import hashlib

        path = shared / "tier3" / "providers" / f"{hashlib.sha256(name.encode()).hexdigest()}.json"
        row = cache(path, key(name=name, programs=programs, options=t3, corpus=corpus_binding),
                    lambda: measure(run, name, spec))
        log("shared tier3", name, row.get("step_ms", row.get("error")))
        return row

    def report(raw, order, path):
        assembled = assemble_report(baseline_run.workload, raw, order, verify=False,
                                    isolation="one spawned process per provider; shared model baselines", **t3)
        assembled["group_execution"] = GROUP
        _write(path, assembled)
        return assembled

    try:
        # Phase A: one case, every agent; controls/calibration are shared via state.
        for cid in sorted({cid for run in runs.values() for cid in run.modules}):
            participating = {agent: run for agent, run in runs.items() if cid in run.modules}
            first = next(iter(participating.values()))
            case = first.cases[cid]
            op = model.reference_op(case.task)
            records = P.tier2_payloads(first.corpus, op, case)
            binding = P.tier2_binding(first.corpus, records)
            with case_payload_cache([run.corpus for run in participating.values()]):
                log("case group", cid, list(participating))
                executable = {}
                for agent, run in participating.items():
                    configure()
                    single = replace(run, modules={cid: run.modules[cid]},
                                     paths={cid: run.paths[cid]}, programs={cid: run.programs[cid]})
                    verdict = verify_cases(single)
                    results[agent]["cases"][cid].update(verdict["candidates"][cid])
                    P.timing_mode()
                    try:
                        P.check_executable(op, run.modules[cid], run.corpus.load(records[0], device=run.device))
                    except P.NotExecutable as exc:
                        results[agent]["cases"][cid].update(executable=False, error=str(exc), tier2_inputs=binding)
                    else:
                        executable[agent] = run.modules[cid]
                        results[agent]["cases"][cid].update(executable=True, tier2_inputs=binding)
                if executable:
                    P.timing_mode()
                    inputs = [(case.workload, partial(first.corpus.load, record, device=first.device)) for record in records]
                    path = shared / "tier2" / f"{cid.replace('/', '__')}.json"
                    timed = cache(path, key(case=cid, inputs=binding, options=P.TIER2_OPTIONS,
                                            programs={agent: participating[agent].programs[cid] for agent in executable}),
                                  lambda: (time_tier2 or group_timing.time_case)(
                                      op, executable, inputs, options=P.TIER2_OPTIONS))
                    for agent, row in timed["rows"].items():
                        results[agent]["cases"][cid].update(tier2=row, shared_tier2_measurement=str(path))
                    log("shared tier2", cid, {agent: row.get("eager", {}).get("speedup")
                                             for agent, row in timed["rows"].items()})

        # Phase B: model-wide baseline measurements, once for the complete cohort.
        individual = {(agent, cid): run for agent, run in runs.items() for cid in sorted(run.modules)
                      if run.cases[cid].level == 2 and results[agent]["cases"][cid].get("executable")}
        base_raw = {}
        if runs:
            P.timing_mode()
            empty = replace(baseline_run, paths={}, programs={}, modules={})
            jobs = {"eager": (empty, "eager", {})}
            jobs["torch_compile"] = (empty, "torch_compile", {})
            for (agent, cid), run in individual.items():
                name = f"{agent}::{cid}"
                single = replace(run, paths={cid: run.paths[cid]}, programs={cid: run.programs[cid]},
                                 modules={cid: run.modules[cid]})
                jobs[name] = (single, (cid,), {cid: run.programs[cid]})
            order = provider_order(jobs, seed=t3["seed"])
            raw = {name: model_timing(name, *jobs[name]) for name in order}
            base_raw = {name: raw[name] for name in ("eager", "torch_compile") if name in raw}
            single_report = report(raw, order, shared / "tier3" / "single_sites.json")
            for agent, cid in individual:
                results[agent]["cases"][cid]["tier3_single_site"] = P._tier3_row(single_report, f"{agent}::{cid}")

        # Phase C: each agent selects and verifies its own combination.
        combinations, combination_jobs = {}, {}
        for agent, run in runs.items():
            chosen = [cid for cid in sorted(run.modules)
                      if run.cases[cid].level == 2 and P._positive(results[agent]["cases"][cid])]
            configure()
            combination = verify_combination(run, chosen)
            combinations[agent] = combination
            results[agent]["tier3"] = {"baselines": {
                name: P._tier3_row(single_report, name) for name in P.ENGINE_BASELINES}, "single_site": {
                cid: row["tier3_single_site"] for cid, row in results[agent]["cases"].items()
                if "tier3_single_site" in row}, "combination": combination}
            if chosen:
                subset = replace(run, paths={cid: run.paths[cid] for cid in chosen},
                                 programs={cid: run.programs[cid] for cid in chosen},
                                 modules={cid: run.modules[cid] for cid in chosen})
                combination_jobs[f"{agent}::combination"] = (subset, tuple(chosen), subset.programs)

        # Phase D: combinations share the same eager/whole-model compile samples.
        if combination_jobs:
            P.timing_mode()
            order = provider_order(combination_jobs, seed=t3["seed"])
            combined_raw = {name: model_timing(name, *combination_jobs[name]) for name in order}
            combined_report = report({**base_raw, **combined_raw}, list(base_raw) + order,
                                     shared / "tier3" / "combinations.json")
            for agent, combination in combinations.items():
                name = f"{agent}::combination"
                if name not in combined_raw:
                    continue
                own = P._tier3_row(combined_report, name)
                combination["tier3"] = own
                if "torch_compile" in base_raw:
                    compiled = P._tier3_row(combined_report, "torch_compile")
                    combination["torch_compile_tier3"] = compiled
                    if own.get("ok") and P.compile_baseline_valid(compiled):
                        combination["speedup_vs_torch_compile"] = compiled["step_ms"] / own["step_ms"]
        for result in results.values():
            result["settings"]["timing"] = P.timing_mode()
            result["shared_measurements"] = str(shared)
        save()
    except Suspended as exc:
        save(str(exc))
        log("suspended:", str(exc))
        return 3
    return 0
