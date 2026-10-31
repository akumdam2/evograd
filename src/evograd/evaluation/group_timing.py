"""Shared operator measurements across agents, with one baseline per block."""
from __future__ import annotations

import random


def measure_case(op, providers, values, *, warmup=10, reps=50, blocks=3, seed=0):
    from evograd.evaluation.tier1 import fair
    from evograd.opdecl.inputs import upstream_grad_values
    from evograd.verification.common.tensors import isolate

    cache = fair.L2Cache()
    allowed = tuple(op.backward_may_overwrite or ())
    errors, raw, order = {}, {name: [] for name in providers}, []

    def fresh():
        own = isolate(values)
        return own, fair.snapshot_tensors(own, may_overwrite=allowed)

    # Copies are made outside timing, one provider at a time to bound GPU memory.
    for name, provider in providers.items():
        try:
            own, snapshot = fresh()
            fair._warm_provider(provider, own, upstream_grad_values(op, own), warmup=warmup)
            fair.assert_tensors_unchanged(own, snapshot, provider=name)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
        finally:
            own = snapshot = None
    rng = random.Random(seed)
    for index in range(blocks):
        names = [name for name in providers if name not in errors]
        rng.shuffle(names)
        for name in names:
            order.append(name)
            try:
                own, snapshot = fresh()
                samples = fair._measure_provider_block(
                    providers[name], own, upstream_grad_values(op, own),
                    reps=reps, clear_l2=cache.clear)
                fair.assert_tensors_unchanged(own, snapshot, provider=name)
                raw[name].append({"pair_index": index, "samples_ms": samples})
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
            finally:
                own = snapshot = None
    summaries = {}
    for name, entries in raw.items():
        if name in errors:
            continue
        summaries[name] = {
            metric: fair._summary([sample for block in entries
                                   for sample in block["samples_ms"][region]])
            for metric, region in (("forward", "forward"), ("backward", "backward"),
                                   ("pair_full", "full"))}
    return {"providers": summaries, "errors": errors, "raw_blocks": raw, "block_order": order}


def time_case(op, modules, inputs, *, options):
    """Each candidate and baseline appears once per block of each recorded call."""
    from evograd.evaluation.common.provider import (
        candidate_autograd_provider, pytorch_autograd_provider,
        renamed_provider, torch_compile_provider)
    from evograd.evaluation import protocol
    from evograd.evaluation.tier1.fair import _block_bootstrap_speedup
    from evograd.opdecl.baselines import baseline_candidate_module
    from evograd.verification.payloads import CorpusError

    providers, errors = {}, {}
    names = {agent: f"candidate:{agent}" for agent in modules}
    for agent, module in modules.items():
        try:
            providers[names[agent]] = renamed_provider(candidate_autograd_provider(op, module), names[agent])
        except Exception as exc:
            errors[names[agent]] = f"{type(exc).__name__}: {exc}"
    providers["eager"] = pytorch_autograd_provider(op)
    providers["torch_compile"] = torch_compile_provider(op, name="torch_compile", dynamic=False)
    baselines = ["eager", "torch_compile", *sorted(op.performance_baselines)]
    for name in sorted(op.performance_baselines):
        try:
            provider = renamed_provider(
                candidate_autograd_provider(op, baseline_candidate_module(op, name)), name)
            workload, load = inputs[0]
            protocol.check_baseline(op, provider, load(), workload)
            providers[name] = provider
        except CorpusError:
            raise
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    calls = []
    for index, (_workload, load) in enumerate(inputs):
        values = load()
        report = measure_case(op, {name: provider for name, provider in providers.items() if name not in errors},
                              values, **{**options, "seed": options["seed"] + index})
        del values
        errors.update(report["errors"])
        calls.append(report)
    rows = {}
    for agent, candidate in names.items():
        rows[agent] = {}
        for baseline in baselines:
            if candidate in errors or baseline in errors:
                rows[agent][baseline] = {"error": errors.get(candidate) or errors[baseline]}
                continue
            def total(name, metric):
                return sum(call["providers"][name][metric]["median_ms"] for call in calls)

            candidate_ms, baseline_ms = total(candidate, "pair_full"), total(baseline, "pair_full")
            rows[agent][baseline] = {
                "route": "autograd", "candidate_ms": candidate_ms, "baseline_ms": baseline_ms,
                "speedup": baseline_ms / candidate_ms,
                "ci95": _block_bootstrap_speedup(calls, candidate, baseline, "full", seed=options["seed"] + 1),
                "speedup_forward": total(baseline, "forward") / total(candidate, "forward"),
                "speedup_backward": total(baseline, "backward") / total(candidate, "backward"),
            }
    return {"rows": rows, "calls": calls,
            "measurement": "one measurement per provider per block; shared baseline blocks across agents"}
