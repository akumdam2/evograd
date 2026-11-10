"""The site adapters give MegaFold's own step, bit for bit: the old capture check, by group.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.megafold.site_check \\
        --out REPORT.json [--order 101]

The recorder changes no arithmetic, so the capture loop no longer tests an adapter.
This does: it installs every site's deployment adapter with a kernel computing the
site's model reference, and asks whether the step still is the plain step (loss,
distogram logits and every parameter gradient, by content hash). Nested sites cannot
be installed together, so they run in non-overlapping groups, one model per group
(``SiteRegistry.groups``). A pass also proves each adapter consumes RNG as the module
it replaces does, since the step's dropout and noise would move otherwise.

The kernels are the references themselves, called directly: no ``autograd.Function``,
so the gradients are autograd's and only the adapters and references are on trial.
A group that differs runs again one site at a time, so the report names the site.
Exits 0 when every group is identical, 2 otherwise.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import MODEL


def compare(plain: dict[str, Any], patched: dict[str, Any], primary: str) -> dict[str, Any]:
    """Which of the step's results differ between two records of it."""
    key = f"{primary}_digest"
    differing = sorted(name for name in plain["grads"] if plain["grads"][name] != patched["grads"].get(name))
    extra = sorted(set(patched["grads"]) - set(plain["grads"]))
    loss_same, primary_same = (plain["loss_digest"] == patched["loss_digest"], plain[key] == patched[key])
    return {"loss_uninstrumented": plain["loss"], "loss_patched": patched["loss"],
            "loss_bitwise": loss_same, f"{primary}_bitwise": primary_same,
            "grads_compared": len(plain["grads"]), "grads_differing": differing,
            "grads_unexpected": extra,
            "identical": loss_same and primary_same and not differing and not extra}


def reference_kernels(registry, sites, reference_op, resolve) -> Any:
    """A kernel set with each of ``sites`` computing its model reference."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    kernels = KernelSet(registry=registry)
    for site in sites:
        op = reference_op(registry.require(site).op)
        kernels = patch(kernels, site, resolve(op), source=KernelSource(site, op.name, None, "site_check"))
    return kernels


def check(order: int = 101, *, device: str = "cuda", log: Callable = print) -> dict[str, Any]:
    import torch

    from evograd.opdecl.oracle import resolve_runtime_forward

    from ..base import configure, objective_of
    from ..objectives import step_record, to_device

    settings = configure()
    workload = MODEL.load_workload()
    objective = objective_of(MODEL).bind(workload)
    batch = to_device(MODEL.train_batches(workload, (order,))[0][order], device)
    primary, registry = objective.primary, workload.site_registry

    def run(kernels, provider=None):
        net, provenance = workload.build_patched(kernels, provider=provider)
        net = net.train()
        started = time.time()
        loss, output = objective.capture_step(net, batch)
        record = step_record(net, loss, output, primary)
        del net, loss, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return record, provenance, time.time() - started

    from evograd.evaluation.tier3.patch import KernelSet

    plain, _provenance, seconds = run(KernelSet(registry=registry))
    log(f"plain step: loss {plain['loss']:.6f}, {seconds:.1f}s")

    def against_plain(sites) -> dict[str, Any]:
        kernels = reference_kernels(registry, sites, MODEL.reference_op, resolve_runtime_forward)
        record, provenance, seconds = run(kernels)
        row = {"sites": list(sites), "paths": provenance.to_dict()["counts"], "seconds": seconds,
               **compare(plain, record, primary)}
        log(f"{', '.join(sites) if len(sites) == 1 else f'{len(sites)} sites'}: "
            f"identical={row['identical']}, loss {row['loss_patched']:.6f}, "
            f"{len(row['grads_differing'])} gradients differ, {seconds:.1f}s")
        return row

    groups = []
    for sites in registry.groups():
        row = against_plain(sites)
        if not row["identical"] and len(sites) > 1:
            row["alone"] = [against_plain((site,)) for site in sites]
        groups.append(row)
    return {"order": order, "execution": settings, "groups": groups,
            "identical": all(row["identical"] for row in groups)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order", type=int, default=101)
    args = parser.parse_args(argv)
    report = check(args.order, log=lambda *p: print(time.strftime("%H:%M:%S"), *p, flush=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"identical": report["identical"], "groups": len(report["groups"])}))
    return 0 if report["identical"] else 2


if __name__ == "__main__":
    sys.exit(main())
