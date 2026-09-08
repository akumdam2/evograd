"""Is the provider a function, or does it remember?

Site preflight calls a kernel a handful of times on a declared grid. A model
calls it once per layer -- for a 28-layer decoder, 28 times, or 56. A provider
that is correct for the first eight calls and wrong afterwards passes the first
and poisons the second, and no amount of whole-model statistics reliably catches
it -- with eight of 28 layers still right, the model's aggregate error stays
inside the hardware's own noise. That fault is not a numerical question at all.
It is a question about whether the provider is a pure function of its arguments.

So: call it, with **identical cloned inputs and identical upstream gradients**,
more times than the model ever will, and compare every call against the first.
Any difference is a state dependence, whatever its magnitude, and it is reported
with the call index and the result name where it first appeared.

Two properties this gate needs and gets:

* two thresholds, because purity is not one question. The declared per-result
  tolerance is the hard gate: a repeat that lands outside it is wrong, full
  stop. But a state-dependent kernel need not be *wrong* -- the control that
  motivated this gate flips to a 2% error, and 2% sits comfortably inside a
  bfloat16 operator's declared 8% rtol. So a second question is asked as well:
  *does this kernel's answer depend on its history?*

  Not "does it repeat bitwise". On this GPU it often does not: attention's
  backward uses atomics, and two identical calls can differ in the last bits
  while the kernel is a perfectly pure function. What separates noise from
  state is not size but *shape*. Noise is stationary -- consecutive calls
  disagree by about as much late in a run as early. State is a step: one call
  differs sharply from the one before it, and every call after that keeps
  differing from call 1 by roughly that much.

  So the kernel's own noise is estimated from the **median** disagreement
  between consecutive calls, which the single transition of a stateful kernel
  cannot move, and drift is flagged when a call's distance from call 1 exceeds
  a fixed multiple of it. Both numbers are reported, so what was applied is
  visible rather than asserted.
* the purity calls happen in a **disposable process**. Constructing a second
  Python object does not reset module-level counters, CUDA graph caches, or an
  autotuner's memo; only a fresh interpreter does. The provider the model later
  runs is reconstructed there, so nothing this gate did can reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import torch

SCHEMA_VERSION = "evograd-t3-purity/2"

#: Calls to make when a workload does not say how many its site runs. A workload
#: that knows should pass twice its canonical invocation count, so a provider
#: that only misbehaves after "more calls than preflight makes" has nowhere to
#: hide; this fallback is merely more than any preflight performs.
DEFAULT_CALLS = 56

#: Where the first call is compared. Around the canonical counts and at the end,
#: because that is where a state-dependent provider changes behaviour, plus a
#: dense early stretch to catch one that changes immediately.
def checkpoints(total: int) -> list[int]:
    marks = {1, 2, 3, 7, 8, 9, 27, 28, 29, 55, 56, 57, 111, 112, total - 1, total}
    return sorted(i for i in marks if 1 <= i <= total)


CHILD_TIMEOUT = int(os.environ.get("EVOGRAD_PURITY_TIMEOUT", "1800"))

#: How far past its own median consecutive-call disagreement a call may sit
#: from call 1 before it is called state rather than noise. Predeclared, and
#: applied identically to every provider.
DETERMINISM_MARGIN = 16.0

#: A deterministic kernel has a median consecutive spread of exactly zero, and
#: a bound of zero would make one differing bit a state dependence. This is the
#: smallest disagreement that is treated as real, relative to the declared
#: tolerance the operator is judged by anyway.
DETERMINISM_FLOOR_FRACTION = 1e-3


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def _one_call(op, kernel, values: dict[str, Any]) -> dict[str, Any]:
    """One invocation on freshly cloned inputs, returning outputs and gradients.

    Every call gets its own clones, so a provider that mutates an input cannot
    make the next call differ for that reason alone -- mutation is detected
    separately and reported as itself.
    """
    from evograd.opdecl.inputs import as_output_tuple, upstream_grad_values

    active = {arg.name for arg in op.active_args()}
    by_grad = {arg.grad_name: arg.name for arg in op.active_args()}
    leaves: dict[str, torch.Tensor] = {}
    positional = []
    snapshots: dict[str, torch.Tensor] = {}
    for arg in op.args:
        value = _clone(values.get(arg.name, getattr(arg, "default", None)))
        if torch.is_tensor(value):
            snapshots[arg.name] = value.detach().clone()
            if arg.name in active:
                value = value.requires_grad_(True)
                leaves[arg.name] = value
        positional.append(value)

    result = kernel(*positional)
    outputs = as_output_tuple(op, result)
    upstream = upstream_grad_values(op, values)
    douts = upstream if isinstance(upstream, tuple) else (upstream,)
    grads = torch.autograd.grad(outputs, list(leaves.values()),
                                tuple(_clone(d) for d in douts))
    by_name = dict(zip(leaves.keys(), grads))

    collected = {
        f"out:{name}": tensor.detach().clone()
        for name, tensor in zip(op.output_names, outputs)
    }
    for grad_name in op.grad_names():
        collected[f"grad:{grad_name}"] = by_name[by_grad[grad_name]].detach().clone()
    mutated = [
        name for name, before in snapshots.items()
        if not torch.equal(before, positional[[a.name for a in op.args].index(name)].detach())
    ]
    return {
        "results": collected,
        "mutated_inputs": mutated,
        "finite": all(bool(torch.isfinite(t).all()) for t in collected.values()),
        "saved_structure": sorted(collected),
    }


def _median(values: list[float]) -> float:
    """The middle disagreement. One transition cannot move it; noise sets it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _spread(first: dict[str, torch.Tensor], later: dict[str, torch.Tensor]):
    """The largest disagreement between two calls, and where it was."""
    worst, where = 0.0, None
    for name, expected in first.items():
        got = later.get(name)
        if got is None or got.shape != expected.shape:
            return {"worst": float("inf"), "where": name}
        delta = float((got.float() - expected.float()).abs().max())
        if delta > worst:
            worst, where = delta, name
    return {"worst": worst, "where": where}


def is_production_default(site: str, kernel, registry) -> bool:
    """Is this the registry's own default rather than something to evaluate?

    The registry comes from the workload -- ``KernelSet.registry`` -- because
    two model families do not share a site namespace, and a site name existing
    somewhere is not a reason to treat this kernel as somebody's default.
    """
    entry = registry.get(site)
    return entry is not None and kernel is entry.default


def _tolerance(op, workload, name: str) -> tuple[float, float]:
    """The declaration's own tolerance for this result, whatever kind it is."""
    return op.tolerance_for(workload, name.split(":", 1)[1])


def check_site(site: str, op_name: str, kernel, *, registry, suite: str,
               device: str = "cuda", calls: int | None = None) -> dict[str, Any]:
    """Call one provider ``calls`` times on identical inputs and compare.

    ``suite`` names the benchmark suite carrying the shapes the model actually
    runs this site at -- purity is a question about the provider at *production*
    width, not on the declaration's small correctness grid.
    """
    from evograd.opdecl.inputs import make_case_inputs
    from evograd.ops import get_op

    op = get_op(op_name)
    if is_production_default(site, kernel, registry):
        # The registry's own production spelling is the model's code, not a
        # provider under evaluation, and it is not even callable with the
        # declared signature -- the adapter reaches it through the live module.
        # Saying so is more honest than inventing a subject to test.
        return {"site": site, "op": op_name, "ok": True, "calls": 0,
                "skipped": "the registry's production default is not a provider"}
    observed = op.benchmark_workloads(suite=suite)
    if not observed:
        return {"site": site, "op": op_name, "ok": False,
                "reason": f"the site declares no {suite!r} configuration to test on"}
    workload = observed[0]
    total = calls or DEFAULT_CALLS
    values = make_case_inputs(op, workload, device=device)

    try:
        first = _one_call(op, kernel, values)
    except Exception as exc:
        return {"site": site, "op": op_name, "ok": False, "calls": 1,
                "reason": f"the provider raised on its first call: "
                          f"{type(exc).__name__}: {exc}"}
    marks = checkpoints(total)
    compared: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None
    mutations: list[dict[str, Any]] = []
    if first["mutated_inputs"]:
        mutations.append({"call": 1, "inputs": first["mutated_inputs"]})

    previous = first
    consecutive: list[float] = []
    from_first: list[dict[str, Any]] = []
    for index in range(2, total + 1):
        try:
            record = _one_call(op, kernel, values)
        except Exception as exc:
            first_divergence = first_divergence or {
                "call": index, "result": "raised",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            break
        spread = _spread(first["results"], record["results"])
        consecutive.append(_spread(previous["results"], record["results"])["worst"])
        from_first.append({"call": index, **spread})
        if record["mutated_inputs"]:
            mutations.append({"call": index, "inputs": record["mutated_inputs"]})
        if record["saved_structure"] != first["saved_structure"]:
            first_divergence = first_divergence or {
                "call": index, "result": "saved_structure",
                "reason": "the set of returned results changed",
            }
        for name, expected in first["results"].items():
            got = record["results"].get(name)
            atol, rtol = _tolerance(op, workload, name)
            same = (
                got is not None
                and got.shape == expected.shape
                and bool(torch.allclose(got.float(), expected.float(),
                                        atol=atol, rtol=rtol))
            )
            if not same and first_divergence is None:
                delta = (float((got.float() - expected.float()).abs().max())
                         if got is not None and got.shape == expected.shape else None)
                first_divergence = {
                    "call": index, "result": name, "atol": atol, "rtol": rtol,
                    "max_abs_err": delta,
                }
        if index in marks:
            worst = max(
                (
                    float((record["results"][n].float()
                           - first["results"][n].float()).abs().max())
                    for n in first["results"]
                    if n in record["results"]
                    and record["results"][n].shape == first["results"][n].shape
                ),
                default=float("inf"),
            )
            compared.append({"call": index, "max_abs_err_vs_first": worst,
                             "finite": record["finite"]})
        del previous
        previous = record
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    noise = _median(consecutive)
    floor = DETERMINISM_FLOOR_FRACTION * max(
        _tolerance(op, workload, name)[0] for name in first["results"]
    )
    bound = max(noise, floor) * DETERMINISM_MARGIN
    drift = next(
        ({"call": s["call"], "result": s["where"], "max_abs_err": s["worst"],
          "median_consecutive_spread": noise, "floor": floor, "bound": bound,
          "regime": "deterministic" if noise == 0.0 else "noisy"}
         for s in from_first if s["worst"] > bound),
        None,
    )

    return {
        "site": site,
        "op": op_name,
        "workload": {"dims": dict(workload.dims), "dtype": workload.dtype},
        "calls": total,
        "checkpoints": marks,
        "compared": compared,
        "results_checked": first["saved_structure"],
        "mutated_inputs": mutations,
        "finite": first["finite"],
        "first_divergence": first_divergence,
        "median_consecutive_spread": noise,
        "max_spread_from_first": max((s["worst"] for s in from_first), default=0.0),
        "determinism_regime": "deterministic" if noise == 0.0 else "noisy",
        "determinism_floor": floor,
        "determinism_bound": bound,
        "first_drift": drift,
        "ok": bool(first_divergence is None and drift is None and not mutations
                   and first["finite"]),
        "gate": "the operator's declared per-result tolerances for correctness, "
                "and the provider's own opening spread for determinism",
    }


def check_kernels(kernels, *, suite: str, device: str = "cuda",
                  calls: dict[str, int] | None = None) -> dict[str, Any]:
    """Every patched site of one kernel set, in this process.

    The registry travels on the kernel set, so this needs to be told only which
    benchmark suite carries the observed shapes and how many calls each site is
    worth -- both facts about the workload, neither about purity.
    """
    registry = kernels.registry
    sites = []
    for site in kernels.patched:
        entry = registry.require(site)
        sites.append(check_site(
            site, entry.op, kernels.kernel_for(site), registry=registry,
            suite=suite, device=device, calls=(calls or {}).get(site),
        ))
    return {
        "schema_version": SCHEMA_VERSION,
        "sites": sites,
        "checked_sites": [s["site"] for s in sites if not s.get("skipped")],
        "skipped_sites": [s["site"] for s in sites if s.get("skipped")],
        "ok": all(s["ok"] for s in sites),
        "isolation": "in-process; use run_isolated for the disposable-process form",
    }


# ── the disposable process ───────────────────────────────────────────────────


def run_isolated(provider: str, *, module: str, device: str = "cuda",
                 sites: tuple[str, ...] | None = None,
                 fault: dict[str, Any] | None = None,
                 workload_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the purity gate in a child, so nothing it touches can reach the model.

    A second Python object is not a reset: module-level counters, lazily built
    caches and autotuner memos all survive it. A new interpreter is the only
    reset this repository can rely on, and the provider the model validation
    later uses is constructed there rather than here.

    ``module`` is the workload's own ``python -m`` entry point. Rebuilding a
    provider from a name means knowing which registry and which adapters that
    name refers to, and only the workload does; this half owns the isolation,
    the argument protocol and the timeout.
    """
    import tempfile

    handle, path = tempfile.mkstemp(prefix="evograd_purity_", suffix=".json")
    os.close(handle)
    argv = ["--provider", provider, "--device", device, "--result-json", path]
    if sites:
        argv += ["--sites", ",".join(sites)]
    if fault:
        argv += ["--fault", json.dumps(fault)]
    if workload_config:
        argv += ["--workload", json.dumps(workload_config)]
    try:
        process = subprocess.run(
            [sys.executable, "-m", module, *argv],
            capture_output=True, text=True, timeout=CHILD_TIMEOUT,
        )
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {"schema_version": SCHEMA_VERSION, "ok": False,
                    "reason": f"purity child exited rc={process.returncode} "
                              "without a result",
                    "stderr_tail": process.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"schema_version": SCHEMA_VERSION, "ok": False,
                "reason": f"purity child exceeded {CHILD_TIMEOUT}s"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


#: Origins the child process knows how to rebuild from a name alone.
REBUILDABLE = {"structural_identity": "structural", "bound_pair_identity": "bound"}


def spec_for(kernels) -> dict[str, Any] | None:
    """Can a fresh interpreter rebuild this provider? If so, how.

    ``None`` means it cannot, and the caller must say so rather than quietly
    running the gate in the process the model will also use.
    """
    origins = {getattr(src, "origin", None) for src in kernels.sources}
    origins.discard(None)
    faults = {o for o in origins if o.startswith("fault:")}
    plain = origins - faults
    if len(faults) > 1 or not plain <= set(REBUILDABLE):
        return None
    if not plain:
        return None
    spec: dict[str, Any] = {
        "provider": REBUILDABLE[sorted(plain)[0]] if len(plain) == 1 else None,
        "sites": tuple(kernels.patched),
    }
    if spec["provider"] is None:
        return None
    if faults:
        name, _, magnitude = sorted(faults)[0][len("fault:"):].partition("@")
        site = next(s.site for s in kernels.sources
                    if getattr(s, "origin", "").startswith("fault:"))
        spec["fault"] = {"name": name, "site": site,
                         "magnitude": float(magnitude), "describes": ""}
    return spec


def run_for(kernels, workload, *, module: str, suite: str,
            calls: dict[str, int] | None = None,
            device: str = "cuda") -> dict[str, Any]:
    """The purity gate for one kernel set, in a child process when it can be.

    A second Python object is not a reset -- module-level counters, lazily built
    caches and autotuner memos all survive one. So the preference is always a
    fresh interpreter; when a provider cannot be named to one, the in-process
    run still happens and the report says which it was, because a gate that
    silently downgrades its own isolation is worse than one that admits it.
    """
    spec = spec_for(kernels)
    if spec is None:
        report = check_kernels(kernels, suite=suite, calls=calls, device=device)
        report["isolation"] = "in-process: this provider has no reconstructible origin"
        report["isolated"] = False
        return report
    report = run_isolated(spec["provider"], module=module, device=device,
                          sites=spec["sites"], fault=spec.get("fault"),
                          workload_config=workload.to_config())
    report["isolated"] = True
    return report


def child_parser(prog: str, description: str | None = None) -> argparse.ArgumentParser:
    """The argument protocol :func:`run_isolated` speaks to its child.

    Owned here so the two halves cannot drift: a workload's entry point builds
    its parser from this, adds nothing, and answers exactly what was sent.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default="structural",
                        choices=("structural", "bound"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sites", default=None)
    parser.add_argument("--fault", default=None)
    parser.add_argument("--workload", default=None)
    parser.add_argument("--result-json", default=None)
    return parser


def child_main(args, kernels, *, suite: str,
               calls: dict[str, int] | None = None) -> int:
    """Run the gate on a rebuilt provider and write the child's result.

    The workload's entry point does the one thing this cannot: turn
    ``--provider`` and ``--fault`` back into a :class:`KernelSet`, which needs
    its registry and its adapters. Everything after that is the same for
    every workload.
    """
    report = check_kernels(kernels, suite=suite, calls=calls, device=args.device)
    report["provider"] = args.provider
    report["isolation"] = "disposable child process"
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as file:
            file.write(text)
    else:
        print(text)
    return 0 if report["ok"] else 1
