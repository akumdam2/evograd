"""**Runner** -- build every provider, time them, and report.

The last of tier 3's three parts:

    tier3_model.py    the workload protocol, and the import path
    tier3_patch.py    kernels, sites, and the two ways to insert one
    tier3_runner.py   this file -- building, timing, and reporting

Knows nothing about any particular model, and nothing about how a kernel got
into one. It receives a workload and a set of providers, and its whole job is
to make sure the only difference between them is the kernels: same weights,
same batches, same optimizer, same step, same order of operations.

Two rules the earlier version of this file did not have, both about not
reporting a number that is not a number:

* **Correctness first.** Every kernel a provider patches in is put through the
  tier-1 pair gate -- the same ``verify_pair_provider`` tiers 1 and 2 use --
  before anything is timed. A provider that fails is marked failed and never
  measured, because a wrong kernel does not produce an error at this tier, it
  produces a throughput.
* **Every loss must be a finite scalar.** A step whose loss is NaN still runs,
  still has a wall time, and still divides into tokens per second. Checking is
  one ``isfinite`` per step and it is the difference between a measurement and
  a number.
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Any, Callable

import torch

from evograd.bench.tier3_model import (
    TrainingWorkload,
    build_with_provenance,
    site_registry_for,
)
from evograd.bench.tier3_patch import KernelSet

TIER3_PROTOCOL_VERSION = "evograd-tier3-model-v2"

#: What the optimizer is, spelled out rather than implied by "AdamW". Every
#: provider gets exactly this, and the report carries it, because a comparison
#: across two runs with different weight decay is not a comparison.
OPTIMIZER = "AdamW"
OPTIMIZER_DEFAULTS: dict[str, Any] = {
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.01,
    "amsgrad": False,
}


class Tier3Error(RuntimeError):
    """A provider cannot be measured. Marks that provider failed, not the run."""


class NonFiniteLoss(Tier3Error):
    """A loss was NaN, infinite, or not a scalar."""


class PreflightFailure(Tier3Error):
    """A patched kernel did not pass the tier-1 correctness gate."""


class ModelCorrectnessFailure(Tier3Error):
    """The assembled model failed its whole-model gate. Never timed."""


# ── correctness, before anything is timed ────────────────────────────────────


def check_loss(value: Any, *, where: str) -> float:
    """Every loss must be a finite scalar. Returns it as a float.

    Not a formality. ``loss.backward()`` accepts a NaN happily, the optimizer
    steps on NaN gradients happily, the step has a wall time, and that wall time
    divides into a perfectly plausible tokens/s. A provider whose kernel
    overflows in bfloat16 therefore reports *faster* than eager, because NaN
    arithmetic is not slower. The check costs one ``isfinite`` per step.
    """
    if not torch.is_tensor(value):
        raise NonFiniteLoss(
            f"{where}: loss is {type(value).__name__}, not a tensor"
        )
    if value.numel() != 1:
        raise NonFiniteLoss(
            f"{where}: loss has shape {tuple(value.shape)}; a training step needs "
            "a scalar to call backward() on"
        )
    scalar = float(value.detach())
    if not (scalar == scalar and abs(scalar) != float("inf")):
        raise NonFiniteLoss(f"{where}: loss is {scalar}")
    return scalar


def preflight(
    kernels: KernelSet, ops: dict[str, Any] | None, *, device: str = "cuda"
) -> dict[str, Any]:
    """Put every patched kernel through the tier-1 correctness gate.

    Reuses ``bench.provider.verify_pair_provider`` -- the same path tier 1's CLI
    and tier 2's ``check_module`` gate on -- against the declaration's own
    correctness workloads. Tier 3 cannot verify the model-shaped call directly:
    at these sites the model's activations are not a declared workload, and the
    only oracle that exists is the declaration's. So the kernel is verified
    where it *is* specified, and the rank adapter is what carries that verdict
    to the model's shapes.

    Raises :class:`PreflightFailure`, so a provider that fails is recorded as
    failed and never reaches the timing loops.
    """
    from evograd.bench.provider import candidate_provider, verify_pair_provider

    registry = kernels.registry
    checked: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    for site in kernels.patched:
        source = kernels.source_for(site)
        if source is None or not source.verifiable or ops is None:
            unverifiable.append(
                {
                    "site": site,
                    "workload_family": registry.name,
                    "reason": (
                        "no declaration-governed pair behind this site; a raw "
                        "callable has no oracle to check it against"
                    ),
                }
            )
            continue
        op = ops[source.op_name]
        # The declaration's own grid, plus whatever model-derived shapes this
        # workload's registry adds for the site. The second half is the point:
        # a kernel correct at 32 rows and wrong at 4096 passes the first and
        # fails the second, and only the model knows which 4096 it presents.
        extra = tuple(registry.require(site).preflight)
        workloads = tuple(op.correctness) + extra
        report = verify_pair_provider(
            op, candidate_provider(op, source.module), workloads, device=device
        )
        entry = {
            "site": site,
            "op": source.op_name,
            "origin": source.origin,
            "workload_family": registry.name,
            "ok": report.ok,
            "cases": len(report.cases),
            "declared_cases": len(op.correctness),
            "workload_supplied_cases": len(extra),
            "checked_configs": [
                {
                    "dims": dict(workload.dims),
                    "dtype": workload.dtype,
                    "source": "declared" if index < len(op.correctness)
                              else "workload_supplied",
                    "id": _workload_id(workload),
                }
                for index, workload in enumerate(workloads)
            ],
        }
        if not report.ok:
            failing = [
                {
                    "dims": case.dims,
                    "dtype": case.dtype,
                    "error": case.error,
                    "failed": [c.name for c in case.checks if not c.ok],
                }
                for case in report.cases
                if not case.ok
            ]
            entry["failures"] = failing
            checked.append(entry)
            raise PreflightFailure(
                f"{site} ({source.op_name}, workload {registry.name}) failed the "
                f"tier-1 correctness gate: {failing[0]}"
            )
        checked.append(entry)
    return {
        "gate": (
            "bench.provider.verify_pair_provider on op.correctness plus the "
            "workload registry's model-derived shapes"
        ),
        "device": device,
        "site_ops": registry.site_ops,
        "workload_family": registry.name,
        "checked": checked,
        "unverifiable": unverifiable,
    }


def _workload_id(workload) -> str:
    """A short, stable name for one checked configuration."""
    provenance = getattr(workload, "provenance", None)
    label = getattr(provenance, "component", None) if provenance else None
    dims = ",".join(f"{k}={v}" for k, v in sorted(workload.dims.items()))
    return f"{label or 'workload'}[{dims}]:{workload.dtype}"


# ── the timed step ───────────────────────────────────────────────────────────


def make_training_step(workload: TrainingWorkload, model, optimizer, batch):
    """THE timed region: the four things a training loop does per step."""

    def step():
        loss = workload.loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    return step


def _sync() -> None:
    """Drain the GPU, or do nothing when there is not one.

    The tier measures training on an accelerator, and every published number
    comes from one. Tolerating its absence is what lets the whole runner --
    ordering, the finiteness gate, the preflight, provenance -- be exercised on
    CPU in the test suite instead of only where a GPU happens to be free.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_memory() -> int | None:
    """Peak allocation, or ``None`` where the allocator does not exist."""
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def model_correctness_check(workload, kernels, *, verify: bool, device: str):
    """Ask the workload for its whole-model gate, and refuse to time a failure.

    Optional by design: a workload that has not calibrated one returns
    ``not_applicable`` and tier 3 proceeds on site preflight alone, which is
    where every workload started. A workload that *has* one and fails it raises,
    so the provider is recorded ``failed_at="model_correctness"`` and never
    reaches a timer.
    """
    check = getattr(workload, "model_correctness", None)
    if not verify or not callable(check):
        return {"gate": "not_applicable" if callable(check) else "none",
                "ok": True, "skipped": not verify}
    if not kernels.patched:
        return {"gate": "unpatched provider", "ok": True, "skipped": True}
    verdict = check(kernels, device=device)
    if not verdict.get("ok"):
        raise ModelCorrectnessFailure(
            f"the assembled model failed its whole-model gate: "
            f"{verdict.get('reason', 'unknown')}"
        )
    return verdict


def _percentile(samples: list[float], q: float) -> float:
    # Same definition as tier 1's, kept local rather than imported across a
    # private name; the two must agree, and a five-line function is a cheaper
    # way to guarantee that than a cross-module import of an underscore name.
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "q20_ms": _percentile(samples, 0.2),
        "q80_ms": _percentile(samples, 0.8),
    }


def measure_step(step, *, warmup: int, steps: int, blocks: int) -> dict[str, Any]:
    """Step latency, throughput, and how much of the step the CPU was busy for.

    ``cpu_bound_fraction`` is the metric this tier exists to expose, and it is
    cheap: time the step twice, once returning as soon as the CPU has stopped
    submitting work and once after the GPU has drained. Near 1.0 means the GPU
    was never the thing being waited on -- at which point a faster kernel cannot
    show up in throughput no matter how much faster it is, and a null result
    means nothing about the kernel.

    It is the **CPU submission-and-blocking fraction**, and nothing narrower.
    The CPU stops running both when it has finished submitting and when it
    blocks -- on an implicit synchronization, on the allocator while a step
    reserves a multi-gigabyte logits tensor, on a ``.item()`` someone left in a
    loss. All of those are real reasons a kernel improvement cannot surface,
    which is why one number covers them, and none of them is separable from the
    others without a profile. Do not report it as dispatch cost.

    No L2 flush anywhere in here, deliberately. Tiers 1 and 2 clear the cache
    before each timed region because they measure one kernel on inputs it would
    otherwise find resident. This measures a training step, whose weights,
    activations and optimizer state are exactly as warm as the previous step
    left them -- that is the thing being measured, and flushing it would price
    a step no training loop ever runs.
    """
    for _ in range(warmup):
        step()
    _sync()

    per_block, cpu_fractions = [], []
    for _ in range(blocks):
        _sync()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        submitted = time.perf_counter()
        _sync()
        finished = time.perf_counter()

        wall = finished - start
        per_block.append(wall / steps * 1e3)
        cpu_fractions.append((submitted - start) / wall if wall > 0 else 1.0)

    return {
        "step_ms": statistics.median(per_block),
        "step_ms_min": min(per_block),
        "per_block_ms": per_block,
        "latency": _summary(per_block),
        "cpu_bound_fraction": statistics.median(cpu_fractions),
        "cpu_bound_fraction_per_block": cpu_fractions,
        "steps": steps,
        "blocks": blocks,
    }


def loss_trajectory(
    workload: TrainingWorkload, model, optimizer, *, steps: int, seed: int
) -> list[float]:
    """The first N losses, on a fresh batch each step, for cross-provider agreement.

    Correctness at this tier is not ``allclose`` on one gradient: it is whether
    the model still trains. Two kernel sets that agree to tolerance on a single
    call can still diverge once their differences compound through an optimizer,
    and that divergence is what a user would actually notice.

    The batch has to change between steps or the metric says nothing. Training
    repeatedly on one batch memorizes it -- a 4-layer Llama drops from 12.5 to
    0.001 within two steps -- after which every provider reads ~0 and agreement
    is trivially satisfied. Each step is seeded, so the sequence is identical
    across providers.

    Every loss is checked for scalarity and finiteness as it is produced, so a
    provider that diverges is failed here rather than timed and reported.
    """
    losses = []
    for index in range(steps):
        batch = workload.batch_for(seed=seed + 1 + index)
        loss = workload.loss(model, batch)
        losses.append(check_loss(loss, where=f"loss trajectory step {index}"))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return losses


def measure_provider(
    workload: TrainingWorkload,
    kernels: KernelSet,
    *,
    warmup: int = 3,
    steps: int = 10,
    blocks: int = 3,
    loss_steps: int = 5,
    learning_rate: float = 1e-4,
    seed: int = 0,
    ops: dict[str, Any] | None = None,
    verify: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """One kernel set on one workload: verify, then loss trajectory, then timing."""
    verification = (
        preflight(kernels, ops, device=device)
        if verify
        else {"gate": "skipped", "checked": [], "unverifiable": []}
    )
    # Whole-model correctness, if the workload has one. Site preflight proves a
    # kernel right at its declared shapes; it cannot prove that a hundred of
    # them assembled into a model still train, and that failure is the one that
    # produces a throughput rather than an error. Outside every timed region.
    model_correctness = model_correctness_check(
        workload, kernels, verify=verify, device=device
    )

    model, provenance = build_with_provenance(workload, kernels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, **OPTIMIZER_DEFAULTS
    )

    losses = loss_trajectory(
        workload, model, optimizer, steps=loss_steps, seed=seed
    )

    batch = workload.batch_for(seed=seed)
    step = make_training_step(workload, model, optimizer, batch)

    # Peak memory over a settled step, measured outside the timing loops: the
    # optimizer's state is allocated on its first step, so measuring before that
    # would report a number no steady-state training run ever sees.
    check_loss(step(), where="timed batch, first step")
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    check_loss(step(), where="timed batch, memory probe")
    _sync()
    peak_memory = _peak_memory()

    timing = measure_step(step, warmup=warmup, steps=steps, blocks=blocks)
    # The timed blocks themselves are not checked: reading a loss forces a
    # device synchronization, and doing that once per step would destroy the
    # very measurement `cpu_bound_fraction` reports. So the settled steps on
    # either side of them are, which catches a kernel that only diverges after
    # the optimizer has moved -- the failure mode a single pre-check misses.
    check_loss(step(), where="after the timed blocks")
    units = workload.units_per_step()
    return {
        **timing,
        "units_per_second": units / (timing["step_ms"] / 1e3),
        "unit_name": workload.unit_name,
        "peak_memory_bytes": peak_memory,
        "losses": losses,
        "patched": list(kernels.patched),
        "patch_provenance": provenance.to_dict(),
        "kernel_sources": [s.to_dict() for s in kernels.sources],
        "verification": verification,
        "model_correctness": model_correctness,
        "optimizer": {
            "name": OPTIMIZER,
            "learning_rate": learning_rate,
            **{k: list(v) if isinstance(v, tuple) else v
               for k, v in OPTIMIZER_DEFAULTS.items()},
        },
    }


# ── ordering and confidence ──────────────────────────────────────────────────


def provider_order(names, *, seed: int) -> list[str]:
    """A seeded random measurement order, so clock drift is not a result.

    Tiers 1 and 2 randomize provider order for the same reason: a GPU that warms
    or throttles over a run gives whichever provider went first a systematic
    advantage, and with a fixed order that advantage is indistinguishable from
    the kernel. Seeded, so the order is reproducible and is recorded in the
    report rather than being an unstated property of the loop.
    """
    ordered = list(names)
    random.Random(seed).shuffle(ordered)
    return ordered


def _bootstrap_ratio(
    reference_blocks: list[float],
    candidate_blocks: list[float],
    *,
    iterations: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """A percentile interval for reference/candidate step time.

    Blocks are resampled with replacement within each provider, independently:
    unlike tier 1, the two providers here are not measured in interleaved paired
    blocks -- each builds and trains its own model -- so there is no pairing to
    preserve and pretending otherwise would narrow the interval on an assumption
    that does not hold. With the default three blocks the interval is wide; that
    is the honest width, and ``--blocks`` is how it narrows.
    """
    if not reference_blocks or not candidate_blocks:
        return {"iterations": 0, "low": float("nan"), "high": float("nan")}
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        reference = statistics.median(
            [rng.choice(reference_blocks) for _ in reference_blocks]
        )
        candidate = statistics.median(
            [rng.choice(candidate_blocks) for _ in candidate_blocks]
        )
        if candidate > 0:
            estimates.append(reference / candidate)
    if not estimates:
        return {"iterations": 0, "low": float("nan"), "high": float("nan")}
    return {
        "iterations": len(estimates),
        "low": _percentile(estimates, 0.025),
        "high": _percentile(estimates, 0.975),
    }


def speedup_intervals(
    report: dict[str, Any], reference: str = "eager", *, seed: int = 0
) -> dict[str, Any]:
    """Per-provider step-time ratio against the reference, with a 95% interval."""
    providers = report.get("providers") or {}
    base = providers.get(reference)
    if not base or not base.get("ok"):
        return {"reference": reference, "available": False}
    out: dict[str, Any] = {"reference": reference, "available": True, "vs_reference": {}}
    for index, (name, entry) in enumerate(sorted(providers.items())):
        if name == reference or not entry.get("ok"):
            continue
        out["vs_reference"][name] = {
            "step_ms_ratio": base["step_ms"] / entry["step_ms"],
            "ci95": _bootstrap_ratio(
                base["per_block_ms"], entry["per_block_ms"], seed=seed + index
            ),
        }
    return out


# ── the run ──────────────────────────────────────────────────────────────────


def run_tier3(
    workload: TrainingWorkload,
    providers: dict[str, KernelSet],
    *,
    warmup: int = 3,
    steps: int = 10,
    blocks: int = 3,
    loss_steps: int = 5,
    learning_rate: float = 1e-4,
    seed: int = 0,
    ops: dict[str, Any] | None = None,
    verify: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Every provider on one workload.

    The harness never asks what the model is. It builds, feeds, and times
    whatever the workload hands back, so a new model is a new
    :class:`TrainingWorkload` and nothing in this file changes.

    Providers run **in this process**. That is what makes it usable for a
    ``ModuleWorkload`` built from closures, which cannot be pickled into a
    child -- and it means a kernel that hangs or wedges the CUDA context takes
    the run with it. ``tier3_cli`` runs each provider in a killable child for
    exactly that reason; use it when the workload can be named on a command
    line.
    """
    from evograd.bench.tier1 import environment_fingerprint

    order = provider_order(providers, seed=seed)
    results: dict[str, Any] = {}
    for name in order:
        results[name] = measure_one(
            workload, name, providers[name], warmup=warmup, steps=steps,
            blocks=blocks, loss_steps=loss_steps, learning_rate=learning_rate,
            seed=seed, ops=ops, verify=verify, device=device,
        )

    return assemble_report(
        workload, results, order,
        warmup=warmup, steps=steps, blocks=blocks, loss_steps=loss_steps,
        learning_rate=learning_rate, seed=seed, verify=verify,
        isolation="in-process (see tier3_cli for one child process per provider)",
    )


def measure_one(
    workload: TrainingWorkload,
    name: str,
    kernels: KernelSet,
    **options: Any,
) -> dict[str, Any]:
    """One provider, with its failure captured rather than raised.

    A kernel that fails preflight, diverges to NaN, or runs out of memory must
    cost its own row and nothing else. The CLI wraps this in a child process so
    that a hang or a wedged CUDA context is also survivable; in-process, those
    two are not.
    """
    try:
        return {"ok": True, "provider": name, **measure_provider(
            workload, kernels, **options
        )}
    except Exception as exc:  # one provider must not take the run down
        return {
            "ok": False,
            "provider": name,
            "error": f"{type(exc).__name__}: {exc}",
            "failed_at": _failure_stage(exc),
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _failure_stage(exc: Exception) -> str:
    if isinstance(exc, ModelCorrectnessFailure):
        return "model_correctness"
    if isinstance(exc, PreflightFailure):
        return "preflight"
    if isinstance(exc, NonFiniteLoss):
        return "loss_finiteness"
    out_of_memory = getattr(torch, "OutOfMemoryError", None) or getattr(
        torch.cuda, "OutOfMemoryError", None
    )
    if out_of_memory is not None and isinstance(exc, out_of_memory):
        return "out_of_memory"
    return "measurement"


def assemble_report(
    workload: TrainingWorkload,
    results: dict[str, Any],
    order: list[str],
    *,
    warmup: int,
    steps: int,
    blocks: int,
    loss_steps: int,
    learning_rate: float,
    seed: int,
    verify: bool,
    isolation: str,
) -> dict[str, Any]:
    """The report shell, filled with whatever the providers produced.

    Shared by the in-process runner and the CLI's isolated one, so a report
    means the same thing however the providers were executed.
    """
    report = {
        "protocol": TIER3_PROTOCOL_VERSION,
        **workload.describe(),
        "seed": seed,
        "provider_order": list(order),
        "isolation": isolation,
        "timing_protocol": {
            "step": "loss = workload.loss(model, batch); loss.backward(); opt.step(); opt.zero_grad()",
            "optimizer": OPTIMIZER,
            "optimizer_config": {
                "learning_rate": learning_rate,
                **{k: list(v) if isinstance(v, tuple) else v
                   for k, v in OPTIMIZER_DEFAULTS.items()},
            },
            "weights": "built by the workload from a fixed seed, identical across providers",
            "batch_data": "seeded; every provider sees the same sequence",
            "warmup_steps": warmup,
            "timed_steps_per_block": steps,
            "blocks": blocks,
            "loss_trajectory_steps": loss_steps,
            "provider_order": "seeded random, recorded; identical settings for every provider",
            "l2_policy": (
                "never flushed inside a step -- this measures a training step, "
                "whose state is as warm as the previous step left it"
            ),
            "loss_trajectory": "fresh batch per step, so agreement is not trivially satisfied",
            "peak_memory": "measured after the optimizer state exists",
            "cpu_bound_fraction": (
                "CPU submission-and-blocking fraction: CPU-busy time / total wall "
                "time. Near 1.0 means the GPU was not the bottleneck -- submission "
                "cost, or blocking on an implicit synchronization such as "
                "allocator pressure. Not a dispatch-only measurement"
            ),
        },
        "verification_policy": verification_policy(workload, verify=verify),
        "site_registry": site_registry_for(workload).to_dict(),
        "environment": _environment(),
        "providers": results,
    }
    report["speedup_intervals"] = speedup_intervals(report, seed=seed)
    return report


def _environment() -> dict[str, Any]:
    """Tier 1's fingerprint, or a report that says there was no accelerator.

    ``environment_fingerprint`` reads the device name, which raises without a
    GPU. A published tier-3 number always has one; a CPU run of the harness
    itself does not, and it should produce a report saying so rather than no
    report at all.
    """
    from evograd.bench.tier1 import environment_fingerprint

    if not torch.cuda.is_available():
        return {"gpu_name": None, "torch": torch.__version__, "cuda": None}
    return environment_fingerprint()


def verification_policy(
    workload: TrainingWorkload, *, verify: bool = True
) -> dict[str, Any]:
    """Exactly what was gated, and what was only observed. Recorded in the report."""
    threshold = getattr(workload, "loss_delta_threshold", None)
    return {
        "preflight": (
            "every patched kernel through bench.provider.verify_pair_provider on "
            "its declaration's correctness workloads; a provider that fails is "
            "not timed"
            if verify
            else "skipped by request (--no-verify): timings are ungated"
        ),
        "model_correctness": (
            "a workload may declare a whole-model gate: one untimed canonical "
            "forward/backward/optimizer step against both the original eager "
            "model and the bound-pair path a candidate replaces, checked "
            "against a calibrated per-role noise envelope, plus its loss "
            "trajectory over the calibrated horizon. A failure is recorded "
            "failed_at='model_correctness' and is never timed"
        ),
        "loss_scalar_and_finite": (
            "every loss in the trajectory, and the settled steps on either side "
            "of the timed blocks, must be a finite scalar; NaN or Inf marks the "
            "provider failed. Losses inside the timed blocks are not read: that "
            "would force a synchronization per step and destroy the measurement"
        ),
        "loss_trajectory": (
            f"gated at max |delta| <= {threshold}"
            if threshold is not None
            else "diagnostic only -- reported, not gated. A defensible threshold "
                 "depends on dtype and horizon, so a workload declares one via "
                 "`loss_delta_threshold` or the trajectory stays an observation "
                 "behind the correctness and finiteness gates"
        ),
        "loss_delta_threshold": threshold,
    }


def loss_agreement(
    report: dict[str, Any], reference: str = "eager", *, threshold: float | None = None
) -> dict[str, Any]:
    """Largest absolute loss difference from the reference, per provider.

    Reported, and gated only when the workload declared a threshold. How much
    divergence is acceptable depends on the dtype and the horizon, and a number
    invented here would be arbitrary in exactly the way that makes a gate worse
    than no gate. What matters is that it is visible -- a kernel that is fast
    and quietly changes the loss curve is not a faster kernel -- and that the
    real gates (tier-1 correctness, finite scalar losses) already ran.
    """
    providers = report.get("providers") or {}
    base = providers.get(reference)
    if not base or not base.get("ok"):
        return {"reference": reference, "available": False}
    if threshold is None:
        threshold = (report.get("verification_policy") or {}).get(
            "loss_delta_threshold"
        )
    expected = base["losses"]
    out: dict[str, Any] = {
        "reference": reference,
        "available": True,
        "threshold": threshold,
        "gated": threshold is not None,
        "max_abs_delta": {},
    }
    if threshold is not None:
        out["within_threshold"] = {}
    for name, entry in providers.items():
        if name == reference or not entry.get("ok"):
            continue
        delta = max(abs(a - b) for a, b in zip(expected, entry["losses"]))
        out["max_abs_delta"][name] = delta
        if threshold is not None:
            out["within_threshold"][name] = delta <= threshold
    return out
