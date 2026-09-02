"""**Runner** — build every provider, time them, and report.

The last of tier 3's three parts:

    tier3_model.py    the workload protocol, and the import path
    tier3_patch.py    kernels, sites, and the two ways to insert one
    tier3_runner.py   this file — building, timing, and reporting

Knows nothing about any particular model, and nothing about how a kernel got
into one. It receives a workload and a set of providers, and its whole job is
to make sure the only difference between them is the kernels: same weights,
same batches, same optimizer, same step, same order of operations.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from evograd.bench.tier3_model import TrainingWorkload
from evograd.bench.tier3_patch import KernelSet

TIER3_PROTOCOL_VERSION = "evograd-tier3-model-v1"

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


def measure_step(step, *, warmup: int, steps: int, blocks: int) -> dict[str, Any]:
    """Step latency, throughput, and how much of the step the CPU was busy for.

    ``cpu_bound_fraction`` is the metric this tier exists to expose, and it is
    cheap: time the step twice, once returning as soon as the CPU has stopped
    submitting work and once after the GPU has drained. Near 1.0 means the GPU
    was never the thing being waited on — at which point a faster kernel cannot
    show up in throughput no matter how much faster it is, and a null result
    means nothing about the kernel.

    Read it as "the GPU was not the bottleneck", not as "this is all dispatch
    cost". The CPU also stops running when it blocks on an implicit
    synchronization, and a step that allocates a 2 GB logits tensor every
    iteration will hit the allocator hard enough to do exactly that. Both are
    real reasons a kernel improvement cannot surface, which is why one number
    covers both — but attributing it specifically to dispatch needs a profile.

    Dispatch is nonetheless the expected term once kernels are patched in: each
    patched site is a Python ``autograd.Function`` callback, and every one of
    them blocks the launch pipeline while the engine acquires the GIL.
    """
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    per_block, cpu_fractions = [], []
    for _ in range(blocks):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            step()
        submitted = time.perf_counter()
        torch.cuda.synchronize()
        finished = time.perf_counter()

        wall = finished - start
        per_block.append(wall / steps * 1e3)
        cpu_fractions.append((submitted - start) / wall if wall > 0 else 1.0)

    return {
        "step_ms": statistics.median(per_block),
        "step_ms_min": min(per_block),
        "per_block_ms": per_block,
        "cpu_bound_fraction": statistics.median(cpu_fractions),
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
    repeatedly on one batch memorizes it — a 4-layer Llama drops from 12.5 to
    0.001 within two steps — after which every provider reads ~0 and agreement
    is trivially satisfied. Each step is seeded, so the sequence is identical
    across providers.
    """
    losses = []
    for index in range(steps):
        batch = workload.batch_for(seed=seed + 1 + index)
        loss = workload.loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
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
) -> dict[str, Any]:
    """One kernel set on one workload: loss trajectory, then throughput and memory."""
    model = workload.build(kernels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = loss_trajectory(
        workload, model, optimizer, steps=loss_steps, seed=seed
    )

    batch = workload.batch_for(seed=seed)
    step = make_training_step(workload, model, optimizer, batch)

    # Peak memory over a settled step, measured outside the timing loops: the
    # optimizer's state is allocated on its first step, so measuring before that
    # would report a number no steady-state training run ever sees.
    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak_memory = int(torch.cuda.max_memory_allocated())

    timing = measure_step(step, warmup=warmup, steps=steps, blocks=blocks)
    units = workload.units_per_step()
    return {
        **timing,
        "units_per_second": units / (timing["step_ms"] / 1e3),
        "unit_name": workload.unit_name,
        "peak_memory_bytes": peak_memory,
        "losses": losses,
        "patched": list(kernels.patched),
    }



def run_tier3(
    workload: TrainingWorkload,
    providers: dict[str, KernelSet],
    *,
    warmup: int = 3,
    steps: int = 10,
    blocks: int = 3,
    loss_steps: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Every provider on one workload.

    The harness never asks what the model is. It builds, feeds, and times
    whatever the workload hands back, so a new model is a new
    :class:`TrainingWorkload` and nothing in this file changes.
    """
    from evograd.bench.tier1 import environment_fingerprint

    results: dict[str, Any] = {}
    for name, kernels in providers.items():
        try:
            results[name] = {"ok": True, **measure_provider(
                workload, kernels, warmup=warmup, steps=steps, blocks=blocks,
                loss_steps=loss_steps, seed=seed,
            )}
        except Exception as exc:  # one provider must not take the run down
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()

    return {
        "protocol": TIER3_PROTOCOL_VERSION,
        **workload.describe(),
        "timing_protocol": {
            "step": "loss = workload.loss(model, batch); loss.backward(); opt.step(); opt.zero_grad()",
            "optimizer": "AdamW",
            "weights": "built by the workload from a fixed seed, identical across providers",
            "batch_data": "seeded; every provider sees the same sequence",
            "loss_trajectory": "fresh batch per step, so agreement is not trivially satisfied",
            "peak_memory": "measured after the optimizer state exists",
            "cpu_bound_fraction": (
                "CPU-busy time / total wall time. Near 1.0 means the GPU was not "
                "the bottleneck — dispatch cost, or blocking on an implicit "
                "synchronization such as allocator pressure"
            ),
        },
        "environment": environment_fingerprint(),
        "providers": results,
    }


def loss_agreement(report: dict[str, Any], reference: str = "eager") -> dict[str, Any]:
    """Largest absolute loss difference from the reference, per provider.

    Reported rather than gated: how much divergence is acceptable depends on the
    dtype and the horizon, and a threshold picked here would be arbitrary. What
    matters is that it is visible — a kernel that is fast and quietly changes
    the loss curve is not a faster kernel.
    """
    providers = report.get("providers") or {}
    base = providers.get(reference)
    if not base or not base.get("ok"):
        return {"reference": reference, "available": False}
    expected = base["losses"]
    out: dict[str, Any] = {"reference": reference, "available": True, "max_abs_delta": {}}
    for name, entry in providers.items():
        if name == reference or not entry.get("ok"):
            continue
        out["max_abs_delta"][name] = max(
            abs(a - b) for a, b in zip(expected, entry["losses"])
        )
    return out
