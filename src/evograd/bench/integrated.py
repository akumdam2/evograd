"""The integrated training step: a declared pair used as a real ``nn.Module``.

A direct forward/backward pair measurement leaves out everything the deployment
path pays for -- ``autograd.Function`` dispatch, the autograd engine, ``.grad``
accumulation, ``save_for_backward``. On a small LayerNorm those costs are the
overwhelming majority of a training step, so a fitness computed on the pair
alone can rank candidates by a quantity the deployment path barely reflects.

Everything here is shared by the evolution fitness and the final benchmark, so
the two cannot drift apart: same module wrapper, same timed region, same
gradient-reset policy, same saved-state semantics.

The measured region is::

    model.zero_grad(set_to_none=True)
    for activation in activations:
        activation.grad = None
    y = model(*activations)
    torch.autograd.backward(y, dy)

Gradient reset is inside it. Real training pays it every step, and the batched
measurement runs many steps under one event pair -- with the reset outside, the
batched and per-step numbers would not describe the same thing.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from evograd.bench.tier2 import eager_module, pair_module
from evograd.opdecl.activity import Active, OpDecl, Workload
from evograd.opdecl.bind import lookup_pair
from evograd.opdecl.inputs import (
    as_output_tuple,
    make_case_inputs,
    upstream_grad_values,
)

def activation_and_parameter_args(
    op: OpDecl,
) -> tuple[tuple[Active, ...], tuple[Active, ...]]:
    """Split the Active tensor args into activations and parameters.

    Read from the declaration's ``parameter_args``, never inferred. This used to
    guess -- "the first Active tensor is the layer's input, the rest are its
    weights" -- which is true of LayerNorm's ``x, weight, bias`` and of nine
    other declarations, and false of nine more. ``geglu(a, b)`` and
    ``swiglu(a, b)`` take two activations and own no weights at all, so the
    guess registered a tensor flowing through the network as an
    ``nn.Parameter``; ``fused_add_rms_norm(x, r, weight)`` did the same to the
    residual. Nothing raised. The module built fine and the training step it
    measured was not the one the operator describes.

    Both are ``Active`` because both take gradients, and at the direct-pair
    level the distinction does not exist -- every argument is passed
    positionally. It appears only here, where an ``nn.Module`` has to hold some
    of them and be called with the rest, which is why the declaration carries it
    explicitly.
    """
    if op.parameter_args is None:
        raise ValueError(
            f"{op.name}: the integrated step wraps the operator as an nn.Module, "
            "which holds parameters and takes activations as call arguments. The "
            "declaration does not say which Active args are which -- set "
            "`parameter_args` to the names of its parameters, or to `()` if it "
            "has none."
        )
    parameters = tuple(a for a in op.active_args() if a.name in op.parameter_args)
    activations = tuple(a for a in op.active_args() if a.name not in op.parameter_args)
    if not activations:
        raise ValueError(
            f"{op.name}: parameter_args names every Active arg, leaving no "
            "activation to call the module with"
        )
    return activations, parameters


def scalar_kwargs(op: OpDecl) -> dict[str, Any]:
    return {
        a.name: a.default
        for a in op.args
        if getattr(a, "shape", None) is None and hasattr(a, "default")
    }


# The module wrapper is `bench.tier2.OperatorModule`, not a second copy of it.
# Both this file and tier 2 answer the same question -- hold the declared
# parameters, take the declared activations, route the call through a
# differentiable callable -- and two implementations of that would drift. What
# stays here is the *protocol*: a training step with the gradient reset inside
# the timed region, batched under one event pair, which is a different
# measurement from tier 2's `do_bench` of forward and step.
#
# Reusing it also fixes a wrapper this file never had: `OperatorModule`
# registers tensor `Inactive` args as buffers, so operators carrying rotary
# tables or label tensors (`rope`, `cross_entropy`, `evoattention`) can be
# wrapped at all. The local wrapper passed only activations, parameters and
# scalars, and raised a `KeyError` on the rest.


def candidate_module(op: OpDecl, program_module, *, values: dict[str, Any]):
    """A candidate program's pair, wrapped as the layer it is meant to replace.

    The autograd wiring is ``opdecl.bind``: it routes the saved state through
    ``save_for_backward`` while keeping plain values in a side layout, checks
    the returned gradient count against ``op.grad_names()``, and places each
    gradient in its declared argument slot. That last part is what a positional
    wrapper cannot do -- once an operator's activations are not its leading
    arguments (``af3_single_repr_block`` has ``pair_bias`` in the middle),
    gradient order and call order diverge.
    """
    lookup_pair(op, program_module)  # fail here, not inside a timed region
    return pair_module(
        op, program_module, adapter_kind="candidate_pair_module", values=values
    )


def eager_layer(op: OpDecl, values: dict[str, Any]):
    """The declared production forward, differentiated by the autograd engine.

    This is the eager PyTorch baseline every ratio is taken against. It calls
    ``runtime_forward`` -- the fused spelling a real model uses -- not the
    unfused reference the oracle differentiates.
    """
    return eager_module(op, values)


def case_tensors(op: OpDecl, workload: Workload, *, device: str = "cuda"):
    """Activations, upstream gradient(s), and parameter values for one workload.

    Returns a *list* of activations: an operator may take more than one --
    ``geglu(a, b)`` takes two and owns no parameters at all. ``dy`` mirrors the
    declared output shape: a Tensor for one output, an ordered tuple of them
    for several. The third element is the full declared input dict, which the
    shared module wrapper reads its parameters and buffers from.
    """
    values = make_case_inputs(op, workload, device=device)
    activation_args, _param_args = activation_and_parameter_args(op)
    activations = [
        values[a.name].detach().clone().requires_grad_(True) for a in activation_args
    ]
    upstream = upstream_grad_values(op, values)
    dy = (
        tuple(d.detach().clone() for d in upstream)
        if isinstance(upstream, tuple)
        else upstream.detach().clone()
    )
    # The whole declared input dict travels, not just the parameters: the shared
    # wrapper takes its buffers from it too, and every provider built from the
    # same dict runs on byte-identical weights.
    return activations, dy, values


def make_training_step(
    model, activations: list[torch.Tensor], dy: Any
) -> Callable[[], None]:
    """THE timed region. Imported by both the fitness and the benchmark.

    ``torch.autograd.backward`` rather than ``y.backward(dy)`` because a
    multi-output layer has no single ``y`` to call it on: every declared output
    is produced inside the region and every one is given its own upstream
    gradient, so none of them can be dropped from what is timed.
    """
    op = model._op
    dys = dy if isinstance(dy, tuple) else (dy,)

    def step() -> None:
        model.zero_grad(set_to_none=True)
        for activation in activations:
            activation.grad = None
        y = model(*activations)
        torch.autograd.backward(as_output_tuple(op, y), dys)

    return step


def batched_step_ms(
    step: Callable[[], None], *, warmup: int, steps: int, blocks: int
) -> dict[str, Any]:
    """One CUDA event pair around many steps, repeated.

    A single event pair per step measures partly its own overhead once a step is
    order 10 us; amortizing over many steps removes that and reports what a
    training loop actually sustains.
    """
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    per_block = []
    for _ in range(blocks):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(steps):
            step()
        end.record()
        torch.cuda.synchronize()
        per_block.append(float(start.elapsed_time(end)) / steps)
    return {
        "median_ms": statistics.median(per_block),
        "min_ms": min(per_block),
        "per_block_ms": per_block,
        "steps": steps,
        "blocks": blocks,
    }


def _cache_path() -> Path | None:
    raw = os.environ.get("EVOGRAD_EAGER_STEP_CACHE_PATH")
    return Path(raw) if raw else None


def eager_step_ms(
    op: OpDecl,
    workload: Workload,
    *,
    device: str = "cuda",
    warmup: int = 10,
    steps: int = 200,
    blocks: int = 3,
) -> float:
    """Eager latency for one workload, cached across candidate evaluations.

    Every candidate is divided by the same eager number, so measuring it once
    per shape keeps the ratio stable and keeps the fitness affordable. The cache
    is keyed by everything that changes the measurement.
    """
    key = f"{op.name}|{sorted(workload.dims.items())}|{workload.dtype}|{device}|{steps}x{blocks}"
    path = _cache_path()
    cache: dict[str, float] = {}
    if path and path.is_file():
        try:
            cache = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
    if key in cache:
        return float(cache[key])

    activations, dy, values = case_tensors(op, workload, device=device)
    model = eager_layer(op, values)
    result = batched_step_ms(
        make_training_step(model, activations, dy),
        warmup=warmup, steps=steps, blocks=blocks,
    )
    value = float(result["median_ms"])
    if path:
        cache[key] = value
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2))
        except OSError:
            pass
    return value


def candidate_step_ms(
    op: OpDecl,
    program_module,
    workload: Workload,
    *,
    device: str = "cuda",
    warmup: int = 10,
    steps: int = 200,
    blocks: int = 3,
) -> float:
    activations, dy, values = case_tensors(op, workload, device=device)
    model = candidate_module(op, program_module, values=values)
    result = batched_step_ms(
        make_training_step(model, activations, dy),
        warmup=warmup, steps=steps, blocks=blocks,
    )
    return float(result["median_ms"])


def integrated_ratio_report(
    op: OpDecl,
    program_module,
    workloads: tuple[Workload, ...],
    *,
    device: str = "cuda",
    warmup: int = 10,
    steps: int = 200,
    blocks: int = 3,
) -> dict[str, Any]:
    """Per-shape candidate/eager ratios and their equal-weight geometric mean.

    Equal weight per shape by construction: the sweep spans a 16384x range in
    rows, and a mean over raw latencies would be decided entirely by its top end.
    """
    per_shape = []
    for workload in workloads:
        eager = eager_step_ms(
            op, workload, device=device, warmup=warmup, steps=steps, blocks=blocks
        )
        candidate = candidate_step_ms(
            op, program_module, workload, device=device,
            warmup=warmup, steps=steps, blocks=blocks,
        )
        per_shape.append(
            {
                "dims": dict(workload.dims),
                "dtype": workload.dtype,
                "eager_ms": eager,
                "candidate_ms": candidate,
                "ratio_vs_eager": candidate / eager,
            }
        )
        torch.cuda.empty_cache()
    ratios = [entry["ratio_vs_eager"] for entry in per_shape]
    geomean = statistics.geometric_mean(ratios)
    return {
        "per_shape": per_shape,
        "ratio_vs_eager_geomean": geomean,
        "speedup_vs_eager_geomean": 1.0 / geomean,
        "shapes": len(per_shape),
    }
