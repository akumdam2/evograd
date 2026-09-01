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
    y.backward(dy)

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
from torch import nn

from evograd.opdecl.activity import Active, OpDecl, Workload
from evograd.opdecl.bind import bind, lookup_pair
from evograd.opdecl.inputs import make_case_inputs
from evograd.opdecl.oracle import resolve_runtime_forward


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


class _LayerModule(nn.Module):
    """Shared wrapping: declared parameters held, declared activations taken.

    Subclasses differ only in the callable they route to. Both are called with
    the activations in declared order and supply the parameters and scalars
    themselves, so the eager baseline and the candidate are invoked through the
    same shape of call.
    """

    def __init__(self, op: OpDecl, *, param_shapes: dict[str, tuple[int, ...]]):
        super().__init__()
        self.op_name = op.name
        self._activation_args, self._param_args = activation_and_parameter_args(op)
        self._scalars = scalar_kwargs(op)
        for arg in self._param_args:
            self.register_parameter(
                arg.name, nn.Parameter(torch.empty(param_shapes[arg.name]))
            )

    def parameter_list(self):
        return [getattr(self, arg.name) for arg in self._param_args]

    def _named(self, activations):
        named = {a.name: v for a, v in zip(self._activation_args, activations)}
        named.update({a.name: getattr(self, a.name) for a in self._param_args})
        named.update(self._scalars)
        return named


class CandidateModule(_LayerModule):
    """A candidate program's pair, wrapped as the layer it is meant to replace.

    The autograd wiring is ``opdecl.bind``, not a local ``autograd.Function``.
    It routes the saved state through ``save_for_backward`` while keeping plain
    values in a side layout, checks the returned gradient count against
    ``op.grad_names()``, and places each gradient in its declared argument slot.
    That last part is what a positional wrapper cannot do: once an operator's
    activations are not its leading arguments -- ``af3_single_repr_block`` has
    ``pair_bias`` in the middle -- gradient order and call order diverge.
    """

    def __init__(self, op: OpDecl, program_module, *, param_shapes: dict[str, tuple[int, ...]]):
        super().__init__(op, param_shapes=param_shapes)
        lookup_pair(op, program_module)  # fail here, not inside a timed region
        self._call = bind(op, program_module)

    def forward(self, *activations):
        return self._call(**self._named(activations))


class EagerModule(_LayerModule):
    """The declared production forward, differentiated by the autograd engine.

    This is the eager PyTorch baseline every ratio is taken against. It calls
    ``runtime_forward`` -- the fused spelling a real model uses -- not the
    unfused reference the oracle differentiates.
    """

    def __init__(self, op: OpDecl, *, param_shapes: dict[str, tuple[int, ...]]):
        super().__init__(op, param_shapes=param_shapes)
        self._arg_names = tuple(arg.name for arg in op.args)
        self._forward = resolve_runtime_forward(op)

    def forward(self, *activations):
        named = self._named(activations)
        return self._forward(*(named[name] for name in self._arg_names))


def case_tensors(op: OpDecl, workload: Workload, *, device: str = "cuda"):
    """Activations, upstream gradient, and parameter values for one workload.

    Returns a *list* of activations: an operator may take more than one --
    ``geglu(a, b)`` takes two and owns no parameters at all.
    """
    values = make_case_inputs(op, workload, device=device)
    activation_args, param_args = activation_and_parameter_args(op)
    activations = [
        values[a.name].detach().clone().requires_grad_(True) for a in activation_args
    ]
    dy = values[op.upstream_grad_name].detach().clone()
    params = {a.name: values[a.name].detach().clone() for a in param_args}
    return activations, dy, params


def load_parameters(model: nn.Module, params: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in params.items():
            getattr(model, name).copy_(value)


def make_training_step(
    model: nn.Module, activations: list[torch.Tensor], dy: torch.Tensor
) -> Callable[[], None]:
    """THE timed region. Imported by both the fitness and the benchmark."""

    def step() -> None:
        model.zero_grad(set_to_none=True)
        for activation in activations:
            activation.grad = None
        y = model(*activations)
        y.backward(dy)

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

    activations, dy, params = case_tensors(op, workload, device=device)
    model = EagerModule(op, param_shapes={k: tuple(v.shape) for k, v in params.items()})
    model = model.to(device=device, dtype=activations[0].dtype)
    load_parameters(model, params)
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
    activations, dy, params = case_tensors(op, workload, device=device)
    model = CandidateModule(
        op, program_module, param_shapes={k: tuple(v.shape) for k, v in params.items()}
    )
    model = model.to(device=device, dtype=activations[0].dtype)
    load_parameters(model, params)
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
