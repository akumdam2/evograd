"""**Patcher** — what a kernel is, and how a model comes to hold one.

The middle of tier 3's three parts:

    tier3_model.py    the architecture, and how to bring your own
    tier3_patch.py    this file — kernels, sites, and the two ways to insert one
    tier3_runner.py   building, timing, and reporting

A :class:`KernelSet` says which implementation each patchable site reaches for.
There are two ways a model can be made to reach for a different one, and which
applies depends on whether you wrote the model:

* **By construction** — the model holds a ``KernelSet`` and calls through it, so
  building with a different set *is* the patch. No surgery. This is what the
  built-in Llama workload does.
* **By module surgery** — :func:`patch_modules` walks a built tree and replaces
  matching submodules, carrying their trained weights across. This is the route
  for a model you did not write, and it is what Liger's
  ``apply_liger_kernel_to_llama`` does.

This module depends on nothing else in tier 3, which is what keeps the layering
acyclic: the model layer imports from here, the runner imports from both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from evograd.opdecl.activity import OpDecl
from evograd.opdecl.bind import bind
from evograd.opdecl.oracle import resolve_runtime_forward
from evograd.ops.level3.llama3_decoder_layer.forward_ref import (
    _default_swiglu,
    _rms_norm_fused,
)

# ── the swappable surface ────────────────────────────────────────────────────


def _eager_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, target: torch.Tensor):
    """Logits materialized, then cross-entropy — what an unfused head costs.

    At Llama-3's 128256 vocabulary this tensor is the memory story: 2.1 GB at
    8192 tokens in bfloat16, and the same again for its gradient. A fused loss
    never materializes it, which is why it moves peak memory rather than
    latency, and why peak memory is a reported metric here.
    """
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float().flatten(0, -2), target.flatten())


@dataclass(frozen=True)
class KernelSet:
    """Which implementation the model reaches for at each patchable site.

    Defaults are the declared production spelling. A provider replaces one or
    more entries; nothing else about the model changes, so a difference in the
    measurement is attributable to the swap.
    """

    rms_norm: Callable = _rms_norm_fused
    swiglu: Callable = _default_swiglu
    cross_entropy: Callable = _eager_cross_entropy
    #: Names of the sites actually replaced, for the report.
    patched: tuple[str, ...] = ()


def _declared_rank(shape: str | None) -> int | None:
    """How many dimensions a declared shape string has. ``None`` if not a tensor."""
    if not shape:
        return None
    inner = shape.strip()[1:-1].strip()
    return len(inner.split(",")) if inner else 0


def kernel_from_pair(op: OpDecl, module) -> Callable:
    """A candidate's forward/backward pair as a callable the model can hold.

    ``bind`` is what makes it differentiable: it routes the saved state through
    ``save_for_backward``, checks the returned gradient count, and places each
    gradient in its declared argument slot.

    Around that sits a rank adapter, because the model and the declaration
    disagree about shape and both are right. A model carries activations as
    ``[batch, tokens, hidden]``; every declaration here is written for rows --
    ``rmsnorm`` takes ``[rows, hidden]``, ``fused_linear_cross_entropy`` takes
    ``target: [rows]``. Eager PyTorch hides the difference by broadcasting over
    leading dimensions, so the unpatched model runs and a patched one fails
    inside someone else's kernel with an unpacking error.

    Flattening the leading dimensions and restoring them afterwards is what a
    real integration does too: HuggingFace flattens before the loss, and Liger's
    patches operate on 2D rows. Doing it here keeps the declaration honest --
    a kernel is measured on exactly the shape it was declared, verified and
    evolved against -- rather than silently requiring every candidate to handle
    a rank its contract never mentioned.
    """
    call = bind(op, module)
    ranks = [_declared_rank(getattr(arg, "shape", None)) for arg in op.args]
    output_rank = _declared_rank(getattr(op.output, "shape", None))

    def flattened(*args):
        leading: tuple[int, ...] | None = None
        adapted = []
        for value, rank in zip(args, ranks):
            if rank is None or not torch.is_tensor(value) or value.dim() <= rank:
                adapted.append(value)
                continue
            extra = value.dim() - rank + 1
            if leading is None:
                leading = tuple(value.shape[:extra])
            adapted.append(value.reshape(-1, *value.shape[extra:]))

        result = call(*adapted)

        # Restore only when the result is actually per-row. A fused loss returns
        # a scalar; unflattening that would be nonsense.
        if (
            leading is not None
            and torch.is_tensor(result)
            and output_rank
            and result.dim() == output_rank
            and result.shape[0] == math.prod(leading)
        ):
            return result.reshape(*leading, *result.shape[1:])
        return result

    return flattened



# ── providers and the run ────────────────────────────────────────────────────

#: Which declared operator each patchable site corresponds to. The site is
#: named by the model; the operator is named by the declaration registry, and
#: an evolved program is only accepted for a site whose operator it implements.
SITE_OPS = {
    "rms_norm": "rmsnorm",
    "swiglu": "swiglu",
    "cross_entropy": "fused_linear_cross_entropy",
}


def patch(kernels: KernelSet, site: str, callable_: Callable) -> KernelSet:
    """Replace one site, recording that it was replaced."""
    if site not in SITE_OPS:
        raise ValueError(f"unknown patch site {site!r}; known: {sorted(SITE_OPS)}")
    return replace(
        kernels, **{site: callable_}, patched=tuple(sorted({*kernels.patched, site}))
    )


def restrict(kernels: KernelSet, sites: tuple[str, ...]) -> KernelSet:
    """Keep only the named sites patched; revert the rest to eager.

    Patching three sites at once and reading one number cannot say which site
    the number came from. Restricting to one at a time is how a blended result
    becomes an attribution.
    """
    unknown = set(sites) - set(SITE_OPS)
    if unknown:
        raise ValueError(f"unknown sites {sorted(unknown)}; known: {sorted(SITE_OPS)}")
    result = KernelSet()
    for site in kernels.patched:
        if site in sites:
            result = patch(result, site, getattr(kernels, site))
    return result


def patched_kernels(candidates: dict[str, Any], ops: dict[str, OpDecl]) -> KernelSet:
    """Build a kernel set from ``{site: candidate module}``.

    Each candidate is bound through ``opdecl.bind``, so the model reaches an
    evolved kernel by exactly the path a deployment would: a
    ``torch.autograd.Function`` built from the declaration, not a hand-wired
    call.
    """
    kernels = KernelSet()
    for site, module in candidates.items():
        op = ops[SITE_OPS[site]]
        kernels = patch(kernels, site, kernel_from_pair(op, module))
    return kernels



@dataclass(frozen=True)
class ModulePatch:
    """Replace every submodule matching ``match`` with ``build(original, kernel)``.

    The route for a model you did not write. ``match`` identifies the sites --
    usually ``lambda m: type(m).__name__ == "LlamaRMSNorm"`` -- and ``build``
    returns the replacement, which is responsible for carrying the original's
    trained weights across. Getting that wrong produces a model that runs and
    has been silently reinitialized, so the replacement receives the original
    rather than just its shapes.
    """

    site: str
    match: Callable[[nn.Module], bool]
    build: Callable[[nn.Module, Callable], nn.Module]


def patch_modules(
    model: nn.Module, patches: tuple[ModulePatch, ...], kernels: KernelSet
) -> list[str]:
    """Swap matching submodules in a built tree. Returns the paths replaced.

    Only sites whose kernel the provider actually replaced are touched: an
    eager :class:`KernelSet` leaves the model alone, so the unpatched provider
    is the original model rather than a rebuilt lookalike.
    """
    replaced: list[str] = []
    for patch_spec in patches:
        if patch_spec.site not in kernels.patched:
            continue
        kernel = getattr(kernels, patch_spec.site)
        for path, submodule in list(model.named_modules()):
            if not patch_spec.match(submodule) or not path:
                continue
            parent_path, _, attribute = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, attribute, patch_spec.build(submodule, kernel))
            replaced.append(path)
    return replaced



def eager_pair_for(op: OpDecl):
    """The declared eager kernel, presented as a forward/backward pair.

    Not a baseline — a *control*. Patching a site with this changes none of the
    mathematics and all of the plumbing: the same ``F.rms_norm`` now arrives
    through ``bind``, a Python ``autograd.Function``, and the rank adapter,
    exactly as an evolved kernel does. Whatever that costs is the harness tax,
    and it is the number standing between "Liger is slower" and "the patched
    path is slower".

    The backward recomputes the forward and differentiates it, where a real
    candidate's backward is a kernel. So this reads as an *upper bound* on the
    tax — it pays one extra forward per site that nothing else pays. If it
    lands near the unpatched eager number the plumbing is free and a slowdown
    is the kernels'; if it lands near the patched number, the plumbing is the
    story; in between, both contribute and ``--sites`` splits it further.
    """
    reference = resolve_runtime_forward(op)
    arg_names = tuple(arg.name for arg in op.args)
    active_names = tuple(arg.name for arg in op.active_args())
    by_grad = {arg.grad_name: arg.name for arg in op.active_args()}

    def forward_with_saved(*args):
        # Save the raw inputs, not a graph. Building the graph here and
        # differentiating it in the backward is the obvious construction and it
        # does not survive contact with `autograd.Function`: forward runs with
        # grad mode disabled, and its returned output is detached, so the graph
        # either never exists or does not reach the backward. Recomputing under
        # `enable_grad` in the backward is the checkpointing pattern and has
        # neither problem.
        with torch.no_grad():
            output = reference(*args)
        return output, tuple(args)

    def backward_from_saved(dout, saved, **_kwargs):
        values = dict(zip(arg_names, saved))
        prepared, leaves = [], []
        for name in arg_names:
            value = values[name]
            if name in active_names and torch.is_tensor(value):
                value = value.detach().requires_grad_(True)
                leaves.append(value)
            prepared.append(value)
        # `enable_grad` because a backward also runs with grad mode off.
        with torch.enable_grad():
            output = reference(*prepared)
        gradients = torch.autograd.grad(output, leaves, dout)
        by_name = dict(zip(active_names, gradients))
        return tuple(by_name[by_grad[name]] for name in op.grad_names())

    module = SimpleNamespace(
        forward_with_saved=forward_with_saved,
        backward_from_saved=backward_from_saved,
    )
    module.__name__ = f"<eager pair control for {op.name}>"
    return module


def identity_control_kernels(
    ops: dict[str, OpDecl], sites: tuple[str, ...] | None = None
) -> KernelSet:
    """Every site patched with the eager kernel it already had.

    The tier-3 analogue of ``--identity-control`` at tiers 1 and 2. There the
    control is one provider timed against itself, because those tiers compare
    two providers in symmetric slots. Here the asymmetry is not the slot, it is
    that a patched model routes through machinery an unpatched one never
    touches — so the control has to be the same mathematics on both sides of
    that machinery, not the same provider in both slots.
    """
    kernels = KernelSet()
    for site in sites or tuple(SITE_OPS):
        op = ops[SITE_OPS[site]]
        kernels = patch(kernels, site, kernel_from_pair(op, eager_pair_for(op)))
    return kernels


