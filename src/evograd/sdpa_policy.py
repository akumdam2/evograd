"""Which scaled-dot-product-attention behaviour every evograd process runs under.

**The hazard (torch 2.11.0+cu128, cuDNN 9.19, GH200; measured 2026-10-10/11).**
PyTorch's default SDPA backend on Hopper is cuDNN, and its backward caches an
execution plan per problem without keying on the upstream gradient's (``dO``)
strides. In one process, once a shape has run with one ``dO`` layout -- the
transposed view a Transformers attention module hands back, or a contiguous
tensor -- a later backward at the same shape with the other layout returns
gradients with rel_l2 ~1.4 against an fp32 reference, silently. Both orders,
every attention shape in this repository (D 64/128/256, GQA 2-8). q/k/v layouts
are *not* affected; flash and math are not affected. Reproducers:
``scripts/qwen3_5_4b/probes/{cudnn_sdpa_plan_cache,sdpa_policy,sdpa_guard}.py``.

A benchmark process mixes layouts at a shape all the time: a Level-1 attention
task's cotangent beside a Level-2 site's, a compiled provider beside eager, a
reference beside a candidate. A reference gradient computed second is then wrong.

**The policy** (``EVOGRAD_SDPA_POLICY``, default ``cudnn_guarded``):

``cudnn_guarded``  keep PyTorch's backend choice and make ``dO`` contiguous
                   before SDPA's backward (an identity autograd node after the
                   call). cuDNN then only ever sees one ``dO`` layout. Measured
                   correct in both orders at every shape; costs ~9% of an
                   attention forward+backward (one copy of ``dO``), against
                   1.5-1.8x for disabling cuDNN -- so attention baselines stay
                   PyTorch's fastest, as the benchmark's baseline rule requires.
``no_cudnn``       ``torch.backends.cuda.enable_cudnn_sdp(False)``: flash (or
                   math) instead. Correct, simplest, slower baseline.
``torch_default``  touch nothing: reproduces pre-2026-10-11 behaviour, hazard
                   included. For reproducing old numbers only.

Neither fixes cuDNN's (or flash's) *nondeterministic* backward. Bitwise
checks run under ``torch.use_deterministic_algorithms(True)``, which removes
cuDNN from SDPA's candidates altogether (flash's deterministic backward runs).

**Installation.** ``import evograd`` calls :func:`install`, which imports no
torch: it applies the policy at once if torch is already loaded, and otherwise
the moment torch's import finishes. Every harness process and child imports
``evograd`` first, so every SDPA call they make is covered. What ran is
reported by :func:`describe`, and the tier-3 environment fingerprint binds it.
Code that captured ``F.scaled_dot_product_attention`` by value *before* torch
finished importing would bypass the guard; nothing in this repository does.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import os
import sys
from typing import Any

POLICY_ENV = "EVOGRAD_SDPA_POLICY"
POLICIES = ("cudnn_guarded", "no_cudnn", "torch_default")
DEFAULT_POLICY = "cudnn_guarded"
#: Set on the installed wrapper so a second ``apply`` (or a reader) can tell.
GUARD_ATTRIBUTE = "_evograd_sdpa_guard"

_applied: str | None = None


def policy() -> str:
    """The policy this process is asked for (validated)."""
    value = os.environ.get(POLICY_ENV, DEFAULT_POLICY).strip() or DEFAULT_POLICY
    if value not in POLICIES:
        raise ValueError(f"{POLICY_ENV}={value!r}; expected one of {POLICIES}")
    return value


def _guarded(original, torch):
    class ContiguousUpstreamGrad(torch.autograd.Function):
        """Identity forward; backward hands SDPA a contiguous ``dO``."""

        @staticmethod
        def forward(ctx, out):
            return out.view_as(out)

        @staticmethod
        def backward(ctx, grad):
            return grad.contiguous()

    @functools.wraps(original)
    def scaled_dot_product_attention(*args, **kwargs):
        out = original(*args, **kwargs)
        if torch.is_grad_enabled() and out.requires_grad:
            out = ContiguousUpstreamGrad.apply(out)
        return out

    setattr(scaled_dot_product_attention, GUARD_ATTRIBUTE, True)
    scaled_dot_product_attention.__wrapped__ = original
    return scaled_dot_product_attention


def guard_installed() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    return bool(getattr(torch.nn.functional.scaled_dot_product_attention, GUARD_ATTRIBUTE, False))


def apply(torch=None) -> str:
    """Put this process under the requested policy. Idempotent."""
    global _applied
    if torch is None:
        import torch
    chosen = policy()
    functional = torch.nn.functional
    if chosen == "cudnn_guarded" and not guard_installed():
        functional.scaled_dot_product_attention = _guarded(
            functional.scaled_dot_product_attention, torch)
    elif chosen == "no_cudnn":
        torch.backends.cuda.enable_cudnn_sdp(False)
    _applied = chosen
    return chosen


def describe() -> dict[str, Any]:
    """What this process actually runs under, read back rather than assumed."""
    torch = sys.modules.get("torch")
    out: dict[str, Any] = {"policy": _applied, "requested": policy(), "guard": guard_installed()}
    if torch is not None:
        backend = torch.backends.cuda
        out["enabled"] = {
            name: bool(getattr(backend, f"{name}_sdp_enabled")())
            for name in ("cudnn", "flash", "mem_efficient", "math")
            if hasattr(backend, f"{name}_sdp_enabled")
        }
    return out


class _AfterTorchImport(importlib.abc.MetaPathFinder):
    """Applies the policy once ``import torch`` has finished, then steps aside."""

    def find_spec(self, name, path, target=None):
        if name != "torch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec("torch")
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_then_apply(module):
            exec_module(module)
            apply(module)

        spec.loader.exec_module = exec_then_apply
        return spec


def install() -> None:
    """Apply now if torch is loaded; otherwise as soon as it is. Never imports torch."""
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "nn") and hasattr(torch, "backends"):
        apply(torch)
        return
    if not any(isinstance(f, _AfterTorchImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _AfterTorchImport())


__all__ = ["DEFAULT_POLICY", "GUARD_ATTRIBUTE", "POLICIES", "POLICY_ENV", "apply", "describe",
           "guard_installed", "install", "policy"]
