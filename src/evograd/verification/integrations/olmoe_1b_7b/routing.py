"""OLMoE's model-scope Gate 2 with its routing held fixed.

Routed experts make top-k choices, and a tiny numerical difference upstream of a
router can flip one. Model Gate 2 compares every output and parameter gradient,
so a correct kernel that flips one token's experts would look like a large error.
So every run of a sample routes as the native model does on that sample: the router
still computes its probabilities and gets its gradient, but the experts are the ones
the native run chose. Only the choice is fixed, and only here. The loss includes
the 0.01 load-balancing term, which counts each expert's tokens from its own top-k
of the live router logits (HF ``load_balancing_loss_func``); that count is not
fixed, so a flip there moves the term and the router gradients' share from it,
never which experts run. This is a known limitation, left open.

Gate 3 keeps free routing. It judges the loss curve against thresholds taken from
native and compile runs, and compile's rounding flips routing as a candidate's does,
so its thresholds already hold that spread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from evograd.evaluation.tier3.patch import KernelSet
from evograd.verification.common.tensors import isolated_rng
from evograd.verification.gate2.adapters.model import WorkloadModelScope

from ..objectives import NextTokenLM

ROUTER = "OlmoeTopKRouter"
RULE = ("each run of a sample routes tokens as the native model does on that sample: "
        "the router's probabilities and gradient are live, its top-k choice is the native run's")


def routers(model) -> list[tuple[str, torch.nn.Module]]:
    """Every router of ``model``, by module path."""
    return [(path, module) for path, module in model.named_modules()
            if type(module).__name__ == ROUTER]


def forced_outputs(router, logits, indices):
    """``OlmoeTopKRouter.forward`` with its top-k choice given.

    The same arithmetic on the same logits, so where ``indices`` is what the router
    would have chosen, the result is what it would have returned, bit for bit.
    """
    probabilities = F.softmax(logits, dtype=torch.float, dim=-1)
    values = probabilities.gather(-1, indices)
    if router.norm_topk_prob:
        values = values / values.sum(dim=-1, keepdim=True)
    return logits, values.to(logits.dtype), indices


def force(model, route: dict[str, torch.Tensor]) -> list:
    """Hold each router of ``model`` to ``route``; returns the hook handles."""
    handles = []
    for path, router in routers(model):
        indices = route[path]

        def hook(module, _args, output, indices=indices):
            return forced_outputs(module, output[0], indices)

        handles.append(router.register_forward_hook(hook))
    return handles


class ForcedRoutingScope(WorkloadModelScope):
    """The model scope, with every run of a sample on the native run's routing."""

    def __init__(self, workload: Any, *, native_sites=None, replaced=None, patch_set=()):
        super().__init__(workload, native_sites=native_sites, replaced=replaced,
                         patch_set=patch_set)
        self._routes: dict[int, dict[str, torch.Tensor]] = {}

    def identity(self) -> dict[str, Any]:
        return {**super().identity(), "routing": RULE}

    def reference_parts(self) -> dict[str, Any]:
        return {**super().reference_parts(), "forced_outputs": forced_outputs}

    def route(self, sample) -> dict[str, torch.Tensor]:
        """The native model's top-k choices on ``sample``, recorded once.

        Isolated, because a build runs inside the caller's seeded block, and this
        one happens only on a sample's first build.
        """
        seed = int(sample.seed)
        if seed not in self._routes:
            seen: dict[str, torch.Tensor] = {}
            with isolated_rng(seed, self.device):
                model, _ = self.workload.build_patched(KernelSet(registry=self.registry))
                model = model.train(self.training)
                for path, router in routers(model):
                    router.register_forward_hook(
                        lambda _m, _a, out, path=path: seen.__setitem__(path, out[2].detach().clone()))
                with torch.no_grad(), self.numerics_context():
                    model(**self.invocation(sample, model).kwargs)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._routes[seed] = seen
        return self._routes[seed]

    def build(self, deployment, sample, hooks):
        route = self.route(sample)  # first: it rebuilds, and the expected call counts are the last build's
        model = super().build(deployment, sample, hooks)
        force(model, route)
        return model


@dataclass(frozen=True)
class ForcedRoutingLM(NextTokenLM):
    """The causal-LM objective, with model Gate 2 on fixed routing."""

    def model_scope(self, workload, native_sites, replaced=None, patch_set=()):
        return ForcedRoutingScope(workload, native_sites=native_sites, replaced=replaced,
                                  patch_set=patch_set)
