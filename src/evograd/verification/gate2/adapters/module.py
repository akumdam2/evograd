"""The generic module adapter: any model built from named sites, at model scope.

:class:`ModuleScope` is the whole adapter for any module whose replaceable
parts are named sites: a factory builds the module from a ``site -> callable``
map (same initial weights every call), and three small functions say how to
call it, which outputs it declares, and which cotangents seed the backward.
Gate 2's core does the rest. :func:`~evograd.verification.gate2.adapters.model.model_scope`
builds one with a scalar loss as the only backward seed; a small test network
with its own outputs and seeded cotangents is a model too. There is no block
scope: a sub-module is verified as an operator or inside the whole model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Invocation, Sample, SiteHooks
from evograd.verification.gate2.adapters.base import ScopeBase, seeded_cotangents
from evograd.verification.gate2.runner import OutputStructureError


@dataclass
class ModuleScope(ScopeBase):
    """A model built from named sites. See the module docstring."""

    scope: str
    name: str
    #: ``factory(sites) -> nn.Module``; ``sites`` maps every site to the
    #: callable to install. Must give identical initial weights on every call.
    factory: Callable[[Mapping[str, Callable]], torch.nn.Module]
    native_sites: Mapping[str, Callable]
    #: ``(sample, device) -> Invocation``.
    make_invocation: Callable[[Sample, str], Invocation]
    #: ``(result, invocation, sample) -> {output name: value}``.
    select_outputs: Callable[[Any, Invocation, Sample], Mapping[str, Any]]
    #: Output names, and which of them are exact (non-numeric) metadata.
    output_names: tuple[str, ...]
    metadata_outputs: tuple[str, ...] = ()
    #: ``(outputs, invocation, sample) -> {output name: cotangent}``. Default:
    #: seeded standard-normal cotangents for every floating output.
    make_cotangents: Callable[[Mapping[str, Any], Invocation, Sample], Mapping[str, torch.Tensor]] | None = None
    #: site -> calls per execution when that site is replaced.
    calls_per_execution: Mapping[str, int] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    dtype: str = "float32"
    device: str = "cpu"
    autocast_dtype: str | None = None
    training: bool = True

    def __post_init__(self) -> None:
        if self.scope != "model":
            raise ValueError(f"ModuleScope is a model-scope adapter; Gate 2 has no {self.scope!r} "
                             "scope (use scope='model')")
        unknown = set(self.calls_per_execution) - set(self.native_sites)
        if unknown:
            raise ValueError(f"call counts for unknown sites {sorted(unknown)}")

    def identity(self) -> dict[str, Any]:
        return {"scope": self.scope, "name": self.name, "config": dict(self.config),
                "sites": sorted(self.native_sites), "outputs": list(self.output_names),
                "metadata_outputs": list(self.metadata_outputs)}

    def numerics(self) -> dict[str, Any]:
        return {**self.base_numerics(), "dtype": self.dtype}

    def reference_parts(self) -> dict[str, Any]:
        return {"factory": self.factory, "native_sites": dict(self.native_sites),
                "make_invocation": self.make_invocation, "select_outputs": self.select_outputs,
                "make_cotangents": self.make_cotangents}

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted(self.native_sites))

    def native_site(self, site: str) -> Callable:
        return self.native_sites[site]

    def build(self, deployment: Deployment, sample: Sample, hooks: SiteHooks) -> torch.nn.Module:
        sites = dict(self.native_sites)
        for site, fn in deployment.sites.items():
            if site not in sites:
                raise KeyError(f"{self.name}: no site {site!r}")
            sites[site] = hooks.wrap(site, fn)
        module = self.factory(sites).to(self.device)
        return module.train(self.training)

    def expected_invocations(self, deployment: Deployment) -> dict[str, int]:
        return {site: int(self.calls_per_execution.get(site, 1)) for site in deployment.sites}

    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation:
        return self.make_invocation(sample, self.device)

    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]:
        specs = []
        for name in self.output_names:
            if name in self.metadata_outputs:
                specs.append(C.ResultSpec(C.PREFIX[C.METADATA] + name, C.METADATA, exact=True))
            else:
                specs.append(C.ResultSpec(C.PREFIX[C.OUTPUT] + name, C.OUTPUT))
        return tuple(specs)

    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> dict[str, Any]:
        outputs = dict(self.select_outputs(result, invocation, sample))
        if set(outputs) != set(self.output_names):
            raise OutputStructureError(f"{self.name}: outputs {sorted(outputs)} != declared "
                                       f"{sorted(self.output_names)}")
        return outputs

    def cotangents(self, outputs: Mapping[str, Any], invocation: Invocation,
                   sample: Sample) -> Mapping[str, torch.Tensor]:
        if self.make_cotangents is not None:
            return self.make_cotangents(outputs, invocation, sample)
        floating = {k: v for k, v in outputs.items() if k not in self.metadata_outputs}
        return seeded_cotangents(floating, sample.seed)
