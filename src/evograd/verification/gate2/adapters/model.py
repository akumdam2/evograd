"""Model scope: a whole model, its declared outputs and loss, one backward.

:func:`model_scope` makes a :class:`ModuleScope` for any model built from named
sites: outputs are what ``select_outputs`` declares plus ``loss``, and the
backward is seeded with ``d loss = 1`` only -- the same as ``loss.backward()``.

:class:`WorkloadModelScope` bridges an existing tier-3 ``TrainingWorkload``
(Qwen3-0.6B, Llama-3.2-1B): the workload builds the model (``build_patched``,
random initialization from its spec, no download) and the batch
(``batch_for(seed=...)``); this adapter installs each replaced site's callable
through a ``KernelSet`` and declares ``logits`` and the token-mean next-token
NLL as the outputs. The old model gates (envelope, KL, global gradient,
protocol 4) are not used.

Model scope uses the ``matched_inputs`` sample protocol: a policy is calibrated
on fixed samples and judges fresh executions of those same samples, matched by
content (see :mod:`evograd.verification.gate2.calibration`).

Not covered: Qwen3-Next, or any model with routed experts. Its routing would
need a fixed-routing execution condition this bridge does not define, so no
such workload is wired here.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import Deployment, Invocation, Sample, SiteHooks
from evograd.verification.gate2.adapters.base import ScopeBase, seeded_samples
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.runner import OutputStructureError

IGNORE_INDEX = -100


def model_scope(*, name: str, factory: Callable[[Mapping[str, Callable]], torch.nn.Module],
                native_sites: Mapping[str, Callable],
                make_invocation: Callable[[Sample, str], Invocation],
                forward_outputs: Callable[[Any, Invocation, Sample], Mapping[str, Any]],
                loss: Callable[[Mapping[str, Any], Invocation, Sample], torch.Tensor],
                output_names: tuple[str, ...], loss_definition: str,
                calls_per_execution: Mapping[str, int] | None = None,
                config: Mapping[str, Any] | None = None, dtype: str = "float32",
                device: str = "cpu", autocast_dtype: str | None = None,
                training: bool = True) -> ModuleScope:
    """A model scope. ``loss`` is computed from the declared outputs; it is an
    output itself (compared like any other) and the only backward seed."""
    if "loss" in output_names:
        raise ValueError("`loss` is added by the model scope; do not declare it")

    def select(result, invocation, sample):
        outputs = dict(forward_outputs(result, invocation, sample))
        outputs["loss"] = loss(outputs, invocation, sample)
        return outputs

    def seed(outputs, invocation, sample):
        value = outputs["loss"]
        if not (torch.is_tensor(value) and value.dim() == 0):
            raise OutputStructureError("the model loss must be a scalar tensor")
        return {"loss": torch.ones_like(value)}

    return ModuleScope(
        scope="model", name=name, factory=factory, native_sites=native_sites,
        make_invocation=make_invocation, select_outputs=select,
        output_names=tuple(output_names) + ("loss",), make_cotangents=seed,
        calls_per_execution=dict(calls_per_execution or {}),
        config={**dict(config or {}), "loss": loss_definition}, dtype=dtype, device=device,
        autocast_dtype=autocast_dtype, training=training)


def next_token_mean_nll(logits: torch.Tensor, labels: torch.Tensor, *,
                        ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Sum of ``-log P(label_{t+1})`` over valid positions / their count, in float32."""
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    count = int((targets != ignore_index).sum())
    if count == 0:
        raise ValueError("a batch with no valid next-token positions")
    total = torch.nn.functional.cross_entropy(shifted.reshape(-1, shifted.shape[-1]),
                                              targets.reshape(-1), ignore_index=ignore_index,
                                              reduction="sum")
    return total / count


class WorkloadModelScope(ScopeBase):
    """A tier-3 ``TrainingWorkload`` at model scope. See the module docstring."""

    scope = "model"

    def __init__(self, workload: Any, *, native_sites: Mapping[str, Callable] | None = None,
                 replaced: Mapping[str, Any] | None = None):
        self.workload = workload
        spec = getattr(workload, "spec", None)
        self.device = str(getattr(spec, "device", "cpu"))
        self.dtype = str(getattr(spec, "dtype", "float32"))
        self.registry = workload.site_registry
        self._native = dict(native_sites or {})
        #: Site -> the configurations (workloads) a deployment replaces there,
        #: where it takes only some of the site's calls; see :meth:`control_site`.
        self._replaced = {site: tuple(ws) for site, ws in (replaced or {}).items()}
        self._expected: dict[str, int] = {}

    def identity(self) -> dict[str, Any]:
        return {"scope": self.scope, "bridge": "tier3.TrainingWorkload",
                "workload": dict(self.workload.describe()),
                "outputs": ["logits", "loss"],
                "loss": "token-mean next-token NLL, float32, ignore_index=-100",
                "site_references": self.site_references(), **self.replaced_identity()}

    def replaced_identity(self) -> dict[str, Any]:
        """The replaced configurations, bound into the policy (nothing when whole sites are)."""
        if not self._replaced:
            return {}
        return {"replaced_configurations": {
            site: [{"dims": dict(w.dims), "dtype": w.dtype,
                    "strides": [[n, list(st)] for n, st in
                                (getattr(getattr(w, "provenance", None), "strides", None) or ())]}
                   for w in workloads]
            for site, workloads in sorted(self._replaced.items())}}

    def control_site(self, site: str, call: Callable) -> Callable:
        """``call`` (a control's implementation of ``site``) installed where the
        deployment replaces: the whole site, or only its replaced configurations
        with the native on every other call."""
        if site not in self._replaced:
            return call
        from evograd.benchmark import get_task
        from evograd.verification.common import dispatch

        op = get_task(self.registry.require(site).op)
        return dispatch.by_configuration(
            op, {dispatch.configuration(op, w): call for w in self._replaced[site]},
            self.native_site(site))

    def site_references(self) -> dict[str, Any]:
        """Each site's reference contract -- what boundary replay judges against.

        Bound into the policy, so a policy calibrated when a site's declared
        reference was a different computation (e.g. the residual site's former
        ``F.rms_norm`` contract) is a binding mismatch, never reinterpreted.
        Sites given an explicit ``native_sites`` callable are recorded as such.
        """
        from evograd.benchmark import get_task
        from evograd.opdecl.oracle import reference_identity

        references = {}
        for site in self.sites():
            if site in self._native:
                references[site] = {"kind": "explicit_native_site"}
            else:
                references[site] = {"task": self.registry.require(site).op,
                                    **reference_identity(get_task(self.registry.require(site).op))}
        return references

    def numerics(self) -> dict[str, Any]:
        return {**self.base_numerics(), "dtype": self.dtype}

    def reference_parts(self) -> dict[str, Any]:
        return {"workload_class": type(self.workload), "loss": next_token_mean_nll,
                "native_sites": {site: self.native_site(site) for site in self.sites()}}

    def sites(self) -> tuple[str, ...]:
        return tuple(self.registry.names)

    def native_site(self, site: str) -> Callable:
        """The site's declared operator in its declared ABI (how replacements are called)."""
        if site in self._native:
            return self._native[site]
        from evograd.benchmark import get_task
        from evograd.opdecl.oracle import resolve_runtime_forward

        return resolve_runtime_forward(get_task(self.registry.require(site).op))

    def samples(self, seeds) -> list[Sample]:
        return seeded_samples(f"{self.workload.name}/model", seeds,
                              generator="workload.batch_for(seed=sample.seed)")

    def build(self, deployment: Deployment, sample: Sample, hooks: SiteHooks) -> torch.nn.Module:
        from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

        kernels = KernelSet(registry=self.registry)
        for site, fn in deployment.sites.items():
            kernels = patch(kernels, site, hooks.wrap(site, fn), source=KernelSource(
                site, self.registry.require(site).op, None, f"verification:{deployment.name}"))
        model, _provenance = self.workload.build_patched(kernels)
        last = getattr(self.workload, "last_build", None)
        expected = dict(getattr(last, "expected", {}) or {})
        self._expected = {site: int(expected.get(site, 0)) for site in deployment.sites}
        return model.train(self.training)

    def expected_invocations(self, deployment: Deployment) -> dict[str, int]:
        return dict(self._expected) if deployment.sites else {}

    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation:
        input_ids, labels = self.workload.batch_for(seed=int(sample.seed))
        return Invocation(kwargs={"input_ids": input_ids, "use_cache": False},
                          context={"labels": labels})

    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]:
        return (C.ResultSpec("out:logits", C.OUTPUT), C.ResultSpec("out:loss", C.OUTPUT))

    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> dict[str, Any]:
        logits = getattr(result, "logits", None)
        if not torch.is_tensor(logits):
            raise OutputStructureError("the model returned no logits")
        return {"logits": logits,
                "loss": next_token_mean_nll(logits, invocation.context["labels"])}

    def cotangents(self, outputs: Mapping[str, Any], invocation: Invocation,
                   sample: Sample) -> dict[str, torch.Tensor]:
        return {"loss": torch.ones_like(outputs["loss"])}


class _PrimaryOutputModule(torch.nn.Module):
    """``model`` behind the workload's own step: ``forward(batch) -> (loss, primary)``."""

    def __init__(self, workload: Any, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self._workload = workload           # a plain attribute, not a submodule

    def forward(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        loss, primary = self._workload.primary_output(self.model, batch)
        return loss, primary


class ObjectiveModelScope(WorkloadModelScope):
    """A tier-3 ``TrainingWorkload`` whose objective is its own (not next-token NLL).

    For classification, diffusion and recommendation models (ConvNeXt V2,
    Qwen2-VL, DiT, DLRM). The workload supplies, besides the tier-3 protocol:

    * ``primary_output(model, batch) -> (loss, primary)``: the scalar its
      training step differentiates (equal bit for bit to ``loss(model, batch)``)
      and one model output beside it (class logits, the diffusion prediction,
      the click logit), named ``primary_output_name``;
    * ``objective_definition``: what that loss is, in words (bound into the policy);
    * ``expected_site_calls() -> {site: calls}``: calls per forward and backward.

    The outputs compared are ``<primary>`` and ``loss``; every trainable
    parameter's gradient is compared too (see ``integration_contract``). The
    backward is seeded with ``d loss = 1``, as training does. The batch is the
    module's one keyword argument, so its every tensor is content-identified
    and checked for mutation. Parameter names carry the wrapper's ``model.``
    prefix.
    """

    def __init__(self, workload: Any, *, native_sites: Mapping[str, Callable] | None = None,
                 replaced: Mapping[str, Any] | None = None):
        super().__init__(workload, native_sites=native_sites, replaced=replaced)
        self.primary = str(workload.primary_output_name)

    def identity(self) -> dict[str, Any]:
        return {"scope": self.scope, "bridge": "tier3.TrainingWorkload.primary_output",
                "workload": dict(self.workload.describe()),
                "outputs": [self.primary, "loss"],
                "loss": str(self.workload.objective_definition),
                "site_references": self.site_references(), **self.replaced_identity()}

    def reference_parts(self) -> dict[str, Any]:
        return {"workload_class": type(self.workload),
                "primary_output": type(self.workload).primary_output,
                "native_sites": {site: self.native_site(site) for site in self.sites()}}

    def build(self, deployment: Deployment, sample: Sample, hooks: SiteHooks) -> torch.nn.Module:
        from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

        kernels = KernelSet(registry=self.registry)
        for site, fn in deployment.sites.items():
            kernels = patch(kernels, site, hooks.wrap(site, fn), source=KernelSource(
                site, self.registry.require(site).op, None, f"verification:{deployment.name}"))
        model, _provenance = self.workload.build_patched(kernels)
        expected = dict(self.workload.expected_site_calls())
        self._expected = {site: int(expected[site]) for site in deployment.sites}
        return _PrimaryOutputModule(self.workload, model).train(self.training)

    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation:
        return Invocation(kwargs={"batch": dict(self.workload.batch_for(seed=int(sample.seed)))})

    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]:
        return (C.ResultSpec(f"out:{self.primary}", C.OUTPUT), C.ResultSpec("out:loss", C.OUTPUT))

    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> dict[str, Any]:
        if not (isinstance(result, tuple) and len(result) == 2):
            raise OutputStructureError("primary_output must return (loss, primary)")
        loss, primary = result
        if not (torch.is_tensor(loss) and loss.dim() == 0):
            raise OutputStructureError("the workload loss must be a scalar tensor")
        if not torch.is_tensor(primary):
            raise OutputStructureError(f"the primary output {self.primary!r} is not a tensor")
        return {self.primary: primary, "loss": loss}
