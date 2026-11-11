"""Model scope: a whole model, its declared outputs and loss, one backward.

:func:`model_scope` makes a :class:`ModuleScope` for any model built from named
sites: outputs are what ``select_outputs`` declares plus ``loss``, and the
backward is seeded with ``d loss = 1`` only -- the same as ``loss.backward()``.

:class:`WorkloadModelScope` bridges an existing tier-3 ``TrainingWorkload``
of a next-token language model (Qwen3-0.6B, Llama-3.2-1B, Qwen3.5-4B,
OLMoE-1B-7B): the workload builds the model (``build_patched``) and the batch
(``batch_for(seed=...)``); this adapter installs each replaced site's callable
through a ``KernelSet`` and calls the model as ``workload.loss`` does,
``model(input_ids=..., labels=labels, use_cache=False)``. The backward is the
``.loss`` it returns, seeded with 1 -- the scalar the tier-3 step
differentiates, with any auxiliary term the model adds (OLMoE's 0.01 router
load-balancing loss) -- so the parameter gradients compared are the ones
training applies. The forward is judged on the logits, which determine the
next-token loss; the loss only seeds the backward and is not a result (an
auxiliary term reaches the verdict through its gradients).

A patch set that replaces :data:`LM_HEAD_LOSS_SITE`, the LM head fused with the
cross-entropy, never materializes logits. There the forward is judged on the
LM head's input (the decoder's last hidden state, after the final norm) and on
the loss the site returns, beside every parameter gradient.
The old model gates (envelope, KL, global gradient, protocol 4) are not used.

Model scope uses the ``matched_inputs`` sample protocol: a policy is calibrated
on fixed samples and judges fresh executions of those same samples, matched by
content (see :mod:`evograd.verification.gate2.calibration`).

Routed experts need a fixed-routing execution condition this bridge does not
define; OLMoE adds it in a subclass (``integrations.olmoe_1b_7b.routing``).
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch

from evograd.verification.common import contracts as C
from evograd.verification.common.execution import (WORKLOAD_LM_LOSS, Deployment, Invocation,
                                                    Sample, SiteHooks)
from evograd.verification.gate2.adapters.base import ScopeBase, seeded_samples
from evograd.verification.gate2.adapters.module import ModuleScope
from evograd.verification.gate2.runner import OutputStructureError

#: The site every LM integration names for the LM head fused with the
#: cross-entropy (Qwen3-0.6B's ``lm_head_ce``): it returns the loss and no logits.
LM_HEAD_LOSS_SITE = "lm_head_ce"


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


class WorkloadModelScope(ScopeBase):
    """A tier-3 ``TrainingWorkload`` at model scope. See the module docstring."""

    scope = "model"

    def __init__(self, workload: Any, *, native_sites: Mapping[str, Callable] | None = None,
                 replaced: Mapping[str, Any] | None = None, patch_set: Iterable[str] = ()):
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
        #: Whether the patch set fuses the LM head with the loss (no logits).
        self.fused_head = LM_HEAD_LOSS_SITE in tuple(patch_set)
        self.output_names = ("lm_head_input", "loss") if self.fused_head else ("logits",)
        self.backward_only = () if self.fused_head else ("loss",)
        self._head_input: torch.Tensor | None = None

    def identity(self) -> dict[str, Any]:
        return {"scope": self.scope, "bridge": "tier3.TrainingWorkload.loss",
                "workload": dict(self.workload.describe()),
                "outputs": list(self.output_names), "backward_only": list(self.backward_only),
                "loss": WORKLOAD_LM_LOSS,
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
            op, {dispatch.configuration(op, w) for w in self.registry.require(site).preflight},
            {dispatch.configuration(op, w): call for w in self._replaced[site]},
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
        return {"workload_class": type(self.workload), "loss": type(self.workload).loss,
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
        if LM_HEAD_LOSS_SITE in deployment.sites and not self.fused_head:
            raise ValueError(f"{deployment.name} replaces {LM_HEAD_LOSS_SITE!r}, which returns "
                             "no logits; build the scope with that patch set")
        model, _provenance = self.workload.build_patched(kernels)
        last = getattr(self.workload, "last_build", None)
        expected = dict(getattr(last, "expected", {}) or {})
        self._expected = {site: int(expected.get(site, 0)) for site in deployment.sites}
        if self.fused_head:
            self._head_input = None
            model.get_decoder().register_forward_hook(self._keep_head_input)
        return model.train(self.training)

    def _keep_head_input(self, _module, _args, output) -> None:
        self._head_input = output.last_hidden_state

    def expected_invocations(self, deployment: Deployment) -> dict[str, int]:
        return dict(self._expected) if deployment.sites else {}

    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation:
        """The model called as ``workload.loss`` calls it, labels included."""
        input_ids, labels = self.workload.batch_for(seed=int(sample.seed))
        return Invocation(kwargs={"input_ids": input_ids, "labels": labels, "use_cache": False})

    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]:
        return tuple(C.ResultSpec(f"out:{name}", C.OUTPUT) for name in self.output_names)

    def structure(self, result: Any) -> Any:
        """The model's return without its logits: they are an output when the head
        is not fused, and a fused head does not return them (native does); every
        other field must be the native one's."""
        tree = super().structure(result)
        if isinstance(tree, dict):
            tree.pop("logits", None)
        return tree

    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> dict[str, Any]:
        loss = getattr(result, "loss", None)
        if not (torch.is_tensor(loss) and loss.dim() == 0):
            raise OutputStructureError("the model returned no scalar loss")
        if self.fused_head:
            head_input, self._head_input = self._head_input, None
            if not torch.is_tensor(head_input):
                raise OutputStructureError("the decoder produced no last hidden state")
            return {"lm_head_input": head_input, "loss": loss}
        logits = getattr(result, "logits", None)
        if not torch.is_tensor(logits):
            raise OutputStructureError("the model returned no logits")
        return {"logits": logits, "loss": loss}

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
        self.backward_only = ()         # the loss is a result here, beside the primary output

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
