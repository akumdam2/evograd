"""Operator scope: one declared operator as an ``nn.Module``, through real autograd.

The module is tier 2's thin parameter holder (``NativeModule``): the
declaration's ``parameter_args`` are ``nn.Parameter`` state, every other
Active argument is a call argument, inactive tensors are buffers. What it calls
is the deployment:

* native     -- ``resolve_runtime_forward(op)``, PyTorch eager autograd (for a
                model-reference declaration, the model's own computation --
                the same callable the Gate 1 oracle differentiates);
* candidate  -- the artifact's own ``DEPLOYMENT_ENTRY`` (validated against the
                artifact contract, never silently demoted), or ``bind(op, pair)``
                for a legacy pair;
* control    -- the native callable under ``torch.compile``.

Reused from the old tree: input recipes, ``bind``, the artifact contract and
the module wrapper. Not reused: ``check_module`` and its oracle comparison.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch

from evograd.opdecl.activity import OpDecl, Workload
from evograd.verification.common import contracts as C
from evograd.verification.common.execution import CANDIDATE, Deployment, Invocation, Sample, SiteHooks
from evograd.verification.common.tensors import isolate
from evograd.verification.gate2.adapters.base import ScopeBase, seeded_samples
from evograd.verification.gate2.runner import OutputStructureError

SITE = "operator"


class OperatorScope(ScopeBase):
    scope = "operator"

    def __init__(self, op: OpDecl, workload: Workload, *, device: str = "cuda",
                 autocast_dtype: str | None = None):
        if op.parameter_args is None:
            raise ValueError(f"{op.name}: the operator scope needs `parameter_args` declared "
                             "(which Active args are module state; `()` for none)")
        self.op = op
        self.workload = workload
        self.device = device
        self.autocast_dtype = autocast_dtype
        self._values: dict[str, dict[str, Any]] = {}

    # identity ---------------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        from evograd.opdecl.oracle import reference_identity

        # `reference` names the contract (model reference vs declared
        # mathematics) and, for a model reference, its pinned source; a policy
        # calibrated under one contract is a binding mismatch under the other.
        return {"scope": self.scope, "op": self.op.name, "forward": self.op.forward,
                "runtime_forward": self.op.runtime_forward, "declaration": self.op.declaration,
                "reference": reference_identity(self.op),
                "parameter_args": list(self.op.parameter_args or ()),
                "dims": dict(self.workload.dims), "dtype": self.workload.dtype}

    def numerics(self) -> dict[str, Any]:
        return {**self.base_numerics(), "dtype": self.workload.dtype}

    def reference_parts(self) -> dict[str, Any]:
        from evograd.evaluation.tier2.runner import NativeModule
        from evograd.opdecl.inputs import make_case_inputs
        from evograd.verification.gate1.adapters import declared_values

        return {"runtime_forward": self.native_site(SITE), "module": NativeModule,
                "inputs": self.op.make_inputs or make_case_inputs, "sampling": declared_values}

    def sites(self) -> tuple[str, ...]:
        return (SITE,)

    def native_site(self, site: str) -> Callable:
        from evograd.opdecl.oracle import resolve_runtime_forward

        if site != SITE:
            raise KeyError(site)
        return resolve_runtime_forward(self.op)

    def samples(self, seeds: Iterable[int]) -> list:
        """Declared inputs at shifted generator seeds (seed 0 is the declared case)."""
        dims = ",".join(f"{k}={v}" for k, v in sorted(self.workload.dims.items()))
        return seeded_samples(f"{self.op.name}/{dims}/{self.workload.dtype}", seeds,
                              generator="declaration inputs, manual_seed shifted by seed")

    def candidate(self, artifact: Any, *, name: str = "candidate") -> Deployment:
        """A generated artifact as a deployment: its own entry, or the binder."""
        from evograd.opdecl.bind import bind
        from evograd.pipelines.shared.artifact import deployment_entry, validate_artifact

        entry = deployment_entry(artifact)
        if entry is not None:
            validate_artifact(self.op, artifact)
            call, kind = entry, "direct_deployment_entry"
        else:
            call, kind = bind(self.op, artifact), "legacy_bind_pair"
        return Deployment(name, CANDIDATE, {SITE: call},
                          {"adapter_kind": kind,
                           "module": getattr(artifact, "__name__", repr(artifact))})

    # execution --------------------------------------------------------------

    def values(self, sample: Sample) -> dict[str, Any]:
        if sample.sample_id not in self._values:
            from evograd.verification.gate1.adapters import declared_values

            self._values[sample.sample_id] = declared_values(
                self.op, self.workload, device=self.device, seed_offset=int(sample.seed))
        return self._values[sample.sample_id]

    def build(self, deployment: Deployment, sample: Sample, hooks: SiteHooks) -> torch.nn.Module:
        from evograd.evaluation.tier2.runner import native_module

        if SITE in deployment.sites:
            call = hooks.wrap(SITE, deployment.sites[SITE])
        else:
            call = self.native_site(SITE)
        # The module registers inactive tensors as buffers by reference; give it
        # its own copy so nothing it does reaches the sample's cached inputs.
        module = native_module(self.op, call, values=isolate(self.values(sample)),
                               adapter_kind=f"verification_{deployment.role}")
        return module.train(self.training)

    def expected_invocations(self, deployment: Deployment) -> dict[str, int]:
        return {SITE: 1} if SITE in deployment.sites else {}

    def invocation(self, sample: Sample, module: torch.nn.Module) -> Invocation:
        values = self.values(sample)
        names = list(module._activation_names)
        # The declaration's overwrite allowance, for every candidate alike:
        # contents of these inputs (and upstream gradients) may be replaced by
        # the backward; their layout may not.
        allowed = set(self.op.backward_may_overwrite)
        mutable = tuple(f"args[{i}]" for i, n in enumerate(names) if n in allowed)
        mutable += tuple(f"cotangent:{o.name}" for o in self.op.outputs if o.grad_name in allowed)
        return Invocation(args=tuple(values[n] for n in names),
                          differentiable_inputs={n: f"args[{i}]" for i, n in enumerate(names)},
                          mutable_inputs=mutable)

    def output_specs(self, invocation: Invocation) -> tuple[C.ResultSpec, ...]:
        return tuple(C.ResultSpec(C.PREFIX[C.OUTPUT] + n, C.OUTPUT) for n in self.op.output_names)

    def outputs(self, result: Any, invocation: Invocation, sample: Sample) -> dict[str, Any]:
        names = self.op.output_names
        if self.op.is_multi_output:
            if torch.is_tensor(result) or not isinstance(result, (tuple, list)) \
                    or len(result) != len(names):
                raise OutputStructureError(f"{self.op.name}: expected outputs {list(names)}")
            return dict(zip(names, result))
        if not torch.is_tensor(result):
            raise OutputStructureError(f"{self.op.name}: expected one tensor {names[0]!r}, "
                                       f"got {type(result).__name__}")
        return {names[0]: result}

    def cotangents(self, outputs: Mapping[str, Any], invocation: Invocation,
                   sample: Sample) -> dict[str, torch.Tensor]:
        values = self.values(sample)
        return {out.name: values[out.grad_name] for out in self.op.outputs}
