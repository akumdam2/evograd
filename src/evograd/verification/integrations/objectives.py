"""How a model's training step is run, recorded and judged, by kind of objective.

A :class:`~evograd.verification.integrations.base.Model` names one of these:

* :class:`NextTokenLM` (the default) -- an HF causal LM. Batches are
  ``(input_ids, labels)``; the step is ``model(input_ids, labels=labels).loss``;
  Gate 2 model scope compares logits and the token-mean next-token NLL
  (``WorkloadModelScope``); Gate 3 trains with ``gate3.workload_training``.
  Every record, key and identity is exactly what the Qwen3 and Qwen3.5
  integrations have always produced.
* :class:`PrimaryOutput` -- any other workload with ``primary_output(model,
  batch) -> (loss, primary)``: classification logits, a diffusion prediction,
  a click logit. Batches are dicts of tensors; Gate 2 model scope compares
  the loss, the primary output and every parameter gradient
  (``ObjectiveModelScope``); Gate 3 trains with
  ``gate3.workload_objective_training``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping


def to_device(tree: Any, device: str) -> Any:
    """Every tensor in ``tree`` on ``device`` (containers rebuilt)."""
    import torch

    if torch.is_tensor(tree):
        return tree.to(device)
    if isinstance(tree, tuple):
        return tuple(to_device(v, device) for v in tree)
    if isinstance(tree, list):
        return [to_device(v, device) for v in tree]
    if isinstance(tree, Mapping):
        return {k: to_device(v, device) for k, v in tree.items()}
    return tree


def step_record(model, loss, primary, primary_name: str) -> dict[str, Any]:
    """Loss, the primary output and every parameter gradient, by content hash."""
    from evograd.verification.common.tensors import tensor_digest

    grads = {name: (tensor_digest(p.grad) if p.grad is not None else None)
             for name, p in model.named_parameters()}
    return {"loss": float(loss.detach()), "loss_digest": tensor_digest(loss.detach()),
            f"{primary_name}_digest": tensor_digest(primary.detach()), "grads": grads}


def generic_parameter_state(model) -> dict[str, Any]:
    """Parameter count, content digest and the tied parameter groups."""
    from evograd.verification.common import contracts as C
    from evograd.verification.common.tensors import tensor_digest

    digests = {name: tensor_digest(p) for name, p in model.named_parameters()}
    return {"parameters": len(digests),
            "tied": [list(g) for g in C.parameter_groups(model) if len(g) > 1],
            "state_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}


@dataclass(frozen=True)
class NextTokenLM:
    """An HF causal language model (the objective of every LM integration)."""

    primary: str = "logits"
    #: The key of a batch identity whose digest names a batch in payload records.
    batch_key: str = "input_ids_sha256"

    @property
    def loss_definition(self) -> str:
        from .base import LOSS_DEFINITION

        return LOSS_DEFINITION

    def capture_step(self, model, batch) -> tuple[Any, Any]:
        """One forward and backward; returns ``(loss, logits)``."""
        from .base import run_step

        ids, labels = batch
        output = run_step(model, ids, labels)
        return output.loss, output.logits

    def parameter_state(self, model) -> dict[str, Any]:
        from .base import parameter_state

        return parameter_state(model)

    def data_recipe(self, workload) -> dict[str, Any]:
        return {"batch_size": workload.batch_size, "seq_len": workload.seq_len}

    def model_scope(self, workload, native_sites: Mapping[str, Callable], replaced=None):
        from evograd.verification.gate2.adapters.model import WorkloadModelScope

        return WorkloadModelScope(workload, native_sites=native_sites, replaced=replaced)

    def training(self, workload, *, train_data, optimizer):
        from evograd.verification import gate3

        return gate3.workload_training(workload, train_data=train_data, optimizer=optimizer)


@dataclass(frozen=True)
class PrimaryOutput:
    """A workload with its own objective: ``workload.primary_output(model, batch)``.

    ``primary`` must equal the workload's ``primary_output_name``; batches are
    dicts of tensors, named by the ``batch_sha256`` of their content.
    """

    primary: str
    loss_definition: str
    batch_key: str = "batch_sha256"
    workload: Any = None

    def capture_step(self, model, batch) -> tuple[Any, Any]:
        if self.workload is None:
            raise ValueError("a PrimaryOutput objective needs its workload for capture")
        loss, primary = self.workload.primary_output(model, batch)
        loss.backward()
        return loss, primary

    def bind(self, workload) -> "PrimaryOutput":
        """This objective over a loaded workload (capture needs its step)."""
        if str(workload.primary_output_name) != self.primary:
            raise ValueError(f"workload primary output {workload.primary_output_name!r} "
                             f"is not {self.primary!r}")
        return PrimaryOutput(self.primary, self.loss_definition, self.batch_key, workload)

    def parameter_state(self, model) -> dict[str, Any]:
        return generic_parameter_state(model)

    def data_recipe(self, workload) -> dict[str, Any]:
        return dict(workload.data_recipe())

    def model_scope(self, workload, native_sites: Mapping[str, Callable], replaced=None):
        from evograd.verification.gate2.adapters.model import ObjectiveModelScope

        return ObjectiveModelScope(workload, native_sites=native_sites, replaced=replaced)

    def training(self, workload, *, train_data, optimizer):
        from evograd.verification import gate3

        return gate3.workload_objective_training(workload, train_data=train_data,
                                                 optimizer=optimizer)


def batch_digest(batch: Mapping[str, Any]) -> str:
    """Content hash of a dict batch: each tensor's digest, by key."""
    from evograd.verification.common.contracts import stable_hash
    from evograd.verification.common.tensors import tensor_digest

    return stable_hash({k: tensor_digest(v) for k, v in sorted(batch.items())})
