"""Gate 3 adapters: optimizers, the language-model objective, workload bridges.

The core (:func:`evograd.verification.gate3.runner.measure_training`) knows no
``input_ids``, ``.logits``, token shift, ``ignore_index`` or AdamW. They live
here, as one explicit adapter for next-token language models:

* :func:`adamw` -- the optimizer, with its full configuration written down;
* :func:`next_token_nll` -- ``(sum of -log P(label_{t+1}), valid tokens)`` in
  float32, the weight being the token count;
* :class:`LanguageModelTraining` -- model/evaluator builders, a training batch
  sequence and fixed held-out batches. Batches are ``(input_ids, labels)``;
  any object with ``batch(index)`` (and optionally ``describe()``) or a plain
  sequence serves;
* :func:`workload_training` -- the same over a tier-3 ``TrainingWorkload``
  (Qwen3, Llama): the workload builds the patched model and the trusted
  evaluator (``eager_evaluator``); this installs replaced sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from evograd.verification.common.contracts import stable_hash
from evograd.verification.common.execution import Deployment, SiteHooks
from evograd.verification.common.tensors import tensor_digest

IGNORE_INDEX = -100


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    config: Mapping[str, Any]
    factory: Callable[[Any], torch.optim.Optimizer]

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.factory(model.parameters())


def adamw(*, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
          weight_decay: float = 0.01) -> OptimizerSpec:
    """``torch.optim.AdamW`` with every setting stated (constant learning rate)."""
    config = {"optimizer": "AdamW", "lr": lr, "betas": list(betas), "eps": eps,
              "weight_decay": weight_decay, "schedule": "constant", "foreach": False}
    return OptimizerSpec("AdamW", config, lambda params: torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=False))


def next_token_nll(logits: torch.Tensor, labels: torch.Tensor, *,
                   ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, int]:
    """``(differentiable sum over valid positions of -log P(label_{t+1}), count)``."""
    if logits.dim() != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(f"logits {tuple(logits.shape)} and labels {tuple(labels.shape)} disagree")
    shifted = logits[:, :-1].float()
    targets = labels[:, 1:]
    count = int((targets != ignore_index).sum())
    if count == 0:
        raise ValueError("a batch with no valid next-token positions")
    total = torch.nn.functional.cross_entropy(shifted.reshape(-1, shifted.shape[-1]),
                                              targets.reshape(-1), ignore_index=ignore_index,
                                              reduction="sum")
    return total, count


def _batch(source: Any, index: int) -> Any:
    return source.batch(index) if hasattr(source, "batch") else source[index]


def _describe(source: Any, count: int | None) -> dict[str, Any]:
    if hasattr(source, "describe"):
        return {"describe": source.describe(), "count": count}
    digests = []
    for i in range(count if count is not None else len(source)):
        digests.append(stable_hash([tensor_digest(t) for t in _batch(source, i)
                                    if torch.is_tensor(t)]))
    return {"content_hash": stable_hash(digests), "count": len(digests)}


def logits_of_hf(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=input_ids, use_cache=False).logits


@dataclass
class LanguageModelTraining:
    """Next-token LM training over named-site models. See the module docstring.

    ``build(deployment, hooks)`` returns the model with the deployment's sites
    installed through ``hooks.wrap``; ``build_native()`` returns the same model
    with nothing replaced (the trusted evaluator). Both must start from the
    same deterministic initial weights.
    """

    build: Callable[[Deployment, SiteHooks], torch.nn.Module]
    build_native: Callable[[], torch.nn.Module]
    train_data: Any
    #: ``None`` for a training-only measurement (``TrainingPlan(validation=False)``).
    validation_data: Any
    optimizer: OptimizerSpec
    validation_count: int | None = None
    train_count: int | None = None
    logits_of: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] = logits_of_hf
    ignore_index: int = IGNORE_INDEX
    device: str = "cpu"
    config: Mapping[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return {"objective": "token-weighted next-token NLL (float32, shifted, "
                             f"ignore_index={self.ignore_index})",
                "backward": "mean over the batch's valid tokens", "config": dict(self.config)}

    def data_identity(self) -> dict[str, Any]:
        return {"train": _describe(self.train_data, self.train_count),
                "validation": None if self.validation_data is None
                else _describe(self.validation_data, self.validation_count)}

    def optimizer_config(self) -> dict[str, Any]:
        return dict(self.optimizer.config)

    def build_model(self, deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module:
        return self.build(deployment, hooks).train()

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.optimizer.build(model)

    def build_evaluator(self) -> torch.nn.Module:
        return self.build_native()

    def train_batch(self, index: int) -> Any:
        if self.train_count is not None and index >= self.train_count:
            raise IndexError(f"training batch {index} beyond the declared {self.train_count}")
        return _batch(self.train_data, index)

    def train_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, float]:
        input_ids, labels = batch
        return next_token_nll(self.logits_of(model, input_ids), labels,
                              ignore_index=self.ignore_index)

    def validation_batches(self) -> Sequence[Any]:
        if self.validation_data is None:
            return []
        count = self.validation_count if self.validation_count is not None else len(self.validation_data)
        return [_batch(self.validation_data, i) for i in range(count)]

    def validation_loss(self, evaluator: torch.nn.Module, batch: Any) -> tuple[float, float]:
        input_ids, labels = batch
        total, count = next_token_nll(self.logits_of(evaluator, input_ids), labels,
                                      ignore_index=self.ignore_index)
        return float(total), float(count)


def site_model_training(factory: Callable[[Mapping[str, Callable]], torch.nn.Module],
                        native_sites: Mapping[str, Callable], **options: Any) -> LanguageModelTraining:
    """``factory(sites)`` builds the model with a full ``site -> callable`` map."""
    def build(deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module:
        sites = dict(native_sites)
        for site, fn in deployment.sites.items():
            if site not in sites:
                raise KeyError(f"no site {site!r}")
            sites[site] = hooks.wrap(site, fn)
        return factory(sites).to(options.get("device", "cpu"))

    return LanguageModelTraining(build=build,
                                 build_native=lambda: factory(dict(native_sites)).to(
                                     options.get("device", "cpu")),
                                 **options)


def workload_training(workload: Any, *, train_data: Any, optimizer: OptimizerSpec,
                      validation_data: Any = None, validation_count: int | None = None,
                      train_count: int | None = None) -> LanguageModelTraining:
    """A tier-3 ``TrainingWorkload`` (Qwen3, Llama) as a Gate 3 adapter.

    For Llama, ``train_data``/``validation_data`` can be the workload's own
    ``SyntheticBatches``; Qwen3's synthetic path is ``batch_for`` (wrap it, e.g.
    ``SeedBatches``) since its ``train_batches`` needs cached real text.
    Without ``validation_data`` the adapter serves training-only plans.
    """
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    registry = workload.site_registry

    def build(deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module:
        kernels = KernelSet(registry=registry)
        for site, fn in deployment.sites.items():
            kernels = patch(kernels, site, hooks.wrap(site, fn), source=KernelSource(
                site, registry.require(site).op, None, f"verification:{deployment.name}"))
        model, _ = workload.build_patched(kernels)
        return model

    return LanguageModelTraining(
        build=build, build_native=workload.eager_evaluator, train_data=train_data,
        validation_data=validation_data, validation_count=validation_count,
        train_count=train_count, optimizer=optimizer,
        device=str(getattr(getattr(workload, "spec", None), "device", "cpu")),
        config={"workload": workload.describe()})


@dataclass(frozen=True)
class SeedBatches:
    """``workload.batch_for(seed=offset + index)`` as a batch source."""

    workload: Any
    offset: int
    role: str

    def batch(self, index: int) -> Any:
        return self.workload.batch_for(seed=self.offset + int(index))

    def describe(self) -> dict[str, Any]:
        return {"source": "workload.batch_for", "offset": self.offset, "role": self.role,
                "workload": getattr(self.workload, "name", None)}
