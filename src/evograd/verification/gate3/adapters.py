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
  evaluator (``eager_evaluator``); this installs replaced sites;
* :func:`workload_objective_training` -- a training-only adapter for a
  workload with its own objective (``primary_output``): vision, diffusion,
  recommendation.
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
    #: Linear warmup: update ``t`` (1-based) runs at ``lr * min(1, t / warmup_steps)``.
    warmup_steps: int = 0
    #: The global gradient norm is clipped to this before each update (``None``: never).
    clip_grad_norm: float | None = None

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.factory(model.parameters())

    def before_update(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                      step: int) -> dict[str, float]:
        """Clip and schedule update ``step``; returns what the step record adds.

        The learning rate is a function of the step alone, so a resumed run
        needs no scheduler state. With neither setting this does nothing and
        the step record is unchanged.
        """
        record: dict[str, float] = {}
        if self.clip_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm,
                                                  foreach=False)
            record["grad_norm"] = float(norm)
        if self.warmup_steps:
            lr = float(self.config["lr"]) * min(1.0, step / self.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            record["lr"] = lr
        return record


def adamw(*, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
          weight_decay: float = 0.01, warmup_steps: int = 0,
          clip_grad_norm: float | None = None) -> OptimizerSpec:
    """``torch.optim.AdamW`` with every setting stated.

    The learning rate is constant unless ``warmup_steps`` ramps it linearly
    over the first updates; ``clip_grad_norm`` clips the global gradient norm
    before each update. Either appears in the configuration only when set, so
    a constant-rate, unclipped run keeps its identity.
    """
    config = {"optimizer": "AdamW", "lr": lr, "betas": list(betas), "eps": eps,
              "weight_decay": weight_decay, "schedule": "constant", "foreach": False}
    if warmup_steps:
        config.update(schedule="linear_warmup", warmup_steps=int(warmup_steps))
    if clip_grad_norm is not None:
        config["clip_grad_norm"] = float(clip_grad_norm)
    return OptimizerSpec("AdamW", config, lambda params: torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, foreach=False),
        warmup_steps=int(warmup_steps),
        clip_grad_norm=None if clip_grad_norm is None else float(clip_grad_norm))


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

    def before_update(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                      step: int) -> dict[str, float]:
        return self.optimizer.before_update(optimizer, model, step)

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


def _workload_build(workload: Any) -> Callable[[Deployment, SiteHooks], torch.nn.Module]:
    """The workload's model with the deployment's sites installed through ``hooks``."""
    from evograd.evaluation.tier3.patch import KernelSet, KernelSource, patch

    registry = workload.site_registry

    def build(deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module:
        kernels = KernelSet(registry=registry)
        for site, fn in deployment.sites.items():
            kernels = patch(kernels, site, hooks.wrap(site, fn), source=KernelSource(
                site, registry.require(site).op, None, f"verification:{deployment.name}"))
        model, _ = workload.build_patched(kernels)
        return model

    return build


@dataclass
class WorkloadObjectiveTraining:
    """Training-only Gate 3 adapter for a workload with its own objective.

    For workloads that are not next-token language models (classification,
    diffusion, recommendation). The step differentiates exactly the loss
    ``workload.primary_output(model, batch)`` returns, with weight 1, so the
    recorded objective is that loss. There is no held-out evaluation: plans
    must be ``validation=False``.
    """

    workload: Any
    train_data: Any
    optimizer: OptimizerSpec
    train_count: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._build = _workload_build(self.workload)

    def identity(self) -> dict[str, Any]:
        return {"objective": str(self.workload.objective_definition),
                "backward": "the workload loss as returned (weight 1)",
                "config": {"workload": self.workload.describe()}}

    def data_identity(self) -> dict[str, Any]:
        return {"train": _describe(self.train_data, self.train_count), "validation": None}

    def optimizer_config(self) -> dict[str, Any]:
        return dict(self.optimizer.config)

    def build_model(self, deployment: Deployment, hooks: SiteHooks) -> torch.nn.Module:
        return self._build(deployment, hooks).train()

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.optimizer.build(model)

    def before_update(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module,
                      step: int) -> dict[str, float]:
        return self.optimizer.before_update(optimizer, model, step)

    def build_evaluator(self) -> torch.nn.Module:
        raise NotImplementedError("a workload-objective adapter serves training-only plans")

    def train_batch(self, index: int) -> Any:
        if self.train_count is not None and index >= self.train_count:
            raise IndexError(f"training batch {index} beyond the declared {self.train_count}")
        return _batch(self.train_data, index)

    def train_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, float]:
        loss, _primary = self.workload.primary_output(model, batch)
        return loss, 1.0

    def validation_batches(self) -> Sequence[Any]:
        return []

    def validation_loss(self, evaluator: torch.nn.Module, batch: Any) -> tuple[float, float]:
        raise NotImplementedError("a workload-objective adapter serves training-only plans")


def workload_objective_training(workload: Any, *, train_data: Any, optimizer: OptimizerSpec,
                                train_count: int | None = None) -> WorkloadObjectiveTraining:
    """A tier-3 workload with ``primary_output`` as a training-only Gate 3 adapter."""
    return WorkloadObjectiveTraining(
        workload=workload, train_data=train_data, optimizer=optimizer, train_count=train_count,
        device=str(getattr(getattr(workload, "spec", None), "device", "cpu")))


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
