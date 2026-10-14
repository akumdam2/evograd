"""Observing what a canonical training step actually invokes.

This is not a tracer. It watches a fixed list of *semantically stable*
boundaries -- the ones a later milestone will turn into replayable tasks -- and
records one event per invocation. Everything below those boundaries (the
reshapes, the residual adds, the softmax inside SDPA) is deliberately invisible:
capturing every ATen call would produce a faithful but useless transcript, and
the units this project evolves are kernels like "RMSNorm at 1024 wide", not
`aten::view`.

Two capture mechanisms, chosen per boundary by what the installed Transformers
actually does:

* **Module hooks** where the boundary is an ``nn.Module`` -- decoder layers,
  attention, MLP, ``nn.Linear``, the model's RMSNorm, the activation instance.
  Hooks are the honest mechanism here: they see exactly the tensors the module
  received, with no risk of the observer's model of the call diverging from the
  real one.
* **Narrow function wrappers** where it is not -- ``apply_rotary_pos_emb``,
  ``scaled_dot_product_attention``, and the causal-LM loss. Each patch names one
  attribute on one module, is installed only inside the context, and is restored
  on both success and failure.

Two rules the rest of the file exists to keep:

**No tensor outlives the hook.** Every hook converts its tensors to
:class:`TensorMeta` -- plain ints and strings -- before returning. An observer
that stashed a tensor would keep 1.2 GiB of logits alive to the end of the run
and change the very memory behaviour it is meant to describe.

**A missing mandatory boundary is an error, not a smaller manifest.** If
``apply_rotary_pos_emb`` moves in a future Transformers release, the harvest must
fail loudly; a manifest that is quietly missing RoPE would poison every task
derived from it.

Which *classes* implement these boundaries, and which module holds
``apply_rotary_pos_emb``, is the one thing that differs between architectures.
A workload states that in an :class:`ObservationPlan` and nothing else here
knows what model is being watched.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import nn

#: Phase currently captured. Backward runs, but nothing observes it -- see the
#: `capture_scope` block in the manifest.
PHASE_FORWARD = "forward"

_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

#: One observer at a time. Two nested contexts would double every event.
_ACTIVE: "Observation | None" = None


class ObserverError(RuntimeError):
    """The observer could not be installed, or was installed twice."""


class MandatoryBoundaryError(RuntimeError):
    """A boundary the manifest is defined to contain produced no events."""


# --------------------------------------------------------------------------
# metadata -- the only thing that ever leaves a hook
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    stride: tuple[int, ...]
    contiguous: bool
    numel: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "TensorMeta":
        return cls(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            # Type only: which physical GPU ran it is environment, not structure.
            device=tensor.device.type,
            requires_grad=bool(tensor.requires_grad),
            stride=tuple(tensor.stride()),
            contiguous=bool(tensor.is_contiguous()),
            numel=int(tensor.numel()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["stride"] = list(self.stride)
        return payload


_MAX_DESCRIBE_DEPTH = 4


def describe(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """A JSON-safe description of one argument, holding no reference to it.

    Opaque objects are reduced to their class name and never ``repr``'d: a repr
    routinely carries an object address, which would make the manifest differ
    between two identical runs.
    """
    if torch.is_tensor(value):
        return {"kind": "tensor", **TensorMeta.of(value).to_dict()}
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool) or isinstance(value, (int, float, str)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": value.type}
    if depth >= _MAX_DESCRIBE_DEPTH:
        return {"kind": "opaque", "type": type(value).__name__}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [describe(item, depth=depth + 1) for item in value],
        }
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": {
                str(k): describe(v, depth=depth + 1) for k, v in sorted(value.items(), key=str)
            },
        }
    return {"kind": "opaque", "type": type(value).__name__}


def describe_all(values: Sequence[Any]) -> list[dict[str, Any]]:
    return [describe(v) for v in values]


def parameter_meta(module: nn.Module) -> dict[str, Any]:
    """Only the module's own parameters -- children report their own."""
    return {
        name: TensorMeta.of(param).to_dict()
        for name, param in module.named_parameters(recurse=False)
    }


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


@dataclass
class Event:
    ordinal: int
    phase: str
    task: str
    module_path: str | None
    module_class: str | None
    role: str | None
    layer_index: int | None
    inputs: list[dict[str, Any]]
    input_kwargs: dict[str, dict[str, Any]]
    outputs: list[dict[str, Any]]
    params: dict[str, Any]
    attrs: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "phase": self.phase,
            "task": self.task,
            "module_path": self.module_path,
            "module_class": self.module_class,
            "role": self.role,
            "layer_index": self.layer_index,
            "inputs": self.inputs,
            "input_kwargs": self.input_kwargs,
            "outputs": self.outputs,
            "params": self.params,
            "attrs": self.attrs,
            "provenance": self.provenance,
        }

    def semantic_key_payload(self) -> dict[str, Any]:
        """What structural deduplication compares.

        Module path, role, layer index and ordinal are all absent by
        construction: two invocations that differ only in *where* they happened
        are the same configuration, and the record keeps all of those as
        provenance instead.
        """
        return {
            "task": self.task,
            "inputs": self.inputs,
            "input_kwargs": self.input_kwargs,
            "outputs": self.outputs,
            "params": self.params,
            "attrs": self.attrs,
        }


def layer_index_of(path: str | None) -> int | None:
    if not path:
        return None
    match = _LAYER_INDEX_RE.search(path)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# per-boundary attribute extraction
# --------------------------------------------------------------------------


def _linear_attrs(module: nn.Module) -> dict[str, Any]:
    return {
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "bias": module.bias is not None,
    }


def _rms_norm_attrs(module: nn.Module) -> dict[str, Any]:
    # Llama and Qwen3 spell the epsilon ``variance_epsilon``; the zero-centred
    # ``(1 + w)`` norms (Gemma, Qwen3.5) spell it ``eps``. Either is the same
    # fact, and a norm with neither is a real incompatibility.
    eps = getattr(module, "variance_epsilon", None)
    if eps is None:
        eps = module.eps
    return {
        "normalized_size": int(module.weight.shape[-1]),
        "eps": float(eps),
    }


def _attention_attrs(module: nn.Module) -> dict[str, Any]:
    config = module.config
    return {
        # Read strictly: these define the operator, and one missing is a real
        # incompatibility rather than an architecture that happens to differ.
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "num_key_value_groups": int(module.num_key_value_groups),
        "head_dim": int(module.head_dim),
        "scaling": float(module.scaling),
        # Read leniently: a decoder that does not do sliding-window attention
        # has no such attribute, and "absent" is the honest record. Qwen3
        # carries one, Llama-3 does not.
        "attention_dropout": float(getattr(module, "attention_dropout", 0.0)),
        "is_causal": bool(getattr(module, "is_causal", True)),
        "sliding_window": getattr(module, "sliding_window", None),
        "attn_implementation": getattr(config, "_attn_implementation", None),
    }


def _mlp_attrs(module: nn.Module) -> dict[str, Any]:
    # A dense MLP carries its own ``intermediate_size``. A routed MoE block in
    # the MLP position does not -- its width is per expert -- so the configured
    # expert widths are recorded instead of failing on an absent attribute.
    config = module.config if hasattr(module, "config") else None
    attrs: dict[str, Any] = {"hidden_size": int(getattr(module, "hidden_size", 0)
                                                or getattr(config, "hidden_size"))}
    if hasattr(module, "intermediate_size"):
        attrs["intermediate_size"] = int(module.intermediate_size)
    else:
        attrs["moe_intermediate_size"] = int(config.moe_intermediate_size)
        attrs["num_experts"] = int(config.num_experts)
    attrs["hidden_act"] = str(config.hidden_act)
    return attrs


def _moe_attrs(module: nn.Module) -> dict[str, Any]:
    """A routed MoE block: router, experts and (optionally) a shared expert."""
    gate, experts = module.gate, module.experts
    shared = getattr(module, "shared_expert", None)
    return {
        "num_experts": int(getattr(gate, "num_experts", getattr(experts, "num_experts", 0))),
        "top_k": int(getattr(gate, "top_k", 0)),
        "norm_topk_prob": bool(getattr(gate, "norm_topk_prob", False)),
        "shared_expert_intermediate_size": (
            int(shared.intermediate_size) if shared is not None else None),
        "shared_expert_gate": getattr(module, "shared_expert_gate", None) is not None,
    }


def _router_attrs(module: nn.Module) -> dict[str, Any]:
    return {
        "num_experts": int(module.num_experts),
        "top_k": int(module.top_k),
        "norm_topk_prob": bool(getattr(module, "norm_topk_prob", False)),
        "hidden_dim": int(getattr(module, "hidden_dim", module.weight.shape[-1])),
    }


def _experts_attrs(module: nn.Module) -> dict[str, Any]:
    config = getattr(module, "config", None)
    return {
        "num_experts": int(module.num_experts),
        "hidden_dim": int(module.hidden_dim),
        "intermediate_dim": int(module.intermediate_dim),
        "experts_implementation": getattr(config, "_experts_implementation", None),
    }


def _decoder_layer_attrs(module: nn.Module) -> dict[str, Any]:
    return {"hidden_size": int(module.hidden_size)}


def _rotary_attrs(module: nn.Module) -> dict[str, Any]:
    return {
        "rope_type": str(getattr(module, "rope_type", None)),
        "attention_scaling": float(getattr(module, "attention_scaling", 1.0)),
        "max_seq_len_cached": int(getattr(module, "max_seq_len_cached", 0)),
    }


def _no_attrs(module: nn.Module) -> dict[str, Any]:
    return {}


# --------------------------------------------------------------------------
# the observation
# --------------------------------------------------------------------------


@dataclass
class Observation:
    """Events plus a statement of what was and was not watched."""

    workload_id: str
    config_hash: str
    events: list[Event] = field(default_factory=list)
    #: The provenance kind every event carries; see ``ObservationPlan``.
    provenance_kind: str = "observed"
    boundaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _ordinal: int = 0
    _pending: dict[int, list[tuple[int, list, dict]]] = field(default_factory=dict)
    _module_stack: list[tuple[str, str]] = field(default_factory=list)
    _layer_stack: list[tuple[str, int]] = field(default_factory=list)

    # -- recording ------------------------------------------------------

    def next_ordinal(self) -> int:
        ordinal = self._ordinal
        self._ordinal += 1
        return ordinal

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.provenance_kind,
            "workload_id": self.workload_id,
            "config_hash": self.config_hash,
        }

    def record(
        self,
        *,
        ordinal: int,
        task: str,
        inputs: list[dict[str, Any]],
        input_kwargs: dict[str, dict[str, Any]],
        outputs: list[dict[str, Any]],
        module_path: str | None = None,
        module_class: str | None = None,
        role: str | None = None,
        layer_index: int | None = None,
        params: dict[str, Any] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            Event(
                ordinal=ordinal,
                phase=PHASE_FORWARD,
                task=task,
                module_path=module_path,
                module_class=module_class,
                role=role,
                layer_index=layer_index,
                inputs=inputs,
                input_kwargs=input_kwargs,
                outputs=outputs,
                params=params or {},
                attrs=attrs or {},
                provenance=self.provenance,
            )
        )

    # -- context for functional boundaries -------------------------------

    @property
    def current_layer(self) -> tuple[str, int] | None:
        return self._layer_stack[-1] if self._layer_stack else None

    @property
    def current_module(self) -> tuple[str, str] | None:
        return self._module_stack[-1] if self._module_stack else None

    def ordered_events(self) -> list[Event]:
        return sorted(self.events, key=lambda e: e.ordinal)

    def counts_by_task(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.task] = counts.get(event.task, 0) + 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# boundary table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleBoundary:
    task: str
    class_name: str
    attrs: Callable[[nn.Module], dict[str, Any]]
    mandatory: bool = True
    #: Pushes a decoder-layer frame so functional boundaries inside it can say
    #: which layer they belong to.
    is_layer: bool = False


#: The boundaries every HuggingFace decoder-only causal LM presents, named by
#: the *task* rather than by the class that happens to implement it. A workload
#: supplies the class names; the tasks, and the attributes read off each, are
#: the same for Llama, Mistral and Qwen because they are the Transformers
#: decoder convention rather than one model's invention.
MANDATORY_DECODER_TASKS: tuple[str, ...] = (
    "decoder_layer",
    "attention",
    "mlp",
    "linear",
    "rms_norm",
    "silu",
    "rope_apply",
    "sdpa",
    "causal_cross_entropy",
)

#: The attribute extractor each decoder task uses. Shared for the same reason:
#: ``num_key_value_groups``, ``variance_epsilon`` and ``intermediate_size`` are
#: spelled identically across these architectures.
DECODER_ATTRS: dict[str, Callable[[nn.Module], dict[str, Any]]] = {
    "decoder_layer": _decoder_layer_attrs,
    "attention": _attention_attrs,
    "mlp": _mlp_attrs,
    "rms_norm": _rms_norm_attrs,
    "linear": _linear_attrs,
    "rotary_embedding": _rotary_attrs,
    # Routed-MoE boundaries, spelled the same way by the HF MoE decoders.
    "moe": _moe_attrs,
    "router": _router_attrs,
    "experts": _experts_attrs,
}


@dataclass(frozen=True)
class FunctionBoundary:
    """A module-level function a workload wants observed, beyond the standard three.

    The standard decoder boundaries (RoPE, SDPA, the loss) are installed for
    every plan. An architecture whose operators are plain functions --
    a gated delta rule, a causal convolution, a grouped GEMM -- names them here:
    which attribute on which namespace, the task they record as, and the role.
    ``role`` may be a callable ``(args, kwargs, module_path) -> str`` when one
    function serves several roles distinguishable only by their arguments.
    """

    task: str
    namespace: str
    attribute: str
    role: str | Callable[[tuple, dict, str | None], str]
    #: Keyword arguments recorded as inputs after the positional ones, in order.
    tensor_kwargs: tuple[str, ...] = ()
    attrs: Callable[[tuple, dict], dict[str, Any]] | None = None

    @property
    def wrapper_name(self) -> str:
        return f"{self.namespace}.{self.attribute}"


@dataclass(frozen=True)
class ObservationPlan:
    """What one workload wants watched: its classes, and where RoPE lives.

    The observer is not a tracer -- it watches a fixed list of semantically
    stable boundaries. *Which* boundaries those are is generic; which classes
    implement them, and which module holds ``apply_rotary_pos_emb``, is not.
    That is the whole of what a workload has to say here.
    """

    #: Human-readable, for the capture scope.
    name: str
    #: The module boundaries to hook, by class name.
    boundaries: tuple[ModuleBoundary, ...]
    #: Dotted path of the modeling module holding ``apply_rotary_pos_emb``.
    modeling_module: str
    #: Tasks a finished manifest must contain.
    mandatory_tasks: tuple[str, ...] = MANDATORY_DECODER_TASKS
    #: Further function boundaries this architecture's operators live behind.
    functions: tuple[FunctionBoundary, ...] = ()
    #: Classes other than the MLP whose ``act_fn`` is an observed activation
    #: (a routed experts module, say). Recorded under the role
    #: ``<owner role>_act_fn`` so it cannot merge with the MLP's ``act_fn``.
    activation_owners: tuple[str, ...] = ()
    #: What the events' provenance calls this observation. ``observed`` for a
    #: whole training step; a block-level harvest says so.
    provenance_kind: str = "observed"


def _names(value: str | Sequence[str]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else tuple(value)


def decoder_boundaries(
    *,
    decoder_layer: str | Sequence[str],
    attention: str | Sequence[str],
    mlp: str | Sequence[str],
    rms_norm: str | Sequence[str],
    rotary_embedding: str | None = None,
    linear: str = "Linear",
    extra: Sequence[ModuleBoundary] = (),
) -> tuple[ModuleBoundary, ...]:
    """The standard decoder boundary table, given one model's class names.

    Each standard task may be implemented by more than one class -- a hybrid
    decoder has two token mixers and, in the MLP position, a routed MoE beside a
    dense shared expert -- and ``extra`` adds the boundaries only such a model
    has. A single class name per task, as every dense decoder passes, gives the
    same table as before.
    """
    table: list[ModuleBoundary] = []
    for name in _names(decoder_layer):
        table.append(ModuleBoundary("decoder_layer", name, DECODER_ATTRS["decoder_layer"],
                                    is_layer=True))
    for name in _names(attention):
        table.append(ModuleBoundary("attention", name, DECODER_ATTRS["attention"]))
    for name in _names(mlp):
        table.append(ModuleBoundary("mlp", name, DECODER_ATTRS["mlp"]))
    for name in _names(rms_norm):
        table.append(ModuleBoundary("rms_norm", name, DECODER_ATTRS["rms_norm"]))
    table.append(ModuleBoundary("linear", linear, DECODER_ATTRS["linear"]))
    if rotary_embedding is not None:
        table.append(
            ModuleBoundary("rotary_embedding", rotary_embedding,
                           DECODER_ATTRS["rotary_embedding"], mandatory=False)
        )
    table.extend(extra)
    return tuple(table)


def _role_of(path: str) -> str:
    """The semantic role is the attribute the module was assigned to.

    ``model.layers.7.self_attn.q_proj`` is a ``q_proj``; a decoder layer, whose
    path ends in its index, is a ``decoder_layer``.
    """
    tail = path.rsplit(".", 1)[-1] if path else ""
    return "decoder_layer" if tail.isdigit() else (tail or "root")


# --------------------------------------------------------------------------
# installation
# --------------------------------------------------------------------------


class _Installer:
    """Owns every hook handle and every patched attribute, so that one
    ``finally`` can undo all of it whatever happened in between."""

    def __init__(self, observation: Observation):
        self.obs = observation
        self._handles: list[Any] = []
        self._patches: list[tuple[Any, str, Any]] = []
        self._undos: list[Callable[[], None]] = []

    # -- module hooks ---------------------------------------------------

    def hook_module(
        self,
        module: nn.Module,
        path: str,
        task: str,
        attrs_fn: Callable[[nn.Module], dict[str, Any]],
        *,
        is_layer: bool = False,
        role: str | None = None,
    ) -> None:
        obs = self.obs
        module_class = type(module).__name__
        resolved_role = role or _role_of(path)
        layer_index = layer_index_of(path)

        def pre_hook(mod, args, kwargs):
            ordinal = obs.next_ordinal()
            obs._pending.setdefault(id(mod), []).append(
                (ordinal, describe_all(args), {k: describe(v) for k, v in kwargs.items()})
            )
            obs._module_stack.append((path, module_class))
            if is_layer and layer_index is not None:
                obs._layer_stack.append((path, layer_index))
            return None

        def post_hook(mod, args, kwargs, output):
            stack = obs._pending.get(id(mod))
            if not stack:  # pragma: no cover - defensive
                return None
            ordinal, inputs, input_kwargs = stack.pop()
            if obs._module_stack:
                obs._module_stack.pop()
            if is_layer and layer_index is not None and obs._layer_stack:
                obs._layer_stack.pop()
            outs = output if isinstance(output, tuple) else (output,)
            obs.record(
                ordinal=ordinal,
                task=task,
                inputs=inputs,
                input_kwargs=input_kwargs,
                outputs=describe_all(outs),
                module_path=path,
                module_class=module_class,
                role=resolved_role,
                layer_index=layer_index,
                params=parameter_meta(mod),
                attrs=attrs_fn(mod),
            )
            return None

        self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._handles.append(module.register_forward_hook(post_hook, with_kwargs=True))

    # -- function patches -----------------------------------------------

    def patch(self, namespace: Any, name: str, factory: Callable[[Any], Any]) -> None:
        original = getattr(namespace, name)
        setattr(namespace, name, factory(original))
        self._patches.append((namespace, name, original))

    def add_undo(self, undo: Callable[[], None]) -> None:
        """For a boundary that is not an attribute -- a dict entry, say."""
        self._undos.append(undo)

    def patched_targets(self) -> list[tuple[Any, str, Any]]:
        return list(self._patches)

    # -- teardown --------------------------------------------------------

    def restore(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for namespace, name, original in reversed(self._patches):
            setattr(namespace, name, original)
        self._patches.clear()
        for undo in reversed(self._undos):
            undo()
        self._undos.clear()


def _activation_modules(model: nn.Module, mlp_classes: Sequence[str],
                        owners: Sequence[str] = ()) -> dict[int, tuple[str, str]]:
    """The activation instances the model actually calls, by identity.

    ``ACT2FN`` yields a fresh ``nn.Module`` per lookup in Transformers 5.16.1, so
    each MLP owns its own SiLU and a hook on it fires once per MLP call. Found
    by walking to each owner's ``act_fn`` rather than by class name, because the
    class behind ``"silu"`` has changed spelling before (``SiLUActivation`` vs
    ``nn.SiLU``) and may again.

    Returns ``{id: (path, role)}``. An MLP's activation keeps the role
    ``act_fn``; another owner's (a routed experts module) is ``<owner>_act_fn``,
    so the two never collapse into one role.
    """
    found: dict[int, tuple[str, str]] = {}
    mlp_classes, owners = set(mlp_classes), set(owners)
    for path, module in model.named_modules():
        name = type(module).__name__
        if name not in mlp_classes and name not in owners:
            continue
        act = getattr(module, "act_fn", None)
        if isinstance(act, nn.Module):
            role = "act_fn" if name in mlp_classes else f"{_role_of(path)}_act_fn"
            found[id(act)] = (f"{path}.act_fn", role)
    return found


def _install_module_hooks(installer: _Installer, model: nn.Module,
                          boundaries: tuple[ModuleBoundary, ...],
                          activation_owners: Sequence[str] = ()) -> dict[str, int]:
    by_class = {b.class_name: b for b in boundaries}
    installed: dict[str, int] = {task: 0 for task in (b.task for b in boundaries)}

    mlp_classes = [b.class_name for b in boundaries if b.task == "mlp"]
    activations = _activation_modules(model, mlp_classes, activation_owners)
    installed["silu"] = 0

    for path, module in model.named_modules():
        boundary = by_class.get(type(module).__name__)
        if boundary is not None:
            installer.hook_module(
                module,
                path,
                boundary.task,
                boundary.attrs,
                is_layer=boundary.is_layer,
            )
            installed[boundary.task] += 1
        elif id(module) in activations:
            act_path, role = activations[id(module)]
            installer.hook_module(module, path, "silu", _no_attrs, role=role)
            installed["silu"] += 1
    return installed


def _install_function_patches(installer: _Installer, obs: Observation,
                              modeling_module: str) -> dict[str, bool]:
    """Wrap the three functional boundaries at the call sites Transformers uses.

    Each wrapper records the enclosing module path and decoder-layer index from
    the observer's stack, so a function with no ``self`` still carries the same
    provenance a module hook would have given it.
    """
    import importlib

    import transformers.loss.loss_utils as loss_utils

    # The model's own modeling module: `apply_rotary_pos_emb` is a module-level
    # function called by name from the attention forward, so rebinding the
    # attribute reaches the call. Which module that is, is the workload's.
    modeling = importlib.import_module(modeling_module)

    available: dict[str, bool] = {}

    def _context() -> tuple[str | None, int | None]:
        return _context_of(obs)

    # RoPE: a module-level function in the model's modeling module, called by
    # name from the attention forward, so rebinding the attribute is enough.
    def rope_factory(original):
        def wrapper(q, k, cos, sin, unsqueeze_dim=1, **kwargs):
            ordinal = obs.next_ordinal()
            inputs = describe_all((q, k, cos, sin))
            out = original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim, **kwargs)
            path, layer = _context()
            obs.record(
                ordinal=ordinal,
                task="rope_apply",
                inputs=inputs,
                input_kwargs={},
                outputs=describe_all(out if isinstance(out, tuple) else (out,)),
                module_path=path,
                module_class=None,
                role="apply_rotary_pos_emb",
                layer_index=layer,
                attrs={"unsqueeze_dim": int(unsqueeze_dim)},
            )
            return out

        return wrapper

    installer.patch(modeling, "apply_rotary_pos_emb", rope_factory)
    available["rope_apply"] = True

    # SDPA: `transformers.integrations.sdpa_attention` calls
    # `torch.nn.functional.scaled_dot_product_attention` by attribute lookup, so
    # this is the call site. Its internal softmax is intentionally not a
    # boundary -- it will be derived from this configuration later.
    def sdpa_factory(original):
        def wrapper(query, key, value, *args, **kwargs):
            ordinal = obs.next_ordinal()
            inputs = describe_all((query, key, value))
            attn_mask = kwargs.get("attn_mask", args[0] if args else None)
            out = original(query, key, value, *args, **kwargs)
            path, layer = _context()
            obs.record(
                ordinal=ordinal,
                task="sdpa",
                inputs=inputs,
                input_kwargs={"attn_mask": describe(attn_mask)},
                outputs=describe_all((out,)),
                module_path=path,
                module_class=None,
                role="scaled_dot_product_attention",
                layer_index=layer,
                attrs={
                    "dropout_p": float(kwargs.get("dropout_p", 0.0)),
                    "is_causal": bool(kwargs.get("is_causal", False)),
                    "scale": kwargs.get("scale"),
                    "enable_gqa": bool(kwargs.get("enable_gqa", False)),
                    "attn_mask_provided": attn_mask is not None,
                },
            )
            return out

        return wrapper

    installer.patch(nn.functional, "scaled_dot_product_attention", sdpa_factory)
    available["sdpa"] = True

    # Causal cross entropy: `PreTrainedModel.loss_function` resolves
    # `LOSS_MAPPING[loss_type]` on every access, so replacing the mapping entry
    # reaches the call without touching the model.
    def causal_ce_factory(original):
        def wrapper(logits, labels, vocab_size, **kwargs):
            ordinal = obs.next_ordinal()
            inputs = describe_all((logits, labels))
            out = original(logits, labels, vocab_size, **kwargs)
            obs.record(
                ordinal=ordinal,
                task="causal_cross_entropy",
                inputs=inputs,
                input_kwargs={
                    k: describe(v) for k, v in kwargs.items() if k in ("shift_labels", "num_items_in_batch")
                },
                outputs=describe_all((out,)),
                module_path=None,
                module_class=None,
                role="loss_function",
                layer_index=None,
                attrs={
                    "vocab_size": int(vocab_size),
                    "ignore_index": int(kwargs.get("ignore_index", -100)),
                    "shift_labels_provided": kwargs.get("shift_labels") is not None,
                },
            )
            return out

        return wrapper

    mapping = loss_utils.LOSS_MAPPING
    original_entry = mapping["ForCausalLM"]
    mapping["ForCausalLM"] = causal_ce_factory(original_entry)
    installer.add_undo(lambda: mapping.__setitem__("ForCausalLM", original_entry))
    available["causal_cross_entropy"] = True

    # The flattened cross entropy inside it: the shape a Level-1/2 cross-entropy
    # task would actually be declared at. Best effort -- it is an internal
    # helper, so a rename downgrades it rather than failing the harvest.
    if hasattr(loss_utils, "fixed_cross_entropy"):

        def flat_ce_factory(original):
            def wrapper(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
                ordinal = obs.next_ordinal()
                inputs = describe_all((source, target))
                out = original(source, target, num_items_in_batch, ignore_index, **kwargs)
                obs.record(
                    ordinal=ordinal,
                    task="cross_entropy",
                    inputs=inputs,
                    input_kwargs={},
                    outputs=describe_all((out,)),
                    module_path=None,
                    module_class=None,
                    role="fixed_cross_entropy",
                    layer_index=None,
                    attrs={
                        "ignore_index": int(ignore_index),
                        "reduction": "sum" if num_items_in_batch is not None else "mean",
                    },
                )
                return out

            return wrapper

        installer.patch(loss_utils, "fixed_cross_entropy", flat_ce_factory)
        available["cross_entropy"] = True
    else:  # pragma: no cover - version dependent
        available["cross_entropy"] = False

    return available


def _context_of(obs: Observation) -> tuple[str | None, int | None]:
    """The enclosing module path and decoder-layer index, from the observer's stacks."""
    module = obs.current_module
    layer = obs.current_layer
    return (module[0] if module else None, layer[1] if layer else None)


def _install_extra_functions(installer: _Installer, obs: Observation,
                             functions: Sequence[FunctionBoundary]) -> dict[str, bool]:
    """Wrap each workload-named function boundary at its call site.

    Same contract as the standard wrappers: the ordinal is taken before the
    call, only metadata leaves the wrapper, and the enclosing module and layer
    come from the observer's stacks.
    """
    import importlib

    available: dict[str, bool] = {}
    for boundary in functions:
        namespace = importlib.import_module(boundary.namespace)

        def factory(original, _b=boundary):
            def wrapper(*args, **kwargs):
                ordinal = obs.next_ordinal()
                tensors = list(args) + [kwargs[k] for k in _b.tensor_kwargs if k in kwargs]
                inputs = describe_all(tensors)
                input_kwargs = {k: describe(v) for k, v in sorted(kwargs.items())
                                if k not in _b.tensor_kwargs}
                out = original(*args, **kwargs)
                path, layer = _context_of(obs)
                role = _b.role(args, kwargs, path) if callable(_b.role) else _b.role
                obs.record(
                    ordinal=ordinal,
                    task=_b.task,
                    inputs=inputs,
                    input_kwargs=input_kwargs,
                    outputs=describe_all(out if isinstance(out, tuple) else (out,)),
                    module_path=path,
                    module_class=None,
                    role=role,
                    layer_index=layer,
                    attrs=_b.attrs(args, kwargs) if _b.attrs is not None else {},
                )
                return out

            return wrapper

        installer.patch(namespace, boundary.attribute, factory)
        available[boundary.task] = True
    return available


@contextmanager
def observe(model: nn.Module, *, workload_id: str, config_hash: str,
            plan: ObservationPlan) -> Iterator[Observation]:
    """Watch ``model`` for the duration of the block.

    Every hook and every patch is removed in the ``finally``, so an exception in
    the observed step leaves the process exactly as it found it.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise ObserverError(
            "an observation is already active; nesting would record every "
            "invocation twice"
        )

    observation = Observation(workload_id=workload_id, config_hash=config_hash,
                              provenance_kind=plan.provenance_kind)
    installer = _Installer(observation)
    _ACTIVE = observation
    try:
        hooked = _install_module_hooks(installer, model, plan.boundaries,
                                       plan.activation_owners)
        patched = _install_function_patches(installer, observation,
                                            plan.modeling_module)
        if plan.functions:
            patched.update(_install_extra_functions(installer, observation, plan.functions))
        missing = [task for task, count in hooked.items()
                   if count == 0 and task in plan.mandatory_tasks]
        if missing:
            raise ObserverError(
                "no module implements these mandatory boundaries in the installed "
                f"Transformers: {missing}. The manifest would be incomplete, so "
                "nothing was observed."
            )
        observation.boundaries = {
            "workload": plan.name,
            "modules_hooked": dict(sorted(hooked.items())),
            "functions_patched": dict(sorted(patched.items())),
        }
        yield observation
    finally:
        installer.restore()
        _ACTIVE = None


def check_mandatory_boundaries(observation: Observation,
                               plan: ObservationPlan) -> None:
    """Fail rather than export a manifest that is missing a defined boundary."""
    counts = observation.counts_by_task()
    missing = [task for task in plan.mandatory_tasks if counts.get(task, 0) == 0]
    if missing:
        raise MandatoryBoundaryError(
            f"these mandatory boundaries produced no events: {missing}. "
            f"Observed: {counts}. The installed Transformers may have moved the "
            "call site; the manifest was not written."
        )


def check_layer_boundaries(observation: Observation, layer_index: int,
                           tasks: Sequence[str]) -> None:
    """Every listed task produced events inside one decoder layer.

    The whole-manifest check above is not enough for a hybrid model: SDPA and
    RoPE run in one layer in four, so "the manifest has an SDPA event" says
    nothing about whether the full-attention layer that should have run it did.
    Each block kind states the tasks it must contain, and each harvested layer
    is held to its own list.
    """
    counts: dict[str, int] = {}
    for event in observation.events:
        if event.layer_index == layer_index:
            counts[event.task] = counts.get(event.task, 0) + 1
    missing = [task for task in tasks if counts.get(task, 0) == 0]
    if missing:
        raise MandatoryBoundaryError(
            f"layer {layer_index} produced no events for {missing}. Observed in that "
            f"layer: {dict(sorted(counts.items()))}. The manifest was not written."
        )


class ObservedOperatorError(RuntimeError):
    """The step ran, but not the operator the architecture describes."""


def check_observed_attention(observation: Observation, arch: dict[str, Any]) -> None:
    """A grouped-query model must be observed *calling* SDPA with grouped keys.

    Transformers can reach SDPA two ways. When it can rely on ``is_causal=True``
    it passes the key and value tensors as the model holds them -- one per KV
    head -- and lets SDPA expand them. When it instead builds an explicit
    attention mask, ``enable_gqa`` is unavailable, so ``repeat_kv`` materialises
    the KV heads up to the query-head count *before* the call.

    Both compute the same attention. They are not the same operator: the second
    presents ``num_attention_heads`` keys rather than ``num_key_value_heads``,
    which is a different shape, a different memory cost, and -- for a benchmark
    whose whole premise is that its shapes come from a real step -- a different
    task. A kernel evolved against it would target tensors four times larger
    than the model actually holds.

    Which path Transformers takes has changed between releases, so this is
    checked against the observation rather than against a version number: the
    fact that matters is what the model ran, not what the installed library is
    expected to do.
    """
    heads = arch.get("num_attention_heads")
    kv_heads = arch.get("num_key_value_heads")
    if not heads or not kv_heads or heads == kv_heads:
        return  # not a grouped-query architecture; nothing to check

    for event in observation.ordered_events():
        if event.task != "sdpa" or len(event.inputs) < 2:
            continue
        key = event.inputs[1]
        if key.get("kind") != "tensor" or len(key.get("shape", ())) != 4:
            continue
        observed = key["shape"][1]
        if observed == kv_heads:
            continue
        raise ObservedOperatorError(
            f"this architecture has {kv_heads} key/value heads for "
            f"{heads} query heads, but SDPA was called with {observed} key "
            f"heads: Transformers materialised them with repeat_kv instead of "
            f"using enable_gqa. The step is numerically fine; the *observed "
            f"operator* is not the model's, so every shape derived from it "
            f"would describe a tensor the model does not hold.\n"
            f"    attn_mask_provided={event.attrs.get('attn_mask_provided')} "
            f"is_causal={event.attrs.get('is_causal')} "
            f"enable_gqa={event.attrs.get('enable_gqa')}\n"
            f"This follows the installed Transformers' choice of attention "
            f"path. The manifest was not written."
        )
