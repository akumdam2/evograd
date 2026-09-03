"""Observing what the canonical Qwen3 training step actually invokes.

This is not a tracer. It watches a fixed list of *semantically stable* Qwen3
boundaries -- the ones a later milestone will turn into replayable tasks -- and
records one event per invocation. Everything below those boundaries (the
reshapes, the residual adds, the softmax inside SDPA) is deliberately invisible:
capturing every ATen call would produce a faithful but useless transcript, and
the units this project evolves are kernels like "RMSNorm at 1024 wide", not
`aten::view`.

Two capture mechanisms, chosen per boundary by what the installed Transformers
actually does:

* **Module hooks** where the boundary is an ``nn.Module`` -- decoder layers,
  attention, MLP, ``nn.Linear``, ``Qwen3RMSNorm``, the activation instance.
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
    return {
        "normalized_size": int(module.weight.shape[-1]),
        "eps": float(module.variance_epsilon),
    }


def _attention_attrs(module: nn.Module) -> dict[str, Any]:
    config = module.config
    return {
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "num_key_value_groups": int(module.num_key_value_groups),
        "head_dim": int(module.head_dim),
        "scaling": float(module.scaling),
        "attention_dropout": float(module.attention_dropout),
        "is_causal": bool(module.is_causal),
        "sliding_window": module.sliding_window,
        "attn_implementation": getattr(config, "_attn_implementation", None),
    }


def _mlp_attrs(module: nn.Module) -> dict[str, Any]:
    return {
        "hidden_size": int(module.hidden_size),
        "intermediate_size": int(module.intermediate_size),
        "hidden_act": str(module.config.hidden_act),
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
            "kind": "observed",
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


MODULE_BOUNDARIES: tuple[ModuleBoundary, ...] = (
    ModuleBoundary("decoder_layer", "Qwen3DecoderLayer", _decoder_layer_attrs, is_layer=True),
    ModuleBoundary("attention", "Qwen3Attention", _attention_attrs),
    ModuleBoundary("mlp", "Qwen3MLP", _mlp_attrs),
    ModuleBoundary("rms_norm", "Qwen3RMSNorm", _rms_norm_attrs),
    ModuleBoundary("linear", "Linear", _linear_attrs),
    ModuleBoundary("rotary_embedding", "Qwen3RotaryEmbedding", _rotary_attrs, mandatory=False),
)

#: Task names that must be present in a finished manifest.
MANDATORY_TASKS: tuple[str, ...] = (
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


def _activation_modules(model: nn.Module) -> dict[int, str]:
    """The activation instances a Qwen3 MLP actually calls, by identity.

    ``ACT2FN`` yields a fresh ``nn.Module`` per lookup in Transformers 5.16.1, so
    each MLP owns its own SiLU and a hook on it fires once per MLP call. Found
    by walking to each MLP's ``act_fn`` rather than by class name, because the
    class behind ``"silu"`` has changed spelling before (``SiLUActivation`` vs
    ``nn.SiLU``) and may again.
    """
    found: dict[int, str] = {}
    for path, module in model.named_modules():
        if type(module).__name__ != "Qwen3MLP":
            continue
        act = getattr(module, "act_fn", None)
        if isinstance(act, nn.Module):
            found[id(act)] = f"{path}.act_fn"
    return found


def _install_module_hooks(installer: _Installer, model: nn.Module) -> dict[str, int]:
    by_class = {b.class_name: b for b in MODULE_BOUNDARIES}
    installed: dict[str, int] = {task: 0 for task in (b.task for b in MODULE_BOUNDARIES)}

    activations = _activation_modules(model)
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
            installer.hook_module(module, path, "silu", _no_attrs, role="act_fn")
            installed["silu"] += 1
    return installed


def _install_function_patches(installer: _Installer, obs: Observation) -> dict[str, bool]:
    """Wrap the three functional boundaries at the call sites Transformers uses.

    Each wrapper records the enclosing module path and decoder-layer index from
    the observer's stack, so a function with no ``self`` still carries the same
    provenance a module hook would have given it.
    """
    import transformers.loss.loss_utils as loss_utils
    import transformers.models.qwen3.modeling_qwen3 as modeling

    available: dict[str, bool] = {}

    def _context() -> tuple[str | None, int | None]:
        module = obs.current_module
        layer = obs.current_layer
        return (module[0] if module else None, layer[1] if layer else None)

    # RoPE: a module-level function in modeling_qwen3, called by name from
    # Qwen3Attention.forward, so rebinding the module attribute is enough.
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


@contextmanager
def observe(model: nn.Module, *, workload_id: str, config_hash: str) -> Iterator[Observation]:
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

    observation = Observation(workload_id=workload_id, config_hash=config_hash)
    installer = _Installer(observation)
    _ACTIVE = observation
    try:
        hooked = _install_module_hooks(installer, model)
        patched = _install_function_patches(installer, observation)
        missing = [task for task, count in hooked.items() if count == 0 and task in MANDATORY_TASKS]
        if missing:
            raise ObserverError(
                "no module implements these mandatory boundaries in the installed "
                f"Transformers: {missing}. The manifest would be incomplete, so "
                "nothing was observed."
            )
        observation.boundaries = {
            "modules_hooked": dict(sorted(hooked.items())),
            "functions_patched": dict(sorted(patched.items())),
        }
        yield observation
    finally:
        installer.restore()
        _ACTIVE = None


def check_mandatory_boundaries(observation: Observation) -> None:
    """Fail rather than export a manifest that is missing a defined boundary."""
    counts = observation.counts_by_task()
    missing = [task for task in MANDATORY_TASKS if counts.get(task, 0) == 0]
    if missing:
        raise MandatoryBoundaryError(
            f"these mandatory boundaries produced no events: {missing}. "
            f"Observed: {counts}. The installed Transformers may have moved the "
            "call site; the manifest was not written."
        )
