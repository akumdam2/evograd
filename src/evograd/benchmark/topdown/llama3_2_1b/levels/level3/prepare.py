"""Preparing the captured Llama-3 layer for replay.

A replay needs the layer rebuilt from the artifact's own architecture record,
its weights loaded, its tensors moved to a device, and its layout checked
against what was captured. All of that describes *the artifact*, so it stays on
the benchmark side.

What the replay concludes is not here. Tolerances, gates, noise measurement and
the verdict live in
:mod:`evograd.evaluation.workloads.llama3_2_1b.level3.replay`.
"""

from __future__ import annotations

import gc
from typing import Any

import torch

from .artifact import ArtifactError, tensor_meta, to_device
from ...levels.level4.model import DTYPES, require_transformers


class ReplayError(RuntimeError):
    """The replay could not be performed, or disagreed with the capture."""


def validate_noise_repeats(value: Any) -> int:
    """0 to skip the measurement, or at least 2 to make one.

    A single replay has nothing to be compared against, so ``1`` would silently
    report no noise floor while looking like it measured one.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"noise_repeats must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"noise_repeats must not be negative, got {value}")
    if value == 1:
        raise ValueError(
            "noise_repeats=1 cannot measure anything: one replay has nothing to "
            "be compared with. Pass 0 to skip the measurement, or >= 2 to make it."
        )
    return value


def build_config(arch: dict[str, Any]):
    require_transformers()
    from transformers import LlamaConfig

    return LlamaConfig(**arch, use_cache=False, attn_implementation="sdpa")


def build_single_layer(arch: dict[str, Any], layer_index: int, *, device: str, dtype: str):
    """One ``LlamaDecoderLayer``. Nothing above it is constructed."""
    require_transformers()
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    config = build_config(arch)
    torch_dtype = DTYPES[dtype]
    layer = LlamaDecoderLayer(config, layer_index)
    # A decoder layer owns no floating-point buffers -- unlike the full model,
    # whose rotary `inv_freq` must stay float32 -- so casting the whole module is
    # safe here. Checked rather than assumed.
    float_buffers = [name for name, buf in layer.named_buffers() if buf.is_floating_point()]
    if float_buffers:  # pragma: no cover - version dependent
        raise ReplayError(
            f"this LlamaDecoderLayer has floating-point buffers {float_buffers}; "
            "casting the module would change their precision"
        )
    layer = layer.to(device=device, dtype=torch_dtype)
    layer.train()
    layer.gradient_checkpointing = False
    return layer


def live_model_instances() -> dict[str, int]:
    """Count live Llama objects on the heap.

    The standalone claim is worth only as much as its evidence, and "we did not
    call the constructor" is weaker evidence than "no such object exists".
    """
    require_transformers()
    from transformers.models.llama import modeling_llama as modeling

    classes = {
        name: getattr(modeling, name)
        for name in ("LlamaForCausalLM", "LlamaModel", "LlamaDecoderLayer", "LlamaAttention")
        if hasattr(modeling, name)
    }
    counts = {name: 0 for name in classes}
    gc.collect()
    for obj in gc.get_objects():
        try:
            for name, cls in classes.items():
                if type(obj) is cls:
                    counts[name] += 1
                    break
        except ReferenceError:  # pragma: no cover - weakref proxies
            continue
    return counts


def prepare_layer(payload: dict[str, Any], device: str):
    """Rebuild the captured layer and its arguments on ``device``.

    Shared by every task derived from a layer artifact, so the MLP and the
    attention block cannot drift into loading the same artifact two different
    ways.
    """
    identity = payload["identity"]
    dtype = str(payload["output"].dtype).replace("torch.", "")
    layer = build_single_layer(payload["arch"], identity["layer_index"], device=device, dtype=dtype)
    layer.load_state_dict({k: v.to(device) for k, v in payload["state_dict"].items()}, strict=True)
    args = to_device(payload["args"], device)
    kwargs = to_device(payload["kwargs"], device)
    grad_output = to_device(payload["grad_output"], device)
    return layer, args, kwargs, grad_output, dtype


def verify_layout(cpu_value: Any, device_value: Any, path: str = "$") -> list[str]:
    """Layout and dtype must survive the move to the device."""
    problems: list[str] = []
    if torch.is_tensor(cpu_value):
        if not torch.is_tensor(device_value):  # pragma: no cover - defensive
            return [f"{path}: expected a tensor"]
        if cpu_value.dtype != device_value.dtype:
            problems.append(f"{path}: dtype {device_value.dtype} != captured {cpu_value.dtype}")
        if tuple(cpu_value.shape) != tuple(device_value.shape):
            problems.append(f"{path}: shape {tuple(device_value.shape)} != {tuple(cpu_value.shape)}")
        if tuple(cpu_value.stride()) != tuple(device_value.stride()):
            problems.append(
                f"{path}: stride {tuple(device_value.stride())} != captured "
                f"{tuple(cpu_value.stride())}"
            )
        return problems
    if isinstance(cpu_value, (tuple, list)):
        for index, item in enumerate(cpu_value):
            problems += verify_layout(item, device_value[index], f"{path}[{index}]")
        return problems
    if isinstance(cpu_value, dict):
        for key, item in cpu_value.items():
            problems += verify_layout(item, device_value[key], f"{path}.{key}")
        return problems
    if cpu_value != device_value:  # pragma: no cover - scalars pass through
        problems.append(f"{path}: {device_value!r} != captured {cpu_value!r}")
    return problems
