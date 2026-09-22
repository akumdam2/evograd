"""Deriving the llama3_qkv_rope case from the canonical layer artifact.

Everything here describes *the task*: where its boundary sits in the
live model, which tensors cross it, in which layout, and whether what
was captured still matches. Nothing here decides whether an
implementation is good enough -- that is
:mod:`evograd.evaluation.workloads.llama3_2_1b.level2.qkv_rope`.

Derive and verify ``llama3_qkv_rope`` from the canonical Layer-16 artifact.

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.qkv_rope derive \
        --source results/llama3-level4/layer8.pt \
        --metadata-out results/llama3-level4/layer8-qkv.json

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.qkv_rope verify \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/layer8-qkv-verify.json

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.qkv_rope calibrate \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/llama3_qkv_rope-tolerance.json

``layer8.pt`` stays the only tensor artifact. This boundary's inputs and
outputs already live inside it, so they are re-derived by replaying the layer
and hooking two points, and only JSON is written.

**What is not here, relative to Qwen3.** Llama-3 has no per-head query/key
RMSNorm, so this boundary is the three projections and the rotation, its
declaration has four active arguments rather than six, and there is no ``eps``:
the operator contains no normalization to have one.

The two hooks bracket the boundary exactly:

* ``self_attn``'s forward pre-hook sees ``hidden_states`` -- the normalized
  residual stream that is this task's ``x`` -- and a tensor hook on it yields
  ``dx``. It is consumed by nothing else in the layer, so its gradient *is* this
  boundary's input gradient.
* the ``scaled_dot_product_attention`` call sees ``q``, ``k`` and ``v`` in the
  exact layout the boundary must produce, and tensor hooks on them yield
  ``dq``, ``dk``, ``dv`` -- which are this boundary's *upstream* gradients, the
  ones the next task hands back.

``cos`` and ``sin`` come from the artifact's ``position_embeddings`` argument.
The three weights and their gradients come from the layer's own modules; each is
used once per step, so a gradient read after one backward belongs to this
boundary alone.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from evograd.benchmark.topdown.llama3_2_1b.levels.level3.artifact import ArtifactError, content_hash_over, identity_hash_over, load_canonical
from evograd.benchmark.topdown.llama3_2_1b.levels.level2.attention.capture import preserve_layout_cpu
from evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-llama3-qkv-task/1"

TASK_NAME = "llama3_qkv_rope"

CONTENT_KEYS = ("inputs", "outputs", "output_grads", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "rope_config_id",
    "q_proj_config_id",
    "kv_proj_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

OUTPUT_NAMES = ("q", "k", "v")
GRAD_NAMES = (
    "dx",
    "dq_weight",
    "dk_weight",
    "dv_weight",
)

#: HF parameter path -> the declaration's argument name. Three entries, not
#: Qwen3's five: ``LlamaAttention`` carries no ``q_norm``/``k_norm``.
WEIGHT_MAP = {
    "q_proj.weight": "q_weight",
    "k_proj.weight": "k_weight",
    "v_proj.weight": "v_weight",
}


class QkvExtractionError(RuntimeError):
    """The QKV invocation could not be derived, or does not check out."""


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


class _QkvCapture:
    def __init__(self) -> None:
        self.x = None
        self.outputs: dict[str, torch.Tensor] = {}
        self.output_grads: dict[str, torch.Tensor] = {}
        self.dx = None
        self.attn_calls = 0
        self.sdpa_calls = 0
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [n for n in ("x", "dx") if getattr(self, n) is None]
        missing += [n for n in OUTPUT_NAMES if n not in self.outputs]
        missing += [f"d{n}" for n in OUTPUT_NAMES if f"d{n}" not in self.output_grads]
        if missing:
            raise QkvExtractionError(f"QKV capture is incomplete, missing {missing}")


@contextmanager
def capture_qkv(layer: torch.nn.Module) -> Iterator[_QkvCapture]:
    """Bracket the boundary: the attention input, and the SDPA inputs."""
    capture = _QkvCapture()
    attention = layer.get_submodule("self_attn")
    original = torch.nn.functional.scaled_dot_product_attention

    def pre_hook(module, args, kwargs):
        capture.attn_calls += 1
        if capture.attn_calls > 1:
            raise QkvExtractionError(
                "the attention module ran more than once inside one derivation; "
                "the task is defined as a single invocation"
            )
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if not torch.is_tensor(hidden):  # pragma: no cover - defensive
            raise QkvExtractionError("could not find the attention module's input")
        capture.x = preserve_layout_cpu(hidden)
        if hidden.requires_grad:
            capture.handles.append(
                hidden.register_hook(
                    lambda grad: setattr(capture, "dx", preserve_layout_cpu(grad))
                )
            )
        return None

    def sdpa_wrapper(query, key, value, *args, **kwargs):
        capture.sdpa_calls += 1
        if capture.sdpa_calls > 1:
            raise QkvExtractionError("SDPA ran more than once inside one derivation")
        for name, tensor in zip(OUTPUT_NAMES, (query, key, value)):
            capture.outputs[name] = preserve_layout_cpu(tensor)
            if tensor.requires_grad:
                capture.handles.append(
                    tensor.register_hook(
                        lambda grad, name=name: capture.output_grads.__setitem__(
                            f"d{name}", preserve_layout_cpu(grad)
                        )
                    )
                )
        return original(query, key, value, *args, **kwargs)

    handle = attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper
    try:
        yield capture
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        handle.remove()
        for h in capture.handles:
            h.remove()
        capture.handles.clear()


def derive_qkv_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    from evograd.benchmark.topdown.llama3_2_1b.levels.level3.prepare import prepare_layer
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise QkvExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    cos, sin = kwargs["position_embeddings"]
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_qkv(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    attention = layer.get_submodule("self_attn")
    weights: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {"dx": capture.dx}
    for hf_name, declared in WEIGHT_MAP.items():
        param = attention.get_parameter(hf_name)
        weights[declared] = preserve_layout_cpu(param)
        if param.grad is None:
            raise QkvExtractionError(f"self_attn.{hf_name} has no gradient after backward")
        grads[f"d{declared}"] = preserve_layout_cpu(param.grad)
    missing = [name for name in GRAD_NAMES if name not in grads]
    if missing:  # pragma: no cover - defensive
        raise QkvExtractionError(f"missing input gradients {missing}")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.self_attn"
    supporting = harvest["supporting"]
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "rope_config_id": harvest["config_id"],
            "q_proj_config_id": supporting["q_projection"]["config_id"],
            "kv_proj_config_id": supporting["kv_projection"]["config_id"],
            "frequency": harvest["frequency"],
            "layer_index": identity["layer_index"],
            "module_path": module_path,
            "source_content_hash": payload["content_hash"],
            "source_artifact_hash": payload["artifact_hash"],
            "provenance_kind": "derived_from_verified_replay",
        },
        "provenance_chain": [
            f"canonical workload {identity['workload_id']}",
            f"harvest manifest {identity['manifest_hash']}",
            f"{identity['module_path']} (event ordinal {identity['event_ordinal']})",
            f"Layer-16 artifact {payload['artifact_hash']}",
            f"q/k/v projections and RoPE at {module_path}",
            TASK_NAME,
        ],
        "attrs": {
            # No `eps`: this boundary contains no normalization to have one.
            "unsqueeze_dim": harvest["attrs"]["unsqueeze_dim"],
            "num_attention_heads": supporting["enclosing_attention"]["attrs"][
                "num_attention_heads"
            ],
            "num_key_value_heads": supporting["enclosing_attention"]["attrs"][
                "num_key_value_heads"
            ],
            "head_dim": supporting["enclosing_attention"]["attrs"]["head_dim"],
        },
        "inputs": {
            "x": capture.x,
            **weights,
            "cos": preserve_layout_cpu(cos),
            "sin": preserve_layout_cpu(sin),
        },
        "outputs": {name: capture.outputs[name] for name in OUTPUT_NAMES},
        "output_grads": {f"d{name}": capture.output_grads[f"d{name}"] for name in OUTPUT_NAMES},
        "grads": {name: grads[name] for name in GRAD_NAMES},
    }
    task_payload["content_hash"] = content_hash_over(task_payload, CONTENT_KEYS)
    task_payload["derivation_hash"] = identity_hash_over(task_payload, IDENTITY_KEYS)

    def meta(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "contiguous": bool(tensor.is_contiguous()),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": task_payload["identity"],
        "provenance_chain": task_payload["provenance_chain"],
        "content_hash": task_payload["content_hash"],
        "derivation_hash": task_payload["derivation_hash"],
        "derived_from": str(source),
        "tensors_written": False,
        "snapshot_hash": snapshot["snapshot_hash"],
        "attrs": task_payload["attrs"],
        "signature": {
            "inputs": {k: meta(v) for k, v in task_payload["inputs"].items()},
            "outputs": {k: meta(v) for k, v in task_payload["outputs"].items()},
            "output_grads": {k: meta(v) for k, v in task_payload["output_grads"].items()},
            "grads": {k: meta(v) for k, v in task_payload["grads"].items()},
        },
        "diagnostics": {
            "note": "diagnostic only -- one derivation pass, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
        },
    }
    return task_payload, metadata


def check_provenance(payload: dict[str, Any], *, snapshot_path: Path | None = None) -> list[str]:
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    supporting = harvest["supporting"]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("rope_config_id", harvest["config_id"]),
        ("q_proj_config_id", supporting["q_projection"]["config_id"]),
        ("kv_proj_config_id", supporting["kv_projection"]["config_id"]),
        ("frequency", harvest["frequency"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "derived_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = f"{snapshot['representative_layer']['module_path']}.self_attn"
    if identity.get("module_path") != expected_path:
        problems.append(
            f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}"
        )
    # The observed output layout, output by output.
    for index, name in enumerate(("q", "k")):
        observed = harvest["output_shapes"][index]
        tensor = payload["outputs"][name]
        if list(tensor.shape) != observed["shape"]:
            problems.append(f"{name} shape {list(tensor.shape)} != harvested {observed['shape']}")
        if list(tensor.stride()) != observed["stride"]:
            problems.append(f"{name} stride {list(tensor.stride())} != harvested {observed['stride']}")
    v_observed = supporting["consumer"]["input_shapes"][2]
    v = payload["outputs"]["v"]
    if list(v.shape) != v_observed["shape"] or list(v.stride()) != v_observed["stride"]:
        problems.append(
            f"v layout {list(v.shape)}/{list(v.stride())} != harvested "
            f"{v_observed['shape']}/{v_observed['stride']}"
        )
    for declared, record in (
        ("q_weight", supporting["q_projection"]),
        ("k_weight", supporting["kv_projection"]),
        ("v_weight", supporting["kv_projection"]),
    ):
        expected_shape = record["params"]["weight"]["shape"]
        got = list(payload["inputs"][declared].shape)
        if got != expected_shape:
            problems.append(f"{declared} shape {got} != harvested {expected_shape}")
    return problems


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?"""
    from evograd.benchmark import get_task

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    supporting = harvest["supporting"]
    declared = get_task(TASK_NAME).benchmark[0]
    batch, heads, tokens, head_dim = harvest["output_shapes"][0]["shape"]
    kv_heads = harvest["output_shapes"][1]["shape"][1]
    hidden = supporting["q_projection"]["input_shapes"][0]["shape"][-1]
    problems: list[str] = []
    for dim, expected in (
        ("B", batch),
        ("T", tokens),
        ("H", hidden),
        ("HQ", heads),
        ("HK", kv_heads),
        ("D", head_dim),
        ("QO", supporting["q_projection"]["params"]["weight"]["shape"][0]),
        ("KVO", supporting["kv_projection"]["params"]["weight"]["shape"][0]),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}")
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}")
    return problems
