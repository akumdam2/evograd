"""Deriving the llama3_residual_rmsnorm case from the canonical layer artifact.

Everything here describes *the task*: where its boundary sits in the
live model, which tensors cross it, in which layout, and whether what
was captured still matches. Nothing here decides whether an
implementation is good enough -- that is
:mod:`evograd.evaluation.workloads.llama3_8b.level2.residual_rmsnorm`.

Derive and verify ``llama3_residual_rmsnorm`` from the canonical Layer-16 artifact.

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_8b.level2.residual_rmsnorm derive \
        --source results/llama3-level4/layer16.pt \
        --metadata-out results/llama3-level4/layer16-residual.json

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_8b.level2.residual_rmsnorm verify \
        --source results/llama3-level4/layer16.pt \
        --report results/llama3-level4/layer16-residual-verify.json

    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_8b.level2.residual_rmsnorm calibrate \
        --source results/llama3-level4/layer16.pt \
        --report results/llama3-level4/llama3_residual_rmsnorm-tolerance.json

``layer16.pt`` stays the only tensor artifact; the boundary is re-derived by
replaying the layer and hooking it, and only JSON is written.

**The representative boundary inside layer 14** is the attention residual add
and the norm that follows it::

    residual   = the decoder layer's input
    x          = the self_attn branch's output
    summed     = residual + x
    normalized = post_attention_layernorm(summed)

**What is and is not directly comparable.** Three of the four gradients are
observable in the layer graph as themselves:

* ``dx`` -- ``x`` is consumed only by the residual add, so its gradient in the
  layer *is* this boundary's ``dx``.
* ``dtotal`` -- ``summed``'s own gradient, which the contract says equals
  ``dsummed + RMSNormBackward(dnormalized)``. Comparing it checks the
  combination rule directly.
* ``dweight`` -- ``post_attention_layernorm.weight`` is used once per step.

``dresidual`` is **not** directly comparable, and this module does not pretend
otherwise. In the real layer the decoder input feeds two consumers -- the
residual add *and* ``input_layernorm`` -- so its ``.grad`` is the sum of this
boundary's ``dresidual`` and a gradient from a path outside the boundary. The
contract's claim about it (``dresidual == dx``) is proved instead against an
isolated autograd reference, and the report records that it was proved that way.

The two upstream gradients are both observed rather than invented:
``dnormalized`` is the gradient arriving at the norm's output, and ``dsummed``
is the layer's own ``grad_output`` -- because the layer ends with
``summed + mlp(...)``, so the gradient reaching ``summed`` from its second
consumer is exactly the gradient of the layer's output.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from evograd.benchmark.topdown.llama3_8b.levels.level3.artifact import ArtifactError, content_hash_over, identity_hash_over, load_canonical
from evograd.benchmark.topdown.llama3_8b.levels.level2.attention.capture import preserve_layout_cpu
from evograd.benchmark.topdown.llama3_8b.harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-llama3-residual-task/1"

TASK_NAME = "llama3_residual_rmsnorm"

CONTENT_KEYS = ("inputs", "outputs", "output_grads", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "rms_norm_config_id",
    "fusion_sites",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

OUTPUT_NAMES = ("out", "summed")
#: Only these are observable as themselves in the layer graph; ``dr`` is not.
DIRECT_GRADS = ("dx", "dweight")


class ResidualExtractionError(RuntimeError):
    """The boundary could not be derived, or does not check out."""


class _ResidualCapture:
    def __init__(self) -> None:
        self.residual = None
        self.x = None
        self.summed = None
        self.normalized = None
        self.dnormalized = None
        self.dsummed_total = None
        self.dx = None
        self.norm_calls = 0

    def require_complete(self) -> None:
        missing = [
            name
            for name in (
                "residual",
                "x",
                "summed",
                "normalized",
                "dnormalized",
                "dsummed_total",
                "dx",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ResidualExtractionError(f"residual capture is incomplete, missing {missing}")


@contextmanager
def capture_residual(layer: torch.nn.Module) -> Iterator[_ResidualCapture]:
    """Hook the attention branch's output and the norm that follows the add."""
    capture = _ResidualCapture()
    attention = layer.get_submodule("self_attn")
    norm = layer.get_submodule("post_attention_layernorm")
    handles: list[Any] = []

    def attention_post(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.x = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "dx", preserve_layout_cpu(grad))
                )
            )
        return None

    def norm_pre(module, args, kwargs):
        capture.norm_calls += 1
        if capture.norm_calls > 1:
            raise ResidualExtractionError(
                "post_attention_layernorm ran more than once inside one "
                "derivation; the task is defined as a single invocation"
            )
        tensor = args[0] if args else kwargs["hidden_states"]
        capture.summed = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            # `summed`'s own gradient: the contract's `dtotal`.
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(
                        capture, "dsummed_total", preserve_layout_cpu(grad)
                    )
                )
            )
        return None

    def norm_post(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.normalized = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            handles.append(
                tensor.register_hook(
                    lambda grad: setattr(
                        capture, "dnormalized", preserve_layout_cpu(grad)
                    )
                )
            )
        return None

    handles.append(attention.register_forward_hook(attention_post, with_kwargs=True))
    handles.append(norm.register_forward_pre_hook(norm_pre, with_kwargs=True))
    handles.append(norm.register_forward_hook(norm_post, with_kwargs=True))
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def derive_residual_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    from evograd.benchmark.topdown.llama3_8b.levels.level3.prepare import prepare_layer
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ResidualExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_residual(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    # The decoder layer's input is this boundary's `residual`.
    capture.residual = preserve_layout_cpu(leaf)
    capture.require_complete()

    norm = layer.get_submodule("post_attention_layernorm")
    if norm.weight.grad is None:
        raise ResidualExtractionError("post_attention_layernorm.weight has no gradient")
    weight = preserve_layout_cpu(norm.weight)
    dweight = preserve_layout_cpu(norm.weight.grad)
    eps = float(norm.variance_epsilon)

    # `summed + mlp(...)` is the layer's output, so the gradient reaching
    # `summed` from its second consumer *is* the layer's upstream gradient.
    dsummed = preserve_layout_cpu(grad_output)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.post_attention_layernorm"
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "rms_norm_config_id": harvest["config_id"],
            "fusion_sites": harvest["fusion_sites"]["total"],
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
            f"attention residual add + {module_path}",
            TASK_NAME,
        ],
        "attrs": {
            "eps": eps,
            "rows": int(capture.summed.numel() // capture.summed.shape[-1]),
            "cols": int(capture.summed.shape[-1]),
            "fusion_sites_per_step": harvest["fusion_sites"]["total"],
            "directly_verified_invocations": harvest["fusion_sites"][
                "directly_verified_invocations"
            ],
        },
        "inputs": {"x": capture.x, "r": capture.residual, "weight": weight},
        "outputs": {"out": capture.normalized, "summed": capture.summed},
        "output_grads": {"dout": capture.dnormalized, "dsummed": dsummed},
        "grads": {
            "dx": capture.dx,
            "dweight": dweight,
            # `summed`'s own gradient, which the contract says is
            # `dsummed + RMSNormBackward(dnormalized)` -- i.e. `dtotal`.
            "dtotal": capture.dsummed_total,
        },
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
        "not_directly_comparable": {
            "dr": (
                "the decoder layer's input feeds both the residual add and "
                "input_layernorm, so its gradient in the layer is this "
                "boundary's dresidual plus a path outside the boundary; the "
                "contract's dresidual == dx is proved against an isolated "
                "autograd reference instead"
            )
        },
        "signature": {
            group: {k: meta(v) for k, v in task_payload[group].items()}
            for group in ("inputs", "outputs", "output_grads", "grads")
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
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("rms_norm_config_id", harvest["config_id"]),
        ("fusion_sites", harvest["fusion_sites"]["total"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "derived_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = (
        f"{snapshot['representative_layer']['module_path']}.post_attention_layernorm"
    )
    if identity.get("module_path") != expected_path:
        problems.append(
            f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}"
        )
    if payload["attrs"]["eps"] != harvest["attrs"]["eps"]:
        problems.append("attrs.eps disagrees with the harvested RMSNorm")
    if payload["attrs"]["cols"] != harvest["attrs"]["normalized_size"]:
        problems.append(
            f"cols {payload['attrs']['cols']} != harvested normalized_size "
            f"{harvest['attrs']['normalized_size']}"
        )
    observed = harvest["output_shapes"][0]["shape"]
    rows = payload["attrs"]["rows"]
    if rows != observed[0] * observed[1]:
        problems.append(f"rows {rows} != harvested tokens {observed[0]} x {observed[1]}")
    return problems


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Llama suite in the declaration still match the snapshot?"""
    from evograd.benchmark import get_task
    from evograd.benchmark.topdown.llama3_8b.levels.level2.residual_rmsnorm.task import (
        LLAMA_FUSION_SITES,
    )

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    op = get_task(TASK_NAME)
    cases = op.benchmark_workloads("llama_3_8b_observed")
    problems: list[str] = []
    if len(cases) != 1:
        problems.append(f"expected one observed Llama case, found {len(cases)}")
        return problems
    case = cases[0]
    observed = harvest["output_shapes"][0]["shape"]
    for dim, expected in (
        ("rows", observed[0] * observed[1]),
        ("cols", harvest["attrs"]["normalized_size"]),
    ):
        if case.dims.get(dim) != expected:
            problems.append(f"declared {dim}={case.dims.get(dim)} != harvested {expected}")
    if case.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {case.dtype!r} != harvested {harvest['dtype']!r}")
    if LLAMA_FUSION_SITES["total"] != harvest["fusion_sites"]["total"]:
        problems.append(
            f"declared fusion sites {LLAMA_FUSION_SITES['total']} != snapshot "
            f"{harvest['fusion_sites']['total']}"
        )
    return problems
