"""Deriving the llama3_attention case from the canonical layer artifact.

Everything here describes *the task*: where its boundary sits in the
live model, which tensors cross it, in which layout, and whether what
was captured still matches. Nothing here decides whether an
implementation is good enough -- that is
:mod:`evograd.evaluation.workloads.llama3_2_1b.level2.attention`.

Derive and verify ``llama3_attention`` from the canonical Layer-16 artifact.

    # describe the derived invocation (metadata only -- no tensors are written)
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.attention derive \
        --source results/llama3-level4/layer8.pt \
        --metadata-out results/llama3-level4/layer8-attention.json

    # check both spellings against what Transformers computed
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.attention verify \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/layer8-attention-verify.json

    # measure what tolerance a correct implementation actually needs
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.attention calibrate \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/llama3_attention-tolerance.json

Like the MLP task, nothing here writes a second tensor file. ``layer8.pt``
already holds every number this boundary needs; the invocation is re-derived by
replaying that artifact, which takes about a second, and only JSON is written.

The boundary is captured at two points inside the replay: the
``scaled_dot_product_attention`` call, which is where q, k and v are visible in
the exact layout the model presented them, and the ``o_proj`` module, which is
where the boundary's output and its upstream gradient are. The gradients of q, k
and v come from tensor hooks on those three tensors -- each is consumed only by
this attention call, so its gradient *is* the boundary's gradient.

Layout is preserved deliberately. q, k and v are non-contiguous head-major views,
and a copy that quietly made them contiguous would describe a different call, so
the CPU copies are allocated with ``empty_strided`` at the observed strides.
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
from evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-llama3-attention-task/1"

TASK_NAME = "llama3_attention"

CONTENT_KEYS = ("q", "k", "v", "o_weight", "out", "grad_output", "grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "sdpa_config_id",
    "o_proj_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

RESULT_NAMES = ("out", "dq", "dk", "dv", "do_weight")


class AttentionExtractionError(RuntimeError):
    """The attention invocation could not be derived, or does not check out."""


# --------------------------------------------------------------------------
# layout-preserving capture
# --------------------------------------------------------------------------


def preserve_layout_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """A CPU copy with the source's exact strides.

    ``.contiguous()`` here would be a silent change of contract: q, k and v
    reach SDPA as head-major transposes, and a contiguous copy is a different
    memory access pattern with different performance and different kernel
    dispatch.
    """
    source = tensor.detach()
    out = torch.empty_strided(
        source.shape, source.stride(), dtype=source.dtype, device="cpu"
    )
    out.copy_(source)
    return out


class _AttentionCapture:
    def __init__(self) -> None:
        self.q = self.k = self.v = None
        self.out = None
        self.grad_output = None
        self.grads: dict[str, torch.Tensor] = {}
        self.sdpa_calls = 0
        self.proj_calls = 0
        self.attrs: dict[str, Any] = {}
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [n for n in ("q", "k", "v", "out", "grad_output") if getattr(self, n) is None]
        missing += [n for n in ("dq", "dk", "dv") if n not in self.grads]
        if missing:
            raise AttentionExtractionError(
                f"attention capture is incomplete, missing {missing}"
            )


@contextmanager
def capture_attention(layer: torch.nn.Module) -> Iterator[_AttentionCapture]:
    """Watch the SDPA call and the output projection inside one decoder layer."""
    capture = _AttentionCapture()
    o_proj = layer.get_submodule("self_attn.o_proj")
    original = torch.nn.functional.scaled_dot_product_attention

    def sdpa_wrapper(query, key, value, *args, **kwargs):
        capture.sdpa_calls += 1
        if capture.sdpa_calls > 1:
            raise AttentionExtractionError(
                "SDPA ran more than once inside one extraction; the task is "
                "defined as a single invocation"
            )
        for name, tensor in (("q", query), ("k", key), ("v", value)):
            setattr(capture, name, preserve_layout_cpu(tensor))
            if tensor.requires_grad:
                capture.handles.append(
                    tensor.register_hook(
                        lambda grad, name=name: capture.grads.__setitem__(
                            f"d{name}", preserve_layout_cpu(grad)
                        )
                    )
                )
        capture.attrs = {
            "dropout_p": float(kwargs.get("dropout_p", 0.0)),
            "is_causal": bool(kwargs.get("is_causal", False)),
            "scale": kwargs.get("scale"),
            "enable_gqa": bool(kwargs.get("enable_gqa", False)),
            "attn_mask_provided": kwargs.get("attn_mask", None) is not None,
        }
        return original(query, key, value, *args, **kwargs)

    def post_hook(module, args, kwargs, output):
        capture.proj_calls += 1
        tensor = output[0] if isinstance(output, tuple) else output
        capture.out = preserve_layout_cpu(tensor)
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_output", preserve_layout_cpu(grad))
                )
            )
        return None

    handle = o_proj.register_forward_hook(post_hook, with_kwargs=True)
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper
    try:
        yield capture
    finally:
        torch.nn.functional.scaled_dot_product_attention = original
        handle.remove()
        for h in capture.handles:
            h.remove()
        capture.handles.clear()


# --------------------------------------------------------------------------
# derivation
# --------------------------------------------------------------------------


def derive_attention_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay Layer 14 and capture the attention boundary inside it."""
    from evograd.benchmark.topdown.llama3_2_1b.levels.level3.prepare import prepare_layer
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AttentionExtractionError(
            "the canonical derivation runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug derivation"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)
    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_attention(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    o_proj = layer.get_submodule("self_attn.o_proj")
    if o_proj.weight.grad is None:
        raise AttentionExtractionError("o_proj.weight has no gradient after backward")
    o_weight = preserve_layout_cpu(o_proj.weight)
    capture.grads["do_weight"] = preserve_layout_cpu(o_proj.weight.grad)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.self_attn"
    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "sdpa_config_id": harvest["config_id"],
            "o_proj_config_id": harvest["supporting"]["output_projection"]["config_id"],
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
            f"scaled_dot_product_attention + o_proj at {module_path}",
            TASK_NAME,
        ],
        "attrs": capture.attrs,
        "q": capture.q,
        "k": capture.k,
        "v": capture.v,
        "o_weight": o_weight,
        "out": capture.out,
        "grad_output": capture.grad_output,
        "grads": dict(capture.grads),
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
        "sdpa_attrs": capture.attrs,
        "signature": {
            name: meta(task_payload[name])
            for name in ("q", "k", "v", "o_weight", "out", "grad_output")
        },
        "gradients": {name: meta(tensor) for name, tensor in sorted(capture.grads.items())},
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
    """Every link, against the tracked snapshot. Empty means clean."""
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("sdpa_config_id", harvest["config_id"]),
        ("o_proj_config_id", harvest["supporting"]["output_projection"]["config_id"]),
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
    # The SDPA configuration is only the same configuration if its scalars match.
    for field, expected in sorted(harvest["attrs"].items()):
        got = payload["attrs"].get(field)
        if field == "scale":
            if got is None or abs(float(got) - float(expected)) > 1e-12:
                problems.append(f"attrs.scale: {got!r} != {expected!r}")
        elif got != expected:
            problems.append(f"attrs.{field}: {got!r} != {expected!r}")
    for index, name in enumerate(("q", "k", "v")):
        observed = harvest["input_shapes"][index]
        tensor = payload[name]
        if list(tensor.shape) != observed["shape"]:
            problems.append(f"{name} shape {list(tensor.shape)} != harvested {observed['shape']}")
        if list(tensor.stride()) != observed["stride"]:
            problems.append(
                f"{name} stride {list(tensor.stride())} != harvested {observed['stride']}"
            )
    expected_weight = harvest["supporting"]["output_projection"]["params"]["weight"]["shape"]
    if list(payload["o_weight"].shape) != expected_weight:
        problems.append(
            f"o_weight shape {list(payload['o_weight'].shape)} != harvested {expected_weight}"
        )
    return problems



#: The declaration's suite for *this* workload. Three of the four Level-2
#: declarations are shared with Qwen3, so ``op.benchmark[0]`` is whichever
#: architecture was harvested first -- Qwen3's shape, not Llama's. The observed
#: suite is the one this workload is declared to run at.
OBSERVED_SUITE = "llama_3_2_1b_observed"


def _declared_case(op):
    """The single ``llama_3_2_1b_observed`` workload, or a problem describing why not."""
    cases = op.benchmark_workloads(OBSERVED_SUITE)
    if len(cases) != 1:
        return None, (
            f"expected one {OBSERVED_SUITE} case on {op.name}, found {len(cases)}; "
            "the declaration carries no observed suite for this workload"
        )
    return cases[0], None


def declaration_problems(snapshot_path: Path | None = None) -> list[str]:
    """Does the Level-2 declaration still describe the canonical snapshot?"""
    from evograd.benchmark import get_task

    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    declared, missing = _declared_case(get_task(TASK_NAME))
    if declared is None:
        return [missing]
    q, k, _ = (entry["shape"] for entry in harvest["input_shapes"])
    hidden, fan_in = harvest["supporting"]["output_projection"]["params"]["weight"]["shape"]
    problems: list[str] = []
    for dim, expected in (
        ("B", q[0]),
        ("HQ", q[1]),
        ("T", q[2]),
        ("D", q[3]),
        ("HK", k[1]),
        ("QO", fan_in),
        ("H", hidden),
    ):
        if declared.dims.get(dim) != expected:
            problems.append(f"declared dim {dim}={declared.dims.get(dim)} != harvested {expected}")
    if declared.dtype != harvest["dtype"].replace("torch.", ""):
        problems.append(f"declared dtype {declared.dtype!r} != harvested {harvest['dtype']!r}")
    return problems
