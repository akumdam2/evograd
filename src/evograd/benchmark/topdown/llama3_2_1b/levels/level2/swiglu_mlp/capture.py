"""Deriving the llama3_swiglu_mlp case from the canonical layer artifact.

Everything here describes *the task*: where its boundary sits in the
live model, which tensors cross it, in which layout, and whether what
was captured still matches. Nothing here decides whether an
implementation is good enough -- that is
:mod:`evograd.evaluation.workloads.llama3_2_1b.level2.swiglu_mlp`.

Extract and verify ``llama3_swiglu_mlp`` from the verified Layer-16 replay.

    # describe the derived invocation (metadata only -- no tensors are written)
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.swiglu_mlp derive \
        --source results/llama3-level4/layer8.pt \
        --metadata-out results/llama3-level4/layer8-mlp.json

    # check the Level-2 declaration's reference against what the model computed
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.swiglu_mlp verify \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/layer8-mlp-verify.json

    # measure what tolerance a correct BF16 implementation actually needs
    PYTHONPATH=src python -m evograd.evaluation.workloads.llama3_2_1b.level2.swiglu_mlp calibrate \
        --source results/llama3-level4/layer8.pt \
        --report results/llama3-level4/llama3_swiglu_mlp-tolerance.json

The source is deliberately the *replay*, not the full model. The Layer-16
artifact has already been shown to reproduce the full model bitwise, so a
capture taken from replaying it inherits that guarantee while costing one layer
instead of 596M parameters -- and, more usefully, the extraction is then
reproducible on any machine that has the artifact, with no 17 GiB training step
in between.

**Nothing here writes a second tensor file.** The MLP's input, output, upstream
gradient and all three weights and their gradients already live inside
``layer8.pt``; a derived ``.pt`` would be 68 MiB of the same numbers under a
different name, and the moment one of the two is regenerated they disagree
silently. The invocation is re-derived by replaying the layer artifact whenever
it is needed, which takes about a second, and only JSON metadata and reports are
written.

The provenance chain is carried explicitly and checked, link by link:

    canonical workload
      -> harvest manifest
        -> model.layers.14
          -> Layer-16 artifact (content and identity hashes)
            -> LlamaMLP invocation
              -> llama3_swiglu_mlp

Verification asks two questions, with two different tolerances, because they are
two different questions.

**Is the extraction wired correctly?** Compared against ``llama3_swiglu_mlp_forward_hf``
-- the BF16 spelling ``LlamaMLP`` actually executes -- this is the same
computation, so it is held to the Level-3 replay tolerances: one BF16 unit
roundoff forward, one epsilon on gradients. If the weight mapping, a transpose,
or the upstream gradient were wrong, this is what would catch it.

**Does the declared contract agree?** The declaration accumulates the gate/up
product in float32 and ``LlamaMLP`` does not, so these are *deliberately not the
same computation* and a rounding-level tolerance would be the wrong instrument.
It is held to the operator's own declared BF16 tolerance -- the one the benchmark
harness gates candidates with -- because that is exactly the question being
asked: is the reference a valid answer for this operator.

Holding the second comparison to the first's tolerance would be a category
error, and loosening the first to accommodate the second would silently weaken a
check that is currently exact.
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from evograd.benchmark.topdown.llama3_2_1b.levels.level3.artifact import (
    ArtifactError,
    content_hash_over,
    identity_hash_over,
    load_canonical,
    tensor_meta,
    to_cpu,
    to_device,
)
from evograd.benchmark.topdown.llama3_2_1b.harvest.snapshot import load as load_snapshot

SCHEMA_VERSION = "evograd-llama3-mlp-task/1"

TASK_NAME = "llama3_swiglu_mlp"

CONTENT_KEYS = ("input", "output", "grad_output", "grad_input", "weights", "weight_grads")

IDENTITY_KEYS = (
    "task",
    "workload_id",
    "config_hash",
    "manifest_hash",
    "harvest_config_id",
    "frequency",
    "layer_index",
    "module_path",
    "source_content_hash",
    "source_artifact_hash",
    "provenance_kind",
)

#: HF parameter name -> the declaration's argument name. The declaration takes
#: the three matrices separately because Qwen3 stores them separately.
WEIGHT_MAP = {
    "gate_proj.weight": "gate_weight",
    "up_proj.weight": "up_weight",
    "down_proj.weight": "down_weight",
}


class MlpExtractionError(RuntimeError):
    """The MLP invocation could not be extracted, or does not check out."""


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


class _MlpCapture:
    def __init__(self) -> None:
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.grad_input: torch.Tensor | None = None
        self.grad_output: torch.Tensor | None = None
        self.calls = 0
        self.handles: list[Any] = []

    def require_complete(self) -> None:
        missing = [
            name
            for name in ("input", "output", "grad_input", "grad_output")
            if getattr(self, name) is None
        ]
        if missing:
            raise MlpExtractionError(f"MLP capture is incomplete, missing {missing}")


@contextmanager
def capture_mlp(layer: torch.nn.Module) -> Iterator[_MlpCapture]:
    """Watch the decoder layer's MLP submodule for the duration of the block."""
    mlp = layer.get_submodule("mlp")
    capture = _MlpCapture()

    def pre_hook(module, args, kwargs):
        capture.calls += 1
        if capture.calls > 1:
            raise MlpExtractionError(
                "the MLP ran more than once inside one extraction; the task "
                "artifact is defined as a single invocation"
            )
        tensor = args[0]
        capture.input = to_cpu(tensor, where="$input")
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_input", grad.detach().clone().to("cpu"))
                )
            )
        return None

    def post_hook(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.output = to_cpu(tensor, where="$output")
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: setattr(capture, "grad_output", grad.detach().clone().to("cpu"))
                )
            )
        return None

    handles = [
        mlp.register_forward_pre_hook(pre_hook, with_kwargs=True),
        mlp.register_forward_hook(post_hook, with_kwargs=True),
    ]
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()
        for handle in capture.handles:
            handle.remove()
        capture.handles.clear()


def derive_mlp_invocation(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay Layer 14 and capture the MLP invocation inside it.

    Returns the tensors in memory and a JSON-safe description of them. Nothing
    is written: ``layer8.pt`` is the authoritative tensor store, and this is a
    view of part of it.
    """
    from evograd.benchmark.topdown.llama3_2_1b.levels.level3.prepare import prepare_layer
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    # Every identity check, with no way to opt out.
    artifact = load_canonical(source, snapshot_path=snapshot_path)
    payload = artifact.payload
    identity = artifact.identity

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise MlpExtractionError(
            "the canonical extraction runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a debug extraction"
        )
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    layer, args, kwargs, grad_output, dtype = prepare_layer(payload, device)

    layer.zero_grad(set_to_none=True)
    leaf = args[0].detach().clone().requires_grad_(True)
    with capture_mlp(layer) as capture:
        out = layer(leaf, **kwargs)
        tensor = out[0] if isinstance(out, tuple) else out
        tensor.backward(grad_output)
    capture.require_complete()

    mlp = layer.get_submodule("mlp")
    weights: dict[str, torch.Tensor] = {}
    weight_grads: dict[str, torch.Tensor] = {}
    for hf_name, declared in WEIGHT_MAP.items():
        param = mlp.get_parameter(hf_name)
        weights[declared] = param.detach().clone().to("cpu")
        if param.grad is None:
            raise MlpExtractionError(f"mlp.{hf_name} has no gradient after backward")
        weight_grads[declared] = param.grad.detach().clone().to("cpu")

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    module_path = f"{identity['module_path']}.mlp"
    if module_path not in harvest["module_paths"]:
        raise MlpExtractionError(
            f"{module_path} is not among the harvested module paths for {TASK_NAME}; "
            "the snapshot and the artifact describe different runs"
        )

    task_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task": TASK_NAME,
            "workload_id": identity["workload_id"],
            "config_hash": identity["config_hash"],
            "manifest_hash": identity["manifest_hash"],
            "harvest_config_id": harvest["config_id"],
            "frequency": harvest["frequency"],
            "layer_index": identity["layer_index"],
            "module_path": module_path,
            "source_content_hash": payload["content_hash"],
            "source_artifact_hash": payload["artifact_hash"],
            "provenance_kind": "captured_from_verified_replay",
        },
        "provenance_chain": [
            f"canonical workload {identity['workload_id']}",
            f"harvest manifest {identity['manifest_hash']}",
            f"{identity['module_path']} (event ordinal {identity['event_ordinal']})",
            f"Layer-16 artifact {payload['artifact_hash']}",
            f"LlamaMLP invocation at {module_path}",
            TASK_NAME,
        ],
        "arch": {
            "hidden_size": harvest["attrs"]["hidden_size"],
            "intermediate_size": harvest["attrs"]["intermediate_size"],
            "hidden_act": harvest["attrs"]["hidden_act"],
            "dtype": dtype,
        },
        "input": capture.input,
        "output": capture.output,
        "grad_output": capture.grad_output,
        "grad_input": capture.grad_input,
        "weights": weights,
        "weight_grads": weight_grads,
    }
    # A fingerprint of the derived numbers, so two derivations on two machines
    # can be compared even though neither writes a file.
    task_payload["content_hash"] = content_hash_over(task_payload, CONTENT_KEYS)
    task_payload["derivation_hash"] = identity_hash_over(task_payload, IDENTITY_KEYS)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": task_payload["identity"],
        "provenance_chain": task_payload["provenance_chain"],
        "content_hash": task_payload["content_hash"],
        "derivation_hash": task_payload["derivation_hash"],
        "derived_from": str(source),
        "tensors_written": False,
        "arch": task_payload["arch"],
        "snapshot_hash": snapshot["snapshot_hash"],
        "signature": {
            "input": tensor_meta(capture.input),
            "output": tensor_meta(capture.output),
            "grad_output": tensor_meta(capture.grad_output),
            "grad_input": tensor_meta(capture.grad_input),
            "weights": {k: tensor_meta(v) for k, v in sorted(weights.items())},
            "weight_grads": {k: tensor_meta(v) for k, v in sorted(weight_grads.items())},
        },
        "diagnostics": {
            "note": "diagnostic only -- one extraction pass, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
            ),
        },
    }
    return task_payload, metadata


def load_or_derive(
    source: Path, *, device: str = "cuda", snapshot_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The only way to obtain the invocation: derive it from the layer artifact.

    There is no ``.pt`` to load. Keeping one would mean two files holding the
    same numbers, and nothing forcing them to stay equal.
    """
    return derive_mlp_invocation(source, device=device, snapshot_path=snapshot_path)


def check_provenance(payload: dict[str, Any], *, snapshot_path: Path | None = None) -> list[str]:
    """Every link in the chain, against the tracked snapshot. Empty means clean."""
    snapshot = load_snapshot(snapshot_path)
    harvest = snapshot["tasks"][TASK_NAME]
    identity = payload["identity"]
    problems: list[str] = []
    for field, expected in (
        ("task", TASK_NAME),
        ("workload_id", snapshot["workload_id"]),
        ("config_hash", snapshot["config_hash"]),
        ("manifest_hash", snapshot["manifest_hash"]),
        ("harvest_config_id", harvest["config_id"]),
        ("frequency", harvest["frequency"]),
        ("layer_index", snapshot["representative_layer"]["layer_index"]),
        ("provenance_kind", "captured_from_verified_replay"),
    ):
        if identity.get(field) != expected:
            problems.append(f"identity.{field}: {identity.get(field)!r} != {expected!r}")
    expected_path = f"{snapshot['representative_layer']['module_path']}.mlp"
    if identity.get("module_path") != expected_path:
        problems.append(f"identity.module_path: {identity.get('module_path')!r} != {expected_path!r}")
    if identity.get("module_path") not in harvest["module_paths"]:
        problems.append(f"{identity.get('module_path')!r} is not a harvested module path")
    arch = payload["arch"]
    for field, expected in (
        ("hidden_size", harvest["attrs"]["hidden_size"]),
        ("intermediate_size", harvest["attrs"]["intermediate_size"]),
        ("hidden_act", harvest["attrs"]["hidden_act"]),
    ):
        if arch.get(field) != expected:
            problems.append(f"arch.{field}: {arch.get(field)!r} != {expected!r}")
    for name, entries in (("input", harvest["input_shapes"]), ("output", harvest["output_shapes"])):
        expected = entries[0]["shape"]
        got = list(payload[name].shape)
        if got != expected:
            problems.append(f"{name} shape {got} != harvested {expected}")
    return problems
