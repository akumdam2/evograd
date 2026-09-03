"""Capture one decoder layer's real inputs, output and gradients from the full run.

    PYTHONPATH=src python -m evograd.bench.workloads.qwen3.levels.level3.capture \
        --layer 14 \
        --harvest results/qwen3-level4/harvest.json \
        --out results/qwen3-level4/layer14.pt

The layer is selected *from the harvest manifest*, not by index alone: the
manifest is checked for self-consistency, checked against the workload the
capture is about to run, and searched for an observed ``decoder_layer`` event at
that index. The event's module path is what the hook is installed on. So the
artifact cannot describe a layer the canonical run did not actually execute, and
its provenance points at a specific line of a specific manifest.

What is captured is everything a standalone replay needs and nothing else:

* the positional and keyword arguments Transformers really passed -- including
  ``attention_mask=None``, which is what sends SDPA down its causal path
* the layer's output
* the upstream gradient the *full-model* backward delivered to that output
* the gradient the layer produced for its input
* the layer's weights, and the gradient full-model backward left on each of them

Every one of those is detached, cloned and moved to CPU inside the hook that sees
it. Nothing here keeps a view into the running graph.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .artifact import (
    SCHEMA_VERSION,
    ArtifactError,
    LayerArtifact,
    artifact_hash,
    content_hash,
    describe,
    tensor_meta,
    to_cpu,
)
from ...harvest.manifest import semantic_hash
from ...levels.level4.model import (
    build_model,
    check_effective_settings,
    effective_settings,
    make_inputs,
    require_transformers,
    training_step,
)
from ...levels.level4.smoke import environment_info, gradient_coverage, workload_info
from ...levels.level4.spec import CANONICAL, WorkloadSpec

#: The representative layer. Deep enough that its inputs are a fully mixed
#: residual stream rather than the first block's near-embedding activations, and
#: far enough from the last layer that its upstream gradient has passed through
#: a realistic amount of the backward chain.
CANONICAL_LAYER_INDEX = 14


class CaptureError(RuntimeError):
    """The capture could not be performed as specified."""


# --------------------------------------------------------------------------
# manifest selection
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    recomputed = semantic_hash(manifest)
    if recomputed != manifest.get("manifest_hash"):
        raise CaptureError(
            f"{path}: manifest hash does not match its own content "
            f"(stored {manifest.get('manifest_hash')}, recomputed {recomputed}). "
            "The file has been edited; capturing against it would produce "
            "provenance that points at something that never ran."
        )
    return manifest


def select_layer_event(
    manifest: dict[str, Any],
    layer_index: int,
    *,
    expect_workload_id: str | None = None,
    expect_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """The observed ``decoder_layer`` event for ``layer_index``, or a clear error."""
    if expect_workload_id is not None and manifest["workload_id"] != expect_workload_id:
        raise CaptureError(
            f"manifest workload is {manifest['workload_id']!r}, expected "
            f"{expect_workload_id!r}"
        )
    if expect_manifest_hash is not None and manifest["manifest_hash"] != expect_manifest_hash:
        raise CaptureError(
            f"manifest hash is {manifest['manifest_hash']!r}, expected "
            f"{expect_manifest_hash!r}"
        )
    events = [
        event
        for event in manifest["events"]
        if event["task"] == "decoder_layer" and event["layer_index"] == layer_index
    ]
    if not events:
        observed = sorted(
            {e["layer_index"] for e in manifest["events"] if e["task"] == "decoder_layer"},
            key=lambda v: (v is None, v),
        )
        raise CaptureError(
            f"the manifest has no decoder_layer event at index {layer_index}; "
            f"observed indices are {observed}"
        )
    if len(events) > 1:  # pragma: no cover - one forward pass, one event
        raise CaptureError(
            f"the manifest has {len(events)} decoder_layer events at index "
            f"{layer_index}; the artifact must describe exactly one invocation"
        )
    return events[0]


# --------------------------------------------------------------------------
# the capture itself
# --------------------------------------------------------------------------


@dataclass
class _Capture:
    module_path: str
    args: tuple | None = None
    kwargs: dict | None = None
    output: torch.Tensor | None = None
    grad_output: torch.Tensor | None = None
    grad_input: torch.Tensor | None = None
    forward_calls: int = 0
    handles: list[Any] = field(default_factory=list)

    def require_complete(self) -> None:
        missing = [
            name
            for name in ("args", "kwargs", "output", "grad_output", "grad_input")
            if getattr(self, name) is None
        ]
        if missing:
            raise CaptureError(
                f"{self.module_path}: capture is incomplete, missing {missing}. "
                "A forward without a backward, or a layer whose input does not "
                "require grad, would produce this."
            )


@contextmanager
def capture_decoder_layer(model, module_path: str) -> Iterator[_Capture]:
    """Watch one decoder layer for the duration of the block.

    The tensor hooks that collect the gradients are registered during the forward
    pass and torn down with everything else in the ``finally``, so a run that
    raises mid-backward leaves no hook behind.
    """
    layer = model.get_submodule(module_path)
    capture = _Capture(module_path=module_path)

    def pre_hook(module, args, kwargs):
        capture.forward_calls += 1
        if capture.forward_calls > 1:
            raise CaptureError(
                f"{module_path} ran {capture.forward_calls} times inside one "
                "capture; the artifact is defined as a single invocation"
            )
        capture.args = to_cpu(args, where="$args")
        capture.kwargs = to_cpu(kwargs, where="$kwargs")
        hidden_states = args[0]
        if not torch.is_tensor(hidden_states):  # pragma: no cover - defensive
            raise CaptureError(f"{module_path}: first argument is not a tensor")
        if hidden_states.requires_grad:
            capture.handles.append(
                hidden_states.register_hook(
                    lambda grad: capture.__setattr__(
                        "grad_input", grad.detach().clone().to("cpu")
                    )
                )
            )
        return None

    def post_hook(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.output = to_cpu(tensor, where="$output")
        if tensor.requires_grad:
            capture.handles.append(
                tensor.register_hook(
                    lambda grad: capture.__setattr__(
                        "grad_output", grad.detach().clone().to("cpu")
                    )
                )
            )
        return None

    handles = [
        layer.register_forward_pre_hook(pre_hook, with_kwargs=True),
        layer.register_forward_hook(post_hook, with_kwargs=True),
    ]
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()
        for handle in capture.handles:
            handle.remove()
        capture.handles.clear()


def layer_state(layer) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The layer's weights and the gradients the full-model backward left on them."""
    state = {name: param.detach().clone().to("cpu") for name, param in layer.named_parameters()}
    missing = [name for name, param in layer.named_parameters() if param.grad is None]
    if missing:
        raise CaptureError(
            f"these layer parameters have no gradient after the full backward: "
            f"{missing}"
        )
    grads = {
        name: param.grad.detach().clone().to("cpu") for name, param in layer.named_parameters()
    }
    return state, grads


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def run_capture(
    spec: WorkloadSpec | None = None,
    *,
    manifest_path: Path,
    layer_index: int = CANONICAL_LAYER_INDEX,
    expect_workload_id: str | None = None,
    expect_manifest_hash: str | None = None,
) -> tuple[LayerArtifact, dict[str, Any]]:
    """Run the canonical step once with one layer under capture."""
    spec = (spec or CANONICAL).validate()
    require_transformers()

    manifest = load_manifest(manifest_path)
    event = select_layer_event(
        manifest,
        layer_index,
        expect_workload_id=expect_workload_id,
        expect_manifest_hash=expect_manifest_hash,
    )
    if manifest["workload_id"] != spec.workload_id:
        raise CaptureError(
            f"the manifest describes {manifest['workload_id']!r} but this capture "
            f"would run {spec.workload_id!r}; the artifact's provenance would be "
            "wrong"
        )
    module_path = event["module_path"]

    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        raise CaptureError(
            "the canonical capture runs on CUDA and no CUDA device is visible; "
            "allocate a GPU node, or pass --device cpu for a (non-canonical) debug run"
        )
    if spec.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model = build_model(spec)
    input_ids, labels = make_inputs(spec)

    effective = effective_settings(model, spec)
    effective["input_ids_checksum"] = int(input_ids.sum().item())
    problems = check_effective_settings(effective, spec)
    if problems:
        raise CaptureError(
            "the built model does not match the requested workload: " + "; ".join(problems)
        )

    with capture_decoder_layer(model, module_path) as capture:
        outputs = training_step(model, input_ids, labels)
    capture.require_complete()

    layer = model.get_submodule(module_path)
    state, param_grads = layer_state(layer)

    if spec.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    loss_value = float(outputs.loss.detach().float().item())
    coverage = gradient_coverage(model)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "workload_id": spec.workload_id,
            "workload_hash": spec.workload_hash,
            "config_hash": spec.config_hash,
            "manifest_hash": manifest["manifest_hash"],
            "manifest_path": str(manifest_path),
            "layer_index": layer_index,
            "module_path": module_path,
            "event_ordinal": event["ordinal"],
            "module_class": event["module_class"],
            "provenance_kind": "captured",
        },
        "arch": spec.arch,
        "args": capture.args,
        "kwargs": capture.kwargs,
        "output": capture.output,
        "grad_output": capture.grad_output,
        "grad_input": capture.grad_input,
        "state_dict": state,
        "param_grads": param_grads,
    }
    payload["content_hash"] = content_hash(payload)
    payload["artifact_hash"] = artifact_hash(payload)

    param_elements = sum(t.numel() for t in state.values())
    param_bytes = sum(t.numel() * t.element_size() for t in state.values())
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": payload["identity"],
        "content_hash": payload["content_hash"],
        "artifact_hash": payload["artifact_hash"],
        "workload": workload_info(spec),
        "signature": {
            "args": [describe(v) for v in capture.args],
            "kwargs": {k: describe(v) for k, v in sorted(capture.kwargs.items())},
            "output": tensor_meta(capture.output),
            "grad_output": tensor_meta(capture.grad_output),
            "grad_input": tensor_meta(capture.grad_input),
        },
        "parameters": {
            "count": len(state),
            "elements": param_elements,
            "bytes": param_bytes,
            "names": sorted(state),
            "grad_names": sorted(param_grads),
        },
        "environment": environment_info(spec),
        "effective": effective,
        "full_model_validation": {
            "loss": loss_value,
            "loss_is_finite": bool(torch.isfinite(outputs.loss.detach()).item()),
            **coverage,
        },
        "diagnostics": {
            "note": "diagnostic only -- one captured step, not a benchmark result",
            "wall_time_s": elapsed,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if spec.device.startswith("cuda") else None
            ),
        },
    }
    return LayerArtifact(payload), metadata


def summarize(metadata: dict[str, Any]) -> str:
    identity = metadata["identity"]
    signature = metadata["signature"]
    parameters = metadata["parameters"]
    lines = [
        f"captured {identity['module_class']} at {identity['module_path']} "
        f"(layer {identity['layer_index']})",
        f"  workload  {identity['workload_id']}",
        f"  manifest  {identity['manifest_hash'][:16]}...  event ordinal "
        f"{identity['event_ordinal']}",
        f"  content   {metadata['content_hash']}",
        f"  artifact  {metadata['artifact_hash']}",
        "",
        "captured call signature",
    ]
    for index, arg in enumerate(signature["args"]):
        lines.append(f"  args[{index}]  {_brief(arg)}")
    for name, value in signature["kwargs"].items():
        lines.append(f"  {name:<20} {_brief(value)}")
    lines += [
        "",
        f"  output        {_brief({'kind': 'tensor', **signature['output']})}",
        f"  grad_output   {_brief({'kind': 'tensor', **signature['grad_output']})}",
        f"  grad_input    {_brief({'kind': 'tensor', **signature['grad_input']})}",
        "",
        f"one layer: {parameters['count']} parameter tensors, "
        f"{parameters['elements']:,} elements, "
        f"{parameters['bytes'] / 2**20:.1f} MiB",
    ]
    validation = metadata["full_model_validation"]
    lines.append(
        f"full model: loss={validation['loss']}  grads "
        f"{validation['params_with_grad']}/{validation['trainable_params']}  "
        f"all-finite={validation['grads_all_finite']}"
    )
    return "\n".join(lines)


def _brief(entry: Any) -> str:
    kind = entry.get("kind")
    if kind == "tensor":
        return f"{entry['shape']} {entry['dtype'].replace('torch.', '')} {entry['device']}" + (
            " grad" if entry.get("requires_grad") else ""
        )
    if kind == "none":
        return "None"
    if kind == "scalar":
        return repr(entry["value"])
    if kind == "sequence":
        return "(" + ", ".join(_brief(item) for item in entry["items"]) + ")"
    if kind == "mapping":
        return "{" + ", ".join(f"{k}={_brief(v)}" for k, v in entry["items"].items()) + "}"
    return str(entry)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    from ...cli import add_override_arguments

    parser = argparse.ArgumentParser(
        prog="python -m evograd.bench.workloads.qwen3.levels.level3.capture",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer", type=int, default=CANONICAL_LAYER_INDEX, help="decoder layer index"
    )
    parser.add_argument(
        "--harvest",
        type=Path,
        default=Path("results/qwen3-level4/harvest.json"),
        help="the harvest manifest the layer is selected from",
    )
    parser.add_argument("--out", type=Path, required=True, help="write the .pt artifact here")
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="write the JSON sidecar here (default: --out with a .json suffix)",
    )
    parser.add_argument("--expect-workload-id", default=None)
    parser.add_argument("--expect-manifest-hash", default=None)
    add_override_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    from ...cli import resolve_spec
    from ...levels.level4.spec import WorkloadSpecError

    args = build_parser().parse_args(argv)
    try:
        spec = resolve_spec(args)
    except WorkloadSpecError as exc:
        print(f"invalid workload: {exc}", file=sys.stderr)
        return 2
    if not spec.is_canonical:
        print(
            "WARNING: non-canonical workload -- this artifact is a debug variant.\n"
            f"         canonical: {CANONICAL.workload_id}\n"
            f"         this run:  {spec.workload_id}",
            file=sys.stderr,
        )

    try:
        artifact, metadata = run_capture(
            spec,
            manifest_path=args.harvest,
            layer_index=args.layer,
            expect_workload_id=args.expect_workload_id,
            expect_manifest_hash=args.expect_manifest_hash,
        )
    except (CaptureError, ArtifactError) as exc:
        print(f"capture failed, nothing written: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    path = artifact.save(args.out)
    metadata["artifact"] = {"path": str(path), "bytes": path.stat().st_size}
    metadata_path = args.metadata_out or path.with_suffix(".json")
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2, default=str) + "\n", "utf-8")

    print(summarize(metadata))
    print(f"\nwrote {path}  ({path.stat().st_size / 2**20:.1f} MiB)")
    print(f"wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
