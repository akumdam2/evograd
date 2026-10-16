"""Capture the representative layer's real inputs, outputs and gradients from the full run.

    PYTHONPATH=src python -m evograd.benchmark.topdown.olmoe_1b_7b.layer.capture \
        --harvest results/experiments/olmoe_1b_7b_cases/<run>/harvest.json \
        --out-dir results/experiments/olmoe_1b_7b_cases/<run>

One canonical Level-3 step (loss + backward) with layer 8 under capture
(``layer8.pt`` plus a JSON sidecar). The layer is selected *from the harvest
manifest*: the manifest's own hash is checked, its workload must be the one
about to run, and the hook goes on the module path of the observed
``decoder_layer`` event at that index -- so an artifact cannot describe a layer
the canonical run did not execute.

Captured, each detached, cloned and moved to CPU inside the hook that sees it:
the positional and keyword arguments Transformers passed (including the rotary
tables and ``attention_mask=None``), the output, the upstream gradient the
full-model backward delivered, the gradient produced for the input, the
weights, and the gradient the full backward left on each weight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

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
from ..architecture import REPRESENTATIVE_LAYERS, layer_kind
from ..harvest.manifest import semantic_hash


class CaptureError(RuntimeError):
    """The capture could not be performed as specified."""


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    recomputed = semantic_hash(manifest)
    if recomputed != manifest.get("manifest_hash"):
        raise CaptureError(f"{path}: manifest hash does not match its own content (stored "
                           f"{manifest.get('manifest_hash')}, recomputed {recomputed})")
    return manifest


def select_layer_event(manifest: dict[str, Any], layer_index: int) -> dict[str, Any]:
    events = [e for e in manifest["events"]
              if e["task"] == "decoder_layer" and e["layer_index"] == layer_index]
    if len(events) != 1:
        raise CaptureError(f"the manifest has {len(events)} decoder_layer events at index "
                           f"{layer_index}; the artifact must describe exactly one invocation")
    return events[0]


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
        missing = [name for name in ("args", "kwargs", "output", "grad_output", "grad_input")
                   if getattr(self, name) is None]
        if missing:
            raise CaptureError(f"{self.module_path}: capture is incomplete, missing {missing}")


@contextmanager
def capture_decoder_layer(model, module_path: str) -> Iterator[_Capture]:
    """Watch one decoder layer for the duration of the block; always unhooked."""
    layer = model.get_submodule(module_path)
    capture = _Capture(module_path=module_path)

    def pre_hook(module, args, kwargs):
        capture.forward_calls += 1
        if capture.forward_calls > 1:
            raise CaptureError(f"{module_path} ran {capture.forward_calls} times inside one "
                               "capture; the artifact is defined as a single invocation")
        capture.args = to_cpu(args, where="$args")
        capture.kwargs = to_cpu(kwargs, where="$kwargs")
        hidden_states = args[0]
        if hidden_states.requires_grad:
            capture.handles.append(hidden_states.register_hook(
                lambda grad: capture.__setattr__("grad_input", grad.detach().clone().to("cpu"))))
        return None

    def post_hook(module, args, kwargs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        capture.output = to_cpu(tensor, where="$output")
        if tensor.requires_grad:
            capture.handles.append(tensor.register_hook(
                lambda grad: capture.__setattr__("grad_output", grad.detach().clone().to("cpu"))))
        return None

    handles = [layer.register_forward_pre_hook(pre_hook, with_kwargs=True),
               layer.register_forward_hook(post_hook, with_kwargs=True)]
    try:
        yield capture
    finally:
        for handle in handles + capture.handles:
            handle.remove()
        capture.handles.clear()


def layer_state(layer) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The layer's weights and the gradients the full-model backward left on them."""
    missing = [name for name, param in layer.named_parameters() if param.grad is None]
    if missing:
        raise CaptureError(f"these layer parameters have no gradient after the full backward: "
                           f"{missing}")
    state = {n: p.detach().clone().to("cpu") for n, p in layer.named_parameters()}
    grads = {n: p.grad.detach().clone().to("cpu") for n, p in layer.named_parameters()}
    return state, grads


def auxiliary_statistics(router_logits, *, num_experts: int, top_k: int, coef: float,
                         layer_index: int) -> dict[str, Any]:
    """The load-balancing loss's global statistics, for one layer's replay.

    ``load_balancing_loss_func`` sums, over every layer and token, the top-k
    assignment counts and the softmax probabilities per expert, divides both by
    the total row count and dots them. Its gradient into layer ``layer_index``'s
    router logits therefore depends on the global assignment fractions
    (``tokens_per_expert``, non-differentiable) and on nothing else of the other
    layers; their probability sums only move the loss *value*. Both are kept, so
    the replay rebuilds the identical autograd chain (``prepare.auxiliary_objective``).
    The arithmetic mirrors the function's unmasked branch exactly.
    """
    import torch.nn.functional as F

    device = router_logits[0].device
    tokens_per_expert_sum = torch.zeros(num_experts, dtype=torch.float32, device=device)
    other_prob_sum = torch.zeros(num_experts, dtype=torch.float32, device=device)
    total_rows = 0.0
    with torch.no_grad():
        for index, layer_gate in enumerate(router_logits):
            routing_weights = F.softmax(layer_gate.to(device), dim=-1)
            _, selected = torch.topk(routing_weights, top_k, dim=-1)
            tokens_per_expert_sum = tokens_per_expert_sum + torch.bincount(
                selected.reshape(-1), minlength=num_experts).float()
            if index != layer_index:
                other_prob_sum = other_prob_sum + routing_weights.float().sum(dim=0)
            total_rows = total_rows + routing_weights.shape[0]
    return {
        "coef": float(coef),
        "num_experts": int(num_experts),
        "total_rows": float(total_rows),
        "tokens_per_expert": (tokens_per_expert_sum / total_rows).detach().clone().to("cpu"),
        "router_prob_sum_other_layers": other_prob_sum.detach().clone().to("cpu"),
        "router_logits_dtype": str(router_logits[layer_index].dtype),
    }


def run_capture(spec=None, *, manifest_path: Path,
                layers: Sequence[int] | None = None) -> dict[int, tuple[LayerArtifact, dict]]:
    """Run the canonical step once with every requested layer under capture."""
    from ..declaration import WORKLOAD
    from ..levels.level3.model import build_model, make_inputs, require_transformers, training_step
    from ..levels.level3.smoke import environment_info, gradient_coverage, workload_info

    spec = WORKLOAD.resolve(spec)
    require_transformers()
    layers = list(layers if layers is not None else sorted(REPRESENTATIVE_LAYERS.values()))
    manifest = load_manifest(manifest_path)
    if manifest["workload_id"] != spec.workload_id:
        raise CaptureError(f"the manifest describes {manifest['workload_id']!r} but this capture "
                           f"would run {spec.workload_id!r}")
    events = {index: select_layer_event(manifest, index) for index in layers}
    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        raise CaptureError("the canonical capture runs on CUDA and no CUDA device is visible")
    if spec.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model = build_model(spec)
    input_ids, labels = make_inputs(spec)
    effective = WORKLOAD.effective_settings(model, spec)
    effective["input_ids_checksum"] = int(input_ids.sum().item())
    problems = WORKLOAD.check_settings(effective, spec)
    if problems:
        raise CaptureError("the built model does not match the requested workload: "
                           + "; ".join(problems))

    with ExitStack() as stack:
        captures = {index: stack.enter_context(
            capture_decoder_layer(model, events[index]["module_path"])) for index in layers}
        outputs = training_step(model, input_ids, labels)
    for capture in captures.values():
        capture.require_complete()
    if spec.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    loss_value = float(outputs.loss.detach().float().item())
    coverage = gradient_coverage(model)
    router_logits = getattr(outputs, "router_logits", None)
    if spec.arch.get("output_router_logits") and not router_logits:
        raise CaptureError("output_router_logits is on but the step returned no router "
                           "logits; the auxiliary loss's statistics cannot be captured")

    results: dict[int, tuple[LayerArtifact, dict]] = {}
    for index in layers:
        event, capture = events[index], captures[index]
        state, param_grads = layer_state(model.get_submodule(event["module_path"]))
        auxiliary = None
        if router_logits:
            auxiliary = auxiliary_statistics(
                router_logits, num_experts=int(spec.arch["num_experts"]),
                top_k=int(spec.arch["num_experts_per_tok"]),
                coef=float(spec.arch["router_aux_loss_coef"]), layer_index=index)
            auxiliary["aux_loss"] = (float(outputs.aux_loss.detach().float().item())
                                     if getattr(outputs, "aux_loss", None) is not None else None)
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "identity": {
                "workload_id": spec.workload_id,
                "workload_hash": spec.workload_hash,
                "config_hash": spec.config_hash,
                "manifest_hash": manifest["manifest_hash"],
                "manifest_path": str(manifest_path),
                "layer_index": index,
                "layer_kind": layer_kind(index, spec.arch),
                "module_path": event["module_path"],
                "event_ordinal": event["ordinal"],
                "module_class": event["module_class"],
                "experts_implementation": spec.experts_implementation,
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
            "auxiliary": auxiliary,
        }
        payload["content_hash"] = content_hash(payload)
        payload["artifact_hash"] = artifact_hash(payload)
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
                "elements": sum(t.numel() for t in state.values()),
                "bytes": sum(t.numel() * t.element_size() for t in state.values()),
                "names": sorted(state),
            },
            "environment": environment_info(spec),
            "effective": effective,
            "auxiliary": (None if auxiliary is None else
                          {k: (describe(v) if torch.is_tensor(v) else v)
                           for k, v in auxiliary.items()}),
            "full_model_validation": {"loss": loss_value,
                                      "loss_is_finite": bool(torch.isfinite(outputs.loss.detach())),
                                      **coverage},
            "diagnostics": {
                "note": "diagnostic only -- one captured step, not a benchmark result",
                "wall_time_s": elapsed,
                "peak_allocated_bytes": (torch.cuda.max_memory_allocated()
                                         if spec.device.startswith("cuda") else None),
            },
        }
        results[index] = (LayerArtifact(payload), metadata)
    return results


def write_captures(results: dict[int, tuple[LayerArtifact, dict]], out_dir: Path) -> list[Path]:
    """``layer<i>.pt`` and ``layer<i>.json`` per captured layer."""
    out_dir = Path(out_dir)
    written = []
    for index, (artifact, metadata) in sorted(results.items()):
        path = artifact.save(out_dir / f"layer{index}.pt")
        metadata["artifact"] = {"path": str(path), "bytes": path.stat().st_size}
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
        written += [path, sidecar]
    return written


def main(argv: list[str] | None = None) -> int:
    from ..cli import add_override_arguments, resolve_spec

    parser = argparse.ArgumentParser(prog="python -m evograd.benchmark.topdown.olmoe_1b_7b."
                                          "layer.capture", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, action="append", default=None,
                        help="layer index (repeatable); default 8")
    add_override_arguments(parser)
    args = parser.parse_args(argv)
    spec = resolve_spec(args)
    try:
        results = run_capture(spec, manifest_path=args.harvest, layers=args.layer)
    except (CaptureError, ArtifactError) as exc:
        print(f"capture failed, nothing written: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    for path in write_captures(results, args.out_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
