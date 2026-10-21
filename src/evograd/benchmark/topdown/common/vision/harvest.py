"""Observe semantic module boundaries during a real whole-model backward.

Unsupported functional boundaries are listed, never inferred as observations.
Hooks store metadata only; they retain no activation graphs.
"""

import argparse
import json
import math
from pathlib import Path
import torch
from .spec import canonical, VisionSpec, digest
from .model import build_model, make_inputs, forward
from .operators import bindings
from .contracts import SHAPES
from evograd.benchmark.topdown.common.snapshot import snapshot_hash

PRIMITIVES = {
    "convnextv2_tiny": ("linear", "layernorm", "spatial_conv2d", "grn", "gelu"),
    "qwen2_vl_2b": ("linear", "linear_no_bias", "layernorm", "rmsnorm", "quick_gelu", "gelu"),
}


def meta(t):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "requires_grad": t.requires_grad,
    }


def collect(spec):
    model = build_model(spec)
    records = []
    handles = []

    def record(level, task, path, dims, values, names, free=None):
        source_inputs = [meta(x) for x in values]
        adapter = "identity"
        if level == 1 and task in (
            "linear",
            "linear_no_bias",
            "layernorm",
            "rmsnorm",
            "gelu",
            "quick_gelu",
        ):
            primary = values[0]
            packed = primary.reshape(-1, primary.shape[-1])
            adapter = (
                "flatten_leading_view"
                if packed.untyped_storage().data_ptr() == primary.untyped_storage().data_ptr()
                else "flatten_leading_copy"
            )
            values = [packed, *values[1:]]
        records.append(
            {
                "level": level,
                "task": task,
                "path": path,
                "dims": dims,
                "dtype": str(values[0].dtype).removeprefix("torch."),
                "inputs": [meta(x) for x in values],
                "source_inputs": source_inputs,
                "input_adapter": adapter,
                "strides": {name: list(value.stride()) for name, value in zip(names, values)},
                "provenance": {"component": task, "free": free or {}, "source": "execution"},
            }
        )

    for site, path, owner, first, pack in bindings(model):

        def hook(module, args, site=site, path=path, pack=pack):
            values = pack(args[0])
            x = values[0]
            if site == "convnext_mlp":
                b, y, w, h = x.shape
                dims = dict(B=b, Y=y, X=w, H=h, I=values[1].shape[0])
            elif site == "text_mlp":
                b, t, h = x.shape
                dims = dict(B=b, T=t, H=h, I=values[1].shape[0])
            else:
                t, h = x.shape
                dims = dict(T=t, H=h, I=values[1].shape[0])
            record(2, spec.model + "_" + site, path, dims, values, [n for n, _ in SHAPES[site]])

        handles.append(first.register_forward_pre_hook(hook))
    for path, module in model.named_modules():
        kind = type(module).__name__

        def hook(m, args, path=path, kind=kind):
            x = args[0]
            if isinstance(m, torch.nn.Linear):
                task = "linear" if m.bias is not None else "linear_no_bias"
                values = [x, m.weight] + ([m.bias] if m.bias is not None else [])
                record(
                    1,
                    task,
                    path,
                    dict(M=math.prod(x.shape[:-1]), K=x.shape[-1], N=m.out_features),
                    values,
                    ["x", "weight", "bias"],
                )
            elif isinstance(m, torch.nn.Conv2d):
                if (
                    m.dilation != (1, 1)
                    or len(set(m.kernel_size)) > 1
                    or len(set(m.stride)) > 1
                    or len(set(m.padding)) > 1
                ):
                    raise ValueError(f"unsupported Conv2D contract at {path}")
                b, c, y, w = x.shape
                k = m.kernel_size[0]
                s = m.stride[0]
                p = m.padding[0]
                record(
                    1,
                    "spatial_conv2d",
                    path,
                    dict(
                        B=b,
                        C=c,
                        Y=y,
                        X=w,
                        O=m.out_channels,
                        CG=c // m.groups,
                        K=k,
                        OY=(y + 2 * p - k) // s + 1,
                        OX=(w + 2 * p - k) // s + 1,
                        S=s,
                        P=p,
                        G=m.groups,
                    ),
                    [x, m.weight, m.bias],
                    ["x", "weight", "bias"],
                )
            elif kind == "ConvNextV2GRN":
                b, y, w, h = x.shape
                record(
                    1,
                    "grn",
                    path,
                    dict(B=b, Y=y, X=w, H=h),
                    [x, m.weight, m.bias],
                    ["x", "weight", "bias"],
                )
            elif isinstance(m, torch.nn.LayerNorm):
                if getattr(m, "data_format", None) == "channels_first":
                    x = x.permute(0, 2, 3, 1)
                record(
                    1,
                    "layernorm",
                    path,
                    dict(rows=math.prod(x.shape[:-1]), hidden=x.shape[-1]),
                    [x, m.weight, m.bias],
                    ["x", "weight", "bias"],
                    {"eps": m.eps},
                )
            elif kind in ("Qwen2RMSNorm", "Qwen2VLRMSNorm"):
                record(
                    1,
                    "rmsnorm",
                    path,
                    dict(rows=math.prod(x.shape[:-1]), hidden=x.shape[-1]),
                    [x, m.weight],
                    ["x", "weight"],
                    {"eps": m.variance_epsilon},
                )
            elif kind in ("GELUActivation", "GELU", "QuickGELUActivation"):
                task = "quick_gelu" if kind == "QuickGELUActivation" else "gelu"
                record(
                    1,
                    task,
                    path,
                    dict(rows=math.prod(x.shape[:-1]), hidden=x.shape[-1]),
                    [x],
                    ["x"],
                )

        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)) or kind in (
            "ConvNextV2GRN",
            "Qwen2RMSNorm",
            "Qwen2VLRMSNorm",
            "GELUActivation",
            "GELU",
            "QuickGELUActivation",
        ):
            handles.append(module.register_forward_pre_hook(hook))
    try:
        result = forward(model, make_inputs(spec), spec.model)
        if not torch.isfinite(result.loss):
            raise ValueError("nonfinite model loss")
        result.loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all()
        ]
        if missing or nonfinite:
            raise ValueError(
                f"invalid training gradients: missing={missing}, nonfinite={nonfinite}"
            )
        expected = (
            {"convnext_mlp": sum(spec.arch["depths"])}
            if spec.model == "convnextv2_tiny"
            else {
                "text_mlp": spec.arch["num_hidden_layers"],
                "vision_mlp": spec.arch["vision_config"]["depth"],
            }
        )
        for site, count in expected.items():
            actual = sum(r["level"] == 2 and r["task"] == spec.model + "_" + site for r in records)
            if actual != count:
                raise ValueError(f"{site}: expected {count} invocations, observed {actual}")
        return {
            "schema_version": "evograd-vision-harvest/1",
            "workload_id": spec.workload_id,
            "config": spec.to_config(),
            "architecture": spec.arch,
            "canonical": spec.is_canonical,
            "execution_scope": "model",
            "loss": float(result.loss.detach()),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "records": records,
            "backward_verified": True,
            "not_bound": (
                [
                    "attention (vision and causal text)",
                    "multimodal RoPE",
                    "Conv3D patch embedding",
                    "SwiGLU activation",
                    "cross entropy",
                ]
                if spec.model == "qwen2_vl_2b"
                else ["cross entropy", "spatial pooling", "residual addition"]
            ),
        }
    finally:
        for handle in handles:
            handle.remove()


def snapshot(manifest):
    from evograd.benchmark.topdown.common.harvest_roles import group_configurations

    groups = {level: group_configurations(manifest["records"], level, digest) for level in (1, 2)}
    payload = {
        "schema_version": f"evograd-{manifest['config']['model']}-snapshot/1",
        "manifest_hash": digest(manifest),
        "workload_id": manifest["workload_id"],
        "config": manifest["config"],
        "canonical": manifest["canonical"],
        "not_bound": manifest["not_bound"],
        "level1": {},
        "tasks": {},
    }
    for level, key in ((1, "level1"), (2, "tasks")):
        payload[key] = {
            task: {
                "configurations": configs,
                "frequency": sum(c["frequency"] for c in configs),
            }
            for task, configs in groups[level].items()
        }
    expected = set(PRIMITIVES[manifest["config"]["model"]])
    if set(payload["level1"]) != expected:
        raise ValueError(
            f'primitive observation mismatch: expected {expected}, got {set(payload["level1"])}'
        )
    payload["snapshot_hash"] = snapshot_hash(payload)
    return payload


def main(model):
    parser = argparse.ArgumentParser(description="Whole-model forward/backward harvest")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON execution overrides (reduced smoke workloads are not canonical)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args()
    config = canonical(model).to_config()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    if config["model"] != model:
        raise ValueError("config names a different model")
    spec = VisionSpec(**config)
    if args.write_snapshot and not spec.is_canonical:
        raise ValueError("only a canonical execution can publish the tracked snapshot")
    manifest = collect(spec)
    payload = snapshot(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.write_snapshot:
        from evograd.benchmark.topdown import snapshot_path

        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "canonical": spec.is_canonical,
                "L1_configurations": sum(
                    len(e["configurations"]) for e in payload["level1"].values()
                ),
                "L2_configurations": sum(
                    len(e["configurations"]) for e in payload["tasks"].values()
                ),
            }
        )
    )


def snapshot_main(model):
    parser = argparse.ArgumentParser(
        description="Derive a tracked snapshot from a canonical harvest"
    )
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.harvest.read_text())
    spec = VisionSpec(**manifest["config"])
    if spec.model != model or not spec.is_canonical or not manifest.get("backward_verified"):
        raise ValueError("a canonical forward/backward harvest for this model is required")
    if manifest["workload_id"] != spec.workload_id or manifest["architecture"] != spec.arch:
        raise ValueError("harvest configuration does not match current architecture")
    payload = snapshot(manifest)
    if args.write:
        from evograd.benchmark.topdown import snapshot_path

        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(json.dumps(payload, indent=2))
