"""Observe semantic module boundaries during a real whole-model backward.

The tracked snapshot comes from :func:`real_run`: the verification capture's
first step (``integrations.base.real_run``: pinned pretrained weights, the
first batch of data order 101 from the frozen real-data pool, the
verification settings), with that run's identity as ``run``. Only such a
harvest writes it (``integrations.base.require_real_run``). :func:`collect` on
a spec alone (random weights, synthetic inputs) serves reduced checks.

Unsupported functional boundaries are listed, never inferred as observations.
Level-2 boundaries are the module boundaries and residual streams of
``operators``; the calls per step they must reach are ``calls_per_step``.
Hooks store metadata only (a residual stream's hooks hold its last sum and
branch until the next norm reads them); they retain no activation graphs.
"""

import argparse
import json
import math
from pathlib import Path
import torch
from .spec import digest
from .model import build_model, make_inputs, forward
from .operators import bindings, calls_per_step, residual_streams
from .contracts import BOUNDARIES, dims_of
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


def collect(spec, *, model=None, batch=None):
    """One forward and backward of ``model`` (default: built from ``spec``) on
    ``batch`` (default: ``spec``'s synthetic inputs), observed."""
    model = build_model(spec) if model is None else model
    batch = make_inputs(spec) if batch is None else batch
    records = []
    handles = []
    streams = []  # per residual stream, the last N_{k-1} input and B_k output (dropped after)
    # The mask each Qwen2-VL text attention call receives. Right-padded rows (the
    # real pool, the synthetic fixture) give an explicit one, which the
    # text_attention_block boundary requires; unpadded rows (None: SDPA
    # is_causal) are refused there.
    masks = {}

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

    def boundary(site, path, values):
        args = BOUNDARIES[site].args
        tensors = [(a.name, v) for a, v in zip(args, values) if getattr(a, "shape", None)]
        free = {a.name: v for a, v in zip(args, values) if not getattr(a, "shape", None)}
        record(2, spec.model + "_" + site, path, dims_of(args, values), [v for _, v in tensors],
               [n for n, _ in tensors], free)

    for binding in bindings(model):

        def hook(module, args, kwargs, binding=binding):
            boundary(binding.site, binding.path, binding.pack(args, kwargs))

        handles.append(binding.first.register_forward_pre_hook(hook, with_kwargs=True))
    for stream in residual_streams(model):
        # Boundary k: N_k's input is the sum of B_k's output and N_{k-1}'s input.
        state = {}
        for k, (path, norm) in enumerate(stream.norms):

            def norm_hook(module, args, k=k, path=path, stream=stream, state=state):
                if k:
                    boundary(stream.site, path, (state.pop("branch"), state["stream"],
                                                 *stream.norm_args(module)))
                state["stream"] = args[0]

            handles.append(norm.register_forward_pre_hook(norm_hook))
        for _, branch in stream.branches:

            def branch_hook(module, args, output, state=state):
                state["branch"] = output[0] if isinstance(output, tuple) else output

            handles.append(branch.register_forward_hook(branch_hook))
        streams.append(state)
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

        if kind == "Qwen2VLAttention":

            def mask_hook(m, args, kwargs):
                mask = kwargs.get("attention_mask")
                key = json.dumps(None if mask is None else meta(mask))
                masks[key] = masks.get(key, 0) + 1

            handles.append(module.register_forward_pre_hook(mask_hook, with_kwargs=True))
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
        result = forward(model, batch, spec.model)
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
        expected = calls_per_step(spec.model, spec.arch)
        for site, count in expected.items():
            actual = sum(r["level"] == 2 and r["task"] == spec.model + "_" + site for r in records)
            if actual != count:
                raise ValueError(f"{site}: expected {count} invocations, observed {actual}")
        return {
            "schema_version": "evograd-vision-harvest/2",
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
            **(
                {"text_attention_masks": [{"mask": json.loads(k), "calls": n}
                                          for k, n in masks.items()]}
                if spec.model == "qwen2_vl_2b" else {}
            ),
            "backward_verified": True,
            "not_bound": (
                [
                    "attention, 2-D RoPE and multimodal RoPE as primitives (bound inside the "
                    "Level-2 attention boundaries only)",
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
        for state in streams:
            state.clear()


def real_run(name, *, device="cuda"):
    """The harvest of ``name``'s verification run, with its run identity as ``run``."""
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import real_run as verification_run

    workload, model, batch, run = verification_run(load(name), device=device)
    return {**collect(workload.spec, model=model, batch=batch), "run": run}


def snapshot(manifest):
    from evograd.benchmark.topdown.common.harvest_roles import group_configurations

    groups = {level: group_configurations(manifest["records"], level, digest) for level in (1, 2)}
    payload = {
        "schema_version": f"evograd-{manifest['config']['model']}-snapshot/2",
        "manifest_hash": digest(manifest),
        "workload_id": manifest["workload_id"],
        "config": manifest["config"],
        "run": manifest.get("run"),
        "not_bound": manifest["not_bound"],
        **({"text_attention_masks": manifest["text_attention_masks"]}
           if "text_attention_masks" in manifest else {}),
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
    parser = argparse.ArgumentParser(
        description="Whole-model forward/backward harvest of the verification run (order 101)"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-snapshot", action="store_true")
    args = parser.parse_args()
    manifest = real_run(model)
    payload = snapshot(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.write_snapshot:
        from evograd.benchmark.topdown import snapshot_path
        from evograd.verification.integrations import load
        from evograd.verification.integrations.base import require_real_run

        require_real_run(load(model), manifest)
        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "workload_id": manifest["workload_id"],
                "workload": manifest["run"]["workload"],
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
        description="Derive a tracked snapshot from a harvest of the verification run"
    )
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.harvest.read_text())
    from evograd.verification.integrations import load
    from evograd.verification.integrations.base import require_real_run

    require_real_run(load(model), manifest)
    payload = snapshot(manifest)
    if args.write:
        from evograd.benchmark.topdown import snapshot_path

        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(json.dumps(payload, indent=2))
