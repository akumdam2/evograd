"""Execution-derived Level-1/2 cases from a complete generated training step."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from evograd.benchmark.topdown.common.snapshot import snapshot_hash
from evograd.benchmark.topdown import snapshot_path
from .spec import GeneratedSpec, canonical, digest
from .model import build_model, make_inputs, forward_loss, dit_mlp, dit_modulation, dlrm_interaction


def _meta(t):
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
            "requires_grad": t.requires_grad}


def collect(spec, batch=None):
    model = build_model(spec)
    records = []
    handles = []
    counter = 0
    weights = {p.data_ptr(): n for n, p in model.named_parameters()}

    def record(level, task, path, dims, names, values, *, note=""):
        records.append({"level": level, "task": task, "path": path, "dims": dims,
                        "dtype": str(values[0].dtype).removeprefix("torch."),
                        "inputs": [_meta(v) for v in values],
                        "source_inputs": [_meta(v) for v in values],
                        "input_adapter": "identity",
                        "strides": {n: list(v.stride()) for n, v in zip(names, values)},
                        "provenance": {"component": task, "free": {},
                                       "source": "execution", "note": note}})

    for path, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            def conv_hook(m, args, path=path):
                x = args[0]
                b, c, y, xx = x.shape
                ky, kx = m.kernel_size
                sy, sx = m.stride
                py, px = m.padding
                if ky != kx or sy != sx or py != px or m.dilation != (1, 1):
                    raise ValueError("Conv2D shape outside spatial_conv2d contract")
                dims = dict(B=b, C=c, Y=y, X=xx, O=m.out_channels, CG=c // m.groups,
                            K=ky, OY=(y + 2 * py - ky) // sy + 1,
                            OX=(xx + 2 * px - kx) // sx + 1, S=sy, P=py, G=m.groups)
                record(1, "spatial_conv2d", path, dims, ("x", "weight", "bias"),
                       (x, m.weight, m.bias))
            handles.append(module.register_forward_pre_hook(conv_hook))
        elif isinstance(module, torch.nn.EmbeddingBag):
            def bag_hook(m, args, path=path):
                indices, offsets = args[:2]
                b = offsets.numel()
                width = indices.numel() // b
                if width * b != indices.numel():
                    raise ValueError("nonuniform bag lengths outside current contract")
                record(1, "embedding_bag", path,
                       dict(N=m.num_embeddings, D=m.embedding_dim, B=b, M=width,
                            L=indices.numel()),
                       ("weight", "indices", "offsets"), (m.weight, indices, offsets))
            handles.append(module.register_forward_pre_hook(bag_hook))
    if spec.model == "dit_s_2":
        for i, block in enumerate(model.blocks):
            def mod_hook(c, weight, bias, i=i):
                b, h = c.shape
                record(2, "dit_s_2_dit_modulation", f"blocks.{i}.modulation",
                       dict(B=b, H=h, J=weight.shape[0]),
                       ("c", "weight", "bias"), (c, weight, bias))
                return dit_modulation(c, weight, bias)
            block.mod_fn = mod_hook
            def mlp_hook(x, w1, b1, w2, b2, i=i):
                b, t, h = x.shape
                record(2, "dit_s_2_dit_mlp", f"blocks.{i}.mlp", dict(B=b, T=t, H=h, I=w1.shape[0]),
                       ("x", "w1", "b1", "w2", "b2"), (x, w1, b1, w2, b2))
                return dit_mlp(x, w1, b1, w2, b2)
            block.mlp_fn = mlp_hook
    else:
        def interaction_hook(dense, sparse):
            b, s, d = sparse.shape
            record(2, "torchrec_dlrm_dlrm_interaction", "interaction", dict(
                B=b, S=s, D=d, O=d + s * (s + 1) // 2),
                ("dense", "sparse"), (dense, sparse))
            return dlrm_interaction(dense, sparse)
        model.interaction_fn = interaction_hook

    original_linear = F.linear
    def observed_linear(x, weight, bias=None):
        nonlocal counter
        path = weights.get(weight.data_ptr(), f"functional_linear_{counter}")
        counter += 1
        # Level-1 linear contracts flatten leading dimensions. This adapter is
        # recorded explicitly so shape and layout remain auditable.
        flat = x.reshape(-1, x.shape[-1])
        vals = (flat, weight, bias) if bias is not None else (flat, weight)
        task = "linear" if bias is not None else "linear_no_bias"
        record(1, task, path, dict(M=flat.shape[0], K=flat.shape[1], N=weight.shape[0]),
               ("x", "weight", "bias"), vals,
               note="leading dimensions flattened for Level-1 contract")
        records[-1]["source_inputs"][0] = _meta(x)
        records[-1]["input_adapter"] = ("flatten_leading_view" if flat.data_ptr() == x.data_ptr()
                                        else "flatten_leading_copy")
        return original_linear(x, weight, bias)

    try:
        F.linear = observed_linear
        if batch is None:
            if spec.input_mode == "real":
                raise ValueError("real-input harvest requires a prepared batch")
            batch = make_inputs(spec, 0)
        loss, output = forward_loss(model, batch, spec.model)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite model loss")
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        nonfinite = [n for n, p in model.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
        if missing or nonfinite:
            raise ValueError(f"invalid training gradients: missing={missing}, nonfinite={nonfinite}")
        expected = 2 * spec.arch["depth"] if spec.model == "dit_s_2" else 1
        if sum(row["level"] == 2 for row in records) != expected:
            raise ValueError("Level-2 invocation count differs from architecture")
        return {"schema_version": "evograd-generated-harvest/1",
                "workload_id": spec.workload_id, "config": spec.to_config(),
                "architecture": spec.arch, "canonical": spec.is_canonical,
                "execution_scope": "model", "loss": float(loss.detach()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "torch": torch.__version__, "records": records, "backward_verified": True,
                "not_bound": (["attention", "affine-free layer norm", "SiLU",
                               "tanh-approximate GELU", "diffusion loss", "residual modulation"]
                              if spec.model == "dit_s_2" else
                              ["ReLU", "binary cross entropy", "distributed sharding and fused sparse optimizer"])}
    finally:
        F.linear = original_linear
        for handle in handles:
            handle.remove()


def snapshot(manifest):
    from evograd.benchmark.topdown.common.harvest_roles import group_configurations

    groups = {level: group_configurations(manifest["records"], level, digest) for level in (1, 2)}
    payload = {"schema_version": f"evograd-{manifest['config']['model']}-snapshot/1",
               "manifest_hash": digest(manifest), "workload_id": manifest["workload_id"],
               "config": manifest["config"], "canonical": manifest["canonical"],
               "not_bound": manifest["not_bound"], "level1": {}, "tasks": {}}
    for level, name in ((1, "level1"), (2, "tasks")):
        payload[name] = {task: {"configurations": configs,
                                "frequency": sum(c["frequency"] for c in configs)}
                         for task, configs in groups[level].items()}
    expected = ({"linear", "spatial_conv2d"} if manifest["config"]["model"] == "dit_s_2"
                else {"linear", "embedding_bag"})
    if set(payload["level1"]) != expected:
        raise ValueError(f"primitive observation mismatch: {set(payload['level1'])} != {expected}")
    payload["snapshot_hash"] = snapshot_hash(payload)
    return payload


def main(model):
    parser = argparse.ArgumentParser(description="Whole-model generated-input forward/backward harvest")
    parser.add_argument("--config", type=Path, help="JSON execution overrides")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--real-input", action="store_true", help="harvest from prepared real inputs")
    parser.add_argument("--data", type=Path, help="prepared real-input artifact path")
    parser.add_argument("--criteo-dir", type=Path, help="TorchRec-preprocessed Criteo Kaggle NumPy directory")
    args = parser.parse_args()
    config = canonical(model).to_config()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    if config["model"] != model:
        raise ValueError("config names a different model")
    config["input_mode"] = "real" if args.real_input else "generated"
    spec = GeneratedSpec(**config)
    if args.write_snapshot and not spec.is_canonical:
        raise ValueError("only a canonical execution can publish the tracked snapshot")
    batch = None
    if args.real_input:
        from .data import ensure_pinned_data, load_data
        from types import SimpleNamespace
        path = ensure_pinned_data(spec, args.data, batch_count=1, source_dir=args.criteo_dir)
        batch = {k: v.to(spec.device) for k, v in load_data(SimpleNamespace(spec=spec, data_path=path))[0]["batches"][0].items()}
    manifest = collect(spec, batch)
    payload = snapshot(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.write_snapshot:
        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.out), "canonical": spec.is_canonical,
                      "L1_configurations": sum(len(e["configurations"]) for e in payload["level1"].values()),
                      "L2_configurations": sum(len(e["configurations"]) for e in payload["tasks"].values())}))


def snapshot_main(model):
    parser = argparse.ArgumentParser(description="Derive a tracked snapshot from canonical harvest")
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.harvest.read_text())
    spec = GeneratedSpec(**manifest["config"])
    if spec.model != model or not spec.is_canonical or not manifest.get("backward_verified"):
        raise ValueError("canonical forward/backward harvest for this model required")
    if manifest["workload_id"] != spec.workload_id or manifest["architecture"] != spec.arch:
        raise ValueError("harvest does not match current architecture")
    payload = snapshot(manifest)
    if args.write:
        snapshot_path(model).write_text(json.dumps(payload, indent=2) + "\n")
    else:
        print(json.dumps(payload, indent=2))
