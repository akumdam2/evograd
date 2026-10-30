"""Harvest MegaFold's site calls from one real training step; derive the tracked snapshot.

    python -m evograd.benchmark.topdown.megafold.harvest.harvest --out HARVEST.json [--write-snapshot]

One step of the verification workload (:mod:`evograd.verification.integrations.
megafold`) under the verification settings, with the capture's recorder in
harvest mode: every training call of every site, in the task arguments its
adapter hands a kernel. Calls that agree on dims, dtype, layouts and adapter
are one configuration (``harvest_roles.group_configurations``); the snapshot
carries them with their calls per step and module paths, and is hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evograd.benchmark.topdown import snapshot_path
from evograd.benchmark.topdown.common.snapshot import snapshot_hash

SCHEMA = "evograd-megafold-snapshot/1"
LEVEL1 = {"layernorm", "swiglu"}
LEVEL2 = {"megafold_layernorm_linear", "megafold_evoattention"}
NOT_BOUND = (
    "LayernormLinear with a linear bias (the triangle multiplication's combined projection)",
    "windowed (local) atom attention",
    "triangle multiplication, outer product mean and MSA pair-weighted averaging (not sites)",
    "LayerNorms outside PreLayerNorm and non-affine LayerNorms (not sites)",
    "calls under no_grad or inside a backward pass (activation-checkpoint recompute, the "
    "confidence head's diffusion mini-rollout)",
    "the template embedder and the recycling projections: no gradient reaches them in this step "
    "(templates masked, one recycling pass), so they hold no sites (workload.NOT_SITES)",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def collect(seed: int = 0) -> dict:
    import torch

    from evograd.verification.integrations.base import build_model, configure
    from evograd.verification.integrations.megafold import WORKLOAD_CONFIG, load_workload
    from evograd.verification.integrations.megafold.capture import MegaFoldCapture

    settings = configure()
    workload = load_workload()
    net = build_model(workload)
    capture = MegaFoldCapture(net).install()
    try:
        loss, primary = workload.primary_output(net, workload.batch_for(seed=seed))
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss {float(loss)}")
        loss.backward()
    finally:
        capture.remove()
    per_site: dict[str, int] = {}
    for call in capture.calls:
        per_site[call["site"]] = per_site.get(call["site"], 0) + 1
    data = workload.data_identity()
    return {"schema_version": "evograd-megafold-harvest/1",
            "workload_id": f"megafold:{data['sample']}@{data['sample_sha256'][:16]}:seed{seed}",
            "config": {**WORKLOAD_CONFIG, "describe": workload.describe()}, "data": data,
            "determinism": settings, "torch": torch.__version__, "loss": float(loss.detach()),
            "primary": {"name": workload.primary_output_name, "shape": list(primary.shape),
                        "dtype": str(primary.dtype)},
            "parameters_without_grad": sorted(n for n, p in net.named_parameters()
                                              if p.requires_grad and p.grad is None),
            "calls_per_site": per_site, "records": capture.calls, "not_bound": list(NOT_BOUND)}


def snapshot(manifest: dict) -> dict:
    from evograd.benchmark.topdown.common.harvest_roles import group_configurations

    payload = {"schema_version": SCHEMA, "manifest_hash": digest(manifest),
               "workload_id": manifest["workload_id"], "config": manifest["config"], "canonical": True,
               "data": manifest["data"], "not_bound": manifest["not_bound"], "level1": {}, "tasks": {}}
    for level, key in ((1, "level1"), (2, "tasks")):
        groups = group_configurations(manifest["records"], level, digest)
        payload[key] = {task: {"configurations": configs,
                               "frequency": sum(c["frequency"] for c in configs)}
                        for task, configs in groups.items()}
    if set(payload["level1"]) != LEVEL1 or set(payload["tasks"]) != LEVEL2:
        raise ValueError(f"observed tasks {set(payload['level1'])} / {set(payload['tasks'])}")
    payload["snapshot_hash"] = snapshot_hash(payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the full harvest (records of every call)")
    parser.add_argument("--write-snapshot", action="store_true", help="also write the tracked snapshot")
    args = parser.parse_args(argv)
    manifest = collect()
    payload = snapshot(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    if args.write_snapshot:
        snapshot_path("megafold").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"harvest": str(args.out), "loss": manifest["loss"],
                      "calls_per_site": manifest["calls_per_site"],
                      "configurations": {k: len(v["configurations"])
                                         for k, v in {**payload["level1"], **payload["tasks"]}.items()},
                      "parameters_without_grad": len(manifest["parameters_without_grad"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
