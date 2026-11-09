"""Harvest MegaFold's calls from one real training step; derive the tracked snapshot.

    python -m evograd.benchmark.topdown.megafold.harvest.harvest --out HARVEST.json [--write-snapshot]
    python -m evograd.benchmark.topdown.megafold.harvest.harvest --from-harvest HARVEST.json --write-snapshot

One step of the verification workload (:mod:`evograd.verification.integrations.
megafold`) under the verification settings, with the capture's recorder in
harvest mode: every training call of every site and primitive, in the task
arguments a kernel would get. Calls that agree on dims, dtype, layouts, adapter
and static parameters are one configuration
(``harvest_roles.group_configurations``); the snapshot carries them with their
calls per step and module paths, and is hashed.

The snapshot also carries a census of every torch call the step made. A name
this module does not classify fails the snapshot rather than being dropped. The
harvest file is written first, so classifying it and rerunning with
``--from-harvest`` needs no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evograd.benchmark.topdown import snapshot_path
from evograd.benchmark.topdown.common.snapshot import snapshot_hash

SCHEMA = "evograd-megafold-snapshot/2"
LEVEL1 = {"layernorm", "layernorm_no_affine", "layernorm_no_bias", "swiglu", "glu", "sigmoid",
          "softmax", "relu", "linear", "linear_no_bias"}
LEVEL2 = {"megafold_layernorm_linear", "megafold_evoattention",
          "megafold_triangle_multiplication_outgoing", "megafold_triangle_multiplication_incoming",
          "megafold_triangle_attention_starting", "megafold_triangle_attention_ending",
          "megafold_transition", "megafold_outer_product_mean",
          "megafold_msa_pair_weighted_averaging"}
NOT_BOUND = (
    "windowed (local) atom attention",
    "attention with pair bias as a whole, the transition without its LayerNorm, AdaLN and its "
    "output gate: they run stock and their primitives are recorded at Level 1 (docs/AF3.md section 10)",
    "calls under no_grad or inside a backward pass (activation-checkpoint recompute, the "
    "confidence head's diffusion mini-rollout)",
    "the template embedder and the recycling projections: no gradient reaches them in this step "
    "(templates masked, one recycling pass), so they hold no sites (workload.NOT_SITES)",
)

#: Operators the step runs that no task binds, and why (``docs/AF3.md`` section 5).
#: Each one's observed configurations are counted in the snapshot's inventory.
UNBOUND = {
    "einsum": "a batched contraction, not matmul's [M, K] @ [K, N]; every trunk einsum is inside "
              "a Level-2 contract",
    "matmul": "batched rigid-frame and coordinate arithmetic",
    "__matmul__": "batched rigid-frame and coordinate arithmetic",
    "bmm": "batched rigid-frame and coordinate arithmetic",
    "cross_entropy": "reduction='none' or ignore_index=-1: not the generic task's mathematics",
    "mse_loss": "a loss on coordinates, no generic task",
    "cdist": "geometry on coordinates, no generic task",
    "pairwise_distance": "geometry on coordinates, no generic task",
    "normalize": "geometry on coordinates, no generic task",
    "cross": "rigid-frame geometry on coordinates, no generic task",
    "svd": "rigid alignment of coordinates, no generic task",
    "det": "rigid alignment of coordinates, no generic task",
    "embedding": "an index lookup, no arithmetic to optimize",
    "silu": "only ever inside the SwiGLU boundary, which swiglu binds",
}

#: Names that are not operators: views, creation, indexing, comparison, control.
#: Every other dunder (``__add__``, ``__getitem__``, ...) counts as one of these.
SCAFFOLDING = frozenset({
    "_set_grad_enabled", "argsort", "bitwise_and", "device", "empty", "eye", "get_device", "indices",
    "nan_to_num", "to_sparse", "tolist", "unique_consecutive", "untyped_storage",
    "abs", "add", "add_", "all", "amax", "amin", "any", "arange", "argmax", "argmin", "backward",
    "bfloat16", "bool", "broadcast_tensors", "bucketize", "cat", "ceil", "chunk", "clamp", "clamp_",
    "clamp_max", "clamp_min", "clip", "clone", "contiguous", "copy_", "cos", "cpu", "cumsum", "cuda",
    "detach", "detach_", "diagonal", "dim", "div", "div_", "double", "dropout", "eq", "exp", "expand",
    "expand_as", "fill_", "fill_diagonal_", "flatten", "flip", "float", "floor", "floor_divide",
    "full", "full_like", "gather", "ge", "gt", "index_add_", "index_select", "int", "is_floating_point",
    "isfinite", "isnan", "item", "le", "lens", "linspace", "log", "log_softmax", "logical_and",
    "logical_not", "logical_or", "long", "lt", "masked_fill", "masked_fill_", "masked_select", "max",
    "mean", "min", "movedim", "mul", "mul_", "nansum", "narrow", "ne", "neg", "new_ones", "new_tensor",
    "new_zeros", "nonzero", "numel", "one_hot", "ones", "ones_like", "pad", "permute", "pow", "prod",
    "rand", "rand_like", "randint", "randn", "randn_like", "repeat", "repeat_interleave", "requires_grad_",
    "reshape", "round", "scatter", "scatter_", "scatter_add", "searchsorted", "sign", "size", "sort",
    "split", "sqrt", "square", "squeeze", "stack", "std", "sub", "sub_", "sum", "t", "tensor", "tensor_split",
    "tile", "to", "topk", "transpose", "tril", "triu", "type", "type_as", "unbind", "unflatten", "unique",
    "unsqueeze", "var", "view", "view_as", "where", "zero_", "zeros", "zeros_like",
})


def classify(census: dict[str, int], bound) -> dict:
    """Every name the step called, as bound, unbound or scaffolding; an unknown name raises."""
    out = {"bound": {}, "unbound": {}, "scaffolding": {}}
    unknown = []
    for name, calls in sorted(census.items()):
        if name in bound:
            out["bound"][name] = calls
        elif name in UNBOUND:
            out["unbound"][name] = {"calls": calls, "reason": UNBOUND[name]}
        elif name in SCAFFOLDING or (name.startswith("__") and name.endswith("__")):
            out["scaffolding"][name] = calls
        else:
            unknown.append((name, calls))
    if unknown:
        raise ValueError(f"the step called {unknown}, which harvest.py does not classify: bind each "
                         "to a task or add it to UNBOUND or SCAFFOLDING, then rerun with --from-harvest")
    return out


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def collect(seed: int = 0) -> dict:
    import torch

    from evograd.verification.integrations.base import build_model, configure
    from evograd.verification.integrations.megafold import WORKLOAD_CONFIG, load_workload
    from evograd.verification.integrations.megafold.capture import PRIMITIVES, MegaFoldCapture

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
            "calls_per_site": per_site, "records": capture.calls, "census": capture.census,
            "unbound": capture.unbound_configurations(), "bound_primitives": sorted(PRIMITIVES),
            "not_bound": list(NOT_BOUND)}


def _require_declared_dims(section: dict) -> None:
    """A configuration without its task's dims fails every later import, this module's included."""
    from evograd.benchmark import get_task

    for task, entry in section.items():
        declared = set(get_task(task).dims)
        wrong = [sorted(c["dims"]) for c in entry["configurations"] if set(c["dims"]) != declared]
        if wrong:
            raise ValueError(f"{task} declares {sorted(declared)}, its configurations have {wrong[0]}")


def snapshot(manifest: dict) -> dict:
    from evograd.benchmark.topdown.common.harvest_roles import group_configurations

    payload = {"schema_version": SCHEMA, "manifest_hash": digest(manifest),
               "workload_id": manifest["workload_id"], "config": manifest["config"], "canonical": True,
               "data": manifest["data"], "not_bound": manifest["not_bound"], "level1": {}, "tasks": {},
               "census": classify(manifest["census"], set(manifest["bound_primitives"])),
               "unbound_configurations": manifest["unbound"]}
    for level, key in ((1, "level1"), (2, "tasks")):
        groups = group_configurations(manifest["records"], level, digest)
        payload[key] = {task: {"configurations": configs,
                               "frequency": sum(c["frequency"] for c in configs)}
                        for task, configs in groups.items()}
    if set(payload["level1"]) != LEVEL1 or set(payload["tasks"]) != LEVEL2:
        raise ValueError(f"observed tasks {set(payload['level1'])} / {set(payload['tasks'])}")
    _require_declared_dims(payload["level1"])
    _require_declared_dims(payload["tasks"])
    payload["snapshot_hash"] = snapshot_hash(payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--out", type=Path, help="run the harvest and write it here (records of every call)")
    source.add_argument("--from-harvest", type=Path, help="derive the snapshot from a saved harvest (no GPU)")
    parser.add_argument("--write-snapshot", action="store_true", help="also write the tracked snapshot")
    args = parser.parse_args(argv)
    if args.from_harvest is not None:
        manifest = json.loads(args.from_harvest.read_text())
    else:
        manifest = collect()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    payload = snapshot(manifest)
    if args.write_snapshot:
        snapshot_path("megafold").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"harvest": str(args.out or args.from_harvest), "loss": manifest["loss"],
                      "calls_per_site": manifest["calls_per_site"],
                      "configurations": {k: len(v["configurations"])
                                         for k, v in {**payload["level1"], **payload["tasks"]}.items()},
                      "parameters_without_grad": len(manifest["parameters_without_grad"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
