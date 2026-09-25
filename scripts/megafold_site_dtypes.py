"""Record the dtype each MegaFold patch site actually receives in one forward.

Built through the tier-3 CLI's own parser and adapter so the model is exactly
the one ``tier3-bench --model megafold`` runs (same autocast, checkpointing,
recycling pins). Forward pre-hooks on the site module classes record input
dtypes; nothing is patched. Output: JSON of {site_class: {dtypes, count}}.
"""
import argparse, inspect, json, sys
from collections import defaultdict
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="megafold"); ap.add_argument("--residues", type=int, default=384)
ap.add_argument("--provider", default="megafold"); ap.add_argument("--out", required=True)
ap.add_argument("--identity", action="store_true", help="patch every site with the identity control and hook the site wrappers")
a = ap.parse_args()

from evograd.evaluation.tier3.cli import _parser
from evograd.evaluation.tier3.workloads import tier3_adapter
from evograd.evaluation.tier3.patch import KernelSet
from evograd.evaluation.tier3.model import build_with_provenance

args = _parser().parse_args(["--model", a.model, "--residues", str(a.residues), "--baseline", "none"])
wl = tier3_adapter(a.model).build(args)
kernels = KernelSet(registry=wl.site_registry)
if a.identity:
    from evograd.evaluation.tier3.patch import identity_control_kernels
    from evograd.benchmark import TASKS
    kernels = identity_control_kernels(TASKS, registry=wl.site_registry)
kw = {"provider": a.provider} if "provider" in inspect.signature(build_with_provenance).parameters else {}
model, prov = build_with_provenance(wl, kernels, **kw)

WATCH = {"LayernormLinear", "FusedTransition", "FusedSwiGLU", "Attend", "LayerNorm",
         "AttentionPairBias", "TriangleAttention", "TritonEvoformer"}
seen = defaultdict(lambda: {"dtypes": set(), "count": 0, "example_shape": None, "by_dtype": defaultdict(int), "shapes_by_dtype": defaultdict(set)})
def hook(name):
    def f(mod, inputs):
        ts = [t for t in inputs if torch.is_tensor(t)]
        if ts:
            seen[name]["dtypes"].update(str(t.dtype) for t in ts)
            seen[name]["by_dtype"][str(ts[0].dtype)] += 1
            seen[name]["shapes_by_dtype"][str(ts[0].dtype)].add(tuple(ts[0].shape))
            seen[name].setdefault("by_dtype_shape", defaultdict(int))[str(ts[0].dtype)+" "+str(tuple(ts[0].shape))] += 1
            seen[name]["count"] += 1
            if seen[name]["example_shape"] is None: seen[name]["example_shape"] = list(ts[0].shape)
    return f
for path, m in model.named_modules():
    n = type(m).__name__
    if n in WATCH or (a.identity and n.startswith("_Patched")): m.register_forward_pre_hook(hook(n))

batch = wl.batch_for(seed=1)
loss = wl.loss(model, batch)
report = {"model": a.model, "residues": a.residues, "provider": a.provider,
          "declared_case_dtype": str(getattr(wl, "dtype", None)),
          "autocast_dtype": str(getattr(wl, "dtype", None)),
          "loss": float(loss.detach()),
          "sites": {k: {"dtypes": sorted(v["dtypes"]), "count": v["count"], "example_shape": v["example_shape"],
                 "by_dtype": dict(v["by_dtype"]),
                 "by_dtype_shape": dict(v.get("by_dtype_shape", {})),
                 "shapes_by_dtype": {dt: sorted(list(map(list, sh)))[:6] for dt, sh in v["shapes_by_dtype"].items()}} for k, v in seen.items()},
          "provenance": prov.to_dict() if hasattr(prov, "to_dict") else str(prov)}
json.dump(report, open(a.out, "w"), indent=2); print(json.dumps(report["sites"], indent=1))
