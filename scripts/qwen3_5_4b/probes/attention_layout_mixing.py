"""Does the cuDNN SDPA plan-cache bug reach the Level-3 gate for layer 15?

    python scripts/qwen3_5_4b/probes/attention_layout_mixing.py ORDER OUT.json

ORDER is a comma list of ``eager`` / ``compiled``: the ``qwen3_5_gated_attention``
production spelling and its ``torch.compile`` (what the trusted-compile control
and the ``--compile-site gated_attention`` provider run), executed forward +
backward in that order in this one process at the observed B1 x T2048 case.
Each is compared with the declaration's float32 oracle (explicit matmuls, no
SDPA, so no cuDNN state can touch it). Run once per order, each in a fresh
process: ``eager,compiled`` is the calibration child's order (native first),
``compiled,eager`` a candidate child's (the compiled kernel first, then the
boundary check's eager recomputation).
"""

from __future__ import annotations

import json
import sys

import torch

from evograd.benchmark import TASKS
from evograd.opdecl.inputs import as_output_tuple, make_case_inputs, upstream_grad_values
from evograd.opdecl.oracle import oracle, resolve_runtime_forward


def main() -> int:
    order = sys.argv[1].split(",")
    op = TASKS["qwen3_5_gated_attention"]
    workload = op.benchmark_workloads(suite="qwen3_5_4b_observed")[0]
    values = make_case_inputs(op, workload, device="cuda")
    _ref_out, ref_grads = oracle(op, values)
    eager = resolve_runtime_forward(op)
    kernels = {"eager": eager, "compiled": torch.compile(eager, dynamic=False, fullgraph=True)}
    douts = upstream_grad_values(op, values)
    active = [a for a in op.args if a.name in {x.name for x in op.active_args()}]
    report = {"order": order, "torch": torch.__version__, "runs": []}
    for name in order:
        leaves = {a.name: values[a.name].detach().clone().requires_grad_(True) for a in active}
        args = [leaves.get(a.name, values.get(a.name, getattr(a, "default", None)))
                for a in op.args]
        outputs = as_output_tuple(op, kernels[name](*args))
        grads = torch.autograd.grad(outputs, list(leaves.values()), douts)
        rel = {a.grad_name: float((g.double() - ref_grads[a.grad_name].double()).norm()
                                  / ref_grads[a.grad_name].double().norm())
               for a, g in zip(active, grads)}
        report["runs"].append({"kernel": name, "rel_l2_vs_fp32_oracle": rel})
        print(json.dumps(report["runs"][-1]), flush=True)
    json.dump(report, open(sys.argv[2], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
