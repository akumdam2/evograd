"""A frozen warm-started state of a generated model, for verification.

DiT-S/2 initializes its adaLN-Zero gates and final layer at zero, so at the
initialization no gradient reaches its blocks, conditioning path or patch
embedding: recorded payloads there carry all-zero cotangents and cannot show a
gradient fault. Verification therefore starts from this state: the
deterministic initialization trained for ``steps`` AdamW updates on real data
order ``order`` (never a calibration, acceptance or Gate 3 order), under the
verification's deterministic settings, saved in float32 with its recipe.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evograd.verification.integrations.warm_start \\
        --model dit_s_2 --out PATH [--steps 1000 --lr 1e-4 --order 909]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

SCHEMA = "evograd-generated-init-state/1"


def default_path(model: str) -> Path:
    from evograd.benchmark.topdown.common.data_paths import prepared_dir

    return Path(os.environ.get(f"EVOGRAD_{model.upper()}_WARM_START",
                               Path(prepared_dir(model)) / "warm_start.pt"))


def warm_start(model_name: str, out=None, *, steps: int = 1000, lr: float = 1e-4, order: int = 909,
               device: str = "cuda", log=print) -> str:
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
    from evograd.verification import gate3
    from evograd.verification.integrations.base import configure

    settings = configure()
    workload = GeneratedWorkload.from_config({"model": model_name, "device": device,
                                              "input_mode": "real", "stream": True})
    torch.manual_seed(workload.spec.seed)
    model = workload.build(KernelSet(registry=workload.site_registry)).train()
    optimizer_spec = gate3.adamw(lr=lr)
    optimizer = optimizer_spec.build(model)
    batches = workload.train_batches(order)
    losses, started = [], time.time()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = workload.loss(model, batches.batch(step))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            log(f"warm start step {step + 1}/{steps} loss {losses[-1]:.4f} "
                f"({time.time() - started:.0f}s)")
    recipe = {"from": "the deterministic initialization (spec seed)", "steps": steps,
              "optimizer": dict(optimizer_spec.config), "data_order": order,
              "data": workload.data_identity(), "workload": workload.spec.workload_id,
              "execution": settings, "losses": losses}
    state = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    path = Path(out) if out is not None else default_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({"schema": SCHEMA, "model": model_name, "recipe": recipe, "state_dict": state},
                   temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="dit_s_2")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--order", type=int, default=909)
    args = parser.parse_args(argv)
    path = warm_start(args.model, args.out, steps=args.steps, lr=args.lr, order=args.order,
                      log=lambda *p: print(time.strftime("%H:%M:%S"), *p, flush=True))
    print(json.dumps({"warm_start": path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
