"""A frozen warm-started state of a model without pretrained weights, for verification.

DiT-S/2 initializes its adaLN-Zero gates and final layer at zero, so at the
initialization no gradient reaches its blocks, conditioning path or patch
embedding: recorded payloads there carry all-zero cotangents and cannot show a
gradient fault. MegaFold's zero-initialized output layers do the same to its
MSA module, diffusion module and input embedder. Verification therefore
starts from this state: the deterministic initialization trained for
``steps`` updates on real data order ``order`` (never a calibration,
acceptance or Gate 3 order), under the verification's deterministic settings,
saved in float32 with its recipe. DiT trains with AdamW; MegaFold with its own
trainer's recipe (:func:`_megafold`).

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

from evograd.evaluation.tier3.workloads.common.init_state import SCHEMA


def default_path(model: str) -> Path:
    from evograd.benchmark.topdown.common.data_paths import prepared_dir

    return Path(os.environ.get(f"EVOGRAD_{model.upper()}_WARM_START",
                               Path(prepared_dir(model)) / "warm_start.pt"))


def _generated(model_name: str, *, steps: int, lr: float, order: int, device: str, log):
    """AdamW at ``lr`` on the generated workload's real data order."""
    from evograd.evaluation.tier3.patch import KernelSet
    from evograd.evaluation.tier3.workloads.common.generated.workload import GeneratedWorkload
    from evograd.verification import gate3

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
              "losses": losses}
    return model, recipe


def _megafold(*, steps: int, order: int, device: str, log):
    """MegaFold's own trainer (``megafold.trainer.Trainer`` at ``configs/megafold_1x1.yaml``):
    ``torch.optim.Adam`` with the trainer's default betas and eps at the yaml's
    lr, its warmup-then-decay ``LambdaLR`` (lr 0 at step 0), the gradient norm
    clipped to the yaml's ``clip_grad_norm``, the confidence head called on every
    ``confidence_head_interval``-th step from step 0, in the trainer's order
    (forward, backward, clip, step, zero, schedule). The data is the pinned
    structure, step ``t`` seeded ``order * 1_000_000 + t``. The weights trained
    are what is kept; MegaFold's EMA copy serves its validation only."""
    import inspect

    import yaml
    from torch.optim.lr_scheduler import LambdaLR

    from .base import build_model
    from .megafold import load_workload

    workload = load_workload(warm=False)
    model = build_model(workload).train()
    import megafold  # importable once the workload has prepared MegaFold
    from megafold.trainer import Trainer, default_lambda_lr_fn

    config_path = Path(megafold.__file__).resolve().parents[1] / "configs" / "megafold_1x1.yaml"
    training = yaml.safe_load(config_path.read_text())["training"]["initial_training"]
    lr, clip = float(training["lr"]), float(training["clip_grad_norm"])
    interval = int(training["confidence_head_interval"])
    adam = dict(inspect.signature(Trainer.__init__).parameters["default_optimizer_kwargs"].default)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, **adam)
    scheduler = LambdaLR(optimizer, lr_lambda=default_lambda_lr_fn)
    losses, norms, started = [], [], time.time()
    for step in range(steps):
        batch = workload.batch_for(seed=order * 1_000_000 + step)
        loss = workload.loss(model, batch, call_confidence_head=step % interval == 0)
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip)))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            log(f"warm start step {step + 1}/{steps} loss {losses[-1]:.4f} grad norm {norms[-1]:.3g} "
                f"lr {scheduler.get_last_lr()[0]:.3g} ({time.time() - started:.0f}s)")
    recipe = {"from": "the deterministic initialization (workload seed)", "steps": steps,
              "trainer": "megafold.trainer.Trainer, configs/megafold_1x1.yaml",
              "optimizer": {"name": "torch.optim.Adam", "lr": lr, **adam,
                            "schedule": "megafold.trainer.default_lambda_lr_fn (warmup 3000 from 0)",
                            "clip_grad_norm": clip},
              "confidence_head": f"called at steps % {interval} == 0",
              "data_order": order, "seeds": f"{order} * 1_000_000 + step",
              "data": workload.data_identity(), "workload": workload.describe(),
              "losses": losses, "grad_norms": norms}
    return model, recipe


def warm_start(model_name: str, out=None, *, steps: int = 1000, lr: float = 1e-4, order: int = 909,
               device: str = "cuda", log=print) -> str:
    from evograd.verification.integrations.base import configure

    settings = configure()
    if model_name == "megafold":
        model, recipe = _megafold(steps=steps, order=order, device=device, log=log)
    else:
        model, recipe = _generated(model_name, steps=steps, lr=lr, order=order, device=device, log=log)
    recipe["execution"] = settings
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
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW lr (generated models; "
                        "MegaFold takes its yaml's)")
    parser.add_argument("--order", type=int, default=909)
    args = parser.parse_args(argv)
    path = warm_start(args.model, args.out, steps=args.steps, lr=args.lr, order=args.order,
                      log=lambda *p: print(time.strftime("%H:%M:%S"), *p, flush=True))
    print(json.dumps({"warm_start": path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
