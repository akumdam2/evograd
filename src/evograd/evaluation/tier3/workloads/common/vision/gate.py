"""Frozen native/compiled calibration for whole-model integration and training.

Calibration and evaluation use disjoint generated or prepared batches. Gate 2 checks loss, logits,
input gradients and every parameter gradient. Gate 3 checks every loss in the
independently updated trajectory (RMS and max absolute difference). Controls
must repeat bitwise before they can establish budgets. No windows or trimming.
"""

from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import torch
from evograd.benchmark.topdown.common.vision.spec import digest
from evograd.benchmark.topdown.common.vision.model import forward

SCHEMA = "evograd-vision-calibration/2"
FLOOR = 1e-5
MARGIN = 2.0
OPTIMIZER = {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01, "amsgrad": False}


def _data_mode(workload):
    if getattr(workload.spec, "input_mode", "generated") == "real":
        return "pinned_real"
    if getattr(workload.spec, "weights", "random") == "pretrained":
        return "pinned_real"
    return "prepared" if getattr(workload, "data_path", None) else "synthetic"


def _gate_name(workload):
    return "vision_model" if workload.spec.model in ("convnextv2_tiny", "qwen2_vl_2b") else "model"


def _schema(workload):
    return SCHEMA if _gate_name(workload) == "vision_model" else "evograd-generated-model-calibration/1"


def fingerprint(workload):
    result = {
        "workload_hash": workload.spec.workload_hash,
        "torch": torch.__version__,
        "device": (
            torch.cuda.get_device_name(workload.spec.device)
            if workload.spec.device.startswith("cuda")
            else "cpu"
        ),
        "cuda": torch.version.cuda,
    }
    if workload.spec.model in ("convnextv2_tiny", "qwen2_vl_2b"):
        result["transformers"] = __import__("transformers").__version__
    if (getattr(workload.spec, "input_mode", "generated") == "real"
            and workload.spec.model == "dit_s_2"):
        result["diffusers"] = __import__("diffusers").__version__
    return result


from evograd.benchmark.topdown.common.vision.data import load_data
from evograd.benchmark.topdown.common.vision.model import make_inputs


def verification_batches(workload, steps):
    """Return disjoint fixed batches for the selected execution identity."""
    if getattr(workload, "data_path", None):
        data, data_hash = getattr(workload, "load_data", load_data)(workload)
        batches = data["batches"]
    else:
        batch_fn = getattr(workload, "generated_batch_cpu", None)
        batches = [batch_fn(i) if batch_fn else make_inputs(workload.spec.replace(device="cpu"), seed=i)
                   for i in range(2 * steps)]
        data_hash = digest({
            "source": "seeded_synthetic",
            "workload_hash": workload.spec.workload_hash,
            "batch_seeds": list(range(2 * steps)),
        })
    if len(batches) < 2 * steps:
        raise ValueError("need disjoint calibration/evaluation sequences, each of length steps")
    return batches, data_hash


@contextmanager
def deterministic(seed):
    # CUBLAS must see this before its first operation in a fresh process.
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        ":4096:8",
        ":16:8",
    ):
        raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def metrics(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("tensor shape or dtype changed")
    a = actual.flatten()
    r = reference.flatten()
    ss = rr = peak = 0.0
    for start in range(0, a.numel(), 1 << 20):
        x = a[start : start + (1 << 20)].double()
        y = r[start : start + (1 << 20)].double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("nonfinite output or gradient")
        d = x - y
        ss += float(d.square().sum())
        rr += float(y.square().sum())
        peak = max(peak, float(d.abs().max()))
    return {
        "relative_l2": math.sqrt(ss) / max(math.sqrt(rr), 1e-30),
        "max_abs_over_rms": peak / max(math.sqrt(rr / max(1, a.numel())), 1e-30),
    }


def trajectory(actual, reference):
    if len(actual) != len(reference) or not actual:
        raise ValueError("trajectory length mismatch or empty trajectory")
    differences = [a - r for a, r in zip(actual, reference)]
    if not all(math.isfinite(v) for v in differences):
        raise ValueError("nonfinite trajectory")
    return {
        "rms": math.sqrt(sum(v * v for v in differences) / len(differences)),
        "max_abs": max(abs(v) for v in differences),
    }


def run(
    workload, kernels, batches, directory, reference=None, require_equal=False, save_reference=True
):
    """Disk-backed per-step reference; only scalar discrepancies survive."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    losses = []
    with deterministic(workload.spec.seed):
        model = workload.build(kernels)
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER)
        for step, source in enumerate(batches):
            batch = {k: v.to(workload.spec.device).detach().clone() for k, v in source.items()}
            input_key = getattr(workload, "input_gradient_key", "pixel_values")
            batch[input_key].requires_grad_(True)
            if hasattr(workload, "forward"):
                loss, logits = workload.forward(model, batch)
            else:
                result = forward(model, batch, workload.spec.model)
                loss, logits = result.loss, result.logits
            loss.backward()
            output_key = "logits" if _gate_name(workload) == "vision_model" else "output"
            values = {
                "loss": loss,
                output_key: logits,
                "input_gradient": batch[input_key].grad,
            }
            values.update(
                {"gradient:" + n: p.grad for n, p in model.named_parameters() if p.requires_grad}
            )
            for name, value in values.items():
                if value is None or not torch.isfinite(value).all():
                    raise ValueError(f"missing/nonfinite {name}")
                value = value.detach().cpu()
                filename = f"{step}-{hashlib.sha256(name.encode()).hexdigest()}.pt"
                if reference:
                    other = torch.load(Path(reference) / filename, weights_only=True)
                    if require_equal and not torch.equal(value, other):
                        raise ValueError(f"control did not repeat: {name}")
                    row = metrics(value, other)
                    prev = all_metrics.setdefault(name, {k: 0.0 for k in row})
                    for key in row:
                        prev[key] = max(prev[key], row[key])
                if save_reference:
                    torch.save(value, directory / filename)
            losses.append(float(loss.detach()))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del values, batch, loss, logits
            if "result" in locals():
                del result
        del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"tensors": all_metrics, "losses": losses}


def compile_kernels(workload, sites, backend="inductor"):
    from evograd.evaluation.tier3.patch import KernelSet, patch

    kernels = KernelSet(registry=workload.site_registry)
    for site in sites:
        kernels = patch(
            kernels,
            site,
            torch.compile(
                workload.site_registry.require(site).default,
                backend=backend,
                dynamic=False,
                fullgraph=True,
            ),
        )
    return kernels


def calibrate(workload, sites, steps=5, backend="inductor"):
    if steps < 2:
        raise ValueError("training calibration requires at least two updates")
    if not sites:
        raise ValueError("select at least one replacement site")
    all_batches, data_hash = verification_batches(workload, steps)
    from evograd.evaluation.tier3.patch import KernelSet

    eager = KernelSet(registry=workload.site_registry)
    compiled = compile_kernels(workload, sites, backend)
    batches = all_batches[:steps]
    with tempfile.TemporaryDirectory(prefix="evograd-calibration-") as tmp:
        root = Path(tmp)
        native = run(workload, eager, batches, root / "native")
        run(workload, eager, batches, root / "native_repeat", root / "native", True, False)
        control = run(workload, compiled, batches, root / "compiled", root / "native")
        run(workload, compiled, batches, root / "compiled_repeat", root / "compiled", True, False)
    row = {
        "schema_version": _schema(workload),
        "fingerprint": fingerprint(workload),
        "data_hash": data_hash,
        "data_mode": _data_mode(workload),
        "sites": sorted(sites),
        "steps": steps,
        "compile_backend": backend,
        "optimizer": OPTIMIZER,
        "calibration_indices": list(range(steps)),
        "evaluation_indices": list(range(steps, 2 * steps)),
        "repeatability": "bitwise native/native and compiled/compiled",
        "floor": FLOOR,
        "margin": MARGIN,
        "tensor_budgets": {
            n: {k: MARGIN * max(v, FLOOR) for k, v in ms.items()}
            for n, ms in control["tensors"].items()
        },
        "trajectory_budgets": {
            k: MARGIN * max(v, FLOOR)
            for k, v in trajectory(control["losses"], native["losses"]).items()
        },
    }
    row["calibration_hash"] = digest(row)
    return row


def evaluate(workload, kernels):
    try:
        if not workload.calibration:
            raise ValueError("missing frozen calibration for this workload and input mode")
        artifact = json.loads(Path(workload.calibration).read_text())
        expected_hash = artifact.pop("calibration_hash")
        if digest(artifact) != expected_hash:
            raise ValueError("calibration hash mismatch")
        if artifact["schema_version"] != _schema(workload) or artifact["fingerprint"] != fingerprint(workload):
            raise ValueError("calibration belongs to another execution environment/workload")
        if artifact["sites"] != sorted(kernels.patched):
            raise ValueError("calibration must cover the exact replacement sites")
        if (
            digest(artifact["optimizer"]) != digest(OPTIMIZER)
            or artifact["floor"] != FLOOR
            or artifact["margin"] != MARGIN
        ):
            raise ValueError("calibration policy or optimizer differs from current gate")
        mode = _data_mode(workload)
        if artifact.get("data_mode") != mode:
            raise ValueError("calibration input mode differs from workload")
        all_batches, data_hash = verification_batches(workload, artifact["steps"])
        if data_hash != artifact["data_hash"]:
            raise ValueError("calibration data hash mismatch")
        if artifact["compile_backend"] != "inductor" and workload.spec.device.startswith("cuda"):
            raise ValueError("CUDA evaluation requires inductor calibration")
        batches = [all_batches[i] for i in artifact["evaluation_indices"]]
        from evograd.evaluation.tier3.patch import KernelSet

        with tempfile.TemporaryDirectory(prefix="evograd-holdout-") as tmp:
            root = Path(tmp)
            native = run(
                workload, KernelSet(registry=workload.site_registry), batches, root / "native"
            )
            candidate = run(
                workload,
                kernels,
                batches,
                root / "candidate",
                root / "native",
                save_reference=False,
            )
        if set(candidate["tensors"]) != set(artifact["tensor_budgets"]):
            raise ValueError("output/gradient coverage changed")
        violations = [
            f"{n}:{k}"
            for n, ms in candidate["tensors"].items()
            for k, v in ms.items()
            if v > artifact["tensor_budgets"][n][k]
        ]
        training = trajectory(candidate["losses"], native["losses"])
        violations.extend(
            "trajectory:" + k for k, v in training.items() if v > artifact["trajectory_budgets"][k]
        )
        ok = not violations
        return {
            "gate": _gate_name(workload),
            "ok": ok,
            "execution_ok": True,
            "numerical_ok": ok,
            "evaluation_complete": True,
            "failed_at": None if ok else "numerical",
            "calibration_hash": expected_hash,
            "violations": violations,
            "trajectory": training,
            "tensor_metrics": candidate["tensors"],
            "reason": "passed" if ok else "frozen budgets exceeded",
            "findings": [],
        }
    except Exception as exc:
        return {
            "gate": _gate_name(workload),
            "ok": False,
            "execution_ok": False,
            "numerical_ok": None,
            "evaluation_complete": False,
            "failed_at": "model_gate",
            "findings": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }


def main():
    import argparse
    from .workload import VisionWorkload

    parser = argparse.ArgumentParser(
        description="Freeze model gate budgets from native and same-site compiled controls"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        help="optional prepared batches; omitted for seeded generated inputs")
    parser.add_argument("--real-input", action="store_true",
                        help="use pinned checkpoint and public dataset batches")
    parser.add_argument("--sites", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--compile-backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["weights"] = "random"
    config["data_path"] = str(args.data) if args.data else None
    if args.real_input:
        from evograd.benchmark.topdown.common.vision.spec import VisionSpec
        from evograd.benchmark.topdown.common.vision.data import ensure_pinned_data
        config["weights"] = "pretrained"
        spec = VisionSpec(**{k: v for k, v in config.items() if k in VisionSpec.__dataclass_fields__})
        config["data_path"] = ensure_pinned_data(spec, config["data_path"], batch_count=2 * args.steps)
    workload = VisionWorkload.from_config(config)
    row = calibrate(workload, args.sites.split(","), args.steps, args.compile_backend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
