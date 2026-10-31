"""Job-local pristine CPU model snapshots for multi-agent formal evaluation.

Only checkpoint-backed models are cached. Every caller receives an independent
copy before patches, gradients or optimizer state exist. T3 children mmap the
same trusted, job-created snapshot, retaining process/GPU-memory isolation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import dataclasses
import fcntl
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path

ENV = "EVOGRAD_EVALUATION_MODEL_CACHE"
QWEN35_RNG = "qwen3.5-checkpoint-cpu-uniform/1"


def _replay_qwen35_rng(heads):
    """Consume the small A_log initialization draws overwritten by the checkpoint.

    Preserve the per-layer calls (one large draw need not have the same RNG
    effect), CPU device, and float32 dtype seen in Transformers' _init_weights.
    Never install the discarded values into the cached model.
    """
    import torch

    for size in heads:
        torch.empty(size, device="cpu", dtype=torch.float32).uniform_(0.01, 16)


def _qwen35_rng_recipe(model, before, after):
    """Accept only the observed Qwen3.5 loader transition, not arbitrary RNG use."""
    import torch

    heads = [int(module.num_v_heads) for module in model.modules()
             if type(module).__name__ == "Qwen3_5GatedDeltaNet"
             and type(module).__module__ == "transformers.models.qwen3_5.modeling_qwen3_5"]
    if not heads or any(size <= 0 for size in heads):
        raise RuntimeError("Qwen3.5 RNG replay cannot identify the Gated DeltaNet layers; "
                           "rerun with --no-weight-cache")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(before)
        _replay_qwen35_rng(heads)
        matches = torch.equal(after, torch.get_rng_state())
    if not matches:
        raise RuntimeError("Qwen3.5 checkpoint RNG changes do not match the supported A_log "
                           "initialization; rerun with --no-weight-cache")
    return heads


WORKLOAD_MODULES = {
    "qwen3_0_6b": "qwen3_0_6b",
    "qwen3_5_4b": "qwen3_5_4b",
    "llama_3_2_1b": "llama3_2_1b",
    "olmoe_1b_7b": "olmoe_1b_7b",
    "convnextv2_tiny": "common.vision",
    "qwen2_vl_2b": "common.vision",
}


class PristineModelCache:
    def __init__(self, directory, *, model_name=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.model_name = model_name
        self.rng_replays = {}
        self.loads = 0
        self.copies = 0
        self.factory = None

    def prime(self, spec):
        # Fail before evaluating candidates if snapshotting is unsupported.
        if self.model_name == "qwen3_5_4b":
            import torch

            # Preloading is preparation, not a measured model construction.
            # Do not advance the caller's stream merely by creating the snapshot.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                model = self.build(self.factory, dataclasses.replace(spec, device="cpu"))
        else:
            model = self.build(self.factory, dataclasses.replace(spec, device="cpu"))
        del model

    def build(self, factory, spec):
        import torch

        if not (getattr(spec, "pretrained", False) or getattr(spec, "weights", None) == "pretrained"):
            return factory(spec)
        cpu_spec = dataclasses.replace(spec, device="cpu")
        qwen35 = self.model_name == "qwen3_5_4b"
        rng_binding = {}
        if qwen35:
            from importlib.metadata import version

            rng_binding = {"rng_replay": QWEN35_RNG, "torch": torch.__version__,
                           "transformers": version("transformers"),
                           "default_dtype": str(torch.get_default_dtype())}
        key = hashlib.sha256(json.dumps({
            "spec": dataclasses.asdict(cpu_spec),
            "factory": f"{factory.__module__}.{factory.__qualname__}",
            "source": inspect.getsource(factory),
            **rng_binding,
        }, sort_keys=True, default=str).encode()).hexdigest()
        fresh_load = False
        if key not in self.models:
            path = self.directory / f"{key}.pt"
            with (self.directory / f"{key}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not path.exists():
                    before = torch.random.get_rng_state()
                    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
                    model = factory(cpu_spec)
                    # A reused snapshot must not alter the RNG semantics of construction.
                    after = torch.random.get_rng_state()
                    cpu_same = torch.equal(before, after)
                    cuda_same = cuda_before is None or all(
                        torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))
                    heads = []
                    if qwen35 and not cpu_same and cuda_same:
                        heads = _qwen35_rng_recipe(model, before, after)
                    elif not cpu_same or not cuda_same:
                        raise RuntimeError("Checkpoint builder changes RNG state; rerun with --no-weight-cache")
                    if qwen35 and cuda_before is None and torch.cuda.is_initialized():
                        raise RuntimeError("Qwen3.5 CPU checkpoint builder initialized CUDA; "
                                           "rerun with --no-weight-cache")
                    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
                        raise RuntimeError("Pristine model cache requires CPU-only parameters and buffers")
                    if any(p.grad is not None for p in model.parameters()):
                        raise RuntimeError("Pristine model unexpectedly contains gradients")
                    temporary = path.with_suffix(f".{os.getpid()}.tmp")
                    try:
                        payload = ({"schema": QWEN35_RNG, "model": model,
                                    "heads": heads, "binding": rng_binding} if qwen35 else model)
                        torch.save(payload, temporary)
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    self.loads += 1
                    self.models[key] = model
                    self.rng_replays[key] = heads
                    if heads:
                        print(f"[model-cache] Qwen3.5 RNG replay validated: {len(heads)} small CPU draws per build", flush=True)
                    fresh_load = True
                    print(f"[model-cache] Loaded checkpoint once: {getattr(spec, 'model_name', type(spec).__name__)}", flush=True)
                else:
                    # This file is generated only by this job, never supplied by a candidate.
                    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                    if qwen35:
                        if (not isinstance(payload, dict) or payload.get("schema") != QWEN35_RNG
                                or payload.get("binding") != rng_binding
                                or not isinstance(payload.get("heads"), list)
                                or any(type(size) is not int or size <= 0 for size in payload["heads"])):
                            raise RuntimeError("Invalid Qwen3.5 RNG replay snapshot")
                        self.models[key] = payload["model"]
                        self.rng_replays[key] = payload["heads"]
                    else:
                        self.models[key] = payload
                    print("[model-cache] Using job-local CPU snapshot", flush=True)
        # A real load already consumed these draws. Memory/disk cache hits
        # must reproduce them from the caller's CURRENT state, not reset to a
        # stored seed or consume the draws a second time on the first build.
        if qwen35 and not fresh_load:
            _replay_qwen35_rng(self.rng_replays[key])
        self.copies += 1
        return copy.deepcopy(self.models[key]).to(device=spec.device)


@contextmanager
def checkpoint_model_cache(model, directory=None):
    directory = directory or os.environ.get(ENV)
    target = WORKLOAD_MODULES.get(model)
    if not directory or not target:
        yield None
        return
    module = importlib.import_module(f"evograd.evaluation.tier3.workloads.{target}.workload")
    factory = module.build_model
    cache = PristineModelCache(directory, model_name=model)
    cache.factory = factory

    @functools.wraps(factory)
    def cached(spec):
        return cache.build(factory, spec)

    module.build_model = cached
    try:
        yield cache
    finally:
        module.build_model = factory
