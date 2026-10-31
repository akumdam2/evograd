"""Opt-in calibration reuse. Default execution remains unchanged.

Gate 2 tensors live in private job-local files, not an unbounded GPU/CPU map.
Calibration reference/repeat and verification reference use different slots;
real execution IDs are retained when an execution is reused.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile

FLAGS = ("share_gate3_native", "share_gate2_calibration_native", "share_gate2_verification_reference")
VERSION = "native-control-reuse/1"
_CURRENT = ContextVar("verification_reuse", default=None)


def add_arguments(parser):
    parser.add_argument("--share-gate3-native", action="store_true",
                        help="reuse native training controls across compatible patch sets")
    parser.add_argument("--share-gate2-calibration-native", action="store_true",
                        help="reuse model Gate 2 native calibration executions across patch sets")
    parser.add_argument("--share-gate2-verification-reference", action="store_true",
                        help="reuse separate model Gate 2 verification references across candidates")


def options(args):
    return {name: bool(getattr(args, name, False)) for name in FLAGS}


def settings():
    current = _CURRENT.get()
    return {"version": VERSION, **{name: bool(current and current.options.get(name)) for name in FLAGS}}


def enabled(name):
    return settings()[name]


@dataclass
class _Session:
    options: dict
    directory: Path
    log: object


@contextmanager
def calibration_reuse(config=None, *, log=print):
    config = dict(config or {})
    unknown = set(config) - set(FLAGS)
    if unknown:
        raise ValueError(f"unknown calibration reuse options: {sorted(unknown)}")
    if any(type(value) is not bool for value in config.values()):
        raise ValueError("calibration reuse options must be booleans")
    if not any(config.values()):
        token = _CURRENT.set(None)
        try:
            yield
        finally:
            _CURRENT.reset(token)
        return
    import os
    with tempfile.TemporaryDirectory(prefix="evograd-native-controls-", dir=os.environ.get("SLURM_TMPDIR")) as directory:
        token = _CURRENT.set(_Session(config, Path(directory), log))
        try:
            log("[calibration-reuse]", settings(), flush=True) if log is print else log("[calibration-reuse]", settings())
            yield
        finally:
            _CURRENT.reset(token)


def native_execution(adapter, sample, *, slot, execute):
    """Reuse only identified model-scope native runs; never candidate/compile runs.

    The cache is job-local. Native/candidate content and weight checks in the
    calibrator/verifier still run on every comparison and reject mismatches.
    """
    from evograd.verification.common.execution import Deployment

    session = _CURRENT.get()
    flag = "share_gate2_verification_reference" if slot == "verification" else "share_gate2_calibration_native"
    if session is None or not session.options.get(flag) or adapter.scope != "model":
        return execute(adapter, Deployment.native(), sample)
    import torch
    from evograd.verification.common import contracts as C
    from evograd.verification.gate2.runner import results_digest
    from evograd.verification.integrations.objectives import to_device

    # Unidentifiable references cannot be shared. The normal calibrator reports
    # why they are unidentifiable; this optimization never weakens that check.
    try:
        reference = adapter.reference_identity()
    except ValueError:
        return execute(adapter, Deployment.native(), sample)
    workload = getattr(adapter, "workload", None)
    binding = {"version": VERSION, "slot": slot, "adapter": adapter.identity(),
               "reference": reference, "numerics": adapter.numerics(),
               "environment": C.environment_fingerprint(adapter.device),
               "sample": sample.to_dict(),
               "data": workload.data_identity() if workload is not None else None}
    key = C.stable_hash(binding)
    path = session.directory / f"{key}.pt"
    if path.exists():
        # Only our own private session files are deserialized.
        saved = torch.load(path, map_location=adapter.device, weights_only=False)
        result = saved["execution"]
        if saved["binding"] != binding or result.results_digest != results_digest(result.results):
            raise ValueError("native execution cache binding or content mismatch")
        result.reused_from = str(path)
        session.log("[calibration-reuse] hit", slot, sample.sample_id)
        return result
    result = execute(adapter, Deployment.native(), sample)
    if not result.error and not result.sample_identity.get("opaque"):
        stored = replace(result, results=to_device(result.results, "cpu"))
        torch.save({"binding": binding, "execution": stored}, path)
        session.log("[calibration-reuse] saved", slot, sample.sample_id)
    return result


def check_report_mode(path):
    """Do not overwrite a report from a different calibration experiment."""
    import json

    path = Path(path)
    if path.exists():
        recorded = json.loads(path.read_text()).get("settings", {}).get("calibration_reuse", {})
        live = settings()
        if any(bool(recorded.get(name, False)) != live[name] for name in FLAGS):
            raise ValueError("calibration reuse options changed; choose a new output directory")
