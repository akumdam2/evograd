"""Opt-in reuse of Gate 3's native training controls. Default execution is unchanged.

``--share-gate3-native``: one native training run serves every patch set whose
training identity matches (``integrations.gates``). Gate 2 keeps nothing to reuse:
both of its scopes measure their trusted controls in the run that judges.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

FLAGS = ("share_gate3_native",)
VERSION = "native-control-reuse/2"
_CURRENT = ContextVar("verification_reuse", default=None)


def add_arguments(parser):
    parser.add_argument("--share-gate3-native", action="store_true",
                        help="reuse native training controls across compatible patch sets")


def options(args):
    return {name: bool(getattr(args, name, False)) for name in FLAGS}


def settings():
    current = _CURRENT.get()
    return {"version": VERSION, **{name: bool(current and current.get(name)) for name in FLAGS}}


def enabled(name):
    return settings()[name]


@contextmanager
def calibration_reuse(config=None, *, log=print):
    """The reuse options for the block (:data:`FLAGS`, all off by default)."""
    config = dict(config or {})
    unknown = set(config) - set(FLAGS)
    if unknown:
        raise ValueError(f"unknown reuse options: {sorted(unknown)}")
    if any(type(value) is not bool for value in config.values()):
        raise ValueError("reuse options must be booleans")
    token = _CURRENT.set(config if any(config.values()) else None)
    try:
        if any(config.values()):
            log("[control-reuse]", settings())
        yield
    finally:
        _CURRENT.reset(token)


def check_report_mode(path):
    """Do not overwrite a report written under other reuse options."""
    import json

    path = Path(path)
    if path.exists():
        recorded = json.loads(path.read_text()).get("settings", {}).get("calibration_reuse", {})
        live = settings()
        if any(bool(recorded.get(name, False)) != live[name] for name in FLAGS):
            raise ValueError("reuse options changed; choose a new output directory")
