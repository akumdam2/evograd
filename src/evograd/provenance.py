"""Where a result was produced: the evograd checkout's commit and the host."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any


def revision() -> dict[str, Any]:
    """The checkout's commit and whether tracked files differ from it."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                     "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "", None
    return {"commit": commit or None, "dirty": dirty}


def provenance() -> dict[str, Any]:
    """:func:`revision` and the host it ran on."""
    return {**revision(), "host": socket.gethostname()}
