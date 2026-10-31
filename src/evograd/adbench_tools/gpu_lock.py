"""Optional serialization of search evaluations on one GPU.

Set ADBENCH_GPU_LOCK to the same file for every search sharing a physical GPU.
Acquire/release events go to stderr and <lock>.log. Waiting does not consume
the evaluation timeout, but still counts against the search/allocation limit.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

LOCK_ENV = "ADBENCH_GPU_LOCK"


def _log(path: str, line: str) -> None:
    data = f"[gpu_lock] {line}\n"
    sys.stderr.write(data)
    sys.stderr.flush()
    fd = os.open(path + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data.encode())
    finally:
        os.close(fd)


@contextlib.contextmanager
def gpu_lock(label: str, *, cancelled=lambda: False):
    """Hold the configured lock; never nest this in a parent and its child."""
    path = os.environ.get(LOCK_ENV)
    if not path:
        yield
        return
    with open(path, "a") as handle:
        asked = time.monotonic()
        while True:
            if cancelled():
                raise InterruptedError("GPU lock wait interrupted")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.05)
        held = time.monotonic()
        try:
            _log(path, f"acquire t={time.time():.3f} pid={os.getpid()} "
                       f"waited={held-asked:.3f} {label}")
            yield
        finally:
            try:
                _log(path, f"release t={time.time():.3f} pid={os.getpid()} "
                           f"held={time.monotonic()-held:.3f} {label}")
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run_evaluation(command, *, env, timeout, label):
    """Run a captured evaluator, serializing its entire GPU process tree.

    A small monitor owns the lock and the evaluation timeout. It receives the
    batch's termination signals and kills the evaluator group before unlocking.
    Child evaluators have the lock variable removed, preventing nested locks.
    Without a configured lock the original subprocess behavior is preserved.
    """
    if not env.get(LOCK_ENV):
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="adbench_gpu_") as scratch:
        metadata = Path(scratch) / "status.json"
        monitor = [sys.executable, "-m", __name__, "_worker", label,
                   str(timeout or 0), str(metadata), *map(str, command)]
        result = subprocess.run(monitor, env=env, capture_output=True, text=True)
        if metadata.is_file():
            state = json.loads(metadata.read_text())
            if state["timed_out"]:
                raise subprocess.TimeoutExpired(command, timeout, result.stdout, result.stderr)
            result.returncode = state["returncode"]
        result.args = command
        return result


def _worker(label, timeout, metadata, command):
    stopped = [0]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.__setitem__(0, signum))
    try:
        with gpu_lock(label, cancelled=lambda: bool(stopped[0])):
            child_env = dict(os.environ)
            child_env.pop(LOCK_ENV, None)
            proc = subprocess.Popen(command, env=child_env, start_new_session=True)
            deadline = time.monotonic() + timeout if timeout else float("inf")
            timed_out = False
            try:
                while proc.poll() is None:
                    timed_out = time.monotonic() >= deadline
                    if stopped[0] or timed_out:
                        break
                    time.sleep(0.05)
            finally:
                # Also remove descendants left behind by a parent that exited.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            Path(metadata).write_text(json.dumps({"timed_out": timed_out,
                                                  "returncode": 128 + stopped[0] if stopped[0] else proc.returncode}))
        return 128 + stopped[0] if stopped[0] else (124 if timed_out else proc.returncode)
    except InterruptedError:
        return 128 + (stopped[0] or signal.SIGTERM)


if __name__ == "__main__":
    if len(sys.argv) < 6 or sys.argv[1] != "_worker":
        raise SystemExit("Internal evaluation monitor; use run_evaluation().")
    raise SystemExit(_worker(sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5:]))
