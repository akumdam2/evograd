"""Diagnose Triton's driver bootstrap. Run on the GPU node that failed:

    PYTHONPATH=src python scripts/triton_env_check.py

Pipeline D's capture never reaches its own code when this is broken: the first
Triton kernel Inductor emits forces `driver.active`, which shells out to `cc` to
build a small CUDA shim (`cuda_utils.c`) against libcuda and the CPython headers.
Triton runs that compile with `stdout=DEVNULL` and re-raises a bare
CalledProcessError, so the compiler's own message is the one thing the traceback
does not carry. This script reruns the exact same compile with stderr captured,
after reporting each input that command depends on.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def head(title: str) -> None:
    print(f"\n=== {title}")


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<{type(exc).__name__}: {exc}>"
    return (out.stdout + out.stderr).strip()


def report_interpreter() -> None:
    head("interpreter")
    print(f"python      {sys.executable}")
    print(f"version     {sys.version.split()[0]}  ({platform.machine()}, {platform.system()})")
    include = sysconfig.get_paths()["include"]
    header = Path(include) / "Python.h"
    # This is the -I<...>/include on Triton's cc line. A venv built from a system
    # python inherits the system include dir; if python3-devel was never
    # installed the directory is absent and every Triton compile fails with
    # "Python.h: No such file or directory".
    print(f"include     {include}")
    print(f"Python.h    {'present' if header.exists() else 'MISSING  <- cc cannot compile without it'}")


def report_compiler() -> None:
    head("compiler")
    env_cc = os.environ.get("CC")
    print(f"$CC         {env_cc or '<unset>'}")
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if not path:
            print(f"{name:<12}<not on PATH>")
            continue
        real = os.path.realpath(path)
        version = run([path, "--version"]).splitlines()
        suffix = f"  -> {real}" if real != path else ""
        print(f"{name:<12}{path}{suffix}")
        print(f"{'':<12}{version[0] if version else '<no --version output>'}")
    # HANDOFF.md records that the previous cluster needed gcc-toolset-13 on
    # LD_LIBRARY_PATH; a half-activated toolset is a compile failure of its own.
    for toolset in sorted(Path("/opt/rh").glob("gcc-toolset-*")) if Path("/opt/rh").is_dir() else []:
        print(f"toolset     {toolset}")
    print(f"LD_LIBRARY_PATH {os.environ.get('LD_LIBRARY_PATH', '<unset>')}")


def report_libcuda() -> None:
    head("libcuda")
    print(f"$TRITON_LIBCUDA_PATH {os.environ.get('TRITON_LIBCUDA_PATH', '<unset>')}")
    cache = run(["/sbin/ldconfig", "-p"])
    hits = [line.strip() for line in cache.splitlines() if "libcuda.so" in line]
    print("ldconfig -p | grep libcuda.so:")
    for line in hits or ["  <no match>"]:
        print(f"  {line}")
    for candidate in {line.split()[-1] for line in hits if "=>" in line}:
        path = Path(candidate)
        if not path.exists():
            print(f"  {candidate}: DANGLING symlink -> {os.readlink(path) if path.is_symlink() else '?'}")
        else:
            print(f"  {candidate}: {path.stat().st_size:,} bytes")
    try:
        from triton.backends.nvidia.driver import libcuda_dirs

        print(f"triton libcuda_dirs() -> {libcuda_dirs()}")
    except Exception as exc:  # noqa: BLE001 - the assertion text is the diagnosis
        print(f"triton libcuda_dirs() raised {type(exc).__name__}: {exc}")


def report_triton() -> None:
    head("triton / torch")
    try:
        import triton
    except ImportError as exc:
        print(f"triton import failed: {exc}")
        return
    print(f"triton      {triton.__version__}  ({Path(triton.__file__).parent})")
    backend = Path(triton.__file__).parent / "backends" / "nvidia"
    header = backend / "include" / "cuda.h"
    print(f"cuda.h      {'present' if header.exists() else f'MISSING at {header}'}")
    try:
        import torch

        print(f"torch       {torch.__version__}")
        print(f"cuda avail  {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"device      {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch check raised {type(exc).__name__}: {exc}")
    # The .so is built in a tempdir and dlopen'd from there; a noexec /tmp
    # fails later than this script's compile, but is worth seeing here.
    tmp = os.environ.get("TMPDIR", "/tmp")
    print(f"TMPDIR      {tmp}")
    mount = next((l for l in run(["findmnt", "-no", "TARGET,OPTIONS", "-T", tmp]).splitlines()), "")
    print(f"mount       {mount or '<findmnt unavailable>'}")


class _CapturingSubprocess:
    """Stand-in for `subprocess` inside triton.runtime.build.

    `_build` calls `subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)`.
    Only stdout is redirected, so stderr normally lands in the job log -- but not
    in the CalledProcessError, and not in the traceback Pipeline D writes to
    `capture_error.txt`. Swapping the module reference merges both streams into
    the exception path instead.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def check_call(self, cmd, *args, **kwargs):
        print("\ncommand:")
        print("  " + " ".join(cmd))
        kwargs.pop("stdout", None)
        kwargs.pop("stderr", None)
        done = self._real.run(cmd, *args, stdout=self._real.PIPE,
                              stderr=self._real.STDOUT, text=True, **kwargs)
        if done.stdout:
            print("compiler output:")
            for line in done.stdout.rstrip().splitlines():
                print(f"  {line}")
        if done.returncode != 0:
            raise self._real.CalledProcessError(done.returncode, cmd, output=done.stdout)
        return 0


def reproduce() -> int:
    head("reproducing the driver build with stderr captured")
    try:
        import triton.runtime.build as build
    except ImportError as exc:
        print(f"cannot import triton.runtime.build: {exc}")
        return 1
    original = build.subprocess
    build.subprocess = _CapturingSubprocess(original)
    try:
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        print(f"\ndriver bootstrap OK -> {target}")
    except Exception as exc:  # noqa: BLE001
        print(f"\ndriver bootstrap FAILED: {type(exc).__name__}: {exc}")
        return 1
    finally:
        build.subprocess = original

    head("end-to-end triton kernel")
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _copy(src, dst, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)

        src = torch.randn(1024, device="cuda")
        dst = torch.empty_like(src)
        _copy[(1,)](src, dst, src.numel(), BLOCK=1024)
        print("kernel compiled and ran;", "values match" if torch.equal(src, dst) else "VALUES DIFFER")
    except Exception as exc:  # noqa: BLE001
        print(f"kernel FAILED: {type(exc).__name__}: {exc}")
        return 1
    return 0


def main() -> int:
    report_interpreter()
    report_compiler()
    report_libcuda()
    report_triton()
    return reproduce()


if __name__ == "__main__":
    raise SystemExit(main())
