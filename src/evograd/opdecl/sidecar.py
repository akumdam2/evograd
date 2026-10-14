"""Libraries a baseline imports from outside the environment.

Flash Linear Attention is installed *beside* the evograd environment, not in
it (``pip install --no-deps --target ~/.cache/evograd-fla/0.5.2 ...``). The
reason is Transformers' import-time kernel dispatch (Qwen3.5 T-dispatch):
``modeling_qwen3_5`` binds ``chunk_gated_delta_rule`` and friends to FLA the
moment it is imported, if ``fla`` is importable then. Anywhere FLA can be
imported, a native reference can silently stop being the torch block.

So a baseline that needs such a library names it here, and two rules follow:

* only the process measuring that baseline may import it. The harness that
  spawns the child sets :data:`ENV` to the sidecar's name (:func:`sidecar_env`);
  no other process -- the parent, the native reference, any other provider --
  ever has the directory on ``sys.path``;
* inside that child the directory joins ``sys.path`` only when the baseline is
  first built (:func:`activate`), and only after the modules that dispatch at
  import time (``Sidecar.dispatches_first``) have been imported -- so they have
  already chosen torch, as they do everywhere else.

Prepending the directory to the child's ``PYTHONPATH`` would make it visible
from the interpreter's first import, before Transformers is imported, and the
child's own native block (built to judge the baseline) would run FLA.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass

#: Comma-separated sidecar names this process may activate.
ENV = "EVOGRAD_SIDECARS"


@dataclass(frozen=True)
class Sidecar:
    name: str
    #: Environment variable overriding the default directory.
    path_env: str
    default_path: str
    #: Modules that pick kernels when imported and must do so *without* this
    #: library visible: imported (if installed) before the path is added.
    dispatches_first: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return os.path.realpath(os.path.expanduser(os.environ.get(self.path_env)
                                                   or self.default_path))


SIDECARS: dict[str, Sidecar] = {
    "fla": Sidecar(
        name="fla",
        path_env="EVOGRAD_FLA_PATH",
        default_path="~/.cache/evograd-fla/0.5.2",
        # Every modeling module whose GDN functions dispatch at import: each is
        # its own module object with its own binding.
        dispatches_first=("transformers.models.qwen3_5.modeling_qwen3_5",),
    ),
}


class SidecarUnavailable(ImportError):
    """The library is not available to this process -- by design or by absence."""


def allowed() -> frozenset[str]:
    return frozenset(n.strip() for n in os.environ.get(ENV, "").split(",") if n.strip())


def sidecar_env(names, base: dict[str, str] | None = None) -> dict[str, str] | None:
    """The environment for a child that may activate ``names``; ``None`` when
    there are none (the child then inherits the parent's environment)."""
    names = sorted(set(names or ()))
    if not names:
        return None
    unknown = [n for n in names if n not in SIDECARS]
    if unknown:
        raise KeyError(f"unknown sidecar(s) {unknown}; known: {sorted(SIDECARS)}")
    env = dict(os.environ if base is None else base)
    env[ENV] = ",".join(names)
    return env


def on_path(name: str) -> bool:
    path = SIDECARS[name].path
    return any(os.path.realpath(p or ".") == path for p in sys.path)


def activate(name: str) -> str:
    """Make sidecar ``name`` importable in this process, if this process may.

    Raises :class:`SidecarUnavailable` in any process the harness did not start
    for it, and when the directory is missing.
    """
    sidecar = SIDECARS[name]
    if name not in allowed():
        raise SidecarUnavailable(
            f"{name} is a sidecar library: it is importable only in a child process "
            f"started for its baseline ({ENV}={name}); run the provider isolated")
    if not os.path.isdir(sidecar.path):
        raise SidecarUnavailable(f"{name}: {sidecar.path} does not exist "
                                 f"(set {sidecar.path_env} or install it there)")
    if not on_path(name):
        for module in sidecar.dispatches_first:
            try:
                importlib.import_module(module)
            except ImportError:
                pass  # not installed: nothing to dispatch
        sys.path.insert(0, sidecar.path)
    return sidecar.path


__all__ = ["ENV", "SIDECARS", "Sidecar", "SidecarUnavailable", "activate", "allowed",
           "on_path", "sidecar_env"]
