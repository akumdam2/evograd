"""Import helpers shared by declarations, extraction, and synthesis.

Forward references may be regular ``package.module:function`` specs or
``/path/to/file.py:function`` specs. The latter is the stable boundary used by
the high-level callable/file interface.
"""

from __future__ import annotations

import importlib
import importlib.util
import uuid
from pathlib import Path


def resolve_callable(spec: str):
    module_or_path, separator, function_name = spec.partition(":")
    if not separator:
        module_or_path, function_name = spec.rsplit(".", 1)

    path = Path(module_or_path)
    if path.suffix == ".py" or "/" in module_or_path:
        if not path.is_file():
            raise FileNotFoundError(f"forward file not found: {path}")
        module_spec = importlib.util.spec_from_file_location(
            f"evograd_forward_{uuid.uuid4().hex}", path
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not import forward file: {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_or_path)

    value = getattr(module, function_name)
    if not callable(value):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return value
