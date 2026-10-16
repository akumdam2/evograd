"""Stable implementation identities, so a frozen policy knows its reference.

A Gate 2 policy is only valid for the reference it was calibrated against. A
name or an import path is not that reference: ``native(x) = 2*x`` and
``native(x) = 3*x`` share both. :func:`fingerprint` identifies what actually
computes:

* a Python function by its bytecode, constants (nested code included, the
  docstring excluded), defaults, closure values, and the globals its code
  reaches -- directly or through module attribute chains (``F.helper``);
* a class defined outside installed libraries by its own methods and
  literal attributes, and its bases;
* ``functools.partial`` by its function, arguments and keywords;
* installed-library and standard-library objects by qualified name plus the
  package version (their code is the environment's, which the policy binds
  separately);
* literals, small containers of identifiable values, tensors (content hash),
  dtypes, devices and modules (by name).

Anything else at a position that participates in the computation -- the
object itself, a default, a closure value -- is refused with
:class:`UnidentifiableReference`, unless it declares an explicit, checkable
identity in ``__verification_identity__``. A *global* the code reaches that
cannot be identified is recorded by name and type under ``opaque``: its
renaming or retyping invalidates the policy, a change of its value does not,
and the identity says so.

Nothing here uses ``id()``, addresses or default ``repr()`` as content; ``id``
only guards recursion.

**Digest version 2** (:data:`FINGERPRINT_VERSION`). Version 1 hashed each code
constant as ``repr()``. A ``frozenset`` constant -- what CPython folds
``x in {"a", "b"}`` into -- iterates in an order set by ``PYTHONHASHSEED``, so
the same function had a different digest in every fresh process and a policy
frozen in one process could not be verified in the next. Version 2 encodes
every constant canonically: a type tag plus a length-framed payload, tuples in
order, frozensets as the sorted multiset of their elements' encodings, nested
code objects by their own digest, floats by ``float.hex``. A constant of any
other type is refused (fail closed). The version is part of every digest, so a
version-1 identity never equals a version-2 one: policies frozen under version 1
must be recalibrated, never reinterpreted.
"""

from __future__ import annotations

import dis
import functools
import hashlib
import json
import os
import sys
import sysconfig
import types
from typing import Any

import torch

#: Explicit identity hook for objects that cannot be analysed.
DECLARED = "__verification_identity__"
MAX_ITEMS = 1024

#: Bumped whenever the same object can produce a different digest. Hashed into
#: every digest and reported beside it; see the module docstring.
FINGERPRINT_VERSION = 2

_LIBRARY_DIRS = tuple(
    os.path.realpath(p) + os.sep
    for p in {sysconfig.get_paths().get(k) for k in ("purelib", "platlib", "stdlib", "platstdlib")}
    if p)


class UnidentifiableReference(ValueError):
    """An object that participates in the reference has no stable identity."""


def _is_library(module_name: str | None) -> bool:
    if not module_name:
        return True
    top = module_name.split(".")[0]
    if top in sys.stdlib_module_names or top in ("builtins", "torch"):
        return True
    module = sys.modules.get(module_name) or sys.modules.get(top)
    if module is None:
        return False                    # not importable (exec'd, synthetic): analyse it
    path = getattr(module, "__file__", None)
    if path is None:
        return True                     # builtin, frozen or extension without a file
    path = os.path.realpath(path)
    return path.startswith(_LIBRARY_DIRS) or f"{os.sep}site-packages{os.sep}" in path


def _version(module_name: str) -> str | None:
    top = module_name.split(".")[0]
    if top in sys.stdlib_module_names or top == "builtins":
        return "python-" + sys.version.split()[0]
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(top)
    except PackageNotFoundError:
        return getattr(sys.modules.get(top), "__version__", None)


def _qualname(obj: Any) -> str:
    return f"{getattr(obj, '__module__', None)}.{getattr(obj, '__qualname__', getattr(obj, '__name__', type(obj).__name__))}"


class _Fingerprinter:
    def __init__(self) -> None:
        self.visiting: set[int] = set()
        self.opaque: set[str] = set()
        #: Finished functions/classes: id -> (object, identity). The object is
        #: held so its id cannot be reused while this walk lives.
        self.done: dict[int, tuple[Any, Any]] = {}

    # values at positions that compute: strict
    def value(self, v: Any, where: str) -> Any:
        if hasattr(v, DECLARED) and not isinstance(v, type):
            return {"declared": json.loads(json.dumps(getattr(v, DECLARED), sort_keys=True,
                                                      default=str))}
        if v is None or isinstance(v, (bool, int, float, str, complex)):
            return [type(v).__name__, repr(v)]
        if isinstance(v, bytes):
            return ["bytes", hashlib.sha256(v).hexdigest()]
        if isinstance(v, (torch.dtype, torch.device, torch.Size)):
            return [type(v).__name__, str(v)]
        if torch.is_tensor(v):
            from evograd.verification.common.tensors import tensor_digest

            return ["tensor", tensor_digest(v)]
        if isinstance(v, types.ModuleType):
            return ["module", v.__name__]
        if isinstance(v, (tuple, list, set, frozenset)):
            if len(v) > MAX_ITEMS:
                raise UnidentifiableReference(f"{where}: container of {len(v)} items")
            items = [self.value(x, f"{where}[]") for x in v]
            if isinstance(v, (set, frozenset)):
                items = sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
            return [type(v).__name__, items]
        if isinstance(v, dict):
            if len(v) > MAX_ITEMS:
                raise UnidentifiableReference(f"{where}: mapping of {len(v)} items")
            keys = list(v)
            if not all(isinstance(k, (str, int, bool, float, type(None))) for k in keys):
                raise UnidentifiableReference(f"{where}: mapping with non-literal keys")
            return ["dict", sorted([[repr(k), self.value(v[k], f"{where}.{k}")] for k in keys])]
        if isinstance(v, functools.partial):
            return {"partial": self.value(v.func, f"{where}.func"),
                    "args": self.value(v.args, f"{where}.args"),
                    "keywords": self.value(dict(v.keywords), f"{where}.keywords")}
        if isinstance(v, (staticmethod, classmethod)):
            return {type(v).__name__: self.value(v.__func__, where)}
        if isinstance(v, property):
            return {"property": [self.value(f, where) for f in (v.fget, v.fset, v.fdel)]}
        if isinstance(v, types.FunctionType):
            return self.function(v)
        if isinstance(v, types.MethodType):
            return {"method": self.value(v.__func__, where), "self": self.value(v.__self__, where)}
        if isinstance(v, type):
            return self.cls(v)
        module = getattr(v, "__module__", None)
        if callable(v) and module is not None and _is_library(module) and \
                hasattr(v, "__qualname__") or isinstance(v, (types.BuiltinFunctionType,
                                                            types.BuiltinMethodType)):
            return {"library": _qualname(v), "version": _version(module or "builtins")}
        raise UnidentifiableReference(
            f"{where}: a {type(v).__module__}.{type(v).__qualname__} has no stable identity; "
            f"give it `{DECLARED}` or pass a plain function")

    # globals the code reaches: lenient (recorded as opaque when unidentifiable)
    def reached(self, v: Any, where: str) -> Any:
        try:
            return self.value(v, where)
        except UnidentifiableReference:
            label = f"{where}:{type(v).__module__}.{type(v).__qualname__}"
            self.opaque.add(label)
            return ["opaque", label]

    def function(self, fn: types.FunctionType) -> Any:
        if _is_library(fn.__module__):
            return {"library": _qualname(fn), "version": _version(fn.__module__ or "builtins")}
        if id(fn) in self.done:
            return self.done[id(fn)][1]
        if id(fn) in self.visiting:
            return ["recursion", _qualname(fn)]
        self.visiting.add(id(fn))
        try:
            code = fn.__code__
            closure = {}
            for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    closure[name] = ["empty"]
                    continue
                closure[name] = self.value(contents, f"{_qualname(fn)}<closure {name}>")
            scope = dict(fn.__globals__)
            scope.update({n: c for n, c in zip(code.co_freevars, (
                cell.cell_contents if _filled(cell) else None for cell in fn.__closure__ or ()))})
            reached = {}
            for chain in sorted(_chains(code)):
                root = chain[0]
                if root not in scope:
                    continue                   # a builtin, or a name bound elsewhere
                value, path = scope[root], [root]
                for attr in chain[1:]:
                    if not isinstance(value, types.ModuleType) or not hasattr(value, attr):
                        break
                    value, path = getattr(value, attr), path + [attr]
                key = ".".join(path)
                if key not in reached:
                    reached[key] = self.reached(value, f"{fn.__module__}:{key}")
            identity = {"function": _qualname(fn), "code": _code_digest(code, fn.__doc__),
                        "defaults": self.value(fn.__defaults__, f"{_qualname(fn)}<defaults>"),
                        "kwdefaults": self.value(fn.__kwdefaults__, f"{_qualname(fn)}<kwdefaults>"),
                        "closure": closure, "globals": reached}
            self.done[id(fn)] = (fn, identity)
            return identity
        finally:
            self.visiting.discard(id(fn))

    def cls(self, c: type) -> Any:
        if _is_library(c.__module__):
            return {"library": _qualname(c), "version": _version(c.__module__)}
        if id(c) in self.done:
            return self.done[id(c)][1]
        if id(c) in self.visiting:
            return ["recursion", _qualname(c)]
        self.visiting.add(id(c))
        try:
            members = {}
            for name, attr in sorted(vars(c).items()):
                computing = isinstance(attr, (types.FunctionType, staticmethod, classmethod,
                                              property, functools.partial))
                if name.startswith("__") and name.endswith("__") and not computing:
                    continue
                members[name] = self.reached(attr, f"{_qualname(c)}.{name}")
            identity = {"class": _qualname(c),
                        "bases": [self.value(b, _qualname(c)) for b in c.__bases__],
                        "members": members}
            self.done[id(c)] = (c, identity)
            return identity
        finally:
            self.visiting.discard(id(c))


def _filled(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def _chains(code: types.CodeType) -> set[tuple[str, ...]]:
    """Name chains the code loads: ``LOAD_GLOBAL F; LOAD_ATTR helper`` -> ``(F, helper)``."""
    out: set[tuple[str, ...]] = set()
    chain: list[str] = []
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_DEREF", "LOAD_NAME"):
            if chain:
                out.add(tuple(chain))
            chain = [ins.argval]
        elif ins.opname in ("LOAD_ATTR", "LOAD_METHOD") and chain:
            chain.append(ins.argval)
        else:
            if chain:
                out.add(tuple(chain))
            chain = []
    if chain:
        out.add(tuple(chain))
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out |= _chains(const)
    return out


def _frame(payload: bytes) -> bytes:
    """``<length>:<payload>``, so concatenated encodings cannot run together."""
    return str(len(payload)).encode("ascii") + b":" + payload


def _constant(value: Any) -> bytes:
    """Canonical, type-preserving bytes for one code constant.

    Exact types only: ``True`` is not ``1``, ``1.0`` is not ``1``, ``-0.0`` is
    not ``0.0``, ``("a",)`` is not ``frozenset({"a"})``. Unordered constants are
    encoded as the sorted encodings of their elements, which depends on the
    values alone and never on the process's string hash seed.
    """
    kind = type(value)
    if value is None:
        return b"N"
    if value is Ellipsis:
        return b"E"
    if kind is bool:
        return b"b1" if value else b"b0"
    if kind is int:
        return b"i" + _frame(str(value).encode("ascii"))
    if kind is float:
        return b"f" + _frame(value.hex().encode("ascii"))
    if kind is complex:
        return b"c" + _frame(f"{value.real.hex()},{value.imag.hex()}".encode("ascii"))
    if kind is str:
        return b"s" + _frame(value.encode("utf-8", "surrogatepass"))
    if kind is bytes:
        return b"y" + _frame(value)
    if kind is tuple:
        return b"t" + _frame(b"".join(_frame(_constant(item)) for item in value))
    if kind is frozenset:
        return b"F" + _frame(b"".join(sorted(_frame(_constant(item)) for item in value)))
    if kind is slice:
        return b"S" + _frame(_constant((value.start, value.stop, value.step)))
    if kind is types.CodeType:
        return b"C" + _frame(_code_digest(value, None).encode("ascii"))
    raise UnidentifiableReference(
        f"code constant of type {kind.__module__}.{kind.__qualname__} has no canonical encoding")


def _code_digest(code: types.CodeType, doc: str | None) -> str:
    """Bytecode, canonical constants (docstring excluded), names and signature."""
    h = hashlib.sha256(b"evograd-code-digest/2\0")
    h.update(_frame(code.co_code))
    for i, const in enumerate(code.co_consts):
        if i == 0 and doc is not None and type(const) is str and const == doc:
            h.update(b"<doc>")
        else:
            h.update(_frame(_constant(const)))
    for field in (code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars):
        h.update(_frame(b"".join(_frame(name.encode("utf-8")) for name in field)))
    h.update(f"{code.co_argcount},{code.co_posonlyargcount},{code.co_kwonlyargcount},"
             f"{code.co_flags}".encode("ascii"))
    return h.hexdigest()


def fingerprint(obj: Any, *, label: str = "reference") -> dict[str, Any]:
    """``{"qualname", "digest", "opaque", "fingerprint_version"}`` for one reference object.

    Raises :class:`UnidentifiableReference` when ``obj`` -- or a default or
    closure value it computes with -- cannot be identified.
    """
    worker = _Fingerprinter()
    detail = worker.value(obj, label)
    payload = json.dumps({"fingerprint_version": FINGERPRINT_VERSION, "detail": detail},
                         sort_keys=True)
    return {"qualname": _qualname(obj) if not isinstance(obj, dict) else label,
            "digest": hashlib.sha256(payload.encode()).hexdigest(),
            "opaque": sorted(worker.opaque),
            "fingerprint_version": FINGERPRINT_VERSION}


def fingerprint_parts(parts: dict[str, Any]) -> dict[str, Any]:
    """:func:`fingerprint` per named part, e.g. ``{"factory": f, "native_sites": {...}}``."""
    return {name: fingerprint(obj, label=name) for name, obj in sorted(parts.items())}
