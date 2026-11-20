"""Gate 2's trusted control: the native implementation of the replaced sites under
``torch.compile``, a correct implementation that rounds differently.
"""

from __future__ import annotations

import itertools
import types
from typing import Any, Mapping, Sequence

import torch

from evograd.verification.common.execution import CONTROL, Deployment


def compile_control(adapter, patch_set: Sequence[str], *, backend: str = "inductor",
                    fullgraph: bool = True, options: Mapping[str, Any] | None = None
                    ) -> tuple[Deployment, dict[str, int]]:
    """The native implementation of every replaced site under ``torch.compile``.

    Installed at exactly the candidate's replacement sites and run through the
    same autograd path; where the adapter's deployment takes only some of a
    site's configurations (``control_site``), at exactly those calls. The
    returned counter records how many graphs the backend compiled per site.
    """
    from torch._dynamo import lookup_backend

    real = lookup_backend(backend)
    graphs: dict[str, int] = {site: 0 for site in patch_set}
    sites = {}
    for site in patch_set:
        def counting(gm, example_inputs, _site=site):
            graphs[_site] += 1
            return real(gm, example_inputs)

        compiled = torch.compile(_private_entry(adapter.native_site(site)), backend=counting,
                                 fullgraph=fullgraph, **dict(options or {}))
        install = getattr(adapter, "control_site", None)
        sites[site] = install(site, compiled) if install is not None else compiled
    identity = {"kind": "torch_compile", "backend": backend, "fullgraph": fullgraph,
                "options": dict(options or {}), "torch": torch.__version__,
                "compiles": "the native implementation of each replaced site"}
    return Deployment(f"torch_compile[{backend}]", CONTROL, sites, identity), graphs


def _private_entry(fn):
    """``fn`` behind a function with its own code object.

    Dynamo keeps compiled entries on the code object it traces. Compiling the
    same native function again (another case in the same process, a new
    counting backend) would pile entries onto one code object until the
    recompile limit turns the next control into a hard failure. A private
    code object gives each control its own cache, freed with the control.
    It also gets its own name: dynamo keeps automatic-dynamic shape state per
    (file, line, name), so entries sharing a name would let one site's shapes
    turn the next site's dynamic (a SymBool then reaches SDPA's bool arguments).
    """
    def site_entry(*args, **kwargs):
        return fn(*args, **kwargs)

    name = f"site_entry_{next(_ENTRY_IDS)}"
    return types.FunctionType(site_entry.__code__.replace(co_name=name, co_qualname=name),
                              site_entry.__globals__, name, None, site_entry.__closure__)


_ENTRY_IDS = itertools.count()
