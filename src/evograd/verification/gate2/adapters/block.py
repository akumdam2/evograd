"""DEPRECATED. Gate 2 has no block scope (removed 2026-10-13).

The generic module adapter moved to :mod:`evograd.verification.gate2.adapters.module`
(``ModuleScope``, model scope only). This module re-exports it for historical
imports and keeps ``LegacyBlockScope`` only as a name that refuses to build:
a block is not a Gate 2 scope. No removed block-evaluation modules are required.
"""

from __future__ import annotations

import warnings

from evograd.verification.gate2.adapters.module import ModuleScope

warnings.warn("evograd.verification.gate2.adapters.block is deprecated: Gate 2 has no block "
              "scope; import ModuleScope from evograd.verification.gate2.adapters.module "
              "(scope='model')", DeprecationWarning, stacklevel=2)

__all__ = ["LegacyBlockScope", "ModuleScope"]


class LegacyBlockScope:
    """Removed: refuses to build."""

    def __init__(self, *args, **kwargs):
        raise ValueError("Gate 2 has no block scope (removed 2026-10-13): a tier-3 BlockAdapter "
                         "is not a Gate 2 scope; verify the sites as operators and inside the "
                         "whole model")
