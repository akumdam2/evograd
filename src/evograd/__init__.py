"""evograd: evolved backward kernels — seed via pipelines, optimize via OpenEvolve."""

__version__ = "0.1.0"

# Every evograd process runs SDPA under one policy (cuDNN's backward returns wrong
# gradients when one process mixes upstream-gradient layouts). Imports no torch:
# applies when torch is, or now if it already is. See evograd.sdpa_policy.
from evograd import sdpa_policy as _sdpa_policy  # noqa: E402

_sdpa_policy.install()
