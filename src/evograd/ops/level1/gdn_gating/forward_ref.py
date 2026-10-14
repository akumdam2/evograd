"""Reference for the Gated DeltaNet's decay and write-strength gates.

``g = -exp(A_log) * softplus(a + dt_bias)`` in float32 -- the log of the
per-step state decay -- and ``beta = sigmoid(b)`` in ``b``'s dtype. Spelled
exactly as ``Qwen3_5GatedDeltaNet.forward`` computes them, so the reference
is also the production spelling: the float32 promotion of ``g`` is part of the
definition (a bfloat16 ``exp(A_log)`` overflows the decay for large ``A``).
"""

import torch
import torch.nn.functional as F


def gdn_gating_forward_ref(a, b, A_log, dt_bias):
    if a.shape != b.shape:
        raise ValueError(f"a {tuple(a.shape)} and b {tuple(b.shape)} must match")
    if A_log.shape != (a.shape[-1],) or dt_bias.shape != (a.shape[-1],):
        raise ValueError("A_log and dt_bias must be [heads]")
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return g, beta
