"""The gated delta rule with in-kernel q/k L2 normalization.

The definition, per head, with ``S`` the ``[Dk, Dv]`` state starting at zero,
``q`` and ``k`` L2-normalized (eps 1e-6 inside the root) and ``q`` scaled by
``Dk^-0.5``::

    S_t   = exp(g_t) * S_{t-1}
    S_t   = S_t + k_t (beta_t * (v_t - S_t^T k_t))^T
    o_t   = S_t^T q_t

Two float32 oracle spellings: the per-token **recurrence** (exactly the
definition) up to :data:`RECURRENT_MAX_T` tokens, and Transformers' **torch
chunk** function on float32 inputs above it, where a recurrence through autograd
would keep one ``[B, HV, Dk, Dv]`` float32 state per step alive. The two agree to
~5e-7 in float32. Production is the torch chunk function on the model-dtype
inputs, reached through ``__wrapped__``. No initial and no final state.
"""

from __future__ import annotations

import torch

RECURRENT_MAX_T = 256
CHUNK_SIZE = 64


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def check(q, k, v, g, beta):
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)} "
                         "must be [B, T, HV, D]")
    if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("g and beta must be [B, T, HV]")


def recurrent_gated_delta_rule(q, k, v, g, beta):
    """The per-token recurrence, in float32. Inputs and output are [B, T, HV, D]."""
    qf = l2norm(q.float()) * q.shape[-1] ** -0.5
    kf = l2norm(k.float())
    vf, gf, bf = v.float(), g.float(), beta.float()
    batch, tokens, heads, dk = qf.shape
    state = qf.new_zeros(batch, heads, dk, vf.shape[-1])
    outputs = []
    for t in range(tokens):
        state = state * gf[:, t].exp()[..., None, None]
        k_t = kf[:, t]
        remembered = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (vf[:, t] - remembered) * bf[:, t][..., None]
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, qf[:, t]))
    return torch.stack(outputs, dim=1)


def forward_ref(q, k, v, g, beta, *, modeling):
    """The float32 oracle (recurrence on short sequences, float32 chunk above)."""
    check(q, k, v, g, beta)
    if q.shape[1] <= RECURRENT_MAX_T:
        out = recurrent_gated_delta_rule(q, k, v, g, beta)
    else:
        out, _ = modeling.torch_chunk_gated_delta_rule.__wrapped__(
            q.float(), k.float(), v.float(), g=g.float(), beta=beta.float(),
            chunk_size=CHUNK_SIZE, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
    return out.to(q.dtype)


def forward_production(q, k, v, g, beta, *, modeling):
    """Transformers' torch chunk implementation, exactly as the model calls it."""
    check(q, k, v, g, beta)
    out, _ = modeling.torch_chunk_gated_delta_rule.__wrapped__(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return out


__all__ = ["CHUNK_SIZE", "RECURRENT_MAX_T", "check", "forward_production", "forward_ref",
           "l2norm", "recurrent_gated_delta_rule"]
