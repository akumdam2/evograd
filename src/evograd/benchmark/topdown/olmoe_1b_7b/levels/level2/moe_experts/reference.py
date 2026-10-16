"""References for OLMoE's routed experts.

The boundary is ``OlmoeExperts.forward(hidden_states, top_k_index, top_k_weights)``:
for every token, the ``K`` experts the router chose each run a SwiGLU MLP on it
and their outputs are summed with the router's weights. The weights of all
``E`` experts arrive as two 3-D tensors, ``gate_up_proj [E, 2I, H]`` (each
expert's gate rows followed by its up rows) and ``down_proj [E, H, I]``.

**Oracle.** Every expert applied to every token in float32, then the sparse
routing weights applied -- a dense ``[T, E, 2I]`` intermediate, which states the
mathematics without any gather, sort or scatter. The spelling is the one the
operator suite's ``fused_moe_swiglu`` task carries, copied here rather than
imported so a workload package depends on no other task's package.

**Production.** The grouped-GEMM experts kernel Transformers 5 dispatches an
unmodified ``OlmoeForCausalLM`` to (``experts_implementation="grouped_mm"``):
tokens sorted by expert, one grouped GEMM for gate/up, SiLU times up, one
grouped GEMM down, the routing weights applied, the tokens restored and summed
over ``K`` in float32. Called through Transformers' own function on a shim that
carries exactly the attributes the function reads, so the site's spelling and
the model's are the same code. ``experts_implementation="eager"`` reaches the
per-expert loop in ``OlmoeExperts.forward`` the same way.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import hf_modules


def _check(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    if x.ndim != 2:
        raise ValueError(f"x must be [T, H], got {tuple(x.shape)}")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert weights must be 3-D: gate_up_proj [E, 2I, H], down_proj [E, H, I]")
    experts, gate_up, hidden = gate_up_proj.shape
    if gate_up % 2:
        raise ValueError("gate_up_proj's second axis must be twice the intermediate width")
    if down_proj.shape != (experts, hidden, gate_up // 2):
        raise ValueError(f"down_proj must be [{experts}, {hidden}, {gate_up // 2}], got "
                         f"{tuple(down_proj.shape)}")
    if x.shape[1] != hidden:
        raise ValueError(f"x's width {x.shape[1]} must match the experts' {hidden}")
    if top_k_index.shape != top_k_weights.shape or top_k_index.ndim != 2:
        raise ValueError("top_k_index and top_k_weights must both be [T, K]")
    if top_k_index.shape[0] != x.shape[0]:
        raise ValueError("top_k_index must have one row per token")
    if top_k_index.dtype not in (torch.int64, torch.int32):
        raise ValueError(f"top_k_index must be an integer tensor, got {top_k_index.dtype}")


def moe_experts_forward_ref(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    _check(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    experts = gate_up_proj.shape[0]
    routing = torch.zeros((x.shape[0], experts), device=x.device,
                          dtype=torch.float32).scatter_add(
        1, top_k_index.to(torch.int64), top_k_weights.float())
    pre = torch.einsum("th,egh->teg", x.float(), gate_up_proj.float())
    gate, up = pre.chunk(2, dim=-1)
    post = F.silu(gate) * up
    expert_out = torch.einsum("tei,ehi->teh", post, down_proj.float())
    return (expert_out * routing.unsqueeze(-1)).sum(dim=1).to(x.dtype)


class _ExpertsShim:
    """The attributes ``grouped_mm_experts_forward`` and the eager forward read
    off an ``OlmoeExperts`` module, on a plain object.

    A class rather than a ``SimpleNamespace``: Dynamo traces attribute access
    on a user-defined instance but refuses the namespace constructor, and the
    Tier-2 compile provider traces this spelling whole.
    """

    has_gate = True
    has_bias = False
    is_transposed = False
    is_concatenated = True

    def __init__(self, gate_up_proj, down_proj):
        self.num_experts = int(gate_up_proj.shape[0])
        self.hidden_dim = int(gate_up_proj.shape[2])
        self.intermediate_dim = int(down_proj.shape[2])
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = torch.nn.SiLU()

    def _apply_gate(self, gate_up_out):
        from transformers.integrations.moe import _default_apply_gate

        return _default_apply_gate(self, gate_up_out)


def _experts_shim(gate_up_proj, down_proj):
    return _ExpertsShim(gate_up_proj, down_proj)


def moe_experts_forward_production(x, gate_up_proj, down_proj, top_k_index, top_k_weights,
                                   *, experts_implementation: str = "grouped_mm"):
    """The experts kernel the model dispatches to, on these tensors."""
    _check(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    shim = _experts_shim(gate_up_proj, down_proj)
    if experts_implementation == "grouped_mm":
        from transformers.integrations.moe import grouped_mm_experts_forward

        return grouped_mm_experts_forward(shim, x, top_k_index, top_k_weights)
    if experts_implementation == "eager":
        return hf_modules.modeling.OlmoeExperts.forward.__wrapped__(
            shim, x, top_k_index, top_k_weights)
    raise ValueError(f"unknown experts_implementation {experts_implementation!r}")


__all__ = ["moe_experts_forward_production", "moe_experts_forward_ref"]
