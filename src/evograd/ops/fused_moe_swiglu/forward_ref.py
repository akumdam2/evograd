"""PyTorch reference for routed grouped-GEMM SwiGLU MoE."""

import torch
import torch.nn.functional as F


def fused_moe_swiglu_forward_ref(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape [T, H]")
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("expert weights must be 3D tensors")
    if top_k_index.shape != top_k_weights.shape:
        raise ValueError("top_k_index and top_k_weights must have matching shape")
    if gate_up_proj.shape[1] % 2:
        raise ValueError("gate_up_proj dimension 1 must be twice the intermediate size")

    expert_index = top_k_index.to(torch.int64)
    routing_weights = torch.zeros(
        (x.shape[0], gate_up_proj.shape[0]),
        device=x.device,
        dtype=top_k_weights.dtype,
    ).scatter_add(1, expert_index, top_k_weights)

    # Compute expert contractions densely, then apply the sparse routing
    # weights. This is semantically identical to indexing selected expert
    # weights, but avoids materializing [T,K,*,H] gathered weight tensors in a
    # primitive Pipeline-B seed. Evolution can fuse this into grouped/top-k
    # GEMMs while retaining a coverage-safe initial program.
    pre_activation = torch.einsum("th,egh->teg", x, gate_up_proj)
    gate, up = pre_activation.chunk(2, dim=-1)
    post_activation = (F.silu(gate.float()) * up.float()).to(x.dtype)

    expert_output = torch.einsum("tei,ehi->teh", post_activation, down_proj)
    return (
        expert_output.float() * routing_weights.float().unsqueeze(-1)
    ).sum(dim=1).to(x.dtype)
