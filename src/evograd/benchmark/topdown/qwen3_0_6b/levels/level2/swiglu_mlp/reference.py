"""PyTorch reference for Qwen3's gated (SwiGLU) MLP block.

The reference accumulates the gate/up product in float32 and casts once, which
is deliberately *more* accurate than what Transformers runs: ``Qwen3MLP`` calls
``self.act_fn(...) * self.up_proj(x)`` entirely in the model dtype. A reference
exists to be the correct answer, not to reproduce a particular rounding, and
every other declaration here follows the same convention (see
``fused_moe_swiglu``). The cost of that choice is measured rather than assumed:
the Level-2 verification reports the reference against the captured
``Qwen3MLP`` invocation *and* against the BF16 spelling, so the difference
between the two is a number in a report instead of a footnote.
"""

import torch
import torch.nn.functional as F


def qwen3_swiglu_mlp_forward_ref(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] != gate_weight.shape[-1]:
        raise ValueError(
            f"x's last dim {x.shape[-1]} must match gate_weight's {gate_weight.shape[-1]}"
        )
    if gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must have the same shape [I, H]")
    if down_weight.shape != (gate_weight.shape[1], gate_weight.shape[0]):
        raise ValueError(
            f"down_weight must be [H, I] = "
            f"{(gate_weight.shape[1], gate_weight.shape[0])}, got {tuple(down_weight.shape)}"
        )

    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = F.silu(gate.float()) * up.float()
    hidden = hidden.to(x.dtype)
    return F.linear(hidden, down_weight)


def qwen3_swiglu_mlp_forward_hf(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """The spelling ``Qwen3MLP.forward`` actually executes: no float32 upcast.

    Not the declared reference -- kept so the verification can report what the
    upcast costs, rather than leaving the two contracts silently different.
    """
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)
