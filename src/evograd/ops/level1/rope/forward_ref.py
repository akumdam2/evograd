"""Trusted PyTorch reference for rotary position embedding (RoPE).

Layout is ``[batch, heads, tokens, head_dim]`` — the HuggingFace convention
after ``view + transpose``, and the layout Liger's kernel expects. Keeping it
means a Liger baseline needs no transposes of its own, so the comparison
measures the kernel rather than layout shuffling.

Rotation follows the half-rotated (GPT-NeoX / Llama) convention:

    y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]

where ``x1``/``x2`` are the two halves of the head dimension. The interleaved
convention (GPT-J style) is a different function of the same inputs and produces
plausible-looking but wrong results, so the semantics say so explicitly.

``cos`` and ``sin`` are ``[tokens, head_dim]`` with the two halves duplicated,
which is what ``LlamaRotaryEmbedding`` emits. They are inputs rather than
constants because a real layer computes them once per step and shares them
across every layer; a kernel that recomputed them would be measuring the wrong
thing.
"""

from __future__ import annotations

import torch


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def rope_runtime_ref(x, cos, sin):
    """The exact spelling ``apply_rotary_pos_emb`` executes, in the model dtype.

    Transformers computes ``q*cos + rotate_half(q)*sin`` entirely at the model's
    precision -- there is no float32 upcast anywhere in it. The oracle above
    upcasts, which makes it a better reference and a *slower* one, so timing the
    eager baseline through it would charge the baseline for casts the model
    never executes and inflate every speedup.

    Broadcasting matches too: Transformers unsqueezes ``cos``/``sin`` at
    ``unsqueeze_dim=1`` to reach ``[1, 1, T, D]`` against ``[B, H, T, D]``, which
    is what ``cos[None, None]`` produces from this task's ``[T, D]`` tables.
    """
    if x.ndim != 4:
        raise ValueError(
            f"rope expects [batch, heads, tokens, head_dim], got {tuple(x.shape)}"
        )
    cos_b = cos[None, None]
    sin_b = sin[None, None]
    return x * cos_b + _rotate_half(x) * sin_b


def rope_forward_ref(x, cos, sin):
    """Apply RoPE to ``x`` of shape ``[batch, heads, tokens, head_dim]``.

    ``cos``/``sin`` are ``[tokens, head_dim]`` and broadcast over batch and
    heads. The rotation is evaluated in float32 and cast back: at bfloat16 the
    products ``x*cos`` and ``x*sin`` lose enough precision to dominate the
    tolerance, so this is the more accurate answer and the right oracle.

    It is *not* what the model runs. ``apply_rotary_pos_emb`` stays in the model
    dtype throughout -- ``rope_runtime_ref`` above reproduces it bitwise -- and
    the two differ by about 5e-3 relative at bfloat16. Timing the eager baseline
    through this spelling would charge it for casts Transformers never executes,
    which is why ``runtime_forward`` names the other one.
    """
    if x.ndim != 4:
        raise ValueError(f"rope expects [batch, heads, tokens, head_dim], got {tuple(x.shape)}")
    cos_b = cos[None, None].float()
    sin_b = sin[None, None].float()
    x32 = x.float()
    return (x32 * cos_b + _rotate_half(x32) * sin_b).to(x.dtype)
