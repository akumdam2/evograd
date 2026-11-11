"""Qwen3's self-attention module, spelled as ``Qwen3Attention.forward`` computes it.

One spelling, the model's own arithmetic in its own order, dtypes and layouts
(Transformers 5.16.1, ``attn_implementation="sdpa"``, training, no cache):

* ``q_proj``/``k_proj``/``v_proj`` as ``F.linear`` on ``x [B, T, H]``, viewed as
  ``[B, T, heads, D]``;
* the per-head ``q_norm``/``k_norm`` as ``Qwen3RMSNorm``: float32 variance and
  ``rsqrt``, cast back to the input dtype, *then* the weight product (two
  roundings);
* the transpose to head-major ``[B, heads, T, D]`` and ``apply_rotary_pos_emb``
  in the model dtype (``rotate_half`` by concatenation);
* ``sdpa_attention_forward``'s call: ``attn_mask=None``, ``dropout_p=0.0``,
  ``scale=D**-0.5`` (``module.scaling``), ``is_causal`` for more than one
  query, and ``enable_gqa=True`` when query heads outnumber KV heads (no mask,
  ``D <= 256``);
* the transpose back, ``contiguous``, the merge to ``[B, T, HQ*D]`` and
  ``o_proj`` as ``F.linear``.

``cos``/``sin`` are the rotary tables' first row ``[1, T, D]``. The model's are
``[B, T, D]`` with identical rows; broadcasting one row is elementwise and
gives the same bits.
"""

import torch
import torch.nn.functional as F

#: ``use_gqa_in_sdpa``: wider heads fall back to ``repeat_kv``, outside this contract.
_GQA_MAX_HEAD_DIM = 256


def _rms_norm(heads: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen3RMSNorm.forward``."""
    input_dtype = heads.dtype
    hidden = heads.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return weight * hidden.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``rotate_half`` of ``modeling_qwen3``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_attention_block_forward(x, q_weight, k_weight, v_weight, o_weight,
                                  q_norm_weight, k_norm_weight, cos, sin, eps=1e-6):
    head_dim = q_norm_weight.shape[-1]
    if k_weight.shape[0] % head_dim or q_weight.shape[0] % k_weight.shape[0]:
        raise ValueError("query heads must be a multiple of the KV heads, both of width D")
    if head_dim > _GQA_MAX_HEAD_DIM:
        raise ValueError(f"head_dim {head_dim} > {_GQA_MAX_HEAD_DIM}: the model would repeat_kv")
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query = _rms_norm(F.linear(x, q_weight).view(hidden_shape), q_norm_weight, eps).transpose(1, 2)
    key = _rms_norm(F.linear(x, k_weight).view(hidden_shape), k_norm_weight, eps).transpose(1, 2)
    value = F.linear(x, v_weight).view(hidden_shape).transpose(1, 2)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)

    gqa = {"enable_gqa": True} if query.shape[1] != key.shape[1] else {}
    attn = F.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, scale=head_dim**-0.5,
        is_causal=query.shape[2] > 1, **gqa)
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.reshape(*input_shape, -1).contiguous()
    return F.linear(attn, o_weight)


__all__ = ["qwen3_attention_block_forward"]
