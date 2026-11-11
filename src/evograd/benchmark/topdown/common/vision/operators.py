"""Executable boundaries taken from the published PyTorch architectures.

Each Level-2 function is the computation the HF module performs at that
boundary, in its op order, dtypes, casts and layouts: the model reference the
recorded payloads are replayed against bit for bit. Two kinds of boundary:

* a module boundary (:func:`bindings`): one call of a module (or, for
  ConvNeXt, a span of one layer's submodules) whose arguments ``pack`` reads
  off that call;
* a residual stream (:func:`residual_streams`): a chain of norms, each
  normalizing the sum of the residual stream and one branch (attention or
  MLP). A boundary there adds and normalizes (``*_residual_*``), across the
  blocks' module boundaries.
"""

from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F


def grn(x, weight, bias):
    norm = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    return weight * (x * (norm / (norm.mean(dim=-1, keepdim=True) + 1e-6))) + bias + x


def convnext_mlp(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    return F.linear(
        grn(F.gelu(F.linear(x, up_weight, up_bias)), grn_weight, grn_bias), down_weight, down_bias
    )


def text_mlp(x, gate_weight, up_weight, down_weight):
    return F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)


def vision_mlp(x, up_weight, up_bias, down_weight, down_bias):
    hidden = F.linear(x, up_weight, up_bias)
    return F.linear(hidden * torch.sigmoid(1.702 * hidden), down_weight, down_bias)


def spatial_conv(x, weight, bias, stride, padding, groups):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)


def dwconv_layernorm(x, dw_weight, dw_bias, ln_weight, ln_bias, eps=1e-6):
    """``ConvNextV2Layer``'s depthwise conv, NCHW -> NHWC, then its channels-last LayerNorm."""
    h = F.conv2d(x, dw_weight, dw_bias, 1, dw_weight.shape[-1] // 2, 1, x.shape[1])
    h = h.permute(0, 2, 3, 1)
    return F.layer_norm(h, (h.shape[-1],), ln_weight, ln_bias, eps)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, groups):
    """Transformers' SDPA ``repeat_kv``: each KV head repeated for its query heads."""
    if groups == 1:
        return x
    b, heads, t, d = x.shape
    return x[:, :, None, :, :].expand(b, heads, groups, t, d).reshape(b, heads * groups, t, d)


def vision_attention(x, qkv_weight, qkv_bias, cos, sin, proj_weight, proj_bias):
    """``VisionAttention.forward`` (SDPA) over one image's tokens (one ``cu_seqlens`` segment).

    The 2-D rotary embedding runs in float32 and is cast back
    (``apply_rotary_pos_emb_vision``); attention is non-causal.
    """
    t, d = x.shape[0], cos.shape[-1]
    q, k, v = F.linear(x, qkv_weight, qkv_bias).reshape(t, 3, x.shape[-1] // d, -1) \
        .permute(1, 0, 2, 3).unbind(0)
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = ((q * cos) + (rotate_half(q) * sin)).to(q_dtype)
    k = ((k * cos) + (rotate_half(k) * sin)).to(k_dtype)
    q, k, v = (s.transpose(0, 1).unsqueeze(0) for s in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=d**-0.5,
                                         is_causal=False)
    out = out.transpose(1, 2).contiguous().reshape(t, -1).contiguous()
    return F.linear(out, proj_weight, proj_bias)


def text_attention_block(x, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, cos, sin,
                         attention_mask, o_weight):
    """``Qwen2VLAttention.forward`` (SDPA) with the explicit mask the model passes.

    ``cos``/``sin`` are the multimodal tables after their section interleave
    (``apply_multimodal_rotary_pos_emb`` before ``unsqueeze``). With a mask,
    Transformers' SDPA path repeats the KV heads and calls SDPA without
    ``is_causal``.
    """
    b, t, _ = x.shape
    d = cos.shape[-1]
    q = F.linear(x, q_weight, q_bias).view(b, t, -1, d).transpose(1, 2)
    k = F.linear(x, k_weight, k_bias).view(b, t, -1, d).transpose(1, 2)
    v = F.linear(x, v_weight, v_bias).view(b, t, -1, d).transpose(1, 2)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    groups = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(q, repeat_kv(k, groups), repeat_kv(v, groups),
                                         attn_mask=attention_mask, dropout_p=0.0, scale=d**-0.5,
                                         is_causal=False)
    out = out.transpose(1, 2).contiguous().reshape(b, t, -1).contiguous()
    return F.linear(out, o_weight)


def vision_residual_layernorm(x, r, weight, bias, eps=1e-6):
    """``residual + branch`` of ``Qwen2VLVisionBlock``, then the next ``nn.LayerNorm``."""
    summed = r + x
    return F.layer_norm(summed, (summed.shape[-1],), weight, bias, eps), summed


def text_residual_rmsnorm(x, r, weight, eps=1e-6):
    """``residual + hidden_states`` of ``Qwen2VLDecoderLayer``, then the next ``Qwen2VLRMSNorm``
    (float32 normalization, cast back, then the weight product)."""
    summed = r + x
    h = summed.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(summed.dtype), summed


FUNCTIONS = {"convnext_mlp": convnext_mlp, "dwconv_layernorm": dwconv_layernorm,
             "text_mlp": text_mlp, "vision_mlp": vision_mlp,
             "vision_attention": vision_attention, "text_attention_block": text_attention_block,
             "vision_residual_layernorm": vision_residual_layernorm,
             "text_residual_rmsnorm": text_residual_rmsnorm}


def calls_per_step(model, arch):
    """``{site: calls}`` of one forward (and its backward), from the architecture."""
    if model == "convnextv2_tiny":
        layers = sum(arch["depths"])
        return {"convnext_mlp": layers, "dwconv_layernorm": layers}
    text, vision = arch["num_hidden_layers"], arch["vision_config"]["depth"]
    return {"text_mlp": text, "vision_mlp": vision, "vision_attention": vision,
            "vision_residual_layernorm": 2 * vision, "text_attention_block": text,
            "text_residual_rmsnorm": 2 * text}


# ── module boundaries ────────────────────────────────────────────────────────


class Binding(NamedTuple):
    """One module boundary: ``pack(args, kwargs)`` of ``first``'s call gives the
    task arguments in declared order; ``last``'s output (the first element of a
    tuple) is the task output."""

    site: str
    path: str
    owner: torch.nn.Module
    first: torch.nn.Module
    last: torch.nn.Module
    pack: Callable


def _hidden(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _sdpa(module):
    if module.config._attn_implementation != "sdpa":
        raise ValueError(f"{type(module).__name__}: the Level-2 attention boundary is the SDPA "
                         f"path, not {module.config._attn_implementation!r}")


def mrope(table, section):
    """``apply_multimodal_rotary_pos_emb``'s section interleave of a ``[3, B, T, D]`` table."""
    return torch.cat([m[i % 3] for i, m in enumerate(table.split(list(section) * 2, dim=-1))],
                     dim=-1)


def bindings(model):
    """Every module boundary of ``model``, in module order."""
    for path, module in model.named_modules():
        kind = type(module).__name__
        if kind == "ConvNextV2Layer":

            def pack(args, kwargs, m=module):
                ln = m.layernorm
                return (_hidden(args, kwargs), m.dwconv.weight, m.dwconv.bias, ln.weight, ln.bias,
                        ln.eps)

            yield Binding("dwconv_layernorm", path, module, module.dwconv, module.layernorm, pack)

            def pack(args, kwargs, m=module):
                return (_hidden(args, kwargs), m.pwconv1.weight, m.pwconv1.bias, m.grn.weight,
                        m.grn.bias, m.pwconv2.weight, m.pwconv2.bias)

            yield Binding("convnext_mlp", path, module, module.pwconv1, module.pwconv2, pack)
        elif kind == "Qwen2MLP":

            def pack(args, kwargs, m=module):
                return (_hidden(args, kwargs), m.gate_proj.weight, m.up_proj.weight,
                        m.down_proj.weight)

            yield Binding("text_mlp", path, module, module, module, pack)
        elif kind == "VisionMlp":

            def pack(args, kwargs, m=module):
                return (_hidden(args, kwargs), m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias)

            yield Binding("vision_mlp", path, module, module, module, pack)
        elif kind == "VisionAttention":

            def pack(args, kwargs, m=module):
                _sdpa(m)
                cu_seqlens = kwargs["cu_seqlens"] if "cu_seqlens" in kwargs else args[1]
                if cu_seqlens.numel() != 2:
                    raise ValueError(f"{path}: {cu_seqlens.numel() - 1} images in one vision call; "
                                     "the boundary attends over one image's tokens")
                cos, sin = kwargs["position_embeddings"]
                return (_hidden(args, kwargs), m.qkv.weight, m.qkv.bias, cos, sin, m.proj.weight,
                        m.proj.bias)

            yield Binding("vision_attention", path, module, module, module, pack)
        elif kind == "Qwen2VLAttention":

            def pack(args, kwargs, m=module):
                _sdpa(m)
                mask = kwargs.get("attention_mask")
                if not (torch.is_tensor(mask) and mask.dtype == torch.bool):
                    raise ValueError(f"{path}: text attention without a boolean padding mask "
                                     "(unpadded rows call SDPA with is_causal); the boundary is "
                                     "the right-padded batch's masked call")
                section = m.config.rope_parameters["mrope_section"]
                cos, sin = (mrope(t, section) for t in kwargs["position_embeddings"])
                return (_hidden(args, kwargs), m.q_proj.weight, m.q_proj.bias, m.k_proj.weight,
                        m.k_proj.bias, m.v_proj.weight, m.v_proj.bias, cos, sin, mask,
                        m.o_proj.weight)

            yield Binding("text_attention_block", path, module, module, module, pack)


# ── residual streams ─────────────────────────────────────────────────────────


class Stream(NamedTuple):
    """One residual stream: ``norms`` are N_0 .. N_n and ``branches`` B_1 .. B_n, in
    execution order. Boundary k (1..n) adds B_k's output to N_{k-1}'s input (the
    stream so far) and N_k normalizes that sum; N_0 has no preceding add. Each
    block holds two boundaries (after its attention and after its MLP); the
    last normalizes into the final norm (``merger.ln_q``, ``model.norm``)."""

    site: str
    blocks: tuple
    norms: tuple
    branches: tuple
    #: ``norm -> (weight, [bias,] eps)``: the task arguments after ``(x, r)``.
    norm_args: Callable


def _layernorm_args(m):
    return m.weight, m.bias, m.eps


def _rmsnorm_args(m):
    return m.weight, m.variance_epsilon


def residual_streams(model):
    """Every residual stream of ``model`` (Qwen2-VL's vision tower and text decoder)."""
    for path, module in model.named_modules():
        kind = type(module).__name__
        if kind == "Qwen2VisionTransformerPretrainedModel":
            site, attn, mlp, first, second, final, args = (
                "vision_residual_layernorm", "attn", "mlp", "norm1", "norm2", "merger.ln_q",
                _layernorm_args)
            blocks = tuple((f"{path}.blocks.{i}", b) for i, b in enumerate(module.blocks))
        elif kind == "Qwen2VLTextModel":
            site, attn, mlp, first, second, final, args = (
                "text_residual_rmsnorm", "self_attn", "mlp", "input_layernorm",
                "post_attention_layernorm", "norm", _rmsnorm_args)
            blocks = tuple((f"{path}.layers.{i}", b) for i, b in enumerate(module.layers))
        else:
            continue
        modules = dict(module.named_modules())
        norms = [(f"{blocks[0][0]}.{first}", getattr(blocks[0][1], first))]
        branches = []
        for i, (block_path, block) in enumerate(blocks):
            after = (f"{blocks[i + 1][0]}.{first}", getattr(blocks[i + 1][1], first)) \
                if i + 1 < len(blocks) else (f"{path}.{final}", modules[final])
            norms += [(f"{block_path}.{second}", getattr(block, second)), after]
            branches += [(f"{block_path}.{attn}", getattr(block, attn)),
                         (f"{block_path}.{mlp}", getattr(block, mlp))]
        yield Stream(site, blocks, tuple(norms), tuple(branches), args)
