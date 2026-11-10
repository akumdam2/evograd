"""MegaFold's eager modules, copied for CPU tests: the mathematics, not the Triton imports.

Each class below is ``megafold/model/megafold.py`` or ``megafold/model/attention.py``
at the pinned commit, with attribute names kept so that the site adapters read
these modules exactly as they read MegaFold's. The collectives, which are no-ops
at ``sp_size`` 1, are left out.

CPU autocast does not run CUDA's op lists, so what CUDA autocast does to each op
is spelled out in :class:`Amp` (the lists are ``ATen/autocast_mode.h``, torch 2.12).
"""

from __future__ import annotations

import contextlib
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from evograd.evaluation.tier3.workloads.megafold import sites as S
from evograd.evaluation.tier3.workloads.megafold import workload as W


class Amp:
    """One op as CUDA autocast at ``dtype`` runs it; ``None`` is autocast off."""

    dtype: torch.dtype | None = None

    def _down(self, *tensors):
        return tuple(t if t is None or self.dtype is None else t.to(self.dtype) for t in tensors)

    def _up(self, *tensors):
        return tuple(t if t is None or self.dtype is None else t.float() for t in tensors)

    def layer_norm(self, x, shape, weight=None, bias=None, eps=1e-5):
        return F.layer_norm(*self._up(x)[:1], shape, *self._up(weight, bias), eps)

    def linear(self, x, weight, bias=None):
        return F.linear(*self._down(x, weight, bias))

    def einsum(self, equation, *tensors):
        return torch.einsum(equation, *self._down(*tensors))

    def softmax(self, x, dim=-1):
        return self._up(x)[0].softmax(dim=dim)


AMP = Amp()


@contextlib.contextmanager
def autocast(dtype):
    """``MegaFold.forward``'s autocast at ``dtype``, for the eager modules and the adapters."""
    AMP.dtype = dtype
    try:
        with mock.patch.object(W, "_autocast_dtype", lambda _t: dtype), \
                mock.patch.object(S, "_autocast_dtype", lambda _t: dtype):
            yield
    finally:
        AMP.dtype = None


class LayernormLinear(nn.Module):
    """``FusedLayernormLinear``'s torch path."""

    def __init__(self, K, N, has_linear_bias=True):
        super().__init__()
        self.K, self.N = K, N
        self.has_layernorm_bias = True
        self.WEIGHT = nn.Parameter(1 + 0.1 * torch.randn(K))
        self.BIAS = nn.Parameter(0.1 * torch.randn(K))
        self.linear_weight = nn.Parameter(torch.randn(K, N) * K ** -0.5)
        self.linear_bias = nn.Parameter(0.1 * torch.randn(N)) if has_linear_bias else None

    def forward(self, X):
        return AMP.linear(AMP.layer_norm(X, (self.K,), self.WEIGHT, self.BIAS),
                          self.linear_weight.T, self.linear_bias)


class Dropout(nn.Module):
    def __init__(self, prob, *, dropout_type=None):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type

    def forward(self, t):
        if self.dropout_type == "row":
            batch, _, col, dim = t.shape
            ones_shape = (batch, 1, col, dim)
        elif self.dropout_type == "col":
            batch, row, _, dim = t.shape
            ones_shape = (batch, row, 1, dim)
        else:
            return self.dropout(t)
        return t * self.dropout(t.new_ones(ones_shape))


def to_pairwise_mask(mask):
    return mask.unsqueeze(-1) & mask.unsqueeze(-2)


class TriangleMultiplication(nn.Module):
    def __init__(self, *, dim, dim_hidden=None, mix="incoming", dropout=0.0, dropout_type=None):
        super().__init__()
        dim_hidden = dim_hidden or dim
        self.dim_hidden = dim_hidden
        self.combined = LayernormLinear(dim, dim_hidden * 5)
        self.glu = nn.GLU(dim=-1)
        self.mix = mix
        self.mix_einsum_eq = {"outgoing": "... i k d, ... j k d -> ... i j d",
                              "incoming": "... k i d, ... k j d -> ... i j d"}[mix]
        self.to_out = nn.Sequential(LayernormLinear(dim_hidden, dim, has_linear_bias=False),
                                    Dropout(dropout, dropout_type=dropout_type))

    def forward(self, x, mask=None):
        if mask is not None:
            mask = to_pairwise_mask(mask)[..., None]
        combined_out = self.combined(x)
        left_and_right = combined_out[..., :self.dim_hidden * 4]
        out_gate = combined_out[..., self.dim_hidden * 4:].sigmoid()
        left, right = self.glu(left_and_right).chunk(2, dim=-1)
        if mask is not None:
            left, right = left * mask, right * mask
        out = AMP.einsum(self.mix_einsum_eq, left, right)
        return self.to_out(out) * out_gate


class Linear(nn.Linear):
    """``nn.Linear`` under autocast."""

    def forward(self, x):
        return AMP.linear(x, self.weight, self.bias)


class LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` under autocast."""

    def forward(self, x):
        return AMP.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class PreLayerNorm(nn.Module):
    def __init__(self, fn, *, dim):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attend(nn.Module):
    """``Attend.forward``'s default path."""

    is_local_attn, scale, enable_attn_softclamp, dropout = False, None, False, 0.0

    def forward(self, q, k, v, mask=None, attn_bias=None, batch_size=1):
        dtype = q.dtype
        q = q * q.shape[-1] ** -0.5
        sim = AMP.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_bias is not None:
            sim = sim + attn_bias
        if mask is not None:
            sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(sim.dtype).max)
        attn = AMP.softmax(sim).to(dtype)
        return AMP.einsum("b h i j, b h j d -> b h i d", attn, v)


class Attention(nn.Module):
    def __init__(self, *, dim, dim_head, heads, init_gate_bias=-2.0):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.attend = Attend()
        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, dim, bias=False)
        self.to_gates = Linear(dim, dim_inner)
        nn.init.normal_(self.to_gates.weight, std=0.1)
        nn.init.constant_(self.to_gates.bias, init_gate_bias)

    def split_heads(self, t):
        b, n, width = t.shape
        return t.reshape(b, n, self.heads, width // self.heads).permute(0, 2, 1, 3)

    def forward(self, seq, mask=None, attn_bias=None, **kwargs):
        q = self.to_q(seq)
        k, v = self.to_kv(seq).chunk(2, dim=-1)
        q, k, v = (self.split_heads(t) for t in (q, k, v))
        out = self.attend(q, k, v, attn_bias=attn_bias, mask=mask, **kwargs)
        b, h, n, d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out(out * self.to_gates(seq).sigmoid())


class Rearrange(nn.Module):
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, t):
        return t.permute(*self.order)


class TriangleAttention(nn.Module):
    def __init__(self, *, dim, heads, node_type, dropout=0.0, dropout_type=None, dim_head=4):
        super().__init__()
        self.need_transpose = node_type == "ending"
        self.to_attn_bias = nn.Sequential(Linear(dim, heads, bias=False), Rearrange(0, 3, 1, 2))
        self.dropout = Dropout(dropout, dropout_type=dropout_type)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)

    def forward(self, pairwise_repr, mask=None, **kwargs):
        b = len(pairwise_repr)
        if self.need_transpose:
            pairwise_repr = pairwise_repr.permute(0, 2, 1, 3)
        attn_bias = self.to_attn_bias(pairwise_repr)
        batch_repeat = pairwise_repr.shape[1]
        attn_bias = attn_bias.repeat_interleave(batch_repeat, dim=0)
        if mask is not None:
            mask = mask.repeat_interleave(batch_repeat, dim=0)
        shape = pairwise_repr.shape
        packed = torch.cat([pairwise_repr.reshape(-1, *shape[2:])])  # ``pack_one``: einops concatenates
        out = self.attn(packed, mask=mask, attn_bias=attn_bias, batch_size=b, **kwargs)
        out = out.reshape(shape)
        if self.need_transpose:
            out = out.permute(0, 2, 1, 3)
        return self.dropout(out)


class SwiGLU(nn.Module):
    """``FusedSwiGLU``'s mathematics: ``silu(gates) * x``, in the dtype it arrives in."""

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class FusedTransition(nn.Module):
    def __init__(self, dim, expansion_factor=4, include_ln=True):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        first = (LayernormLinear(dim, dim_inner * 2, has_linear_bias=False) if include_ln
                 else Linear(dim, dim_inner * 2, bias=False))
        self.ff = nn.Sequential(first, SwiGLU(), Linear(dim_inner, dim, bias=False))

    def forward(self, x):
        return self.ff(x)


class AttentionPairBias(nn.Module):
    """Non-windowed: the windowing and extra-bias branches are left out."""

    def __init__(self, *, heads, dim_pairwise, window_size=None, dim, dim_head=4):
        super().__init__()
        self.window_size = window_size
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head)
        self.to_attn_bias = nn.Sequential(LayernormLinear(dim_pairwise, heads, has_linear_bias=False),
                                          Rearrange(0, 3, 1, 2))

    def forward(self, single_repr, *, pairwise_repr, **kwargs):
        attn_bias = self.to_attn_bias(pairwise_repr) + 0.0
        return self.attn(single_repr, attn_bias=attn_bias, batch_size=pairwise_repr.shape[0], **kwargs)


class OuterProductMean(nn.Module):
    def __init__(self, *, dim_msa=16, dim_pairwise=12, dim_hidden=4, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.to_hidden = LayernormLinear(dim_msa, dim_hidden * 2, has_linear_bias=False)
        self.to_pairwise_repr = Linear(dim_hidden ** 2, dim_pairwise)

    def forward(self, msa, *, mask=None, msa_mask=None):
        dtype = msa.dtype
        a, b = self.to_hidden(msa).chunk(2, dim=-1)
        msa_mask, num_true_in_msa_mask = msa_mask
        a = a * msa_mask[..., None, None].type(dtype)
        b = b * msa_mask[..., None, None].type(dtype)
        outer_product = AMP.einsum("b s i d, b s j e -> b i j d e", a, b)
        outer_product = outer_product.flatten(-2)
        if mask is not None:
            outer_product = outer_product * to_pairwise_mask(mask)[..., None].type(dtype)
        pairwise_repr = self.to_pairwise_repr(outer_product)
        return pairwise_repr / num_true_in_msa_mask[..., None, None, None].clamp(min=self.eps)


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, *, dim_msa=16, dim_pairwise=8, dim_head=3, heads=2, dropout=0.0, dropout_type=None):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.msa_to_values_and_gates = nn.Sequential(
            LayernormLinear(dim_msa, dim_inner * 2, has_linear_bias=False), Unstack(heads))
        self.pairwise_repr_to_attn = nn.Sequential(
            LayernormLinear(dim_pairwise, heads, has_linear_bias=False), Rearrange(0, 3, 1, 2))
        self.to_out = nn.Sequential(MergeHeads(), Linear(dim_inner, dim_msa, bias=False),
                                    Dropout(dropout, dropout_type=dropout_type))

    def forward(self, *, msa, pairwise_repr, mask=None):
        b = self.pairwise_repr_to_attn(pairwise_repr)
        values, gates = self.msa_to_values_and_gates(msa)
        gates = gates.sigmoid()
        if mask is not None:
            b = b.masked_fill(~mask[:, None, None, :], -torch.finfo(b.dtype).max)
        weights = AMP.softmax(b)
        out = AMP.einsum("b h i j, b h s j d -> b h s i d", weights, values)
        out = out * gates
        return self.to_out(out)


class Unstack(nn.Module):
    """``Rearrange("b s n (gv h d) -> gv b h s n d", gv=2, h=heads)``."""

    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, t):
        b, s, n, width = t.shape
        return t.reshape(b, s, n, 2, self.heads, width // (2 * self.heads)).permute(3, 0, 4, 1, 2, 5)


class MergeHeads(nn.Module):
    """``Rearrange("b h s n d -> b s n (h d)")``."""

    def forward(self, t):
        b, h, s, n, d = t.shape
        return t.permute(0, 2, 3, 1, 4).reshape(b, s, n, h * d)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, *, dim, dim_cond):
        super().__init__()
        self.norm = LayerNorm(dim, elementwise_affine=False)
        self.norm_cond = LayerNorm(dim_cond, bias=False)
        self.to_gamma = nn.Sequential(Linear(dim_cond, dim), nn.Sigmoid())
        self.to_beta = Linear(dim_cond, dim, bias=False)

    def forward(self, x, cond):
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)
        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta


class ConditionWrapper(nn.Module):
    def __init__(self, fn, *, dim, dim_cond):
        super().__init__()
        self.fn = fn
        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.to_adaln_zero_gamma = nn.Sequential(Linear(dim_cond, dim), nn.Sigmoid())

    def forward(self, x, *, cond, **kwargs):
        x = self.adaptive_norm(x, cond=cond)
        out = self.fn(x, **kwargs)
        return out * self.to_adaln_zero_gamma(cond)


class Tree(nn.Module):
    """One of everything a step runs, in MegaFold's order, with its residual adds."""

    def __init__(self, dim=16):
        super().__init__()
        self.outer_product_mean = OuterProductMean(dim_msa=dim, dim_pairwise=dim)
        self.msa_pair_weighted_avg = MSAPairWeightedAveraging(dim_msa=dim, dim_pairwise=dim, dropout=0.15,
                                                              dropout_type="row")
        self.msa_transition = FusedTransition(dim)
        self.tri_mult_outgoing = TriangleMultiplication(dim=dim, mix="outgoing", dropout=0.25, dropout_type="row")
        self.tri_mult_incoming = TriangleMultiplication(dim=dim, mix="incoming", dropout=0.25, dropout_type="row")
        self.tri_attn_starting = PreLayerNorm(TriangleAttention(
            dim=dim, heads=2, node_type="starting", dropout=0.25, dropout_type="row"), dim=dim)
        self.tri_attn_ending = PreLayerNorm(TriangleAttention(
            dim=dim, heads=2, node_type="ending", dropout=0.25, dropout_type="col"), dim=dim)
        self.pairwise_transition = FusedTransition(dim)
        self.pair_bias_attn = PreLayerNorm(AttentionPairBias(heads=2, dim_pairwise=dim, dim=dim), dim=dim)
        self.single_transition = FusedTransition(dim)
        self.conditioned_attn = ConditionWrapper(AttentionPairBias(heads=2, dim_pairwise=dim, dim=dim),
                                                 dim=dim, dim_cond=dim)
        self.conditioned_transition = ConditionWrapper(FusedTransition(dim, 2, include_ln=False),
                                                       dim=dim, dim_cond=dim)

    def forward(self, msa, pair, single, mask, msa_mask):
        count = msa_mask.float().sum(-1)
        pair = self.outer_product_mean(msa, mask=mask, msa_mask=(msa_mask, count)) + pair
        msa = self.msa_pair_weighted_avg(msa=msa, pairwise_repr=pair, mask=mask) + msa
        msa = self.msa_transition(msa) + msa
        pair = self.tri_mult_outgoing(pair, mask=mask) + pair
        pair = self.tri_mult_incoming(pair, mask=mask) + pair
        pair = self.tri_attn_starting(pair, mask=mask) + pair
        pair = self.tri_attn_ending(pair, mask=mask) + pair
        pair = self.pairwise_transition(pair) + pair
        single = self.pair_bias_attn(single, pairwise_repr=pair, mask=mask) + single
        single = self.single_transition(single) + single
        with _autocast_off():
            single, pair = single.float(), pair.float()
            rows = single.expand(4, *single.shape[1:])
            rows = self.conditioned_attn(rows, cond=rows, pairwise_repr=pair,
                                         mask=mask.expand(4, mask.shape[1])) + rows
            rows = self.conditioned_transition(rows, cond=rows) + rows
        return msa.float().sum() + pair.sum() + rows.sum()


@contextlib.contextmanager
def _autocast_off():
    """``disable_*_casting``: the diffusion module runs with autocast off."""
    before = AMP.dtype
    AMP.dtype = None
    with mock.patch.object(W, "_autocast_dtype", lambda _t: None), \
            mock.patch.object(S, "_autocast_dtype", lambda _t: None):
        try:
            yield
        finally:
            AMP.dtype = before
