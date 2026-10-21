"""Reviewed Liger RoPE autograd-pair adapter.

Liger's kernel rotates *q and k together* — ``rope_forward(q, k, cos, sin)``
returns both — while this declaration is a single-tensor operator, because
``OpDecl`` describes one output. The adapter bridges that by passing a
one-head slice as ``k``.

The overhead this costs the baseline is bounded and worth stating precisely.
The kernel launches one program per token and loads ``pad_n_q_head`` and
``pad_n_kv_head`` head tiles; with ``n_kv_head = 1`` the block size is unchanged
(``BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)``), so the launch geometry is
identical to a q-only rotation and the extra work is one head's load/store per
token — about ``1/n_heads``, i.e. ~3% at Llama-3's 32 heads.

The alternative, passing the same tensor as both q and k, would have doubled the
baseline's work and made Liger look 2x slower than it is.

Liger also calls ``.contiguous()`` on q and k internally after transposing, so
handing it a slice costs no more than a real model's call does.
"""


def make_liger_rope_autograd_pair_fns():
    from liger_kernel.ops.rope import rope_backward, rope_forward

    def forward_with_saved(x, cos, sin):
        # k is a one-head slice: the smallest tensor that keeps the joint API
        # satisfied. See the module docstring for the overhead this implies.
        #
        # It must be a copy, not a view. Liger rotates in place after calling
        # `.contiguous()`, and for a single kv head that call is a no-op —
        # transposing a size-1 dimension leaves the tensor contiguous, so the
        # kernel writes straight back into `x`. A view here silently mutates a
        # benchmark input; one head's worth of copy is the price of not doing
        # that, which is the same ~1/n_heads order as the overhead already
        # documented above.
        k = x[:, :1].clone()
        # The declaration's cos/sin are ``[T, head_dim]``, broadcast over batch
        # and heads. Liger reads ``cos.shape[0]`` as a batch size and selects
        # its addressing branch from it (``tl.where(cos_bs == 1, ...)``): a 2-D
        # table makes ``cos_bs == T``, which takes the per-batch branch and
        # reads past the end of the table for every ``batch_idx > 0``. A leading
        # 1 selects the broadcast branch. ``B == 1`` hides this entirely — the
        # batch offset is zero there — so it only appears at ``B > 1``.
        cos = cos.unsqueeze(0) if cos.dim() == 2 else cos
        sin = sin.unsqueeze(0) if sin.dim() == 2 else sin
        q_rot, _k_rot, cos_out, sin_out = rope_forward(x, k, cos, sin)
        return q_rot, (cos_out, sin_out)

    def backward_from_saved(dy, saved):
        cos, sin = saved
        # Same reason as the forward: `rope_backward` writes its result into the
        # tensors it is handed, and the single-head slice survives
        # `.contiguous()` unchanged, so a view would rotate part of `dy` in
        # place. `dy` itself is safe — transposing its n_heads > 1 layout leaves
        # it non-contiguous, so Liger's `.contiguous()` really does copy.
        dk = dy[:, :1].clone()
        # ``rope_backward`` returns two tensors, not four: it hands back dq and
        # dk only, while ``rope_forward`` also passes cos and sin through.
        dq, _dk = rope_backward(dy, dk, cos, sin)
        return (dq,)

    return forward_with_saved, backward_from_saved
