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
        k = x[:, :1]
        q_rot, _k_rot, cos_out, sin_out = rope_forward(x, k, cos, sin)
        return q_rot, (cos_out, sin_out)

    def backward_from_saved(dy, saved):
        cos, sin = saved
        dk = dy[:, :1]
        dq, _dk, _cos, _sin = rope_backward(dy, dk, cos, sin)
        return (dq,)

    return forward_with_saved, backward_from_saved
