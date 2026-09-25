"""MegaFold's fused EvoformerAttention Triton kernels as an evograd pair.

``megafold/model/FusedEvoAttention/evoattention.py`` exposes the kernels only
through an ``autograd.Function``. The pair protocol needs the forward's saved
state and a backward that consumes it without the engine, so the adapter calls
the Function's static ``forward`` and ``backward`` with a minimal context
object that records what ``save_for_backward`` is given. The tensors flow
exactly as they would under ``apply``; only the engine is absent.

Layouts match the declaration with no transposes in the glue: ``q, k, v`` are
``[B, S, N, H, D]``, ``res_mask`` is ``[B, S, 1, 1, N]`` additive float32 and
``pair_bias`` is ``[B, 1, H, N, N]`` float32. MegaFold returns ``d_pair_bias``
in the output gradient's dtype; the declaration asks for ``pair_bias``'s dtype,
so the adapter casts it back. That cast is the one operation outside the
kernels.
"""


class _Ctx:
    """The subset of ``autograd.function.FunctionCtx`` the kernels use."""

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tuple(tensors)


def make_megafold_evoattention_autograd_pair_fns():
    from megafold.model.FusedEvoAttention.evoattention import EvoformerAttention

    def forward_with_saved(q, k, v, res_mask, pair_bias):
        ctx = _Ctx()
        out = EvoformerAttention.forward(ctx, q, k, v, res_mask, pair_bias)
        return out, (ctx,)

    def backward_from_saved(do, saved):
        (ctx,) = saved
        dq, dk, dv, _d_res_mask, d_pair_bias = EvoformerAttention.backward(ctx, do)
        pair_bias = ctx.saved_tensors[4]
        return dq, dk, dv, d_pair_bias.to(pair_bias.dtype)

    return forward_with_saved, backward_from_saved
