"""MegaFold's fused LayerNorm -> Linear Triton kernels as an evograd pair.

``megafold/model/FusedLayernormLinear/fused_layernorm_linear.py`` is the kernel
the ``layernorm_linear`` site of the MegaFold workload runs in production.
Its forward takes ``(x, linear_weight[K, N], linear_bias, gamma, beta)`` and
returns the output with the row statistics; its backward returns
``(dx, dlinear_weight, dlinear_bias, dgamma, dbeta)``. The declaration's
gradient order is ``(dx, dlinear_weight, dweight, dbias)`` with no linear
bias, so the glue drops the ``None`` and reorders nothing else. ``eps`` is
fixed at 1e-5 inside MegaFold's kernel, which is the declaration's default;
the adapter refuses any other value rather than silently ignoring it.
"""


def make_megafold_layernorm_linear_autograd_pair_fns():
    from megafold.model.FusedLayernormLinear.fused_layernorm_linear import (
        layernorm_linear_backward,
        layernorm_linear_forward,
    )

    def forward_with_saved(x, weight, bias, linear_weight, eps=1e-5):
        if abs(float(eps) - 1e-5) > 1e-12:
            raise ValueError(f"MegaFold's layernorm_linear kernel fixes eps=1e-5, got {eps}")
        out, x_2d, b, mean, rstd = layernorm_linear_forward(
            x.contiguous(), linear_weight.contiguous(), None, weight, bias,
            has_layernorm_bias=True, has_linear_bias=False,
        )
        return out, (x_2d, b, mean, rstd, weight, bias)

    def backward_from_saved(dout, saved):
        x_2d, b, mean, rstd, weight, bias = saved
        dx, dlinear_weight, _dlinear_bias, dweight, dbias = layernorm_linear_backward(
            dout.contiguous(), x_2d, b, mean, rstd, weight, bias,
            has_layernorm_bias=True, has_linear_bias=False,
        )
        return dx, dlinear_weight, dweight, dbias

    return forward_with_saved, backward_from_saved
