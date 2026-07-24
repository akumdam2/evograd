"""Fairness-reviewed Liger PolyNorm adapter.

Liger exposes the row-statistics forward used here. Its raw backward only
supports scalar affine parameters, so the vector-affine benchmark computes the
same derivatives from those Liger-produced statistics.
"""

import torch

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass


def _vector_forward(x_2d, weight, bias, rstd):
    cols = x_2d.shape[-1]
    xf, wf = x_2d.float(), weight.float()
    x2, x3 = xf * xf, xf * xf * xf
    return (
        wf[0].reshape(1, cols) * (x3 * rstd[:, 0:1].float())
        + wf[1].reshape(1, cols) * (x2 * rstd[:, 1:2].float())
        + wf[2].reshape(1, cols) * (xf * rstd[:, 2:3].float())
        + bias.float().reshape(1, cols)
    ).to(x_2d.dtype)


def _vector_backward(dout, x, weight, eps):
    rows, cols = x.shape
    inv_cols = 1.0 / float(cols)
    xf, gf, wf = x.float(), dout.float(), weight.float()
    x2, x3, x5 = xf.pow(2), xf.pow(3), xf.pow(5)
    r1 = torch.rsqrt(x2.mean(dim=1, keepdim=True) + eps)
    r2 = torch.rsqrt(xf.pow(4).mean(dim=1, keepdim=True) + eps)
    r3 = torch.rsqrt(xf.pow(6).mean(dim=1, keepdim=True) + eps)
    gw0, gw1, gw2 = (
        gf * wf[0].reshape(1, cols),
        gf * wf[1].reshape(1, cols),
        gf * wf[2].reshape(1, cols),
    )
    dweight = torch.empty((3, cols), dtype=torch.float32, device=x.device)
    dweight[0] = (gf * (x3 * r3)).sum(dim=0)
    dweight[1] = (gf * (x2 * r2)).sum(dim=0)
    dweight[2] = (gf * (xf * r1)).sum(dim=0)
    dbias = gf.sum(dim=0)
    s3 = (gw0 * x3).sum(dim=1, keepdim=True)
    s2 = (gw1 * x2).sum(dim=1, keepdim=True)
    s1 = (gw2 * xf).sum(dim=1, keepdim=True)
    dx = (
        3.0 * gw0 * x2 * r3 - (3.0 * inv_cols) * r3.pow(3) * x5 * s3
        + 2.0 * gw1 * xf * r2 - (2.0 * inv_cols) * r2.pow(3) * x3 * s2
        + gw2 * r1 - inv_cols * r1.pow(3) * xf * s1
    )
    return dx, dweight, dbias


def make_liger_poly_norm_autograd_pair_fns():
    from liger_kernel.ops.poly_norm import poly_norm_forward

    def forward_with_saved(x, weight, bias, eps):
        with torch.no_grad():
            x, weight, bias = x.contiguous(), weight.contiguous(), bias.contiguous()
            dummy_weight = torch.zeros((3,), dtype=torch.float32, device=x.device)
            dummy_bias = torch.zeros((), dtype=torch.float32, device=x.device)
            _unused, x_2d, rstd, *_ = poly_norm_forward(
                x, dummy_weight, dummy_bias, eps
            )
            output = _vector_forward(x_2d, weight, bias, rstd)
            return output.view_as(x), (x_2d, weight, bias, rstd)

    def backward_from_saved(dout, saved, eps):
        with torch.no_grad():
            x_2d, weight, bias, _rstd = saved
            dx, dweight, dbias = _vector_backward(
                dout.contiguous().view_as(x_2d), x_2d, weight, eps
            )
            return (
                dx.to(x_2d.dtype).view_as(dout),
                dweight.to(weight.dtype).reshape_as(weight),
                dbias.to(bias.dtype).reshape_as(bias),
            )

    return forward_with_saved, backward_from_saved
