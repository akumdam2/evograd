"""Explicit autograd-pair baseline backed by PyTorch's cuBLAS matmul path."""

import torch

from evograd.ops._common import make_pair_baseline


def _cublas_pair_factory():
    def forward(a, b):
        output = torch.mm(a, b)
        return output, (a, b)

    def backward(dc, saved):
        a, b = saved
        return torch.mm(dc, b.T), torch.mm(a.T, dc)

    return forward, backward


measure_cublas_pair_baseline = make_pair_baseline(
    _cublas_pair_factory,
    ("a", "b"),
)
