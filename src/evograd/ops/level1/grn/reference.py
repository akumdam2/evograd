import torch


def forward(x, weight, bias):
    g = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    return weight * (x * (g / (g.mean(dim=-1, keepdim=True) + 1e-6))) + bias + x
