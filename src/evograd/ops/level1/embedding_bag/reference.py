"""PyTorch sum-mode EmbeddingBag oracle."""

import torch.nn.functional as F


def forward(weight, indices, offsets):
    return F.embedding_bag(indices, weight, offsets, mode="sum")
