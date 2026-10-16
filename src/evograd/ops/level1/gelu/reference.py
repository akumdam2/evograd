import torch
import torch.nn.functional as F


def forward(x):
    return F.gelu(x)
