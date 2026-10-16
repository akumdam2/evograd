import torch
import torch.nn.functional as F


def forward(x):
    return x * torch.sigmoid(1.702 * x)
