import torch.nn.functional as F


def forward(x, weight, bias, stride, padding, groups):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)
