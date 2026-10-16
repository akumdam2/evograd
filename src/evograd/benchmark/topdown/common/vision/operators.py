"""Executable boundaries taken from the published PyTorch architectures."""

import torch
import torch.nn.functional as F


def grn(x, weight, bias):
    norm = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    return weight * (x * (norm / (norm.mean(dim=-1, keepdim=True) + 1e-6))) + bias + x


def convnext_mlp(x, up_weight, up_bias, grn_weight, grn_bias, down_weight, down_bias):
    return F.linear(
        grn(F.gelu(F.linear(x, up_weight, up_bias)), grn_weight, grn_bias), down_weight, down_bias
    )


def text_mlp(x, gate_weight, up_weight, down_weight):
    return F.linear(F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight), down_weight)


def vision_mlp(x, up_weight, up_bias, down_weight, down_bias):
    hidden = F.linear(x, up_weight, up_bias)
    return F.linear(hidden * torch.sigmoid(1.702 * hidden), down_weight, down_bias)


def spatial_conv(x, weight, bias, stride, padding, groups):
    return F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)


FUNCTIONS = {"convnext_mlp": convnext_mlp, "text_mlp": text_mlp, "vision_mlp": vision_mlp}


def bindings(model):
    """(site, owner path, owner module, first child, argument packer)."""
    for path, module in model.named_modules():
        kind = type(module).__name__
        if kind == "ConvNextV2Layer":

            def pack(x, m=module):
                return (
                    x,
                    m.pwconv1.weight,
                    m.pwconv1.bias,
                    m.grn.weight,
                    m.grn.bias,
                    m.pwconv2.weight,
                    m.pwconv2.bias,
                )

            yield "convnext_mlp", path, module, module.pwconv1, pack
        elif kind == "Qwen2MLP":

            def pack(x, m=module):
                return (x, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)

            yield "text_mlp", path, module, module, pack
        elif kind == "VisionMlp":

            def pack(x, m=module):
                return (x, m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias)

            yield "vision_mlp", path, module, module, pack
