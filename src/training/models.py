"""Model definitions: Logistic Regression, MLP, CNN, WRN, and ViT.

All models accept (batch, C, H, W) inputs and output (batch, num_classes) logits.
They are designed to be compatible with Opacus (no in-place ops, BatchNorm → GroupNorm/LayerNorm).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Multinomial logistic regression (flatten → linear)."""

    def __init__(self, input_shape: tuple[int, ...], num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(math.prod(input_shape), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.flatten(x))


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations.

    Opacus-compatible (no BatchNorm).
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128]
        in_dim = math.prod(input_shape)

        layers: list[nn.Module] = [nn.Flatten()]
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """Small CNN for CIFAR-10/MNIST.

    Uses GroupNorm instead of BatchNorm for Opacus compatibility.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        channels: list[int] | None = None,
    ):
        super().__init__()
        channels = channels or [32, 64]
        c_in = input_shape[0]
        h, w = input_shape[1], input_shape[2]

        conv_layers: list[nn.Module] = []
        for c_out in channels:
            conv_layers.extend(
                [
                    nn.Conv2d(c_in, c_out, 3, padding=1),
                    nn.GroupNorm(min(8, c_out), c_out),  # Opacus-safe normalization
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            c_in = c_out
            h, w = h // 2, w // 2

        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# WideResNet (WRN-16-4) — Opacus-compatible (GroupNorm instead of BatchNorm)
# Reference: Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
# Used by Steinke, Nasr & Jagielski (2023) for one-run auditing on CIFAR-10.
# ---------------------------------------------------------------------------


class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with weight standardization (De et al. 2022).

    Standardizes weights per output channel (over fan-in dimensions)
    before the convolution. Does not change privacy guarantees.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        # Standardize over (in_channels, kH, kW) per output channel
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        fan_in = weight.shape[1] * weight.shape[2] * weight.shape[3]
        scale = torch.rsqrt(torch.clamp(var * fan_in, min=1e-4))
        weight = (weight - mean) * scale
        return nn.functional.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WRNBlock(nn.Module):
    """Single WideResNet basic block: two 3×3 convs with GroupNorm + ReLU."""

    def __init__(
        self, in_ch: int, out_ch: int, stride: int = 1, weight_standardization: bool = False
    ):
        super().__init__()
        Conv = WeightStandardizedConv2d if weight_standardization else nn.Conv2d
        self.gn1 = nn.GroupNorm(min(16, in_ch), in_ch)
        self.relu1 = nn.ReLU()
        self.conv1 = Conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(16, out_ch), out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv(out_ch, out_ch, 3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.gn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.gn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    """WideResNet (WRN-depth-widen) for CIFAR-10/100.

    Default is WRN-16-4 (~2.75M params), matching the paper's setup.
    Uses GroupNorm (16 groups, per De et al. 2022) instead of BatchNorm
    for Opacus compatibility. Optionally uses weight standardization.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        depth: int = 16,
        widen_factor: int = 4,
        weight_standardization: bool = False,
    ):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6k+4"
        n_blocks = (depth - 4) // 6  # blocks per group
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        self._ws = weight_standardization

        Conv = WeightStandardizedConv2d if weight_standardization else nn.Conv2d
        self.conv1 = Conv(input_shape[0], channels[0], 3, padding=1, bias=False)

        self.group1 = self._make_group(
            channels[0], channels[1], n_blocks, stride=1, ws=weight_standardization
        )
        self.group2 = self._make_group(
            channels[1], channels[2], n_blocks, stride=2, ws=weight_standardization
        )
        self.group3 = self._make_group(
            channels[2], channels[3], n_blocks, stride=2, ws=weight_standardization
        )

        self.gn_final = nn.GroupNorm(min(16, channels[3]), channels[3])
        self.relu_final = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_group(
        in_ch: int, out_ch: int, n_blocks: int, stride: int, ws: bool = False
    ) -> nn.Sequential:
        layers = [_WRNBlock(in_ch, out_ch, stride, weight_standardization=ws)]
        for _ in range(1, n_blocks):
            layers.append(_WRNBlock(out_ch, out_ch, stride=1, weight_standardization=ws))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.relu_final(self.gn_final(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ---------------------------------------------------------------------------
# Vision Transformer (ViT) — Opacus-compatible (LayerNorm, no in-place ops)
# Uses Conv2d patch embedding + plain-Linear attention for full Opacus safety.
# Default: ViT-Tiny (embed_dim=192, 3 heads, 6 layers, patch 4×4) ≈ 1.3M params.
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention using plain Linear layers (Opacus-safe)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with LayerNorm (Opacus-compatible)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for small images (CIFAR-10, MNIST).

    Uses Conv2d patch embedding and LayerNorm — fully Opacus-compatible.
    Default: ViT-Tiny (embed_dim=192, 3 heads, 6 layers) ≈ 1.3M params on CIFAR-10.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        patch_size: int = 4,
        embed_dim: int = 192,
        num_heads: int = 3,
        num_layers: int = 6,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        C, H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0
        num_patches = (H // patch_size) * (W // patch_size)

        self.patch_embed = nn.Conv2d(
            C,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(
            *[_TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, embed_dim, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x[:, 0])  # CLS token only
        return self.head(x)


def build_model(cfg: dict[str, Any], input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """Instantiate a model from config.

    Parameters
    ----------
    cfg : dict
        Full experiment config (uses cfg["model"]).
    input_shape : tuple
        (C, H, W) of the dataset.
    num_classes : int
        Number of output classes.

    Returns
    -------
    nn.Module
    """
    arch = cfg["model"]["arch"]
    if arch == "logreg":
        return LogisticRegression(input_shape, num_classes)
    elif arch == "mlp":
        hidden = cfg["model"].get("mlp_hidden", [128])
        return MLP(input_shape, num_classes, hidden_dims=hidden)
    elif arch == "cnn":
        channels = cfg["model"].get("cnn_channels", [32, 64])
        return SmallCNN(input_shape, num_classes, channels=channels)
    elif arch == "wrn":
        depth = cfg["model"].get("wrn_depth", 16)
        widen = cfg["model"].get("wrn_widen", 4)
        ws = cfg["model"].get("weight_standardization", False)
        return WideResNet(
            input_shape, num_classes, depth=depth, widen_factor=widen, weight_standardization=ws
        )
    elif arch == "vit":
        return VisionTransformer(
            input_shape,
            num_classes,
            patch_size=cfg["model"].get("vit_patch_size", 4),
            embed_dim=cfg["model"].get("vit_embed_dim", 192),
            num_heads=cfg["model"].get("vit_num_heads", 3),
            num_layers=cfg["model"].get("vit_num_layers", 6),
            mlp_ratio=cfg["model"].get("vit_mlp_ratio", 2.0),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
