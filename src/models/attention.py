"""
models/attention.py — Convolutional Block Attention Module (CBAM).

CBAM applies two sequential attention gates:
  1. Channel Attention — answers "WHAT features matter?"
  2. Spatial Attention — answers "WHERE does the signal live?"

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel attention gate.

    Pools the feature map spatially (average-pool + max-pool), passes both
    through a shared two-layer MLP, sums the outputs, and applies a sigmoid
    to produce per-channel weights in [0, 1].

    Args:
        in_ch     : Number of input channels.
        reduction : Bottleneck reduction ratio for the MLP (default 16).
    """

    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, in_ch // reduction)
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_ch, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))        # (B, C)
        max_pool = x.amax(dim=(2, 3))        # (B, C)
        attn = self.sigmoid(
            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        )                                    # (B, C)
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial attention gate.

    Aggregates feature channels (average + max across the channel dim),
    concatenates them, applies a single convolution, and uses sigmoid to
    produce a per-pixel weight map in [0, 1].

    Args:
        kernel_size : Size of the conv kernel (7 recommended by the paper).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_out = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        scale = self.sigmoid(
            self.conv(torch.cat([avg_out, max_out], dim=1))
        )                                        # (B, 1, H, W)
        return x * scale


class CBAM(nn.Module):
    """
    Full CBAM block: ChannelAttention → SpatialAttention.

    Drop this module anywhere in a CNN to let the network learn
    *what* and *where* to focus on.

    Args:
        in_ch          : Number of input channels.
        reduction      : Channel-attention MLP reduction ratio.
        spatial_kernel : Kernel size for the spatial-attention conv.
    """

    def __init__(self, in_ch: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_ch, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
