"""
models/prep_network.py — Preparation Network.

Transforms the secret image from 3 RGB channels into a richer N-channel
feature representation before embedding.  This intermediate representation
reduces direct spectral interference with the cover image when concatenated
in the Hiding Network.
"""

import torch
import torch.nn as nn


class PrepNetwork(nn.Module):
    """
    Lightweight CNN: secret image (3 ch) → high-dimensional feature map (N ch).

    All convolutions use padding=1 so the spatial dimensions are preserved
    (256×256 in, 256×256 out).

    Architecture:
        Conv(3→64)  → BN → LeakyReLU
        Conv(64→64) → BN → LeakyReLU
        Conv(64→N)  → BN → LeakyReLU

    Args:
        in_ch  : Input channels (3 for RGB).
        out_ch : Output channels N; concatenated with the cover image before
                 the Hiding Network.  Defaults to 16.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, secret: torch.Tensor) -> torch.Tensor:
        """
        Args:
            secret: (B, 3, H, W) secret image tensor in [0, 1].

        Returns:
            Feature map of shape (B, N, H, W).
        """
        return self.net(secret)
