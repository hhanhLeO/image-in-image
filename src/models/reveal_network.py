"""
models/reveal_network.py — Reveal Network (U-Net + CBAM).

Reconstructs the hidden secret image from the stego image.
CBAM is placed at the same depth levels as in HidingNetwork (bottleneck
and dec3), so both networks learn complementary attention patterns over
the same spatial regions.
"""

import torch
import torch.nn as nn

from models.attention import CBAM
from models.hiding_network import DoubleConv   # reuse the same building block


class RevealNetwork(nn.Module):
    """
    U-Net encoder–decoder with CBAM attention for secret extraction.

    Input shape  : (B, 3, 256, 256)  — stego image
    Output shape : (B, 3, 256, 256)  — revealed secret image in [0, 1]

    The architecture mirrors HidingNetwork.  Skip connections preserve
    the high-frequency structural components of the hidden signal,
    enabling high-fidelity reconstruction.

    Args:
        base_ch : Base channel width *b*.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        b = base_ch

        # ── Encoder ────────────────────────────────────────────
        self.enc1 = DoubleConv(3,     b)        # 256×256
        self.enc2 = DoubleConv(b,     b * 2)    # 128×128
        self.enc3 = DoubleConv(b * 2, b * 4)   #  64×64
        self.enc4 = DoubleConv(b * 4, b * 8)   #  32×32
        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck + CBAM ───────────────────────────────────
        self.bottleneck      = DoubleConv(b * 8, b * 16)   # 16×16
        self.cbam_bottleneck = CBAM(b * 16)

        # ── Decoder ────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)

        self.up3       = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3      = DoubleConv(b * 8, b * 4)
        self.cbam_dec3 = CBAM(b * 4)              # mirror HidingNetwork's CBAM

        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1  = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        # ── Output ─────────────────────────────────────────────
        self.out_conv = nn.Conv2d(b, 3, kernel_size=1)
        self.out_act  = nn.Sigmoid()

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stego : (B, 3, H, W) stego image in [0, 1].

        Returns:
            Revealed secret image of shape (B, 3, H, W) in [0, 1].
        """
        # Encode
        e1 = self.enc1(stego)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck with attention
        bn = self.cbam_bottleneck(self.bottleneck(self.pool(e4)))

        # Decode with skip connections
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.cbam_dec3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_act(self.out_conv(d1))
