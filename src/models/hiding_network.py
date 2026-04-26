"""
models/hiding_network.py — Hiding Network (U-Net + CBAM).

Takes the concatenated [cover ‖ prep_features] tensor and produces a
stego image that is perceptually indistinguishable from the cover image
while secretly containing the embedded secret.

CBAM placement rationale
─────────────────────────
• Bottleneck (16×16): global embedding decisions; full image context visible.
• dec3     (64×64) : mid-level texture; best scale for hiding information
                     without triggering visible artefacts.
"""

import torch
import torch.nn as nn

from models.attention import CBAM


class DoubleConv(nn.Module):
    """Standard U-Net building block: Conv→BN→LeakyReLU repeated twice."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HidingNetwork(nn.Module):
    """
    U-Net encoder–decoder with CBAM attention for steganographic embedding.

    Input shape  : (B, prep_out_ch + 3, 256, 256)
    Output shape : (B, 3, 256, 256)  — stego image in [0, 1]

    Encoder depth:
        enc1  256×256   b ch
        enc2  128×128   2b ch
        enc3   64×64    4b ch
        enc4   32×32    8b ch
        btk    16×16   16b ch  ← CBAM

    Decoder depth (with skip connections):
        dec4   32×32    8b ch
        dec3   64×64    4b ch  ← CBAM
        dec2  128×128   2b ch
        dec1  256×256    b ch
        out   256×256    3 ch  (Sigmoid)

    Args:
        in_ch   : Total input channels = prep_out_ch + 3.
        base_ch : Base channel width *b* (all others are multiples of this).
    """

    def __init__(self, in_ch: int = 19, base_ch: int = 32):
        super().__init__()
        b = base_ch

        # ── Encoder ────────────────────────────────────────────
        self.enc1 = DoubleConv(in_ch, b)        # 256×256
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
        self.cbam_dec3 = CBAM(b * 4)              # CBAM at 64×64

        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)

        self.up1  = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)

        # ── Output ─────────────────────────────────────────────
        self.out_conv = nn.Conv2d(b, 3, kernel_size=1)
        self.out_act  = nn.Sigmoid()

    def forward(
        self,
        cover: torch.Tensor,
        prep_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cover         : (B, 3, H, W) cover image in [0, 1].
            prep_features : (B, N, H, W) prep-network output.

        Returns:
            Stego image of shape (B, 3, H, W) in [0, 1].
        """
        x = torch.cat([cover, prep_features], dim=1)   # (B, N+3, H, W)

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck with global attention
        bn = self.cbam_bottleneck(self.bottleneck(self.pool(e4)))

        # Decode with skip connections
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.cbam_dec3(self.dec3(torch.cat([self.up3(d4), e3], dim=1)))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_act(self.out_conv(d1))
