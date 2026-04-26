"""
models/stega_net.py — StegaNet: the full encoder–decoder system.

Combines PrepNetwork, HidingNetwork, and RevealNetwork into a single
end-to-end trainable module.

    Secret S  ──► PrepNetwork ──► PrepOut (N ch)
                                       │
    Cover  C  ──────────────────► Concatenate (N+3 ch)
                                       │
                                       ▼
                               HidingNetwork (U-Net + CBAM)
                                       │
                               Stego C' ────────────────────► output
                                       │
                                       ▼
                               RevealNetwork (U-Net + CBAM)
                                       │
                               Revealed S' ─────────────────► output
"""

import torch
import torch.nn as nn

from models.prep_network   import PrepNetwork
from models.hiding_network import HidingNetwork
from models.reveal_network import RevealNetwork
from typing import Tuple


class StegaNet(nn.Module):
    """
    Full deep steganography model.

    Args:
        prep_out_ch  : Output channels of the PrepNetwork (default 16).
        unet_base_ch : Base channel width for both U-Nets (default 32).
    """

    def __init__(self, prep_out_ch: int = 16, unet_base_ch: int = 32):
        super().__init__()
        self.prep   = PrepNetwork(in_ch=3, out_ch=prep_out_ch)
        self.hiding = HidingNetwork(in_ch=prep_out_ch + 3, base_ch=unet_base_ch)
        self.reveal = RevealNetwork(base_ch=unet_base_ch)

    def forward(
        self,
        cover: torch.Tensor,
        secret: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cover  : (B, 3, H, W) cover image in [0, 1].
            secret : (B, 3, H, W) secret image in [0, 1].

        Returns:
            stego    : (B, 3, H, W) stego image in [0, 1].
            revealed : (B, 3, H, W) reconstructed secret image in [0, 1].
        """
        prep_out = self.prep(secret)
        stego    = self.hiding(cover, prep_out)
        revealed = self.reveal(stego)
        return stego, revealed

    # ── Convenience helpers ─────────────────────────────────────

    def hide(self, cover: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        """Embed *secret* into *cover*; return stego image only."""
        with torch.no_grad():
            stego, _ = self.forward(cover, secret)
        return stego

    def reveal_secret(self, stego: torch.Tensor) -> torch.Tensor:
        """Extract hidden secret from *stego*; return revealed image only."""
        with torch.no_grad():
            return self.reveal(stego)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
