"""
metrics.py — Evaluation metrics for Deep Steganography.

Metrics computed:
  • PSNR  — Peak Signal-to-Noise Ratio          (higher is better)
  • SSIM  — Structural Similarity Index          (higher is better, max 1)
  • LPIPS — Learned Perceptual Image Patch Sim.  (lower is better)

All metrics are applied to both cover↔stego and secret↔revealed pairs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio. Higher = better. Formula: 10*log10(max²/MSE)"""
    mse = F.mse_loss(img1, img2, reduction="none").mean(dim=(1, 2, 3)).clamp(min=1e-10)
    return (10 * torch.log10(max_val ** 2 / mse)).mean()


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """Structural Similarity Index. Range [0,1]. Higher = better."""
    B, C, H, W = img1.shape
    kernel = _gaussian_kernel(window_size).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    def conv(x): return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1, mu2   = conv(img1), conv(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq  = conv(img1*img1) - mu1_sq
    sigma2_sq  = conv(img2*img2) - mu2_sq
    sigma12    = conv(img1*img2) - mu1_mu2
    ssim_map   = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean()


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity using VGG-16. Lower = better."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg.eval()
        for p in vgg.parameters(): p.requires_grad = False
        feats = vgg.features
        self.s1 = nn.Sequential(*list(feats.children())[:4])
        self.s2 = nn.Sequential(*list(feats.children())[4:9])
        self.s3 = nn.Sequential(*list(feats.children())[9:16])
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def _norm(self, x): return (x - self.mean) / self.std

    def forward(self, img1, img2):
        def nm(a, b):
            return F.mse_loss(F.normalize(a,dim=1), F.normalize(b,dim=1))
        x1, x2 = self._norm(img1), self._norm(img2)
        h1 = self.s1(x1); h2 = self.s1(x2)
        h3 = self.s2(h1); h4 = self.s2(h2)
        h5 = self.s3(h3); h6 = self.s3(h4)
        return nm(h1,h2) + nm(h3,h4) + nm(h5,h6)


class MetricsCalculator:
    """Computes PSNR, SSIM, LPIPS for both cover↔stego and secret↔revealed pairs."""
    def __init__(self, device="cpu"):
        self.device = device
        self.lpips_fn = LPIPS().to(device).eval()

    @torch.no_grad()
    def compute(self, cover, stego, secret, revealed) -> dict:
        return {
            "psnr_cover":   psnr(cover, stego).item(),
            "ssim_cover":   ssim(cover, stego).item(),
            "lpips_cover":  self.lpips_fn(cover.to(self.device), stego.to(self.device)).item(),
            "psnr_secret":  psnr(secret, revealed).item(),
            "ssim_secret":  ssim(secret, revealed).item(),
            "lpips_secret": self.lpips_fn(secret.to(self.device), revealed.to(self.device)).item(),
        }

    def format(self, m: dict) -> str:
        return (f"[Cover↔Stego]     PSNR={m['psnr_cover']:.2f}dB  "
                f"SSIM={m['ssim_cover']:.4f}  LPIPS={m['lpips_cover']:.4f}"
                f"[Secret↔Revealed] PSNR={m['psnr_secret']:.2f}dB  "
                f"SSIM={m['ssim_secret']:.4f}  LPIPS={m['lpips_secret']:.4f}")