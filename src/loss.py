"""
loss.py — Combined steganography loss function.

    L_total = alpha  * MSE(cover, stego)
            + beta   * MSE(secret, revealed)
            + beta_ssim * (1 - SSIM(secret, revealed))
            + gamma  * Perceptual(cover, stego)       [VGG-16]
            + delta  * Perceptual(secret, revealed)   [VGG-16]

The perceptual loss uses VGG-16 features from layers relu1_2, relu2_2,
and relu3_3 so that structural and textural similarity is captured beyond
raw pixel error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    VGG-16 perceptual loss — reusable for both cover and secret pairs.
    Compares high-level feature representations (relu1_2, relu2_2, relu3_3).
    Ref: Zeng et al. (2023)
    """
    VGG_LAYERS = {"relu1_1": 1,"relu1_2": 4, "relu2_2": 9, "relu3_3": 16}

    def __init__(self, layer_weights: dict = None):
        super().__init__()
        if layer_weights is None:
            layer_weights = {k: 1.0 / len(self.VGG_LAYERS) for k in self.VGG_LAYERS}
        self.layer_weights = layer_weights

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.extractors = nn.ModuleDict()
        for name, stop_idx in self.VGG_LAYERS.items():
            self.extractors[name] = nn.Sequential(*list(vgg.features.children())[:stop_idx + 1])

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between any two image pairs (img_a, img_b)."""
        a_n, b_n = self._normalize(img_a), self._normalize(img_b)
        loss = torch.tensor(0.0, device=img_a.device)
        for name, extractor in self.extractors.items():
            w = self.layer_weights.get(name, 1.0)
            loss = loss + w * F.mse_loss(extractor(b_n), extractor(a_n))
        return loss


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor,
              window_size: int = 11, C1: float = 1e-4, C2: float = 9e-4) -> torch.Tensor:
    """
    SSIM-based structural loss: L_ssim = 1 - SSIM(img1, img2).
    Range [0, 2], practically ~[0, 1]. Lower = better structural similarity.
    Penalizes blurring, contrast loss, and structural distortion that MSE misses.
    This directly optimises the SSIM metric used in evaluation.
    """
    B, C, H, W = img1.shape
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2

    def conv(x): return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1, mu2   = conv(img1), conv(img2)
    sigma1_sq  = conv(img1 * img1) - mu1 ** 2
    sigma2_sq  = conv(img2 * img2) - mu2 ** 2
    sigma12    = conv(img1 * img2) - mu1 * mu2
    ssim_map   = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                 ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


class SteganographyLoss(nn.Module):
    """
    Improved combined loss:
      L = alpha * MSE_cover + beta * MSE_secret  
        + beta_ssim * SSIM_secret
        + gamma * Percep_cover + delta * Percep_secret

    Args:
        alpha     : weight for cover MSE (imperceptibility)
        beta_mse  : weight for secret pixel MSE (recovery accuracy)
        beta_ssim : weight for secret SSIM loss (structural quality)
        gamma     : weight for cover perceptual loss (VGG cover)
        delta     : weight for secret perceptual loss (VGG secret)
    """
    def __init__(self,
                 alpha:     float = 1.0,
                 beta_mse:  float = 1.0,
                 beta_ssim: float = 0.5,
                 gamma:     float = 0.1,
                 delta:     float = 0.05):
        super().__init__()
        self.alpha, self.beta_mse, self.beta_ssim = alpha, beta_mse, beta_ssim
        self.gamma, self.delta = gamma, delta
        self.perceptual = PerceptualLoss()
        self.mse = nn.MSELoss()

    def forward(self, cover, stego, secret, revealed) -> dict:
        cover_mse     = self.mse(stego, cover)
        secret_mse    = self.mse(revealed, secret)
        secret_ssim   = ssim_loss(secret, revealed)       
        percep_cover  = self.perceptual(cover, stego)
        percep_secret = self.perceptual(secret, revealed)  

        total = (self.alpha     * cover_mse
               + self.beta_mse  * secret_mse
               + self.beta_ssim * secret_ssim
               + self.gamma     * percep_cover
               + self.delta     * percep_secret)

        return {
            "total":         total,
            "cover_mse":     cover_mse,
            "secret_mse":    secret_mse,
            "secret_ssim":   secret_ssim,
            "percep_cover":  percep_cover,
            "percep_secret": percep_secret,
        }