"""
config.py — Centralised hyperparameters for Deep Steganography.

Adjust any field here before training; nothing else needs to change.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Network architecture hyperparameters."""
    prep_out_ch:  int = 16    # Output channels of PrepNetwork
    unet_base_ch: int = 32    # Base channel width of both U-Nets
    image_size:   int = 256   # Spatial resolution (height = width)


@dataclass
class LossConfig:
    """
    Weights for the combined training loss:

        L = alpha * MSE_cover
          + beta  * MSE_secret
          + beta_ssim * SSIM_secret
          + gamma * Perceptual_cover   (VGG-16)
          + delta * Perceptual_secret  (VGG-16)
    """
    alpha:     float = 1.0   # Cover MSE weight
    beta:      float = 1.0   # Secret MSE weight
    beta_ssim: float = 0.5   # Secret SSIM loss weight
    gamma:     float = 0.1   # Cover perceptual loss weight
    delta:     float = 0.05  # Secret perceptual loss weight


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

@dataclass
class TrainConfig:
    """Training loop, optimizer, and I/O settings."""

    # ── Dataset paths ──────────────────────────────────────────
    # train_root:     str = "../data/imagenet/test"
    # val_root:       str = "../data/imagenet/val"
    train_root:    Path = ROOT_DIR / "data/imagenet/test"
    val_root:      Path = ROOT_DIR / "data/imagenet/val"
    train_max_size: int = 200   # Cap on training images (None = use all)
    val_max_size:   int = 20    # Cap on validation images

    # ── Training ───────────────────────────────────────────────
    epochs:      int = 10
    batch_size:  int = 8
    num_workers: int = 4

    # ── Optimizer ──────────────────────────────────────────────
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    lr_step_size: int   = 10    # Reduce LR every N epochs
    lr_gamma:     float = 0.5

    # ── Checkpointing ──────────────────────────────────────────
    # checkpoint_dir: str          = "../checkpoints"
    checkpoint_dir:          Path = ROOT_DIR / "checkpoints"
    save_every:               int = 3          # Save periodic checkpoint every N epochs
    resume_from:    Optional[str] = None      # Path to .pth to resume from

    # ── Logging & Visualisation ────────────────────────────────
    log_every:     int = 50   # Print batch loss every N steps
    val_every:     int = 1
    vis_every:     int = 3    # Save visualisation grid every N epochs
    # vis_dir:       str = "../outputs/vis"
    vis_dir:      Path = ROOT_DIR / "output/vis"
    n_vis_samples: int = 4


@dataclass
class Config:
    """Top-level config that bundles all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    loss:  LossConfig  = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ── Default singleton (import and use directly) ────────────────
cfg = Config()
