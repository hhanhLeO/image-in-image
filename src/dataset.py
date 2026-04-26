"""
dataset.py — Dataset and DataLoader utilities for Deep Steganography.

Each sample returns a (cover, secret) pair of DIFFERENT images.
Supported input: any folder of .jpg files (ImageNet layout or flat).
"""

import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

from config import Config


# ── Supported image extensions ─────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def find_images(root: str) -> list:
    """Recursively find all image files under *root*."""
    root = Path(root)
    images = [str(p) for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    images.sort()
    return images


def get_transform(split: str = "train", size: int = 256) -> T.Compose:
    """
    Return a torchvision transform pipeline.

    Train : Resize → RandomHorizontalFlip → ToTensor  ([0, 1])
    Val   : Resize → ToTensor                          ([0, 1])

    Note: No ImageNet mean/std normalisation — the networks operate on
    raw [0, 1] pixel values.
    """
    if split == "train":
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor()])


class StegoDataset(Dataset):
    """
    Generic steganography dataset.

    Each ``__getitem__`` call returns a ``(cover, secret)`` tuple where
    cover and secret are always **different** images drawn from the same
    folder.  Compatible with ImageNet, COCO, or any flat image directory.

    Args:
        root      : Path to the image directory (searched recursively).
        split     : ``"train"`` or ``"val"`` — controls augmentation.
        size      : Spatial resolution to resize images to.
        transform : Optional custom transform; overrides ``split``/``size``.
        max_size  : Truncate the dataset to this many images (``None`` = all).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        size: int = 256,
        transform=None,
        max_size: Optional[int] = None,
    ):
        self.images = find_images(root)
        if not self.images:
            raise FileNotFoundError(f"No images found under '{root}'.")
        if max_size is not None:
            self.images = self.images[:max_size]
        self.transform = transform or get_transform(split, size)
        self.n = len(self.images)
        print(f"[StegoDataset] {split}: {self.n} images from '{root}'")

    def __len__(self) -> int:
        return self.n

    def _load(self, path: str) -> torch.Tensor:
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cover = self._load(self.images[idx])
        # Always pick a DIFFERENT image as the secret
        secret_idx = idx
        while secret_idx == idx:
            secret_idx = random.randint(0, self.n - 1)
        secret = self._load(self.images[secret_idx])
        return cover, secret


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """
    Construct train and validation ``DataLoader`` objects from *cfg*.

    Returns:
        ``(train_loader, val_loader)``
    """
    pin_memory = torch.cuda.is_available()

    train_ds = StegoDataset(
        cfg.train.train_root,
        split="train",
        size=cfg.model.image_size,
        max_size=cfg.train.train_max_size,
    )
    val_ds = StegoDataset(
        cfg.train.val_root,
        split="val",
        size=cfg.model.image_size,
        max_size=cfg.train.val_max_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
