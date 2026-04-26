"""
scripts/prepare_data.py — Split and copy ImageNet images into train/val/test.

The raw ImageNet folder contains ~300k images organised by class (abacus/,
abaya/, ...).  This script randomly samples a fixed subset, flattens the
class hierarchy, and writes flat numbered files into data/imagenet/.

Usage:
    python scripts/prepare_data.py --root imagenet-256 --output data/imagenet

Output layout:
    data/imagenet/
    ├── train/   (30 000 images  →  000001.jpg … 030000.jpg)
    ├── val/     ( 2 000 images)
    └── test/    ( 2 000 images)
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

# Match the extensions recognised by StegoDataset in src/dataset.py
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: str) -> List[Path]:
    """Recursively collect all image files under *root*."""
    images = [
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not images:
        raise ValueError(f"No images found in '{root}'.")
    return images


def split_dataset(
    root: str,
    train_size: int = 30_000,
    val_size:   int =  2_000,
    test_size:  int =  2_000,
    seed:       int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Randomly sample and split images into train / val / test subsets.

    Sorting before shuffling guarantees the same split across platforms
    regardless of filesystem ordering.

    Args:
        root       : Path to the raw ImageNet folder.
        train_size : Number of training images.
        val_size   : Number of validation images.
        test_size  : Number of test images.
        seed       : Random seed for reproducibility.

    Returns:
        (train, val, test) — three lists of Path objects.
    """
    all_images = sorted(find_images(root))   # sort first → reproducible shuffle

    total_needed = train_size + val_size + test_size
    if len(all_images) < total_needed:
        raise ValueError(
            f"Not enough images: need {total_needed}, found {len(all_images)}."
        )

    random.seed(seed)
    random.shuffle(all_images)

    train = all_images[:train_size]
    val   = all_images[train_size : train_size + val_size]
    test  = all_images[train_size + val_size : total_needed]

    return train, val, test


def copy_split(paths: List[Path], dest_dir: Path, split_name: str) -> None:
    """
    Copy *paths* into *dest_dir*, renaming files to zero-padded indices
    (000001.jpg, 000002.jpg, …) so the flat output is class-agnostic.

    Args:
        paths      : Source image paths.
        dest_dir   : Destination directory (created if absent).
        split_name : Human-readable label used in progress messages.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for i, src in enumerate(paths, start=1):
        dst = dest_dir / f"{i:06d}{src.suffix.lower()}"
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            print(f"  [skip] {src}: {exc}")
            skipped += 1

        if i % 5_000 == 0:
            print(f"  {split_name}: {i}/{len(paths)} copied …")

    ok = len(paths) - skipped
    print(f"  {split_name}: {ok} copied, {skipped} skipped → {dest_dir}")


def main(root: str, output: str, train_size: int, val_size: int, test_size: int, seed: int):
    print(f"Source : {root}")
    print(f"Output : {output}")
    print(f"Split  : {train_size} / {val_size} / {test_size}  (seed={seed})")
    print()

    train, val, test = split_dataset(root, train_size, val_size, test_size, seed)

    print(f"Copying {len(train):,} train images …")
    copy_split(train, Path(output) / "train", "train")

    print(f"Copying {len(val):,} val images …")
    copy_split(val,   Path(output) / "val",   "val")

    print(f"Copying {len(test):,} test images …")
    copy_split(test,  Path(output) / "test",  "test")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ImageNet split for Deep Steganography")
    parser.add_argument("--root",       default="imagenet-256",  help="Raw ImageNet root folder")
    parser.add_argument("--output",     default="data/imagenet", help="Output folder")
    parser.add_argument("--train_size", type=int, default=30_000)
    parser.add_argument("--val_size",   type=int, default=2_000)
    parser.add_argument("--test_size",  type=int, default=2_000)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    main(args.root, args.output, args.train_size, args.val_size, args.test_size, args.seed)