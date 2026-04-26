"""
demo.py — Inference and demo script for Deep Steganography.

Loads a trained checkpoint, hides a secret image inside a cover image,
reveals the hidden secret, prints metrics, and saves all outputs.

Usage:
    python demo.py \\
        --cover   path/to/cover.jpg  \\
        --secret  path/to/secret.jpg \\
        --checkpoint path/to/best_model.pth \\
        --output_dir outputs/demo
"""

import os
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image

from config  import cfg
from models  import StegaNet
from metrics import MetricsCalculator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load an image from *path* as a (1, 3, H, W) tensor in [0, 1]."""
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def save_image_tensor(tensor: torch.Tensor, path: str):
    """Save a (1, 3, H, W) or (3, H, W) tensor as a PNG file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(tensor.squeeze(0).clamp(0, 1), path)
    print(f"  Saved → {path}")


def load_model(checkpoint_path: str) -> StegaNet:
    """Restore a StegaNet from a ``.pth`` checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device)
    model = StegaNet(
        prep_out_ch=cfg.model.prep_out_ch,
        unet_base_ch=cfg.model.unet_base_ch,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Model loaded from '{checkpoint_path}'")
    if state.get("metrics"):
        print(f"  Checkpoint metrics: {state['metrics']}")
    return model


def show_results(cover, stego, secret, revealed, title=""):
    """Display the four images side-by-side in a matplotlib figure."""
    imgs   = [cover, stego, secret, revealed]
    labels = ["Cover (C)", "Stego (C')", "Secret (S)", "Revealed (S')"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Main demo ──────────────────────────────────────────────────

@torch.no_grad()
def run_demo(
    cover_path:      str,
    secret_path:     str,
    checkpoint_path: str,
    output_dir:      str = "outputs/demo",
):
    """
    Full hide-and-reveal pipeline.

    1. Load cover and secret images.
    2. Embed secret into cover → stego image.
    3. Extract hidden secret from stego → revealed image.
    4. Print PSNR / SSIM / LPIPS metrics.
    5. Save all four images and a comparison grid.

    Returns:
        dict of computed metrics.
    """
    model  = load_model(checkpoint_path)
    cover  = load_image(cover_path,  cfg.model.image_size).to(device)
    secret = load_image(secret_path, cfg.model.image_size).to(device)
    print(f"Cover  : {cover_path}")
    print(f"Secret : {secret_path}")

    stego, revealed = model(cover, secret)

    # Metrics
    calc    = MetricsCalculator(device=str(device))
    metrics = calc.compute(cover, stego, secret, revealed)
    print("\n" + "─" * 60)
    print(calc.format(metrics))
    print("─" * 60)

    # Visualise
    show_results(cover, stego, secret, revealed,
                 title="Deep Steganography — Demo")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    save_image_tensor(cover,    f"{output_dir}/1_cover.png")
    save_image_tensor(stego,    f"{output_dir}/2_stego.png")
    save_image_tensor(secret,   f"{output_dir}/3_secret.png")
    save_image_tensor(revealed, f"{output_dir}/4_revealed.png")

    # Comparison grid (1 row × 4 columns)
    grid = torch.cat([cover, stego, secret, revealed], dim=0)
    save_image_tensor(
        vutils.make_grid(grid, nrow=4, padding=4, normalize=False),
        f"{output_dir}/comparison_grid.png",
    )
    print(f"\nAll outputs saved to '{output_dir}/'")
    return metrics


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Steganography Demo")
    parser.add_argument("--cover",      required=True, help="Path to cover image")
    parser.add_argument("--secret",     required=True, help="Path to secret image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output_dir", default="outputs/demo",
                        help="Directory to save output images")
    args = parser.parse_args()

    run_demo(
        cover_path=args.cover,
        secret_path=args.secret,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
