"""
train.py — Training and validation loop for Deep Steganography.

Usage (Colab / terminal):
    python train.py

Edit config.py to change hyperparameters, dataset paths, and output dirs.
"""

import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from config  import Config, cfg
from dataset import build_dataloaders
from models  import StegaNet
from loss    import SteganographyLoss
from metrics import MetricsCalculator


# ── Device setup ───────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Checkpoint helpers ─────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer  and "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
    if scheduler  and "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
    return state.get("epoch", 0), state.get("metrics", {})


# ── Visualisation helpers ──────────────────────────────────────

def show_comparison(cover, stego, secret, revealed, n=4, title=""):
    """Display a side-by-side grid: Cover | Stego | Secret | Revealed."""
    n = min(n, cover.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes[None]
    for j, label in enumerate(["Cover (C)", "Stego (C')", "Secret (S)", "Revealed (S')"]):
        axes[0, j].set_title(label, fontsize=12, fontweight="bold")
    for i in range(n):
        for j, imgs in enumerate([cover, stego, secret, revealed]):
            axes[i, j].imshow(imgs[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j].axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def save_vis_grid(cover, stego, secret, revealed, path, n=4):
    """Save a 4-column comparison grid image to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imgs = torch.cat([cover[:n], stego[:n], secret[:n], revealed[:n]], dim=0)
    grid = vutils.make_grid(imgs, nrow=n, normalize=False, padding=2)
    vutils.save_image(grid, path)


# ── Per-epoch functions ────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch, cfg):
    model.train()
    keys   = ["total", "cover_mse", "secret_mse", "secret_ssim", "percep_cover", "percep_secret"]
    totals = {k: 0.0 for k in keys}
    n      = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
    for batch_idx, (cover, secret) in enumerate(pbar):
        cover, secret = cover.to(device), secret.to(device)
        optimizer.zero_grad()

        stego, revealed = model(cover, secret)
        losses = criterion(cover, stego, secret, revealed)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n += 1

        if (batch_idx + 1) % cfg.train.log_every == 0:
            pbar.set_postfix({"loss": f"{totals['total'] / n:.4f}"})

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, metrics_calc, cfg):
    model.eval()
    keys    = ["total", "cover_mse", "secret_mse", "secret_ssim", "percep_cover", "percep_secret"]
    met_keys = ["psnr_cover", "ssim_cover", "lpips_cover", "psnr_secret", "ssim_secret", "lpips_secret"]
    totals  = {k: 0.0 for k in keys}
    met_sum = {k: 0.0 for k in met_keys}
    n       = 0
    vis_batch = None

    for cover, secret in tqdm(loader, desc="Validating", leave=False):
        cover, secret = cover.to(device), secret.to(device)
        stego, revealed = model(cover, secret)

        losses  = criterion(cover, stego, secret, revealed)
        metrics = metrics_calc.compute(cover, stego, secret, revealed)

        for k in totals:  totals[k]  += losses[k].item()
        for k in met_sum: met_sum[k] += metrics[k]
        n += 1

        if vis_batch is None:
            vis_batch = (cover.cpu(), stego.cpu(), secret.cpu(), revealed.cpu())

    return (
        {k: v / n for k, v in totals.items()},
        {k: v / n for k, v in met_sum.items()},
        vis_batch,
    )


# ── Plot training history ──────────────────────────────────────

def plot_history(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["psnr_cover"],  label="PSNR Cover (C↔C')",   marker="o", markersize=3)
    axes[1].plot(epochs, history["psnr_secret"], label="PSNR Secret (S↔S')",  marker="o", markersize=3)
    axes[1].axhline(y=33, color="green",  linestyle="--", alpha=0.5, label="Target cover ≥33 dB")
    axes[1].axhline(y=28, color="orange", linestyle="--", alpha=0.5, label="Target secret ≥28 dB")
    axes[1].set_title("PSNR (higher = better)"); axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["ssim_cover"],  label="SSIM Cover (C↔C')",  marker="o", markersize=3)
    axes[2].plot(epochs, history["ssim_secret"], label="SSIM Secret (S↔S')", marker="o", markersize=3)
    axes[2].axhline(y=0.95, color="green",  linestyle="--", alpha=0.5, label="Target cover ≥0.95")
    axes[2].axhline(y=0.90, color="orange", linestyle="--", alpha=0.5, label="Target secret ≥0.90")
    axes[2].set_title("SSIM (higher = better)"); axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ── Main training entry point ──────────────────────────────────

def run_training(cfg: Config):
    print("=" * 60)
    print("  Deep Steganography — Training")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {cfg.train.epochs}")
    print(f"  Batch size : {cfg.train.batch_size}")
    print(f"  Train imgs : {cfg.train.train_max_size}")
    print(f"  Val imgs   : {cfg.train.val_max_size}")
    print("=" * 60)

    # Build model
    model = StegaNet(
        prep_out_ch=cfg.model.prep_out_ch,
        unet_base_ch=cfg.model.unet_base_ch,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Loss, optimiser, scheduler
    criterion = SteganographyLoss(
        alpha=cfg.loss.alpha,
        beta_mse=cfg.loss.beta,
        beta_ssim=cfg.loss.beta_ssim,
        gamma=cfg.loss.gamma,
        delta=cfg.loss.delta,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=cfg.train.lr_step_size,
        gamma=cfg.train.lr_gamma,
    )
    metrics_calc = MetricsCalculator(device=str(device))

    # Data
    train_loader, val_loader = build_dataloaders(cfg)

    # Resume if requested
    start_epoch = 0
    if cfg.train.resume_from and os.path.exists(cfg.train.resume_from):
        start_epoch, _ = load_checkpoint(
            cfg.train.resume_from, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}")
        start_epoch += 1

    history = {
        "train_loss": [], "val_loss": [],
        "psnr_cover": [], "psnr_secret": [],
        "ssim_cover": [], "ssim_secret": [],
    }
    best_psnr_secret = 0.0

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, epoch, cfg)
        scheduler.step()

        val_losses, val_metrics, vis_batch = validate(
            model, val_loader, criterion, metrics_calc, cfg
        )

        # Record history
        history["train_loss"].append(train_losses["total"])
        history["val_loss"].append(val_losses["total"])
        history["psnr_cover"].append(val_metrics["psnr_cover"])
        history["psnr_secret"].append(val_metrics["psnr_secret"])
        history["ssim_cover"].append(val_metrics["ssim_cover"])
        history["ssim_secret"].append(val_metrics["ssim_secret"])

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch + 1}/{cfg.train.epochs}]  ({elapsed:.1f}s)  "
            f"LR={scheduler.get_last_lr()[0]:.1e}"
        )
        print(
            f"  Train : total={train_losses['total']:.4f}  "
            f"cov_mse={train_losses['cover_mse']:.4f}  "
            f"sec_mse={train_losses['secret_mse']:.4f}  "
            f"sec_ssim={train_losses['secret_ssim']:.4f}  "
            f"p_cov={train_losses['percep_cover']:.4f}  "
            f"p_sec={train_losses['percep_secret']:.4f}"
        )
        print(f"  Val   : total={val_losses['total']:.4f}")
        print("  " + metrics_calc.format(val_metrics))

        # Save best model
        if val_metrics["psnr_secret"] > best_psnr_secret:
            best_psnr_secret = val_metrics["psnr_secret"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(cfg.train.checkpoint_dir, "best_model.pth"),
            )

        # Periodic checkpoint
        if (epoch + 1) % cfg.train.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, {},
                os.path.join(cfg.train.checkpoint_dir, f"epoch_{epoch + 1:03d}.pth"),
            )

        # Visualisation
        if (epoch + 1) % cfg.train.vis_every == 0 and vis_batch:
            show_comparison(
                *vis_batch, n=cfg.train.n_vis_samples,
                title=f"Epoch {epoch + 1} — Cover | Stego | Secret | Revealed",
            )
            save_vis_grid(
                *vis_batch,
                path=os.path.join(cfg.train.vis_dir, f"epoch_{epoch + 1:03d}.png"),
                n=cfg.train.n_vis_samples,
            )

    print(f"\n✓ Training complete.  Best PSNR (secret): {best_psnr_secret:.2f} dB")
    return model, history


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    model, history = run_training(cfg)
    plot_history(history)
