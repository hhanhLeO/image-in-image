# Deep Learning Steganography — Image-in-Image

**Final Project | Data Hiding and Secret Sharing**

> Hide an entire secret image inside a cover image using deep neural networks, then perfectly reconstruct it — with no visible trace in the stego image.

---

## Overview

Traditional steganography hides data by manually flipping bits. This project takes a completely different approach: an **end-to-end trainable encoder–decoder** that _learns_ where and how to embed a full secret image inside a cover image of the same size.

The system outputs:

- A **stego image C′** that looks visually identical to the original cover.
- A **revealed secret image S′** accurately reconstructed from the stego.

Unlike traditional methods, this approach achieves **full-image embedding capacity** and learns optimal hiding patterns automatically, without any handcrafted rules.

---

## Architecture

```
Secret S  ──► PrepNetwork ──► PrepOut (16 ch)
                                    │
Cover  C  ──────────────────► Concat (19 ch)
                                    │
                                    ▼
                          HidingNetwork (U-Net + CBAM)
                                    │
                     ┌──────── Stego C′ ───────────────► output
                     │
                     ▼
              RevealNetwork (U-Net + CBAM)
                     │
              Revealed S′ ───────────────────────────► output
```

### Sub-networks

| Module            | Input                   | Output              | Key design                        |
| ----------------- | ----------------------- | ------------------- | --------------------------------- |
| **PrepNetwork**   | Secret S (3 ch)         | Feature map (16 ch) | Lightweight CNN, no downsampling  |
| **HidingNetwork** | Cover ‖ PrepOut (19 ch) | Stego C′ (3 ch)     | U-Net + CBAM at bottleneck & dec3 |
| **RevealNetwork** | Stego C′ (3 ch)         | Revealed S′ (3 ch)  | Symmetric U-Net + CBAM            |

### U-Net Encoder–Decoder

Both HidingNetwork and RevealNetwork use a 4-level U-Net:

```
Encoder:  256×256 (b) → 128×128 (2b) → 64×64 (4b) → 32×32 (8b) → 16×16 (16b)
Decoder:  16×16 → 32×32 → 64×64 → 128×128 → 256×256 → output (3 ch)
```

Skip connections transfer high-frequency structural detail from encoder to decoder, ensuring the stego image accurately preserves the cover's appearance.

### CBAM — Convolutional Block Attention Module

CBAM is placed at two depths inside both U-Nets:

- **Bottleneck (16×16)** — global embedding decisions with full image context visible.
- **dec3 (64×64)** — mid-level texture, the optimal scale for hiding information without triggering visible artefacts.

Each CBAM block applies two sequential attention gates:

- **Channel Attention** — answers _what_ features matter, by weighting each channel map.
- **Spatial Attention** — answers _where_ to focus, by weighting each spatial position.

---

## Loss Function

```
L_total = α      · MSE(cover,  stego)
        + β      · MSE(secret, revealed)
        + β_ssim · (1 − SSIM(secret, revealed))
        + γ      · Perceptual(cover,  stego)
        + δ      · Perceptual(secret, revealed)
```

| Term                | Purpose                                        | Default weight |
| ------------------- | ---------------------------------------------- | -------------- |
| `MSE_cover`         | Stego looks identical to cover at pixel level  | α = 1.0        |
| `MSE_secret`        | Revealed image matches secret at pixel level   | β = 1.0        |
| `SSIM_secret`       | Revealed image matches secret structurally     | β_ssim = 0.5   |
| `Perceptual_cover`  | Cover texture preserved in stego (VGG-16)      | γ = 0.1        |
| `Perceptual_secret` | Secret structure recovered faithfully (VGG-16) | δ = 0.05       |

The perceptual loss uses three **sequential VGG-16 stages** (relu1_2 → relu2_2 → relu3_3), capturing shallow edges, mid-level textures, and high-level structures respectively.

---

## Evaluation Metrics

| Metric    | Measures                           | Direction | Target                         |
| --------- | ---------------------------------- | --------- | ------------------------------ |
| **PSNR**  | Pixel-level reconstruction quality | ↑ higher  | Cover > 30 dB · Secret > 25 dB |
| **SSIM**  | Structural/perceptual similarity   | ↑ higher  | Cover > 0.95 · Secret > 0.90   |
| **LPIPS** | Deep-feature perceptual distance   | ↓ lower   | As low as possible             |

Metrics are computed for both pairs: **cover ↔ stego** (imperceptibility) and **secret ↔ revealed** (recovery accuracy).

---

## Project Structure

```
image-in-image/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   └── prepare_data.py       ← Split raw ImageNet into train/val/test
│
├── src/
│   ├── config.py             ← All hyperparameters in one place
│   ├── dataset.py            ← StegoDataset + DataLoader builder
│   ├── loss.py               ← SteganographyLoss + PerceptualLoss (VGG-16)
│   ├── metrics.py            ← PSNR, SSIM, LPIPS, MetricsCalculator
│   ├── train.py              ← Full training loop
│   ├── demo.py               ← CLI: hide + reveal + save outputs
│   ├── app.py                ← Gradio web demo
│   └── models/
│       ├── __init__.py
│       ├── attention.py      ← CBAM (ChannelAttention + SpatialAttention)
│       ├── prep_network.py   ← PrepNetwork
│       ├── hiding_network.py ← HidingNetwork (U-Net + CBAM)
│       ├── reveal_network.py ← RevealNetwork (U-Net + CBAM)
│       └── stega_net.py      ← StegaNet (full system)
│
├── notebooks/
│   └── deep_steganography.ipynb   ← Original Colab notebook
│
├── checkpoints/
│   └── best_model.pth   ← Best model checkpoint
|
├── samples/
│   ├── cover.jpg             ← Quick-start example images
│   └── secret.jpg
│
└── outputs/                  ← Generated files
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/hhanhLeO/image-in-image.git
cd image-in-image
pip install -r requirements.txt
```

### 2. Prepare the dataset

- Download Dataset from Kaggle: [ImageNet 256×256](https://www.kaggle.com/datasets/dimensi0n/imagenet-256)

```bash
pip install kaggle
kaggle datasets download -d dimensi0n/imagenet-256
```

- Unzip the downloaded file `imagenet-256.zip` and place the extracted folder in the project root directory, so that the path to the raw dataset is `imagenet-256`. After unzipping, you should have the following structure:

```
imagenet-256/
  abacus/
    000.jpg
    0001.jpg
    ...
  abaya/
    ...
  accordion/
    ...
  ...
```

- Run the preprocessing script to create train/val/test splits:

```bash
python scripts/prepare_data.py \
    --root       imagenet-256 \
    --output     data/imagenet \
    --train_size 30000 \
    --val_size    2000 \
    --test_size   2000
```

`--root` should be the path to the unzipped folder, the one that directly contains the class subfolders (`abacus`, `abaya`,...). This creates a flat, class-agnostic split under `data/imagenet/train`, `val`, `test`.

### 3. Configure paths

Edit `src/config.py` to point `train_root`, `val_root`, and `checkpoint_dir` at your local paths.

---

## Usage

### Train

```bash
python src/train.py
```

Checkpoints are saved to `checkpoint_dir` (configured in `config.py`). The best model by PSNR-secret is saved as `best_model.pth`.

### Pre-trained model

The trained model checkpoint (`best_model.pth`) is not included in this zip due to file size. Download it separately and place it in the `checkpoints/` folder.

**Download:** [best_model.pth](https://drive.google.com/drive/folders/1Jim9D9y8nSet25A3tXQbH23WCW8rVzta?usp=sharing)

Then place it at:

```
checkpoints/best_model.pth
```

### CLI demo

```bash
python src/demo.py \
    --cover      samples/cover.jpg \
    --secret     samples/secret.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output_dir outputs/demo
```

Saves `1_cover.png`, `2_stego.png`, `3_secret.png`, `4_revealed.png`, and `comparison_grid.png`.

### Web demo (Gradio)

```bash
python src/app.py --checkpoint checkpoints/best_model.pth

# Share a public URL
python src/app.py --checkpoint checkpoints/best_model.pth --share
```

Open `http://localhost:7860` in your browser. Upload a cover and a secret image, click **Hide & Reveal**, and inspect the stego output, revealed secret, difference maps, and PSNR / SSIM / LPIPS metrics.

### Colab notebook

Open `notebooks/deep_steganography.ipynb` and run all cells end-to-end for an interactive walkthrough of training and evaluation.

---

## Dataset

- **Primary:** ImageNet 256×256 — 30 000 train / 2 000 val / 2 000 test

**Preprocessing pipeline:**

1. Resize to 256×256
2. ToTensor — pixel values in [0, 1]
3. Random horizontal flip (train only)
4. Cover and secret are always drawn from _different_ indices in the dataset

---

## References

[1] S. Baluja, “Hiding Images in Plain Sight: Deep Steganography,” inAdvances in Neural Information Processing Systems I. Guyon andothers, editors, volume 30, Curran Associates, Inc., 2017. url: https://proceedings.neurips.cc/paper_files/paper/2017/file/838e8afb1ca34354ac209f53d90c3a43-Paper.pdf.  
[2] A. Kumar, P. Singla and A. Yadav, StegaVision: Enhancing Steganography with Attention Mechanism, 2024. arXiv: 2411.05838 [cs.CV]. url: https://arxiv.org/abs/2411.05838.  
[3] S. G. Dilara S¸ener, “Enhancing steganography in 256×256 colored images with U-Net: A study on PSNR and SSIM metrics with variable-sized hidden images” IIETA, 2024. url: https://doi.org/10.18280/rces.110202.  
[4] L. Zeng, N. Yang, X. Li, A. Chen, H. Jing and J. Zhang, “Advanced Image Steganography Using a U-Net-Based Architecture with Multi-Scale Fusion and Perceptual Loss,” Electronics, jourvol 12, number 18, 2023. url: https://www.mdpi.com/2079-9292/12/18/3808.
