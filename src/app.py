"""
src/app.py — Gradio web demo for Deep Steganography.

Usage:
    python src/app.py --checkpoint checkpoints/best_model.pth
    python src/app.py --checkpoint checkpoints/best_model.pth --share   # public URL
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import gradio as gr
import numpy as np
from PIL import Image

from config import cfg
from models import StegaNet
from metrics import MetricsCalculator


# ── Device & model ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: StegaNet = None
metrics_calc = MetricsCalculator(device=str(device))


def load_model(checkpoint_path: str) -> str:
    """Load StegaNet from checkpoint. Returns status message."""
    global model
    if not Path(checkpoint_path).exists():
        return f"❌ Checkpoint not found: {checkpoint_path}"
    try:
        state = torch.load(checkpoint_path, map_location=device)
        model = StegaNet(
            prep_out_ch=cfg.model.prep_out_ch,
            unet_base_ch=cfg.model.unet_base_ch,
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        ckpt_metrics = state.get("metrics", {})
        msg = f"✅ Model loaded from `{checkpoint_path}` (device: {device})"
        if ckpt_metrics:
            psnr_s = ckpt_metrics.get("psnr_secret", "—")
            ssim_s = ckpt_metrics.get("ssim_secret", "—")
            msg += f"\n📊 Checkpoint — PSNR secret: **{psnr_s:.2f} dB** · SSIM secret: **{ssim_s:.4f}**"
        return msg
    except Exception as exc:
        return f"❌ Failed to load model: {exc}"


def preprocess(img: Image.Image, size: int = 256) -> torch.Tensor:
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tf(img.convert("RGB")).unsqueeze(0).to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def difference_map(t1: torch.Tensor, t2: torch.Tensor, amplify: int = 10) -> Image.Image:
    """Amplified absolute difference between two image tensors."""
    diff = (t1 - t2).abs() * amplify
    return tensor_to_pil(diff.clamp(0, 1))


@torch.no_grad()
def run_hide_and_reveal(cover_img: Image.Image, secret_img: Image.Image):
    """Main inference function called by Gradio."""
    if model is None:
        raise gr.Error("No model loaded. Please load a checkpoint first.")
    if cover_img is None or secret_img is None:
        raise gr.Error("Please upload both a cover image and a secret image.")

    cover  = preprocess(cover_img)
    secret = preprocess(secret_img)

    stego, revealed = model(cover, secret)

    m = metrics_calc.compute(cover, stego, secret, revealed)

    metrics_md = f"""
| | PSNR (↑) | SSIM (↑) | LPIPS (↓) |
|---|---|---|---|
| **Cover ↔ Stego** | {m['psnr_cover']:.2f} dB | {m['ssim_cover']:.4f} | {m['lpips_cover']:.4f} |
| **Secret ↔ Revealed** | {m['psnr_secret']:.2f} dB | {m['ssim_secret']:.4f} | {m['lpips_secret']:.4f} |
"""

    stego_pil    = tensor_to_pil(stego)
    revealed_pil = tensor_to_pil(revealed)
    diff_cover   = difference_map(cover, stego)
    diff_secret  = difference_map(secret, revealed)

    return stego_pil, revealed_pil, diff_cover, diff_secret, metrics_md


# ── Custom CSS ─────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #111118;
    --card:     #16161f;
    --border:   #2a2a3a;
    --accent:   #7c6aff;
    --accent2:  #ff6a9e;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --success:  #4ade80;
    --radius:   12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Header */
.stego-header {
    text-align: center;
    padding: 48px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.stego-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.stego-header p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
    letter-spacing: 0.04em;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
}

/* Cards */
.gr-group, .gr-box {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 28px !important;
    color: #fff !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.gr-button-primary:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.gr-button-secondary {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Inputs */
.gr-textbox textarea, .gr-textbox input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Image panels */
.gr-image {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* Metrics table */
.metrics-box table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.metrics-box th {
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.metrics-box td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-family: 'DM Mono', monospace;
}

/* Status bar */
.status-ok  { color: var(--success) !important; }
.status-err { color: var(--accent2) !important; }

/* Accordion */
.gr-accordion { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
"""


# ── Build Gradio UI ────────────────────────────────────────────

def build_ui(default_checkpoint: str = ""):
    with gr.Blocks(css=CSS, title="Deep Steganography") as demo:

        # ── Header ─────────────────────────────────────────────
        gr.HTML("""
        <div class="stego-header">
            <h1>Deep Steganography</h1>
            <p>Image-in-Image · U-Net + CBAM · Hide a secret image inside a cover image</p>
        </div>
        """)

        # ── Model loader ───────────────────────────────────────
        with gr.Accordion("⚙️  Model checkpoint", open=not bool(default_checkpoint)):
            with gr.Row():
                ckpt_input = gr.Textbox(
                    value=default_checkpoint,
                    label="Checkpoint path (.pth)",
                    placeholder="checkpoints/best_model.pth",
                    scale=4,
                )
                load_btn = gr.Button("Load model", variant="secondary", scale=1)
            load_status = gr.Markdown(value="*No model loaded yet.*")

        load_btn.click(fn=load_model, inputs=ckpt_input, outputs=load_status)

        # Auto-load if path provided at startup
        if default_checkpoint:
            demo.load(fn=lambda: load_model(default_checkpoint), outputs=load_status)

        gr.HTML("<hr style='border-color:#2a2a3a; margin: 8px 0 24px'>")

        # ── Main panel ─────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left — inputs
            with gr.Column(scale=1):
                gr.HTML("<div class='section-label'>01 — Input images</div>")
                cover_input = gr.Image(
                    label="Cover image (C)",
                    type="pil",
                    height=256,
                )
                secret_input = gr.Image(
                    label="Secret image (S)",
                    type="pil",
                    height=256,
                )
                run_btn = gr.Button("▶  Hide & Reveal", variant="primary")
                clear_btn = gr.ClearButton(
                    components=[cover_input, secret_input],
                    value="✕  Clear",
                    variant="secondary",
                )

            # Right — outputs
            with gr.Column(scale=2):
                gr.HTML("<div class='section-label'>02 — Outputs</div>")
                with gr.Row():
                    stego_out = gr.Image(
                        label="Stego image (C′)  — looks like cover",
                        type="pil",
                        height=256,
                        interactive=False,
                    )
                    revealed_out = gr.Image(
                        label="Revealed secret (S′)",
                        type="pil",
                        height=256,
                        interactive=False,
                    )
                gr.HTML("<div class='section-label' style='margin-top:16px'>03 — Difference maps  (×10 amplified)</div>")
                with gr.Row():
                    diff_cover_out = gr.Image(
                        label="Cover vs Stego",
                        type="pil",
                        height=200,
                        interactive=False,
                    )
                    diff_secret_out = gr.Image(
                        label="Secret vs Revealed",
                        type="pil",
                        height=200,
                        interactive=False,
                    )

        # ── Metrics ────────────────────────────────────────────
        gr.HTML("<div class='section-label' style='margin-top:24px'>04 — Metrics</div>")
        with gr.Group(elem_classes="metrics-box"):
            metrics_out = gr.Markdown(value="*Run the model to see metrics.*")

        # ── Wire up ────────────────────────────────────────────
        run_btn.click(
            fn=run_hide_and_reveal,
            inputs=[cover_input, secret_input],
            outputs=[stego_out, revealed_out, diff_cover_out, diff_secret_out, metrics_out],
        )

        # ── Examples ───────────────────────────────────────────
        sample_dir = Path(__file__).parent.parent / "samples"
        sample_cover  = str(sample_dir / "cover.jpg")
        sample_secret = str(sample_dir / "secret.jpg")
        if Path(sample_cover).exists() and Path(sample_secret).exists():
            gr.HTML("<div class='section-label' style='margin-top:24px'>05 — Quick examples</div>")
            gr.Examples(
                examples=[[sample_cover, sample_secret]],
                inputs=[cover_input, secret_input],
                label="",
            )

        # ── Footer ─────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding: 32px 0 16px; color: #6b6b80; font-size: 0.78rem; letter-spacing: 0.06em;">
            DATA HIDING AND SECRET SHARING · FINAL PROJECT · GROUP 2 · LÊ THỊ HỒNG HẠNH — 22127103
        </div>
        """)

    return demo


# ── Entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Steganography — Gradio Demo")
    parser.add_argument("--checkpoint", default="", help="Path to best_model.pth")
    parser.add_argument("--share",  action="store_true", help="Create a public Gradio URL")
    parser.add_argument("--port",   type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(default_checkpoint=args.checkpoint)
    demo.launch(share=args.share, server_port=args.port)