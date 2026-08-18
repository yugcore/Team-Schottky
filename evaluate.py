"""
Evaluate a trained checkpoint on the validation split (or any GT/NoisyLR
folder pair) using PSNR and SSIM, and save a visual comparison grid.

Usage:
    python evaluate.py \
        --gt_dir "Y:/Team-Schottky/datasets/train/train/GT" \
        --noisy_dir "Y:/Team-Schottky/datasets/train/train/NoisyLR" \
        --weights models/weights.pth \
        --n_samples 8 \
        --val_split 0.1 --seed 42
"""

import argparse
import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    def psnr(image_true, image_test, data_range=1.0):
        mse = np.mean((image_true.astype(np.float64) - image_test.astype(np.float64)) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10((data_range ** 2) / mse))

    def ssim(image_true, image_test, data_range=1.0):
        # Basic SSIM fallback
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        x = image_true.astype(np.float64)
        y = image_test.astype(np.float64)
        mu_x, mu_y = np.mean(x), np.mean(y)
        sigma_x2 = np.var(x)
        sigma_y2 = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
        return float(num / den)

from models.ffc_restoration import FFCRestorationNet
from train import find_pairs, _is_junk  # reuse the same junk-filtering / pairing logic


def load_array(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return arr


def restore(model, noisy, device, use_tta=True):
    from run import restore_array
    return restore_array(model, noisy, device, use_tta=use_tta)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (TTA={'disabled' if args.no_tta else 'enabled'})")

    model = FFCRestorationNet(in_ch=1, base_ch=args.base_ch, n_blocks=args.n_blocks, scale=args.scale)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    all_files = find_pairs(args.gt_dir, args.noisy_dir)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_split))
    val_files = all_files[:n_val]
    print(f"Evaluating on {len(val_files)} validation pairs "
          f"(same split used during training, seed={args.seed}).")

    psnr_scores, ssim_scores = [], []
    baseline_psnr_scores = []  # naive bicubic upsample, for comparison

    for fname in val_files:
        gt = load_array(os.path.join(args.gt_dir, fname))
        noisy = load_array(os.path.join(args.noisy_dir, fname))

        restored = restore(model, noisy, device, use_tta=not args.no_tta)

        psnr_scores.append(psnr(gt, restored, data_range=1.0))
        ssim_scores.append(ssim(gt, restored, data_range=1.0))

        # naive baseline: bicubic upsample of the raw noisy input, no denoising/SR model
        import torch.nn.functional as F
        noisy_t = torch.from_numpy(np.clip(noisy, 0, 1)).unsqueeze(0).unsqueeze(0).float()
        bicubic = F.interpolate(noisy_t, scale_factor=args.scale, mode="bicubic", align_corners=False)
        bicubic = np.clip(bicubic.squeeze().numpy(), 0, 1)
        baseline_psnr_scores.append(psnr(gt, bicubic, data_range=1.0))

    print("\n--- Results over validation set ---")
    print(f"Model   PSNR: {np.mean(psnr_scores):.2f} dB  (std {np.std(psnr_scores):.2f})")
    print(f"Model   SSIM: {np.mean(ssim_scores):.4f}  (std {np.std(ssim_scores):.4f})")
    print(f"Naive bicubic-only PSNR: {np.mean(baseline_psnr_scores):.2f} dB")

    # --- save a visual comparison grid for a handful of samples ---
    n_show = min(args.n_samples, len(val_files))
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for i, fname in enumerate(val_files[:n_show]):
        gt = load_array(os.path.join(args.gt_dir, fname))
        noisy = load_array(os.path.join(args.noisy_dir, fname))
        restored = restore(model, noisy, device, use_tta=not args.no_tta)

        axes[i, 0].imshow(np.clip(noisy, 0, 1), cmap="gray")
        axes[i, 0].set_title(f"Input (degraded)\n{fname}", fontsize=8)
        axes[i, 1].imshow(restored, cmap="gray")
        axes[i, 1].set_title(f"Model output\nPSNR={psnr(gt, restored, data_range=1.0):.1f}dB "
                              f"SSIM={ssim(gt, restored, data_range=1.0):.3f}", fontsize=8)
        axes[i, 2].imshow(gt, cmap="gray")
        axes[i, 2].set_title("Ground truth", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = args.save_fig
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved visual comparison grid to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--weights", default="models/weights.pth")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=8, help="How many images to show in the visual grid")
    p.add_argument("--save_fig", default="eval_comparison.png")
    p.add_argument("--no_tta", action="store_true", help="Disable 8-fold test-time augmentation")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
