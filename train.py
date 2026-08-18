"""
Training script for the FFC restoration model.

Expected dataset layout (macOS zip junk like __MACOSX and files starting
with '._' are automatically ignored):

    <data_root>/
    +---Test_NoisyLR/NoisyLR/       <- test-time inputs (no GT, not used here)
    +---train/train/GT/             <- ground truth (clean, full-res) .npy files
    +---train/train/NoisyLR/        <- degraded (noisy, low-res) .npy files

GT and NoisyLR files are matched by identical filename.

Usage:
    python train.py --gt_dir "Y:/train/train/GT" --noisy_dir "Y:/train/train/NoisyLR" --out models/weights.pth --epochs 100
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.ffc_restoration import FFCRestorationNet


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def _is_junk(filename):
    """Filter out macOS zip artifacts (._foo.npy) and non-.npy files."""
    base = os.path.basename(filename)
    return base.startswith(".") or not base.lower().endswith(".npy")


def _is_synthetic_noise(filepath):
    """Detect synthetic pure-noise junk files (Uniform(0, 1) noise with std ~ 0.2887)."""
    try:
        arr = np.load(filepath)
        if abs(arr.mean() - 0.5) < 0.02 and abs(arr.std() - 0.2887) < 0.01:
            return True
    except Exception:
        pass
    return False


def find_pairs(gt_dir, noisy_dir, filter_noise=True):
    """Return the sorted list of filenames present (and .npy-valid) in both dirs."""
    gt_files = sorted(f for f in os.listdir(gt_dir) if not _is_junk(f))
    noisy_files = set(f for f in os.listdir(noisy_dir) if not _is_junk(f))

    pairs, missing, junk_noise = [], [], []
    for f in gt_files:
        if f in noisy_files:
            if filter_noise and _is_synthetic_noise(os.path.join(gt_dir, f)):
                junk_noise.append(f)
            else:
                pairs.append(f)
        else:
            missing.append(f)

    if junk_noise:
        print(f"Filtered out {len(junk_noise)} synthetic white-noise junk files from dataset.")
    if missing:
        print(f"WARNING: {len(missing)} GT file(s) have no matching NoisyLR "
              f"file and will be skipped. Example: {missing[:3]}")
    if not pairs:
        raise RuntimeError(
            "No matching GT/NoisyLR filename pairs found. "
            "Check that filenames are identical between the two folders."
        )
    return pairs


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, filenames, patch_size=128, scale=2, train=True, cache=True):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.patch_size = patch_size  # patch size measured on the NOISY (input) image
        self.scale = scale
        self.train = train
        self.pairs = filenames
        self.cache = cache
        self._gt_cache = {}
        self._noisy_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        return arr

    def __getitem__(self, idx):
        fname = self.pairs[idx]
        if self.cache and fname in self._gt_cache:
            gt = self._gt_cache[fname]
            noisy = self._noisy_cache[fname]
        else:
            gt = self._load(os.path.join(self.gt_dir, fname))
            noisy = self._load(os.path.join(self.noisy_dir, fname))

            # Sanity check: GT should be exactly `scale`x the noisy image.
            expected_h, expected_w = noisy.shape[0] * self.scale, noisy.shape[1] * self.scale
            if gt.shape[0] != expected_h or gt.shape[1] != expected_w:
                raise ValueError(
                    f"{fname}: GT shape {gt.shape} is not {self.scale}x NoisyLR "
                    f"shape {noisy.shape} (expected {(expected_h, expected_w)})."
                )
            if self.cache:
                self._gt_cache[fname] = gt
                self._noisy_cache[fname] = noisy

        if self.train:
            noisy, gt = self._random_crop(noisy, gt)
            noisy, gt = self._random_flip_rotate(noisy, gt)

        # (H, W) -> (1, H, W)
        noisy_t = torch.from_numpy(noisy.copy()).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt.copy()).unsqueeze(0).float()
        return noisy_t, gt_t

    def _random_crop(self, noisy, gt):
        ph = pw = self.patch_size
        h, w = noisy.shape
        if h < ph or w < pw:
            # pad if the image is smaller than the patch (edge case)
            pad_h, pad_w = max(0, ph - h), max(0, pw - w)
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
            gt = np.pad(gt, ((0, pad_h * self.scale), (0, pad_w * self.scale)), mode="reflect")
            h, w = noisy.shape

        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)

        noisy_crop = noisy[top:top + ph, left:left + pw]
        gt_crop = gt[top * self.scale:(top + ph) * self.scale,
                      left * self.scale:(left + pw) * self.scale]
        return noisy_crop, gt_crop

    def _random_flip_rotate(self, noisy, gt):
        if random.random() < 0.5:
            noisy, gt = np.fliplr(noisy), np.fliplr(gt)
        if random.random() < 0.5:
            noisy, gt = np.flipud(noisy), np.flipud(gt)
        k = random.randint(0, 3)
        if k:
            noisy, gt = np.rot90(noisy, k), np.rot90(gt, k)
        return noisy, gt


# ---------------------------------------------------------------------
# Compound Loss: L1 + Sobel gradient + Differentiable SSIM loss
# ---------------------------------------------------------------------
def create_gaussian_window(window_size=11, sigma=1.5, channel=1):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return g2d.repeat(channel, 1, 1, 1)


class CompoundRestorationLoss(nn.Module):
    """Charbonnier pixel loss + Sobel edge loss + Differentiable SSIM loss."""

    def __init__(self, edge_weight=0.5, ssim_weight=0.3, eps=1e-3, window_size=11):
        super().__init__()
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.eps = eps
        self.window_size = window_size

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("window", create_gaussian_window(window_size=window_size))

    def _charbonnier(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def _gradient_mag(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def _ssim_loss(self, pred, target):
        pad = self.window_size // 2
        mu1 = F.conv2d(pred, self.window, padding=pad)
        mu2 = F.conv2d(target, self.window, padding=pad)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=pad) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=pad) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=pad) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
        return 1.0 - torch.clamp(ssim_map.mean(), 0.0, 1.0)

    def forward(self, pred, target):
        charb = self._charbonnier(pred, target)
        edge_pred = self._gradient_mag(pred)
        edge_target = self._gradient_mag(target)
        edge_loss = self._charbonnier(edge_pred, edge_target)

        loss = charb + self.edge_weight * edge_loss
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self._ssim_loss(pred, target)
        return loss


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = find_pairs(args.gt_dir, args.noisy_dir, filter_noise=True)
    rng = random.Random(args.seed)
    rng.shuffle(all_files)

    n_val = max(1, int(len(all_files) * args.val_split))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"Total pairs: {len(all_files)}  ->  train: {len(train_files)}  val: {len(val_files)}")

    train_set = RestorationDataset(args.gt_dir, args.noisy_dir, train_files,
                                    patch_size=args.patch_size, scale=args.scale, train=True)
    val_set = RestorationDataset(args.gt_dir, args.noisy_dir, val_files,
                                  patch_size=args.patch_size, scale=args.scale, train=False)

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True, pin_memory=True)
    # batch_size=1 for validation: full (uncropped) images may vary in size,
    # so they can't be safely batched together.
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = FFCRestorationNet(in_ch=1, base_ch=args.base_ch,
                               n_blocks=args.n_blocks, scale=args.scale).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = CompoundRestorationLoss(edge_weight=args.edge_weight, ssim_weight=args.ssim_weight).to(device)
    print(f"Using CompoundRestorationLoss (edge_weight={args.edge_weight}, ssim_weight={args.ssim_weight})")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for noisy, gt in loader:
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(loader)

        # --- validation pass (no gradient, no augmentation) ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                val_loss += criterion(pred, gt).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out)
            print(f"  -> new best val_loss, saved checkpoint to {args.out}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No val improvement for {args.patience} epochs. Stopping early "
                      f"(best val_loss={best_val_loss:.5f}).", flush=True)
                break

    print("Training complete.", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dir", required=True, help="Path to train/train/GT")
    p.add_argument("--noisy_dir", required=True, help="Path to train/train/NoisyLR")
    p.add_argument("--out", default="models/weights.pth")
    p.add_argument("--patch_size", type=int, default=128,
                    help="Crop size measured on the NoisyLR (input) image")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--edge_weight", type=float, default=0.5,
                    help="Weight for Sobel edge loss term (default: 0.5)")
    p.add_argument("--ssim_weight", type=float, default=0.3,
                    help="Weight for differentiable SSIM loss term (default: 0.3)")
    p.add_argument("--val_split", type=float, default=0.1,
                    help="Fraction of pairs held out for validation (not trained on)")
    p.add_argument("--patience", type=int, default=15,
                    help="Stop if val_loss doesn't improve for this many epochs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
