#!/usr/bin/env python3
"""
Entry point for AI-Based Restoration of Degraded Images for Semiconductor
Inspection.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir>, restores it (denoise + super-resolve
2x), and writes one .npy file per input to <output-dir> with the same
filename. Runs fully offline on CPU or GPU, no internet access or
manual configuration required.
"""

import os
import sys
import glob

import numpy as np
import torch

from models.ffc_restoration import FFCRestorationNet

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "models", "weights.pth")
SCALE = 2          # fixed 2x upsampling (128->256 or 256->512)
BASE_CH = 64
N_BLOCKS = 8


def load_model(device):
    model = FFCRestorationNet(in_ch=1, base_ch=BASE_CH, n_blocks=N_BLOCKS, scale=SCALE)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_array(path):
    """Load a .npy file and return a 2D float32 array (H, W)."""
    arr = np.load(path).astype(np.float32)

    if arr.ndim == 3:
        # (H, W, 1) -> (H, W)
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Unexpected channel dim in {path}: shape {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected array shape in {path}: {arr.shape}")

    return arr


def restore_array(model, arr, device, use_tta=True):
    """Run one (H, W) numpy array through the model, return (H_out, W_out) numpy array."""
    # NOTE: input may legitimately exceed [0, 1] due to speckle overshoot.
    # We do NOT clip the input -- the model is trained to handle this.
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)

    with torch.no_grad():
        if not use_tta:
            out = model(tensor)
        else:
            # 8-fold test-time augmentation (4 rotations x 2 flips)
            preds = []
            for k in range(4):
                t_rot = torch.rot90(tensor, k, dims=[2, 3])
                out_rot = model(t_rot)
                out_unrot = torch.rot90(out_rot, -k, dims=[2, 3])
                preds.append(out_unrot)

                t_flip = torch.flip(t_rot, dims=[3])
                out_flip = model(t_flip)
                out_unflip = torch.flip(out_flip, dims=[3])
                out_unrot_flip = torch.rot90(out_unflip, -k, dims=[2, 3])
                preds.append(out_unrot_flip)

            out = torch.stack(preds, dim=0).mean(dim=0)

    out = out.squeeze(0).squeeze(0).cpu().numpy()

    # Safety net: clamp to [0, 1] and sanitize any NaN/Inf, per spec.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory does not exist: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_files:
        print(f"Warning: no .npy files found in {input_dir}")
        return

    print(f"Found {len(input_files)} input file(s).")

    for i, in_path in enumerate(input_files, 1):
        filename = os.path.basename(in_path)
        out_path = os.path.join(output_dir, filename)

        try:
            arr = load_array(in_path)
            restored = restore_array(model, arr, device)
            np.save(out_path, restored)
            print(f"[{i}/{len(input_files)}] {filename}: "
                  f"{arr.shape} -> {restored.shape}  saved -> {out_path}")
        except Exception as e:
            print(f"[{i}/{len(input_files)}] FAILED on {filename}: {e}")
            raise

    print("Done.")


if __name__ == "__main__":
    main()
