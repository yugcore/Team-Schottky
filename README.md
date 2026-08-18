# AI-Based Restoration of Degraded Images for Semiconductor Inspection

Team: `Team-Schottky`

## Approach

A joint denoising + super-resolution model built on **Classical-Prior Hybrid ResFFC (Residual Fast Fourier Convolution)** architecture with homomorphic log-domain speckle processing:

1. **Classical Lee Speckle Prior**: An internal GPU-vectorized Lee filter computes local variance and adaptive SNR weighting to pre-filter multiplicative speckle noise while preserving raw edge transitions.
2. **Dual-Channel Input Stem**: Feeds both the raw log-domain input and the clean Lee prior through the encoder.
3. **Prior-Anchored Residual Learning**: The bicubic baseline is anchored on the clean Lee prior, allowing the network to focus purely on high-frequency super-resolution detail.
4. **ResFFC Blocks**: Combines 3x3 local spatial convolutions with 2D FFT global frequency-domain convolutions, backed by full local and global residual connections.
5. **8-Fold Test-Time Augmentation (TTA)**: Evaluates multiple orthogonal orientations at inference for maximum fidelity and edge consistency.

Model size: ~425K parameters (`base_ch=64, n_blocks=8`), optimized for high reconstruction expressiveness and ultra-fast inference (<15ms per image on GPU).

See `models/ffc_restoration.py` for the full architecture.

## Directory structure

```
Team-Schottky/
├── run.py                    # entry point (with 8-fold TTA)
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
└── models/
    ├── ffc_restoration.py    # Classical Lee Prior + ResFFC model definition
    └── weights.pth           # trained model weights
```

## Setup

```bash
pip install -r requirements.txt
```

No internet access, API keys, or additional downloads are required at
runtime -- model weights are bundled in `models/weights.pth`.

## Usage

```bash
python run.py <input-dir> <output-dir>
```

- Reads every `.npy` file in `<input-dir>`.
- Creates `<output-dir>` if it does not already exist.
- Writes one restored `.npy` file per input file, using the same filename.
- Automatically uses GPU (`cuda`) if available, otherwise falls back to CPU.
- Automatically applies 8-fold test-time augmentation (TTA) for maximum restoration quality.

### Example

```bash
python run.py ./data/test_inputs ./data/restored_outputs
```

## Input / output contract

- **Input**: grayscale `.npy` arrays of shape `(H, W)` or `(H, W, 1)`.
  Input values may legitimately exceed `[0, 1]` due to speckle noise
  overshoot -- this is expected and handled internally by the model.
- **Output**: grayscale `.npy` arrays of shape `(H_out, W_out)`, where
  `H_out = 2 * H` and `W_out = 2 * W` (matching the dataset's 128->256 and
  256->512 restoration targets). Values are clipped to `[0, 1]` and
  sanitized against NaN/Inf as a final safety step.

## Notes on training

`models/weights.pth` contains the trained model weights used for
inference. Training was performed on the provided paired dataset
(excluding non-semantic synthetic white noise outliers) using a Compound Restoration Loss (L1 + Sobel gradient + Differentiable SSIM) with reflect-padding before all FFT operations to eliminate circular-convolution boundary artifacts.
