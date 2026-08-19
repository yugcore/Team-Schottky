# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**Team:** `Team-Schottky`  
**GitHub Repository:** [github.com/yugcore/Team-Schottky](https://github.com/yugcore/Team-Schottky.git)  
**Submission Artifacts:** [`DOCUMENTATION.md`](DOCUMENTATION.md) (Slide Presentation Brief) | [`USER.md`](USER.md) (Plain-English Guide & Q&A)

---

## ⚡ Quick Start: Verify the Model for Yourself

You can easily evaluate the trained model, calculate quantitative metrics (PSNR & SSIM), and generate a side-by-side visual comparison grid in just a couple of commands.

### 1. Installation & Environment Setup

```bash
# Clone the repository (if not already local)
git clone https://github.com/yugcore/Team-Schottky.git
cd Team-Schottky

# Install required dependencies
pip install -r requirements.txt
pip install matplotlib scikit-image
```

> [!NOTE]
> **100% Self-Contained & Offline:** The model weights (`models/weights.pth`, 1.87 MB) are bundled directly in the repository. No internet connection, cloud API keys, or external model downloads are needed at runtime.

---

### 2. Self-Check & Visual Evaluation (`evaluate.py`)

Run the following command to evaluate on the validation set, compare against the naive bicubic baseline, and generate a **20-sample visual comparison grid**:

```bash
python evaluate.py --gt_dir "datasets/train/train/GT" --noisy_dir "datasets/train/train/NoisyLR" --weights models/weights.pth --n_samples 20
```

#### What this command does:
1. **Quantitative Benchmarking**: Evaluates the model on the validation split (`--val_split 0.1`, `--seed 42`) and computes:
   - **Model PSNR (Peak Signal-to-Noise Ratio)** in dB
   - **Model SSIM (Structural Similarity Index)**
   - **Baseline PSNR** (naive bicubic interpolation of the noisy degraded input)
2. **Visual Inspection Grid (`eval_comparison.png`)**: Exports a high-resolution 3-column image grid showing the first 20 samples:
   - **Column 1:** Degraded Input (Low-Resolution with Multiplicative Speckle Noise)
   - **Column 2:** Restored Model Output (Denoised + 2x Super-Resolved, with individual PSNR/SSIM scores)
   - **Column 3:** Ground Truth Reference

#### Evaluation CLI Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--gt_dir` | *(Required)* | Path to the directory containing ground-truth `.npy` files |
| `--noisy_dir` | *(Required)* | Path to the directory containing degraded `.npy` files |
| `--weights` | `models/weights.pth` | Path to trained PyTorch weights checkpoint |
| `--n_samples` | `8` | Number of sample image triplets to display in the visual comparison grid (e.g. `20`) |
| `--save_fig` | `eval_comparison.png` | Filename / path where the visual comparison PNG is saved |
| `--val_split` | `0.1` | Fraction of dataset reserved for validation (10%) |
| `--seed` | `42` | Random seed for deterministic train/val partition |
| `--no_tta` | `False` | Pass this flag to disable 8-fold Test-Time Augmentation for faster evaluation |

---

### 3. Run Inference on Any Folder (`run.py`)

To run the restoration model on any arbitrary folder of `.npy` degraded images (e.g., test set or unseen scans):

```bash
python run.py <input-dir> <output-dir>
```

#### Example:
```bash
python run.py ./datasets/test_inputs ./outputs/restored_outputs
```

- Reads all `.npy` grayscale files from `<input-dir>`.
- Automatically uses GPU (`cuda`) if available, or seamlessly falls back to CPU.
- Applies **8-fold Test-Time Augmentation (TTA)** for optimal edge consistency and noise suppression.
- Outputs clean 2x super-resolved `.npy` files to `<output-dir>` with matching filenames.

---

### 4. Verify Output Validity & Contract (`check_outputs.py`)

To automatically check that all generated output arrays satisfy all competition constraints (shape doubling, finite numbers, no NaN/Inf, strict `[0.0, 1.0]` range):

```bash
python check_outputs.py
```

---

## 🔬 Approach & Technical Architecture

Our solution is a single-stage, physics-informed **Classical-Prior Hybrid ResFFC (Residual Fast Fourier Convolution)** network designed specifically for semiconductor inspection metrology.

```
Degraded Input (H, W)
       │
       ▼
[Homomorphic Log-Transform: log(1 + max(0, I))]
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
[Raw Log Input]                             [Classical Lee Filter Prior]
       │                                    (GPU-Vectorized Local Stats)
       └──────────────────────┬───────────────────────┘
                              ▼
               [Dual-Channel Input Stem (2 -> 64)]
                              │
               [FiLM Noise-Variance Conditioning]
                              │
             [8x SE-ResFFC Spectral-Spatial Blocks]
              (Local 3x3 Conv + Global 2D FFT Conv)
                              │
               [Long Global Skip Connection]
                              │
               [PixelShuffle 2x Upsampler]
                              │
                              ▼
            Residual + Bicubic(Lee Prior Baseline)
                              │
                              ▼
              [Inverse Homomorphic: exp(y) - 1]
                              │
                              ▼
                 [Clamped [0, 1] Output (2H, 2W)]
```

### Core Architectural Pillars:
1. **Homomorphic Log-Domain Processing**: Maps multiplicative speckle noise ($I = R \cdot \eta$) into an additive domain ($\log(1 + I) \approx \log(1 + R) + \log(\eta)$), linearizing the restoration problem.
2. **GPU-Vectorized Classical Lee Filter Prior**: Calculates local adaptive Signal-to-Noise Ratio (SNR) statistics directly on GPU to pre-filter speckle while preserving raw step edges with zero boundary blurring.
3. **Dual-Channel Input Stem & Prior-Anchored Baseline**: Feeds both the raw noisy log array and the clean Lee prior into the network. The upscaled Lee prior acts as the spatial anchor, allowing the deep network to focus exclusively on reconstructing high-frequency super-resolution details.
4. **Fast Fourier Convolutions (ResFFC)**: Combines local $3\times 3$ spatial convolutions with global 2D FFT spectral convolutions, achieving full-image receptive field with $\mathcal{O}(N \log N)$ computational efficiency.
5. **Zero-Hallucination Guarantee (Anti-GAN)**: Trained with a deterministic **Compound Restoration Loss** (Charbonnier + Sobel Gradient + Differentiable SSIM). Generative adversarial and diffusion losses are avoided to guarantee zero hallucinated artifacts or false defects.
6. **8-Fold Test-Time Augmentation (TTA)**: Averages predictions across 4 orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) and horizontal flips to leverage Manhattan circuit geometry and gain $+0.30\text{ dB}$ PSNR.

---

## 📊 Key Specifications & Performance

| Specification | Value | Context / Benchmark |
| :--- | :--- | :--- |
| **Model Parameters** | **446,465** (~446K) | Lightweight, compact, avoids over-parameterization |
| **Model Checkpoint Size** | **1.87 MB** (`models/weights.pth`) | Fully bundled, zero external asset downloads |
| **Inference Latency (GPU)** | **< 15 ms / image** | Real-time capable for high-speed wafer fab inspection |
| **VRAM Footprint** | **< 250 MB** | Can run on edge industrial microscopes and standard laptops |
| **Input Shape & Format** | `(H, W)` or `(H, W, 1)` `.npy` | Grayscale arrays; handles out-of-bounds speckle overshoot |
| **Output Shape & Format** | `(2H, 2W)` `.npy` | Grayscale arrays; guaranteed finite and strictly clamped in `[0.0, 1.0]` |
| **Execution Environment** | PyTorch 2.x, CPU/GPU | Fully automated offline execution |

---

## 📁 Repository Structure

```
Team-Schottky/
├── run.py                    # Official CLI entry point (restores a folder of .npy images with 8-fold TTA)
├── evaluate.py               # Evaluation engine (calculates PSNR/SSIM & exports visual comparison grid)
├── check_outputs.py          # Validation script (verifies output shapes, [0, 1] range, and NaN/Inf safety)
├── train.py                  # Training pipeline with compound loss and synthetic outlier filtering
├── requirements.txt          # Python dependencies
├── README.md                 # Setup, quick-start, and execution guide
├── DOCUMENTATION.md          # Complete hackathon slide deck presentation briefing
├── USER.md                   # Intuitive plain-English guide & interview/defense Q&A cheat sheet
└── models/
    ├── ffc_restoration.py    # Classical Lee Prior + ResFFC PyTorch model architecture definition
    └── weights.pth           # Bundled trained model checkpoint (1.87 MB)
```

---

## 📑 Input / Output Contract

- **Input Contract:**
  - Reads grayscale `.npy` arrays with shape `(H, W)` or `(H, W, 1)`.
  - Input pixel values may legitimately exceed the standard `[0, 1]` range due to multiplicative speckle noise overshoot. These values are ingested safely without pre-truncation.
- **Output Contract:**
  - Generates 2D grayscale `.npy` arrays with exact $2\times$ spatial super-resolution `(2H, 2W)`.
  - Output pixel values are strictly bounded within `[0.0, 1.0]`.
  - All output arrays are sanitized against non-finite values (`NaN`, `+Inf`, `-Inf`).

---

## 📚 Further Reading & References

- 📄 **Slide Briefing:** Check [`DOCUMENTATION.md`](DOCUMENTATION.md) for the complete slide-by-slide Hackathon technical documentation.
- 💡 **Concepts & Q&A:** Check [`USER.md`](USER.md) for the intuitive explanation of every pipeline step, mathematical derivations, and presentation Q&A answers.
