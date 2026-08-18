"""
FFC (Fast Fourier Convolution) based joint denoising + super-resolution
model for semiconductor inspection image restoration.

Signals & Systems intuition:
- Local branch: standard spatial conv -> fine texture / local structure.
- Global (spectral) branch: rFFT2 -> 1x1 conv on frequency-domain features
  -> irFFT2. By the convolution theorem, a 1x1 conv in frequency space
  is equivalent to a full-image-size spatial convolution -> global
  receptive field for the cost of one FFT.
- Reflect-padding is used before FFT to reduce circular-convolution /
  wraparound boundary artifacts (DFT assumes periodicity).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Spectral (frequency-domain) transform unit
# ---------------------------------------------------------------------
class SpectralTransform(nn.Module):
    """
    Global branch of an FFC block.
    Operates on the magnitude+phase (as real/imag stacked channels)
    of the 2D FFT of the input feature map.
    """
    def __init__(self, in_channels, out_channels, reduction=2):
        super().__init__()
        mid = max(out_channels // reduction, 8)

        # Pre/post conv in spatial domain around the FFT step
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Operates on [real, imag] stacked -> 2*mid channels
        self.freq_conv = nn.Sequential(
            nn.Conv2d(mid * 2, mid * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid * 2),
            nn.ReLU(inplace=True),
        )

        self.post_conv = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        b, c, h, w = x.shape

        # Reflect pad to reduce boundary wraparound artifacts from
        # treating the image as circularly periodic.
        pad = max(h, w) // 8
        x_p = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        # Real 2D FFT (assumes real-valued input, saves compute vs full fft2)
        freq = torch.fft.rfft2(x_p, norm="ortho")  # complex tensor
        real, imag = freq.real, freq.imag
        freq_feat = torch.cat([real, imag], dim=1)

        freq_feat = self.freq_conv(freq_feat)

        real2, imag2 = torch.chunk(freq_feat, 2, dim=1)
        freq_out = torch.complex(real2, imag2)

        out = torch.fft.irfft2(freq_out, s=x_p.shape[-2:], norm="ortho")
        # undo reflect pad
        out = out[:, :, pad:pad + h, pad:pad + w]

        return self.post_conv(out)


# ---------------------------------------------------------------------
# Squeeze-and-Excitation Channel Attention
# ---------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention Block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


# ---------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation) for Noise Conditioning
# ---------------------------------------------------------------------
class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation based on estimated noise variance."""
    def __init__(self, channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels * 2)
        )

    def forward(self, x, noise_std):
        params = self.mlp(noise_std)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


# ---------------------------------------------------------------------
# FFC block: local (spatial) + global (spectral) + SE channel attention
# ---------------------------------------------------------------------
class FFCBlock(nn.Module):
    def __init__(self, channels, global_ratio=0.5):
        super().__init__()
        g_ch = int(channels * global_ratio)
        l_ch = channels - g_ch
        self.l_ch, self.g_ch = l_ch, g_ch

        # Local branch: standard conv, local <-> local and global <-> local
        self.conv_l2l = nn.Conv2d(l_ch, l_ch, 3, padding=1, bias=False)
        self.conv_g2l = nn.Conv2d(g_ch, l_ch, 3, padding=1, bias=False) if g_ch > 0 else None

        # Global branch: spectral transform, local <-> global and global <-> global
        self.conv_l2g = nn.Conv2d(l_ch, g_ch, 3, padding=1, bias=False) if g_ch > 0 else None
        self.spectral = SpectralTransform(g_ch, g_ch) if g_ch > 0 else None

        self.bn_l = nn.BatchNorm2d(l_ch)
        self.bn_g = nn.BatchNorm2d(g_ch) if g_ch > 0 else None
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(channels, reduction=4)

    def forward(self, x):
        x_l, x_g = x[:, :self.l_ch], x[:, self.l_ch:]

        out_l = self.conv_l2l(x_l)
        if self.g_ch > 0:
            out_l = out_l + self.conv_g2l(x_g)
        out_l = self.act(self.bn_l(out_l))

        if self.g_ch > 0:
            out_g = self.conv_l2g(x_l) + self.spectral(x_g)
            out_g = self.act(self.bn_g(out_g))
            out = torch.cat([out_l, out_g], dim=1)
        else:
            out = out_l
        out = self.se(out)  # SE Channel Attention
        return x + out  # Local residual connection


# ---------------------------------------------------------------------
def apply_lee_filter(x, kernel_size=5, noise_var=0.04):
    """
    Classical Lee Speckle Filter (differentiable PyTorch implementation).
    Multiplicative noise model: I = R * eta with Var(eta) = noise_var.
    Weights local mean vs. raw pixel based on local variance / noise ratio:
      - In flat areas (low variance): weight -> 0, smooths speckle noise.
      - At sharp edges (high variance): weight -> 1, preserves true structural boundaries.
    """
    pad = kernel_size // 2
    mean = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    sq_mean = F.avg_pool2d(x * x, kernel_size, stride=1, padding=pad)
    var = torch.clamp(sq_mean - mean * mean, min=0.0)

    denom = var + (mean * mean) * noise_var + 1e-6
    weight = torch.clamp(var / denom, 0.0, 1.0)
    return mean + weight * (x - mean)


# ---------------------------------------------------------------------
# Full restoration network: Classical Lee Prior + FiLM + SE-ResFFC
# ---------------------------------------------------------------------
class FFCRestorationNet(nn.Module):
    """
    Joint denoise + super-resolve network with Classical Lee Prior + FiLM + SE-ResFFC.
    scale: upsampling factor (2 for 256->512 or 128->256, etc.)
    """
    def __init__(self, in_ch=1, base_ch=64, n_blocks=8, scale=2, global_ratio=0.5):
        super().__init__()
        # Dual-channel stem: channel 0 = raw log input, channel 1 = classical Lee prior
        self.stem = nn.Conv2d(2, base_ch, 3, padding=1)
        self.film = FiLMModulation(base_ch)

        self.blocks = nn.Sequential(
            *[FFCBlock(base_ch, global_ratio=global_ratio) for _ in range(n_blocks)]
        )

        self.fuse = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # Upsample via pixel shuffle (sub-pixel conv), avoids checkerboard
        # artifacts common with transposed convolutions.
        self.upsample = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )

        self.to_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)
        self.scale = scale

    def forward(self, x):
        # 1. Homomorphic log transform to linearize multiplicative speckle noise
        x_clamped = torch.clamp(x, min=0.0)
        x_log = torch.log1p(x_clamped)

        # 2. Extract classical Lee speckle filter prior on the GPU
        x_lee_log = apply_lee_filter(x_log, kernel_size=5, noise_var=0.04)

        # 3. Estimate per-image noise standard deviation for FiLM degradation conditioning
        diff = x_log - x_lee_log
        noise_std = torch.std(diff, dim=[1, 2, 3], keepdim=False).unsqueeze(1)  # (B, 1)

        # 4. Concatenate raw input and classical prior as dual input channels
        x_dual = torch.cat([x_log, x_lee_log], dim=1)

        # 5. Bicubic baseline anchored on the clean Lee prior
        base_log = F.interpolate(x_lee_log, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # 6. Deep feature extraction with FiLM noise conditioning and SE-ResFFC blocks
        feat_stem = self.stem(x_dual)
        feat_stem = self.film(feat_stem, noise_std)
        feat = self.blocks(feat_stem)
        feat = self.fuse(feat) + feat_stem  # Long global residual connection
        feat = self.upsample(feat)
        residual_log = self.to_out(feat)

        out_log = base_log + residual_log
        # Convert back from log domain: expm1(y) = exp(y) - 1
        out = torch.expm1(out_log)

        # Ground truth is always [0,1]; clamp the final output accordingly.
        return torch.clamp(out, 0.0, 1.0)


if __name__ == "__main__":
    # Quick sanity check with a synthetic batch, mimicking a
    # noisy/low-res input that exceeds [0,1] (speckle overshoot).
    model = FFCRestorationNet(in_ch=1, base_ch=64, n_blocks=8, scale=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}")

    x = torch.rand(2, 1, 128, 128) * 1.3 - 0.1  # simulate values outside [0,1]
    y = model(x)
    print("Input shape :", x.shape, "range:", x.min().item(), x.max().item())
    print("Output shape:", y.shape, "range:", y.min().item(), y.max().item())
