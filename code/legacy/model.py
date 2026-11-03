"""
model.py
--------
Convolutional Autoencoder for LTE IQ spectrogram reconstruction.

Architecture
------------
Input  : (B, 1, 128, 64)  — normalised magnitude spectrogram

Encoder
  Conv2d(1→16)  + BN + ReLU  → MaxPool2d  →  (B, 16,  64, 32)
  Conv2d(16→32) + BN + ReLU  → MaxPool2d  →  (B, 32,  32, 16)
  Conv2d(32→64) + BN + ReLU  → MaxPool2d  →  (B, 64,  16,  8)

Bottleneck: (B, 64, 16, 8)  — 8 192-dim latent space

Decoder
  Upsample ×2 → Conv2d(64→32) + BN + ReLU  →  (B, 32, 32, 16)
  Upsample ×2 → Conv2d(32→16) + BN + ReLU  →  (B, 16, 64, 32)
  Upsample ×2 → Conv2d(16→1)  + Sigmoid    →  (B,  1,128, 64)

Output matches input shape exactly — no size mismatch.
"""

import torch
import torch.nn as nn
from pathlib import Path

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv → BN → ReLU encoder block (spatial size unchanged)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Upsample ×2 → Conv → BN → ReLU decoder block."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for unsupervised anomaly detection on
    LTE magnitude spectrograms.

    The model learns a compressed latent representation of normal LTE
    signals. Anomalies produce a high reconstruction error because the
    decoder has never seen them during training.

    Parameters
    ----------
    config : Config
        enc_channels : list[int], default [16, 32, 64]
            Output channel counts for the three encoder stages.
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        ch = config.model.enc_channels   # e.g. [16, 32, 64]

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = _enc_block(1,     ch[0])   # (B,  1, H,   W  ) → (B, 16, H,   W  )
        self.enc2 = _enc_block(ch[0], ch[1])   # after pool          → (B, 32, H/2, W/2)
        self.enc3 = _enc_block(ch[1], ch[2])   # after pool          → (B, 64, H/4, W/4)
        self.pool = nn.MaxPool2d(2)             # used between stages
        # after 3 pools:                         → (B, 64, H/8, W/8)

        # ── Decoder ───────────────────────────────────────────────────────
        self.dec3 = _dec_block(ch[2], ch[1])   # upsample ×2  → (B, 32, H/4, W/4)
        self.dec2 = _dec_block(ch[1], ch[0])   # upsample ×2  → (B, 16, H/2, W/2)

        # Final block: upsample + conv → sigmoid squashes output to [0, 1]
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch[0], 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),                        # → (B,  1, H,   W  )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input spectrogram to bottleneck representation."""
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct spectrogram from bottleneck representation."""
        z = self.dec3(z)
        z = self.dec2(z)
        z = self.dec1(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode. Returns reconstruction of x."""
        return self.decode(self.encode(x))

    # ── Reconstruction error ──────────────────────────────────────────────

    def reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Per-sample MSE between input and reconstruction.

        Parameters
        ----------
        x         : (B, 1, H, W)
        reduction : "mean" — returns (B,) tensor (one score per spectrogram)
                    "none" — returns (B, 1, H, W) pixel-wise errors

        Returns
        -------
        torch.Tensor of shape (B,) for reduction="mean"
        """
        with torch.no_grad():
            x_hat = self(x)
            err   = (x - x_hat) ** 2
        if reduction == "mean":
            return err.mean(dim=(1, 2, 3))
        return err

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)
        print(f"[model] saved → {path}")

    def load(self, path: Path, device: torch.device | None = None) -> "ConvAutoencoder":
        device = device or torch.device("cpu")
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"[model] loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(config: Config = cfg) -> ConvAutoencoder:
    """Instantiate and return a new ConvAutoencoder with seeded initial weights."""
    torch.manual_seed(config.train.seed)
    model = ConvAutoencoder(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] ConvAutoencoder  trainable params: {n_params:,}")
    return model
