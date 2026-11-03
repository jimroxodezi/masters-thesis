"""
models.py
---------
Reconstruction models of design section 7 (input: normalised 1 x 128 x 128).

conv_ae
    Encoder: three stages [Conv 3x3 -> BatchNorm -> ReLU], channels 16, 32, 64,
    each followed by 2x max-pooling (128 -> 64 -> 32 -> 16), then a 1x1 conv to
    c_lat = 8 channels: latent 8 x 16 x 16 = 2048 values (8x compression).
    The design says "max-pool 2x between stages"; three pools are needed for the
    stated 16 x 16 latent, as in the author's original model, so the pool also
    follows the third stage. The 1x1 bottleneck conv is linear.
    Decoder (mirror): 1x1 conv c_lat -> 64 + BN + ReLU, then
    [Upsample 2x -> Conv 3x3 -> BN -> ReLU] 64 -> 32 and 32 -> 16, and a final
    Upsample 2x -> Conv 3x3 16 -> 1 with no BN and no activation (linear
    output). Upsampling is bilinear, as in the original model.
fc_ae
    flatten 16384 -> 256 -> 128 (latent, linear) -> 256 -> 16384, ReLU after the
    hidden layers, linear output.
conv_vae
    Encoder stages as conv_ae, then two 1x1 convs give mu and log sigma^2
    (8 x 16 x 16); z = mu + sigma * eps; decoder as conv_ae. Loss: per-pixel
    mean MSE + beta_KL * KL, with the KL averaged per latent element. For
    scoring (and for the validation loss) the reconstruction is decode(mu).

loss(x) and val_loss(x) return the loss and its parts (mse, and kl for the
VAE) as tensors on the model's device; train.py reads them once per epoch.

Scoring (design section 10) uses inference_model(model): a copy of the trained
model in which every BatchNorm that follows a convolution is folded into that
convolution, and the convolutional models use the channels-last memory layout,
which the CPU runs faster. The reconstruction changes only by floating-point
rounding (about 1e-6 of its range). Evaluation, latency and the demonstration
all score with this copy, so the timed computation is the evaluated one;
training and the validation loss use the model as trained.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .config import Config

INPUT_SIZE = 128


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _up_block(cin: int, cout: int, mode: str) -> nn.Sequential:
    return nn.Sequential(nn.Upsample(scale_factor=2, mode=mode,
                                     **({"align_corners": False} if mode in ("bilinear", "bicubic") else {})),
                         _conv_block(cin, cout))


def _init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class ConvEncoder(nn.Module):
    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        c = [1, *channels]
        self.stages = nn.ModuleList(_conv_block(c[i], c[i + 1]) for i in range(len(channels)))
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        for st in self.stages:
            x = self.pool(st(x))
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_channels: int, channels=(16, 32, 64), mode: str = "bilinear"):
        super().__init__()
        rev = list(channels)[::-1]                       # 64, 32, 16
        self.inp = nn.Sequential(nn.Conv2d(latent_channels, rev[0], 1, bias=False),
                                 nn.BatchNorm2d(rev[0]), nn.ReLU(inplace=True))
        self.ups = nn.ModuleList(_up_block(rev[i], rev[i + 1], mode) for i in range(len(rev) - 1))
        up_kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        self.out = nn.Sequential(nn.Upsample(scale_factor=2, mode=mode, **up_kw),
                                 nn.Conv2d(rev[-1], 1, 3, padding=1, bias=True))   # linear output

    def forward(self, z):
        h = self.inp(z)
        for up in self.ups:
            h = up(h)
        return self.out(h)


class ConvAE(nn.Module):
    kind = "conv_ae"

    def __init__(self, channels=(16, 32, 64), latent_channels: int = 8, upsample_mode: str = "bilinear"):
        super().__init__()
        self.encoder = ConvEncoder(channels)
        self.to_latent = nn.Conv2d(channels[-1], latent_channels, 1)       # linear bottleneck
        self.decoder = ConvDecoder(latent_channels, channels, upsample_mode)
        _init(self)

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def forward(self, x):
        return self.decoder(self.encode(x))

    def reconstruct(self, x):
        return self(x)

    def loss(self, x):
        mse = F.mse_loss(self(x), x)
        return mse, {"mse": mse.detach()}

    def val_loss(self, x):
        return self.loss(x)


class FCAE(nn.Module):
    kind = "fc_ae"

    def __init__(self, n_in: int = INPUT_SIZE * INPUT_SIZE, hidden: int = 256, latent: int = 128):
        super().__init__()
        self.n_in = n_in
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(n_in, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, n_in))

    def forward(self, x):
        return self.decoder(self.encoder(x)).view_as(x)

    def reconstruct(self, x):
        return self(x)

    def loss(self, x):
        mse = F.mse_loss(self(x), x)
        return mse, {"mse": mse.detach()}

    def val_loss(self, x):
        return self.loss(x)


class ConvVAE(nn.Module):
    kind = "conv_vae"

    def __init__(self, channels=(16, 32, 64), latent_channels: int = 8, upsample_mode: str = "bilinear",
                 beta_kl: float = 1.0):
        super().__init__()
        self.beta_kl = beta_kl
        self.encoder = ConvEncoder(channels)
        self.to_mu = nn.Conv2d(channels[-1], latent_channels, 1)
        self.to_logvar = nn.Conv2d(channels[-1], latent_channels, 1)
        self.decoder = ConvDecoder(latent_channels, channels, upsample_mode)
        _init(self)
        nn.init.zeros_(self.to_logvar.weight)     # start near unit variance
        nn.init.zeros_(self.to_logvar.bias)

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

    def reconstruct(self, x):
        mu, _ = self.encode(x)
        return self.decoder(mu)

    @staticmethod
    def kl_per_element(mu, logvar):
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()

    def loss(self, x):
        x_hat, mu, logvar = self(x)
        mse = F.mse_loss(x_hat, x)
        kl = self.kl_per_element(mu, logvar)
        total = mse + self.beta_kl * kl
        return total, {"mse": mse.detach(), "kl": kl.detach()}

    def val_loss(self, x):
        """Deterministic validation loss: reconstruction at z = mu, plus beta*KL."""
        mu, logvar = self.encode(x)
        mse = F.mse_loss(self.decoder(mu), x)
        kl = self.kl_per_element(mu, logvar)
        return mse + self.beta_kl * kl, {"mse": mse.detach(), "kl": kl.detach()}


# ── model specifications ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """A model variant: kind + latent size, with a stable name used in paths."""
    kind: str                 # conv_ae | fc_ae | conv_vae
    latent: int
    ablation: bool = False

    @property
    def name(self) -> str:
        return self.base_name if not self.ablation else f"{self.base_name}_lat{self.latent}"

    @property
    def base_name(self) -> str:
        return self.kind

    def to_dict(self) -> dict:
        return {"kind": self.kind, "latent": self.latent, "ablation": self.ablation, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(d["kind"], int(d["latent"]), bool(d.get("ablation", False)))


def main_specs(cfg: Config) -> list[ModelSpec]:
    m = cfg.models
    lat = {"conv_ae": m.conv_ae.latent_channels, "fc_ae": m.fc_ae.latent,
           "conv_vae": m.conv_ae.latent_channels}
    return [ModelSpec(k, lat[k]) for k in m.main]


def ablation_specs(cfg: Config) -> list[ModelSpec]:
    """Latent-size ablations; the main latent size is the main model itself."""
    m = cfg.models
    out = [ModelSpec("conv_ae", c, True) for c in m.ablation_conv_latent if c != m.conv_ae.latent_channels]
    out += [ModelSpec("fc_ae", c, True) for c in m.ablation_fc_latent if c != m.fc_ae.latent]
    return out


def build_model(spec: ModelSpec, cfg: Config) -> nn.Module:
    m = cfg.models
    if spec.kind == "conv_ae":
        return ConvAE(tuple(m.conv_ae.channels), spec.latent, m.conv_ae.upsample_mode)
    if spec.kind == "fc_ae":
        return FCAE(INPUT_SIZE * INPUT_SIZE, m.fc_ae.hidden, spec.latent)
    if spec.kind == "conv_vae":
        return ConvVAE(tuple(m.conv_ae.channels), spec.latent, m.conv_ae.upsample_mode, m.conv_vae.beta_kl)
    raise ValueError(f"unknown model kind {spec.kind!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── scoring copy of a trained model ─────────────────────────────────────────

CONV_KINDS = ("conv_ae", "conv_vae")


def inference_model(model: nn.Module) -> nn.Module:
    """Copy of a trained model for scoring: BatchNorm folded, channels-last.

    The trained model itself is not changed. Calling it again on a prepared
    copy returns that copy.
    """
    if getattr(model, "prepared_for_inference", False):
        return model
    m = copy.deepcopy(model).eval()
    for seq in [s for s in m.modules() if isinstance(s, nn.Sequential)]:
        for i in range(len(seq) - 1):
            if isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d):
                seq[i] = fuse_conv_bn_eval(seq[i], seq[i + 1])
                seq[i + 1] = nn.Identity()
    if getattr(m, "kind", None) in CONV_KINDS:
        m = m.to(memory_format=torch.channels_last)
    for p in m.parameters():
        p.requires_grad_(False)
    m.prepared_for_inference = True
    return m


def model_input(model: nn.Module, x: np.ndarray) -> torch.Tensor:
    """(B, 128, 128) normalised spectrograms -> (B, 1, 128, 128) in the model's layout."""
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)[:, None])
    if getattr(model, "kind", None) in CONV_KINDS:
        xt = xt.contiguous(memory_format=torch.channels_last)
    return xt


# ── PER score (Tian et al., 2022, eq. 10; design section 7, detector 3) ───────

def per_score(x: torch.Tensor, x_hat: torch.Tensor, gamma: int = 3, alpha: float = 0.05,
              beta: float = 2.0, xi: float = 90.0, eta: float = 99.0) -> torch.Tensor:
    """PER = beta * q_xi(f_gamma(E) over background) + q_eta(f_gamma(E) over signal region).

    E = |X - X_hat|; f_gamma = gamma x gamma min-pooling, stride 1, same padding
    (padded cells never win the minimum); background = [maxpool_gamma(X01) < alpha]
    with X01 the frame min-max scaled to [0, 1] (masking only); percentiles are
    taken over the pixels of each region (0 if a region is empty), with linear
    interpolation. x, x_hat: (B, 1, H, W). Returns (B,).
    The percentiles are computed with numpy's linear method, which gives the
    same values as torch.quantile (tests/test_cost_and_demo.py) in less time.
    """
    pad = gamma // 2
    err = (x - x_hat).abs()
    f = -F.max_pool2d(-err, gamma, stride=1, padding=pad)                # min-pooling
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    x01 = (x - lo) / (hi - lo).clamp_min(1e-12)
    bg = (F.max_pool2d(x01, gamma, stride=1, padding=pad) < alpha).flatten(1).numpy()
    f_np = f.detach().flatten(1).double().numpy()
    out = np.zeros(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        b = f_np[i][bg[i]]
        s = f_np[i][~bg[i]]
        qb = np.percentile(b, xi) if b.size else 0.0
        qs = np.percentile(s, eta) if s.size else 0.0
        out[i] = beta * qb + qs
    return torch.from_numpy(out)


def per_score_torch(x: torch.Tensor, x_hat: torch.Tensor, gamma: int = 3, alpha: float = 0.05,
                    beta: float = 2.0, xi: float = 90.0, eta: float = 99.0) -> torch.Tensor:
    """The PER score as first implemented, with torch.quantile: the reference
    for per_score (same values) and the "as trained" variant timed by latency.py."""
    pad = gamma // 2
    f = -F.max_pool2d(-(x - x_hat).abs(), gamma, stride=1, padding=pad)
    flat = x.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    x01 = (x - lo) / (hi - lo).clamp_min(1e-12)
    bg = (F.max_pool2d(x01, gamma, stride=1, padding=pad) < alpha).flatten(1)
    f64 = f.detach().double().flatten(1)
    out = torch.zeros(x.shape[0], dtype=torch.float64)
    for i in range(x.shape[0]):
        b, s = f64[i][bg[i]], f64[i][~bg[i]]
        qb = torch.quantile(b, xi / 100.0) if b.numel() else torch.tensor(0.0, dtype=torch.float64)
        qs = torch.quantile(s, eta / 100.0) if s.numel() else torch.tensor(0.0, dtype=torch.float64)
        out[i] = beta * qb + qs
    return out
