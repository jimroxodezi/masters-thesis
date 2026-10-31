"""
sdr_capture.py
--------------
Captures IQ samples from the RTL-SDR Blog V4 dongle.

Falls back to a synthetic OFDM-like LTE signal generator when the
hardware is not connected (useful for unit-testing the pipeline).
"""

import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)   # suppress pyrtlsdr pkg_resources warning

from config import Config, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Live capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_iq(config: Config = cfg) -> np.ndarray:
    """
    Capture IQ samples from the RTL-SDR Blog V4 dongle.

    Returns
    -------
    np.ndarray, dtype complex64, shape (N,)
        Raw IQ samples at config.sdr.sample_rate.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        raise ImportError("pyrtlsdr not installed. Run: uv add 'pyrtlsdr==0.2.91'")

    sdr = RtlSdr()
    try:
        sdr.sample_rate = config.sdr.sample_rate
        sdr.center_freq = config.sdr.center_freq
        sdr.gain        = config.sdr.gain

        n_samples = int(config.sdr.sample_rate * config.sdr.capture_duration)
        # read_samples returns complex128 — downcast immediately to save memory
        samples = sdr.read_samples(n_samples).astype(np.complex64)
        print(f"[sdr_capture] Captured {len(samples):,} samples "
              f"@ {config.sdr.center_freq/1e6:.1f} MHz "
              f"({config.sdr.capture_duration:.0f} s)")
        return samples

    finally:
        sdr.close()


def save_iq(samples: np.ndarray, path: Path) -> None:
    """Persist raw IQ data to disk as a .npy file."""
    path = Path(path)
    np.save(path, samples)
    print(f"[sdr_capture] Saved {len(samples):,} samples → {path}")


def load_iq(path: Path) -> np.ndarray:
    """Load previously captured IQ samples."""
    path = Path(path)
    samples = np.load(path)
    print(f"[sdr_capture] Loaded {len(samples):,} samples ← {path}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic signal generator (hardware-free testing)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_lte(
    config: Config = cfg,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic OFDM signal that approximates a 1.4 MHz LTE carrier.

    The signal comprises 72 active QPSK-modulated subcarriers at 15 kHz
    spacing (ITU-R LTE numerology), embedded in AWGN at the specified SNR.

    Parameters
    ----------
    config  : Config
    snr_db  : Signal-to-noise ratio in dB
    seed    : RNG seed for reproducibility

    Returns
    -------
    np.ndarray, dtype complex64, shape (N,)
    """
    rng = np.random.default_rng(seed)

    n_samples    = int(config.sdr.sample_rate * config.sdr.capture_duration)
    sample_rate  = config.sdr.sample_rate
    t            = np.arange(n_samples) / sample_rate

    n_subcarriers       = 72           # LTE 1.4 MHz: 72 active RBs
    subcarrier_spacing  = 15e3         # 15 kHz (LTE numerology 0)

    signal = np.zeros(n_samples, dtype=np.complex128)

    for k in range(-n_subcarriers // 2, n_subcarriers // 2):
        # Random QPSK symbols per sample (simplified — real LTE uses blocks)
        i_bits = rng.choice([-1.0, 1.0], n_samples)
        q_bits = rng.choice([-1.0, 1.0], n_samples)
        qpsk   = (i_bits + 1j * q_bits) / np.sqrt(2.0)
        signal += qpsk * np.exp(2j * np.pi * k * subcarrier_spacing * t)

    # Normalise to unit power
    signal /= np.sqrt(np.mean(np.abs(signal) ** 2))

    # Add AWGN
    noise_power = 10 ** (-snr_db / 10.0)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )

    return (signal + noise).astype(np.complex64)


def inject_anomaly(
    samples: np.ndarray,
    config: Config = cfg,
    anomaly_type: str = "jammer",
    start_frac: float = 0.3,
    duration_frac: float = 0.05,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject a synthetic anomaly into a copy of the IQ stream.

    Parameters
    ----------
    samples       : clean IQ samples, complex64
    anomaly_type  : "jammer"   — narrowband CW interferer
                    "power"    — sudden amplitude spike
                    "silence"  — carrier dropout
    start_frac    : anomaly start as a fraction of total length
    duration_frac : anomaly duration as a fraction of total length
    rng_seed      : RNG seed

    Returns
    -------
    (corrupted_samples, label_array)
        label_array is 1 where anomaly is present, 0 elsewhere.
    """
    rng = np.random.default_rng(rng_seed)
    corrupted = samples.copy()
    N         = len(samples)
    labels    = np.zeros(N, dtype=np.int8)

    start = int(start_frac * N)
    end   = min(N, start + int(duration_frac * N))
    labels[start:end] = 1
    t     = np.arange(end - start) / config.sdr.sample_rate

    if anomaly_type == "jammer":
        # Narrowband CW tone at +200 kHz offset
        offset_freq = 200e3
        jammer = 2.0 * np.exp(2j * np.pi * offset_freq * t).astype(np.complex64)
        corrupted[start:end] += jammer

    elif anomaly_type == "power":
        # Amplitude spike ×5
        corrupted[start:end] *= 5.0

    elif anomaly_type == "silence":
        # Carrier dropout — replace with noise floor
        noise = (0.01 * (rng.standard_normal(end - start) +
                         1j * rng.standard_normal(end - start))).astype(np.complex64)
        corrupted[start:end] = noise

    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type!r}")

    return corrupted, labels
