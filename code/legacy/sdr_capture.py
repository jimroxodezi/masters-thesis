"""
sdr_capture.py
--------------
Captures IQ samples from the RTL-SDR Blog V4 dongle.

Also provides a synthetic OFDM-like LTE signal generator and an anomaly
injector, for running the pipeline without hardware.
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
    except ImportError as exc:
        # Also raised when pyrtlsdr is installed but the native librtlsdr is missing
        raise ImportError(f"cannot load pyrtlsdr: {exc}. "
                          "Run uv sync, and check that librtlsdr is installed.") from exc

    sdr = RtlSdr()
    try:
        sdr.sample_rate = config.sdr.sample_rate
        sdr.center_freq = config.sdr.center_freq
        sdr.gain        = config.sdr.gain

        n_samples = int(config.sdr.sample_rate * config.sdr.capture_duration)
        samples   = np.empty(n_samples, dtype=np.complex64)
        filled    = 0

        def _store_block(block: np.ndarray, _context) -> None:
            # Blocks arrive as complex128 — downcast on copy to save memory
            nonlocal filled
            n = min(len(block), n_samples - filled)
            samples[filled:filled + n] = block[:n]
            filled += n
            if filled >= n_samples:
                sdr.cancel_read_async()

        # Streaming keeps the samples contiguous and each USB transfer small;
        # one synchronous read of the whole capture would exceed the usbfs limit.
        sdr.read_samples_async(_store_block, num_samples=config.sdr.read_block_samples)

        if filled < n_samples:
            raise IOError(f"Capture stopped early: {filled:,} of {n_samples:,} samples")

        print(f"[sdr_capture] Captured {n_samples:,} samples "
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
    spacing (3GPP TS 36.211), embedded in AWGN at the specified SNR.

    Known limitation: symbols change on every sample rather than once per
    OFDM symbol, so each subcarrier is spectrally white and the sum has a
    flat spectrum with no visible 1.08 MHz occupied band. Use it to exercise
    the pipeline, not as a model of a real LTE carrier.

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

    n_subcarriers       = 72           # LTE 1.4 MHz: 6 RBs × 12 subcarriers
    subcarrier_spacing  = 15e3         # 15 kHz (LTE)

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
