# LTE Anomaly Detection — RTL-SDR Blog V4 + PyTorch Autoencoder

Unsupervised anomaly detection on LTE radio signals using a convolutional
autoencoder trained on normal IQ captures from the RTL-SDR Blog V4 dongle.

---

## Project structure

```
lte_anomaly/
├── config.py                   ← all hyperparameters (edit here)
├── sdr_capture.py              ← live SDR capture + synthetic signal generator
├── preprocessing.py            ← IQ → spectrogram → normalised dataset
├── model.py                    ← convolutional autoencoder (Conv2d)
├── trainer.py                  ← training loop, early stopping, checkpointing
├── detector.py                 ← threshold-based anomaly scorer + evaluator
├── visualize.py                ← training curves, spectrograms, ROC curve
├── main.py                     ← CLI entry point (capture / train / detect / demo)
├── lte_anomaly_detection.ipynb ← interactive Jupyter notebook
│
├── data/                       ← IQ .npy files (created at runtime)
├── checkpoints/                ← model weights + normaliser (created at runtime)
└── results/                    ← PNG plots (created at runtime)
```

---

## Setup

Install all dependencies into your uv virtual environment:

```bash
uv add torch torchvision torchaudio          # from pytorch-cpu index (see pyproject.toml)
uv add numpy scipy matplotlib scikit-learn
uv add "pyrtlsdr==0.2.91"
uv add jupyter ipykernel notebook
```

---

## Quickstart — no hardware needed

Run the full pipeline with a synthetic LTE signal:

```bash
uv run python main.py --mode demo
```

This generates a synthetic OFDM signal, trains the autoencoder, injects a
narrowband jammer anomaly, evaluates detection performance, and saves all
plots to `results/`.

---

## With the RTL-SDR Blog V4

### Step 1 — set your LTE carrier frequency

Edit `config.py` or pass `--freq` on the CLI:

```
Band 28 (700 MHz APT, common in NG/Africa): 796 MHz
Band 20 (800 MHz EU):                       806 MHz
Band 3  (1800 MHz):                        1842 MHz
```

### Step 2 — capture normal LTE signal (training data)

Point the antenna at a clear LTE downlink and run:

```bash
uv run python main.py --mode capture --freq 796 --duration 60
```

Saves `data/normal_iq.npy`.

### Step 3 — train the autoencoder

```bash
uv run python main.py --mode train --freq 796
```

Saves `checkpoints/best_model.pt`, `checkpoints/normalizer.pkl`,
`checkpoints/detector.pkl`, and plots to `results/`.

### Step 4 — run anomaly detection on a new capture

```bash
uv run python main.py --mode detect --freq 796
```

Scores each segment and prints how many were flagged.
Saves `results/anomaly_scores.png` and `results/anomaly_scores.npy`.

---

## Interactive notebook

Open `lte_anomaly_detection.ipynb` in VS Code (kernel: `Python (pytorch-env)`).

Set `USE_HARDWARE = False` (default) for the synthetic demo, or
`USE_HARDWARE = True` to use the live dongle.

---

## How it works

```
IQ samples (complex64)
        │
        ▼
  Segment into tiles (2048 samples each)
        │
        ▼
  2-sided STFT → magnitude spectrogram (dB)
  shape: (128 freq bins, 64 time frames)
        │
        ▼
  Min-max normalise to [0, 1]
        │
        ▼
  ConvAutoencoder
  Encoder: (1,128,64) → (16,64,32) → (32,32,16) → (64,16,8)
  Decoder: (64,16,8)  → (32,32,16) → (16,64,32) → (1,128,64)
        │
        ▼
  Reconstruction MSE per segment
        │
        ▼
  Threshold (99th percentile of training errors)
        │
        ▼
  score > threshold  →  ANOMALY
  score ≤ threshold  →  NORMAL
```

### Anomaly types (synthetic testing)

| Type      | Description                              |
|-----------|------------------------------------------|
| `jammer`  | Narrowband CW tone at +200 kHz offset    |
| `power`   | Sudden amplitude spike (×5)              |
| `silence` | Carrier dropout (replaced with noise)    |

---

## Key configuration knobs (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `sdr.center_freq` | 796 MHz | LTE carrier to tune to |
| `sdr.capture_duration` | 30 s | Length of each IQ capture |
| `preprocess.segment_length` | 2048 | IQ samples per tile |
| `preprocess.n_fft` | 128 | STFT FFT size (= freq bins) |
| `train.epochs` | 100 | Max training epochs |
| `train.patience` | 15 | Early stopping patience |
| `train.learning_rate` | 1e-3 | Adam LR |
| `anomaly.threshold_percentile` | 99.0 | Threshold aggressiveness |

Lowering `threshold_percentile` increases sensitivity (more detections,
more false positives). Raising it increases specificity.
