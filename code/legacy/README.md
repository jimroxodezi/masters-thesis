# LTE Anomaly Detection — RTL-SDR Blog V4 + PyTorch Autoencoder

Unsupervised anomaly detection on LTE radio signals using a convolutional
autoencoder trained on normal IQ captures from the RTL-SDR Blog V4 dongle.

---

## Project structure

```
codelistings/
├── config.py                   ← all hyperparameters (edit here)
├── sdr_capture.py              ← live SDR capture + synthetic signal generator
├── sdr_smoke_test.py           ← minimal dongle check (tune, read one block, print)
├── preprocessing.py            ← IQ → spectrogram → normalised dataset
├── model.py                    ← convolutional autoencoder (Conv2d)
├── trainer.py                  ← training loop, early stopping, checkpointing
├── detector.py                 ← threshold-based anomaly scorer + evaluator
├── visualize.py                ← training curves, spectrograms, ROC curve
├── main.py                     ← CLI entry point (capture / train / detect / demo)
├── lte_anomaly_detection.ipynb ← interactive Jupyter notebook
│
├── data/                       ← IQ .npy files (created at runtime)
├── checkpoints/                ← model weights + normaliser (created at runtime; demo/ for demo mode)
└── results/                    ← PNG plots, metrics.json (created at runtime; demo/ for demo mode)
```

---

## Setup

Install the locked dependencies into a uv virtual environment:

```bash
uv sync
```

---

## Quickstart — no hardware needed

Run the full pipeline with a synthetic LTE signal:

```bash
uv run python main.py --mode demo
```

This generates a synthetic OFDM signal, trains the autoencoder, injects a
narrowband jammer anomaly, evaluates detection performance, and saves all
plots to `results/demo/`, with its checkpoints in `checkpoints/demo/`. Demo never
reads or writes `data/normal_iq.npy` and never replaces the detector that
`--mode train` saved, so it cannot collide with real work.

Pass `--anomaly all` to evaluate the jammer, the power spike and the dropout in
one run. The files of each anomaly then carry its name, such as
`roc_curve_jammer.png`, instead of the plain `roc_curve.png`.

On a CPU-only laptop (Intel Core i5-8250U) one training epoch at the default 30 s
capture took about 6 minutes: 360 s measured, for 399 steps at 0.85 s each plus
validation. Steps took about 3 s each while other CPU-heavy programs were running.
Generating one 30 s synthetic stream took a further 12 minutes, and demo generates
two. For a quick check of the whole pipeline, this took about 3 minutes:

```bash
uv run python main.py --mode demo --duration 1 --epochs 2 --anomaly all
```

---

## With the RTL-SDR Blog V4

The capture path has been tested only against a mock (a fake `rtlsdr` package, and
the real pyrtlsdr classes with stubbed librtlsdr calls), not a physical dongle.

### Step 1 — set your LTE carrier frequency

Edit `config.py` or pass `--freq` on the CLI:

```
Band 28 (700 MHz APT, common in NG/Africa): 796 MHz
Band 20 (800 MHz EU):                       806 MHz
```

LTE Band 3 (1805–1880 MHz) is above the tuner's 1766 MHz limit, so `--freq` rejects it.

### Step 2 — capture normal LTE signal (training data)

Point the antenna at a clear LTE downlink and run:

```bash
uv run python main.py --mode capture --freq 796 --duration 60
```

Saves `data/normal_iq.npy`.

### Step 3 — train the autoencoder

```bash
uv run python main.py --mode train
```

Saves `checkpoints/best_model.pt`, `checkpoints/normalizer.pkl`,
`checkpoints/detector.pkl`, `results/metrics.json`, and plots to `results/`.
Training deletes the old `detector.pkl` first and writes the new one last, so if
a run is interrupted, `--mode detect` refuses the incomplete set instead of
pairing a new model with an old threshold.
Pass `--input path/to/iq.npy` to train on another capture.

### Step 4 — run anomaly detection on a new capture

```bash
uv run python main.py --mode detect --freq 796
```

Scores each segment and prints how many were flagged.
Saves `results/anomaly_scores.png` and `results/anomaly_scores.npy`.
Pass `--input path/to/iq.npy` to score a saved capture instead of recording one.
Never writes to `data/normal_iq.npy`.

---

## Command-line options

| Option | Used by | Effect |
|---|---|---|
| `--mode {capture,train,detect,demo}` | all | Pipeline stage to run (required) |
| `--freq MHZ` | capture, detect | Centre frequency in MHz; above 1766 is rejected |
| `--duration SECONDS` | capture, detect, demo | Length of the capture or synthetic stream |
| `--epochs N` | train, demo | Maximum training epochs, overriding `train.epochs` |
| `--anomaly {jammer,power,silence,all}` | demo | Anomaly to inject (default `jammer`); `all` evaluates each in turn |
| `--input PATH` | train, detect | IQ `.npy` file to use instead of `data/normal_iq.npy` (train) or a new capture (detect) |

### `results/metrics.json`

`--mode train` records the run in `results/metrics.json`, and `--mode demo` in
`results/demo/metrics.json`:
`run` (command line, timestamps), `environment` (package versions, CPU), `config`,
`data`, `model`, `normalizer`, `training` and `threshold`. Demo adds `psd`,
`padding`, `detection` (one object per anomaly) and `threshold_sweep`. The file
is first written when training finishes, so an interrupted run keeps the training
summary. A value that is undefined (NaN) is written as `null`.

---

## Interactive notebook

Open `lte_anomaly_detection.ipynb` in VS Code (kernel: `Python (pytorch-env)`).

Set `USE_HARDWARE = False` (default) for the synthetic demo, or
`USE_HARDWARE = True` to use the live dongle. The synthetic stream is cached as
`data/synthetic_iq.npy`, apart from the real capture in `data/normal_iq.npy`.

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

The synthetic generator has a flat spectrum, with no visible 1.08 MHz occupied
band, so it does not model a real LTE carrier. It is for exercising the pipeline.

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
