# Method design (Chapter 3 and `code/`)

Single source of truth for the experiment. Chapter 3 describes this design, and `code/` implements it. Approved by the author on 2 November 2026 (Chapter 3 outline, decisions D1–D5).

**Labels:**
- **[fixed]** settings follow from approved decisions or from the standard.
- **[default]** settings are starting values that may be tuned on training and validation data only. Chapter 3 shows them as `\TBC` until the final configuration is frozen.
- **[field]** settings are measured or chosen during data collection.

## 1. Receiver and capture

- **Receiver** [fixed]: RTL-SDR Blog V4, through `pyrtlsdr` and librtlsdr (the RTL-SDR Blog driver build that supports the V4).
- **Host** [fixed]: Dell Latitude 5490, CPU only, Ubuntu 26.04. It runs capture, preprocessing, scoring, evaluation, latency and the demo; the neural networks are trained on a Google Colab GPU (§7).
- **Sample rate** [fixed, D2]: f_s = 1.92 MS/s, which is 30.72 MHz / 16. 30.72 MHz = 1/T_s with T_s = 1/(15000 × 2048) s (TS 36.211 clause 4).
- **Frequency correction:** 0 ppm [default]; the V4 has a 1 ppm TCXO. The value is logged.
- **Gain** [fixed rule]: manual, never automatic. Two fixed gain settings, g1 (primary) and g2 (for the gain-change test), are chosen during the band survey [field] to avoid ADC clipping on the strongest Class C capture. Clipping fraction = share of I or Q samples at the 8-bit extremes; it is logged per capture, and captures with more than 0.1% clipped samples are flagged [default].
- **Bias tee** [fixed]: off for all captures (Class A has a 50 Ω terminator, and the antenna is passive). Its state is logged.
- **Duration** [default]: 30 s per capture. Raw IQ is stored as interleaved uint8 (2 bytes per complex sample), about 115 MB per capture, with a JSON metadata sidecar.
- **Metadata per capture:**
  - identity: capture_id, session_id, position_id, class (A, B, C_centre or C_offset), UTC start and end time;
  - radio settings: centre frequency, sample rate, gain setting (dB), ppm, bias-tee state, receiver model and serial (EEPROM string), antenna;
  - measurements: temperature if available, number of samples, clipping fraction, mean power (dB);
  - context: cell-search result (tool, PCI, carrier centre, bandwidth if known), operator notes;
  - provenance: software version (git hash of `code/`).

## 2. Band survey

- `rtl_power` sweeps of the candidate downlink ranges (proposal §8.1) at the first capture position, before the first session, by day and at night (starting at 07:00 and 22:00 WAT, as the sessions), with a fixed gain of 40.2 dB (a tuner step; the proposal's 40 dB is applied as 40.2 dB anyway). 10 minutes per range [field guide], bins of at most 25 kHz averaged over 10 s (60 readings per bin); mean, max-hold and waterfall plots. Purpose and reasoning: thesis Section 3.3.2 (Table "Settings decided by the band survey"). The sweep CSVs are saved and plotted as mean power against frequency. LTE carriers appear as flat plateaus 5–20 MHz wide.
- **Cell search** [fixed] (author, 4 November 2026): `CellSearch` of LTE-Cell-Scanner, Jiao's fork (github.com/JiaoXianjun/LTE-Cell-Scanner, commit e7f71cb), built on the laptop without OpenCL (installation: thesis Appendix C). It steps 100 kHz across a range around each plateau at a fixed gain (`-g`, dB; use a tuner step, e.g. 40.2), finds PSS/SSS and decodes the MIB, and reports cell ID, carrier frequency, frequency offset, RX power and bandwidth in RBs. A plateau counts as a carrier only if a cell is found at its centre. Only `CellSearch` is used (not `LTE-Tracker` or its SIB decoding). At each session it is re-run at the chosen carrier with `--recbin` to keep the samples.
- The chosen band stays `\TargetBand` until the author confirms it.

## 3. Data classes and sessions

- **Class A** (receiver noise): antenna input terminated with 50 Ω. Captured at g1 and g2.
- **Class B** (idle, antenna connected): a frequency inside the surveyed range with no LTE carrier (guard region or unused channel), confirmed by survey and cell search at capture time. Captured at g1 and g2. Ground truth is set at capture level, and frames are **not** screened individually with detector statistics (that would bias the evaluation).
- **Class C_centre** (LTE active, window centred on the carrier centre): contains the PSS and SSS. Captured at g1 and g2.
- **Class C_offset** (LTE active, window inside the carrier but excluding its central 6 RBs): no sync signals present. Captured at g1 and g2.
  - The window centre is offset from the carrier centre by f_off = (0.54 MHz + W_occ/2)/2, where W_occ = N_RB × 0.18 MHz is the occupied width. This places the window midway between the edge of the central 6 RBs (±0.54 MHz) and the occupied edge (±W_occ/2).
  - Feasible only if W_occ/2 − 0.54 MHz ≥ 1.92 MHz, i.e. channel bandwidth ≥ 10 MHz. Example: a 10 MHz carrier gives f_off = 2.52 MHz, and the window spans 1.56–3.48 MHz from the centre.
  - For narrower carriers C_offset is not captured, and this is recorded.
- **Class B window placement:** a stretch of the surveyed range at least 1.92 MHz wide with no carrier (survey plus cell search), as close as possible to the C frequencies. CellSearch finds a carrier at its centre, so the check uses each nearby carrier's centre and bandwidth: the window must lie outside the band the carrier occupies. **Quiet rule** [impl] (4 November 2026, after the test survey): in every bin of the window the survey's mean power is within 3 dB, and its maximum over the sweeps within 6 dB, of the floor (5th percentile of the mean), so narrow (e.g. GSM) and intermittent signals are excluded; the dongle's own spike at each rtl_power hop centre (±3 bins) is ignored; 250 kHz margin from any signal (`survey.quiet_mask`, `suggest_b_windows`).
- **Ground truth for C:** an active FDD carrier transmits reference signals in every downlink subframe (TS 36.211 clause 6.10.1), so every frame of a C capture is labelled occupied once cell search has confirmed the cell.
- **Sessions** [D4, field]: at least 12 sessions over at least 4 days, by day and at night, from at least 2 positions in Benin City.
- **Positions** [fixed] (author, 4 November 2026): both at the author's home in Benin City: P1 = sitting-room (parlour) balcony, P2 = bedroom balcony; they face different directions. Antenna at a marked spot, same height and orientation every session, >= 30 cm from metal railings/bars, photographed; laptop indoors. The survey is made at P1. Before the first P2 session: CellSearch on the chosen carrier and one 10-min sweep of its range at P2 (carrier received, idle window quiet). Limitation: one residence, so transfer to other buildings/areas is not tested.
- **Session times** [fixed] (author, 4 November 2026): day sessions start at 07:00 and night sessions at 22:00 WAT (UTC+1). A session is labelled from the start of its first capture; a start more than 2 h [impl] from both times is labelled `off-schedule` and reported separately (`dataset.time_of_day`; `session` warns; `evaluate` writes `time_of_day.csv`). With one day and one night slot per day, 12 sessions take at least 6 days. Each session captures A, B, C_centre and C_offset at g1 and g2: 8 captures per session.
- **Capture order within a session:** A, B, C_centre, C_offset, each at g1 then g2, so the terminator is fitted and removed only once. This is the order implemented by `occupancy session`. The adaptive energy detector processes each session in this order.
- **Labelling check:** on a later day, a second pass over the capture-level labels (cell search re-run where possible, and inspection of each capture's mean spectrum and a spectrogram sample). Agreement is reported.

## 4. Frames and preprocessing

- **Frame** [fixed]: N = 16384 consecutive complex samples = 8.533 ms at 1.92 MS/s. Since the PSS and SSS recur every 5 ms, every frame of a C_centre capture contains at least one complete PSS and SSS. Non-overlapping frames.
- **DC removal** [fixed]: subtract the frame mean of the complex samples before any detector sees the frame. All detectors use the same DC-removed frames.
- **STFT** [fixed]:
  - K = 128-point FFT, giving a bin spacing of 15 kHz, equal to the LTE subcarrier spacing;
  - periodic Hann window of 128 samples, hop 128 (no overlap), so 128 columns per frame;
  - spectrogram S = 10·log10(|X|² + 1e-12) in dB, FFT-shifted so that frequency increases from bottom to top. Shape 128 (frequency) × 128 (time).
  - One column lasts 66.7 µs, close to one OFDM symbol (128 + 9 or 10 samples at 1.92 MS/s).
- **Normalization** [fixed rule]: global mean μ and standard deviation s of S over all training frames (one scalar each). Applied unchanged to validation and test: x = (S − μ)/s. No min–max scaling, and no clipping.
- Preprocessed frames are stored on disk as float16 memory-mapped arrays, one per capture, with a frame index.

## 5. Splits and cross-validation

- **Session-level 4-fold cross-validation** [fixed]. The 12 or more sessions are partitioned into 4 test groups of 3 sessions. For each fold:
  - test = its 3 sessions;
  - validation = 2 other sessions;
  - training = the remaining sessions.
  - The validation sessions rotate deterministically: with the folds in ascending order, validation = the first 2 sessions, in ascending session order, of the next fold's test group (cyclic).
- Every session is tested exactly once, by models that never saw it.
- **Training data** [D3]: Class B frames at g1 from the training sessions.
- **Validation data:** Class B frames at g1 from the validation sessions. Used for early stopping and threshold calibration.
- **Test data (main):** Class B and Class C frames at g1 from the test sessions, plus the injection set built from the test sessions only.
- **Gain-change test (RQ3, H2):** the same models and thresholds (calibrated at g1), applied to g2 captures of the test sessions.
- **Class A:** tracks the receiver noise floor per session and gain (H2 analysis), plus a receiver-only false-alarm check. Not used for training or as injection noise.
- **Normalization statistics** come from the fold's training data only.

## 6. Controlled-SNR injection set

- **SNR definition:** the ratio of LTE signal power to noise power over the whole 1.92 MHz window.
- **Noise power:**
  - P_n(b) = mean |y|² of an idle frame b (Class B, same test session and gain).
  - P_n,C = median of P_n over the session's Class B frames at the same gain.
  - P_s(c) = P(c) − P_n,C, for a Class C frame c.
- **Mixture:** x = α·c + b, with α² = SNR_target · P_n(b) / P_s(c). The effective SNR, which counts the scaled noise inside c, is SNR_eff = α²·P_s(c) / (P_n(b) + α²·P_n,C). The label is occupied. SNR_eff is stored and used for stratification.
- **Admissible c:** natural SNR P_s/P_n,C at least 10 dB above the target [default]. The c and b frames come from different times of the same test session; Class C frames from C_centre and C_offset are kept separate.
- **SNR grid** [D5]: −20 to +10 dB in 2 dB steps (16 levels). [default] 500 mixtures per level per session.
- **Low SNR (H1)** [fixed definition]: SNR ≤ −4 dB on the grid, the levels below the energy detector's SNR wall for 1 dB of noise uncertainty (−3.3 dB, Tandra & Sahai 2008, eq. 2).

## 7. Detectors (all on identical frames)

1. **Convolutional autoencoder** (primary, `conv_ae`):
   - Input 1×128×128.
   - Encoder: three stages of [Conv 3×3 → BatchNorm → ReLU] with max-pool 2× between stages, channels 16, 32, 64 (as in the author's original model), then a 1×1 conv to c_lat = 8 latent channels [default] → latent 8×16×16 (2048 values, 8× compression).
   - Decoder: mirror, with Upsample 2× + Conv 3×3 + BN + ReLU, and a linear output conv.
   - Loss: MSE.
2. **Fully connected autoencoder** (`fc_ae`): flatten 16384 → 256 → 128 (latent) → 256 → 16384, ReLU, linear output [default]. Loss: MSE.
3. **Convolutional VAE** (`conv_vae`):
   - Encoder as in `conv_ae`, giving μ and log σ² maps (8×16×16); reparameterization; decoder as in `conv_ae`.
   - Loss: per-pixel mean MSE + β_KL · KL, with the KL averaged per latent element, β_KL = 1 [default].
   - At scoring time, the reconstruction is the decoder output at z = μ.
   - Scores: MSE, and the percentile score PER of Tian et al. (2022, eq. 10):
     - PER = β·q_ξ(f_γ(|X−X̂|) over the background region) + q_η(f_γ(|X−X̂|) over the signal region);
     - background mask = [maxpool_γ(X01) < α], where X01 is the frame min–max scaled to [0, 1] for masking only;
     - f_γ = γ×γ min-pooling (stride 1, same padding);
     - γ = 3, α = 0.05, β = 2, ξ = 90, η = 99;
     - percentiles are taken over the pixels inside each region (0 if a region is empty). This and the X01 masking are choices made where the paper's description leaves the details open.
4. **PCA** (`pca`): on flattened normalized training frames; d = the smallest number of components explaining ≥ 95% of training variance, capped at 256 [default]; IncrementalPCA if memory requires. Score: mean squared reconstruction error.
5. **Energy detector, fixed threshold** (`ed_fixed`): T = (1/N) Σ|y(n)|² on the DC-removed frame. The threshold is the validation quantile (Section 8). This is equivalent to freezing the noise estimate at its validation value. For comparison, the theoretical threshold for each target P_FA is also reported: λ = σ̂²/(2N) · F⁻¹_{χ²(2N)}(1 − P_FA), with σ̂² = the median of T on the validation frames (Chapter 2, eq. 2.5).
6. **Energy detector, adaptive threshold** (`ed_adaptive`):
   - The running noise estimate σ̂² is the median of T over the most recent M = 64 frames declared idle [default], initialized to the validation median.
   - Decision statistic: T / σ̂².
   - κ is calibrated by running the same procedure over the validation stream and taking the (1 − P_FA) quantile of T/σ̂².
   - Each session is processed as a stream in capture order (A is excluded; B, then C_centre, then C_offset).
7. **PSS correlator** (`pss`):
   - Templates: the three PSS sequences d_u(n), u ∈ {25, 29, 34} (TS 36.211 Table 6.11.1.1-1), length 62. n = 0…30 maps to subcarrier offsets −31…−1, and n = 31…61 to +1…+31 (DC unused). A 128-point IFFT gives 128-sample time-domain templates at 1.92 MS/s; the cyclic prefix is not included.
   - Statistic: max over the three roots and all lags of the normalized correlation ρ = |Σ y(n+m) p*(m)|² / (Σ|y(n+m)|² · Σ|p(m)|²).
   - Applicable to C_centre frames. Results on C_offset frames are reported to show its failure.
   - Training: none.

**Training** [defaults, from the author's original config]:
- Adam, learning rate 1e-3 (1e-4 for the VAE), weight decay 1e-5;
- batch size 64, at most 100 epochs, early stopping on validation loss (patience 15, min_delta 1e-6);
- ReduceLROnPlateau (factor 0.5, patience 7); gradient clipping at L2 norm 1.0.
- Seeds [D5]: 5 per model and fold (seeds 0–4). [default] Ablations (latent size c_lat ∈ {4, 8, 16}; FC latent ∈ {32, 128, 512}) use 3 seeds.
- `torch.set_num_threads(4)` on the laptop.
- **Where training runs** [fixed, 2 November 2026]: on a Google Colab GPU. Only the pack is uploaded: the preprocessed Class B g1 spectrograms of all sessions, the fold definitions, the per-fold normalization statistics, the frozen config and a SHA-256 manifest. The trained models come back to the laptop for calibration, evaluation, latency and the demo.
- **On the GPU:** cuDNN deterministic mode, benchmark off, `torch.use_deterministic_algorithms(True, warn_only=True)`. Results are not guaranteed bit-identical, because bilinear upsampling backward has no deterministic CUDA kernel. The device, GPU name and library versions are recorded per run.
- **Resume:** each run checkpoints every epoch (atomic writes) and resumes after a disconnect.

## 8. Thresholds

- For every detector, P_FA ∈ {0.01, 0.05} [D5].
- Threshold = empirical (1 − P_FA) quantile of its scores on the fold's idle validation frames (g1). For the adaptive ED it is κ on T/σ̂² (above).
- Larger score = more likely occupied. The decision is occupied if score > threshold.
- The realized test P_FA (on test Class B frames) is reported alongside the target.

## 9. Evaluation

- **Per fold, per seed, per detector, at each P_FA:**
  - P_D (on Class C frames and on injection mixtures by SNR level);
  - realized P_FA (on test Class B);
  - confusion matrix, precision, recall, F1;
  - ROC curve and AUC, precision–recall curve.
  - Class balance is reported because precision and F1 depend on it.
- **Pooling:** test predictions are pooled over the 4 folds, so every session is a test session once.
- **Seeds:** mean ± standard deviation over the 5 seeds.
- **Bootstrap:** 1,000 resamples of sessions with replacement [D5]. Metrics are recomputed on the frames of the resampled sessions, giving 95% percentile intervals. Also used for the differences between detectors (paired, same resampled sessions).
- **H1:** compare conv_ae and ed_adaptive on the injection set at SNR ≤ −4 dB, at P_FA 1% and 5%. ΔP_D = P_D(conv_ae) − P_D(ed_adaptive), pooled over the low-SNR levels. H1 is supported at an operating point if the 95% bootstrap interval of ΔP_D is entirely above 0.
- **H2:** for g2 test data (thresholds from g1), departure = |P_FA,realized − P_FA,target| and |P_D(g2) − P_D(g1)| on matched injection SNR levels. Compare conv_ae with ed_fixed (ed_adaptive as reference). H2 is supported if conv_ae's departures are smaller, with 95% bootstrap intervals of the paired differences entirely on the favourable side. Also reported: the noise-floor change between g1 and g2 (Class A and Class B medians).
- **RQ3:** cross-session results are the fold-pooled ones; the gain change is as in H2. Positions are compared where sessions at more than one position exist.
- **Failure analysis:** a sample of false alarms and misses per detector is inspected (spectrogram, time of day, session, gain).

## 10. Computational cost (RQ4)

- On the laptop, with `torch.set_num_threads(4)`, after 50 warm-up frames.
- Per-frame latency over 1,000 frames at batch size 1, reported as median and 95th percentile, separately for preprocessing (DC removal + STFT + normalization) and for each detector's scoring, and end to end.
- Throughput in frames per second at batch size 64.
- Real-time factor = processing time per frame / 8.533 ms (< 1 means real time).
- Timer: `time.perf_counter`. CPU model, thread count, library versions and power mode are logged.
- **Power condition** [fixed] (author, 3 November 2026): mains power and the Performance power profile, for every timing and for the demo. The `latency` command refuses any other known power state (`--ignore-power` only for test runs); the demo warns. A timing-only check with untrained models showed the convolutional networks 4–5 times slower on battery with the Power Saver profile.
- **Settling** [impl]: before any timing, every detector runs in turn for 5 s (`settle_s`); without it, the first detector timed in a process was up to 1.5 times slower than when timed again in the same process (a start-up effect lasting a few seconds, longer than the 50-frame warm-up). Found 3 November 2026.
- **Optimization effect** [impl]: each neural network is also timed "as trained" (model as saved, default memory layout, PER with `torch.quantile`), alternating frame by frame with its scoring form on the same frames; `latency.json` reports both medians and their ratio (Chapter 4).
- **Scoring copy** [impl]: every score (evaluation, latency, demo) is computed with `models.inference_model`: BatchNorm folded into the preceding convolution, channels-last layout for the convolutional models. Reconstructions change only by floating-point rounding (tested to < 1e-5 of their range); the PER percentiles use numpy's linear method, identical to `torch.quantile`.

## 11. Real-time demonstration

- Streaming capture at 1.92 MS/s in blocks of N samples, then DC removal, STFT, normalization and the chosen detector with its calibrated threshold.
- A CSV log per frame: UTC timestamp, centre frequency, gain, score, decision, processing time.
- Optional live view: rolling spectrogram and an occupancy strip, redrawn at most 4 times per second and not while more than 4 frames wait (one redraw costs about 40 ms).
- Queue of 32 frames (0.27 s) between the USB reader and the processing; when it is full, the OLDEST frames are dropped and counted, so decisions stay current.
- Samples lost before reaching the program (USB overflow) are estimated from the block arrival times and reported (`missing_blocks_estimate`).
- Mains power and the Performance profile, as in section 10.
- A short logged run in the chosen band illustrates occupancy over time.

## 12. Reproducibility and ethics

- Reception is passive. The PSS correlator only tests for known sequences; nothing is decoded, and no system information or user data is recorded.
- Configs, seeds, git hashes and library versions are logged with every result file.
- Code and, subject to the supervisor's approval, data and labels are released.
