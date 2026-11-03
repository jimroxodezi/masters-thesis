# Chapter 3 (Methodology): outline for approval

Status: **approved by the author on 2 November 2026, with all recommendations (D1–D5).** The detailed settings are fixed in `notes/method-design.md`, which both Chapter 3 and the rewritten `code/` follow.

**Refinements made while fixing the design** (reported to the author):
- Injection noise comes from Class B, not Class A as D3 suggested, because thresholds are calibrated on Class B. Class A tracks the noise floor and gives a receiver-only false-alarm check.
- Splits use session-level 4-fold cross-validation, so that all 12 sessions are tested once and the session bootstrap has 12 units.
- Class C is captured with the window both centred on the carrier and offset inside it, so the sync correlator can be tested where it should and should not work.
- "Low SNR" for H1 is defined as SNR ≤ −4 dB, below the energy detector's SNR wall for 1 dB of noise uncertainty.

Governing plan: `notes/proposal.md`, Section 8, together with the decisions in CLAUDE.md (scope confirmed 2 November 2026). Chapters 1 and 2 already commit the method to:
- three data classes and session-level splits;
- a convolutional autoencoder (primary), a fully connected autoencoder, and a VAE with the percentile score;
- baselines: a PCA detector, fixed- and adaptive-threshold energy detectors, and a sync-signal correlator;
- thresholds calibrated on idle validation data at P_FA = 1% and 5%;
- bootstrap intervals over sessions, latency on the laptop CPU, and a real-time demonstration.

## The main problem: the code implements a different experiment

CLAUDE.md requires the method chapter to match `code/`. The code (read 2 November 2026) does not yet implement the plan:

| Item | Plan (proposal, Ch. 1–2) | Code now (`code/config.py`, `main.py`, `detector.py`) |
|---|---|---|
| Normal class | Idle spectrum (Classes A/B); LTE is what is detected | LTE capture is "normal"; anomalies are synthetic (jammer tone, power spike, dropout) |
| Split | By capture session | Random 15% validation split (`random_split`) |
| Threshold | (1 − P_FA) quantile of idle **validation** scores, P_FA 1% and 5% | 99th percentile of **training** errors |
| Gain | Fixed per session, logged | `gain = "auto"` |
| Sample rate | 2.4 MS/s (proposal) | 2.048 MS/s |
| Preprocessing | FFT 256/512, patches ~128×128, tuned; normalization from training data only | 2048-sample segments, 128-point FFT, hop 32 → 128×64 tiles; min–max normalization |
| Detectors | Conv AE, FC AE, VAE+PER, PCA, fixed/adaptive ED, sync correlator | Conv AE only (channels 16/32/64) |
| Metadata, sessions, injection at set SNR, bootstrap, latency, demo | Required | Absent or partial |
| Results | — | `results/` is empty; only a mock dongle has been used |

**Proposed handling.** Draft Chapter 3 now as the method *design*:
- written in the present tense ("each capture is labelled …"), so nothing is claimed to have been done;
- every setting not yet fixed in code is a red `\TBC{}` placeholder naming the config field it will come from.

Once the code implements the design and the captures exist, the chapter is converted to the past tense with the actual values. This keeps CLAUDE.md's rules (no invented settings, no implemented-method claims before `code/` has them). See decision D1.

## Structure (target ~7,500 words, 9 figures, 8 tables)

### 3.1 Introduction (~150 words)
Roadmap; how the method answers RQ1–RQ5.

### 3.2 Research Design (~400 words)
- Overview: comparative, quantitative evaluation of detectors on identical frames of real captures, plus a controlled-SNR set.
- → **Figure 3.1** (end-to-end pipeline) and **Table 3.1** (each RQ/hypothesis → data, detectors, metric, analysis).

### 3.3 Hardware, Software and Band Selection (~900 words)
- **3.3.1 Receiver and host.** RTL-SDR Blog V4 and the Dell Latitude 5490, CPU only, Ubuntu 26.04 → **Table 3.2**, with software versions from `code/pyproject.toml` and the lock file.
- **3.3.2 Band survey.**
  - `rtl_power` sweeps of the candidate ranges in Benin City, by day and at night, then cell search on the best candidate.
  - The chosen band stays `\TargetBand` until you confirm it.
  - → **Figure 3.2** (survey and window-placement flow).
- **3.3.3 Capture configuration.**
  - Sample rate (decision D2).
  - Gain fixed per session and logged.
  - Window placement relative to the carrier centre, which decides whether the sync correlator can work (Figure 2.2).
  - Treatment of the DC spike.
- **3.3.4 Metadata.** One record per capture → **Table 3.3** (fields from proposal §8.1).

### 3.4 Data Collection (~900 words)
- **Classes:**
  - A: terminated 50 Ω input, bias tee off.
  - B: antenna connected, frequencies with no LTE carrier, screened by cell search and power.
  - C: antenna connected, on an active carrier, at several positions and times.
- Sessions as the unit of independence (days, times, locations within Benin City).
- Screening of Class B for LTE contamination.
- → **Figure 3.3** (capture set-up for the three classes) and **Table 3.4** (dataset summary: sessions, durations, frames per class). The table's numbers stay `\TBC` until the captures exist.

### 3.5 Ground Truth and Labelling (~400 words)
- Frame-level labels from cell search plus spectrogram inspection.
- A subset relabelled on a later day, with agreement reported.
- Labels are never used to train the reconstruction models.

### 3.6 Controlled-SNR Injection (~600 words)
- SNR defined as signal power over noise power in the analysed bandwidth.
- Signal power estimated from Class C frames minus the noise floor; then scaled and added to independent idle frames, so labels are exact.
- SNR grid (decision D5).
- Limitation: the "signal" already carries the receiver's noise and 8-bit quantization.
- → **Figure 3.4** and equations.

### 3.7 Preprocessing (~700 words)
- Framing, STFT (window type, length, hop, FFT size), dB scaling, DC handling.
- Normalization statistics from training sessions only.
- Frame/patch geometry.
- → **Table 3.5** (parameters, from the final config) and **Figure 3.5** (preprocessing chain with actual sizes).
- Justify the window choice with Harris (1978) and the resolution trade-off with Allen & Rabiner (1977).

### 3.8 Data Splits (~400 words)
- Training: idle sessions only (decision D3: Class B, A, or both).
- Validation: held-out idle sessions (early stopping and threshold calibration).
- Test: unseen sessions with idle and LTE frames, plus the injection set.
- No frame-level mixing between sets.
- → **Figure 3.6** (session-level split scheme).

### 3.9 Detectors (~1,500 words)
- **3.9.1 Convolutional autoencoder (primary):** architecture from code → **Figure 3.7**.
- **3.9.2 Fully connected autoencoder:** same data and budget; the CNN-vs-dense ablation.
- **3.9.3 VAE with percentile score** (Kingma & Welling, 2014; Tian et al., 2022).
- **3.9.4 PCA detector:** number of components chosen on training data (Ruff et al., 2021).
- **3.9.5 Energy detectors:**
  - Statistic (2.3) per frame.
  - Fixed threshold, with the noise estimate frozen at validation.
  - Adaptive threshold, with the noise estimate updated from recent frames judged idle (window length and update rule to be set).
  - Cite Axell et al. (2010) and Yücek & Arslan (2009).
- **3.9.6 Sync-signal correlator:** correlation with the three PSS Zadoff–Chu sequences (TS 36.211 §6.11.1). Detection statistic and its threshold are calibrated like the others; it works only when the window covers the carrier centre.
- **Training protocol:** PyTorch, MSE, Adam, early stopping on idle validation loss, several seeds, latent-size/depth ablations.
- → **Table 3.6** (detector, input, training data, score, threshold source).

### 3.10 Thresholds and Operating Points (~300 words)
The (1 − P_FA) quantile of idle validation scores at P_FA = 1% and 5%, the same for every detector (Figure 2.10); the realized test P_FA is also reported.

### 3.11 Evaluation Protocol (~1,000 words)
- **Metrics** → **Table 3.7** (definitions): P_D at fixed P_FA, confusion matrix, precision, recall, F1, ROC/AUC, precision–recall.
- **SNR-stratified results:** on the injection set, and on natural captures with estimated SNR.
- **Uncertainty:** mean ± SD over seeds; bootstrap confidence intervals resampling sessions.
- **Robustness (RQ3, H2):** cross-session, gain change, different day or location; drift of the realized P_FA → **Table 3.8**.
- **Failure analysis.**
- **Hypothesis tests:** how H1 and H2 are judged.

### 3.12 Computational Cost (~300 words)
- Per-frame latency (warm-up, repeats, batch size 1, `time.perf_counter`, CPU thread count fixed) and throughput.
- Real-time factor against the sample rate, on the laptop.

### 3.13 Real-Time Demonstration (~250 words)
- Streaming capture → frames → detector → timestamped log and live view.
- → **Figure 3.8** (demo architecture).

### 3.14 Ethics, Reproducibility and Threats to Validity (~500 words)
- Passive reception, no decoding.
- Release of code and data subject to supervisor approval.
- Threats: one receiver, one site, one band, labelling error, injection approximation, idle-data contamination, session count.
- → **Figure 3.9** (threats mapped to mitigations) — optional; drop it if you prefer fewer figures.

### 3.15 Summary (~150 words)

## Decisions needed before drafting

- **D1. The code.**
  - **(Recommended)** I bring `code/` in line with the design after this outline is approved, and draft Chapter 3 in the present tense with `\TBC` placeholders now. Your current pipeline stays available (e.g. moved to `code/legacy/`).
  - Alternatively, you update the code yourself.
- **D2. Sample rate.**
  - **1.92 MS/s (recommended):** 30.72 MHz/16, LTE's basic rate divided by 16 (TS 36.211 clause 4). A 128-point FFT then gives 15 kHz bins, equal to the subcarrier spacing, and the PSS correlator can use the standard 128-point PSS waveform without resampling.
  - **2.4 MS/s** (proposal): the widest slice within the stable range.
  - **2.048 MS/s** (current code).
- **D3. Training data for the reconstruction models.**
  - **Class B (recommended)**, because test idle frames are antenna-connected. Class A is then used for injection noise and for tracking the noise floor across sessions.
  - Alternatively Class A, or A + B.
- **D4. Session plan.** Proposed minimum: 12 sessions over at least 4 days, covering day and night and two fixed gain settings, with 2 or more positions in Benin City; Classes A, B and C captured in every session. Bootstrap intervals over sessions need enough sessions. Please confirm or give your own numbers.
- **D5. Evaluation settings.** Proposed defaults:
  - injection SNR grid −20 to +10 dB in 2 dB steps;
  - 5 training seeds;
  - 1,000 bootstrap resamples (percentile intervals);
  - P_FA 1% and 5%.

## Sources needed for Chapter 3

- **Already in the bibliography:** 3GPP TS 36.211 (PSS, sampling unit T_s), Axell et al. (2010), Yücek & Arslan (2009), Tandra & Sahai (2008), Tian et al. (2022), Kingma & Welling (2014), Ruff et al. (2021), Goodfellow et al. (2016), Allen & Rabiner (1977), Harris (1978), RTL-SDR Blog (2024).
- **New, open access; I would fetch and read them:**
  - Kingma & Ba (2015, Adam, arXiv);
  - Paszke et al. (2019, PyTorch, NeurIPS);
  - Pedregosa et al. (2011, scikit-learn, JMLR);
  - Efron (1979, the bootstrap, Annals of Statistics, Project Euclid);
  - the documentation pages of `rtl-sdr`/`rtl_power` (Osmocom) and `pyrtlsdr`.
- **To choose:** an RTL-SDR-compatible LTE cell-search tool for ground truth (candidate: LTE-Cell-Scanner, to be verified that it builds on Ubuntu 26.04 and supports the V4).

## Draft status and deviations (2 November 2026)

- **Drafted** in `chapters/chapter3.tex` (about 5,200 words), in the present tense. Field values are red `\TBC{}` and provisional settings are red `\Prov{}`. Claims are in `notes/chapter3-claims.md`.
- **Length:** about 5,200 words instead of about 7,500. The difference is mainly the content that needs data: the list of sessions captured, the survey results and the final tuned settings. These are added after data collection.
- **Figures: 8 instead of 9.** The optional threats figure was dropped. The planned detector-overview figure is covered by Table 3.6 and Figure 3.1. The figures are 3.1 overview, 3.2 window placement (to scale), 3.3 capture set-up, 3.4 frame geometry, 3.5 cross-validation folds, 3.6 controlled-SNR construction, 3.7 convolutional autoencoder, and 3.8 real-time demo.
- **Tables: 9 instead of 8.** A one-session capture plan (Table 3.4) was added.
- **Still to verify against `code/`:** the decoder sizes in Figure 3.7, and the parameter counts of the networks, which should be quoted from the code. Settings marked `\Prov` are frozen once validation-based tuning is done.
