# Revised Thesis Proposal

**Working title:** Real-Capture Evaluation of Autoencoder-Based and Classical Detection for LTE Downlink Spectrum Occupancy Using a Low-Cost RTL-SDR

Jimrox Odezi Ogume, Department of Electrical and Electronic Engineering, University of Benin
Revision of the August 2025 proposal. Supervisor: `<FILL IN>`

> `[CITE]` marks a claim that still needs a source. Located sources and their verification status are in Section 12.

---

## 0. What changed from the August 2025 proposal, and why

| Topic | August 2025 proposal | This revision | Reason |
|---|---|---|---|
| Target band | Band 3 (1805–1830 MHz) | A downlink band **within the RTL-SDR tuning range** (at or below 1766 MHz), confirmed active locally | The RTL-SDR Blog V4 tunes only up to about 1.766 GHz, so the original example band cannot be received |
| Training/pre-training data | Public datasets (e.g. DeepSig RadioML) for early prototyping | **Only data captured with the RTL-SDR** | Author's decision; also avoids a domain mismatch with modulation-classification datasets |
| Novelty claim | "Practical and novel approach" | Reproducible, controlled **evaluation** with strong baselines on real low-cost captures | Autoencoder spectrum anomaly detection and autoencoder-vs-power-detection comparisons already exist (Section 3) |
| Expected outcome | AE expected to outperform ED, with specific percentages | Testable hypotheses; **a null or mixed result is an acceptable, reportable outcome** | The literature is mixed on whether learned detectors beat a well-tuned energy detector |
| Unsourced numbers | 20% detection at −20 dB; 85–90%; 4–5x improvement | **Removed** | No verified source |
| Baselines | Energy detector only | Fixed- and adaptive-threshold ED, linear (PCA) reconstruction baseline, LTE sync-signal correlator | A weak baseline makes any win uninformative |
| Ground truth | Visual labelling of spectrograms, cell scanner as an example | Cell-search confirmation plus **controlled digital SNR injection** | Gives exact labels at low SNR, where visual labelling fails |
| Data splitting | Not specified | **Split by capture session/day/location, not by frame** | Adjacent frames are correlated, so frame-level splits leak and inflate results |
| Extras | LangChain, transfer learning, C++ deployment | Dropped or reduced to a measured CPU latency study on the host laptop | Not needed for the research questions |

---

## 1. Introduction and motivation

LTE remains a major carrier of mobile broadband traffic even as 5G is deployed `[CITE]`. Knowing when and where a downlink band is occupied is useful for three reasons:

- **Interference management:** regulators and operators need to find unexpected or unauthorized emissions and sources of harmful interference `[CITE: regulator or operator source]`.
- **Network optimization:** long-term occupancy statistics show how a band is actually used over the day and week `[CITE]`.
- **Dynamic spectrum access planning:** cognitive radio systems need reliable sensing of whether a band is in use before any secondary use is considered (Yücek & Arslan, 2009; Axell et al., 2012).

Unsupervised deep learning offers one route to sensing: train a model on noise-only data so that it reconstructs the noise floor well, then treat poorly reconstructed frames as occupied. This thesis tests that idea on **real signals captured with a US$30-class RTL-SDR**, against classical detectors tuned as carefully as the learned model.

## 2. Classical spectrum sensing background

The thesis will survey the standard families of non-cooperative sensing: energy detection, matched filtering, cyclostationary feature detection, eigenvalue-based detection, and waveform-based detection. Energy detection is simple and needs no prior signal knowledge, but its reliability at low SNR is limited by noise-power uncertainty (the "SNR wall"; Gao et al., 2019). Cyclostationary and matched-filter detectors can do better but need more computation or prior knowledge of the signal (Yücek & Arslan, 2009; Axell et al., 2012).

## 3. Related work and gap

**Learned anomaly detection for spectrum.** Rajendran et al. (2019) proposed an adversarial-autoencoder detector on power spectral density data. Tian et al. (2022) applied a variational autoencoder to spectrograms with a percentile-based anomaly score and reported higher recall than a comparison method at a fixed false alarm rate for QPSK anomalies. Kang et al. (2022) used deep support vector data description for physical-layer anomaly detection. Torío et al. (2025) used a convolutional autoencoder and a plain power-measurement detector on calibrated VHF recordings and found the two agreed closely, with the autoencoder more sensitive to low-power transmissions; their analysis was offline.

**Learned spectrum sensing versus energy detection.** Several studies report learned detectors beating energy detection at low SNR. Gao et al. (2019) reported a deep learning detector with a lower SNR wall than the energy detector; their experiments used digitally modulated single-carrier signals (BPSK, QPSK, 8PSK, CPFSK, QAM16, QAM64, GFSK, PAM4) from a public dataset, **not OFDM**. Zeghdoud et al. (2023) implemented a CNN-LSTM detector on a USRP with GNU Radio and reported better low-SNR detection than CNN, LSTM and energy detection across FM, GSM and OFDM scenarios. However, Grosu and Terebeș (2021) found, on simulated OFDM signals over AWGN and Rayleigh fading channels, that an energy detector with a dynamic threshold outperformed a residual/recurrent convolutional neural network on average, especially at low SNR; the network was ahead only when trained on, and tested within, a narrow SNR range. The evidence is therefore **mixed**.

**Gap addressed.** A search for work that combines all of the following did not surface one: LTE downlink occupancy, a low-cost RTL-SDR, entirely real captured training and test data, and a benchmark of an autoencoder against well-tuned classical baselines with session-level generalization tests. This will be re-checked systematically during the literature review; if a close match is found, the framing will be adjusted.

## 4. Problem statement

Energy detection is the standard low-cost occupancy detector, but a fixed threshold is fragile when the noise floor drifts, which is common on low-cost hardware (gain changes, temperature drift, local interference). A learned noise model could in principle adapt to this. It is not established that an autoencoder trained on idle captures beats a *carefully tuned* adaptive energy detector on real RTL-SDR data, nor how either method behaves when conditions change between sessions. This thesis measures that.

## 5. Aim, research questions and hypotheses

**Aim:** to design, implement and rigorously evaluate an autoencoder-based occupancy detector for an LTE downlink band using only RTL-SDR captures, and to compare it with classical baselines under controlled and uncontrolled conditions.

**Research questions**
- **RQ1:** At fixed false alarm rates, how does the convolutional autoencoder's detection probability compare with fixed-threshold and adaptive-threshold energy detection across SNR?
- **RQ2:** Does the autoencoder add value beyond a linear (PCA) reconstruction detector and an LTE synchronization-signal correlator that exploits known signal structure?
- **RQ3:** How do the methods generalize across capture sessions, days and receiver gain settings, and across locations where captures from more than one location are available?
- **RQ4:** What are the per-frame latency and throughput of each method on the host laptop's CPU, and is real-time operation at the capture sample rate feasible?
- **RQ5:** How do the autoencoder's architecture (convolutional vs fully connected) and anomaly score (reconstruction MSE vs the VAE percentile score) affect detection?

**Hypotheses (to be tested, not assumed):**
- **H1:** the autoencoder achieves higher detection probability than adaptive energy detection at low SNR for the same false alarm rate.
- **H2:** when a test session's noise floor differs from that of the sessions used to set the thresholds, the autoencoder's realized false alarm rate and detection probability depart less from their calibrated values than those of the fixed-threshold energy detector. The adaptive-threshold energy detector shows how much adapting the noise estimate alone recovers.

Rejecting H1 and/or H2 is a valid and publishable result.

## 6. Objectives

1. Select a downlink band within the RTL-SDR tuning range that is confirmed active in the study area, and configure the receiver for it.
2. Build a capture protocol and collect a labelled dataset: terminated-input noise, antenna-connected idle spectrum, and LTE-active spectrum, across multiple sessions.
3. Implement the preprocessing pipeline (STFT spectrograms, normalization).
4. Train and tune autoencoders (convolutional primary, fully connected comparison) and a VAE with percentile scoring on idle data only.
5. Implement the baselines: fixed-threshold and adaptive-threshold energy detection, a PCA reconstruction detector and an LTE synchronization-signal correlator.
6. Evaluate all methods with identical frames, fixed-false-alarm operating points, ROC analysis, SNR-stratified results and session-level generalization tests.
7. Measure inference latency and throughput and build a real-time demonstration with logging.

### Priority tiers (to keep the plan achievable)
- **Core (must have):** band survey and capture protocol; Class A/B/C data; session-level splits; convolutional autoencoder; adaptive-threshold and fixed-threshold energy detectors; fixed-false-alarm evaluation with SNR-stratified results; controlled-SNR injection set.
- **Stretch (strongly preferred):** PCA reconstruction detector (cheap, and it tests whether non-linearity matters); fully connected autoencoder comparison (answers the supervisor's CNN-vs-dense question); cross-session and gain-change robustness tests; bootstrap confidence intervals over sessions; CPU latency measurement.
- **Optional (drop first if time runs short):** LTE sync-signal correlator, VAE with percentile score, dashboard polish.
- **Confirmed by the author on 2 November 2026:** all Stretch items are in scope. From the Optional tier, the LTE sync-signal correlator and the VAE with percentile score are in scope; dashboard polish stays optional. The CFAR-style detector is dropped. The fixed-threshold energy detector (the adaptive detector with its noise estimate frozen at the validation value) was added so that H2 can be tested.

## 7. Scope

**Included**
- One LTE downlink band within the tuner range (**not** Band 3, whose downlink lies above 1766 MHz), observed through the dongle's instantaneous bandwidth of about 2.4 MHz. A slice of one or more LTE carriers is observed, not a full wide carrier. Findings apply to sub-1.77 GHz carriers and should not be generalized to Band 3 or Band 7 carriers.
- All training, validation and test data captured by the author with the RTL-SDR. No external datasets are used.
- Autoencoder vs classical detectors for binary occupancy per frame.
- Real-time pipeline demonstration and latency measurement on the host laptop's CPU (Section 8.1).

**Excluded**
- Demodulating or decoding LTE (no cell IDs beyond what a cell-search tool reports for ground truth, no user data). Apart from the sync-signal correlator, which only tests for the presence of known synchronization sequences, only power and spectral structure are analysed.
- Uplink monitoring; any transmission or active probing (passive reception only).
- Cooperative multi-sensor sensing; GAN, diffusion and other deep models (the VAE is included as an ablation).
- CFAR-style, cyclostationary, eigenvalue-based and matched-filter detectors (reviewed in Chapter 2, not implemented).
- LTE network performance metrics (throughput, latency of the network).
- Operator-level or regulatory conclusions; implications are discussed conceptually.
- Deployment on a Raspberry Pi or any other embedded board; all processing runs on the host laptop.

## 8. Methodology

### 8.1 Hardware and band selection
- **Receiver:** RTL-SDR Blog **V4**. The unit has the Rafael Micro R828D tuner, the RTL2832U 8-bit ADC, a 1 PPM TCXO, a software-switchable bias tee and a built-in HF upconverter. Published V4 specifications (RTL-SDR Blog, 2024): 500 kHz–1.766 GHz, about 2.56 MHz stable bandwidth (up to 3.2 MHz with dropped samples), 50 Ω input. Sample rate 2.4 MS/s.
  - The HF upconverter only affects reception below about 28.8 MHz and plays no part in this study.
  - The bias tee supplies DC to the antenna port. Its state (on/off) is logged for every capture. It must be off when the 50 Ω terminator is fitted (Class A) and when a passive antenna is used.
  - A later model, the RTL-SDR Blog V4L, uses a different tuner (R828S) and front end. If any further dongle is bought, it must be a V4 (R828D), or the model must be recorded per capture and treated as a separate receiver.
- **Host computer:** Dell Latitude 5490 laptop with an Intel Core i5-8250U (8th generation, 4 cores/8 threads), 16 GB RAM and integrated Intel UHD Graphics 620 (no CUDA GPU, so PyTorch runs on the CPU), running Ubuntu 26.04 LTS. The dongle connects to it over USB. It runs capture, training, evaluation, latency measurement and the real-time demonstration; no embedded board is used.
- **Band choice (not yet decided):** candidates within range, in the order to survey them in Benin City (band edges `[CITE: 3GPP TS 36.101]`):
  1. **800 MHz band** (Band 20, downlink 791–821 MHz, or another 800/850 MHz plan). News reports confirm that the Nigerian Communications Commission (NCC) approved MTN Nigeria's use of 2 × 10 MHz in the 800 MHz band for LTE, but do not state the band plan. Secondary sources list MTN LTE in Band 20 at 791–811 MHz; an 850 MHz plan (e.g. Band 5, downlink 869–894 MHz) cannot be ruled out, so the survey covers 790–894 MHz.
  2. **Band 28**, downlink 758–803 MHz. Secondary sources list Glo LTE on 700 MHz, with an MTN layer also reported at 768–778 MHz.
  3. **Band 8**, downlink 925–960 MHz. Operators hold 900 MHz layers (reported at 925–935, 935–940 and 950–955 MHz), but these may carry GSM or other technologies, so LTE presence must be confirmed.
  These sources are secondary and partly dated, and an allocation is not the same as a live LTE carrier. **The band is chosen only after a local survey.**
- **Survey protocol:** sweep each candidate downlink range with `rtl_power` (for example `rtl_power -f 790M:894M:25k -g 40 -i 10 -e 10m b800.csv` for the 800/850 MHz range, `rtl_power -f 758M:803M:25k -g 40 -i 10 -e 10m b28.csv` and `rtl_power -f 925M:960M:25k -g 40 -i 10 -e 10m b8.csv`) at the capture site, in the daytime and at night. LTE carriers appear as flat, roughly 5–20 MHz wide plateaus above the noise floor. Confirm the best candidate with an RTL-SDR-compatible LTE cell-search tool and record cell identities for the ground-truth log. Record the survey outputs in the thesis as evidence for the band choice.
- **Excluded bands:** Band 3 (1805 MHz and above) and Band 7 (2.6 GHz) cannot be tuned by the V4 and are outside the study.
- **Antenna and gain:** a band-appropriate antenna, with gain settings fixed per session and **logged as metadata** (gain is a main driver of noise-floor changes).
- **Metadata per capture:** time, location, receiver model and serial/EEPROM identifier, antenna, gain, bias-tee state, center frequency, sample rate, temperature if available, cell-search result.

### 8.2 Data collection (three classes)
- **Class A, receiver noise:** the antenna input terminated with a matched 50 Ω load. Captures the receiver's own noise floor, with no external signals.
- **Class B, idle environment:** antenna connected, tuned to spectrum with no LTE carrier (guard regions, unused channels, or quiet periods), confirmed by cell search and power screening. Captures environmental noise and local interference.
- **Class C, LTE active:** antenna connected, tuned onto active carrier(s), at several positions, times and conditions to vary natural SNR.
- Multiple sessions across different days, times and (if possible) locations. Each session is an independent unit for splitting.

### 8.3 Ground truth
- Presence of LTE is confirmed with a cell-search tool and inspection of the spectrogram, at the **frame level**.
- A subset is labelled twice (second pass on a later day) to check labelling consistency.
- Labels are never used to train the autoencoder, only to select frames for the idle training set and to score detection.

### 8.4 Controlled-SNR test set (digital injection)
- Take LTE-active frames and idle noise frames from separate captures and combine them in software at target SNR levels spanning the range of interest, so that the label is exact.
- Define SNR explicitly (signal power over noise power in the analysed bandwidth) and document how the noise estimate is obtained.
- Limitation to state: the LTE capture already contains its own noise and the RTL-SDR's 8-bit quantization, so injection is an approximation of a clean signal. It is **complemented by** real over-the-air captures at naturally varying SNR.

### 8.5 Preprocessing
- STFT of the IQ stream (initial FFT lengths 256 or 512, tuned), assembled into fixed-size time-frequency patches (initial size 128×128, tuned).
- Log (dB) scaling; normalization statistics computed from the **training set only** and then applied unchanged to validation and test data.
- Remove or flag known hardware artifacts (e.g. the DC spike), documenting any treatment.

### 8.6 Data splits (leakage control)
- Split **by capture session** (and by day/location where available), never by random frame.
- Training: idle frames only (Classes A and/or B, screened for LTE contamination).
- Validation: held-out idle frames, used for early stopping and **threshold calibration**.
- Test: unseen sessions containing idle and LTE-active frames, plus the injection set.

### 8.7 Models
- **Convolutional autoencoder (primary):** captures local time-frequency structure of spectrogram patches.
- **Fully connected autoencoder (comparison):** for the flattened-vector input. The CNN-vs-dense choice is justified by the ablation, not assumed.
- **PCA reconstruction detector:** a linear baseline, to show whether non-linearity adds anything.
- **VAE (score ablation):** a variational autoencoder with the percentile score of Tian et al. (2022).
- Training: PyTorch, MSE reconstruction loss, Adam, early stopping on idle validation loss. Fixed seeds and **multiple runs** with mean and standard deviation reported. Latent size and depth studied as ablations.

### 8.8 Baseline detectors
- **Energy detector:** per-frame power statistic, threshold from a target false alarm rate using an estimated noise power (theoretical chi-square/Gaussian approximation as a starting point), with an **adaptive** noise estimate from recent frames judged idle. A **fixed-threshold** variant uses the same statistic with the noise estimate frozen at its validation value; it is the comparator for H2.
- **LTE sync-signal correlator:** detects the primary/secondary synchronization signals when a carrier's center lies within the captured slice.
- All baselines are tuned on the same idle validation data as the autoencoder.

### 8.9 Operating points and evaluation
- **Thresholds** for every method are set on idle validation data at fixed false alarm rates (for example 1% and 5%), never on test data.
- **Metrics:** confusion matrix, detection probability at fixed false alarm rate, false alarm rate, precision, recall, F1, ROC and AUC, precision-recall curves.
- **SNR-stratified results** on the injection set and on naturally varying captures.
- **Uncertainty:** bootstrap confidence intervals resampled over **sessions** (not frames), and variability across training seeds.
- **Robustness tests (RQ3):** train on some sessions and test on others; change receiver gain between train and test; test on a different day and, if possible, location.
- **Failure analysis:** inspect false alarms and misses, and relate them to interference, gain changes or drift.
- **Compute (RQ4):** per-frame inference latency and throughput on CPU, with a measured real-time factor against the 2.4 MS/s stream, measured on the host laptop (Section 8.1).

### 8.10 Real-time demonstration
- Streaming capture, live spectrogram, per-frame occupancy decision, and timestamped logging; a simple dashboard (notebook or small web app). A sample of the logged data is used for a brief occupancy-over-time illustration of the chosen band.

### 8.11 Ethics and reproducibility
- Reception is passive and only power and spectral features are analysed, apart from the sync-signal correlator, which only tests for known synchronization sequences. No content is decoded and no user data is recorded.
- Code, configuration and, subject to supervisor approval, the captured dataset and labels are released so the benchmark can be reproduced.

## 9. Expected contributions
1. A documented, labelled dataset of real RTL-SDR captures for an LTE downlink band, with idle, active and controlled-SNR subsets.
2. A controlled benchmark of a learned detector against strong classical and linear baselines, with fixed-false-alarm operating points.
3. An analysis of generalization across sessions, days and gain settings on low-cost hardware.
4. An open implementation with measured latency and real-time feasibility on a commodity laptop.

## 10. Thesis structure (five chapters)

1. **Introduction:** background and motivation (interference management, network optimization, dynamic spectrum access planning), problem statement, aim, research questions and hypotheses, objectives, scope, contributions, thesis outline.
2. **Literature Review:** LTE downlink structure; cognitive radio systems; taxonomy of classical spectrum sensing (energy, matched-filter, cyclostationary, eigenvalue-based, waveform-based); SDR and the RTL-SDR; STFT and spectrograms; autoencoders (convolutional vs fully connected); related work on learned anomaly detection and learned sensing versus energy detection; research gap.
3. **Methodology:** hardware and band selection, three-class data collection, ground truth, controlled-SNR injection, preprocessing, session-level splits, models, baseline detectors, fixed-false-alarm thresholding, evaluation protocol, real-time demonstration, ethics and reproducibility.
4. **Results and Discussion:** detection performance by method and SNR, session and gain generalization, bootstrap confidence intervals, latency and throughput, failure cases; interpretation against RQ1–RQ5 and H1–H2; limitations and threats to validity.
5. **Conclusion and Recommendations:** summary of findings, contributions, recommendations, future work.

## 11. Risks and mitigations
- **The autoencoder may not beat adaptive energy detection.** Mitigation: hypotheses are testable; report the result honestly and analyse why (for example whether reconstruction error tracks signal power).
- **Truly idle LTE spectrum is hard to find in an urban area.** Mitigation: use the terminated-input class and guard/unused-channel captures, and screen frames for contamination.
- **Low-cost hardware impairments** (8-bit ADC, drift, DC spike, gain changes). Mitigation: log metadata, test robustness explicitly, document artifact handling.
- **Only a ~2.4 MHz slice of a wider carrier is observed.** Mitigation: state clearly; the analysis targets occupancy detection, not full-carrier characterization.
- **Close prior work may exist.** Mitigation: systematic literature search before finalizing the contribution statement.
- **Ground-truth noise.** Mitigation: cell-search confirmation, double labelling of a subset, and the exact-label injection set.

## 12. Sources located so far

Bibliographic details were checked online on 2 November 2026; BibTeX is in `references-verified.bib` (keys in brackets). "Metadata verified" means the details are confirmed, not that the paper has been read. Read each paper in full before citing it.

| Source [key] | Use in the thesis | Status |
|---|---|---|
| Tian, Y., Liao, H., Xu, J., Wang, Y., Yuan, S., & Liu, N. (2022). Unsupervised spectrum anomaly detection method for unauthorized bands. *Space: Science & Technology*, *2*, Article 9865016. https://doi.org/10.34133/2022/9865016 [`tian2022`] | VAE on spectrograms, percentile (PER) score | Metadata verified (publisher page gives volume 2); abstract read |
| Rajendran, S., Meert, W., Lenders, V., & Pollin, S. (2019). Unsupervised wireless spectrum anomaly detection with interpretable features. *IEEE Transactions on Cognitive Communications and Networking*, *5*(3), 637–647. https://doi.org/10.1109/TCCN.2019.2911524 [`rajendran2019`] | AAE on PSD data, SDR and Electrosense data | Metadata verified (IEEE Xplore); abstract read |
| Torío, P., Sánchez, M. G., Cuiñas, Í., & Santalla del Río, V. (2025). Unsupervised detection of offending radio transmissions by means of a deep learning autoencoder. *EURASIP Journal on Wireless Communications and Networking*, *2025*, Article 104. https://doi.org/10.1186/s13638-025-02496-3 [`torio2025`] | Closest prior comparison: autoencoder vs power measurement | Full text read; metadata verified (four authors, Article 104 per dblp) |
| Gao, J., Yi, X., Zhong, C., Chen, X., & Zhang, Z. (2019). Deep learning for spectrum sensing. *IEEE Wireless Communications Letters*, *8*(6), 1727–1730. https://doi.org/10.1109/LWC.2019.2939314 [`gao2019`] | SNR wall and learned sensing (single-carrier modulated signals, **not OFDM**) | Metadata verified; arXiv preprint (1909.02730) read; check the published version |
| Kang, Y., Wu, H., Zhao, Z., Li, Y., & Meng, J. (2022). DL-based anomaly detection at the physical-layer of cognitive radio by deep support vector data description. *IEEE Transactions on Cognitive Communications and Networking*, *8*(4), 1689–1705. https://doi.org/10.1109/TCCN.2022.3200377 [`kang2022`] | Related learned anomaly detection | Metadata verified (OpenAlex) |
| Yücek, T., & Arslan, H. (2009). A survey of spectrum sensing algorithms for cognitive radio applications. *IEEE Communications Surveys & Tutorials*, *11*(1), 116–130. https://doi.org/10.1109/SURV.2009.090109 [`yucek2009`] | Classical sensing taxonomy | Metadata verified (dblp) |
| Axell, E., Leus, G., Larsson, E. G., & Poor, H. V. (2012). Spectrum sensing for cognitive radio: State-of-the-art and recent advances. *IEEE Signal Processing Magazine*, *29*(3), 101–116. https://doi.org/10.1109/MSP.2012.2183771 [`axell2012`] | Classical sensing review | Metadata verified (TU Delft repository) |
| Grosu, G., & Terebeș, R. (2021). A Python-based framework for advanced research and development on spectrum sensing for cognitive radio. *Acta Technica Napocensis – Electronics and Telecommunications*, *61*(2), 11–18. https://users.utcluj.ro/~atn/papers/ATN_2_2021_3.pdf (no DOI) [`grosu2021`] | Adaptive ED outperformed an NN on average, especially at low SNR (simulated OFDM data) | Full text read; metadata verified (journal contents page) |
| Zeghdoud, S., Teguig, D., Tanougast, C., Mesloub, A., Sadoudi, S., & Nesraoui, O. (2023). SDR implementation of spectrum sensing using deep learning. *International Journal of Sensors, Wireless Communications and Control*, *13*(4), 226–236. https://doi.org/10.2174/2210327913666230719152400 [`zeghdoud2023`] | CNN-LSTM on USRP/GNU Radio vs ED, real-time | Metadata verified (publisher page prints names surname first); abstract read |
| Baset, A., Becker, C., Derr, K., Sarkar, S., & Kasera, S. K. (2023). AviSense: A real-time system for detection, classification, and analysis of aviation signals. *ACM Transactions on Sensor Networks*, *19*(1), Article 8, 1–35. https://doi.org/10.1145/3526089 [`baset2023`] | SDR-based real-time autoencoder system | Metadata verified (ACM DL); online 8 Dec 2022, issue dated Feb 2023; abstract read |
| RTL-SDR Blog. (2024). *RTL-SDR Blog V4 datasheet* (V1.0). https://www.rtl-sdr.com/wp-content/uploads/2024/12/RTLSDR_V4_Datasheet_V_1_0.pdf [`rtlsdrblog2024`] | Tuner, ADC, frequency range, bandwidth | Datasheet read; year taken from upload date (PDF itself is undated) |
| News reports on NCC approvals of MTN's 800 MHz spectrum (e.g. Nairametrics, 22 Mar 2021 and 22 Jan 2025; BusinessDay, 2019) | MTN holds 2 × 10 MHz in the 800 MHz band for LTE, licences to 2034 | Secondary; band plan and frequencies **not stated**; confirm with NCC records and the local scan |
| Spectrum Tracker, Nigeria page (spectrum-tracker.com/Nigeria); Wikipedia, "List of LTE networks in Africa" | Candidate operator bands in Nigeria | Secondary sources, partly dated; **confirm against the national regulator and a local scan** |

## 13. Items to confirm with the author before writing
1. The **target band** and center frequency. Pending the survey in Section 8.1; until then write band-specific text with placeholders.
2. Stretch and Optional items: **resolved 2 November 2026** (see the priority tiers in Section 6).
3. Whether any existing captures were taken with hardware or settings that differ from this plan (check the logs before describing them as part of the dataset).
4. Supervisor requirements for dataset release, ethics and format.
