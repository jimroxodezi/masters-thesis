# Chapter 2: claims and sources

Every cited claim in `chapters/chapter2.tex`, with the place in the source where it appears, so that any claim can be spot-checked. All passages were checked on 2 November 2026.

**Page conventions.**
- Plain page numbers are the printed page numbers of the published version, which is the version in `papers/` or the one downloaded.
- "Preprint p." and "ms. p." are pages of a preprint or manuscript copy. The published version is cited, so these sources are cited **without** page numbers in the text; the pages here are only for checking.
  - Preprints: Tandra & Sahai 2008 (30-page manuscript); Zeng & Liang 2009 (arXiv 0804.2960v1); Gao et al. 2019; Zheng et al. 2020; Peng et al. 2020; Tandiya et al. 2018; Rajendran et al. 2018, 2019 and 2020; Calvo-Palomino et al. 2020; Ruff et al. 2021 (arXiv 2009.11732v3).
  - Manuscripts: Baset et al. 2023 (accepted manuscript; printed page = PDF page − 3); Saber et al. 2021 (manuscript without the book's pages).
- "PDF p." is a page of a PDF that prints no page numbers: Tandra & Sahai 2005, Calvo-Palomino et al. 2015 and 2017, and the RTL-SDR Blog datasheet. Tandra 2005 and Calvo-Palomino 2015/2017 are cited without page numbers. The datasheet is cited as "p. 1".
- 3GPP specifications are cited by clause or table number (ETSI TS 136 211 V19.3.0, 2026-04; ETSI TS 136 101 V19.6.0, 2026-08).

Sources read for this chapter: the full papers in `papers/`, plus the open-access copies downloaded on 2 November 2026 (Zeng & Liang; Kingma & Welling; Ruff et al.; Masci et al.; Goodfellow et al., Chapters 9 and 14; the RTL-SDR Blog V4 datasheet). The 13 related-work papers of Section 2.8 were first extracted by a helper agent (`scratchpad/relwork-extraction.md`, with verbatim snippets). Every claim used below was then re-checked against the paper text.

## Section 2.2: The LTE downlink

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 1 | The LTE downlink transmits OFDM symbols | `3gpp2026a` | Clause 6.12 (OFDM baseband signal generation) | |
| 2 | Normal CP, 15 kHz: a resource block is 12 subcarriers (180 kHz) × 7 OFDM symbols (one slot) | `3gpp2026a` | Table 6.2.3-1 | 180 kHz = 12 × 15 kHz |
| 3 | FDD: uplink and downlink separated in frequency; frame structure type 1; 10 ms frame of ten 1 ms subframes numbered 0–9; each subframe two 0.5 ms slots | `3gpp2026a` | Clause 4.1 | "20 slots, numbered 0 to 19" follows from subframe i = slots 2i, 2i+1 (clause 4.1) |
| 4 | Six channel bandwidths 1.4–20 MHz with fixed numbers of resource blocks (6, 15, 25, 50, 75, 100) | `3gpp2026b` | Table 5.6-1 | Table 2.1; occupied width = N_RB × 180 kHz (derived) |
| 5 | The receiver observes at most about 2.56 MHz | `rtlsdrblog2024` | p. 1 ("2.56 MHz stable (up to 3.2 MHz with drops)") | Table 2.1 last column derived: only 1.08 MHz < 2.56 MHz |
| 6 | Downlink ranges: B28 758–803, B20 791–821, B5 869–894, B8 925–960, B3 1805–1880, B7 2620–2690 MHz | `3gpp2026b` | Table 5.5-1 | Table 2.2 |
| 7 | V4 tuning range 500 kHz to 1.766 GHz | `rtlsdrblog2024` | p. 1 | |
| 8 | 504 physical-layer cell identities in 168 groups of three | `3gpp2026a` | Clause 6.11 | |
| 9 | PSS generated from the identity within the group; SSS from the group | `3gpp2026a` | Clause 6.11.1.1 (root index from N_ID^(2), Table 6.11.1.1-1); clause 6.11.2.1 (m0, m1 from N_ID^(1)) | |
| 10 | PSS is a frequency-domain Zadoff–Chu sequence of length 62 | `3gpp2026a` | Clause 6.11.1.1 (d_u(n), n = 0…61) | |
| 11 | FDD: PSS in the last OFDM symbol of slots 0 and 10 (every 5 ms) | `3gpp2026a` | Clause 6.11.1.2 | |
| 12 | SSS: interleaved concatenation of two length-31 binary sequences, in the symbol before the PSS (l = N_symb^DL − 2) of slots 0 and 10 | `3gpp2026a` | Clauses 6.11.2.1, 6.11.2.2 | |
| 13 | Both mapped to 62 subcarriers around the centre, 5 reserved subcarriers each side (central 72 subcarriers, 6 RBs, 1.08 MHz) | `3gpp2026a` | Clauses 6.11.1.2, 6.11.2.2 (k = n − 31 + N_RB·N_sc/2; reserved n = −5…−1, 62…66) | |
| 14 | A UE may assume CRS are transmitted in all downlink subframes of an FDD cell supporting PDSCH | `3gpp2026a` | Clause 6.10.1 | "unless otherwise stated in [4, clause 12]" |
| 15 | In MBSFN subframes CRS only in the non-MBSFN region, the first one or two OFDM symbols | `3gpp2026a` | Clause 6.10.1.2 (CRS only in non-MBSFN region); clause 6.1 (non-MBSFN region = first one or two symbols for Δf = 15 kHz) | |
| F1 | Figure 2.1 (frame structure, PSS/SSS positions) | `3gpp2026a` | Clauses 4.1, 6.11.1.2, 6.11.2.2; Table 6.2.3-1 | Source: `figures/src/fig2-lte-frame-sync.svg` |
| F2 | Figure 2.2 (10 MHz carrier, 50 RBs, central 6 RBs, windows ≤ 2.56 MHz, to scale) | `3gpp2026b`; `3gpp2026a`; `rtlsdrblog2024` | Table 5.6-1; Table 6.2.3-1, clause 6.11; p. 1 | Drawn at 54 px/MHz |

## Section 2.3: Cognitive radio and dynamic spectrum access

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 16 | The term "cognitive radio" was coined by Joseph Mitola | `haykin2005` | p. 203 | |
| 17 | Cognitive radio extends software radio with model-based reasoning about radio etiquette; definition of radio etiquette | `mitola1999` | p. 13 (abstract) | Scanned PDF with OCR errors |
| 18 | Software radio is an ideal platform for cognitive radio | `mitola1999` | p. 13 | Also used in Section 2.5 |
| 19 | Cognition cycle: observe, orient, plan, decide, act; some supervised and unsupervised learning | `mitola1999` | p. 18 ("The Cognition Cycle") | |
| 20 | Haykin's definition of cognitive radio, its two objectives | `haykin2005` | p. 201 (abstract) | "input stimuli" paraphrased as incoming RF stimuli (p. 202: "passive sensing of RF stimuli") |
| 21 | Some bands largely unoccupied most of the time, some partially occupied, the rest heavily used | `haykin2005` | p. 201 | Haykin attributes this to his refs [2]–[4] |
| 22 | Spectrum hole definition (direct quotation) | `haykin2005` | p. 201 | Quoted with page number |
| 23 | Secondary user accessing a hole at the right place and time improves utilization | `haykin2005` | p. 201 | |
| 24 | Three cognitive tasks; tasks 1 and 2 in the receiver, task 3 in the transmitter | `haykin2005` | p. 202 | |
| 25 | Black, grey and white spaces; white (and grey to a lesser extent) are candidates for secondary use | `haykin2005` | p. 206 | |
| 26 | Spectrum sensing is one of the most challenging issues in cognitive radio systems | `yucek2009` | p. 116 (abstract) | |
| 27 | Motivation for CR: measurements show unused resources in frequency, time and space | `axell2010` | p. 322 | As in Chapter 1 |
| 28 | CRs must be able to detect very weak primary signals | `axell2010` | p. 323 | |
| 29 | Primary users must be detected to avoid harmful interference | `tandra2008` | ms. p. 1 (abstract) | |
| 30 | Nigeria: average occupancy < 10% over 80–2200 MHz at five sites; hopeful future for DSA | `popoola2018` | pp. 948, 951, 960 | See Chapter 1 claims 3 and 25 |
| F3 | Figure 2.3 (spectrum holes, schematic) | `haykin2005` | p. 201 | Occupancy pattern invented for illustration; caption says schematic |

## Section 2.4: Classical spectrum sensing

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 31 | H0: y = w; H1: y = x + w | `axell2010` | p. 322, eq. (1) | Thesis writes s(n) for the signal |
| 32 | Noise i.i.d. zero-mean circularly symmetric complex Gaussian, variance σ² | `axell2010` | p. 322 | |
| 33 | Definitions of P_D and P_F | `yucek2009` | p. 121, eqs. (5)–(6) | |
| 34 | P_F should be small so that transmission opportunities are not wasted | `yucek2009` | p. 121 | |
| 35 | Signal power hard to know (transmission, distance); threshold chosen for a given false-alarm rate; noise variance suffices | `yucek2009` | p. 121 | |
| 36 | Neyman–Pearson: optimal test compares the log-likelihood ratio with a threshold | `axell2010` | p. 322 | |
| 37 | Yücek & Arslan reviewed ED, waveform-based, cyclostationarity, radio identification, matched filtering, other methods (multitaper, wavelet); compared accuracy and complexity | `yucek2009` | pp. 120–124 (Section IV; Fig. 4, p. 124) | |
| 38 | ED needs no knowledge of the PU signal; MF needs complete knowledge | `yucek2009` | pp. 120, 123 | |
| 39 | Axell: ED assumes no known structure; feature detectors exploit cyclostationarity or eigenvalue structure | `axell2010` | pp. 322–324 | |
| 40 | Cooperation proposed against noise uncertainty, fading and shadowing | `yucek2009` | p. 124 (Section V) | |
| F4 | Figure 2.4 (taxonomy by prior knowledge) | `yucek2009`; `axell2010`; `zeng2009` | as rows 37–39, 59, 63 | Colour coding of "implemented" follows the confirmed scope (Chapter 1) |
| T3 | Table 2.3 rows | `yucek2009`; `axell2010`; `zeng2009`; `tandra2008` | ED: yucek pp. 120–121, axell p. 323, tandra ms. p. 8; MF: yucek p. 123, tandra ms. p. 1; cyclostationary: yucek pp. 122, 124, zeng preprint p. 1; eigenvalue: zeng preprint pp. 1, 3–4; waveform: yucek p. 122 | |
| 41 | Urkowitz: deterministic signal of unknown structure in flat band-limited Gaussian noise of known PSD; ideal band-pass filter, energy over an interval vs threshold; noncentral chi-square statistic | `kostylev2002` | p. 1606 (abstract, Sections I–II) | **Secondary citation**: "Urkowitz, 1967, as cited in Kostylev, 2002"; Urkowitz (1967) not obtained |
| 42 | Random amplitude from fading → statistic not chi-square; Kostylev derived P_D for Rayleigh, Rice, Nakagami and other amplitudes | `kostylev2002` | p. 1606 | |
| 43 | T = (1/N) Σ |y(n)|² vs λ | `axell2010`; `yucek2009` | Axell p. 322, eq. (2); Yücek p. 121, eq. (2) | Both use the sum without 1/N (stated in text) |
| 44 | With a Gaussian signal, the ED is Neyman–Pearson optimal | `axell2010` | p. 322 | |
| 45 | NT scaled chi-square, 2N DOF: σ²/2 under H0, (σ² + σs²)/2 under H1 | `yucek2009` | p. 121, eq. (7) | |
| 46 | P_FA, P_D and threshold expressions (eqs. 2.4–2.5 of the thesis) | `axell2010` | pp. 322–323, eqs. (3)–(5) | Adapted from the unnormalized statistic: λ = η_e/N |
| 47 | SNR defined as σs²/σw² | `yucek2009` | p. 121 | |
| 48 | ED universal; cannot exploit available knowledge; needs the noise power | `axell2010` | p. 323 | |
| 49 | ED the most common method because of low computational and implementation complexity | `yucek2009` | p. 120 | |
| 50 | ED challenges: threshold selection, cannot differentiate interference from PU and noise, poor at low SNR | `yucek2009` | pp. 120–121 | |
| 51 | ED limited when noise is not stationary or its variance unknown | `yucek2009` | p. 124 | |
| 52 | Device noise uncertainty (non-linearity, non-uniform time-varying thermal noise) vs environmental (other transmissions); accurate noise power very difficult to obtain | `zeng2009` | preprint p. 1 | |
| 53 | A small noise-power estimation error causes significant performance loss | `yucek2009` | p. 121 | |
| 54 | Noise power known only within [σn²/ρ, ρσn²], ρ = 10^(x/10) | `tandra2008` | ms. pp. 7–8 | Thesis writes δ for Tandra's x |
| 55 | Radiometer cannot robustly detect if P ≤ (ρ − 1/ρ)σn² | `tandra2008` | ms. p. 8 | |
| 56 | SNR_wall = (ρ² − 1)/ρ; definition of non-robustness | `tandra2008` | ms. p. 8, eq. (2); ms. p. 6 | |
| 57 | 1 dB uncertainty → wall ≈ −3.3 dB; sample complexity → ∞ near the wall | `tandra2008` | ms. p. 9 (Fig. 3; x = 1 dB marked) | −3.33 dB computed from eq. (2) |
| 58 | Without knowledge of the modulation, every detector has an SNR wall; any detector essentially as non-robust as the radiometer | `tandra2008` | ms. p. 1 (abstract) | |
| 59 | Extension to other model uncertainties (white/stationary noise, flat fading, ideal filters, infinite-precision A/D); noise calibration improves robustness | `axell2010` | p. 323 | Axell cites this to Tandra & Sahai |
| 60 | Finite dynamic range → any detector useless at low SNR under moderate noise uncertainty; all physical radios have such a limit | `tandra2005` | PDF p. 1 (abstract and introduction); Section IV, PDF p. 5 | |
| F6 | Figure 2.6 (analytic P_D vs SNR, N = 10³ and 10⁵, P_FA = 0.05, 1 dB worst case) | `axell2010`; `yucek2009`; `tandra2008` | rows 45–46, 54–56 | Computed by `figures/src/fig2-ed-pd-snr.py` (exact chi-square, scipy). Results printed by the script: known noise P_D = 0.5 at −12.75 dB (N = 10³) and −22.85 dB (N = 10⁵); uncertain at −2.75 and −3.25 dB |
| 61 | Noise level estimated dynamically (smallest eigenvalue of the autocorrelation, MUSIC), estimate used for a constant false-alarm rate; iterative threshold for a given P_F | `yucek2009` | p. 121 | |
| 62 | Occupancy campaigns compare measured power with pre-established thresholds (ED) | `reyesmoncayo2018` | p. 35 | |
| 63 | Matched filter optimal for a perfectly known signal | `axell2010` | p. 323 | |
| 64 | MF: short time to reach P_F/P_MD; needs demodulation and perfect knowledge of PU features; receiver per signal type; impractical complexity; large power consumption | `yucek2009` | p. 123 | |
| 65 | Coherent and cyclostationary detectors non-robust with finite coherence times; SNR-wall scaling worse for cyclostationary than coherent | `tandra2008` | ms. p. 1 (abstract) | |
| 66 | Man-made signals periodic (symbol rate, chip rate, channel code, cyclic prefix); second-order cyclostationary = periodic covariance | `axell2010` | p. 323 | |
| 67 | Cyclic correlation instead of PSD; noise WSS vs modulated cyclostationary; can distinguish transmission types | `yucek2009` | p. 122 | |
| 68 | OFDM used by WiFi, WiMAX, LTE, DVB-T; well-known spectral correlation; great interest in its detection | `axell2010` | p. 323 | |
| 69 | Cyclostationary: worse than ED in stationary noise; unaffected by non-stationary interferer noise that defeats ED; features lost in fading; SNR wall under model uncertainty; sensitive to sampling-clock offsets | `yucek2009` | p. 124 | |
| 70 | Cyclostationary detection requires the cyclic frequencies | `zeng2009` | preprint p. 1 | |
| 71 | Eigenvalues: all σ² under H0; δi + σ² under H1; large spread | `axell2010` | p. 324 | |
| 72 | Max/min and average/min eigenvalue tests; random-matrix thresholds for given P_FA | `zeng2009` | preprint p. 1 (abstract) | |
| 73 | No knowledge of signal, channel or noise power; overcome noise uncertainty; better than ideal ED for highly correlated signals | `zeng2009` | preprint p. 1 | |
| 74 | ED optimal for i.i.d. signals, not for correlated | `zeng2009` | preprint p. 2 | |
| 75 | Sample covariance matrix from consecutive samples (smoothing factor), extreme eigenvalues | `zeng2009` | preprint pp. 3–4 (Algorithm 1) | |
| 76 | Known patterns (preambles, midambles, pilots, spreading sequences); correlation with a known copy = waveform-based (coherent) sensing; only for known patterns | `yucek2009` | p. 122 | |
| 77 | Waveform-based beats ED in reliability and convergence time; improves with pattern length; short measurement time; susceptible to synchronization errors | `yucek2009` | p. 122 | |
| 78 | More robust than ED and cyclostationary (coherent processing); needs a priori information | `yucek2009` | pp. 123–124 | |
| 79 | LTE sync signals used as reference to estimate the LO frequency offset of RTL-SDRs | `calvopalomino2017` | PDF p. 1 (abstract) | |

## Section 2.5: Software-defined radio and the RTL-SDR

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 80 | SDR a practical reality through convergence of digital radio and computer software; term SDR also coined by Mitola | `haykin2005` | p. 202; p. 203 (footnote 5) | |
| 81 | CR relies on SDR for reconfigurability | `haykin2005` | p. 202 | |
| 82 | RTL-SDR dongles designed for DVB-T, found by hardware hackers usable as general-purpose SDR | `rtlsdrblog2024` | p. 1 | |
| 83 | RTL-SDR popularity: price below 25 USD, ease of use, open-source software | `calvopalomino2017` | PDF p. 1 (abstract) | |
| 84 | V4: R828D with three inputs; SMA input triplexed into HF, VHF, UHF; notch filters for AM, FM, DAB attenuate a few dB | `rtlsdrblog2024` | p. 1 | |
| T4 | Table 2.4 (V4 specifications) | `rtlsdrblog2024` | p. 1 (specification table) | HF upconverter deliberately omitted (CLAUDE.md) |
| 85 | Architecture not fully documented; generic model with a single LO feeding down-conversion and sampling | `calvopalomino2017` | PDF p. 2, Section 2.1 and Fig. 1 | Basis of Figure 2.7 |
| 86 | Dongle outputs digital I/Q to a host that processes it | `calvopalomino2015` | PDF p. 1 | |
| 87 | SBC + RTL-SDR platform; scanning in 2.4 MHz steps; 2.4 MS/s, 8-bit, 2.4 MHz sensed at a time | `calvopalomino2015` | PDF p. 2 | |
| 88 | Electrosense: crowdsourced low-cost sensors; little knowledge of actual use over time and space; open spectrum data as a service | `rajendran2018` | preprint p. 1 (abstract) | |
| 89 | Electrosense+: decoding on sensors; RTL-SDR v3 with TCXO; no raw I/Q sent to users | `calvopalomino2020` | preprint pp. 2–5 | |
| 90 | Reyes Moncayo: low-cost monitor (Dragonboard 410C, Python, RTL-SDR); channel occupancy; earlier equipment too costly; low-cost solutions needed to monitor constantly in more places | `reyesmoncayo2018` | p. 34 (abstract, Section I) | |
| 91 | Molina-Tenorio: linked affordable SDRs incl. RTL-SDR; wavelets, machine learning, Higuchi fractal dimension; 95% success for SNR > 0 dB | `molinatenorio2021` | p. 1 (abstract); pp. 4–5 (HackRF One, RTL-SDR, LimeSDR Mini) | |
| 92 | Saber used an RTL-SDR to capture real signals | `saber2021` | ms. p. 1 | |
| 93 | Oscillator inaccuracies ~50 ppm → temperature-dependent frequency error of a few kHz | `calvopalomino2015` | PDF p. 2 | |
| 94 | RTL-SDR models with TCXO well within the 1 ppm tolerance | `calvopalomino2017` | PDF p. 1 (abstract) | |
| 95 | Processing chain with DC removal before windowing and FFT | `calvopalomino2015` | PDF p. 2 | |
| 96 | Finite dynamic range limits detection at low SNR | `tandra2005` | as row 60 | |
| F7 | Figure 2.7 (receive chain, functional) | `rtlsdrblog2024`; `calvopalomino2017` | rows 84–85 | "gain" in the tuner box and "CPU only" from CLAUDE.md hardware description |

## Section 2.6: Time–frequency representation

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 97 | STFT definition (eq. 2.7 of the thesis) | `allen1977` | p. 1558, eq. (2.1) | Notation adapted: y for x, h for w |
| 98 | The window selects the portion of the signal analysed | `allen1977` | p. 1558 | |
| 99 | STFT details depend on the window (convolution of transforms); design the window for the desired time and frequency resolution | `allen1977` | p. 1558 | |
| 100 | Long vs short windows: resolution differences (Fig. 2, Hamming N = 400 and 50) | `allen1977` | p. 1559 | |
| 101 | For spectrum displays and signal detection, STFT used as a non-stationary representation; no synthesis needed | `allen1977` | p. 1559 | |
| 102 | Spectral leakage from finite records; windows are weighting functions that reduce leakage | `harris1978` | p. 52 | |
| 103 | Windows affect detectability, resolution, dynamic range, confidence and ease of implementation | `harris1978` | p. 53 | |
| 104 | ENBW definition; larger ENBW lowers processing gain | `harris1978` | p. 54; p. 56 (PG = 1/ENBW) | |
| 105 | Peak sidelobe level and fall-off rate indicate leakage suppression | `harris1978` | p. 57 | |
| 106 | Xu and Tian: STFT spectrograms down-sampled to 64 × 64 | `xu2021`; `tian2022` | Xu p. 4; Tian p. 4 | |
| 107 | Tandiya: spectrogram images from a waterfall display | `tandiya2018` | preprint p. 3 | |
| 108 | Rajendran 2020: patches of consecutive PSD vectors (6 × 300) | `rajendran2020` | preprint p. 6 | |
| F8 | Figure 2.8 (STFT pipeline, schematic) | `allen1977` | rows 97–101 | Image shading illustrative |

## Section 2.7: Autoencoders and anomaly detection

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 109 | Anomaly: an observation that deviates considerably from some concept of normality | `ruff2021` | preprint pp. 1, 3 | |
| 110 | Unsupervised setting: unlabelled data assumed normal ("clean"); in practice noise or contamination | `ruff2021` | preprint p. 6 | |
| 111 | Reconstruction methods reconstruct normal data well and detect anomalies by failing to; among the earliest and most common neural approaches | `ruff2021` | preprint pp. 13–14 (Section V) | |
| 112 | Torío: the normal state of a channel is no transmission | `torio2025` | p. 5 | |
| 113 | PCA: orthogonal basis maximizing variance; eigenvectors of the covariance matrix; equivalent minimum-reconstruction-error projection; default reconstruction baseline | `ruff2021` | preprint p. 15 (Section V-B, eqs. 25–26) | |
| 113a | Autoencoders are neural networks and belong to machine learning, used for many but not all approaches to AI (Section 2.7 lead) | `goodfellow2016` | p. 499 (Chapter 14, first sentence); p. 9, Figure 1.4 (shallow autoencoders as an example of representation learning, a kind of machine learning) | Opened 2026-11-04 |
| 114 | Autoencoder: encoder h = f(x), decoder r = g(h), trained to copy input to output; loss such as MSE | `goodfellow2016` | pp. 499–500 (Section 14 intro; eq. 14.1) | |
| 115 | Undercomplete: code smaller than input; captures the most salient features | `goodfellow2016` | p. 500 (Section 14.1) | |
| 116 | Linear decoder + MSE → same subspace as PCA; non-linear f, g → more powerful non-linear generalization of PCA | `goodfellow2016` | p. 500 | |
| 117 | Linear autoencoder with a bottleneck recovers the PCA subspace | `ruff2021` | preprint p. 15 (Section V-C) | |
| 118 | Autoencoder objective and reconstruction-error score | `ruff2021` | preprint pp. 14–15, eqs. (23), (28) | Score normalized by D in the thesis |
| 119 | Three ideas of convolution: sparse interactions, parameter sharing, equivariant representations | `goodfellow2016` | p. 329 (Section 9.2) | |
| 120 | Dense: m × n parameters; sparse: k × n | `goodfellow2016` | pp. 330–331 | |
| 121 | Parameter sharing; equivariance to translation | `goodfellow2016` | p. 331; p. 334 | |
| 122 | FC AEs ignore 2-D structure, redundancy, global features; CAEs share weights, preserve spatial locality | `masci2011` | p. 54 | Author copy with printed pages 52–59 |
| 123 | 64 × 64 spectrograms used by Xu and Tian; parameter counts | `xu2021`; `tian2022` | p. 4 each | Counts computed: 4096 × 256 + 256 = 1,048,832; 32 × 9 + 32 = 320 |
| 124 | PER score: FC-VAE AUC higher than Conv-VAE for GMSK (0.9761 vs 0.9678) and CHIRP (0.9697 vs 0.9671), lower for QPSK (0.9749 vs 0.9836) and 16QAM (0.9775 vs 0.9891) | `tian2022` | p. 8 (Tables 3 and 4) | |
| 125 | Xu: convolutional VAE AUC 0.016 higher than FC; better at low FPR | `xu2021` | p. 5 | |
| 126 | Baset: shallow conv AE sufficient; more layers raise prediction time without significant gain | `baset2023` | ms. p. 20 (PDF p. 23) | |
| 127 | VAE: recognition model q_φ(z|x) as probabilistic encoder; generative model p_θ(x|z); variational lower bound with reparameterization, SGD; neural recognition model → VAE | `kingma2014` | arXiv pp. 1–3 | Version 11 read (typo fix of v10, 2014) |
| 128 | Tian: MSE gives equal attention, dilutes short or narrow anomalies | `tian2022` | p. 2 | |
| 129 | PER score combines percentiles of pixel errors in background and signal regions | `tian2022` | p. 4 | |
| 130 | Recall 80% → 93% at 5% false-alarm rate for QPSK | `tian2022` | p. 6 (also abstract, p. 1) | |
| 131 | Xu: score weighted towards low-amplitude (noise) regions; anomalies raise the noise floor of the reconstruction | `xu2021` | pp. 1, 3 | |
| 132 | A threshold can be derived from the empirical score distribution | `ruff2021` | preprint p. 5 | |
| 133 | Rajendran 2019: n-sigma rule on reconstruction and discriminator losses, training mean and SD, n chosen empirically | `rajendran2019` | preprint p. 5 | |
| 134 | Baset: 98th percentile of training errors = 2% FPR on training data | `baset2023` | ms. pp. 14, 21 (PDF pp. 17, 24) | |
| 135 | Torío: threshold a chosen number of dB above the least active (reference) channel | `torio2025` | p. 9 (reference channel defined p. 7) | |
| 136 | Gao: lack of precise false-alarm control is a drawback of DL detectors | `gao2019` | preprint p. 2 | |
| F9 | Figure 2.9 (dense vs convolutional AE) | `goodfellow2016`; `masci2011` | rows 119–122 | Sizes illustrative (caption says so) |
| F10 | Figure 2.10 (threshold calibration) | `ruff2021` (concept) | row 132 | Procedure of this study per CLAUDE.md; curve schematic |

## Section 2.8: Related work

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 137 | O'Shea: recurrent predictors (2-layer LSTM + FC) on raw IQ recorded over the air with a USRP B200-mini in FM, GSM, LTE and ISM bands | `oshea2016` | pp. 1–3 | arXiv, cited with pages |
| 138 | Five classes of synthetic anomalies injected; detection from the likelihood of the prediction error | `oshea2016` | pp. 2, 4 | |
| 139 | Neural predictors slightly beat the Kalman predictor, with slightly larger improvements in cellular bands | `oshea2016` | p. 6 | |
| 140 | Tandiya: PredNet on spectrogram and spectral-correlation images of normal behaviour, generated in GNU Radio Companion; ~100% (high SNR) and 90% (medium SNR) detection at FAR 0.15 | `tandiya2018` | preprint pp. 2–4 | Paper calls its results simulation results (p. 1) |
| 141 | SAIFE: adversarial autoencoder on PSD with interpretable features, semi-supervised | `rajendran2019` | preprint p. 1 | |
| 142 | Evaluated on synthetic data, HackRF sweeps in two Belgian cities and Electrosense data; synthetic anomalies injected; random equal split | `rajendran2019` | preprint pp. 3–4, 6 | |
| 143 | Rajendran 2020: 56 days, four Electrosense sensors, low-cost RTL-SDRs at 2.4 MS/s | `rajendran2020` | preprint p. 5 | |
| 144 | Trained on 7 days, tested on 49 from non-adjacent periods | `rajendran2020` | preprint p. 6 | |
| 145 | At constant FAR 5% and 10%, SAIFE better than OSVM, isolation forest, LODA; LODA expansion | `rajendran2020` | preprint pp. 6–7 (Fig. 6); p. 2 (LODA) | |
| 146 | Unsupervised threshold selection difficult even with full anomaly ground truth | `rajendran2020` | preprint p. 6 | |
| 147 | Xu: ISM band 2.4 GHz; single capture 14:41–15:22 (≈41 min); random split; chirps added digitally | `xu2021` | pp. 3–4 | |
| 148 | Tian: laboratory ISM capture; anomalies QPSK, 16QAM, GMSK, chirp transmitted from a second USRP; random split | `tian2022` | p. 4 | |
| 149 | Neither compared with ED; Tian compared PER with MSE, MAE and LOF, OCSVM, EE, IF | `tian2022`; `xu2021` | Tian p. 6; Xu p. 5 (baselines one-class SVM, LOF, deep AE) | |
| 150 | AviSense: signal detector + AE anomaly classification | `baset2023` | ms. p. 3 (PDF p. 6) | |
| 151 | ED refined by spectral flatness; threshold recomputed at run time from blocks judged noise | `baset2023` | ms. pp. 3, 11 | |
| 152 | Different-day threshold: 74.20% of noise blocks passed as signal vs 32.79%–49.54% same day; static threshold impractical | `baset2023` | ms. p. 25 (Table 6); p. 11 | FPR measured against SFM decisions as ground truth (ms. p. 24) |
| 153 | AEs trained on normal data only; normal data recorded at four locations with USRP X310, HackRF One or RTL-SDR | `baset2023` | ms. pp. 13, 17–18 | RTL-SDR used at one location (Table 3) |
| 154 | Test data varied receiver, gain, day, location, time | `baset2023` | ms. p. 18 | |
| 155 | Ran on a laptop CPU; only study reporting inference throughput | `baset2023` | ms. pp. 24 (Table 5), 29 | Keyword search of all 13 papers |
| 156 | Torío: 1-D conv AE trained on synthetic Rayleigh noise; applied to day-long amplitude recordings from a regulator's station, VHF maritime and land mobile bands | `torio2025` | pp. 4–5, 7 | |
| 157 | Power detector with threshold relative to least active channel | `torio2025` | p. 8 | |
| 158 | Practically identical detections; AE more sensitive to low power; power detector tolerates calibration imperfections, no GPU | `torio2025` | pp. 12, 14 (Table 2) | |
| 159 | Torío reported no P_D or P_FA | `torio2025` | Keyword search (probability, false alarm, ROC) | |
| 160 | Gao: DetectNet (Conv + LSTM) on simulated single-carrier signals per RadioML2016.10a | `gao2019` | preprint pp. 1–2 | |
| 161 | SNR wall 3.15 dB lower at P_FA 5.92%, P_D 90% (16QAM); about 5 dB lower for GFSK across sample lengths | `gao2019` | preprint p. 3 (text; Table III) | |
| 162 | ED SNR wall calculated from a formula (Eq. 3, M → ∞), not run on the same frames | `gao2019` | preprint p. 3 | |
| 163 | Zheng: residual CNN on power spectra; better than max–min eigenvalue and entropy detectors at P_FA 0.1, 0.01, 0.001, especially at low SNR | `zheng2020` | preprint pp. 2–5 | |
| 164 | Zheng: simulated training; ACARS over-the-air signals with simulated noise | `zheng2020` | preprint pp. 3–5 | |
| 165 | Peng: CNN on simulated raw IQ; beats ED and near optimal in domain; degrades under mismatch; unsupervised DA still worse than ED; fine-tuning with a few hundred labelled examples beats ED | `peng2020` | preprint pp. 1–4 | ~300 (QPSK→Gaussian), ~100 (Gaussian→QPSK) |
| 166 | Grosu: simulated OFDM over AWGN, flat and frequency-selective fading; adaptive-threshold ED beat the NN on average, especially at low SNR; tailored NN wins only within its SNR range | `grosu2021` | p. 11 (abstract); p. 14 | |
| 167 | Saber: ASK/FSK at 433 MHz with RTL-SDR; P_D 0.994 vs 0.961; P_FA 0.0023 vs 0.0120; ED described only by reference; "CNN" described as fully connected | `saber2021` | ms. pp. 1, 8–9 (Table 2) | |
| 168 | Zeghdoud: CNN–LSTM better than CNN, LSTM and ED at low SNR in simulation; real-time USRP implementation | `zeghdoud2023` | Abstract (publisher page) | Abstract only, as in Chapter 1 claim 36 |
| 169 | Evaluation-practice bullets (data, normal class, ED baselines, operating points, splits, computation) | as rows 137–168 | — | Summaries of the per-study rows. Negative claims ("none reported confidence intervals", "none reported per-frame latency", "none calibrated on held-out noise-only data") rest on keyword searches of the 13 texts on 2 November 2026 |
| T5–T6 | Tables 2.5 and 2.6 | as rows 137–168 | — | Last row is the design of this study (CLAUDE.md, Chapter 1) |

## Section 2.9: Research gap

Synthesis of rows 137–169; no new source claims.

## Sources requested for Chapter 2 but not cited

- **Urkowitz (1967):** not obtained. It is cited only as a secondary source through Kostylev (2002).
- **Chandola et al. (2009), Sakurada & Yairi (2014):** ACM refused the download (HTTP 403). Ruff et al. (2021) covers the same points.
- **Hinton & Salakhutdinov (2006), Jolliffe & Cadima (2016):** the hosts were unreachable or refused the download. Goodfellow et al. (2016) and Ruff et al. (2021) cover the same points.
- **Axell et al. (2012), Kang et al. (2022):** not obtainable. Their entries were moved to `notes/removed.md`.
- **Thakur & Dewangan (2012)** (`papers/admin,+152.pdf`, Mediterranean Journal of Social Sciences): not cited. It is a general overview in a journal outside the field, and Haykin (2005) and Yücek & Arslan (2009) cover the same ground.
- **Calvo-Palomino, TIWDC 2017 and "Radio-signal correlation…" (`papers/TIWDC17.pdf`, `papers/paper (1).pdf`):** not needed for this chapter.
