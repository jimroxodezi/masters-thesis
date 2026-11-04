# Chapter 1: claims and sources

Every cited claim in `chapters/chapter1.tex`, with the place in the source where it appears. Page numbers are printed page numbers unless marked "PDF p." (page of the copy in `papers/`) or "ms." (manuscript copy). All passages were re-checked on 2 November 2026.

## Section 1.1.1: Spectrum use and its measurement

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 1 | Spectrum is the resource wireless communication depends on; traditionally regulated by long-term assignment to licensed users, aiming at interference-free operation | `popoola2018` | p. 948, §1 | |
| 2 | Measurements show much assigned spectrum idle much of the time; use varies with time, frequency and location | `axell2010`; `popoola2018` | Axell p. 322 (§I); Popoola p. 948 (abstract), p. 960 (§5) | |
| 3 | Occupancy measured over 80–2200 MHz at Abuja, Lafia, Lokoja, Gusau and Port Harcourt; average below 10% | `popoola2018` | p. 948 (abstract); p. 951 (§3.1, sites); p. 960 (§5, "average occupancy less than 10.00%") | |
| 4 | Occupancy measurement is the main quantitative tool giving regulators, researchers, engineers and policymakers usage information | `popoola2018` | p. 949, §2 | Quoted page given in text |
| 5 | ITU: IMT-Advanced considered "4G"; LTE-Advanced to be designated IMT-Advanced; the term "4G" may also be applied to LTE | `itu2010` | Press release, 6 Dec 2010, Geneva | Opened 2 Nov 2026 |
| 6 | Aug 2026 shares of subscribers by generation: 4G 54.77%, 2G 36.05%, 3G 4.27%, 5G 4.92% | `ncc2026` | "Market Share by Generation" table, column Aug'26 | Page last updated 12 Oct 2026; opened 2 Nov 2026. The page says "4G", not "LTE"; row 5 provides the link |
| 7 | LTE is one of several systems using OFDM signalling | `axell2010` | p. 323 | 3GPP TS 36.211 to be added in Chapter 2 |

## Section 1.1.2: Why downlink occupancy matters

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 8 | Spectrum authorities' functions include monitoring emissions to identify, locate and eliminate harmful interference and irregularities | `torio2025` | p. 2, §1.1 (list of authorities' functions) | |
| 9 | Surveillance is needed to identify unauthorized emissions and mitigate interference; manual surveillance is time-consuming, resource-intensive, error-prone over large spectrum in real time | `torio2025` | p. 1, §1 | |
| 10 | Measurements mainly by government agencies with bulky, expensive equipment; does not scale; poor awareness of actual use | `calvopalomino2015` | PDF p. 1 (abstract) | |
| 11 | Risk of interference and malicious use has grown | `tian2022` | p. 1, §1 | |
| 12 | Illegal repeaters frequently disrupt operators' cell planning, causing poor coverage and dropouts | `tian2022` | p. 1, §1 | A statement in the authors' introduction; the only operator-side support found (see Open items) |
| 13 | Occupancy measurements give engineers and regulators information on actual use | `popoola2018` | p. 949, §2 | |
| 14 | Use varies with time, frequency and location | `popoola2018` | p. 948 (abstract) | |
| 15 | Low-cost sensors proposed for continuous, large-scale measurement | `calvopalomino2015` | PDF p. 1 (§1: continuous measurements; wide-scale monitoring) | |
| 16 | Licensed spectrum is intended for its owners; cognitive radio reuses it in an unlicensed manner, motivated by measurements showing unused spectrum | `axell2010` | p. 322, §I | |
| 17 | Secondary users would use idle licensed spectrum opportunistically; which portions are idle must first be established | `popoola2018` | p. 960, §5 | |
| 18 | Primary users must be detected to avoid harmful interference | `tandra2008` | Abstract (ms. p. 1) | |
| 19 | Low occupancy suggests a hopeful future for dynamic spectrum access in Nigeria | `popoola2018` | p. 948 (abstract); p. 960 | |

## Section 1.1.3: Low-cost SDR as a sensor

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 20 | In an SDR, receiver functions traditionally in hardware (mixing, filtering, demodulation, detection) are implemented in software | `saber2021` | PDF p. 6 (manuscript copy; published pp. 930–941) | |
| 21 | RTL-SDR-compatible dongle outputs digital I/Q samples that a host computer processes; sensor covered 24–1766 MHz at under US$100, 50–500 times cheaper than professional equipment | `calvopalomino2015` | PDF p. 1 (abstract; §1; §2.1) | Their host was a single-board computer; Chapter 1 says only "host computer" |
| 22 | RTL-SDR Blog V4: 500 kHz–1.766 GHz, 8-bit ADC, about 2.56 MHz stable bandwidth | `rtlsdrblog2024` | Datasheet V1.0 | Datasheet read (proposal §12) |
| 23 | Only bands below about 1.77 GHz can be received; only a slice a few MHz wide at once; limited resolution and dynamic range | — | Follows from row 22 | Author's inference |

## Section 1.1.4: Learned detection by reconstruction error

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 24 | An autoencoder compresses its input into a smaller latent representation and reconstructs it | `torio2025` | pp. 2–3, §1.2 | |
| 25 | Torío et al. trained on synthetic Rayleigh noise, treated "no transmission" as the normal state, and took large reconstruction error as a transmission | `torio2025` | p. 5, §2.2 (also p. 7) | Closest prior work to the thesis idea |
| 26 | Tian et al. modelled spectrograms with a VAE and a percentile-based score of reconstruction error | `tian2022` | p. 1 (abstract; §1) | |

| G1 | Learned (machine-learning) detectors are AI-based: machine learning is used for many, though not all, approaches to AI | `goodfellow2016` | p. 9, Figure 1.4 caption ("deep learning is a kind of representation learning, which is in turn a kind of machine learning, which is used for many but not all approaches to AI") | Online book (deeplearningbook.org, Chapter 1), opened 2026-11-04 |
| G2 | An autoencoder is a neural network trained to copy its input to its output (Section 1.3, aim) | `goodfellow2016` | p. 499, first sentence of Chapter 14 | Online book, opened 2026-11-04 |

## Section 1.2: Statement of the problem

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 27 | Energy detection is a good choice for blind detection | `gao2019` | Abstract (arXiv p. 1) | |
| 28 | The energy detector compares received energy over an interval with a threshold; detects any signal; needs no signal knowledge | `axell2010` | p. 322 (§II), p. 323 | |
| 29 | Threshold needs the noise variance; imperfect knowledge gives a wrong threshold and quick deterioration | `axell2010` | p. 323 | |
| 30 | SNR wall: without modulation knowledge, no robustness below some SNR however long the observation; any such detector is essentially as non-robust as the radiometer | `tandra2008` | Abstract (ms. p. 1) | |
| 31 | Practical noise aggregates thermal noise, leakage through non-linearity, aliasing from imperfect filters and distant interference; residual uncertainty after run-time calibration | `tandra2008` | ms. p. 6 (§III) | Published-version page not checked |
| 32 | Finite dynamic range gives an SNR wall that no detector can overcome, for a primary signal without a deterministic component | `tandra2005` | Abstract; §IV, Theorem 2 | Proved for BPSK; the paper says it generalizes when no deterministic component is present. LTE has known components, which is why the sync correlator is included |
| 33 | Feature detectors exploit known structure of the transmitted signal | `axell2010` | p. 322 (abstract) | |
| 34 | Coherent and cyclostationary feature detectors are non-robust when coherence times are finite | `tandra2008` | Abstract | |
| 35 | Gao et al.: SNR wall 3.15 dB lower than the energy detector's at one operating point; simulated single-carrier signals, not OFDM | `gao2019` | arXiv p. 3, §III (Pf = 5.92%, Pd = 90%: −8.5 dB vs −5.35 dB); p. 2, §II-B (signals generated as per RadioML2016.10a; noise negatives) | **Correction:** the proposal says "from a public dataset". The signals were *generated following* RadioML2016.10a. The published version is still to be checked |
| 36 | Zeghdoud et al.: in numerical simulations CNN-LSTM outperformed CNN, LSTM and energy detection, with high detection at low SNR; implemented for real-time sensing on a USRP | `zeghdoud2023` | Abstract (publisher page, eurekaselect.com/218915, opened 2 Nov 2026) | **Correction:** the comparison with energy detection was from simulation. The USRP/GNU Radio work demonstrated real-time detection (FM, GSM, OFDM scenarios). Abstract only |
| 37 | Grosu & Terebeș: simulated OFDM; dynamic-threshold energy detector beat the NN on average, especially at low SNR; NN better only in scenarios matching its constrained training data | `grosu2021` | p. 11 (abstract) | Full text read (CLAUDE.md) |
| 38 | All three comparisons used simulated signals | — | Rows 35–37 | Author's synthesis |
| 39 | Torío et al.: amplitude-only recordings of VHF land and maritime mobile channels with highly calibrated instruments | `torio2025` | p. 4, §2.1 | |
| 40 | Both methods effective; autoencoder more sensitive to low-power transmissions | `torio2025` | p. 12 | |
| 41 | Analysis afterwards on recorded files, not real time; deep learning needed more demanding computing (GPU) | `torio2025` | p. 14 (§4); p. 2 | |
| 42 | Saber et al.: RTL-SDR captures of ASK/FSK signals at 433 MHz from an Arduino Uno and wireless transmitter; supervised CNN better than energy detection, ANN and SVM | `saber2021` | Abstract | |
| 43 | The reviewed studies do not establish the answers; a fixed-normalization autoencoder *may* react to a raised noise floor like a fixed threshold | — | — | Author's argument, hedged; tested by H2 |

## Sections 1.3–1.7

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 44 | Stable bandwidth about 2.56 MHz | `rtlsdrblog2024` | Datasheet | |
| 45 | RTL-SDR-based sensors cost a small fraction of professional equipment | `calvopalomino2015` | PDF p. 1 | |

Aims, research questions, hypotheses, objectives, scope and contributions are design statements taken from the approved outline and `notes/proposal.md`. Host laptop details are from CLAUDE.md. None of them is a result.

## Sources considered and not used

- **Najashi, Feng & Kadri (2013)**, *IJCSI* 10(1). It was dropped because the journal was reported as listed on Beall's list of potentially predatory publishers and is the subject of a 2020 investigation hosted by Retraction Watch. Popoola & Otuu (2018) covers the same claims.
- **`yucek2009`, `axell2012`**: no PDFs in `papers/`, so they were not read. Chapter 1 uses `axell2010` and `tandra2008` instead. Both remain in the bib for Chapter 2.

## Open items

- **Operator-side support for network optimization:** row 12 is the only direct source. A stronger operator or regulator source would help.
- **Published versions:** `gao2019` (published version not checked) and `tandra2008` (manuscript page numbers).
- **Primary sources for Chapter 2:** 3GPP specifications for the LTE downlink structure (OFDM, synchronization signals), referred to in Chapter 1 only by forward reference.
