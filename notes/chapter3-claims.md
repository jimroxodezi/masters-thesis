# Chapter 3: claims and sources

Every cited claim in `chapters/chapter3.tex`. Conventions are as in `notes/chapter2-claims.md`. Settings that are design decisions (not claims about sources) are recorded in `notes/method-design.md`, not here. Checked on 2 November 2026.

| # | Claim (as used) | Key | Where in source | Notes |
|---|---|---|---|---|
| 1 | V4: R828D tuner, RTL2832U 8-bit, 500 kHz–1.766 GHz, about 2.56 MHz stable bandwidth, 1 ppm TCXO, software-switchable bias tee | `rtlsdrblog2024` | p. 1 | As in Chapter 2, Table 2.4 |
| 2 | PyTorch used for the neural networks | `paszke2019` | arXiv abstract | Software citation |
| 3 | scikit-learn used for PCA | `pedregosa2011` | JMLR abstract page | Software citation |
| 4 | Candidate ranges are those of Table 2.2 within the tuning range | `3gpp2026b`; `rtlsdrblog2024` | Table 5.5-1; p. 1 | Via Chapter 2 |
| 5 | LTE time unit T_s = 1/(15000 × 2048) s, so 1/T_s = 30.72 MHz | `3gpp2026a` | Clause 4 | 1.92 MHz = 30.72/16 (arithmetic) |
| 6 | Useful OFDM symbol 2048 T_s; normal CP 160 T_s (l = 0) or 144 T_s (l = 1…6), i.e. 128 and 10 or 9 samples at 1.92 MS/s | `3gpp2026a` | Clause 6.12, Table 6.12-1 | Samples = T_s units / 16 |
| 7 | Subcarrier spacing 15 kHz = 128-point FFT bin spacing at 1.92 MS/s | `3gpp2026a` | Table 6.2.3-1 | 1.92 MHz/128 = 15 kHz |
| 8 | Window 1.92 MHz within the stable bandwidth (about 2.56 MHz) | `rtlsdrblog2024` | p. 1 | |
| 9 | No frequency correction needed: TCXO tolerance 1 ppm (≤ about 1 kHz at the candidate bands) | `rtlsdrblog2024` | p. 1 | 1 ppm × 960 MHz = 0.96 kHz (arithmetic) |
| 10 | Every frame of an active carrier is occupied, because CRS are sent in every downlink subframe | `3gpp2026a` | Clause 6.10.1 | Justifies capture-level labels |
| 11 | DC offset removed by subtracting the frame mean (DC removal in RTL-SDR processing) | `calvopalomino2015` | PDF p. 2 | Their pipeline includes DC removal |
| 12 | Hann window chosen to reduce spectral leakage | `harris1978` | p. 52 (windows reduce leakage) | |
| 13 | An undercomplete autoencoder must capture the most salient structure | `goodfellow2016` | p. 500 | |
| 14 | VAE: Gaussian code with mean and log-variance; KL term | `kingma2014` | arXiv pp. 1–3 | |
| 15 | PER score settings: γ = 3 pooling kernels, background weight 2, step threshold 0.05, 90th and 99th percentiles | `tian2022` | p. 4 | Interpretation choices stated in the text |
| 16 | PCA reconstruction error as the linear baseline | `ruff2021` | preprint p. 15 | |
| 17 | CFAR principle: estimate the noise level and set the threshold relative to it | `yucek2009` | p. 121 | |
| 18 | Theoretical chi-square threshold of the energy detector | — | Chapter 2, eq. (2.5) (`axell2010`, pp. 322–323) | |
| 19 | PSS root indices 25, 29, 34 | `3gpp2026a` | Table 6.11.1.1-1 | |
| 20 | PSS sequence d_u(n), length 62 | `3gpp2026a` | Clause 6.11.1.1 | |
| 21 | PSS mapped to subcarriers either side of the unused DC: n = 0…30 → −31…−1, n = 31…61 → +1…+31 | `3gpp2026a` | Clause 6.11.1.2 (k = n − 31 + N_RB·N_sc/2, DC not used) | The DC skip follows from the OFDM mapping in clause 6.12 |
| 22 | Adam optimizer | `kingma2015` | arXiv abstract | |
| 23 | The bootstrap estimates the sampling distribution of a statistic from the observed data | `efron1979` | Abstract (Crossref) | Resampling sessions with replacement is this study's design, not attributed to Efron |
| 24 | SNR wall of the energy detector at 1 dB uncertainty: −3.3 dB | `tandra2008` | ms. p. 8, eq. (2) | Basis of the H1 low-SNR definition |
| F | Figure 3.2 (window placement, to scale) | `3gpp2026b`; `3gpp2026a` | Table 5.6-1; clause 6.11 | f_off = (0.54 + 4.5)/2 = 2.52 MHz for a 10 MHz carrier |
| F | Figure 3.4 (frame geometry, PSS/SSS timing) | `3gpp2026a` | Clauses 4.1, 6.11, 6.12 | |
