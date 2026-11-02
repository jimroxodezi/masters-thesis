# Chapter 2 (Literature Review): outline for approval

Status: **draft outline, awaiting approval.** No chapter text has been written. Two **preview figures** were drawn to fix the visual style (Figures 2.1 and 2.2 below).
Governing plan: `notes/proposal.md` §10. Supervisor feedback carried here: cognitive radio background, the five-method sensing taxonomy, and the CNN-vs-dense justification. The energy-detector benchmark is the core contribution, so energy detection gets the fullest treatment.

Target: about 8,000 words, 9 figures, 5 tables. Drafting in two reviewable batches is proposed: §2.1–2.5 first, then §2.6–2.10.

## How the figures are made

- **Drawing:** schematics are drawn as SVG with the project skill `thesis-figures` (its templates and style rules). Sources are kept in `figures/src/`. The skill's exporter turns them into vector PDF in `figures/`, run through `uv run --with cairosvg` so that neither the system Python nor `code/.venv` is modified.
- **Analytic plots:** plots computed from formulas (Figure 2.5) use matplotlib through `uv`, with the generating script kept in `figures/src/`.
- **No synthetic data stands in for results.** Chapter 2 contains no data plots.
- **Style:** serif labels, colour always paired with text labels (so figures survive grayscale printing), and captions that cite the source each figure is adapted from. Frequency diagrams are drawn to scale and say so; illustrative ones say "not to scale".

## Structure

### 2.1 Introduction (~100 words)
Roadmap of the chapter.

### 2.2 The LTE Downlink (~900 words)
- **2.2.1 OFDM resources and frame structure.**
  - 15 kHz subcarriers; a resource block is 12 subcarriers × 7 OFDM symbols with normal cyclic prefix (TS 36.211 Table 6.2.3-1).
  - The FDD frame is 10 ms, with ten 1 ms subframes of two 0.5 ms slots each (TS 36.211 §4.1).
  - → **Figure 2.1** (preview drawn).
- **2.2.2 Channel bandwidths and bands.**
  - 1.4–20 MHz channels ↔ 6–100 resource blocks (TS 36.101 Table 5.6-1) → **Table 2.1**.
  - FDD downlink edges of Bands 3, 5, 7, 8, 20 and 28 (TS 36.101 Table 5.5-1) against the V4's 1.766 GHz limit → **Table 2.2**. Bands 5, 8, 20 and 28 are receivable; Bands 3 and 7 are not. This resolves the proposal's `[CITE: 3GPP TS 36.101]`.
- **2.2.3 Synchronization and reference signals.**
  - There are 504 physical cell identities in 168 groups of 3. The PSS is a Zadoff–Chu sequence on the central 62 subcarriers (plus 5 reserved each side), sent in the last OFDM symbol of slots 0 and 10; the SSS sits in the symbol before it (TS 36.211 §6.11).
  - Cell-specific reference signals are sent in every FDD downlink subframe unless configured otherwise (TS 36.211 §6.10.1), so an active carrier is never silent.
  - These facts explain when the sync-signal correlator can work and what "occupancy" means here → **Figure 2.2** (preview drawn).

### 2.3 Cognitive Radio and Dynamic Spectrum Access (~700 words)
- Origin and definition of cognitive radio; the cognitive cycle; primary and secondary users; spectrum holes in time, frequency and space.
- Sensing as the enabling function; Nigerian measurements as context.
- Sources: Mitola & Maguire (1999), Haykin (2005), Axell et al. (2010), Tandra & Sahai (2008), Popoola & Otuu (2018).
- → **Figure 2.3**.

### 2.4 Classical Spectrum Sensing (~1,800 words)
- **2.4.1 Detection problem.**
  - H0 (noise only) versus H1 (signal plus noise); P_D and P_FA; a Neyman–Pearson threshold set for a target P_FA.
  - Sources: Axell et al. (2010, 2012).
- **2.4.2 Taxonomy.**
  - Energy, matched-filter, cyclostationary, eigenvalue-based and waveform-based detection, plus cooperative sensing (out of scope), with learned detectors shown as a separate, data-driven branch.
  - Sources: Yücek & Arslan (2009), Axell et al. (2012).
  - → **Figure 2.4** (taxonomy tree) and **Table 2.3**: prior knowledge needed, complexity, robustness to noise uncertainty, and fit to a 2.56 MHz slice of an LTE carrier.
- **2.4.3 Energy detection, in full (core).**
  - Test statistic, threshold for a target P_FA, and noise-power estimation.
  - Noise uncertainty and the SNR wall; finite dynamic range.
  - Fixed versus adaptive thresholds; real-time SDR implementations.
  - Sources: Urkowitz (1967), Axell et al. (2010), Tandra & Sahai (2005, 2008), Gao et al. (2019, eq. 3), Grosu & Terebeș (2021), Molina-Tenorio et al. (2021).
  - → **Figure 2.5**.
- **2.4.4 Matched filtering.** Coherent detection needs the waveform and synchronization; SNR-wall behaviour with finite coherence time (Tandra & Sahai, 2008).
- **2.4.5 Cyclostationary feature detection.** For OFDM, the cyclic prefix creates cyclostationarity (Axell et al., 2010, 2012).
- **2.4.6 Eigenvalue-based detection.** Zeng & Liang (2009).
- **2.4.7 Waveform-based detection.** Known preambles, pilots or synchronization sequences (Yücek & Arslan, 2009). The LTE PSS/SSS correlator used in this thesis is an instance.

### 2.5 Software-Defined Radio and the RTL-SDR (~900 words)
- **SDR concept:** Saber et al. (2021), Calvo-Palomino et al. (2015).
- **RTL-SDR Blog V4:** R828D tuner, RTL2832U 8-bit ADC, 1 PPM TCXO, software-switchable bias tee, 500 kHz–1.766 GHz, about 2.56 MHz stable bandwidth (`rtlsdrblog2024`) → **Table 2.4** and **Figure 2.6**.
- **Low-cost monitoring systems:** Calvo-Palomino et al. (2015), Electrosense (Rajendran et al., 2018), Electrosense+ (Calvo-Palomino et al., 2020), Reyes Moncayo et al. (2018), Molina-Tenorio et al. (2021).
- **Impairments:**
  - 8-bit dynamic range (Tandra & Sahai, 2005).
  - Oscillator frequency offset; LTESS-track uses the LTE sync signals to estimate it (Calvo-Palomino et al., 2017).
  - DC component, removed in processing (Calvo-Palomino et al., 2015).
  - Gain settings.

### 2.6 Time–Frequency Representation (~600 words)
- **STFT and windows:** the STFT; window length versus frequency resolution; windows and spectral leakage (Allen & Rabiner, 1977; Harris, 1978).
- **Spectrograms as images:** spectrogram as |STFT|² in dB, used as two-dimensional input to learned models (Tian et al., 2022).
- → **Figure 2.7**.

### 2.7 Autoencoders for Anomaly Detection (~1,300 words)
- **Anomaly detection by reconstruction:** Chandola et al. (2009), Ruff et al. (2021), Sakurada & Yairi (2014).
- **PCA as linear reconstruction:** Jolliffe & Cadima (2016). The autoencoder is its non-linear generalization (Hinton & Salakhutdinov, 2006).
- **Fully connected vs convolutional (supervisor's question):** local receptive fields and weight sharing suit the time–frequency structure of a spectrogram. A short worked parameter count compares a dense layer with a convolutional layer on a spectrogram patch (Goodfellow et al., 2016; Masci et al., 2011) → **Figure 2.8**.
- **VAE and the percentile score:** Kingma & Welling (2014), Tian et al. (2022).
- **Thresholding:** the threshold is set at a quantile of idle validation errors, so the false-alarm rate is fixed (Torío et al., 2025) → **Figure 2.9**.

### 2.8 Related Work, by Theme (~1,500 words)
- **2.8.1 Learned spectrum anomaly detection:**
  - O'Shea et al. (2016), Tandiya et al. (2018)
  - Rajendran et al. (2019; 2020)
  - Xu et al. (2021), Tian et al. (2022), Kang et al. (2022)
  - Baset et al. (2023), Torío et al. (2025)
- **2.8.2 Learned sensing versus energy detection:**
  - Gao et al. (2019), Zheng et al. (2020), Peng et al. (2020)
  - Grosu & Terebeș (2021), Saber et al. (2021), Zeghdoud et al. (2023)
- **2.8.3 Evaluation practice across these studies:** simulated versus real data; accuracy versus fixed-P_FA operating points; random versus session-level splits; fixed versus adaptive energy-detector baselines.
- → **Table 2.5**: comparison matrix of the related work.

### 2.9 Research Gap (~400 words)
- The gap is the *combination* of four things: real LTE downlink captures, a low-cost RTL-SDR, entirely captured training and test data, and a noise-trained autoencoder benchmarked against tuned classical baselines at fixed P_FA with session-level generalization tests.
- Builds on Table 2.5. It replaces the provisional gap statement in Chapter 1, which will be updated to match.

### 2.10 Summary (~150 words)

## Figure plan

| # | Figure | What it shows | Source (for the caption) | Status |
|---|---|---|---|---|
| 2.1 | LTE FDD frame and sync symbols | Frame → subframes (0 and 5 highlighted) → slots → 7 symbols; SSS in symbol 5, PSS in symbol 6 | Adapted from TS 36.211 §4.1, Table 6.2.3-1, §6.11 | **Preview drawn**: `figures/fig2-lte-frame-sync.pdf` |
| 2.2 | Observation window versus an LTE carrier | Example 10 MHz carrier (50 RBs, 9 MHz occupied); central 6 RBs carry PSS/SSS; two windows of at most 2.56 MHz, one centred and one off-centre. To scale in frequency | TS 36.101 Table 5.6-1; TS 36.211 Table 6.2.3-1, §6.11.1.2; RTL-SDR Blog (2024) | **Preview drawn**: `figures/fig2-observation-window.pdf` |
| 2.3 | Spectrum holes | Time–frequency grid of primary-user occupancy, the holes, and opportunistic secondary use. Not to scale | Haykin (2005) | To draw |
| 2.4 | Sensing taxonomy | Tree of the five non-cooperative methods (plus cooperative, plus learned), annotated with the prior knowledge each needs | Yücek & Arslan (2009); Axell et al. (2012) | To draw |
| 2.5 | Energy detector | (a) Block diagram: band-limit, square, integrate, compare with threshold. (b) Analytic curves of P_D against SNR at fixed P_FA, with perfect and uncertain noise power, showing the SNR wall | Urkowitz (1967); Tandra & Sahai (2008) | To draw and compute; formulas to be checked against the source |
| 2.6 | RTL-SDR receive chain | Antenna → R828D tuner → RTL2832U (8-bit ADC, digital down-conversion) → USB → host laptop (IQ samples, Python) | RTL-SDR Blog (2024) | To draw; blocks to be checked against the datasheet |
| 2.7 | STFT to spectrogram | IQ stream → overlapping windows → FFT → magnitude squared → dB → time–frequency patch | Allen & Rabiner (1977) | To draw |
| 2.8 | Fully connected vs convolutional autoencoder | Two rows: flatten → dense → latent → dense → reshape, against conv → latent maps → transposed conv; local versus global connectivity marked | Masci et al. (2011); Goodfellow et al. (2016) | To draw, from the skill template |
| 2.9 | Detection by reconstruction error | Train on idle data → error on idle validation data → threshold at the (1 − P_FA) quantile → decision on test frames | Torío et al. (2025); Sakurada & Yairi (2014) | To draw |

## Table plan

| # | Table | Source |
|---|---|---|
| 2.1 | LTE channel bandwidths, resource blocks and occupied bandwidth (N_RB × 180 kHz) | TS 36.101 Table 5.6-1; TS 36.211 Table 6.2.3-1 |
| 2.2 | FDD bands 3, 5, 7, 8, 20, 28: downlink range and whether the V4 can receive it | TS 36.101 Table 5.5-1; RTL-SDR Blog (2024) |
| 2.3 | Classical sensing methods compared | Yücek & Arslan (2009); Axell et al. (2010, 2012); Tandra & Sahai (2008) |
| 2.4 | RTL-SDR Blog V4 specifications | RTL-SDR Blog (2024) |
| 2.5 | Related-work comparison matrix: signal and band, data (simulated or real), hardware, method, baseline, operating point, generalization test | Studies in §2.8 |

## Sources

Already verified this session: the publication details of every source below (Crossref or publisher), and the two 3GPP specifications (opened: ETSI TS 136 211 V19.3.0, 2026-04; ETSI TS 136 101 V19.6.0). Full texts still have to be read before drafting.

**A. In `papers/`; to be read for Chapter 2.**
- Specification and standards background: Axell et al. 2010; Tandra & Sahai 2005, 2008; Popoola & Otuu 2018.
- Learned sensing and anomaly detection:
  - Gao et al. 2019 (preprint); Grosu & Terebeș 2021; Saber et al. 2021; Torío et al. 2025; Tian et al. 2022.
  - O'Shea et al. 2016 (arXiv only); Tandiya et al. 2018 (ICC Workshops).
  - Rajendran et al. 2019, SAIFE (preprint of `rajendran2019`); Rajendran et al. 2020, *IEEE TCCN* 6(2) (preprint).
  - Xu et al. 2021 (arXiv only); Zheng et al. 2020, *China Communications* 17(2) (preprint); Peng et al. 2020, *IEEE WCL* 9(1) (preprint).
  - Baset et al. 2023 (accepted manuscript).
- Low-cost SDR systems:
  - Electrosense 2018, *IEEE Commun. Mag.* 56(1) (preprint); Electrosense+ 2020, *Computer Networks* 174 (preprint).
  - Calvo-Palomino et al. 2015; LTESS-track 2017 (WiNTECH).
  - Molina-Tenorio et al. 2021, *Sensors* 21(10); Reyes Moncayo et al. 2018, *Scientia et Technica* 23(1).
  - Two further Calvo-Palomino papers in `papers/` (TIWDC 2017; radio-signal correlation), to use only if relevant.

**B. Open access; I can fetch and read them.**
- Axell et al. 2012 (TU Delft repository)
- Zeng & Liang 2009 (arXiv version)
- Kingma & Welling 2014 (arXiv)
- Goodfellow et al. 2016 (free online book)
- Hinton & Salakhutdinov 2006 and Masci et al. 2011 (author copies)
- Jolliffe & Cadima 2016 (open access)
- Ruff et al. 2021 (arXiv)
- The RTL-SDR datasheet

**C. Paywalled; please add the PDFs to `papers/` if your library gives you access.**
- **Needed:** Yücek & Arslan 2009 (taxonomy, including waveform-based sensing), Haykin 2005 (spectrum holes), Mitola & Maguire 1999 (origin of cognitive radio), Urkowitz 1967 (energy detection), and Kang et al. 2022 (already in the bib).
- **Preferred:** Allen & Rabiner 1977, Harris 1978, Chandola et al. 2009, Sakurada & Yairi 2014.
- **If unavailable:**
  - Group C sources are cited only after reading. Where one can't be obtained, an open source replaces it: Axell et al. 2012 for the taxonomy, Ruff et al. 2021 for anomaly detection, and Goodfellow et al. 2016 for neural networks.
  - Haykin's spectrum-hole definition has no good open substitute.

## Findings that affect other chapters (decisions needed)

1. **An active LTE carrier is never silent.** Reference signals are sent in every FDD downlink subframe (TS 36.211 §6.10.1). So:
   - (a) The proposal's Class B ("idle", §8.2) cannot use "quiet periods" of an active carrier. Idle frames must come from guard regions, unused channels or the terminated input.
   - (b) "Occupancy" in this thesis means the presence of an LTE carrier in the observed slice, not its traffic load.
   - I propose adding one sentence to the Chapter 1 scope and correcting §8.2 of the proposal.
2. **Band edges are now sourced** (TS 36.101 Table 5.5-1). This closes the proposal's `[CITE: 3GPP TS 36.101]`.
3. **The gap statement** in Chapter 1 will be updated to match §2.9 once it is drafted.
