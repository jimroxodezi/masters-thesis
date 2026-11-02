# Chapter 1 (Introduction): outline for approval

Status: **approved 2 November 2026 and drafted** in `chapters/chapter1.tex` (claims in `notes/chapter1-claims.md`). Deviations made during drafting are listed at the end of this file.
Governing plan: `notes/proposal.md` (the revised proposal; CLAUDE.md still calls it `notes/proposal-revised.md`). Bibliography: `bib/references.bib`.

## Decisions applied (confirmed 2 November 2026)

- **Stretch tier, all included:** PCA reconstruction detector; fully connected vs convolutional autoencoder ablation; cross-session/day/gain robustness tests with session-level bootstrap CIs; CPU latency and throughput on the host laptop.
- **Optional tier:** LTE synchronization-signal correlator and VAE with percentile score **included**. CFAR-style detector **dropped**.
- **H2:** a **fixed-threshold energy detector** is added (the same detector with its noise estimate frozen at the validation value), so H2 has a fixed-threshold comparator.
- Band: still undecided. Chapter 1 stays band-agnostic and uses a visible placeholder (`[BAND TBC]`) where the band must be named.

## Changes from the revised proposal that this outline introduces

Please approve or reject each one. If approved, I will also update `notes/proposal.md` (§5–§7), moving replaced text to `notes/removed.md`, so the proposal and the thesis do not disagree.

1. **CFAR removed** from RQ2 and the objectives.
2. **H2 reworded** against the fixed-threshold energy detector (wording in §1.3 below).
3. **New RQ5** for the architecture and score ablations (CNN vs dense; MSE vs VAE percentile score). RQ1–RQ4 keep their proposal numbers, so references such as "Robustness tests (RQ3)" in the proposal stay valid.
4. **Scope wording:** "only power and spectral structure are analysed" no longer holds once the sync correlator is in. New wording: no demodulation or decoding of LTE content; the correlator detects the presence of known synchronization sequences only. "Other deep models (VAE, GAN, diffusion)" becomes "GAN, diffusion and other deep models", since the VAE is now in.
5. **Problem statement:** the proposal's "a learned noise model could in principle adapt to this" has no source. It becomes an open question that H2 tests. An autoencoder whose normalization is fixed from training data may be just as sensitive to noise-floor drift as a fixed threshold.
6. **Gap statement narrowed.** Torío et al. (2025) already trained an autoencoder on noise only and flagged high reconstruction error as "transmission present". Saber et al. already compared a CNN with energy detection on real RTL-SDR captures. The gap is therefore the *combination* described in §1.2 below, not the idea itself.
7. **"US$30-class RTL-SDR" removed** (no source). It is replaced by Calvo-Palomino et al. (2015): an RTL-SDR-based sensor at under US$100 per device, 50–500 times cheaper than professional equipment.

## Structure, content and sources

Target length is about 3,200 words (roughly 9–11 pages at 12 pt with 1.5 spacing). Page numbers below are printed page numbers unless marked "PDF p." or "ms." (manuscript copy).

### 1.1 Background and Motivation (~1,300 words)

**1.1.1 Spectrum use and its measurement**
- Radio spectrum is a managed public resource; traditional regulation assigns bands on a long-term basis to ensure interference-free operation. `popoola2018` p. 948.
- Occupancy measurements worldwide, and in Nigeria, show allocated spectrum is underused and that use varies in time, frequency and space. Popoola & Otuu measured 80–2200 MHz in five Nigerian cities and found occupancy below 10% (`popoola2018` p. 948). Najashi et al. made an indoor measurement at 700 MHz–2.5 GHz in Abuja (`najashi2013` p. 394).
- Occupancy measurement is the main quantitative tool that gives regulators, researchers, engineers and policymakers information on usage patterns. `popoola2018` p. 949.
- LTE remains a major carrier of mobile broadband. **`[CITE]` open** (see Open items). LTE uses OFDM signalling (`axell2010` p. 323). LTE downlink structure is left to Chapter 2.

**1.1.2 Why knowing downlink occupancy matters** (supervisor's "why"; three short paragraphs)
- *Interference management.* Effective spectrum surveillance is needed to identify unauthorized emissions and mitigate interference. Manual surveillance is time-consuming, resource-intensive and error-prone across wide spectrum in real time (`torio2025` p. 1). Spectrum authorities' functions include identifying, locating and eliminating harmful interference (`torio2025` p. 2). Measurements are still mostly made by agencies using bulky, expensive equipment, which gives poor situational awareness (`calvopalomino2015` PDF p. 1). Spectrum interference and malicious use are increasing (`tian2022` p. 1).
- *Network optimization.* Illegal repeaters can disrupt mobile operators' cell planning, causing poor coverage and dropped connections (`tian2022` p. 1). Usage patterns vary in time and place (`popoola2018` p. 948), so planning needs measured rather than assumed occupancy (`popoola2018` p. 949). **Caveat:** this support is indirect. No source in `papers/` says operators use occupancy statistics for network optimization; an operator-side source would strengthen it (see Open items).
- *Dynamic spectrum access planning.* Cognitive radio is motivated by measurements that show unused resources in frequency, time and space (`axell2010` p. 322). Secondary users must detect primary users reliably to avoid harmful interference (`tandra2008` abstract, ms. p. 1). Occupancy measurements identify bands suited to cognitive radio (`najashi2013` p. 394); low occupancy in Nigeria suggests scope for dynamic spectrum access (`popoola2018` p. 948). Cognitive radio itself is reviewed in Chapter 2.

**1.1.3 Low-cost software-defined radio as a sensor**
- An RTL-SDR-based monitoring sensor costs under US$100 and is 50–500 times cheaper than professional equipment; it covers 24–1766 MHz (`calvopalomino2015` PDF p. 1).
- The RTL-SDR Blog V4 used here: 500 kHz–1.766 GHz, 8-bit ADC, about 2.56 MHz stable bandwidth (`rtlsdrblog2024`). This bounds which LTE bands can be observed and how wide a slice is seen.
- The trade-off is limited dynamic range and resolution. This sets up the problem statement.

**1.1.4 Learned detection by reconstruction error**
- Idea: train a model on noise-only data. Small reconstruction error means only noise is present; large error means a transmission is present. `torio2025` p. 5 uses exactly this logic, with an autoencoder trained on synthetic Rayleigh-distributed noise.
- Other unsupervised spectrum anomaly detectors exist, e.g. a VAE with a percentile score on spectrograms (`tian2022` p. 1). A full review is in Chapter 2.
- In this thesis the "normal" class is the receiver's own idle spectrum captured with the RTL-SDR, so an LTE carrier is the anomaly.

### 1.2 Statement of the Problem (~600 words)

1. Energy detection is the standard low-cost occupancy detector. It is universal and needs no knowledge of the signal (`axell2010` p. 323).
2. But its threshold needs the noise variance. With imperfect noise knowledge the threshold is wrong, performance deteriorates quickly, and below an "SNR wall" detection becomes impossible (`axell2010` p. 323; `tandra2008` abstract, ms. pp. 1, 8).
3. Noise in a practical receiver aggregates thermal noise, leakage through receiver non-linearity, aliasing from imperfect front-end filters and distant interference, so some uncertainty remains even after calibration (`tandra2008` ms. p. 6).
4. A low-cost receiver has an 8-bit ADC (`rtlsdrblog2024`). With finite dynamic range, any detector can be rendered useless at low enough SNR under moderate noise uncertainty (`tandra2005` abstract).
5. Learned detectors are reported to beat energy detection at low SNR: `gao2019` (single-carrier signals from a public dataset, **not OFDM**) and `zeghdoud2023` (CNN-LSTM on a USRP; abstract only). But `grosu2021` found the reverse on **simulated** OFDM: a dynamic-threshold energy detector was better on average, especially at low SNR (p. 11). The evidence is mixed.
6. Evidence from real, low-cost captures is thin. `torio2025` used calibrated monitoring-station recordings of VHF channels, trained on synthetic noise, analysed offline and needed a GPU (p. 2). `saber20xx` compared a supervised CNN with energy detection on RTL-SDR captures of self-generated 433 MHz ASK/FSK signals.
7. Problem: it is not established whether an autoencoder trained only on idle RTL-SDR captures detects LTE downlink occupancy better than carefully tuned classical detectors at fixed false-alarm rates. Nor is it known whether either kind of detector keeps its calibrated operating point when the noise floor changes between sessions. *Provisional gap statement, to re-check after the Chapter 2 literature search.*

### 1.3 Aim, Research Questions and Hypotheses (~450 words)

**Aim** (proposal §5, unchanged): to design, implement and rigorously evaluate an autoencoder-based occupancy detector for an LTE downlink band using only RTL-SDR captures, and to compare it with classical baselines under controlled and uncontrolled conditions.

**Research questions** (proposed wording)
- **RQ1:** At fixed false-alarm rates, how does the convolutional autoencoder's detection probability compare with fixed-threshold and adaptive-threshold energy detection across SNR?
- **RQ2:** Does the autoencoder add detection value beyond a linear (PCA) reconstruction detector and an LTE synchronization-signal correlator that exploits known signal structure?
- **RQ3:** How do the methods generalize across capture sessions, days and receiver gain settings (and locations, where captured)?
- **RQ4:** What are the per-frame latency and throughput of each method on the host laptop's CPU, and is real-time operation at the capture sample rate feasible?
- **RQ5 (new):** How do the autoencoder's architecture (convolutional vs fully connected) and anomaly score (reconstruction MSE vs the VAE percentile score) affect detection?

**Hypotheses** (tested, not assumed)
- **H1:** At low SNR and the same false-alarm rate, the convolutional autoencoder achieves a higher detection probability than the adaptive-threshold energy detector. "Low SNR" will be defined in Chapter 3.
- **H2 (reworded):** When a test session's noise floor differs from that of the calibration sessions, the autoencoder's realised false-alarm rate and detection probability depart less from their calibrated values than those of the fixed-threshold energy detector. The adaptive detector shows how much adaptation alone recovers.
- Text will state that rejecting H1 and/or H2 is a valid result and will be reported as such.

**Table 1.1 (proposed):** map of RQ → hypothesis → objective(s) → chapter section where it is answered. It needs only `booktabs`.

### 1.4 Objectives (~350 words)

1. Survey candidate LTE downlink bands within the V4's tuning range at the study site and select one confirmed active band `[BAND TBC]`.
2. Design a capture protocol and collect a labelled multi-session dataset in three classes (terminated-input receiver noise; antenna-connected idle spectrum; LTE-active spectrum) with per-capture metadata, plus a controlled-SNR digital-injection set with exact labels.
3. Implement preprocessing (STFT log-magnitude spectrograms; normalization statistics from training data only) and splits by capture session.
4. Train on idle data only: a convolutional autoencoder (primary), a fully connected autoencoder (architecture ablation) and a VAE with percentile scoring (score ablation).
5. Implement baselines: fixed-threshold and adaptive-threshold energy detectors, a PCA reconstruction detector and an LTE synchronization-signal correlator. All are tuned on the same idle validation data.
6. Evaluate all methods on identical frames at fixed false-alarm rates (planned: 1% and 5%). Report ROC/AUC, confusion-matrix metrics, SNR-stratified results, cross-session/day/gain tests, seed variability and session-level bootstrap CIs.
7. Measure per-frame latency, throughput and real-time factor on the host laptop's CPU, and demonstrate a real-time occupancy-logging pipeline.

### 1.5 Scope and Delimitations (~300 words)

- **Included:** one LTE downlink band within the V4's tuning range `[BAND TBC]`; observed as a slice a few megahertz wide (exact sample rate to come from the final capture configuration); captures in Benin City (confirmed by the author, 2 November 2026); one RTL-SDR Blog V4; all processing on the Dell Latitude 5490 laptop (CPU only); binary per-frame occupancy; only data captured by the author.
- **Excluded:** demodulation or decoding of LTE content (the sync correlator only detects known synchronization sequences); uplink; any transmission or active probing; cooperative multi-sensor sensing; GAN, diffusion and other deep models; CFAR, cyclostationary, eigenvalue-based and matched-filter detectors (reviewed in Chapter 2, not implemented); network performance metrics; operator-level or regulatory conclusions (implications discussed conceptually); embedded deployment; Band 3 and higher bands.
- Findings apply to the band, receiver and site studied. A brief note says reception is passive and no content or user data is recorded (detail in Chapter 3).

### 1.6 Contributions of the Study (~250 words)

1. A documented, labelled, multi-session dataset of real RTL-SDR captures in an LTE downlink band, with idle, active and controlled-SNR subsets. Release is **subject to supervisor approval** (open item).
2. A controlled benchmark of noise-trained autoencoders against fixed and adaptive energy detection, a PCA detector and an LTE sync correlator, on identical frames at fixed false-alarm operating points.
3. An analysis of generalization across sessions, days and gain settings on low-cost hardware, with session-level uncertainty estimates.
4. An implementation with measured CPU latency and throughput on a commodity laptop. Whether the code is released openly is also an open item.

Contributions are stated as design contributions only, with no results. This section will be revised once Chapter 4 exists.

### 1.7 Organisation of the Thesis (~150 words)

One paragraph per chapter (2–5), following CLAUDE.md's five-chapter plan.

## Sources for Chapter 1

| Key (proposed) | Source | In `papers/` | In bib | What has been checked | Before drafting |
|---|---|---|---|---|---|
| `popoola2018` | Popoola & Otuu (2018), *Eskişehir Tech. Univ. J. Sci. Technol. A*, 19(4), 948–962, doi 10.18038/aubtda.425027 | yes (file misnamed `.docx`; it is a PDF) | no | abstract; p. 949 | read in full; verify metadata via DOI; add entry |
| `najashi2013` | Najashi, Feng & Kadri (2013), *IJCSI*, 10(1, No. 1), 394–399 | yes | no | abstract | read in full; verify metadata (3rd author's name prints oddly); add entry |
| `axell2010` | Axell, Leus & Larsson (2010), CIP 2010, pp. 322–327, doi 10.1109/CIP.2010.5604136 | yes | no | pp. 322–323 passages | read in full; verify DOI; add entry. Distinct from `axell2012`. |
| `tandra2008` | Tandra & Sahai, "SNR walls for signal detection", *IEEE JSTSP* 2(1), 4–17, 2008 | yes (manuscript copy) | no | abstract; ms. pp. 6, 8. Venue cross-checked in two reference lists (Axell 2010; Gao 2019 preprint) | verify DOI and pages from the publisher; add entry |
| `tandra2005` | Tandra & Sahai, "Fundamental limits on detection in low SNR under noise uncertainty", WirelessCom 2005 | yes | no | abstract | verify venue, pages and DOI; add entry |
| `calvopalomino2015` | Calvo-Palomino, Pfammatter, Giustiniano & Lenders (2015), IPSN '15 demo abstract, ACM | yes | no | PDF p. 1 | verify pages and DOI; add entry |
| `saber20xx` | Saber, Chehri, El Rharras, Saadane & Wahbi, "A cognitive radio spectrum sensing implementation based on deep learning and real signals" | yes | no | abstract | **find venue and year**; read; add entry |
| `torio2025` | in bib | yes | yes | read in full (CLAUDE.md); pp. 1, 2, 5 re-checked | none |
| `tian2022` | in bib | yes (full PDF now present) | yes | abstract; p. 1 | read in full (needed for the VAE ablation anyway) |
| `gao2019`, `grosu2021` | in bib | yes | yes | read (CLAUDE.md) | check the published version of `gao2019` |
| `zeghdoud2023` | in bib | no | yes | abstract only | use only at abstract level |
| `rtlsdrblog2024` | in bib | URL | yes | datasheet read | none |
| `yucek2009`, `axell2012` | in bib | **no** | yes | metadata only | not used in Chapter 1 unless the PDFs are added and read. `axell2010` and `tandra2008` cover the Chapter 1 claims. |

## Open items

1. **LTE's role in mobile broadband** `[CITE]`. Plan: during drafting, look for Nigerian Communications Commission (NCC) subscription data by technology (preferred), or GSMA *Mobile Economy* data, and cite only what I open. If nothing verifiable is found, the sentence is cut. Tell me if you already have a source.
2. **Operator-side source for network optimization** (see 1.1.2). Same plan as item 1; otherwise the paragraph keeps its indirect support.
3. **Study site:** Benin City, confirmed by the author on 2 November 2026.
4. **Operating points:** confirm 1% and 5% false-alarm rates.
5. **Dataset and code release:** needs supervisor approval (proposal §13, item 4).
6. **Gap statement:** provisional until the Chapter 2 literature search is done.
7. **University format:** do you have University of Benin SPGS formatting rules or a template? (See Drafting setup.)

## Drafting setup (applies once the outline is approved)

`main.tex` and `preamble.tex` are empty, so the first draft needs a minimal build. Defaults, unless SPGS rules say otherwise:
- `report` class, 12 pt, A4, 1.5 line spacing, default Latin Modern font.
- `babel` (British), `csquotes`, `biblatex` (`style=apa`, `backend=biber`) with `\addbibresource{bib/references.bib}`.
- `amsmath`, `siunitx`, `booktabs`, `graphicx`, `microtype`, `hyperref`, `cleveref`.
- A visible placeholder macro for band-specific text.
- Spelling follows the proposal's Oxford convention (normalization, analysed, labelled).
- Acronyms defined on first use; a hand-maintained abbreviations list in the front matter (no `glossaries` package).
- `notes/chapter1-claims.md` is created alongside the draft.

## Deviations made during drafting (2 November 2026)

- **Najashi et al. (2013) dropped.** The journal (*IJCSI*) was reported as listed on Beall's list of potentially predatory publishers and is the subject of a 2020 investigation. Popoola & Otuu (2018) supports the same claims.
- **Open item 1 resolved.** LTE's role is sourced to NCC subscription shares by generation (`ncc2026`), with ITU's 2010 press release (`itu2010`) linking LTE to the term "4G". NCC's page itself does not mention LTE.
- **Saber et al.** were published in 2021 (Springer, Lecture Notes in Networks and Systems vol. 183), so the key is `saber2021`.
- **Corrections found while checking sources:**
  - Gao et al. *generated* their signals following RadioML2016.10a, rather than taking them from a public dataset.
  - Zeghdoud et al.'s comparison with energy detection came from numerical simulation.
  - All three learned-vs-energy-detector comparisons are therefore simulation-based, which the problem statement now says.
- **Tandra & Sahai (2005)** is cited only for signals without a deterministic component, which is the condition of their proof. This motivates the sync correlator as a baseline.
- **Table 1.1** maps each research question to its objectives and principal evidence rather than to chapter sections, since Chapters 2–5 are not yet written.
- **Remaining open items:**
  - Study site (Benin City, taken from proposal §8.1) is used in the text; confirmed by the author on 2 November 2026.
  - The 1% and 5% false-alarm rates are used.
  - Dataset and code release are still open.
  - The gap statement is still provisional.
  - Formatting uses the defaults (see `preamble.tex`) until the SPGS rules are known.
