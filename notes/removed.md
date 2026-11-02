# Removed text

Text taken out of `notes/proposal-revised.md` and `CLAUDE.md` on 2 November 2026 because it was stale, resolved or redundant. Kept here per the "never delete content" rule.

## From notes/proposal-revised.md

- Status note (replaced with a shorter one): "Status of this document: `[CITE]` marks a claim that still needs a source. Sources already located are listed in Section 11 and are labelled by how far they have been verified. Read each paper in full before relying on it."
- Section 1, after "useful for three reasons": "each of which the thesis will motivate with a source". Redundant with the `[CITE]` markers.
- Section 1, DSA bullet: "metadata verified, full text still to be read, see Section 11". Status belongs in Section 12.
- Section 2 heading suffix: "(to be covered in the thesis)".
- Section 2: "Gao et al., 2019, discuss the SNR wall under practical noise uncertainty". Shortened to "(the "SNR wall"; Gao et al., 2019)".
- Section 2: `[CITE]` after "Cyclostationary and matched-filter detectors can do better but need more computation or prior knowledge of the signal". Replaced by (Yücek & Arslan, 2009; Axell et al., 2012), which already cover the detector taxonomy.
- Section 8.1, receiver: "(confirmed by the author from the unit itself; the original proposal's V3 is superseded)". Resolved.
- Section 8.1, band 1: "News reports confirm that the Nigerian Communications Commission (NCC) approved MTN Nigeria's use of 10 MHz in the 800 MHz band acquired from Visafone for 4G LTE (2019), approved a further 10 MHz from Intercellular Nigeria (2021), and renewed both 800 MHz licences to 31 December 2034. These reports do **not** state the band plan or exact frequencies." Condensed; licence history is not needed for band selection.
- Section 8.1, band 1: `[CITE: 3GPP TS 36.101 band table for all band edges in this section; NCC frequency assignment record]`. Replaced by one `[CITE: 3GPP TS 36.101]` for all band edges. The NCC record was redundant because the local survey decides the band.
- Section 8.1, excluded bands: "which sources report several Nigerian operators using". Unsourced and not needed for the exclusion.
- Section 11 (now 12) intro: "All bibliographic details below were checked online on 2 November 2026 against publisher pages, DOI landing pages, dblp or institutional repositories. ... it does **not** mean the paper has been read. Read each paper in full before citing it in the thesis." Shortened.
- Section 12 (now 13), item 1: "~~V3 or V4~~ **Resolved: V4** (R828D tuner, RTL2832U, 1 PPM TCXO, bias tee, HF upconverter, confirmed on the unit)."
- Sections renumbered: new Section 10 (Thesis structure) added; Risks 10 → 11; Sources 11 → 12; Items to confirm 12 → 13.

## From CLAUDE.md

- Hardware item (moved from "Things to resolve" to "Decisions already made", shortened). Removed detail: "Datasheet specs (`rtlsdrblog2024`): 500 kHz–1.766 GHz, 2.56 MHz stable bandwidth (3.2 MHz with drops), 50 Ω input." "Never write V4L (a different model with the R828S tuner)." "The HF upconverter is irrelevant above ~28.8 MHz". The specs are in Section 8.1 of the proposal.
- Band item detail: "News reports confirm MTN holds 2 × 10 MHz in the 800 MHz band for LTE (licences to 2034) but do not give the band plan, so "Band 20" is unconfirmed and the survey covers 790–894 MHz." This is in Section 8.1 of the proposal.
- "Results" item (moved to "Decisions already made"): "The proposals state hypotheses and predictions."
- "That is Tian et al. (2022), key `tian2022`, DOI 10.34133/2022/9865016, volume 2." Details are in the .bib file.
- "Metadata verified is not the same as read" (sentence condensed).
- Corrections now held only in `references-verified.bib` and proposal Section 12:
  - Gao et al. modulation list: "(BPSK, QPSK, 8PSK, CPFSK, QAM16, QAM64, GFSK, PAM4)".
  - "Torío, Sánchez, Cuiñas, Santalla del Río".
  - "**AviSense (`baset2023`)** was published online on 8 Dec 2022, in an issue dated Feb 2023. Use 2023."
  - "**Zeghdoud et al.:** the publisher prints names surname first ("Zeghdoud Sabrina"), so the family names are Zeghdoud, Teguig, Tanougast, Mesloub, Sadoudi, Nesraoui."
- Open `[CITE]` list items no longer open: "the cost and knowledge needs of cyclostationary and matched-filter detectors" (now cited to Yücek & Arslan and Axell et al.); "an NCC frequency assignment record" (dropped as redundant).
- "Claims like "prior studies show deep learning detectors outperform energy detectors at low SNR" need real references." Resolved: Gao et al. (2019), Zeghdoud et al. (2023) and Grosu & Terebeș (2021) now cover this.
- Misplaced note under supervisor feedback: "(All of the above are already addressed in `notes/proposal-revised.md`; carry them into the thesis chapters.)" Moved to the top of that section and now covers every item.
- Seven-chapter plan, replaced by the five-chapter plan:
  1. Introduction: motivation (with the "why" above), problem, aim, objectives, scope, contributions, outline
  2. Background: LTE downlink structure, spectrum sensing, SDR/RTL-SDR, STFT and spectrograms, autoencoders, cognitive radio
  3. Related work: classical sensing methods and deep learning methods, organised by theme
  4. Methodology: hardware and band selection, three-class data collection (terminated-input noise, antenna-connected idle, LTE-active), ground truth, controlled-SNR digital injection, preprocessing, session-level splits, autoencoder/PCA models, baseline detectors, fixed-false-alarm thresholding, evaluation protocol, real-time demo
  5. Results and analysis: quantitative and qualitative, SNR sensitivity, session/gain generalization, bootstrap confidence intervals over sessions, latency and throughput, failure cases
  6. Discussion: findings, limitations, scope exclusions (no demodulation, downlink only, passive, single receiver)
  7. Conclusion and future work

## Raspberry Pi dropped (2 November 2026)

The host computer is a Dell Latitude 5490 laptop, not a Raspberry Pi or other embedded board. Text taken out of `notes/proposal.md`:

- Section 0, "Extras" row: "Dropped or reduced to a measured CPU/edge latency study". Now "a measured CPU latency study on the host laptop".
- Section 6, Optional tier: "Raspberry Pi test".
- Section 8.9, Compute (RQ4): "Optional run on an embedded board such as a Raspberry Pi."
- Section 9, contribution 4: "on commodity hardware". Now "on a commodity laptop".

## Scope decisions for Chapter 1 (2 November 2026)

The author confirmed the scope tiers: all Stretch items in; from the Optional tier, the LTE sync-signal correlator and the VAE in; the CFAR-style detector dropped; a fixed-threshold energy detector added so that H2 can be tested. The Chapter 1 outline (`notes/chapter1-outline.md`) was approved. Text replaced in `notes/proposal.md`:

- Section 0, "Baselines" row, revision column: "Adaptive-threshold ED, CFAR-style detector, linear (PCA) reconstruction baseline; optional LTE sync-signal correlator".
- Section 5, research questions (RQ5 was added after them):
  - "**RQ1:** At a fixed false alarm rate, how does the autoencoder's detection probability compare with adaptive-threshold energy detection across SNR?"
  - "**RQ2:** Does the autoencoder add value beyond a linear (PCA) reconstruction detector and a CFAR-style detector?"
  - "**RQ3:** How do both approaches generalize across capture sessions, days, locations and receiver gain settings?"
  - "**RQ4:** What are the latency and throughput of each method on commodity CPU hardware, and is real-time operation feasible?"
- Section 5, hypothesis: "**H2:** the autoencoder degrades less than fixed-threshold methods across sessions with different noise floors." The only planned energy detector was adaptive, and the autoencoder's own threshold is fixed once set, so H2 had no fixed-threshold comparator.
- Section 6, objective 4: "Train and tune autoencoders (convolutional primary, fully connected comparison) on idle data only."
- Section 6, objective 5: "Implement the baselines: adaptive-threshold energy detection, CFAR-style detection, and a PCA reconstruction detector (optional: LTE synchronization-signal correlator)."
- Section 6, priority tiers: Core read "...; convolutional autoencoder; adaptive-threshold energy detector; ..."; Optional read "CFAR-style detector, LTE sync-signal correlator, VAE with percentile score, dashboard polish."
- Section 7, excluded: "Demodulating or decoding LTE (no cell IDs beyond what a cell-search tool reports for ground truth, no user data); only power and spectral structure are analysed." and "Cooperative multi-sensor sensing; full evaluation of other deep models (VAE, GAN, diffusion) beyond optional ablations."
- Section 8.7: "**Optional:** a variational autoencoder with the percentile score of Tian et al. (2022)."
- Section 8.8: "**CFAR-style detector:** threshold set relative to a local noise estimate across frequency or time cells." and the label "**Optional LTE sync-signal correlator:**".
- Section 8.11: "Reception is passive and only power and spectral features are analysed. No content is decoded and no user data is recorded."
- Section 10, Chapter 4 item: "interpretation against RQ1–RQ4 and H1–H2".
- Section 13, item 2: "Which Stretch and Optional items will actually be done (see the priority tiers). Core items are assumed."

## From bib/references.bib (2 November 2026)

- `grosu2021`: the field `note = {No DOI listed by the journal}` was moved into a comment below the entry, because biblatex-apa printed it in the reference list as "[No DOI listed by the journal]".
