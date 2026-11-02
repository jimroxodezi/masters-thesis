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
