# CLAUDE.md: MSc Thesis (LaTeX)

## Project
**Working title:** Real-Capture Evaluation of Autoencoder-Based and Classical Detection for LTE Downlink Spectrum Occupancy Using a Low-Cost RTL-SDR (confirm the final title with me).

Author: Jimrox Odezi Ogume. Department of Electrical and Electronic Engineering, University of Benin. Supervisor: Prof. Kingsley Ogbeide. Deadline: `<FILL IN>`. Citation style: APA 7.

**Core idea:** train an autoencoder (PyTorch) on idle spectrum captured with an RTL-SDR so it learns the noise floor; frames it reconstructs poorly (high MSE) are flagged as LTE-occupied. Benchmark against strong classical baselines (fixed- and adaptive-threshold energy detectors, PCA reconstruction, an LTE sync-signal correlator) across SNR and across capture sessions. Signals are converted to STFT spectrograms. A null or mixed result is acceptable and must be reported honestly.

**The governing plan is the revised proposal, `notes/proposal.md`. Read it before writing any chapter.** Its Section 0 records what changed from the original August 2025 proposal.

## Decisions already made (do not reopen)
- **All data is captured by me with the RTL-SDR.** No public datasets (no RadioML or similar) anywhere in the thesis's own data.
- **Hardware:** RTL-SDR Blog **V4** for all captures (R828D tuner, RTL2832U 8-bit ADC, 1 PPM TCXO, software-switchable bias tee, built-in HF upconverter). Specs are in Section 8.1 of the proposal (`rtlsdrblog2024`). Never write V3 or V4L. Don't describe the HF upconverter as part of the method. Bias-tee state is logged per capture and is off with the 50 Ω terminator. If a capture log shows any other receiver, ask me.
- **Host computer:** a Dell Latitude 5490 laptop (Intel Core i5-8250U, 8th generation, 4 cores/8 threads; 16 GB RAM; integrated Intel UHD Graphics 620, so PyTorch runs on the CPU only) running Ubuntu 26.04 LTS. Capture, training, evaluation, latency measurement and the real-time demo all run on it. **No Raspberry Pi or other embedded board is used**; never describe embedded deployment as part of the method.
- **Study site:** all captures are made in Benin City, Nigeria.
- **Scope (confirmed 2 November 2026; Section 6 of the proposal):** all Core and Stretch items, plus the LTE sync-signal correlator and the VAE with percentile score from the Optional tier. Dashboard polish stays optional, the CFAR-style detector is dropped, and a fixed-threshold energy detector was added so that H2 can be tested. These are planned methods: describe a method as implemented only once it exists in `code/`.
- The target band must lie **within the RTL-SDR tuning range (about 1.766 GHz max)**. LTE Band 3 downlink (1805 MHz and above) is out of range and must not be described as captured.
- Splits are by capture session/day/location, never by random frame. Normalization statistics come from training data only. Thresholds are set on idle validation data only.
- The thesis reports **actual** results from `code/` and `results/`, never the proposal's hypotheses or predictions.
- The thesis has **five chapters** (see Chapter plan).

## Things to resolve with me before writing (do not guess)
1. **Band:** not yet decided. Candidates and the survey protocol are in Section 8.1 of the proposal. Until I confirm a band, use placeholders in band-specific text and never state a band or center frequency.

## Repo layout
```
main.tex              # root file, includes chapters
preamble.tex          # packages, page layout, thesis metadata and macros
chapters/             # one .tex file per chapter (five)
front/                # abstract, acknowledgements, declaration
figures/              # all figures (PDF); SVG sources and plot scripts in figures/src/
code/                 # Python pipeline (source of truth for methods/results)
results/              # metrics, logs, saved outputs
papers/               # PDFs of cited sources (only cite what is here or DOI-verified)
bib/references.bib    # single bibliography file
notes/                # proposal.md (the revised proposal), removed.md, claim tables, outlines

```

## Build
- `latexmk -pdf -interaction=nonstopmode main.tex`
- Bibliography: `biblatex` with `style=apa`, backend `biber`.
- Compile after every meaningful edit and fix all errors and warnings (undefined references, missing citations, overfull boxes) before reporting a task as done. Never leave the project non-compiling.

## Citation rules (strict)
1. APA 7. In-text: `\parencite{}` for (Author, Year), `\textcite{}` for Author (Year).
2. **Only cite sources in `papers/` or verified by a DOI/URL you have actually opened.** Never cite from memory.
3. **Never invent or guess** authors, years, titles, venues, pages, DOIs, or findings. If a source can't be verified, don't cite it; mark `% TODO: needs source` and tell me.
4. Every new reference gets a full `bib/references.bib` entry with DOI where available. Keys: `firstauthorYEAR` (e.g. `tian2022`).
5. Every in-text citation needs a `.bib` entry, and every `.bib` entry must be cited. Check before finishing a chapter.
6. Paraphrase; don't copy. Direct quotes need quotation marks and page numbers.
7. Keep `notes/<chapter>-claims.md`: a table of claim, source key, and where it appears in the source, so I can spot-check.

## Sources and known problems
- Verified sources are listed in Section 12 of the proposal; BibTeX is in `bib/references.bib` (`tian2022`, `rajendran2019`, `torio2025`, `gao2019`, `yucek2009`, `grosu2021`, `zeghdoud2023`, `baset2023`, `rtlsdrblog2024`, plus the entries added for each chapter). `kang2022` and `axell2012` were moved to `notes/removed.md` because their full texts could not be obtained. Read so far: `torio2025`, `grosu2021`, `tian2022`, `yucek2009` and `rtlsdrblog2024` (published versions), and `gao2019`, `rajendran2019` and `baset2023` (preprint or accepted-manuscript copies, cited without page numbers because the published pagination differs). `zeghdoud2023` has been read in abstract only. The version read for every cited source is recorded in `notes/<chapter>-claims.md`. Read a source before citing a finding from it.
- The original proposal's raw key `tian2022unsupervised` is `tian2022`. Its numbers "20% detection probability at -20 dB", "85-90%" and "4-5x improvement" have no source and must not reappear.
- Corrections to carry forward:
  - **Gao et al. (2019)** used single-carrier modulated signals, not OFDM. Never cite them for OFDM.
  - **Grosu & Terebeș (2021)** used **simulated** data only; say so when citing.
  - **Zeghdoud et al. (2023)** found the opposite of Grosu & Terebeș; cite both to support "the evidence is mixed".
  - **Torío et al. (2025)** has four authors; APA 7 lists all of them.
- `[CITE]` markers in the proposal and how the thesis handles them:
  - LTE's share of mobile broadband: Chapter 1 cites the share of subscriptions by generation (`ncc2026`, with `itu2010` linking "4G" to LTE). It is a share of subscriptions, not of traffic.
  - Interference management: Chapter 1 cites `torio2025` and `tian2022`; a regulator or operator source is still not cited.
  - Long-term occupancy statistics: Chapter 1 cites `popoola2018`.
  - Band edges: Chapter 2 cites 3GPP TS 36.101 (`3gpp2026b`, Table 5.5-1).

## Supervisor feedback to address
All of these are addressed in the proposal; carry them into the chapters:
- Explain **why** interference management, network optimization and dynamic spectrum access planning matter (Chapter 1).
- Add background on **cognitive radio systems** (Chapter 2).
- Add a taxonomy of classical spectrum sensing: energy detection, matched filtering, cyclostationary feature detection, eigenvalue-based detection, waveform-based detection (Chapter 2).
- Justify the autoencoder architecture choice: CNN for spectrogram structure vs dense model for flattened-vector input (Chapters 2 and 3, with the ablation in Chapter 4).
- The benchmark against the classical energy detector is the core contribution; give it full, careful treatment.

## Writing standards
- Academic, precise, impersonal voice. Past tense for what was done and found. No filler, hype, or unsupported claims.
- State limitations, assumptions, and threats to validity. Do not overstate results.
- Methods and results must match `code/` and `results/`. Read the code first. Never invent numbers, metrics, or settings; if a value is unknown, ask.
- Define acronyms on first use (LTE, SDR, STFT, PSD, SNR, ROC, AUC, MSE, etc.). Keep notation consistent, with a notation list.
- Figures and tables: numbered, captioned, referenced with `\cref{}`. Plots as vector PDF with readable labels and units. Use `siunitx` and `booktabs`.
- Detection metrics to report: detection probability at fixed false alarm rates (e.g. 1% and 5%), confusion matrix, precision, recall, F1, ROC/AUC, and SNR-stratified results, for the autoencoder and every baseline, on identical frames. Report variability over training seeds and bootstrap confidence intervals resampled over sessions. Also report per-frame latency and throughput for each method, measured on the host laptop.

## Chapter plan (five chapters; detail in Section 10 of the proposal)
1. **Introduction:** motivation (with the "why" above), problem statement, aim, research questions and hypotheses, objectives, scope, contributions, thesis outline
2. **Literature Review:** LTE downlink structure, cognitive radio, classical sensing taxonomy, SDR/RTL-SDR, STFT and spectrograms, autoencoders; related work on learned anomaly detection and learned sensing vs energy detection, organised by theme; research gap
3. **Methodology:** hardware and band selection, three-class data collection (terminated-input noise, antenna-connected idle, LTE-active), ground truth, controlled-SNR digital injection, preprocessing, session-level splits, autoencoder/PCA models, baseline detectors, fixed-false-alarm thresholding, evaluation protocol, real-time demo
4. **Results and Discussion:** quantitative and qualitative results, SNR sensitivity, session/gain generalization, bootstrap confidence intervals over sessions, latency and throughput, failure cases; interpretation against the research questions; limitations and scope exclusions (no demodulation, downlink only, passive, single receiver)
5. **Conclusion and Recommendations:** summary of findings, contributions, recommendations, future work

## Working rules
- One chapter at a time: outline first, get my approval, then draft.
- Small, reviewable edits. Don't touch files I didn't ask about.
- Never delete content; move removed text to `notes/removed.md`.
- Minimal package set; ask before adding heavy packages or changing the document class.
- If something is ambiguous or missing (supervisor requirements, data details, results), ask. Don't guess.
- At the end of each task: summarise what changed, what still needs verification, and open TODOs.
