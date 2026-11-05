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

## Idle class corrected (2 November 2026)

3GPP TS 36.211 V19.3.0, clause 6.10.1, states that a UE may assume cell-specific reference signals are transmitted in all downlink subframes of an FDD cell that supports PDSCH transmission (in MBSFN subframes, only in the first one or two OFDM symbols). An active carrier is therefore never silent, and an idle class cannot be drawn from its "quiet periods". The author approved the correction. Text replaced in `notes/proposal.md`, Section 8.2:

- "**Class B, idle environment:** antenna connected, tuned to spectrum with no LTE carrier (guard regions, unused channels, or quiet periods), confirmed by cell search and power screening. Captures environmental noise and local interference."

In `chapters/chapter1.tex` (Section 1.5) a sentence defining occupancy as carrier presence was added; no text was removed there.

## Uncited bibliography entries removed (2 November 2026)

Rule 5 of CLAUDE.md requires every `.bib` entry to be cited. These two entries, listed among the verified sources in CLAUDE.md, could not be read and are not cited in Chapters 1 or 2, so they were moved here from `bib/references.bib`. Restore them if the full texts become available and the papers are cited.

- `kang2022`: the author could not obtain the PDF (2 November 2026). No abstract was retrievable: Crossref holds none, IEEE Xplore did not render for the fetch tool, and the Semantic Scholar API refused the request (rate limit).
- `axell2012`: the TU Delft repository record has no file and refused connections; no other open copy was found. Chapter 2 uses Axell et al. (2010, `axell2010`) instead.

```bibtex
@article{kang2022,
  author  = {Kang, Ying and Wu, Hao and Zhao, Zhihua and Li, Yaxing and Meng, Jin},
  title   = {{DL}-based anomaly detection at the physical-layer of cognitive radio by deep support vector data description},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  volume  = {8},
  number  = {4},
  pages   = {1689--1705},
  year    = {2022},
  doi     = {10.1109/TCCN.2022.3200377}
}
@article{axell2012,
  author  = {Axell, Erik and Leus, Geert and Larsson, Erik G. and Poor, H. Vincent},
  title   = {Spectrum sensing for cognitive radio: State-of-the-art and recent advances},
  journal = {IEEE Signal Processing Magazine},
  volume  = {29},
  number  = {3},
  pages   = {101--116},
  year    = {2012},
  doi     = {10.1109/MSP.2012.2183771}
}
```

## Chapter 1 wording corrected after the Chapter 2 reading (2 November 2026)

Saber et al. (2021) call their classifier a CNN, but the only network they describe is fully connected (1024-1024-8-1), and they report each detector at a different false-alarm rate. Text replaced in `chapters/chapter1.tex`, Section 1.2:

- "\Textcite{saber2021} captured real signals with an RTL-SDR receiver and found that a supervised CNN classified them better than energy detection, an artificial neural network and a support vector machine."

Comment replaced at the top of `chapters/chapter1.tex`:

- "% TODO: re-check the gap statement (end of Section 1.2) after the Chapter 2 / literature search."

## From CLAUDE.md: outdated file references (2 November 2026)

At the author's request, file references in CLAUDE.md that no longer matched the repository were corrected: `notes/proposal-revised.md` does not exist (the revised proposal is `notes/proposal.md`), no transcription of the original proposal is kept, and the bibliography is `bib/references.bib`. The supervisor's name was filled in and the study site (Benin City) added to the decisions. Replaced text:

- "Supervisor: `<FILL IN>`."
- "**The governing plan is `notes/proposal-revised.md`. Read it before writing any chapter.** `notes/proposal.md` is the transcription of the original August 2025 proposal (with the supervisor's handwritten notes); use it only for history. Where they differ, the revised proposal wins."
- Repository layout lines: "references.bib        # single bibliography file"; "notes/                # proposal.md, proposal-revised.md, removed.md, claim tables, outlines, TODOs"; "figures/              # all figures (PDF/PNG)".
- "BibTeX is in `references-verified.bib` (`tian2022`, `rajendran2019`, `torio2025`, `gao2019`, `kang2022`, `yucek2009`, `axell2012`, `grosu2021`, `zeghdoud2023`, `baset2023`, `rtlsdrblog2024`)."
- Citation rule 4: "Every new reference gets a full `references.bib` entry with DOI where available."

## From CLAUDE.md and the bibliography header: outdated status (2 November 2026)

At the author's request, statements that no longer matched the project's state were updated. Replaced text:

- CLAUDE.md, "Things to resolve with me before writing", item 2: "2. **Scope tiers:** I have not confirmed which Stretch and Optional items will be done; ask before describing any method as implemented." Scope was confirmed for Chapter 1 and is now listed under "Decisions already made".
- CLAUDE.md, "Sources and known problems": "Only `torio2025`, `grosu2021` and the arXiv preprint of `gao2019` have been read in full; read the others before citing a finding from them."
- CLAUDE.md, "Sources and known problems": "Open `[CITE]` markers in the proposal: LTE's share of mobile broadband; interference management (regulator or operator source); long-term occupancy statistics; 3GPP TS 36.101 for band edges."
- `bib/references.bib`, header comment:

```
% Verified entries for Section 11 of notes/proposal-revised.md
% Checked online on 2026-11-02 against publisher/indexing pages (not full-text reads
% unless noted). Paste into references.bib after reading each paper in full.
% Use biblatex + biber (style=apa).
```
- CLAUDE.md, "Core idea": "Benchmark against strong classical baselines (adaptive energy detector, CFAR-style, PCA reconstruction) across SNR and across capture sessions." The CFAR-style detector was dropped from the scope; the baseline list now matches the confirmed scope.

## Method design approved (2 November 2026)

The author approved the Chapter 3 outline with recommendations D1–D5; the detailed design is in `notes/method-design.md`. Text replaced in `notes/proposal.md`, Section 8.1:

- "50 Ω input. Sample rate 2.4 MS/s."

## Capture order aligned with the code (2 November 2026)

The guided session in `code/occupancy/session.py` captures each class at g1 and then g2 before moving on, so the 50 Ω terminator is fitted and removed only once. Chapter 3 and the design file were aligned with it. Text replaced in `chapters/chapter3.tex`, Section 3.4.2:

- "A session is one visit to one position, during which all classes are captured at both gain settings in a fixed order: A, B, C\textsubscript{centre}, C\textsubscript{offset}, first at $g_1$ and then at $g_2$."
- Table 3.4, order column: "1, 5 / 2, 6 / 3, 7 / 4, 8".

## Chapter 3 aligned with the implemented code (2 November 2026)

After the rewritten `code/` was verified (71 tests passing), Chapter 3 was made exact where the draft was less specific. Replaced phrases in `chapters/chapter3.tex`:

- "and the frequency axis is ordered from the lowest to the highest frequency of the window." (orientation now stated as in the code)
- "The decoder mirrors the encoder with $2 \times$ upsampling and $3 \times 3$ convolutions, and its output layer is linear."
- "Results are reported separately for C\textsubscript{centre} and C\textsubscript{offset} mixtures." (the code's primary H1 test pools both sources)

The other changes only added sentences: the 68 ms discard, the parameter counts, the per-target adaptive runs, the PSS lag range, the quantile rule, the pooled-ROC scaling and the seed-averaged bootstrap.

## Training moved to Google Colab (2 November 2026)

The author decided that the neural networks are trained on a Google Colab GPU, and that everything else stays on the laptop. Replaced text:

- `CLAUDE.md`: "Capture, training, evaluation, latency measurement and the real-time demo all run on it."
- `chapters/chapter1.tex`: "All processing, including capture, training, evaluation, latency measurement and the real-time demonstration, runs on the CPU of a Dell Latitude 5490 laptop (Intel Core i5-8250U processor, \qty{16}{\giga\byte} of memory)."
- `chapters/chapter3.tex`: "All processing, from capture to evaluation, runs on the laptop's CPU."
- `notes/proposal.md`: "It runs capture, training, evaluation, latency measurement and the real-time demonstration; no embedded board is used."
- `chapters/chapter3.tex`, Table 3.7 row: "CPU threads & 4".

## Appendix B: tests described, not listed (3 November 2026)

At the author's request, the full listings of the 14 test files were removed from Appendix B and replaced by Table B.2, which describes each file and counts its tests. The bullet list of test descriptions became that table. The test code remains in `code/tests/`.

## Chapter 3: Colab training and reproducibility aligned with the implemented code (3 November 2026)

Replaced after the Colab workflow was implemented in `code/` (pack, unpack, device, budget modules). Each sentence was replaced by a more precise one; the old text was:

- because training every model, fold and seed on the laptop's CPU would take weeks.
- Raw captures, active captures and test data stay on the laptop.
- The trained models are returned to the laptop, where they are checked for completeness and then calibrated, evaluated and timed.
- Every result file records the configuration, the seeds, the Git commit of the code and the library versions, and every trained model also records the device it was trained on.

Appendix B intro, old sentence: The version of the code is identified by the Git commit recorded in every result file (\TBC{commit of the final run}).

## Chapter 3: power condition, scoring copy and demo behaviour (3 November 2026)

The author fixed the timing condition (mains power, Performance profile; no battery condition) and approved the demo fixes. Two sentences were extended; the old versions were:

- Section 3.12 (sec:m-compute): "The CPU model, the thread count and the library versions are recorded with each measurement."
- Section 3.13 (sec:m-demo): "The time, centre frequency, gain, score, decision and processing time of every frame are logged, and an optional live view shows the rolling spectrogram and the occupancy decisions."

## Chapter 3, Section 3.12 (Computational Cost) restructured (3 November 2026)

At the author's request, the code optimizations were written up in their own subsection (3.12.1 Optimization of Scoring, sec:m-optim), and the timing protocol became 3.12.2 (Timing, sec:m-timing), with the settling period added. The previous single paragraph was:

> Latency and throughput are measured on the host laptop with the library versions of \cref{tab:m-hw-sw} and PyTorch limited to four threads. After \num{50} warm-up frames, each detector scores \num{1000} frames one at a time, and the time per frame is measured with Python's high-resolution performance counter. The median and the 95th percentile are reported separately for preprocessing (DC removal, spectrogram and standardization), for scoring, and end to end. Throughput is measured in frames per second with batches of 64 frames. The real-time factor is the processing time per frame divided by the frame duration, \qty{8.533}{\milli\second}; a value below one means the detector keeps up with the receiver. All timings are taken with the laptop on mains power and set to its Performance power profile, so that the processor runs at full speed and the timings are comparable; the timing program refuses any other known power state. The CPU model, the thread count, the library versions and the power state are recorded with each measurement. The detectors are timed exactly as they are evaluated: for scoring, every batch-normalization layer of the convolutional networks is folded into the convolution before it and the data are stored in the channels-last memory layout, which the processor handles faster. This changes the reconstructions only by floating-point rounding, and the same scoring code is used in the evaluation, in the timing and in the demonstration.

## Appendix B trimmed to the relevant listings (3 November 2026)

At the author's request, Appendix B now lists only the modules that compute what the thesis reports (dataset, preprocess, injection, models, pss, detectors, calibrate, evaluate, latency); the rest of the code is on GitHub. Table B.1 now gives a one-line purpose for every module. The descriptions of the modules no longer listed were:

- **occupancy/init.py and occupancy/main.py**: The package marker, with a summary of what the package contains, and the entry point that makes \texttt{python -m occupancy} run the command line of \texttt{cli.py}.
- **occupancy/config.py**: Every setting of the experiment, and no setting anywhere else. Each value carries the label it has in the design (fixed, default, field or implementation detail), and a run can override values with a JSON file whose unknown keys are rejected, so a typing error cannot silently fall back to a default. It also holds the training-device setting: by default a GPU when one is available, as on Colab, and otherwise the CPU. Derived quantities, such as the frame duration, are computed here from the basic settings.
- **occupancy/provenance.py**: Records what produced a file: the version of the code, the library versions, the host computer and its power mode. Because Git is not available on Colab, the code version is a checksum of the package's source files, recorded together with the Git state where Git is available. Every capture record and every result file embeds this record, which is what makes each number in Chapter~4 traceable to the code and settings that produced it.
- **occupancy/iqio.py**: Reads and writes the raw capture files: interleaved unsigned 8-bit in-phase and quadrature samples, exactly as the receiver delivers them, and the JSON metadata record written last. A capture without a valid record is never used. The file also defines the conversion of the 8-bit values to complex samples.
- **occupancy/capture.py**: Records one capture from the RTL-SDR Blog V4 together with its metadata (\cref{sec:m-capture}). It enforces the capture rules: manual gain only, a gain that is one of the tuner's steps, the bias tee off, \qty{1.92}{MS/s}, a frequency inside the tuning range, and a receiver that identifies itself as a V4. Samples are streamed in blocks, the first \num{131072} samples are discarded, and the clipping fraction and mean power are measured. The file also contains the rule that places the C\textsubscript{offset} window.
- **occupancy/session.py**: The guided capture session (\cref{sec:m-sessions}). It asks for the session details and frequencies, plans the C\textsubscript{offset} window and checks the Class~B window against the carrier. It then walks the operator through the eight captures in order, telling them when to fit and remove the terminator, and reports each capture's clipping and power. A capture can be retaken, and a session manifest is rewritten after every step so that an interrupted session leaves a consistent record.
- **occupancy/survey.py**: The band survey (\cref{sec:m-survey}). It runs \texttt{rtl\_power} at a fixed gain, reads its CSV output, plots mean power against frequency and a waterfall, and lists the plateaus that may be LTE carriers and the idle stretches at least \qty{1.92}{\mega\hertz} wide that could serve as Class~B windows. These lists are aids for the operator; carriers are confirmed with a cell scanner.
- **occupancy/quicklook.py**: The quick-look figure of each capture (its mean spectrum and a sample spectrogram), shown right after recording, and the labelling check of \cref{sec:m-labels}: a template for the second labelling pass and the computation of the agreement between the two passes.
- **occupancy/train.py**: Trains the neural networks and fits the PCA detector for every fold (\cref{sec:m-training}). The normalization is fitted on the fold's training frames and shared by every model of that fold. The weights with the lowest validation loss are kept. Training reads either the local data or a training pack (used on Colab), runs on the CPU or a GPU, saves a checkpoint after every epoch, and resumes an interrupted run from its last checkpoint. A finished run is marked complete and is skipped when training is started again.
- **occupancy/scoring.py**: Decides which frames each fold scores: the idle validation frames used for calibration, and every capture of the test sessions with their controlled-SNR mixtures. It also defines how score files are stored.
- **occupancy/realtime.py**: The real-time demonstration (\cref{sec:m-demo}). It reads the receiver one frame at a time, scores each frame with the chosen detector and its calibrated threshold, and logs every decision. A short queue separates reading from processing. If processing falls behind, the oldest blocks are dropped and counted, so that the decisions stay current, and samples lost on the USB link are estimated from the arrival times of the blocks. An optional live view, redrawn at most four times per second, can be enabled.
- **occupancy/cli.py**: The command line, \texttt{python -m occupancy} followed by a subcommand: \begin{itemize} \item field work: \texttt{survey}, \texttt{capture}, \texttt{session} and \texttt{inspect}; \item the experiment: \texttt{preprocess}, \texttt{train}, \texttt{calibrate} and \texttt{evaluate}; \item training on Colab: \texttt{pack} and \texttt{unpack} on the laptop, and \texttt{train} with the option \texttt{--pack} on Colab (with \texttt{--time-epoch}, it estimates the compute budget); \item cost and demonstration: \texttt{latency} and \texttt{demo}; \item testing only: \texttt{synthetic}. \end{itemize} The commands that read captures refuse synthetic ones unless explicitly allowed, and then read from and write to separate folders.
- **occupancy/device.py**: Chooses where training runs (a GPU when one is available, otherwise the CPU) and switches on the deterministic options of the GPU libraries. It also writes a record of the runtime environment into every training output: the device and GPU, the library and driver versions, the determinism settings, the thread count and the code version. The backward pass of the bilinear upsampling in the convolutional decoders has no deterministic GPU implementation, so training on a GPU is not guaranteed to give bit-identical weights when a run is repeated; the seed still fixes the initial weights and the order of the training frames.
- **occupancy/pack.py**: Builds the training pack on the laptop: the spectrograms of every session's idle captures at $g_1$, stored once per session, together with the folds, each fold's normalization statistics, the frozen configuration, the code version and a manifest of SHA-256 checksums. It also writes a zip file of the pack and of the code for uploading. Raw captures, active captures, captures at $g_2$, the controlled-SNR sets and all scores stay on the laptop.
- **occupancy/unpack.py**: Imports the models trained on Colab into the laptop's working folder. Before copying anything, it checks that the models were trained on a pack made from this working folder, with the same folds, training settings, code and normalization statistics, and that every expected run is complete and matches its checksums. Every check and its result is recorded.
- **occupancy/budget.py**: Estimates the compute needed for training. It times one training epoch of each of the three main networks on one fold, and the writing of one checkpoint, on the chosen device. From these it estimates the time of every run in the design at 30, 60 and 100 epochs per run (early stopping ends most runs before 100), the Colab compute units at the hourly rate Colab shows, and the volume of checkpoints written to Google Drive.
- **colab/colabtrain.py**: The logic behind the Colab notebook. It checks the environment and the GPU, copies the training pack from Google Drive to the runtime and verifies its checksums, times an epoch, runs (and after a disconnect resumes) the training of every model, fold and seed, retries an attempt that fails (for example on a Google Drive input/output error), reports progress, and zips the trained models, without checkpoints, for download. It never needs the receiver.
- **colab/trainoncolab.ipynb**: The notebook that runs the training on Colab. Its explanatory text cells guide the author through ten steps: settings, mounting Google Drive, checking the GPU, copying the code and the pack, installing missing packages, timing one epoch, training (re-run after a disconnect), checking progress, zipping the outputs, and the steps that follow on the laptop. Its code cells, most of which call \texttt{colab\_train.py}, are listed below in order. % Generated from the notebook (appendices/generated/train_on_colab_cells.py); % regenerate whenever the notebook changes.
- **occupancy/synthetic.py**: Generates simple LTE-like signals and noise for testing the code without a receiver. Nothing generated here is used for any result in this thesis. Every synthetic capture is marked as synthetic in its metadata and written to a separate folder, and the rest of the package refuses it unless explicitly allowed.

Also removed: the section introductions "Training on Google Colab" ("The neural networks are trained on a Colab GPU (\cref{sec:m-training}). The following modules move the training data to Colab and the trained models back, and estimate the compute needed.") and the generated listing of the notebook's code cells (appendices/generated/train_on_colab_cells.py, deleted; it can be regenerated from code/colab/train_on_colab.ipynb). The previous Table B.1 grouped the modules by stage with columns Stage, Modules and Method.


## Chapter 3, Reproducibility (3 November 2026)

Old sentence, replaced when the code listings were trimmed and the GitHub address added: "The code is released, and the captures and labels are released if the supervisor approves."

## Chapter 3, evaluation made self-contained (3 November 2026)

The listing of evaluate.py was removed from Appendix B at the author's request ("rely on chapter 3's description of the evaluation"), so three details that were only in the code were added to Sections 3.11.3 and 3.11.5. The old sentences were:

- "\emph{H1.} On the controlled-SNR set at $\gamma \le \qty{-4}{\decibel}$, the difference"
- "The first is the absolute difference between the realized and the target $\Pfa$. The second is the absolute difference between $\Pd$ at $g_2$ and at $g_1$, compared on the same controlled-SNR levels. H2 is supported if both departures of the convolutional autoencoder are smaller than those of the fixed-threshold energy detector, with \pct{95} bootstrap intervals of the paired differences entirely on that side."
- "Variability due to training is the mean and standard deviation of each metric over the five seeds."

## Appendix B: evaluate.py listing removed (3 November 2026)

At the author's request ("rely on chapter 3's description of the evaluation"). Its description was: "The evaluation protocol of \cref{sec:m-eval}. It pools the test predictions of the four folds, computes the metrics of \cref{tab:m-metrics}, overall and by SNR level, and gives the mean and standard deviation over seeds and the session-bootstrap intervals. It also performs the H1 and H2 tests exactly as defined in \cref{sec:m-hypotheses}, and writes every result with its provenance." Section B.3 was renamed from "Calibration and Evaluation" to "Calibration".

## code/chapter4_implementation_report.docx deleted (3 November 2026)

Deleted at the author's request as no longer relevant: a 12,400-word Word report, "Chapter 4: System Implementation", describing the earlier pipeline (an autoencoder trained on normal LTE signal, framed as anomaly detection), which the current design replaced. It was not referenced anywhere. It is kept in git: `git show 1c4d7f6:code/chapter4_implementation_report.docx > chapter4_implementation_report.docx`.

## Chapter 3: session times and the cell-search tool (4 November 2026)

The author fixed the session and survey times (day 07:00, night 22:00 WAT) and chose LTE-Cell-Scanner (`CellSearch`) as the cell-search tool. The sentences replaced were:

- "The software is written in Python and uses PyTorch \parencite{paszke2019} for the neural networks and scikit-learn \parencite{pedregosa2011} for PCA. \Cref{tab:m-hw-sw} lists the hardware and software."
- "    Survey tool & \texttt{rtl\_power} (from the RTL-SDR driver package) \\
    Cell-search tool & \TBC{RTL-SDR-compatible LTE cell-search tool, to be chosen} \\"
- "Each range is swept with \texttt{rtl\_power} at a fixed gain, by day and at night, and the mean power is plotted against frequency. An active LTE carrier appears as a flat plateau of raised power whose width matches one of the channel bandwidths of \cref{tab:lte-bandwidths}. A cell-search tool then confirms the carrier and records its physical-layer cell identity and centre frequency."
- "At least twelve sessions are planned, spread over at least four days, by day and at night, and over at least two positions in Benin City."
- "A Class~B capture is labelled idle in every frame if neither the survey nor the cell search finds a carrier in its window."
- "they are also broken down by session, by time of day and, where more than one position exists, by position."
- "The \syncsignal{} correlator tests only for the presence of known sequences; no system information, user data or other content is decoded or recorded."

## Appendix C driver section corrected (4 November 2026)

The first version quoted the RTL-SDR Blog V4 users guide, which now builds the Osmocom library; the laptop actually uses the RTL-SDR Blog fork (rtl-sdr-blog, commit aed0ea1). The old section and the bib entry it cited were:

```
\section{Receiver Driver}
\label{app:install-driver}

The RTL-SDR Blog V4 needs a version of the RTL-SDR library that supports its R828D tuner. The manufacturer's guide installs it from source \parencite{rtlsdrblognd}. It first removes any earlier copies of the library and its tools (\texttt{sudo apt purge \textasciicircum librtlsdr}, followed by the removal of leftover files from the system folders), then builds and installs the library, and finally stops the television driver of the Linux kernel from claiming the receiver:
\begin{lstlisting}[language=bash]
sudo apt-get install libusb-1.0-0-dev git cmake pkg-config build-essential
git clone https://github.com/osmocom/rtl-sdr
cd rtl-sdr
mkdir build
cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo make install
sudo cp ../rtl-sdr.rules /etc/udev/rules.d/
sudo ldconfig
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee --append /etc/modprobe.d/blacklist-dvb_usb_rtl28xxu.conf
\end{lstlisting}
The laptop is then restarted. The same build installs \texttt{rtl\_test} and the survey tool \texttt{rtl\_power}. On the host laptop the library built from source is in \texttt{/usr/local/lib}, and \texttt{rtl\_test -t} identifies the receiver:
\begin{lstlisting}[language=bash]
Found 1 device(s):
  0:  RTLSDRBlog, Blog V4, SN: 00000001
Found Rafael Micro R828D tuner
RTL-SDR Blog V4 Detected
\end{lstlisting}
It also lists the 29 gain steps of the tuner, from \qty{0.0}{\decibel} to \qty{49.6}{\decibel}; the gains $g_1$ and $g_2$ are chosen from these steps.
```

```
@online{rtlsdrblognd,
  author  = {{RTL-SDR Blog}},
  title   = {{RTL-SDR Blog V4} users guide},
  url     = {https://www.rtl-sdr.com/V4/},
  urldate = {2026-11-04}
}
% Opened 2026-11-04; no date on the page. Linux driver steps: purge old
% librtlsdr packages, build the Osmocom rtl-sdr library with
% -DINSTALL_UDEV_RULES=ON, install the udev rules, ldconfig, blacklist
% dvb_usb_rtl28xxu, reboot.
```

## Appendix C and README: Ubuntu's librtlsdr purged (4 November 2026)

At the author's request Ubuntu's librtlsdr0, librtlsdr2, librtlsdr-dev and rtl-sdr packages (2.0.2) were purged; the RTL-SDR Blog fork in /usr/local is now the only librtlsdr. Old Appendix C sentence: "The fork's instructions first remove any copy of the library installed by the package manager. On the host laptop, Ubuntu's own copy (version 2.0.2) is still installed, but programs load the fork's library from \texttt{/usr/local/lib}, which is searched first: the cell-search tool is linked to it (\cref{app:install-cellsearch}), and the capture program records the library it loaded with every capture. \texttt{rtl\_test -t} identifies the receiver:" Old README lines: "On this laptop the Blog build is in `/usr/local/lib` and the distribution library is still in `/usr/lib/x86_64-linux-gnu`. The dynamic linker currently picks `/usr/local/lib` first."

## Appendix C: purge paragraph removed (4 November 2026)

The author judged it irrelevant to the thesis. Removed text:

```
The fork's instructions first remove any copy of the library installed by the package manager. On the host laptop, Ubuntu's own packages of the library and its tools (version 2.0.2) were removed after the fork had been installed, so that the fork's library is the only one:
\begin{lstlisting}[language=bash]
sudo apt purge librtlsdr0 librtlsdr2 librtlsdr-dev rtl-sdr
sudo udevadm control --reload-rules
\end{lstlisting}
The fork's own udev rule gives the \texttt{plugdev} group access to the receiver. The cell-search tool is linked to the fork's library (\cref{app:install-cellsearch}), and the capture program records the library it loaded with every capture. \texttt{rtl\_test -t} identifies the receiver:
```

## code/README.md: purge note removed (4 November 2026)

At the author's request. Removed lines:
>    Ubuntu's own librtlsdr packages (2.0.2) were purged on 4 November 2026, so
>    `/usr/local/lib/librtlsdr.so.0.6git` is the only librtlsdr on the laptop.

## Chapter 3, Band Survey expanded (4 November 2026)

At the author's request, the purpose of the survey is explained in full (Table 3.x, settings decided by the survey; sweeps; idle window; times). Previous subsection text:

```
\subsection{Band Survey}
\label{sec:m-survey}

The band is chosen after a survey at the capture site, because an allocation is not the same as a live LTE carrier. The candidate downlink ranges are those of \cref{tab:lte-bands} that the receiver can tune: \qtyrange{758}{803}{\mega\hertz} (Band~28), \qtyrange{790}{894}{\mega\hertz} (Bands~20 and~5 together) and \qtyrange{925}{960}{\mega\hertz} (Band~8). Each range is swept with \texttt{rtl\_power} at a fixed gain, once by day and once at night, starting at the session times of \cref{sec:m-sessions}, and the mean power is plotted against frequency. An active LTE carrier appears as a flat plateau of raised power whose width matches one of the channel bandwidths of \cref{tab:lte-bandwidths}.

Each plateau is then examined with \texttt{CellSearch}, the cell-search program of LTE-Cell-Scanner, which was written for RTL2832-based receivers and is built here for the RTL-SDR \parencite{peroulas2024}. It tunes the receiver, at a fixed gain, to each frequency of a range around the plateau in steps of \qty{100}{\kilo\hertz}, looks for the primary and secondary synchronization signals of a cell, and decodes the master information block (MIB) that the cell broadcasts. For each cell found it reports the physical-layer cell identity, the carrier frequency, the frequency offset, the received power and the bandwidth in resource blocks \parencite{peroulas2024}, which gives the carrier's channel bandwidth (\cref{tab:lte-bandwidths}). A plateau counts as a carrier only if a cell is found at its centre. The band studied is the one with the best confirmed carrier at the site: \TargetBand. The survey outputs are kept as evidence for the choice (\cref{app:survey}).

The survey also fixes where the receiver's \qty{1.92}{\mega\hertz} window is placed (\cref{fig:m-windows}). For idle captures (Class~B, \cref{sec:m-data}), the window is placed on a stretch of the surveyed range at least \qty{1.92}{\mega\hertz} wide with no carrier, as close as possible to the carrier used for the active captures. For active captures (Class~C) it is placed either on the carrier's centre, where it contains the synchronization signals, or inside the carrier away from its centre, where it does not (\cref{sec:lte-sync}).
```

Chapter 2 (Section 2.6.1), old ending: "any LTE energy left in the training frames would be learned as normal, which is why idle frames are screened before training (Chapter~3)." (it implied frame-level screening, which Chapter 3 excludes).

## Chapter 3: idle-window rule made explicit (4 November 2026)

After the test survey showed GSM-like narrow peaks and an intermittent signal that the plateau-only rule missed. Replaced text:

- chapters/chapter3.tex: "    Idle (Class~B) window & A stretch of at least \qty{1.92}{\mega\hertz} with no plateau, outside every carrier found &"
- chapters/chapter3.tex: "The readings are plotted as the mean power against frequency, together with the maximum over the readings, and as a waterfall of power against frequency and time."
- chapters/chapter3.tex: "the window has to be shown to be empty before any idle data are captured: the survey must show no plateau in it, and it must lie outside the band occupied by every carrier that the cell search finds near it."
- chapters/chapter3.tex: "A Class~B capture is labelled idle in every frame if the survey shows no plateau in its window and no carrier found by the cell search occupies any part of it."

## Chapter 3: positions recorded (4 November 2026)

The author chose two balconies at home (P1 parlour, P2 bedroom). Replaced sentences:

- chapters/chapter3.tex: "At least twelve sessions are planned, spread over at least four days and over at least two positions in Benin City."
- chapters/chapter3.tex: "The survey is made at the first capture position, before the first session; each of its two runs takes about half an hour for the sweeps, followed by the cell searches."
- chapters/chapter3.tex: "Different positions and times within Benin City widen the conditions but do not remove this limit."

## Chapter 3: cell search why and how (4 November 2026)

At the author's request the reasons for the cell search and its procedure were written out. Old paragraph:

> \emph{Cell search.} Each plateau is then examined with \texttt{CellSearch}, the cell-search program of LTE-Cell-Scanner, which was written for RTL2832-based receivers and is built here for the RTL-SDR \parencite{peroulas2024}. It tunes the receiver, at a fixed gain, to each frequency of a range around the plateau in steps of \qty{100}{\kilo\hertz}, looks for the primary and secondary synchronization signals of a cell, and decodes the master information block (MIB) that the cell broadcasts. For each cell found it reports the physical-layer cell identity, the carrier frequency, the frequency offset, the received power and the bandwidth in resource blocks \parencite{peroulas2024}, which gives the carrier's channel bandwidth (\cref{tab:lte-bandwidths}). A plateau counts as a carrier only if a cell is found at its centre.

## Thesis title changed (4 November 2026)

The author chose the final title "AI-Based LTE Spectrum Occupancy Detection with a Low-Cost RTL-SDR: A Session-Level Evaluation Against Classical Detectors". Previous working title: "Real-Capture Evaluation of Autoencoder-Based and Classical Detection for LTE Downlink Spectrum Occupancy Using a Low-Cost RTL-SDR" (still the title of the revised proposal, notes/proposal.md, which is left as written).

## Thesis title revised (4 November 2026)

The author added "Autoencoders": now "AI-Based LTE Spectrum Occupancy Detection Using Autoencoders and a Low-Cost RTL-SDR: A Session-Level Evaluation Against Classical Detectors". Previous: "AI-Based LTE Spectrum Occupancy Detection with a Low-Cost RTL-SDR: A Session-Level Evaluation Against Classical Detectors".

## Title alignment: AI-based, autoencoders, session-level (4 November 2026)

At the author's request, Chapters 1, 2 and 5 were aligned with the new title. Replaced text:

- chapters/chapter1.tex: "This thesis evaluates whether an autoencoder trained only on idle spectrum, captured with a low-cost RTL-SDR receiver, can detect when a Long Term Evolution (LTE) downlink band is occupied, and compares it with classical detectors tuned with equal care."
- chapters/chapter1.tex: "Learned detectors have been proposed as a way past these limits, with mixed results."
- chapters/chapter1.tex: "The aim of this study is to design, implement and rigorously evaluate an autoencoder-based occupancy detector for an LTE downlink band using only RTL-SDR captures, and to compare it with classical baselines under controlled and uncontrolled conditions."
- chapters/chapter1.tex: "\item a controlled benchmark of noise-trained autoencoders against fixed-threshold"
- chapters/chapter1.tex: "\item an analysis of how these detectors generalize across capture sessions,"
- chapters/chapter2.tex: "\section{Autoencoders and Anomaly Detection}
\label{sec:ae}"
- chapters/chapter5.tex: "This thesis examines whether an autoencoder trained only on idle spectrum, captured with a low-cost RTL-SDR receiver, detects the occupancy of an LTE downlink band better than carefully tuned classical detectors."
- chapters/chapter5.tex: "Four are learned from idle frames: a convolutional autoencoder, a fully connected autoencoder, a variational autoencoder and a PCA detector."
- chapters/chapter3.tex: "Seven detectors are compared (\cref{tab:m-detectors}). Four are learned from idle data, two are classical energy detectors and one exploits the known LTE synchronization signals."

## Band, carrier and idle window confirmed (5 November 2026)

Replaced text:

- preamble.tex: "% The band is not yet chosen (CLAUDE.md, "Things to resolve").
\newcommand{\TargetBand}{\TBC{target band to be confirmed after the band survey}}"
- chapters/chapter3.tex: "The band studied is therefore the one with the best confirmed carrier at the site: \TargetBand."
- chapters/chapter3.tex: "For idle captures (Class~B), it is placed on a stretch of the surveyed range at least \qty{1.92}{\mega\hertz} wide with no carrier, as close as possible to the carrier, so that the idle and active captures are made at nearly the same tuner frequency."
- chapters/chapter3.tex: "A short logged run in \TargetBand{} illustrates occupancy over time in Chapter~4."
- appendices/appA_details.tex: "\TBC{The reasons for the choice of carrier, idle window and band, once confirmed by the author.}"
- CLAUDE.md: "## Things to resolve with me before writing (do not guess)
1. **Band:** not yet decided. Candidates and the survey protocol are in Section 8.1 of the proposal. Until I confirm a band, use placeholders in band-specific text and never state a band or center frequency."
- notes/method-design.md: "- **Positions** [fixed] (author, 4 November 2026):"

## Gains chosen (5 November 2026)

Replaced text:

- appendices/appA_details.tex: "because the power of a downlink carrier rises and falls with its traffic."
- chapters/chapter3.tex: "Two fixed gain settings are used: $g_1$, the primary setting, and $g_2$, used only for the gain-change test of \cref{sec:m-robustness}. Their values, \TBC{$g_1$ and $g_2$ in decibels}, are chosen during the survey so that the strongest active capture does not clip the 8-bit converter. For every capture the fraction of in-phase or quadrature samples at the extremes of the 8-bit range is recorded, and captures with more than \Prov{\pct{0.1}} clipped samples are flagged."
- chapters/chapter3.tex: "Across gain settings, the models and thresholds calibrated at $g_1$ are applied to the same test sessions' captures at $g_2$, which shifts the recorded noise floor by a controlled amount."
- chapters/chapter3.tex: "  \item \emph{Controlled-SNR mixtures.}"
- notes/method-design.md: "- **Gains** [field]: rule agreed 5 November 2026: g1 = highest tuner step without clipping (> 0.1 %) on the carrier, g2 about 10 dB below g1. Values pending (gain trials of 5 November 2026: see the conversation summary / Appendix A once fixed)."
- CLAUDE.md: "## Things to resolve with me before writing (do not guess)
1. **Gains g1 and g2:** rule agreed (g1 = highest tuner step without clipping, g2 about 10 dB below g1), values not yet fixed (see `notes/method-design.md`)."
- code/occupancy/config.py: "    g1_db: float | None = None                   # [field] primary gain, chosen in the survey
    g2_db: float | None = None                   # [field] gain-change test gain"

## Field guide: session plan replaced (5 November 2026)

At the author's request, the plan was replaced by a dated plan in time order starting with S01 on 5 November at 22:00 (the 07:00 slot that day was used by the survey). Old text:

```
There is one day and one night slot per day, so 12 sessions take at least 6 days. A possible plan, balanced over times and positions:

\begin{center}
\begin{tabular}{@{} l l l l @{\hspace{12mm}} l l l l @{}}
\toprule
Session & Day & Start & Position & Session & Day & Start & Position \\
\midrule
S01 & 1 & 07:00 & P1 & S07 & 4 & 07:00 & P2 \\
S02 & 1 & 22:00 & P1 & S08 & 4 & 22:00 & P1 \\
S03 & 2 & 07:00 & P2 & S09 & 5 & 07:00 & P1 \\
S04 & 2 & 22:00 & P2 & S10 & 5 & 22:00 & P2 \\
S05 & 3 & 07:00 & P1 & S11 & 6 & 07:00 & P2 \\
S06 & 3 & 22:00 & P2 & S12 & 6 & 22:00 & P1 \\
\bottomrule
\end{tabular}
\end{center}
```
