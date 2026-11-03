# Chapter 5 (Conclusion and Recommendations): outline for approval

Status: **approved by the author on 2 November 2026 (D1, D2).** The [now] parts and the Table 5.1 frame are drafted in `chapters/chapter5.tex`; the rest are red `\TBC` notes. CLAUDE.md chapter plan item 5: summary of findings, contributions, recommendations, future work.

Like Chapter 4, most of this chapter depends on results that do not exist yet. Each section below is marked **[now]** (can be drafted before data, from Chapters 1–3) or **[after results]**. Nothing marked [after results] will be written until the evaluation has run, and no finding will be anticipated.

## Structure (target ~2,000–2,500 words, no new citations expected)

### 5.1 Summary of the Study (~300 words) [now]
Problem, aim and method in brief: real RTL-SDR captures, idle-trained autoencoders, matched classical baselines, session-level evaluation. Restates Chapters 1 and 3 and needs no results.

### 5.2 Findings by Research Question (~700 words) [after results]
- One paragraph per RQ1–RQ5, and the verdicts on H1 and H2, each pointing to its Chapter 4 table.
- → **Table 5.1:** each research question and hypothesis, its answer, and the evidence (table or figure). Generated or checked against the Chapter 4 outputs.

### 5.3 Contributions (~400 words) [after results]
- The contributions listed in Section 1.6, revised to what was actually delivered: the dataset (if released), the benchmark protocol (session-level splits, idle-calibrated thresholds, controlled-SNR set), the comparison itself, and the code.
- The Chapter 1 TODO ("revisit contributions after Chapter 4") is resolved here and in Section 1.6 together.

### 5.4 Recommendations (~500 words) [after results]
- For practitioners who monitor occupancy with low-cost receivers: which detector to use under which conditions, and how to calibrate it. These follow from the results, so they cannot be written now.
- For researchers evaluating learned detectors:
  - [now] Methodological recommendations that follow from the design and from Chapter 2's review: session-level splits, thresholds calibrated on held-out idle data, matched classical baselines, uncertainty over sessions.
  - [after results] Confirming them with what the study found.

### 5.5 Limitations and Future Work (~600 words) [now, with additions after results]
- [now] Directions that follow from the scope set in Chapter 1:
  - several receivers and cooperative sensing;
  - other bands and receivers (e.g. Band 3 with a receiver that tunes above 1.766 GHz);
  - the cyclostationary and eigenvalue detectors reviewed but not implemented;
  - other learned models (generative adversarial networks, diffusion models);
  - longer-term occupancy logging;
  - embedded deployment (excluded from this study).
- [after results] Directions suggested by the results, e.g. adaptive normalization for the autoencoder if H2 is not supported, or the specific failure causes found.

### 5.6 Closing Remarks (~150 words) [after results]

## Decisions needed

- **D1. Draft the [now] parts immediately (recommended):** 5.1, the methodological part of 5.4 and the scope-based part of 5.5, in the present tense, with red `\TBC` markers where results will go. Alternatively, wait and write the whole chapter after the results.
- **D2. Table 5.1 (recommended):** a one-page table answering each research question and hypothesis, with pointers to the evidence. Alternatively, prose only.
