# Figure catalog

Templates live in `assets/templates/`. "Level" tells you how to escalate: start at simple, move to detailed when asked.

## Ready-made templates

| Template | What it shows | Level | Caption starter |
|---|---|---|---|
| `base-station-simple.svg` | Phone, tower, radio waves | simple | "Figure X: A mobile phone connected to a base station over the radio interface." |
| `hex-cell-handover.svg` | Seven hexagonal cells, frequency groups, phone crossing a handover point | medium | "Figure X: Cellular layout with frequency reuse and a handover between neighbouring cells." |
| `sector-top-view.svg` | Three 120 degree sectors from one tower, top view | medium | "Figure X: Three-sector base station, top view." |
| `downtilt-side-view.svg` | Antenna downtilt, main beam, weak spot under the tower | medium | "Figure X: Antenna downtilt, side view (not to scale)." |
| `two-towers-signal-graph.svg` | Two towers with overlapping beams and signal curves with handover margin (hysteresis) | detailed | "Figure X: Beam overlap and signal strength between two base stations; handover triggers once the target exceeds the serving cell by the hysteresis margin (not to scale)." |
| `lte-epc-architecture.svg` | UE, eNodeB, MME, HSS, S-GW, P-GW, internet | detailed | "Figure X: LTE network architecture (adapted from 3GPP TS 23.401)." |
| `lte-frame-structure.svg` | Frame, subframe, slot, OFDM symbol hierarchy | detailed | "Figure X: LTE FDD frame structure with normal cyclic prefix (adapted from 3GPP TS 36.211)." |
| `thesis-pipeline.svg` | RTL-SDR capture to anomaly decision | thesis core | "Figure X: Processing pipeline of the proposed anomaly detector." |
| `conv-autoencoder.svg` | Encoder, latent code, decoder | thesis core | "Figure X: Convolutional autoencoder used for reconstruction-based anomaly detection." |

Changing a template: keep the viewBox width (680 for flowchart-style, 380 for illustrative ones), keep boxes at least `max(title_chars*8, subtitle_chars*7)+24` px wide, and re-check that no arrow crosses a box.

## Data plots (`scripts/plot_figures.py`)

| Command | Needs from the user | Caption starter |
|---|---|---|
| `roc` | CSV with `label` (1 = anomaly) and `score` | "Figure X: ROC curve of the detector on the test set." |
| `loss` | CSV with `epoch,train_loss,val_loss` | "Figure X: Training and validation loss per epoch." |
| `errors` | same CSV as `roc`; optional threshold | "Figure X: Reconstruction error of normal and anomalous samples with the decision threshold." |
| `spectrogram` | `.npy` of complex IQ, sample rate, optional centre frequency | "Figure X: STFT spectrogram of a captured LTE downlink segment." |

If the user has no data yet, produce the watermarked example so they can see the layout, and tell them what file to export from their pipeline (for `roc` and `errors`, one row per test sample).

## Recipes for figures without a template

Draw these with the visualizer using the same style rules, then export.

- **SDR hardware chain** (flowchart, left-to-right wraps to two rows): antenna, RTL-SDR dongle (tuner and ADC), USB, host PC with driver, Python pipeline. At most four boxes per row.
- **Evaluation flow** (flowchart): dataset, normal-only training split, test split with injected anomalies, threshold selection, metrics (precision, recall, F1, ROC-AUC). Colour: data grey, training teal, evaluation purple.
- **Handover sequence diagram** (SVG with three vertical lifelines: UE, source eNodeB, target eNodeB, plus MME if needed): measurement report, handover decision, handover request, acknowledgment, handover command, attach to target. Number the arrows; put message names in 12 px labels centred above each arrow. State in the caption that it is simplified.
- **Resource grid** (grid of small squares, time on x, subcarriers on y): one resource block is 12 subcarriers by 7 symbols; highlight one resource element and one resource block. Label axes "OFDM symbol" and "Subcarrier".
- **5G NSA versus SA** (two structural diagrams side by side, or two calls): NSA uses an LTE eNodeB anchor with the EPC and a 5G NR secondary node; SA uses gNB with the 5G core. Verify current terminology against 3GPP TS 37.340 and TS 23.501 before labelling interfaces.
- **Block diagram of a custom model or script**: use the `thesis-pipeline.svg` layout, one box per stage, and keep labels identical to the names in the code and thesis text.
