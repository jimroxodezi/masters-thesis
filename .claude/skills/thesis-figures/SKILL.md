---
name: thesis-figures
description: Create figures for a telecom / SDR / deep-learning master's thesis, as quick previews in chat and as print-ready PNG, SVG and PDF files for Word or LaTeX. Covers block diagrams (processing pipeline, convolutional autoencoder, LTE EPC architecture, LTE frame structure), base-station and coverage drawings (tower, cells and handover, sector antennas, downtilt side view, two-tower overlap with signal-strength graph), and data plots (ROC, training loss, reconstruction-error histogram, IQ spectrogram). Use this skill whenever the user asks to draw, diagram, sketch, illustrate or plot anything for their thesis, report, chapter, slides or paper about LTE, 5G, base stations, SDR, RTL-SDR, spectrograms, autoencoders or anomaly detection, including vague asks like "draw me an example diagram", "add a figure", "make it simpler" or "export that for my thesis".
---

# Thesis figures

Produces figures for a master's thesis on SDR and deep-learning LTE signal anomaly detection, though the diagram types work for any telecom report. Two modes share one drawing:

1. **Preview**: draw inline in chat so the user can react quickly.
2. **Export**: once the user likes it, convert the same drawing into a white-background figure at print resolution, because chat drawings use theme variables that Word and LaTeX cannot render.

## Workflow

1. **Pick the figure from `references/catalog.md`.** If the request is vague ("a random LTE diagram"), choose the closest catalog entry rather than asking. Start with the simplest version that makes the point. The user has asked for "super simple, nothing serious" first and "more complicated" only on request, so escalate one step at a time and offer the next step in one line.
2. **Preview in chat.** Read the `diagram` module of the visualizer (`read_me`) and draw with `show_widget`, following its rules. Start from the matching file in `assets/templates/` instead of drawing from scratch: the templates are already verified to have no overlaps or clipping. Adapt labels, colours and counts to the request. One diagram per call, with a short prose paragraph between calls.
3. **Export when asked** ("export", "for my thesis", "Word", "high resolution", "PNG"), or offer it once after the user approves a design:
   - Save the final SVG to `/home/claude/work/<name>.svg` (viewBox must stay; keep chat-style classes and `var(--s)` colours, the exporter resolves them).
   - Run `python <skill-folder>/scripts/export_figure.py <svg> /mnt/user-data/outputs --name <fig-name> --width-in 6`. Add `--font serif` if the thesis body font is Times-like, `--grayscale` for black-and-white printing, `--formats svg,png,pdf` (default). Whitespace is trimmed automatically.
   - Call `present_files` on the PNG first, then the PDF/SVG. Do not publish these as web artifacts: the user needs files.
   - Install on first use if missing: `pip install cairosvg --break-system-packages`.
4. **Data plots** use `scripts/plot_figures.py` (`roc`, `loss`, `errors`, `spectrogram`). With the user's real CSV/NPY files it produces clean figures. Without files it draws synthetic data with a diagonal "EXAMPLE DATA, NOT REAL RESULTS" watermark. Say clearly that a watermarked figure only shows layout and must be regenerated from real data before it goes in the thesis. Never remove the watermark from synthetic data and never invent numbers (AUC, accuracy, thresholds) that the user did not produce.
5. **Hand over text with the figure**: a one-sentence caption in thesis style ("Figure 3.2: ..."), and a one-line alt text. Mention the standard to cite when the content is based on one (see catalog), because thesis examiners expect it.

## Style rules (apply to every figure)

- Flat, minimal, sentence-case labels; two font sizes (14 px labels, 12 px notes); at most three colour ramps per figure, with colour encoding meaning and a one-line legend when it does.
- Colour must never be the only cue: pair it with position, labels or line style, so the figure survives grayscale printing. Test with `--grayscale` when the user prints in black and white.
- Box subtitles five words or fewer. Detail belongs in the caption, not the box.
- Draw original schematics. Do not trace figures from 3GPP specifications, textbooks or papers; redraw the idea and cite the source in the caption ("adapted from 3GPP TS 36.211").
- Technical facts in a figure must be right. LTE FDD frame: 10 ms frame, ten 1 ms subframes, two 0.5 ms slots each, seven OFDM symbols per slot with normal cyclic prefix. If unsure about a parameter, search the spec or leave the number out and say so.
- Illustrative geometry (downtilt angle, beam width, overlap) is exaggerated for clarity. Add the words "not to scale" to the caption for those.

## Tailoring to the user's thesis

Use the user's own terms when a figure touches their pipeline: RTL-SDR capture, IQ samples, STFT spectrograms, PyTorch convolutional autoencoder trained on normal signals, reconstruction error versus a threshold. Check with the user before changing these names; they appear in the thesis text and the figure must match it.

## Files

- `references/catalog.md`: every figure type, its template, when to use it, caption starter, and recipes for figures without a template (SDR hardware chain, 5G NSA/SA comparison, handover sequence diagram, resource grid, evaluation flow).
- `assets/templates/*.svg`: verified starting points (nine diagrams).
- `scripts/export_figure.py`: chat-style SVG to print-ready SVG/PNG/PDF.
- `scripts/plot_figures.py`: ROC, loss, error histogram, IQ spectrogram.

## Using this skill in Claude Code (this thesis project)

Added when the skill was installed into `.claude/skills/` on 2 November 2026. The rest of this file was written for the claude.ai chat interface; in Claude Code:

- There is no `show_widget` or `present_files`. To preview, export a PNG to the session scratchpad and open it with the Read tool.
- Run the scripts through `uv` so that nothing is installed into the system Python or `code/.venv`:
  `uv run --no-project --with cairosvg --with pillow python .claude/skills/thesis-figures/scripts/export_figure.py figures/src/NAME.svg figures --name NAME --width-in 5.9 --font lmodern --formats pdf`
  (add `--with matplotlib --with numpy` for `plot_figures.py`).
- `--font lmodern` (Latin Modern Roman) matches the thesis body font; use it for every thesis figure.
- Keep SVG sources in `figures/src/` and exported PDFs in `figures/`. 5.9 in is the text width (A4, 35 mm left and 25 mm right margins).
- Check every technical fact in a figure against a primary source and cite that source in the caption. Never present synthetic data as results.
