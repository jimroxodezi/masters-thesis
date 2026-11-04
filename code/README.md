# occupancy: LTE downlink occupancy detection with an RTL-SDR Blog V4

Research code for the MSc thesis. It tests whether an autoencoder trained only
on idle (no-LTE) spectrum captured with an RTL-SDR Blog V4 detects LTE downlink
occupancy better than classical detectors: fixed- and adaptive-threshold energy
detectors, PCA reconstruction and a PSS correlator. All detectors run on the
same frames and are compared at fixed false-alarm rates (1 % and 5 %) that are
calibrated on held-out idle data. Sessions are split by 4-fold cross-validation
and confidence intervals come from a session bootstrap.

The experiment is specified in `../notes/method-design.md`, which is the single
source of truth. Every number in it lives in `occupancy/config.py` and nowhere
else. The earlier pipeline (LTE treated as normal, synthetic jammer anomalies)
is kept unchanged in `legacy/`.

**Status:** no real data has been captured yet. Everything except the receiver
access has been run on synthetic data only. The capture code has been tested
against a fake device, not against the dongle. The Google Colab training path
(`pack`, `train --pack`, `unpack`, `colab/`) has been run end to end on the
laptop's CPU with synthetic data; it has not yet been run on a Colab GPU.

## Contents

```
occupancy/            the package (python -m occupancy <subcommand>)
  config.py           every setting, labelled [fixed] / [default] / [field] / [impl]
  capture.py          RTL-SDR capture: cu8 file + JSON sidecar, manual gain, bias tee off
  session.py          guided 8-capture session (design section 3)
  survey.py           rtl_power survey, CSV parser, plots, plateau and idle-window aids
  quicklook.py        per-capture quick-look figure; labelling-check template and agreement
  dataset.py          capture index (synthetic guard), frames, session-level folds
  preprocess.py       DC removal, STFT (128 x 128 dB), normaliser, float16 feature stores
  pss.py              PSS templates (roots 25, 29, 34) and FFT-based correlator
  injection.py        controlled-SNR mixtures x = alpha c + b (design section 6)
  models.py           conv_ae, fc_ae, conv_vae and the PER score
  train.py            training loop (Adam, early stopping, plateau scheduler, clipping),
                      per-epoch checkpoints, resume after interruptions, completion markers
  device.py           CPU or CUDA GPU, deterministic settings, run-environment record
  pack.py             training pack for Colab (Class B g1 spectrograms, folds, normalisers,
                      frozen config, checksums) and training from a pack
  unpack.py           checks and imports models trained on Colab into work/
  budget.py           train --time-epoch: seconds per epoch, estimated GPU-hours of the design
  detectors.py        scores of every detector, PCA, adaptive ED stream, theoretical ED threshold
  scoring.py          which frames each fold scores; score files
  calibrate.py        (1 - P_FA) quantile thresholds on idle validation frames
  evaluate.py         metrics, fold pooling, seeds, session bootstrap, H1, H2, result files
  latency.py          per-frame latency, throughput, real-time factor (design section 10)
  realtime.py         streaming demonstration with CSV log (design section 11)
  synthetic.py        synthetic LTE-like test signal (TESTING ONLY)
  provenance.py       git hash, code version (SHA-256 of the sources), library versions, CPU,
                      power mode in every output; checksums and atomic writes
colab/
  colab_train.py      the Colab steps (check-env, setup, time-epoch, train, status, zip-runs)
  train_on_colab.ipynb  thin notebook around colab_train.py (see "Training on Google Colab")
tests/                unittest suite (tiny inputs, about 1 minute)
legacy/               the earlier pipeline, unchanged
```

## Installation

1. Python environment (already locked): `uv sync` in `code/`. The project uses
   only the locked dependencies: numpy, scipy, scikit-learn, torch (CPU build),
   matplotlib and pyrtlsdr.
2. The V4 needs a librtlsdr with V4 (R828D) support. This laptop uses the
   RTL-SDR Blog fork (https://github.com/rtlsdrblog/rtl-sdr-blog, commit
   aed0ea1, source in `~/projects/rtl-sdr-blog`), built from source with its
   udev rules as in the fork's README (steps in thesis Appendix C), with the
   `dvb_usb_rtl28xxu`, `rtl2832` and `rtl2830` kernel modules blacklisted. The same build provides
   `rtl_power`, which the survey uses, and `rtl_eeprom`.
   Each capture sidecar records which librtlsdr file was loaded and whether
   it is the Blog build (`radio.librtlsdr`).
3. Check the receiver (this opens the dongle but does not stream):

   ```bash
   uv run python -m occupancy capture --info
   ```

   It should report product `Blog V4`, tuner `R828D`, `blog_build: True` and
   the tuner's gain steps. **Gains must be one of these steps**; any other
   value is refused, so that the logged gain is the applied gain.

All commands below are run from `code/` as `uv run python -m occupancy ...`.
`--help` works on every subcommand.

## Safeguards built into the code

- **Manual gain only.** `auto` (or any non-step value) is refused by capture,
  session, survey (`rtl_power -g` is always passed) and demo. The RTL2832
  digital AGC is switched off explicitly.
- **Bias tee off.** The V4 keeps its bias-tee state between programs, so it
  is switched off through `rtlsdr_set_bias_tee` before every capture and
  before every survey (rtl_power itself can only switch it on, with `-T`,
  which is never passed). The call is logged.
- **Tuning range.** A window that would exceed 1.766 GHz is refused, so LTE
  Band 3 (1805 MHz and above) cannot be captured by mistake.
- **Receiver identity.** A receiver that does not report a `Blog V4` product
  string and an R828D tuner is refused unless `--allow-other-receiver` is
  given; the mismatch is then written to the sidecar.
- **Complete files only.** The raw file is streamed to `<id>.cu8.partial`,
  renamed when complete, and the sidecar is written last. Raw file and sidecar
  are flushed to disk (power cuts). An interrupted capture is never indexed.
  A capture is refused if free disk space is below twice its size.
- **Synthetic data can never become a result.** Synthetic captures carry
  `"synthetic": true` and a warning in the sidecar. They live in
  `data_synthetic/`, and writing them under `data/` is refused. Every
  downstream command refuses them unless `--allow-synthetic` is given; with
  that flag all outputs go to `work_synthetic/` and carry
  `"synthetic_data": true`. Real and synthetic captures are never mixed.

## Field protocol (design sections 2 and 3)

### Before the first session (band survey)

1. Survey the candidate downlink ranges at the capture site, starting at
   07:00 (`--label day`) and at 22:00 (`--label night`) WAT, with a fixed gain:

   ```bash
   uv run python -m occupancy survey --range 758M:803M --gain 40.2 --label day --position P1 --duration 10m
   ```

   It first opens the dongle to switch the bias tee off, then runs
   `rtl_power`. It writes `data/survey/survey_<time>_<range>_<label>_<position>.csv`,
   a JSON sidecar and two PNGs (mean power against frequency, waterfall).
   Ctrl-C stops the sweep early, and the sweeps already written are kept and
   analysed. It also
   prints plateaus (LTE carriers appear as flat plateaus 5-20 MHz wide) and
   quiet stretches that can hold a 1.92 MHz window: every bin within 3 dB of
   the floor in the mean and 6 dB in the max hold, so narrow (e.g. GSM) and
   intermittent signals are avoided; the dongle's own spike at each rtl_power
   hop centre is ignored. An existing CSV can be
   re-analysed with `--parse FILE --carrier 796M`, which ranks candidate Class B
   windows by distance from the carrier. These lists are visual aids only.
2. Confirm the carrier with `CellSearch` from LTE-Cell-Scanner (Jiao's fork,
   commit e7f71cb, built in `~/tools/LTE-Cell-Scanner`; not part of this
   package; installation in thesis Appendix C). Search 0.5 MHz either side of
   each plateau at a fixed gain step:

   ```bash
   ~/tools/LTE-Cell-Scanner/build/src/CellSearch -s 795.5e6 -e 796.5e6 -g 40.2 -n 2
   ```

   Note CID (the PCI), fc (carrier centre) and nRB (6/15/25/50/75/100 =
   1.4/3/5/10/15/20 MHz). Use only `CellSearch`, never `LTE-Tracker`.
3. Choose **g1** (primary) and **g2** (gain-change test): two of the tuner's
   gain steps that do not clip on the strongest Class C capture. Short test
   captures (`capture --duration 2 ...`) report the clipping fraction; above
   0.1 % the capture is flagged.
4. Choose the **Class B** frequency: a stretch of the surveyed range at least
   1.92 MHz wide with no carrier (survey plus cell search), as close as
   possible to the carrier. CellSearch finds a carrier only at its centre, so
   check that the window lies outside every nearby carrier (centre +- half
   its bandwidth) rather than searching the window itself.
5. **C_offset** is placed at f_off = (0.54 MHz + W_occ/2)/2 from the carrier
   centre, with W_occ = N_RB x 0.18 MHz. This is feasible only for a channel
   bandwidth of 10 MHz or more (a 10 MHz carrier gives f_off = 2.52 MHz, window
   1.56-3.48 MHz from the centre). For a narrower carrier the session helper
   leaves C_offset out and records why.

### One session (8 captures, about 1 GB)

At least 12 sessions over at least 4 days, from at least 2 positions. Day
sessions start at 07:00 and night sessions at 22:00 WAT; `session` prints the
label and warns if the start is more than 2 h from both times (such a session
is labelled `off-schedule`). For each session:

1. Re-run cell search at the chosen carrier and keep the samples
   (`CellSearch -s <fc> -g <gain> -n 2 --recbin <session>_cellsearch.bin`):
   the carrier must be on air with the same CID. Run it before `session`;
   both need the dongle.
2. Start the guided session:

   ```bash
   uv run python -m occupancy session
   ```

   It asks for the session ID (the next free `S01`, `S02`, ...), position, g1,
   g2, ppm (default 0), antenna, temperature (optional) and notes. It then asks
   for the carrier centre, PCI and bandwidth, the C_offset side (+ or -), the
   Class B frequency and whether cell search at the Class B frequency found no
   cell. The Class A frequency defaults to the Class B frequency. The last
   answers are remembered in `data/field_defaults.json`.
3. It shows the plan and the disk space, then walks through the fixed order:
   **A g1, A g2** (50 ohm terminator on the antenna input), **B g1, B g2**
   (antenna connected), **C_centre g1, C_centre g2**, **C_offset g1,
   C_offset g2**. The order A, B, C_centre, C_offset is fixed (the adaptive
   energy detector processes each session in that order). Within a class, g1
   comes before g2, so the terminator is fitted and removed only once.
4. After each capture it prints the mean power and the clipping fraction (with
   a flag above 0.1 %) and writes `<capture>_quicklook.png` (mean spectrum and
   one spectrogram). Answer Enter to continue, `r` to retake or `q` to quit.
   A retaken capture is kept but marked `"status": "retaken"` and is never
   used. If a capture fails (USB error, full disk, invalid gain step) or is
   stopped with Ctrl-C, the helper reports it and offers retry or quit.
   Running `session` again with the same session ID resumes the session:
   captures already done are skipped.
5. Files go to `data/raw/<session>/`: one `<capture_id>.cu8` (interleaved
   uint8 I/Q, 1.92 MS/s, 30 s, about 115 MB) and one `<capture_id>.json` per
   capture, plus `session.json` (frequencies, cell search, C_offset plan,
   captures, anything not captured). Back the directory up after each session.

A single capture without the helper:

```bash
uv run python -m occupancy capture --session S01 --position P1 --class B --gain-role g1 \
    --gain 28 --freq 783M --cell-tool LTE-Cell-Scanner --cell-result "no cell found"
```

To rehearse the session flow without a dongle, use `session --synthetic`. It
writes SYNTHETIC captures to `data_synthetic/` only.

### Labelling check (a later day)

```bash
uv run python -m occupancy inspect            # quick-look PNGs + work/inspect/labelling_check.csv
# fill in second_pass_class (and cell_search_rerun) for each capture, then:
uv run python -m occupancy inspect --agreement work/inspect/labelling_check.csv
```

## Running the experiment

```bash
uv run python -m occupancy preprocess          # index, splits, spectrograms, PSS, injection sets
uv run python -m occupancy train               # per fold: normaliser, conv_ae/fc_ae/conv_vae (seeds 0-4), PCA
uv run python -m occupancy train --ablation --models conv_ae fc_ae   # latent-size ablations (seeds 0-2)
uv run python -m occupancy calibrate           # thresholds at P_FA 0.01 and 0.05 on validation frames
uv run python -m occupancy evaluate            # test scores, metrics, bootstrap, H1/H2 -> work/results/
uv run python -m occupancy latency             # latency, throughput, real-time factor (RQ4)
uv run python -m occupancy demo --detector conv_ae --freq 796M --gain 28 --pfa 0.01 --duration 120 --live
```

Each command skips work that is already done (`--force` redoes it). Training,
calibration and evaluation take `--models`, `--seeds` and `--ablation`; train
and calibrate also take `--folds`. Use the same selection in all three. Any
setting can be overridden with `--config my.json`, using the nested structure
of `config.py`, for example `{"train": {"max_epochs": 50}}`. Unknown keys are
rejected. A frozen configuration file is the way to record the final
`[default]` values.

`train --device auto|cpu|cuda` chooses where the neural networks train
(default `auto`: a CUDA GPU if PyTorch sees one, else the CPU; config
`train.device`). Every run checkpoints after every epoch and **resumes by
default**: after an interruption, the same command skips finished runs and
continues an unfinished one from its last checkpoint. `--no-resume` restarts
unfinished runs from scratch, `--force` retrains finished ones too. A
checkpoint written for other settings, data or code is refused, never
resumed. The training of conv_ae, fc_ae and conv_vae can also run on a Google
Colab GPU (next section).

**What the steps do** (design sections 4-9):

- *preprocess* indexes `data/raw` and writes `work/index.json` and
  `work/splits.json`. The sessions, in natural order, are cut into 4
  contiguous test groups. For fold k, validation is the first 2 sessions of
  test group k+1 (cyclic), and training is the rest. Each capture is cut into
  non-overlapping frames of N = 16384 samples (8.533 ms). Each frame is
  DC-removed, and the step computes the 128 x 128 dB spectrogram (periodic
  Hann, hop 128, 15 kHz bins, stored with frequency increasing upwards: row
  0 = +945 kHz), the frame energy T and the PSS statistic. These go to
  `work/features/<capture>/` (float16 memory map). It also builds the
  injection sets: 16 SNR levels x 500 mixtures per session, gain and source
  (C_centre and C_offset kept separate).
- *train* fits the normaliser per fold on Class B frames at g1 of the
  training sessions only. It trains with early stopping on the validation
  sessions and writes `work/folds/fold<k>/models/<model>/seed<s>/`
  (`model.pt`, `history.csv`, `train_summary.json`). PCA keeps the components
  that explain 95 % of the training variance, capped at 256.
- *calibrate* scores the validation frames (Class B at g1 of the validation
  sessions) and sets each threshold to the empirical (1 - P_FA) quantile. For
  the adaptive ED, kappa is calibrated on the validation stream. For ed_fixed,
  the theoretical chi-square threshold is also stored, for comparison only.
  Output: `work/folds/fold<k>/thresholds.json` and
  `work/results/thresholds.csv`.
- *evaluate* scores the test sessions (A, B, C at g1 and g2, plus the
  injection sets) and writes to `work/results/`:

| file | content |
|---|---|
| `metrics_summary.csv` | per detector, P_FA, gain and eval set: P_D, realized P_FA, precision, recall, F1, AUC as mean, SD over seeds and 95 % session-bootstrap interval; n_pos, n_neg (class balance) |
| `metrics_by_seed.csv`, `metrics_by_fold.csv` | the same per seed (pooled over folds) and per fold, with confusion counts; the per-fold file adds the AUC on raw scores within the fold |
| `pd_vs_snr.csv` | P_D by SNR stratum (grid level nearest SNR_eff) and source |
| `paired_differences.csv` | conv_ae minus every other detector (P_D, AUC), paired bootstrap intervals |
| `h1.json/.csv`, `h2.json/.csv` | the hypothesis tests exactly as defined in design section 9 |
| `noise_floor.csv` | Class A and B median frame power at g1 and g2 per session, and the change |
| `ed_theoretical_threshold.csv` | chi-square vs empirical ed_fixed thresholds and their realized P_FA |
| `roc_curves.csv`, `pr_curves.csv` | curves on fixed grids, per detector and seed |
| `positions.csv` | P_D and P_FA per position (when there are several) |
| `failure_samples.csv` | a reproducible sample of false alarms and misses (`--failure-figures` adds spectrogram PNGs) |
| `run.json` | configuration, seeds, folds, git hash, library versions, CPU |

Eval sets: `natural` (test B vs C_centre and C_offset), `natural_centre`,
`natural_offset`, `inj_<all|centre|offset>_<level>dB`, `inj_*_low`
(SNR <= -4 dB, H1), `inj_*_alllevels` and `receiver_A` (Class A only, the
receiver-only false-alarm check). Each set exists at g1 and at g2 (the g2
sets use the g1 thresholds). The adaptive ED is decision-directed, so its
statistic is computed per target P_FA; it appears as variants
`ed_adaptive@0.01` and `ed_adaptive@0.05`.

**Latency** (`latency`) uses 4 threads, 50 warm-up frames and 1000 frames at
batch 1 (median and 95th percentile), timing preprocessing, scoring and end
to end separately. It also measures throughput at batch 64 and the real-time
factor against 8.533 ms. CPU, threads, versions and power mode are logged.
Run it with the laptop on mains power and the Performance power profile
(`powerprofilesctl set performance`), with other programs closed: `latency`
refuses any other known power state (`--ignore-power` for test runs only),
because on battery with Power Saver the networks run 4-5 times slower.
Without trained models it times freshly initialised models (same arithmetic)
and says so. Every score (evaluation, latency, demo) uses the scoring copy
of a model (`models.inference_model`: BatchNorm folded into the convolutions,
channels-last layout), which gives the same scores up to floating-point
rounding. Before any timing, every detector runs for 5 s (`settle_s`), which
removes a start-up effect that slowed the first detector timed. Each neural
network is also timed as trained, alternating frame by frame with its scoring
form; `latency.json` (`optimisation`) gives both medians and their ratio.

**Demo** (`demo`) streams 16384-sample blocks from the dongle and applies one
detector with its calibrated threshold (`--fold`, `--seed`, `--pfa`). It logs
every frame to `work/results/demo/*.csv` (UTC time, frequency, gain, score,
threshold, decision, processing time, dropped blocks). Use the g1 gain, on
mains power with the Performance profile (the demo warns otherwise). A
32-frame queue (0.27 s) separates the reader from the processing; if the
detector falls behind, the OLDEST blocks are dropped and counted, so the
decisions stay current. Samples lost before reaching the program (USB
overflow) are estimated from the block arrival times and reported at the end
(`missing_blocks_estimate`). `--live` shows a rolling spectrum and an
occupancy strip, redrawn at most 4 times per second and never while frames
wait.

## Training on Google Colab

The neural networks (conv_ae, fc_ae, conv_vae and the latent-size ablations)
can be trained on a Colab GPU, because on the laptop's CPU the design takes on
the order of weeks (next section). Everything else stays on the laptop:
capture, preprocessing, PCA, the energy detectors, the PSS correlator,
calibration, evaluation, latency and the demo. Colab never sees raw IQ, Class
A or C captures, g2 captures or test injection sets, and nothing on Colab
needs `pyrtlsdr`.

**1. Laptop: pack and upload**

```bash
uv run python -m occupancy preprocess
uv run python -m occupancy pack          # -> work/colab/train_pack/, train_pack.zip, occupancy_code.zip
```

`pack` writes the training pack: the float16 spectrograms of every session's
Class B capture at g1 (once per session), the folds, each fold's
normalisation statistics (computed here), the frozen configuration, the code
version and a manifest with the size and SHA-256 checksum of every file. It
prints the pack size and records the pack under `work/packs/`. Upload
`work/colab/train_pack.zip` and `work/colab/occupancy_code.zip` to Google
Drive, folder `MyDrive/msc-thesis/`, together with
`colab/train_on_colab.ipynb` (or open the notebook from the laptop with
Colab's *File > Upload notebook*). After any change to the data, the
configuration or the code, run `pack --force` and upload both zips again.

**2. Colab: open the notebook and run the cells**

Open `train_on_colab.ipynb` in Colab, choose a GPU runtime (*Runtime > Change
runtime type*), and run the cells in order. They mount Drive, check the GPU,
copy and unzip the code and the pack to the runtime's local disk (verifying
every checksum), install only missing packages (never PyTorch), time one
epoch per model, train, show progress, and zip the outputs. All paths are
set in the first cell (`MyDrive/msc-thesis/`). The training cell runs

```bash
python colab/colab_train.py train --pack /content/msc-thesis/train_pack \
    --out /content/drive/MyDrive/msc-thesis/runs --ablation --max-attempts 5
```

which calls `python -m occupancy train --pack ... --out ... --device cuda`
for the main models and then for the ablations (on a runtime without a GPU it
stops at once rather than training on the CPU). Every epoch of every run is checkpointed
to `runs/` on Drive (written to a temporary file and renamed, keeping the
previous checkpoint), and a finished run gets `complete.json`. **After a
disconnect**, reconnect and run the cells again (the timing cell can be
skipped): finished runs are skipped and the unfinished run continues from its
last checkpoint. Failed attempts (for example a Drive I/O error) are retried;
refusals (another pack, other settings, other code) are not. The last cell
writes `MyDrive/msc-thesis/runs.zip` (without checkpoints).

**3. Laptop: download, import, continue**

```bash
uv run python -m occupancy unpack --runs ~/Downloads/runs.zip
uv run python -m occupancy train --models pca     # PCA is fitted on the laptop
uv run python -m occupancy calibrate --ablation   # leave out --ablation if the ablations were not trained
uv run python -m occupancy evaluate --ablation
```

`unpack` (alias `import-models`) accepts the zip or the unzipped `runs`
directory. Before copying anything into `work/folds/` it checks the pack ID
against `work/packs/` (or `--pack DIR`), the folds and their captures, the
training settings (other configuration differences are listed), the code
version of the pack, of every run and of the laptop, the normalisation
statistics, and the completion marker and checksums of every expected run
(model x latent size x fold x seed; the ablations when any ablation run is
present or `--ablation` is given). It lists every missing, unfinished or
damaged run and exits with code 1 if anything is missing; complete runs are
imported, so it can be run again later. The report is
`work/colab_import.json`. Code that differs is refused unless
`--allow-code-change` is given, and is then recorded with the changed files.

**Data sizes** (design defaults, 30 s captures): a session's Class B g1
spectrograms are 115 MB (3515 frames of 128 x 128 float16), so the pack is
1.38 GB for 12 sessions; the zip is somewhat smaller (82 % of the directory
on synthetic data; `pack` prints both). Trained models: 0.21 MB per conv_ae or
conv_vae run and 34 MB per fc_ae run, about 1.5 GB for all 108 runs (60 main
runs: 3 models x 4 folds x 5 seeds; 48 ablation runs: 4 variants x 4 folds x
3 seeds), most of it fc_ae weights, which hardly compress. During an fc_ae run
its checkpoints take up to about 0.27 GB on Drive (current and previous, 136
MB each); checkpoints are deleted when a run completes. On the laptop, the
pack directory takes little extra space (the spectrograms are hard links to
`work/features` when possible) but the zip is a copy.

**Compute units.** Colab bills the time a runtime is connected, idle or not.
The timing cell (`train --time-epoch`) measures one epoch of each model on
the assigned GPU, including a checkpoint write to Drive, and estimates the
hours of the whole design for 30, 60 and 100 epochs per run (early stopping
ends most runs earlier); with `--units-per-hour` (the rate Colab shows for
the runtime) it converts them to compute units. The result is saved in
`runs/time_epoch.json` and copied to `work/results/time_epoch_colab.json` by
`unpack`. Copying the pack, start-up and idle time come on top. The estimate
also gives the checkpoint data written to Drive. An fc_ae epoch is short on
a GPU, but its checkpoint is about 136 MB, so for fc_ae the checkpoint write
can take as long as the epoch itself; the 44 fc_ae runs (main and ablation)
write about 6 GB to Drive per epoch of training, for example about 270 GB if
they average 45 epochs. Each checkpoint replaces the previous one, so the
space used on Drive stays small. Run only one training notebook at a time on
the same `runs` folder.

**Reproducibility.** Training sets `torch.backends.cudnn.deterministic`,
switches off `cudnn.benchmark`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
CUDA starts and calls `torch.use_deterministic_algorithms(True,
warn_only=True)`. The backward pass of bilinear upsampling (the conv_ae and
conv_vae decoders) has no deterministic CUDA kernel, so **GPU runs are not
guaranteed to be bit-identical** when repeated; the architecture is not
changed to avoid this. The seeds still fix the initial weights (drawn on the
CPU) and the order of the training frames. On the CPU, a run is repeatable,
also across an interruption and resume (tested). Every run records in
`train_summary.json` the device, GPU name, CUDA, cuDNN and driver versions,
PyTorch and the other library versions, Python, the determinism flags, the
code version (SHA-256 of the source files, plus the git state) and every
resume (`segments`). Should Colab assign a different GPU after a reconnect,
the run continues on it, and the change is recorded there.

**Rehearsal without real data:** `synthetic`, then `preprocess --allow-synthetic`
and `pack --allow-synthetic` write `work_synthetic/colab/train_pack_SYNTHETIC.zip`;
set `PACK_ZIP` to that name and `ALLOW_SYNTHETIC = True` in the notebook, and
use `unpack --allow-synthetic` on the laptop. Synthetic and real packs, runs
and work directories are never mixed.

## Computational budget (measured while other programs were running)

On this laptop, with 4 threads and other programs open, one conv_ae or
conv_vae training step of 64 frames took about 2.4 s. One epoch over the
training frames of a 12-session fold (about 24,600 frames) would therefore
take about 16 minutes. Inference took about 14 ms per frame at batch 64. The
fc_ae is about 15 times cheaper. With the design defaults (100 epochs at most,
5 seeds, 4 folds, 500 mixtures per level), the convolutional models alone
need on the order of weeks of CPU time. Before the full run, decide whether to
subsample training frames (`train.frame_stride`, default 1 = all frames, as in
the design), reduce `train.max_epochs`, `injection.mixtures_per_level` or the
number of seeds, or use other hardware. Any such change must be reported in
the thesis. `train` prints the duration of the first epoch, and
`train --time-epoch` (with `--time-batches 20` for a quick CPU estimate)
times one epoch per model and estimates the hours of the whole design on the
selected device (see "Training on Google Colab").

## Synthetic data (testing only)

```bash
uv run python -m occupancy synthetic --sessions 8 --frames 8      # writes data_synthetic/raw
uv run python -m occupancy preprocess --allow-synthetic           # ... and so on, outputs in work_synthetic/
```

The generator produces an LTE-like OFDM signal (1.4 MHz numerology at
1.92 MS/s, PSS on the last symbol of slots 0 and 10, an SSS-like sequence,
CRS-like pilots, random resource-block load) in complex Gaussian noise,
quantised to 8 bits. It exercises the pipeline only. It is not a model of a
real carrier, and nothing computed from it is a thesis result.

## Tests

```bash
cd code
uv run python -m unittest discover -s tests
```

The tests (86) use tiny inputs, 2 torch threads and synthetic data in
temporary directories, and take about a minute on the laptop. They cover the
PSS templates and correlator, the STFT (shape, 15 kHz bins, orientation), the
normaliser, the fold rotation, the quantile thresholds, the theoretical ED
threshold, injection SNR and admissibility, the adaptive ED stream and kappa
calibration, the bootstrap (reproducibility, AUC), the synthetic guard, the
models and the PER score, capture I/O with a fake device, the survey parser,
the guided session (with a retake) and an end-to-end run including latency
and demo. For training elsewhere they cover: resume after an interruption
(also with a damaged newest checkpoint) ending in exactly the state of an
uninterrupted run; refusal to resume a checkpoint of other settings or data;
device `auto` falling back to the CPU; pack contents (Class B g1 only, once per
session) and detection of a corrupted pack; pack -> training through the Colab
script with `pyrtlsdr` unimportable -> unpack -> calibrate, including refusal
of other settings or code and the report of missing runs; training from the
pack giving the same weights as training from the work directory; the
`--time-epoch` output; the notebook; and that no training module imports
`pyrtlsdr`.

## Interpretations of the design (also in the code comments)

- The conv_ae pools after each of its three stages (needed for the stated
  8 x 16 x 16 latent). The 1x1 bottleneck is linear, the decoder starts with
  a 1x1 conv + BN + ReLU, and upsampling is bilinear, as in the original model.
- The fc_ae latent layer is linear. The VAE validation loss uses z = mu.
- The PCA solver is IncrementalPCA when the training matrix exceeds 1 GB.
- Thresholds use the `inverted_cdf` quantile (at most P_FA of the
  calibration scores exceed it). Decisions are `score > threshold`.
- Adaptive ED: the buffer starts as 64 copies of the validation median. Kappa
  is the fixed point of "run the stream at kappa, take the quantile". Each
  injection mixture is scored against the stream's noise estimate just before
  its idle frame b. Class A is scored as a separate stream.
- The PSS statistic uses the lags at which the template lies wholly inside the
  frame.
- Folds use contiguous test groups of sessions in natural order.
- Pooled AUC, ROC and PR use scores mapped through each fold's validation
  ECDF. The bootstrap averages seeds inside each replicate. The H2 P_D
  departure is the mean over matched SNR levels.
- The first 131072 samples after streaming starts (68 ms) are discarded. The
  sidecar's UTC start is accurate to about one USB block (68 ms).
- Training: the initial weights are always drawn on the CPU and then moved to
  the device, so a seed gives the same initial weights on the CPU and the GPU.
  Losses are summed on the device in float64 and read once per epoch (the
  same values as a per-batch sum). A resumed run restarts at the beginning of
  the epoch after its last checkpoint. The checkpoint identity is the model,
  latent size, seed, fold, training data (captures, frame counts,
  normalisation statistics), training settings and code version.
- "Same configuration" for models trained elsewhere means the same training
  settings: preprocessing, splits, the learned models and training (not
  paths, device or threads, nor the detectors that are not trained).
