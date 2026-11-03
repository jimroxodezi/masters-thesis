"""
train.py
--------
Training of the reconstruction models and fitting of the PCA baseline
(design section 7, "Training").

Per fold: the normaliser is fitted on the fold's training frames (Class B at
g1 of the training sessions) and shared by every model of the fold. Each model
is trained with Adam (lr 1e-3, 1e-4 for the VAE; weight decay 1e-5), batch 64,
at most 100 epochs, early stopping on the validation loss (patience 15,
min_delta 1e-6), ReduceLROnPlateau (factor 0.5, patience 7) and gradient
clipping at L2 norm 1.0; the weights with the lowest validation loss are kept.
Seeds 0-4 for the main models and 0-2 for the latent-size ablations.

Device. Training runs on the CPU or on a CUDA GPU (config train.device;
"auto" takes the GPU when there is one). The seed fixes the initial weights,
which are always drawn on the CPU before the model is moved to the device,
and the order of the training frames. device.py explains why a GPU run is not
guaranteed to be bit-identical when repeated.

Interruption and resume. Every run (model, latent size, fold, seed) has its
own directory, folds/fold<k>/models/<model>/seed<s>/. At the end of every
epoch the complete training state is saved to checkpoint.pt: the weights, the
optimiser, the learning-rate scheduler, the early-stopping state, the epoch,
the history, the best weights so far, and the random-number states of Python,
NumPy (global and the frame-order generator), PyTorch and CUDA. The file is
first written under a temporary name and then renamed, and the previous
checkpoint is kept as checkpoint_prev.pt, so an interruption in the middle of a
write (for example a Google Colab disconnect while writing to Google Drive)
never leaves only a damaged checkpoint. By default an unfinished run resumes
from the newest readable checkpoint, at the start of the next epoch; a
checkpoint written for other settings, data or code is refused rather than
resumed. On the CPU, an interrupted and resumed run ends in exactly the same
state as an uninterrupted one.

A run is finished when complete.json exists. It is written last, after
model.pt (the best weights), history.csv (one row per epoch) and
train_summary.json (results, device and environment, resume segments, code
version), and it records their sizes and SHA-256 checksums. The checkpoints
are deleted once the run is complete. A finished run is skipped unless
--force is given.
"""

from __future__ import annotations

import csv
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from . import provenance
from .config import Config
from .dataset import CaptureIndex, Fold, train_records, val_records
from .device import configure_determinism, describe, runtime_environment, select_device
from .models import ModelSpec, build_model, count_parameters
from .preprocess import FeatureStore, Normalizer, open_capture_store

CHECKPOINT = "checkpoint.pt"
CHECKPOINT_PREV = "checkpoint_prev.pt"
CHECKPOINT_TMP = CHECKPOINT + ".tmp"
CHECKPOINT_FILES = (CHECKPOINT, CHECKPOINT_TMP, CHECKPOINT_PREV)
MARKER = "complete.json"
FINAL_FILES = ("model.pt", "history.csv", "train_summary.json")
CHECKPOINT_FORMAT = "occupancy-train-checkpoint-1"
HISTORY_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "train_mse", "val_mse", "train_kl", "val_kl",
                  "grad_norm_mean", "clip_fraction", "epoch_s", "improved"]


class ResumeError(RuntimeError):
    """A checkpoint exists but was written for other settings, data or code."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds every CUDA device


def set_threads(n: int) -> None:
    torch.set_num_threads(int(n))


class FrameSet:
    """Concatenation of feature stores with (optionally strided) frame selection."""

    def __init__(self, stores: list[FeatureStore], normalizer: Normalizer, stride: int = 1):
        self.stores = stores
        self.norm = normalizer
        sel = [(i, np.arange(0, st.n, stride)) for i, st in enumerate(stores)]
        self.store_idx = np.concatenate([np.full(a.size, i) for i, a in sel]) if sel else np.zeros(0, int)
        self.frame_idx = np.concatenate([a for _, a in sel]) if sel else np.zeros(0, int)

    def __len__(self):
        return int(self.frame_idx.size)

    def head(self, n: int) -> "FrameSet":
        """The first n frames (for timing on part of an epoch)."""
        fs = FrameSet.__new__(FrameSet)
        fs.stores, fs.norm = self.stores, self.norm
        fs.store_idx, fs.frame_idx = self.store_idx[:n], self.frame_idx[:n]
        return fs

    def batch(self, positions: np.ndarray) -> torch.Tensor:
        positions = np.asarray(positions)
        out = np.empty((positions.size, 1, *self.stores[0].spec.shape[1:]), dtype=np.float32)
        si, fi = self.store_idx[positions], self.frame_idx[positions]
        for s in np.unique(si):
            m = si == s
            order = np.argsort(fi[m])                     # sorted reads from the memory map
            rows = np.asarray(self.stores[s].spec[fi[m][order]], dtype=np.float32)
            dest = np.flatnonzero(m)[order]
            out[dest, 0] = self.norm.transform(rows)
        return torch.from_numpy(out)

    def batches(self, batch_size: int, rng: np.random.Generator | None = None):
        n = len(self)
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for s in range(0, n, batch_size):
            yield self.batch(order[s:s + batch_size])


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.bad = float("inf"), 0, 0

    def step(self, loss: float, epoch: int) -> bool:
        """True if this epoch improved on the best loss by more than min_delta."""
        if loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = loss, epoch, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad": self.bad}

    def load_state_dict(self, d: dict) -> None:
        self.best, self.best_epoch, self.bad = float(d["best"]), int(d["best_epoch"]), int(d["bad"])


def fold_dir(work_dir: Path, k: int) -> Path:
    return Path(work_dir) / "folds" / f"fold{k}"


def model_dir(work_dir: Path, k: int, spec: ModelSpec, seed: int) -> Path:
    return fold_dir(work_dir, k) / "models" / spec.name / f"seed{seed}"


def fit_fold_normalizer(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path,
                        force: bool = False) -> Normalizer:
    path = fold_dir(work_dir, fold.k) / "normalizer.json"
    stores = [open_capture_store(work_dir, r.capture_id) for r in train_records(index, fold, cfg)]
    ids = [st.unit_id for st in stores]
    if path.exists() and not force:
        norm = Normalizer.load(path)
        if norm.sources == ids:
            return norm
    norm = Normalizer().fit(stores, frame_stride=cfg.train.frame_stride)
    norm.save(path)
    return norm


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _epoch(model, frames: FrameSet, cfg: Config, opt=None, rng=None, device=None) -> dict:
    """One pass over frames (training if opt is given). Losses are summed on the
    device in float64 and read once at the end, so the GPU never waits for the
    CPU between batches; the values equal a per-batch float64 sum."""
    train = opt is not None
    device = torch.device("cpu") if device is None else torch.device(device)
    gpu = device.type == "cuda"
    model.train(train)
    tot = torch.zeros((), dtype=torch.float64, device=device)
    n, parts_sum, grad_norms = 0, {}, []
    with torch.set_grad_enabled(train):
        for xb in frames.batches(cfg.train.batch_size, rng):
            if gpu:
                xb = xb.pin_memory().to(device, non_blocking=True)
            if train:
                loss, parts = model.loss(xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                grad_norms.append(gn.detach())
                opt.step()
            else:
                loss, parts = model.val_loss(xb)
            b = xb.shape[0]
            tot += loss.detach().double() * b
            n += b
            for k, v in parts.items():
                v = torch.as_tensor(v, dtype=torch.float64, device=device).detach()
                parts_sum[k] = parts_sum.get(k, 0.0) + v * b
    if n == 0:
        raise ValueError("empty data set")
    out = {"loss": float(tot) / n, **{k: float(v) / n for k, v in parts_sum.items()}}
    if train:
        g = torch.stack(grad_norms).cpu().numpy().astype(np.float64)
        out["grad_norm_mean"] = float(np.mean(g))
        out["clip_fraction"] = float(np.mean(g > cfg.train.grad_clip))
    return out


# ── random-number states ──────────────────────────────────────────────────────

def rng_state(data_rng: np.random.Generator, device: torch.device) -> dict:
    """Python, NumPy (global and the frame-order generator), PyTorch CPU and CUDA states."""
    g = np.random.get_state(legacy=False)
    return {"python": random.getstate(),
            "numpy_global": {"bit_generator": g["bit_generator"], "key": g["state"]["key"].tolist(),
                             "pos": int(g["state"]["pos"]), "has_gauss": int(g["has_gauss"]),
                             "gauss": float(g["gauss"])},
            "data_order": data_rng.bit_generator.state,
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "device_type": device.type}


def restore_rng_state(state: dict, data_rng: np.random.Generator, device: torch.device) -> list[str]:
    """Inverse of rng_state(); returns notes when the device type changed."""
    notes = []
    random.setstate(state["python"])
    g = state["numpy_global"]
    np.random.set_state({"bit_generator": g["bit_generator"],
                         "state": {"key": np.asarray(g["key"], dtype=np.uint32), "pos": g["pos"]},
                         "has_gauss": g["has_gauss"], "gauss": g["gauss"]})
    data_rng.bit_generator.state = state["data_order"]
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(state["torch_cuda"], device)
        else:
            notes.append("resumed on a GPU from a checkpoint written on the CPU: the CUDA random state "
                         "starts from the seed")
    elif state.get("device_type") == "cuda":
        notes.append("resumed on the CPU from a checkpoint written on a GPU")
    return notes


# ── checkpoints and completion ────────────────────────────────────────────────

def checkpoint_state(identity, epoch, finished, model, opt, sched, stopper, best_state, history, rng, device,
                     segments, elapsed_s, checkpoint_s, notes) -> dict:
    """Everything needed to continue a run after `epoch` exactly as if it had not stopped."""
    return {"format": CHECKPOINT_FORMAT, "identity": identity, "epoch": int(epoch), "finished": bool(finished),
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "stopper": stopper.state_dict(), "best_state": best_state, "history": history,
            "rng": rng_state(rng, torch.device(device)), "segments": segments, "elapsed_s": float(elapsed_s),
            "checkpoint_s": float(checkpoint_s), "notes": notes}


def save_checkpoint(run_dir: Path, state: dict) -> float:
    """Write checkpoint.pt atomically, keeping the previous one as checkpoint_prev.pt.

    Returns the seconds taken."""
    t0 = time.perf_counter()
    run_dir = Path(run_dir)
    cur, tmp = run_dir / CHECKPOINT, run_dir / CHECKPOINT_TMP
    with open(tmp, "wb") as f:
        torch.save(state, f)
    if cur.exists():
        provenance.replace_file(cur, run_dir / CHECKPOINT_PREV)
    provenance.replace_file(tmp, cur)
    return time.perf_counter() - t0


def load_checkpoint(run_dir: Path, log=print) -> dict | None:
    """The newest readable checkpoint (checkpoint.pt, a complete .tmp, or checkpoint_prev.pt)."""
    best, bad = None, []
    for name in CHECKPOINT_FILES:
        p = Path(run_dir) / name
        if not p.exists():
            continue
        try:
            st = torch.load(p, map_location="cpu", weights_only=True)
            if not isinstance(st, dict) or st.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("not a training checkpoint")
        except Exception as exc:  # noqa: BLE001 - a damaged file is skipped, never trusted
            bad.append(f"{name} ({exc.__class__.__name__})")
            continue
        if best is None or int(st["epoch"]) > int(best["epoch"]):
            st["_file"] = name
            best = st
    if bad:
        log(f"  note: unreadable checkpoint file(s) ignored: {', '.join(bad)}")
    return best


def is_complete(run_dir: Path) -> bool:
    """complete.json exists and the files it lists exist with the recorded sizes."""
    run_dir = Path(run_dir)
    try:
        marker = provenance.read_json(run_dir / MARKER)
    except (OSError, ValueError):
        return False
    files = marker.get("files") or {}
    for name in FINAL_FILES:
        p = run_dir / name
        if name not in files or not p.exists() or p.stat().st_size != files[name].get("bytes"):
            return False
    return bool(marker.get("complete"))


def _history_rows(path: Path) -> int:
    try:
        with open(path) as f:
            return max(sum(1 for _ in f) - 1, 0)
    except OSError:
        return 0


def run_status(run_dir: Path) -> dict:
    """complete | unfinished (with the epochs done) | not started."""
    run_dir = Path(run_dir)
    if is_complete(run_dir):
        m = provenance.read_json(run_dir / MARKER)
        return {"state": "complete", "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run")}
    if any((run_dir / n).exists() for n in CHECKPOINT_FILES + FINAL_FILES):
        return {"state": "unfinished", "epochs_done": _history_rows(run_dir / "history.csv")}
    return {"state": "not started"}


def clear_run(run_dir: Path) -> None:
    """Remove the outputs and checkpoints of one run (known file names only)."""
    run_dir = Path(run_dir)
    for name in FINAL_FILES + (MARKER,) + CHECKPOINT_FILES:
        for p in (run_dir / name, run_dir / (name + ".tmp")):
            p.unlink(missing_ok=True)


def _write_history(path: Path, rows: list[dict], durable: bool = False) -> None:
    def write(f):
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    provenance.atomic_write(path, write, binary=False, durable=durable)


def _cpu_state(model: torch.nn.Module) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _unique(items) -> list:
    out = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def _check_resume(saved: dict, current: dict, allow_code_change: bool) -> list[str]:
    """Refuse to resume a checkpoint written for another run, settings, data or code."""
    notes, problems = [], []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) == current.get(key):
            continue
        if key == "code_sha256" and allow_code_change:
            notes.append(f"resumed with different code: {str(saved.get(key))[:12]} -> {str(current.get(key))[:12]}")
            continue
        problems.append(key)
    if problems:
        hint = " (--allow-code-change accepts a code change)" if "code_sha256" in problems else ""
        raise ResumeError(f"the checkpoint in this run directory was written with a different "
                          f"{', '.join(problems)}{hint}; it is not resumed. Restart the run with --no-resume "
                          f"(or --force), or use another output directory.")
    return notes


class _DeterminismNotes:
    """Collects PyTorch's warnings about operations without a deterministic kernel
    (once per message) instead of printing them every batch."""

    def __init__(self, log):
        self.log, self.seen = log, []

    def __enter__(self):
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.filterwarnings("always", message=".*deterministic")
        original = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if "deterministic" not in text:
                return original(message, category, filename, lineno, file, line)
            first = text.strip().splitlines()[0][:300]
            if first not in self.seen:
                self.seen.append(first)
                self.log(f"  note: {first}")
        warnings.showwarning = show
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def train_model(spec: ModelSpec, train_frames: FrameSet, val_frames: FrameSet, cfg: Config,
                seed: int, out_dir: Path, meta: dict | None = None, log=print, *, device=None,
                resume: bool = True, force: bool = False, identity: dict | None = None,
                allow_code_change: bool = False, epoch_callback=None) -> dict:
    """Train one model (or resume it) and keep the weights with the lowest validation loss.

    resume:   continue an unfinished run from its newest checkpoint (default);
              False restarts it from scratch.
    force:    retrain even a finished run.
    identity: what else the run depends on (fold, data); with the spec, seed,
              training settings and code version it must match the checkpoint.
    epoch_callback(epoch) is called after each epoch's checkpoint is written.
    Returns the training summary ("status": trained, resumed or skipped).
    """
    t = cfg.train
    out_dir = Path(out_dir)
    tag = f"[{spec.name} seed {seed}]"
    if force:
        clear_run(out_dir)
    elif is_complete(out_dir):
        log(f"  {tag} already trained, skipped (use --force to retrain)")
        return {**provenance.read_json(out_dir / "train_summary.json"), "status": "skipped"}
    configure_determinism()
    device = select_device(t.device) if device is None else torch.device(device)
    set_threads(t.num_threads)
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(spec, cfg).to(device)        # initial weights drawn on the CPU from the seed
    lr = t.lr_vae if spec.kind == "conv_vae" else t.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=t.sched_factor,
                                                       patience=t.sched_patience)
    stopper = EarlyStopping(t.patience, t.min_delta)
    rng = np.random.default_rng(seed)
    code = provenance.code_version()
    ident = {"spec": spec.to_dict(), "seed": int(seed), "training_fingerprint": cfg.training_fingerprint(),
             "code_sha256": code["code_sha256"], **(identity or {})}
    env = runtime_environment(device)

    history, best_state, segments, notes = [], None, [], []
    epoch, finished, elapsed, ckpt_total = 0, False, 0.0, 0.0
    state = load_checkpoint(out_dir, log) if resume else None
    if state is None:
        if not resume and any((out_dir / n).exists() for n in CHECKPOINT_FILES):
            log(f"  {tag} --no-resume: existing checkpoint discarded, training from scratch")
        clear_run(out_dir)                  # no usable checkpoint: nothing stale may survive a fresh start
    else:
        notes = list(state.get("notes", [])) + _check_resume(state["identity"], ident, allow_code_change)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        stopper.load_state_dict(state["stopper"])
        history, best_state = list(state["history"]), state["best_state"]
        notes += restore_rng_state(state["rng"], rng, device)
        epoch, finished = int(state["epoch"]), bool(state["finished"])
        segments = list(state["segments"])
        elapsed, ckpt_total = float(state["elapsed_s"]), float(state["checkpoint_s"])
        log(f"  {tag} resumed after epoch {epoch} ({state['_file']}) on {describe(device)}")
    segment = {"from_epoch": epoch + 1, "to_epoch": epoch, "utc_start": provenance.utc_iso(), "utc_end": None,
               "device": str(device), "gpu_name": env["gpu_name"], "threads": torch.get_num_threads(),
               "torch": env["torch"], "torch_cuda_build": env["torch_cuda_build"],
               "cudnn_version": env["cudnn_version"], "code_sha256": code["code_sha256"]}
    segments.append(segment)
    first = True
    with _DeterminismNotes(log) as nondet:
        while not finished and epoch < t.max_epochs:
            epoch += 1
            cur_lr = opt.param_groups[0]["lr"]
            t0 = time.perf_counter()
            tr = _epoch(model, train_frames, cfg, opt, rng, device)
            va = _epoch(model, val_frames, cfg, device=device)
            _sync(device)
            dt = time.perf_counter() - t0
            sched.step(va["loss"])
            improved = stopper.step(va["loss"], epoch)
            if improved:
                best_state = _cpu_state(model)
            history.append({"epoch": epoch, "lr": cur_lr, "train_loss": tr["loss"], "val_loss": va["loss"],
                            "train_mse": tr.get("mse"), "val_mse": va.get("mse"), "train_kl": tr.get("kl"),
                            "val_kl": va.get("kl"), "grad_norm_mean": tr["grad_norm_mean"],
                            "clip_fraction": tr["clip_fraction"], "epoch_s": dt, "improved": int(improved)})
            finished = stopper.stop or epoch >= t.max_epochs
            elapsed += dt
            segment.update(to_epoch=epoch, utc_end=provenance.utc_iso())
            notes = _unique(notes + nondet.seen)
            ckpt_total += save_checkpoint(out_dir, checkpoint_state(
                ident, epoch, finished, model, opt, sched, stopper, best_state, history, rng, device,
                segments, elapsed, ckpt_total, notes))
            _write_history(out_dir / "history.csv", history)
            if first:
                rest = t.max_epochs - epoch
                log(f"  {tag} epoch {epoch} took {dt:.1f} s on {device.type}; at most {dt * rest / 3600:.2f} h "
                    f"for the remaining {rest} epoch(s)")
                first = False
            log(f"  {tag} epoch {epoch:3d} train {tr['loss']:.5f} val {va['loss']:.5f} lr {cur_lr:.1e} "
                f"{dt:.1f}s{' *' if improved else ''}")
            if epoch_callback is not None:
                epoch_callback(epoch)
    status = "resumed" if len(segments) > 1 else "trained"
    summary = _finalize(spec, seed, out_dir, cfg, model, stopper, best_state, history, segments, elapsed,
                        ckpt_total, _unique(notes), env, code, ident, meta, len(train_frames),
                        len(val_frames), lr)
    log(f"  {tag} complete: best epoch {summary['best_epoch']} of {summary['epochs_run']}, "
        f"val loss {summary['best_val_loss']:.5f}")
    return {**summary, "status": status}


def _finalize(spec, seed, out_dir, cfg, model, stopper, best_state, history, segments, elapsed, ckpt_total,
              notes, env, code, ident, meta, n_train, n_val, lr) -> dict:
    """Write model.pt, history.csv, train_summary.json and, last, complete.json."""
    t = cfg.train
    if stopper.best_epoch == 0 or best_state is None:
        raise RuntimeError(f"{spec.name} seed {seed}: no finite validation loss")
    t0 = time.perf_counter()
    ckpt = {"state_dict": best_state, "spec": spec.to_dict(), "epoch": stopper.best_epoch,
            "val_loss": stopper.best, "seed": int(seed), "trained_on": env, "segments": segments}
    provenance.atomic_write(out_dir / "model.pt", lambda f: torch.save(ckpt, f), durable=True)
    _write_history(out_dir / "history.csv", history, durable=True)
    synthetic = bool((meta or {}).get("synthetic_data", False))
    summary = {"spec": spec.to_dict(), "seed": int(seed), "best_epoch": stopper.best_epoch,
               "best_val_loss": stopper.best, "epochs_run": len(history),
               "stopped_early": stopper.stop and len(history) < t.max_epochs,
               "train_frames": n_train, "val_frames": n_val, "parameters": count_parameters(model),
               "lr_initial": lr, "wall_time_s": elapsed + ckpt_total + (time.perf_counter() - t0),
               "train_time_s": elapsed, "checkpoint_time_s": ckpt_total,
               "threads": torch.get_num_threads(), "device": env["device"], "gpu_name": env["gpu_name"],
               "environment": env, "segments": segments, "resumed": len(segments) - 1,
               "nondeterminism_notes": notes, "identity": ident,
               "training_fingerprint": cfg.training_fingerprint(), "code_version": code}
    provenance.write_json(out_dir / "train_summary.json",
                          {**summary, **(meta or {}), "run": provenance.run_record(cfg, synthetic=synthetic)},
                          durable=True)
    marker = {"complete": True, "run": f"fold{ident.get('fold')}/{spec.name}/seed{seed}",
              "fold": ident.get("fold"), "spec": spec.to_dict(), "seed": int(seed),
              "best_epoch": stopper.best_epoch, "epochs_run": len(history), "device": env["device"],
              "gpu_name": env["gpu_name"], "code_sha256": code["code_sha256"],
              "training_fingerprint": ident["training_fingerprint"], "synthetic_data": synthetic,
              "finished_utc": provenance.utc_iso(), "environment": env, "resumed": len(segments) - 1,
              "files": {n: provenance.file_record(out_dir / n) for n in FINAL_FILES}}
    provenance.write_json(out_dir / MARKER, marker, durable=True)
    for name in CHECKPOINT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    return summary


def load_model(path: Path, cfg: Config) -> tuple[torch.nn.Module, ModelSpec]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    spec = ModelSpec.from_dict(ck["spec"])
    model = build_model(spec, cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, spec


def specs_and_seeds(cfg: Config, models=None, ablation: bool = False, seeds=None):
    """(spec, seeds) pairs to train/score. models: list of names to restrict to."""
    from .models import ablation_specs, main_specs
    pairs = [(s, tuple(seeds) if seeds is not None else tuple(cfg.train.seeds)) for s in main_specs(cfg)]
    if ablation:
        pairs += [(s, tuple(seeds) if seeds is not None else tuple(cfg.train.ablation_seeds))
                  for s in ablation_specs(cfg)]
    if models:
        pairs = [(s, sd) for s, sd in pairs if s.name in models or s.kind in models]
    return pairs


def expected_runs(cfg: Config, fold_ids, models=None, seeds=None, ablation: bool = False) -> list[tuple]:
    """Every (fold, spec, seed) of the selection, in training order."""
    return [(int(k), spec, int(s)) for k in fold_ids
            for spec, sds in specs_and_seeds(cfg, models, ablation, seeds) for s in sds]


def data_identity(train_frames: FrameSet, val_frames: FrameSet, norm: Normalizer) -> dict:
    """The training and validation data of a fold, as recorded in checkpoints and summaries."""
    return {"train_units": [st.unit_id for st in train_frames.stores],
            "val_units": [st.unit_id for st in val_frames.stores],
            "train_frames": len(train_frames), "val_frames": len(val_frames),
            "mu": norm.mu, "sigma": norm.sigma}


def fold_frames_from_work(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path,
                          force: bool = False) -> tuple[FrameSet, FrameSet, Normalizer]:
    """Training and validation frames of a fold from the local feature stores."""
    norm = fit_fold_normalizer(index, fold, cfg, work_dir, force)
    tr_stores = [open_capture_store(work_dir, r.capture_id) for r in train_records(index, fold, cfg)]
    va_stores = [open_capture_store(work_dir, r.capture_id) for r in val_records(index, fold, cfg)]
    return FrameSet(tr_stores, norm, cfg.train.frame_stride), FrameSet(va_stores, norm, 1), norm


def train_runs(train_frames: FrameSet, val_frames: FrameSet, norm: Normalizer, fold: Fold, cfg: Config,
               root: Path, models=None, seeds=None, ablation: bool = False, force: bool = False,
               resume: bool = True, device=None, meta: dict | None = None, allow_code_change: bool = False,
               log=print) -> list[dict]:
    """Train (or resume, or skip) every selected run of one fold under root/folds/fold<k>/models/."""
    log(f"fold {fold.k}: train {fold.train} ({len(train_frames)} frames), val {fold.val} "
        f"({len(val_frames)} frames), test {fold.test}; mu={norm.mu:.3f} dB s={norm.sigma:.3f} dB")
    full_meta = {"fold": fold.to_dict(), "normalizer": norm.to_dict(), **(meta or {})}
    ident = {"fold": int(fold.k), "data": data_identity(train_frames, val_frames, norm)}
    results = []
    for spec, seeds_ in specs_and_seeds(cfg, models, ablation, seeds):
        for seed in seeds_:
            summ = train_model(spec, train_frames, val_frames, cfg, seed, model_dir(root, fold.k, spec, seed),
                               full_meta, log, device=device, resume=resume, force=force, identity=ident,
                               allow_code_change=allow_code_change)
            if summ.get("status") != "skipped":
                results.append(summ)
    return results


def train_fold(index: CaptureIndex, fold: Fold, cfg: Config, work_dir: Path, models=None,
               seeds=None, ablation: bool = False, include_pca: bool = True, force: bool = False,
               log=print, device=None, resume: bool = True, allow_code_change: bool = False) -> list[dict]:
    """Train the learned models of one fold from the local work directory, then fit PCA."""
    from .detectors import fit_pca_for_fold
    train_frames, val_frames, norm = fold_frames_from_work(index, fold, cfg, work_dir, force)
    results = train_runs(train_frames, val_frames, norm, fold, cfg, work_dir, models, seeds, ablation, force,
                         resume, device, {"synthetic_data": index.synthetic}, allow_code_change, log)
    if include_pca and (models is None or "pca" in models):
        fit_pca_for_fold(train_frames, cfg, fold_dir(work_dir, fold.k), force=force, log=log)
    return results
