"""
budget.py
---------
Compute budget of the training (`train --time-epoch`).

On the selected device, for each main model (conv_ae, fc_ae, conv_vae), it
times one full training epoch on one fold (default fold 0): the training pass
and the validation pass, exactly as train.py runs them, after two warm-up
batches that are not timed. It also times writing one checkpoint to the
output directory (on Colab, Google Drive), which training does after every
epoch. --time-batches N times only N batches of each pass and scales the time
per frame up to the full epoch (a quick estimate on the laptop's CPU).

From the time per training frame, per validation frame and per checkpoint it
estimates the time of the whole design: every run of the configuration (main
models x folds x seeds, plus the latent-size ablations x folds x ablation
seeds; an ablation is costed as the main model of the same kind, whose cost
its latent size hardly changes), with each fold's own number of frames, for
30, 60 and 100 epochs per run (and the configured maximum, if different).
Early stopping ends most runs before the maximum of 100 epochs, so the
actual time lies within this range. Not included:
copying the pack, start-up, and the time the runtime sits idle. With
--units-per-hour X (the rate Colab shows for the connected runtime) the hours
are also given in Colab compute units; Colab charges for the time a runtime
is connected, whatever the GPU is doing. It also gives the amount of
checkpoint data written to the output directory (Google Drive on Colab):
a checkpoint is written after every epoch, and an fc_ae checkpoint holds its
8.5 million weights four times (the weights, the two Adam moments and the
best weights so far), about 136 MB.

The result is written as JSON (time_epoch.json) with the run environment and
provenance.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import provenance
from .config import Config
from .device import configure_determinism, describe, runtime_environment
from .models import build_model, count_parameters, main_specs
from .train import (CHECKPOINT, EarlyStopping, FrameSet, _epoch, _sync, checkpoint_state, expected_runs,
                    save_checkpoint, set_seed)

WARMUP_BATCHES = 2


def _time_model(spec, train_frames: FrameSet, val_frames: FrameSet, cfg: Config, device, ckpt_root: Path,
                max_batches: int | None) -> dict:
    t = cfg.train
    bs = t.batch_size
    set_seed(0)
    model = build_model(spec, cfg).to(device)
    lr = t.lr_vae if spec.kind == "conv_vae" else t.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=t.sched_factor,
                                                       patience=t.sched_patience)
    rng = np.random.default_rng(0)
    _epoch(model, train_frames.head(min(WARMUP_BATCHES * bs, len(train_frames))), cfg, opt, rng, device)
    _epoch(model, val_frames.head(min(bs, len(val_frames))), cfg, device=device)
    _sync(device)
    tr = train_frames if max_batches is None else train_frames.head(max_batches * bs)
    va = val_frames if max_batches is None else val_frames.head(max_batches * bs)
    t0 = time.perf_counter()
    res = _epoch(model, tr, cfg, opt, rng, device)
    _sync(device)
    t1 = time.perf_counter()
    va_res = _epoch(model, va, cfg, device=device)
    _sync(device)
    t2 = time.perf_counter()
    sched.step(va_res["loss"])
    stopper = EarlyStopping(t.patience, t.min_delta)
    stopper.step(va_res["loss"], 1)
    best = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    history = [{"epoch": 1, "train_loss": res["loss"], "val_loss": va_res["loss"]}]
    ck_dir = ckpt_root / f".time_epoch_{spec.name}"
    ck_dir.mkdir(parents=True, exist_ok=True)
    try:
        times = [save_checkpoint(ck_dir, checkpoint_state({"timing": True}, e, False, model, opt, sched, stopper,
                                                          best, history, rng, device, [], 0.0, 0.0, []))
                 for e in (1, 2)]                       # the second write also rotates the previous file
        ck_bytes = (ck_dir / CHECKPOINT).stat().st_size
    finally:
        shutil.rmtree(ck_dir, ignore_errors=True)
    n_tr, n_va = len(tr), len(va)
    return {"kind": spec.kind, "latent": spec.latent, "parameters": count_parameters(model),
            "train_frames_timed": n_tr, "val_frames_timed": n_va,
            "train_s": t1 - t0, "val_s": t2 - t1, "s_per_train_frame": (t1 - t0) / n_tr,
            "s_per_val_frame": (t2 - t1) / n_va, "checkpoint_s": times[-1], "checkpoint_bytes": ck_bytes}


def time_epochs(train_frames: FrameSet, val_frames: FrameSet, cfg: Config, device, fold: int,
                fold_sizes: list[dict], out_json: Path, models=None, max_batches: int | None = None,
                units_per_hour: float | None = None, data: str = "", synthetic: bool = False,
                log=print) -> dict:
    """Time one epoch per main model on one fold and estimate the training time of the design."""
    configure_determinism()
    device = torch.device(device)
    out_json = Path(out_json)
    specs = [s for s in main_specs(cfg) if not models or s.name in models or s.kind in models]
    if not specs:
        raise ValueError("no model selected for timing")
    log(f"timing one epoch per model on fold {fold} ({len(train_frames)} training, {len(val_frames)} validation "
        f"frames) on {describe(device)}" + (f", {max_batches} batch(es) per pass" if max_batches else ""))
    timed = {}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".time_epoch_", dir=out_json.parent) as ck_root:
        for spec in specs:
            r = _time_model(spec, train_frames, val_frames, cfg, device, Path(ck_root), max_batches)
            fs = next(f for f in fold_sizes if f["fold"] == fold)
            r["epoch_s"] = fs["train_frames"] * r["s_per_train_frame"] + fs["val_frames"] * r["s_per_val_frame"]
            timed[spec.name] = r
            log(f"  {spec.name:9s} {r['epoch_s']:9.2f} s per epoch on fold {fold} "
                f"(training {r['train_s']:.2f} s for {r['train_frames_timed']} frames, validation "
                f"{r['val_s']:.2f} s for {r['val_frames_timed']} frames); checkpoint "
                f"{r['checkpoint_s']:.2f} s ({r['checkpoint_bytes'] / 1e6:.1f} MB)")

    # whole design: every run of the configuration, each fold with its own frame counts
    sizes = {f["fold"]: f for f in fold_sizes}
    per_kind = {r["kind"]: r for r in timed.values() if r["latent"] == _main_latent(cfg, r["kind"])}
    runs = expected_runs(cfg, sorted(sizes), ablation=True)
    costed = [(k, spec) for k, spec, _ in runs if spec.kind in per_kind]
    skipped = sorted({spec.name for _, spec, _ in runs if spec.kind not in per_kind})

    def run_epoch_s(k, spec):
        r = per_kind[spec.kind]
        return (sizes[k]["train_frames"] * r["s_per_train_frame"] + sizes[k]["val_frames"] * r["s_per_val_frame"]
                + r["checkpoint_s"])

    epochs_options = sorted({30, 60, 100, int(cfg.train.max_epochs)})
    main = [(k, s) for k, s in costed if not s.ablation]
    estimate = {"epochs_per_run": epochs_options, "runs_total": len(runs), "runs_costed": len(costed),
                "runs_main": sum(1 for _, s, _ in runs if not s.ablation),
                "runs_ablation": sum(1 for _, s, _ in runs if s.ablation),
                "not_costed": skipped, "units_per_hour": units_per_hour, "hours": {}, "hours_main_only": {},
                "compute_units": {} if units_per_hour else None, "checkpoint_gb_written": {}}
    for e in epochs_options:
        h = e * sum(run_epoch_s(k, s) for k, s in costed) / 3600.0
        estimate["hours"][str(e)] = h
        estimate["hours_main_only"][str(e)] = e * sum(run_epoch_s(k, s) for k, s in main) / 3600.0
        estimate["checkpoint_gb_written"][str(e)] = e * sum(per_kind[s.kind]["checkpoint_bytes"]
                                                            for _, s in costed) / 1e9
        if units_per_hour:
            estimate["compute_units"][str(e)] = h * float(units_per_hour)
    env = runtime_environment(device)
    result = {"kind": "time_epoch", "device": str(device), "gpu_name": env["gpu_name"], "data": data,
              "fold": fold, "train_frames": len(train_frames), "val_frames": len(val_frames),
              "batch_size": cfg.train.batch_size, "timed_batches_per_pass": max_batches,
              "extrapolated": max_batches is not None, "warmup_batches": WARMUP_BATCHES, "models": timed,
              "fold_sizes": fold_sizes, "estimate": estimate,
              "notes": ["ablation runs are costed as the main model of the same kind",
                        "the estimate covers training passes, validation passes and checkpoint writes only",
                        "Colab bills connected runtime time, idle or not"],
              "environment": env, "synthetic_data": synthetic,
              "run": provenance.run_record(cfg, synthetic=synthetic)}
    provenance.write_json(out_json, result)
    log(f"design: {estimate['runs_total']} runs ({estimate['runs_main']} main + {estimate['runs_ablation']} "
        f"ablation) over {len(sizes)} folds" + (f"; not costed (kind not timed): {', '.join(skipped)}"
                                               if skipped else ""))
    for e in epochs_options:
        line = (f"  {e:3d} epochs per run: {estimate['hours'][str(e)]:8.2f} h "
                f"(main models only {estimate['hours_main_only'][str(e)]:.2f} h)")
        if units_per_hour:
            line += f", {estimate['compute_units'][str(e)]:.1f} compute units at {units_per_hour:g}/h"
        line += f"; checkpoints written: {estimate['checkpoint_gb_written'][str(e)]:.1f} GB"
        log(line)
    log(f"timing written to {out_json}")
    return result


def _main_latent(cfg: Config, kind: str) -> int:
    m = cfg.models
    return m.fc_ae.latent if kind == "fc_ae" else m.conv_ae.latent_channels
