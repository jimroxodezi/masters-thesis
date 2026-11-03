"""
cli.py
------
python -m occupancy <subcommand> [options]     (run from code/, e.g. with "uv run")

Field work          survey, capture, session, inspect
Experiment          preprocess, train, calibrate, evaluate
Training elsewhere  pack (laptop: data for Colab), train --pack (Colab GPU), unpack (laptop: import models)
Cost and demo       latency, demo, train --time-epoch
Testing only        synthetic   (writes synthetic captures under data_synthetic/)

Downstream commands refuse synthetic captures unless --allow-synthetic is given;
with it, data come from data_synthetic/ and every output goes to work_synthetic/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CLASSES, GAIN_ROLES, Config


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_cfg(args) -> Config:
    return Config.load(args.config)


def _index_and_work(args, cfg: Config):
    from .dataset import SyntheticDataError, index_captures
    allow = bool(getattr(args, "allow_synthetic", False))
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir(synthetic=allow)
    index = index_captures(data_dir, cfg, allow_synthetic=allow)
    work = Path(args.work_dir) if args.work_dir else cfg.work_dir(index.synthetic)
    if index.synthetic and work.resolve() == cfg.work_dir(False).resolve():
        raise SyntheticDataError("synthetic data may not be processed into the real work directory")
    if index.synthetic:
        print("*** SYNTHETIC DATA: outputs are for testing only and are written to", work, "***")
    return index, work


def _folds(index, cfg: Config, which=None):
    from .dataset import make_folds
    folds = make_folds(index.sessions(), cfg)
    return [f for f in folds if which is None or f.k in which]


def _freq(s):
    from .capture import parse_freq
    return parse_freq(s)


def _range(s):
    a, b = s.split(":")
    return _freq(a), _freq(b)


def _seeds(values):
    return None if values is None else [int(v) for v in values]


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_survey(args):
    from . import survey
    cfg = _load_cfg(args)
    if args.parse:
        survey.analyse_survey(cfg, Path(args.parse), carrier_hz=args.carrier)
        return 0
    if not args.range or args.gain is None:
        raise SystemExit("survey needs --range START:STOP and a fixed --gain (or --parse CSV)")
    f0, f1 = args.range
    out = Path(args.out) if args.out else cfg.data_dir(False) / "survey"
    # the V4 keeps its bias-tee state between programs and rtl_power never switches it off
    from .capture import receiver_info
    bias = receiver_info(cfg, args.device)["bias_tee"]
    print(f"bias tee off before the sweep: {bias}")
    survey.run_survey(cfg, f0, f1, args.gain, out, label=args.label or "", position_id=args.position or "",
                      notes=args.notes or "", carrier_hz=args.carrier, bin_hz=args.bin,
                      integration_s=args.integration, duration=args.duration, device_index=args.device,
                      bias_tee_note=f"off: {bias}; -T not passed")
    return 0


def cmd_capture(args):
    from . import capture, provenance
    cfg = _load_cfg(args)
    if args.info:
        info = capture.receiver_info(cfg, args.device)
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0
    missing = [n for n in ("session", "position", "cls", "gain_role", "gain", "freq") if getattr(args, n) is None]
    if missing:
        raise SystemExit("capture needs --" + ", --".join(m.replace("cls", "class").replace("_", "-")
                                                         for m in missing))
    cell = {k: v for k, v in (("tool", args.cell_tool), ("pci", args.pci), ("carrier_centre_hz", args.carrier),
                              ("bandwidth_mhz", args.bandwidth), ("result", args.cell_result)) if v is not None}
    extra = {}
    if args.cls == "C_offset" and args.carrier is not None:
        extra["c_offset_plan"] = capture.c_offset_plan(cfg, args.carrier, args.bandwidth)
    if args.ppm is not None:
        cfg.capture.ppm = args.ppm
    if args.synthetic:
        import numpy as np
        from .synthetic import SessionSignalParams, stable_seed, write_synthetic_capture
        out = (Path(args.data_dir) if args.data_dir else cfg.data_dir(True)) / "raw" / args.session
        t = provenance.utc_now().replace(microsecond=0)
        n = int(round((args.duration or cfg.capture.duration_s) * cfg.capture.sample_rate_hz))
        params = SessionSignalParams.draw(np.random.default_rng(stable_seed(args.session, "params")))
        meta = write_synthetic_capture(cfg, out, session_id=args.session, position_id=args.position,
                                       cls=args.cls, gain_role=args.gain_role,
                                       gain_db=capture.validate_gain(args.gain), centre_freq_hz=args.freq,
                                       n_samples=n, params=params, t_start=t,
                                       seed=stable_seed(args.session, args.cls, args.gain_role, t.isoformat()),
                                       notes=args.notes or "", cell_search=cell, extra_context=extra)
    else:
        out = (Path(args.data_dir) if args.data_dir else cfg.data_dir(False)) / "raw" / args.session
        meta = capture.record_capture(cfg, out, session_id=args.session, position_id=args.position,
                                      cls=args.cls, gain_role=args.gain_role, gain_db=args.gain,
                                      centre_freq_hz=args.freq, duration_s=args.duration,
                                      antenna=args.antenna or "", temperature_c=args.temperature,
                                      cell_search=cell, notes=args.notes or "", extra_context=extra,
                                      device_index=args.device, allow_other_receiver=args.allow_other_receiver)
    m = meta["measurements"]
    print(f"wrote {out / meta['file']}: clipping {100 * m['clipping_fraction']:.4f} % "
          f"({'FLAGGED' if m['clipping_flagged'] else 'ok'}), mean power {m['mean_power_db']:.2f} dB")
    return 0


def cmd_session(args):
    from .session import run_session
    cfg = _load_cfg(args)
    data_dir = Path(args.data_dir) if args.data_dir else cfg.data_dir(args.synthetic)
    if args.synthetic and data_dir.resolve() == cfg.data_dir(False).resolve():
        raise SystemExit("synthetic sessions are written to the synthetic data directory only")
    run_session(cfg, data_dir, synthetic=args.synthetic, duration_s=args.duration, device_index=args.device,
                allow_other_receiver=args.allow_other_receiver, quicklook=not args.no_quicklook)
    return 0


def cmd_synthetic(args):
    from .synthetic import generate_synthetic_dataset
    cfg = _load_cfg(args)
    root = Path(args.data_dir) if args.data_dir else cfg.data_dir(True)
    if (root / "raw").exists() and any((root / "raw").iterdir()) and not args.append:
        raise SystemExit(f"{root / 'raw'} is not empty; remove it or pass --append")
    metas = generate_synthetic_dataset(cfg, root, n_sessions=args.sessions, frames_per_capture=args.frames,
                                       seed=args.seed)
    print(f"wrote {len(metas)} SYNTHETIC captures under {root / 'raw'} (testing only)")
    return 0


def cmd_preprocess(args):
    from . import provenance
    from .injection import build_all_injection_sets
    from .preprocess import preprocess_all
    cfg = _load_cfg(args)
    index, work = _index_and_work(args, cfg)
    provenance.write_json(work / "index.json", {**index.summary(),
                                                "run": provenance.run_record(cfg, index.synthetic)})
    folds = _folds(index, cfg)
    provenance.write_json(work / "splits.json", {"folds": [f.to_dict() for f in folds],
                                                 "sessions": index.sessions(), "synthetic_data": index.synthetic,
                                                 "rule": "contiguous test groups; validation = first 2 "
                                                         "sessions of the next fold's test group"})
    preprocess_all(index, cfg, work, force=args.force)
    if not args.no_injection:
        build_all_injection_sets(index, cfg, work, force=args.force)
    print(f"done: {len(index)} captures, {len(index.sessions())} sessions, folds in {work / 'splits.json'}")
    return 0


def cmd_train(args):
    from .device import configure_determinism, describe, select_device
    configure_determinism()                      # CUBLAS_WORKSPACE_CONFIG before CUDA starts
    from .train import set_threads
    pack = None
    if args.pack:
        from .dataset import SyntheticDataError
        from .pack import open_pack
        if args.config or args.data_dir or args.work_dir:
            raise SystemExit("--pack trains with the pack's frozen configuration and data: --config, "
                             "--data-dir and --work-dir cannot be combined with it (re-pack to change them)")
        if not args.out:
            raise SystemExit("--pack needs --out DIR (where checkpoints, histories and models are written)")
        pack = open_pack(Path(args.pack))
        if pack.synthetic and not args.allow_synthetic:
            raise SyntheticDataError(f"{args.pack} is a SYNTHETIC pack; it is refused unless --allow-synthetic "
                                     "is given (testing only)")
        cfg = pack.config()
    else:
        if args.out and not args.time_epoch:
            raise SystemExit("--out is used with --pack (or --time-epoch); without a pack, models are written "
                             "to the work directory")
        cfg = _load_cfg(args)
    if args.device:
        cfg.train.device = args.device
    if args.threads:
        cfg.train.num_threads = args.threads
    set_threads(cfg.train.num_threads)
    device = select_device(cfg.train.device)
    print(f"device: {describe(device)} (train.device = {cfg.train.device})")
    if args.time_epoch:
        return _time_epoch(args, cfg, device, pack)
    common = dict(models=args.models, seeds=_seeds(args.seeds), ablation=args.ablation, force=args.force,
                  resume=not args.no_resume, device=device, allow_code_change=args.allow_code_change)
    if pack is not None:
        from .pack import train_from_pack
        train_from_pack(pack, cfg, Path(args.out), folds=args.folds, **common)
        return 0
    from .train import train_fold
    index, work = _index_and_work(args, cfg)
    for fold in _folds(index, cfg, args.folds):
        train_fold(index, fold, cfg, work, include_pca=not args.no_pca, **common)
    return 0


def _time_epoch(args, cfg, device, pack) -> int:
    from .budget import time_epochs
    fold = args.folds[0] if args.folds else 0
    if pack is not None:
        tr, va, _, _ = pack.fold_frames(fold, cfg)
        sizes, synthetic = pack.fold_sizes(cfg.train.frame_stride), pack.synthetic
        data, out = f"pack {pack.pack_id}", Path(args.out) / "time_epoch.json"
    else:
        from .preprocess import open_capture_store
        from .train import fold_frames_from_work
        index, work = _index_and_work(args, cfg)
        folds = _folds(index, cfg)
        sel = [f for f in folds if f.k == fold]
        if not sel:
            raise SystemExit(f"no fold {fold}")
        tr, va, _ = fold_frames_from_work(index, sel[0], cfg, work)
        from .dataset import train_records, val_records
        sizes = []
        for f in folds:
            n_tr = [open_capture_store(work, r.capture_id).n for r in train_records(index, f, cfg)]
            n_va = [open_capture_store(work, r.capture_id).n for r in val_records(index, f, cfg)]
            sizes.append({"fold": f.k, "train_frames": sum(len(range(0, n, cfg.train.frame_stride)) for n in n_tr),
                          "val_frames": sum(n_va)})
        synthetic, data = index.synthetic, f"work directory {work}"
        out = (Path(args.out) if args.out else work / "results") / "time_epoch.json"
    time_epochs(tr, va, cfg, device, fold, sizes, out, models=args.models, max_batches=args.time_batches,
                units_per_hour=args.units_per_hour, data=data, synthetic=synthetic)
    return 0


def cmd_pack(args):
    from .pack import make_pack
    cfg = _load_cfg(args)
    index, work = _index_and_work(args, cfg)
    out = Path(args.out) if args.out else work / "colab"
    res = make_pack(index, cfg, work, out, force=args.force)
    print(f"upload {res['zip'].name} and {res['code_zip'].name} (from {out}) to Google Drive "
          f"(MyDrive/msc-thesis/), then open colab/train_on_colab.ipynb in Colab")
    return 0


def cmd_unpack(args):
    from .unpack import import_runs
    cfg = _load_cfg(args)
    index, work = _index_and_work(args, cfg)
    res = import_runs(Path(args.runs), index, _folds(index, cfg), cfg, work, pack_dir=args.pack,
                      fold_ids=args.folds, models=args.models, seeds=_seeds(args.seeds), ablation=args.ablation,
                      allow_code_change=args.allow_code_change, force=args.force)
    return 0 if res["complete"] else 1


def cmd_calibrate(args):
    from .calibrate import calibrate_fold, write_threshold_table
    from .train import set_threads
    cfg = _load_cfg(args)
    set_threads(cfg.train.num_threads)
    index, work = _index_and_work(args, cfg)
    res = [calibrate_fold(index, f, cfg, work, models=args.models, ablation=args.ablation,
                          seeds=_seeds(args.seeds), force=args.force) for f in _folds(index, cfg, args.folds)]
    write_threshold_table(res, work / "results" / "thresholds.csv")
    print(f"thresholds: {work / 'results' / 'thresholds.csv'}")
    return 0


def cmd_evaluate(args):
    from .evaluate import evaluate, score_test_fold
    from .train import set_threads
    cfg = _load_cfg(args)
    set_threads(cfg.train.num_threads)
    index, work = _index_and_work(args, cfg)
    folds = _folds(index, cfg)
    for f in folds:
        score_test_fold(index, f, cfg, work, models=args.models, ablation=args.ablation,
                        seeds=_seeds(args.seeds), force=args.force_scores)
    out = Path(args.out) if args.out else work / "results"
    res = evaluate(index, folds, cfg, work, out, models=args.models, ablation=args.ablation,
                   seeds=_seeds(args.seeds), failure_figures=args.failure_figures)
    print("H1 supported by P_FA:", res["h1"]["supported_by_pfa"])
    print("H2 supported by P_FA:", res["h2"]["supported_by_pfa"])
    return 0


def cmd_latency(args):
    from .latency import run_latency
    cfg = _load_cfg(args)
    if args.threads:
        cfg.latency.num_threads = args.threads
    if args.frames:
        cfg.latency.n_frames = args.frames
    if args.warmup is not None:
        cfg.latency.warmup_frames = args.warmup
    work, cap, synthetic = None, None, False
    try:
        index, work = _index_and_work(args, cfg)
        synthetic = index.synthetic
        if args.capture_id:
            cap = index.by_id(args.capture_id).raw_path
    except Exception as exc:  # noqa: BLE001 - latency also works without data
        if args.capture_id:
            raise
        print(f"note: no data/work directory used ({exc}); timing untrained models on noise")
    out = Path(args.out) if args.out else (work or cfg.work_dir(False)) / "results" / "latency"
    run_latency(cfg, work, out, fold=args.fold, seed=args.seed, capture_path=cap, synthetic_data=synthetic,
                require_performance=not args.ignore_power)
    print(f"latency results: {out}")
    return 0


def cmd_demo(args):
    from .realtime import run_demo
    cfg = _load_cfg(args)
    if args.synthetic and not args.allow_synthetic:
        raise SystemExit("--synthetic demo runs need --allow-synthetic (testing only)")
    work = Path(args.work_dir) if args.work_dir else cfg.work_dir(bool(args.allow_synthetic))
    out = Path(args.out) if args.out else work / "results" / "demo" / (
        f"demo_{args.detector}_{'SYNTHETIC_' if args.synthetic else ''}"
        f"{__import__('time').strftime('%Y%m%dT%H%M%S')}.csv")
    seed = None if args.detector in ("ed_fixed", "ed_adaptive", "pss", "pca") else args.seed
    run_demo(cfg, work, out, detector=args.detector, fold=args.fold, seed=seed, pfa=args.pfa,
             centre_freq_hz=args.freq, gain_db=args.gain, duration_s=args.duration,
             synthetic_class=args.synthetic, live=args.live, max_frames=args.max_frames,
             pace=not args.no_pace, device_index=args.device, allow_other_receiver=args.allow_other_receiver)
    return 0


def cmd_inspect(args):
    import json
    from .quicklook import capture_quicklook, labelling_agreement, labelling_template
    from . import provenance
    cfg = _load_cfg(args)
    if args.agreement:
        res = labelling_agreement(Path(args.agreement))
        print(json.dumps(res, indent=2))
        provenance.write_json(Path(args.agreement).with_suffix(".agreement.json"), res)
        return 0
    index, work = _index_and_work(args, cfg)
    out = Path(args.out) if args.out else work / "inspect"
    out.mkdir(parents=True, exist_ok=True)
    for r in index.records:
        if args.session and r.session_id not in args.session:
            continue
        meta = provenance.read_json(r.meta_path)
        capture_quicklook(Path(r.raw_path), meta, cfg, out / f"{r.capture_id}_quicklook.png")
    tpl = labelling_template(index, out / "labelling_check.csv")
    print(f"quick-look figures and the labelling-check template ({tpl}) are in {out}")
    return 0


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="JSON file overriding config.py defaults")
    common.add_argument("--data-dir", help="captures root (default: data/, or data_synthetic/ with --allow-synthetic)")
    common.add_argument("--work-dir", help="derived data and results (default: work/ or work_synthetic/)")
    syn = argparse.ArgumentParser(add_help=False)
    syn.add_argument("--allow-synthetic", action="store_true",
                     help="accept synthetic captures (TESTING ONLY; outputs go to work_synthetic/)")
    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument("--models", nargs="+", help="restrict to these models (conv_ae fc_ae conv_vae pca ...)")
    sel.add_argument("--seeds", nargs="+", help="restrict to these seeds")
    sel.add_argument("--ablation", action="store_true", help="include the latent-size ablations")

    p = argparse.ArgumentParser(prog="python -m occupancy", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("survey", parents=[common], help="rtl_power band survey and plots")
    s.add_argument("--range", type=_range, help="START:STOP, e.g. 758M:803M")
    s.add_argument("--gain", help="fixed tuner gain in dB (automatic gain is refused)")
    s.add_argument("--label", help="e.g. day or night")
    s.add_argument("--position", help="position ID")
    s.add_argument("--notes")
    s.add_argument("--carrier", type=_freq, help="carrier centre, to rank candidate Class B windows")
    s.add_argument("--bin", type=_freq, help="max bin size (default from config)")
    s.add_argument("--integration", type=int, help="seconds per sweep")
    s.add_argument("--duration", help="rtl_power exit timer, e.g. 5m")
    s.add_argument("--out", help="output directory (default data/survey)")
    s.add_argument("--parse", help="analyse an existing rtl_power CSV instead of running a survey")
    s.add_argument("--device", type=int, default=0)
    s.set_defaults(func=cmd_survey)

    c = sub.add_parser("capture", parents=[common], help="one capture (cu8 + JSON sidecar)")
    c.add_argument("--info", action="store_true", help="print receiver identity and gain steps, then exit")
    c.add_argument("--session")
    c.add_argument("--position")
    c.add_argument("--class", dest="cls", choices=CLASSES)
    c.add_argument("--gain-role", choices=GAIN_ROLES)
    c.add_argument("--gain", help="manual gain in dB (must be a tuner gain step; 'auto' is refused)")
    c.add_argument("--freq", type=_freq, help="centre frequency, e.g. 796M")
    c.add_argument("--duration", type=float, help="seconds (default 30)")
    c.add_argument("--ppm", type=int)
    c.add_argument("--antenna")
    c.add_argument("--temperature", type=float, help="deg C, if known")
    c.add_argument("--notes")
    c.add_argument("--cell-tool")
    c.add_argument("--cell-result", help="e.g. 'no cell found' for Class B")
    c.add_argument("--pci", type=int)
    c.add_argument("--carrier", type=_freq, help="carrier centre from cell search")
    c.add_argument("--bandwidth", type=float, help="channel bandwidth in MHz, if known")
    c.add_argument("--device", type=int, default=0)
    c.add_argument("--allow-other-receiver", action="store_true")
    c.add_argument("--synthetic", action="store_true", help="write a SYNTHETIC capture (testing only)")
    c.set_defaults(func=cmd_capture)

    ss = sub.add_parser("session", parents=[common], help="guided 8-capture session")
    ss.add_argument("--duration", type=float, help="seconds per capture (default 30)")
    ss.add_argument("--device", type=int, default=0)
    ss.add_argument("--allow-other-receiver", action="store_true")
    ss.add_argument("--no-quicklook", action="store_true")
    ss.add_argument("--synthetic", action="store_true", help="rehearse with SYNTHETIC captures (no dongle)")
    ss.set_defaults(func=cmd_session)

    sy = sub.add_parser("synthetic", parents=[common], help="generate a SYNTHETIC data set (testing only)")
    sy.add_argument("--sessions", type=int, default=8)
    sy.add_argument("--frames", type=int, default=8, help="frames per capture")
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--append", action="store_true")
    sy.set_defaults(func=cmd_synthetic)

    pp = sub.add_parser("preprocess", parents=[common, syn], help="frames, spectrograms, injection sets, splits")
    pp.add_argument("--force", action="store_true")
    pp.add_argument("--no-injection", action="store_true")
    pp.set_defaults(func=cmd_preprocess)

    t = sub.add_parser("train", parents=[common, syn, sel], help="train models and fit PCA per fold")
    t.add_argument("--folds", nargs="+", type=int)
    t.add_argument("--threads", type=int)
    t.add_argument("--device", help="auto (default: CUDA GPU if available, else CPU), cpu, cuda or cuda:N")
    t.add_argument("--no-pca", action="store_true")
    t.add_argument("--force", action="store_true", help="retrain finished runs from scratch")
    t.add_argument("--resume", dest="no_resume", action="store_false",
                   help="continue unfinished runs from their last checkpoint (the default)")
    t.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="restart unfinished runs from scratch (finished runs are still skipped)")
    t.add_argument("--pack", help="train from this verified training pack (no raw data or work directory needed)")
    t.add_argument("--out", help="with --pack: output directory for checkpoints, histories and models")
    t.add_argument("--allow-code-change", action="store_true",
                   help="accept code that differs from the pack's or the checkpoint's (recorded)")
    t.add_argument("--time-epoch", action="store_true",
                   help="time one epoch per model and estimate the GPU-hours of the whole design")
    t.add_argument("--time-batches", type=int, help="with --time-epoch: time only N batches per pass")
    t.add_argument("--units-per-hour", type=float, help="with --time-epoch: Colab compute units per hour")
    t.set_defaults(func=cmd_train, no_resume=False)

    pk = sub.add_parser("pack", parents=[common, syn],
                        help="training pack for Colab: Class B g1 spectrograms, folds, normalisers, config")
    pk.add_argument("--out", help="output directory (default: <work>/colab)")
    pk.add_argument("--force", action="store_true", help="replace an existing pack")
    pk.set_defaults(func=cmd_pack)

    up = sub.add_parser("unpack", aliases=["import-models"], parents=[common, syn, sel],
                        help="verify models trained on Colab and copy them into the work directory")
    up.add_argument("--runs", required=True, help="the downloaded output directory or runs .zip")
    up.add_argument("--pack", help="the pack directory (default: the record written by 'pack' in <work>/packs)")
    up.add_argument("--folds", nargs="+", type=int, help="expect only these folds")
    up.add_argument("--allow-code-change", action="store_true",
                    help="import even if the code differs (recorded in colab_import.json)")
    up.add_argument("--force", action="store_true", help="replace different local runs")
    up.set_defaults(func=cmd_unpack)

    cb = sub.add_parser("calibrate", parents=[common, syn, sel], help="thresholds at P_FA 0.01 and 0.05")
    cb.add_argument("--folds", nargs="+", type=int)
    cb.add_argument("--force", action="store_true", help="rescore validation frames")
    cb.set_defaults(func=cmd_calibrate)

    e = sub.add_parser("evaluate", parents=[common, syn, sel], help="test scores, metrics, bootstrap, H1/H2")
    e.add_argument("--force-scores", action="store_true")
    e.add_argument("--failure-figures", action="store_true")
    e.add_argument("--out")
    e.set_defaults(func=cmd_evaluate)

    la = sub.add_parser("latency", parents=[common, syn], help="per-frame latency and throughput")
    la.add_argument("--fold", type=int, default=0)
    la.add_argument("--seed", type=int, default=0)
    la.add_argument("--capture-id", help="time on frames of this capture (default: noise)")
    la.add_argument("--frames", type=int)
    la.add_argument("--warmup", type=int)
    la.add_argument("--threads", type=int)
    la.add_argument("--ignore-power", action="store_true",
                    help="time even if not on mains power with the Performance profile (test runs only)")
    la.add_argument("--out")
    la.set_defaults(func=cmd_latency)

    d = sub.add_parser("demo", parents=[common, syn], help="real-time demonstration with CSV log")
    d.add_argument("--detector", required=True,
                   help="conv_ae, fc_ae, conv_vae_mse, conv_vae_per, pca, ed_fixed, ed_adaptive or pss")
    d.add_argument("--freq", type=_freq, required=True)
    d.add_argument("--gain", required=True, help="the g1 gain the thresholds were calibrated at")
    d.add_argument("--fold", type=int, default=0)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--pfa", type=float, default=0.01)
    d.add_argument("--duration", type=float)
    d.add_argument("--max-frames", type=int)
    d.add_argument("--live", action="store_true", help="live waterfall and occupancy strip")
    d.add_argument("--synthetic", choices=CLASSES, help="SYNTHETIC source of this class (testing only)")
    d.add_argument("--no-pace", action="store_true", help="synthetic source as fast as possible")
    d.add_argument("--device", type=int, default=0)
    d.add_argument("--allow-other-receiver", action="store_true")
    d.add_argument("--out")
    d.set_defaults(func=cmd_demo)

    i = sub.add_parser("inspect", parents=[common, syn], help="quick-look figures and the labelling check")
    i.add_argument("--session", nargs="+")
    i.add_argument("--out")
    i.add_argument("--agreement", help="filled labelling_check.csv: report the agreement")
    i.set_defaults(func=cmd_inspect)
    return p


# Errors reported as one line (exit code 2) instead of a traceback.
REPORTED_ERRORS = {"occupancy.capture": ("GainError", "ReceiverError"),
                   "occupancy.dataset": ("DatasetError", "SyntheticDataError"),
                   "occupancy.session": ("SessionAbort",), "occupancy.pack": ("PackError",),
                   "occupancy.train": ("ResumeError",), "occupancy.device": ("DeviceError",)}


def _reported_errors() -> tuple:
    """The error classes of the modules already loaded: a module that raised one is loaded, and
    no module (in particular none that imports torch) is imported just to catch its errors."""
    return tuple(getattr(sys.modules[m], n) for m, names in REPORTED_ERRORS.items() if m in sys.modules
                 for n in names)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - only the reported error types are caught
        if isinstance(exc, _reported_errors()):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise
