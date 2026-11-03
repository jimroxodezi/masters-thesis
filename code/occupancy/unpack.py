"""
unpack.py
---------
Import of models trained elsewhere (on Google Colab) into the local work
directory, after checking that they belong to it.

`python -m occupancy unpack --runs DIR|ZIP` takes the output directory of
`train --pack ... --out ...`, or the runs.zip the Colab notebook writes, and
copies the trained runs to work/folds/fold<k>/models/<model>/seed<s>/, where
calibrate, evaluate, latency and demo look for them. Before anything is
copied it checks:
  1. pack identity: the runs were trained on a pack made from this work
     directory (pack ID recorded by `pack` under work/packs/, or a pack
     directory given with --pack, which is verified);
  2. data: real and synthetic runs are never mixed with the other kind; the
     folds (test, validation and training sessions, and the training and
     validation captures in order) equal those of the local data;
  3. configuration: the training settings (preprocessing, splits, learned
     models, training) equal the local configuration; other differences,
     which do not affect the trained models, are listed;
  4. code version: the SHA-256 of the occupancy source files is the same
     locally, in the pack and for every run; a difference is refused unless
     --allow-code-change is given, and the changed files are recorded;
  5. normalisation: each fold's local normalizer.json holds the statistics
     the runs were trained with (if it is missing it is written from the pack);
  6. completeness: every expected run (model x latent size x fold x seed;
     with the latent-size ablations when any ablation run was trained or
     --ablation is given) has its completion marker, and the size and SHA-256
     checksum of model.pt, history.csv and train_summary.json match it, and
     its summary names the expected model, seed, fold, data and pack.
Complete runs are copied (to a temporary directory, then renamed); missing,
unfinished and damaged runs are reported and not copied. An existing local
run is replaced only if --force is given (an identical one is left as it is).
Every check and its result is written to work/colab_import.json.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from pathlib import Path

from . import provenance
from .config import Config, config_differences
from .dataset import CaptureIndex, train_records, val_records
from .models import ablation_specs
from .pack import IDENTITY_FILE, PackError, pack_record_path, safe_extract, verify_pack
from .preprocess import Normalizer
from .train import FINAL_FILES, MARKER, expected_runs, fold_dir, is_complete, model_dir, run_status

REPORT = "colab_import.json"


class ImportRefused(PackError):
    """The downloaded runs do not belong to this work directory."""


def _runs_root(path: Path) -> Path:
    """The directory holding pack_identity.json (the given one or one level below)."""
    if (path / IDENTITY_FILE).exists():
        return path
    found = [p.parent for p in path.glob(f"*/{IDENTITY_FILE}")]
    if len(found) == 1:
        return found[0]
    raise ImportRefused(f"{path} is not an output directory of 'train --pack' (no {IDENTITY_FILE})")


def _local_pack(work_dir: Path, pack_id: str, pack_dir) -> dict:
    """The local record of the pack: from --pack (verified) or work/packs/<pack_id>.json."""
    if pack_dir is not None:
        m = verify_pack(Path(pack_dir))
        if m["pack_id"] != pack_id:
            raise ImportRefused(f"the runs were trained on pack {pack_id}, but --pack is pack {m['pack_id']}")
        return {"pack_id": m["pack_id"], "source": f"verified pack directory {pack_dir}", "manifest": m}
    rec = pack_record_path(work_dir, pack_id)
    if not rec.exists():
        raise ImportRefused(f"this work directory ({work_dir}) has no record of pack {pack_id} (no {rec}); "
                            "the runs were trained on a pack made elsewhere. Give the pack directory with --pack.")
    r = provenance.read_json(rec)
    return {"pack_id": r["pack_id"], "source": str(rec), "manifest": r["manifest"]}


def _check_folds(ident: dict, index: CaptureIndex, folds, cfg: Config) -> list[str]:
    problems = []
    local = {f.k: f for f in folds}
    remote = {int(f["fold"]): f for f in ident["folds"]}
    if set(local) != set(remote):
        return [f"folds differ: local {sorted(local)}, runs {sorted(remote)}"]
    for k, f in local.items():
        r = remote[k]
        for part in ("test", "val", "train"):
            if list(getattr(f, part)) != list(r[part]):
                problems.append(f"fold {k} {part} sessions: local {list(getattr(f, part))}, runs {r[part]}")
        tr = [x.capture_id for x in train_records(index, f, cfg)]
        va = [x.capture_id for x in val_records(index, f, cfg)]
        if tr != list(r["train_units"]) or va != list(r["val_units"]):
            problems.append(f"fold {k}: training or validation captures differ from the local data")
    return problems


def _verify_run(src: Path, k: int, spec, seed: int, ident: dict, local_code: str, allow_code_change: bool,
                fold_rec: dict, norm: dict) -> dict:
    status = run_status(src)
    if status["state"] != "complete":
        return status
    marker = provenance.read_json(src / MARKER)
    for name in FINAL_FILES:
        rec = marker["files"][name]
        if provenance.sha256_file(src / name) != rec["sha256"]:
            return {"state": "invalid", "reason": f"{name}: checksum differs from the completion marker"}
    s = provenance.read_json(src / "train_summary.json")
    checks = [
        (s.get("spec") == spec.to_dict(), f"model is {s.get('spec')}"),
        (int(s.get("seed", -1)) == seed, f"seed is {s.get('seed')}"),
        (int((s.get("fold") or {}).get("fold", -1)) == k, "fold differs"),
        ((s.get("pack") or {}).get("pack_id") == ident["pack_id"], "trained on another pack"),
        (s.get("training_fingerprint") == ident["training_fingerprint"], "other training settings"),
        (bool(s.get("synthetic_data")) == bool(ident["synthetic_data"]), "synthetic flag differs"),
        ((s.get("identity") or {}).get("data", {}).get("train_units") == list(fold_rec["train_units"]),
         "training captures differ"),
        ((s.get("identity") or {}).get("data", {}).get("val_units") == list(fold_rec["val_units"]),
         "validation captures differ"),
        ((s.get("identity") or {}).get("data", {}).get("mu") == norm["mu"]
         and (s.get("identity") or {}).get("data", {}).get("sigma") == norm["sigma"],
         "normalisation statistics differ"),
    ]
    for ok, reason in checks:
        if not ok:
            return {"state": "invalid", "reason": reason}
    run_code = (s.get("code_version") or {}).get("code_sha256")
    out = {"state": "complete", "best_epoch": s.get("best_epoch"), "epochs_run": s.get("epochs_run"),
           "device": s.get("device"), "gpu_name": s.get("gpu_name"), "resumed": s.get("resumed"),
           "code_sha256": run_code, "model_sha256": marker["files"]["model.pt"]["sha256"]}
    if run_code != local_code:
        if not allow_code_change:
            return {"state": "invalid", "reason": f"trained with code {str(run_code)[:12]}, local code is "
                                                  f"{local_code[:12]} (--allow-code-change accepts it)"}
        out["note"] = "trained with different code (allowed)"
    return out


def _copy_run(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and any(dst.iterdir()):
        if is_complete(dst):
            a = provenance.read_json(dst / MARKER)["files"]
            b = provenance.read_json(src / MARKER)["files"]
            if all(a[n]["sha256"] == b[n]["sha256"] for n in FINAL_FILES):
                return "already present"
        if not force:
            return "conflict"
    tmp = dst.with_name(dst.name + ".importing")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in FINAL_FILES + (MARKER,):                    # the completion marker last
        shutil.copy2(src / name, tmp / name)
    marker = provenance.read_json(tmp / MARKER)
    for name in FINAL_FILES:
        if provenance.sha256_file(tmp / name) != marker["files"][name]["sha256"]:
            shutil.rmtree(tmp)
            raise OSError(f"copy of {src / name} is damaged")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return "imported"


def import_runs(runs: Path, index: CaptureIndex, folds, cfg: Config, work_dir: Path, pack_dir=None,
                fold_ids=None, models=None, seeds=None, ablation: bool = False, allow_code_change: bool = False,
                force: bool = False, log=print) -> dict:
    """Verify downloaded runs against this work directory and copy the complete ones into it.

    Returns the report (also written to work/colab_import.json); report["complete"]
    is True when every expected run was imported or already present.
    Raises ImportRefused when the runs do not belong to this work directory.
    """
    runs, work_dir = Path(runs), Path(work_dir)
    if not runs.exists():
        raise ImportRefused(f"{runs} does not exist")
    with contextlib.ExitStack() as stack:
        if runs.is_file():
            if runs.suffix.lower() != ".zip":
                raise ImportRefused(f"{runs}: give the runs directory or the runs .zip")
            work_dir.mkdir(parents=True, exist_ok=True)
            tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=".unpack_", dir=work_dir)))
            log(f"extracting {runs} ...")
            safe_extract(runs, tmp)
            root = _runs_root(tmp)
        else:
            root = _runs_root(runs)
        return _import(root, runs, index, folds, cfg, work_dir, pack_dir, fold_ids, models, seeds, ablation,
                       allow_code_change, force, log)


def _import(root, source, index, folds, cfg, work_dir, pack_dir, fold_ids, models, seeds, ablation,
            allow_code_change, force, log) -> dict:
    ident = provenance.read_json(root / IDENTITY_FILE)
    pack_id = ident.get("pack_id")
    report = {"source": str(source), "pack_id": pack_id, "work_dir": str(work_dir),
              "synthetic_data": bool(ident.get("synthetic_data")), "checks": {}, "warnings": []}

    # 1-2. pack identity and data
    if bool(ident.get("synthetic_data")) != bool(index.synthetic):
        raise ImportRefused("the runs were trained on " + ("SYNTHETIC" if ident.get("synthetic_data") else "real")
                            + " data but this work directory holds " + ("synthetic" if index.synthetic else "real")
                            + " data: real and synthetic data are never mixed")
    local_pack = _local_pack(work_dir, pack_id, pack_dir)
    report["checks"]["pack"] = f"pack {pack_id} matches {local_pack['source']}"
    if local_pack["manifest"]["training_fingerprint"] != ident["training_fingerprint"]:
        raise ImportRefused("the pack record and the runs disagree on the training settings")
    fold_problems = _check_folds(ident, index, folds, cfg)
    if fold_problems:
        raise ImportRefused("the folds of the runs differ from the local data: " + "; ".join(fold_problems))
    report["checks"]["folds"] = "identical"

    # 3. configuration
    remote_cfg = Config.from_dict(ident["config"])
    if cfg.training_fingerprint() != ident["training_fingerprint"]:
        diffs = config_differences(cfg.training_dict(), remote_cfg.training_dict(), ignore=())
        raise ImportRefused("the training settings differ from the local configuration: " + "; ".join(diffs))
    other = config_differences(cfg.to_dict(), remote_cfg.to_dict())
    report["checks"]["config"] = "training settings identical"
    if other:
        report["warnings"].append("configuration differences that do not affect training: " + "; ".join(other))

    # 4. code version
    local_code = provenance.code_version()
    code_issues = []
    for label, other_code in (("the pack", ident.get("pack_code_version") or {}),
                              ("the training", ident.get("training_code_version") or {})):
        if other_code.get("code_sha256") != local_code["code_sha256"]:
            files = provenance.changed_code_files(other_code, local_code)
            code_issues.append(f"local code {local_code['code_sha256'][:12]} differs from {label} code "
                               f"{str(other_code.get('code_sha256'))[:12]} (files: {', '.join(files) or 'unknown'})")
    for ch in ident.get("code_changes") or []:
        code_issues.append(f"the code changed during training at {ch.get('utc')} "
                           f"(files: {', '.join(ch.get('changed_files') or []) or 'unknown'})")
    if code_issues and not allow_code_change:
        raise ImportRefused("; ".join(code_issues) + ". Pass --allow-code-change to import anyway (recorded).")
    report["checks"]["code"] = ("identical " + local_code["code_sha256"][:12]) if not code_issues else \
        "DIFFERENT (allowed with --allow-code-change): " + "; ".join(code_issues)

    # 5. normalisation statistics
    notes = []
    for f in folds:
        remote_norm = ident["normalizers"][str(f.k)]
        path = fold_dir(work_dir, f.k) / "normalizer.json"
        if path.exists():
            local = Normalizer.load(path).to_dict()
            if any(local[x] != remote_norm[x] for x in ("mu", "sigma", "n_frames", "fitted_on")):
                raise ImportRefused(f"fold {f.k}: the local normalizer.json differs from the statistics the "
                                    "runs were trained with")
        else:
            Normalizer.from_dict(remote_norm).save(path)
            notes.append(f"fold {f.k}: normalizer.json written from the pack")
    report["checks"]["normalizers"] = "identical" + (f" ({'; '.join(notes)})" if notes else "")

    # 6. completeness, then copy
    fold_recs = {int(fr["fold"]): fr for fr in ident["folds"]}
    ablation_names = {s.name for s in ablation_specs(cfg)}
    trained_ablation = any(next((root / "folds").glob(f"fold*/models/{n}"), None) is not None
                           for n in ablation_names)
    with_ablation = bool(ablation or trained_ablation)
    ids = sorted(fold_recs) if fold_ids is None else [k for k in sorted(fold_recs) if k in fold_ids]
    expected = expected_runs(cfg, ids, models, seeds, with_ablation)
    rows, counts = [], {}
    for k, spec, seed in expected:
        name = f"fold{k}/{spec.name}/seed{seed}"
        res = _verify_run(model_dir(root, k, spec, seed), k, spec, seed, ident, local_code["code_sha256"],
                          allow_code_change, fold_recs[k], ident["normalizers"][str(k)])
        if res["state"] == "complete":
            res["copy"] = _copy_run(model_dir(root, k, spec, seed), model_dir(work_dir, k, spec, seed), force)
            key = res["copy"]
        else:
            key = res["state"]
        counts[key] = counts.get(key, 0) + 1
        rows.append({"run": name, **res})
    complete = all(r.get("copy") in ("imported", "already present") for r in rows)

    timing = root / "time_epoch.json"
    if timing.exists():
        dst = work_dir / "results" / "time_epoch_colab.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(timing, dst)
        report["time_epoch"] = str(dst)
    report.update(expected=len(expected), with_ablation=with_ablation, counts=counts, complete=complete,
                  runs=rows, run=provenance.run_record(cfg, synthetic=index.synthetic))
    provenance.write_json(work_dir / REPORT, report)

    log(f"unpack: pack {pack_id}, {len(expected)} expected runs"
        f"{' (with ablations)' if with_ablation else ''}: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for w in report["warnings"]:
        log("note: " + w)
    if code_issues:
        log("WARNING: " + report["checks"]["code"])
    for r in rows:
        if r.get("copy") not in ("imported", "already present"):
            detail = r.get("reason") or (f"{r.get('epochs_done')} epoch(s) done" if r["state"] == "unfinished"
                                         else "")
            state = r.get("copy") or r["state"]
            log(f"  {state:12s} {r['run']}{(': ' + detail) if detail else ''}")
    if complete:
        log(f"all expected runs are in {work_dir / 'folds'}; next: train --models pca, calibrate, evaluate")
    else:
        log("INCOMPLETE: the runs listed above were not imported (report: " + str(work_dir / REPORT) + ")")
    return report
