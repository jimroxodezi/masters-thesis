"""
pack.py
-------
Training pack: the data the neural-network training needs, and nothing else,
in one self-contained directory (and a .zip of it) that can be copied to
another machine, such as a Google Colab GPU runtime.

`python -m occupancy pack` is run on the laptop after `preprocess`. It writes
    train_pack/
      sessions/<session>/spec.npy   the session's Class B spectrograms at gain g1
                                    (float16, un-normalised dB, n x 128 x 128),
                                    stored once per session and shared by every fold;
      sessions/<session>/unit.json  the capture they come from (capture ID, class,
                                    gain, number of frames, preprocessing fingerprint);
      folds.json                    the folds: test, validation and training sessions,
                                    and the training and validation captures in the
                                    order in which training reads them;
      normalizers/fold<k>.json      each fold's normalisation statistics (mu, s),
                                    computed on the laptop from that fold's training frames;
      config.json                   the frozen configuration;
      code_version.json             SHA-256 of the occupancy source files and the git state;
      manifest.json                 the pack ID and the size and SHA-256 checksum
                                    of every other file.
    train_pack.zip                  the same directory as one file for uploading;
    occupancy_code.zip              the occupancy package and the Colab script and
                                    notebook (code/colab/), to run on the Colab runtime.
Raw IQ, Class A and C captures, gain g2 captures, the injection sets and all
scores stay on the laptop. Every session's Class B g1 frames are included
because each session is a training or validation session of some fold;
training on fold k reads only that fold's training and validation sessions.
A pack made from synthetic data is named train_pack_SYNTHETIC and is refused
for training unless --allow-synthetic is given.

The pack ID is a hash of the file checksums, the training settings and the
code version. open_pack() verifies the manifest (every file present with its
size and checksum, no extra files, the pack ID, and that config.json and
code_version.json agree with it) before any training reads the pack.

train_from_pack() trains from a verified pack into an output directory with
the same layout as the work directory (folds/fold<k>/models/<model>/seed<s>/).
The output directory records the pack it belongs to in pack_identity.json
(pack ID, frozen configuration, folds, normalisation statistics and code
versions); runs from another pack are never mixed into it. Training refuses
code that differs from the code the pack was made with, unless
--allow-code-change is given, in which case the changed files are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

from . import provenance
from .config import CODE_DIR, Config
from .dataset import CaptureIndex, Fold, make_folds, natural_key, train_records, val_records
from .preprocess import Normalizer, capture_source_signature, feature_fingerprint, open_capture_store, \
    store_is_current
from .train import FrameSet, fit_fold_normalizer, train_runs

PACK_KIND = "occupancy-training-pack"
PACK_FORMAT = 1
MANIFEST = "manifest.json"
IDENTITY_FILE = "pack_identity.json"          # in a training output directory
OUTPUT_KIND = "occupancy-training-output"
CODE_ZIP = "occupancy_code.zip"
CODE_ZIP_ROOT = "occupancy_code"
CODE_EXTRA_FILES = ("colab/colab_train.py", "colab/train_on_colab.ipynb", "pyproject.toml", "README.md")


class PackError(RuntimeError):
    """A training pack, or an output directory trained from one, failed a check."""


def pack_name(synthetic: bool) -> str:
    return "train_pack_SYNTHETIC" if synthetic else "train_pack"


def fmt_bytes(n: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def _pack_id(files: list, training_fingerprint: str, code_sha256: str, synthetic: bool) -> str:
    blob = json.dumps({"format": PACK_FORMAT, "files": [[f["path"], f["bytes"], f["sha256"]] for f in files],
                       "training_fingerprint": training_fingerprint, "code_sha256": code_sha256,
                       "synthetic_data": bool(synthetic)}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hard link (no extra disk space; the work stores are never modified in place) or copy."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def _zip_dir(src: Path, zpath: Path, arc_root: str, compresslevel: int = 1) -> None:
    tmp = zpath.with_name(zpath.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel,
                         allowZip64=True) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, f"{arc_root}/{p.relative_to(src).as_posix()}")
    provenance.replace_file(tmp, zpath)


def code_files() -> list[tuple[Path, str]]:
    """(path, archive name) of the code shipped to Colab: the package and code/colab/."""
    out = [(p, f"occupancy/{p.relative_to(provenance.PACKAGE_DIR).as_posix()}")
           for p in sorted(provenance.PACKAGE_DIR.rglob("*.py")) if "__pycache__" not in p.parts]
    out += [(CODE_DIR / rel, rel) for rel in CODE_EXTRA_FILES if (CODE_DIR / rel).exists()]
    return out


def write_code_zip(path: Path) -> dict:
    """occupancy_code.zip: occupancy/, colab/, pyproject.toml, README.md and code_version.json."""
    path = Path(path)
    version = provenance.code_version()
    tmp = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p, arc in code_files():
            z.write(p, f"{CODE_ZIP_ROOT}/{arc}")
        z.writestr(f"{CODE_ZIP_ROOT}/code_version.json", json.dumps(version, indent=2))
    provenance.replace_file(tmp, path)
    return {"path": str(path), "bytes": path.stat().st_size, "code_sha256": version["code_sha256"]}


def safe_extract(zpath: Path, dest: Path) -> list[str]:
    """Extract a zip after refusing absolute paths, '..', empty path parts and symbolic links."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = []
        for info in z.infolist():
            name = info.filename
            parts = name.rstrip("/").split("/")
            if name.startswith("/") or "\\" in name or ":" in parts[0] or any(p in ("", "..") for p in parts):
                raise PackError(f"{zpath}: unsafe member name {name!r}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise PackError(f"{zpath}: symbolic link {name!r} refused")
            target = (dest / name).resolve()
            if target != dest and dest not in target.parents:
                raise PackError(f"{zpath}: member {name!r} would be written outside {dest}")
            names.append(name)
        z.extractall(dest)
    return names


# ── writing a pack (laptop) ───────────────────────────────────────────────────

def make_pack(index: CaptureIndex, cfg: Config, work_dir: Path, out_dir: Path, force: bool = False,
              log=print) -> dict:
    """Write the training pack, its zip and occupancy_code.zip to out_dir; record it in work_dir/packs/."""
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    folds = make_folds(index.sessions(), cfg, log=log)
    synthetic = index.synthetic
    name = pack_name(synthetic)
    final, zpath = out_dir / name, out_dir / f"{name}.zip"
    if (final.exists() or zpath.exists()) and not force:
        raise PackError(f"{final} or {zpath} exists; pass --force to replace it")

    # folds, normalisation statistics and the units every fold reads
    fold_info, normalizers, units = [], {}, {}
    for f in folds:
        norm = fit_fold_normalizer(index, f, cfg, work_dir)
        tr, va = train_records(index, f, cfg), val_records(index, f, cfg)
        for r in tr + va:
            units[r.capture_id] = r
        fold_info.append({**f.to_dict(), "train_units": [r.capture_id for r in tr],
                          "val_units": [r.capture_id for r in va]})
        normalizers[f.k] = norm.to_dict()
    fp = feature_fingerprint(cfg)
    by_session, stores = {}, {}
    for cid, r in units.items():
        if r.cls != "B" or r.gain_role != cfg.split.primary_gain:
            raise PackError(f"{cid}: only Class B frames at {cfg.split.primary_gain} may enter a training pack")
        if r.synthetic != synthetic:
            raise PackError(f"{cid}: synthetic and real captures are never mixed")
        if r.session_id in by_session:
            raise PackError(f"session {r.session_id} has two training captures")
        st = open_capture_store(work_dir, cid)
        if not store_is_current(st.dir, fp, capture_source_signature(r)):
            raise PackError(f"{cid}: the preprocessed features are out of date; run preprocess first")
        by_session[r.session_id], stores[cid] = r, st
    sessions = sorted(by_session, key=natural_key)
    data_bytes = sum((stores[by_session[s].capture_id].dir / "spec.npy").stat().st_size for s in sessions)
    out_dir.mkdir(parents=True, exist_ok=True)
    same_fs = os.stat(work_dir).st_dev == os.stat(out_dir).st_dev
    need = int(data_bytes * (1.05 if same_fs else 2.05)) + 50_000_000
    free = shutil.disk_usage(out_dir).free
    if free < need:
        raise PackError(f"not enough disk space in {out_dir}: {fmt_bytes(free)} free, about "
                        f"{fmt_bytes(need)} needed for the pack and its zip")

    # the pack directory, written under a temporary name and renamed when complete
    tmp = out_dir / f"{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    how = set()
    for sid in sessions:
        r = by_session[sid]
        st = stores[r.capture_id]
        d = tmp / "sessions" / sid
        d.mkdir(parents=True)
        how.add(_link_or_copy(st.dir / "spec.npy", d / "spec.npy"))
        provenance.write_json(d / "unit.json", {
            "session_id": sid, "capture_id": r.capture_id, "class": r.cls, "gain_role": r.gain_role,
            "gain_db": r.gain_db, "position_id": r.position_id, "utc_start": r.utc_start, "n_frames": st.n,
            "shape": list(st.spec.shape), "dtype": str(st.spec.dtype), "synthetic": synthetic,
            "feature_fingerprint": st.meta.get("fingerprint"),
            "content": "un-normalised spectrograms S (dB), frequency x time"})
    provenance.write_json(tmp / "folds.json", {
        "folds": fold_info, "sessions": index.sessions(),
        "units_by_session": {s: by_session[s].capture_id for s in sessions},
        "sessions_without_training_capture": [s for s in index.sessions() if s not in by_session],
        "rule": "contiguous test groups; validation = first 2 sessions of the next fold's test group"})
    for k, nd in normalizers.items():
        provenance.write_json(tmp / "normalizers" / f"fold{k}.json", nd)
    cfg.save(tmp / "config.json")
    code = provenance.code_version()
    provenance.write_json(tmp / "code_version.json", code)
    files = [{"path": p.relative_to(tmp).as_posix(), **provenance.file_record(p)}
             for p in sorted(tmp.rglob("*")) if p.is_file()]
    pack_id = _pack_id(files, cfg.training_fingerprint(), code["code_sha256"], synthetic)
    total = sum(f["bytes"] for f in files)
    manifest = {
        "kind": PACK_KIND, "format": PACK_FORMAT, "pack_id": pack_id, "name": name,
        "created_utc": provenance.utc_iso(), "synthetic_data": synthetic,
        "training_fingerprint": cfg.training_fingerprint(), "config_fingerprint": cfg.fingerprint(),
        "code_sha256": code["code_sha256"], "git": code["git"],
        "contents": f"Class B spectrograms at {cfg.split.primary_gain} of every session (float16), folds, "
                    "per-fold normalisation statistics, frozen configuration, code version",
        "excluded": "raw IQ, Class A and C captures, gain g2 captures, injection sets, scores",
        "sessions": sessions, "n_folds": len(folds), "total_bytes": total, "files": files,
        "run": provenance.run_record(cfg, synthetic=synthetic)}
    provenance.write_json(tmp / MANIFEST, manifest)
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    log(f"pack {pack_id}: {len(sessions)} sessions, {len(folds)} folds, {fmt_bytes(total)} in {final} "
        f"(spectrograms {'hard-linked' if how == {'link'} else 'copied'} from {work_dir / 'features'})")
    log(f"zipping {final.name} ...")
    _zip_dir(final, zpath, name)
    code_zip = write_code_zip(out_dir / CODE_ZIP)
    record = {"pack_id": pack_id, "pack_dir": str(final.resolve()), "zip": str(zpath.resolve()),
              "zip_bytes": zpath.stat().st_size, "manifest": manifest,
              "folds": fold_info, "normalizers": {str(k): v for k, v in normalizers.items()}}
    provenance.write_json(pack_record_path(work_dir, pack_id), record)
    log(f"pack size {fmt_bytes(total)}; zip {fmt_bytes(zpath.stat().st_size)} ({zpath})")
    log(f"code {fmt_bytes(code_zip['bytes'])} ({out_dir / CODE_ZIP}, code {code['code_sha256'][:12]})")
    if synthetic:
        log("*** SYNTHETIC pack: for rehearsing the workflow only ***")
    return {"pack_id": pack_id, "pack_dir": final, "zip": zpath, "code_zip": out_dir / CODE_ZIP,
            "total_bytes": total, "zip_bytes": zpath.stat().st_size, "sessions": sessions,
            "manifest": manifest}


def pack_record_path(work_dir: Path, pack_id: str) -> Path:
    return Path(work_dir) / "packs" / f"{pack_id}.json"


# ── reading and verifying a pack (Colab or laptop) ────────────────────────────

def verify_pack(directory: Path) -> dict:
    """Check every file against manifest.json; return the manifest or raise PackError."""
    d = Path(directory)
    if not (d / MANIFEST).is_file():
        raise PackError(f"{d} is not a training pack (no {MANIFEST})")
    try:
        m = provenance.read_json(d / MANIFEST)
    except (OSError, ValueError) as exc:
        raise PackError(f"{d / MANIFEST} is unreadable: {exc}") from exc
    if m.get("kind") != PACK_KIND or m.get("format") != PACK_FORMAT:
        raise PackError(f"{d}: not a training pack of format {PACK_FORMAT}")
    listed = {f["path"]: f for f in m.get("files", [])}
    present = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()} - {MANIFEST}
    problems = [f"missing file {p}" for p in sorted(set(listed) - present)]
    problems += [f"file not in the manifest: {p}" for p in sorted(present - set(listed))]
    for rel in sorted(set(listed) & present):
        f, p = listed[rel], d / rel
        if p.stat().st_size != f["bytes"]:
            problems.append(f"{rel}: {p.stat().st_size} bytes, manifest says {f['bytes']}")
        elif provenance.sha256_file(p) != f["sha256"]:
            problems.append(f"{rel}: SHA-256 checksum differs from the manifest")
    if _pack_id(m.get("files", []), m.get("training_fingerprint"), m.get("code_sha256"),
                m.get("synthetic_data")) != m.get("pack_id"):
        problems.append("the pack ID does not match the manifest contents (manifest edited)")
    if not problems:
        try:
            if Config.from_dict(provenance.read_json(d / "config.json")).training_fingerprint() \
                    != m["training_fingerprint"]:
                problems.append("config.json does not match the manifest's training fingerprint")
            if provenance.read_json(d / "code_version.json").get("code_sha256") != m["code_sha256"]:
                problems.append("code_version.json does not match the manifest")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"config.json or code_version.json unreadable: {exc}")
    if problems:
        shown = problems[:10] + ([f"... and {len(problems) - 10} more"] if len(problems) > 10 else [])
        raise PackError(f"training pack {d} failed verification: " + "; ".join(shown))
    return m


class PackUnit:
    """One session's Class B g1 spectrograms in a pack (read-only memory map)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.meta = provenance.read_json(self.dir / "unit.json")
        self._spec = None

    @property
    def unit_id(self) -> str:
        return self.meta["capture_id"]

    @property
    def n(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def spec(self):
        if self._spec is None:
            import numpy as np
            self._spec = np.load(self.dir / "spec.npy", mmap_mode="r")
            if list(self._spec.shape) != list(self.meta["shape"]) or str(self._spec.dtype) != self.meta["dtype"]:
                raise PackError(f"{self.dir}: spec.npy is {self._spec.shape} {self._spec.dtype}, "
                                f"unit.json says {self.meta['shape']} {self.meta['dtype']}")
        return self._spec


class TrainingPack:
    """A verified training pack: frozen configuration, folds, normalisers and frames."""

    def __init__(self, directory: Path, manifest: dict):
        self.dir, self.manifest = Path(directory), manifest
        self.folds_record = provenance.read_json(self.dir / "folds.json")
        self.units = {cid: PackUnit(self.dir / "sessions" / sid)
                      for sid, cid in self.folds_record["units_by_session"].items()}

    @property
    def pack_id(self) -> str:
        return self.manifest["pack_id"]

    @property
    def synthetic(self) -> bool:
        return bool(self.manifest["synthetic_data"])

    def config(self) -> Config:
        return Config.from_dict(provenance.read_json(self.dir / "config.json"))

    def code_version(self) -> dict:
        return provenance.read_json(self.dir / "code_version.json")

    def fold_record(self, k: int) -> dict:
        for f in self.folds_record["folds"]:
            if int(f["fold"]) == int(k):
                return f
        raise PackError(f"the pack has no fold {k}")

    def folds(self) -> list[Fold]:
        return [Fold(int(f["fold"]), tuple(f["test"]), tuple(f["val"]), tuple(f["train"]))
                for f in self.folds_record["folds"]]

    def normalizer(self, k: int) -> Normalizer:
        return Normalizer.load(self.dir / "normalizers" / f"fold{k}.json")

    def fold_sizes(self, stride: int = 1) -> list[dict]:
        """Training and validation frames per fold (training frames with the frame stride)."""
        out = []
        for f in self.folds_record["folds"]:
            out.append({"fold": int(f["fold"]),
                        "train_frames": sum(len(range(0, self.units[c].n, stride)) for c in f["train_units"]),
                        "val_frames": sum(self.units[c].n for c in f["val_units"])})
        return out

    def fold_frames(self, k: int, cfg: Config) -> tuple[FrameSet, FrameSet, Normalizer, Fold]:
        """Training and validation frames of fold k, normalised with the laptop's statistics."""
        rec = self.fold_record(k)
        norm = self.normalizer(k)
        if list(norm.sources) != list(rec["train_units"]):
            raise PackError(f"fold {k}: the normalisation statistics were fitted on other captures")
        tr = [self.units[c] for c in rec["train_units"]]
        va = [self.units[c] for c in rec["val_units"]]
        fold = Fold(int(rec["fold"]), tuple(rec["test"]), tuple(rec["val"]), tuple(rec["train"]))
        return FrameSet(tr, norm, cfg.train.frame_stride), FrameSet(va, norm, 1), norm, fold

    def identity(self) -> dict:
        m = self.manifest
        return {"pack_id": m["pack_id"], "training_fingerprint": m["training_fingerprint"],
                "code_sha256": m["code_sha256"], "synthetic_data": m["synthetic_data"],
                "created_utc": m["created_utc"]}


def open_pack(directory: Path, verify: bool = True, log=print) -> TrainingPack:
    d = Path(directory)
    if verify:
        m = verify_pack(d)
        log(f"pack {m['pack_id']} verified: {len(m['files'])} files, {fmt_bytes(m['total_bytes'])}"
            + (" (SYNTHETIC)" if m["synthetic_data"] else ""))
    else:
        m = provenance.read_json(d / MANIFEST)
    return TrainingPack(d, m)


# ── training from a pack (Colab) ──────────────────────────────────────────────

def _inside(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def check_output_dir(out_root: Path, pack: TrainingPack, cfg: Config, allow_code_change: bool = False,
                     log=print) -> dict:
    """Create or check out_root/pack_identity.json: one output directory per pack."""
    out_root = Path(out_root)
    path = out_root / IDENTITY_FILE
    code = provenance.code_version()
    pack_code = pack.code_version()
    changed = provenance.changed_code_files(pack_code, code)
    if code["code_sha256"] != pack_code["code_sha256"]:
        msg = (f"the code here ({code['code_sha256'][:12]}) differs from the code the pack was made with "
               f"({pack_code['code_sha256'][:12]}); changed files: {', '.join(changed) or 'unknown'}")
        if not allow_code_change:
            raise PackError(msg + ". Use the occupancy_code.zip written with the pack, or pass "
                                  "--allow-code-change (the change is recorded).")
        log("WARNING: " + msg + " (allowed and recorded)")
    if path.exists():
        ident = provenance.read_json(path)
        if ident.get("pack_id") != pack.pack_id:
            raise PackError(f"{out_root} holds runs trained on pack {ident.get('pack_id')}, not on pack "
                            f"{pack.pack_id}; use another --out directory")
        if bool(ident.get("synthetic_data")) != pack.synthetic:
            raise PackError(f"{out_root}: synthetic and real runs are never mixed")
        if ident.get("training_code_sha256") != code["code_sha256"]:
            msg = (f"{out_root} was started with code {str(ident.get('training_code_sha256'))[:12]}, this is "
                   f"{code['code_sha256'][:12]}")
            if not allow_code_change:
                raise PackError(msg + "; use another --out directory or pass --allow-code-change")
            ident.setdefault("code_changes", []).append({
                "utc": provenance.utc_iso(), "from": ident.get("training_code_sha256"),
                "to": code["code_sha256"],
                "changed_files": provenance.changed_code_files(ident.get("training_code_version", {}), code)})
            ident["training_code_sha256"], ident["training_code_version"] = code["code_sha256"], code
            provenance.write_json(path, ident)
        return ident
    ident = {"kind": OUTPUT_KIND, "pack_id": pack.pack_id, "pack": pack.identity(),
             "synthetic_data": pack.synthetic, "created_utc": provenance.utc_iso(),
             "training_fingerprint": cfg.training_fingerprint(), "config": cfg.to_dict(),
             "folds": pack.folds_record["folds"],
             "normalizers": {str(f.k): pack.normalizer(f.k).to_dict() for f in pack.folds()},
             "pack_code_version": pack_code, "training_code_sha256": code["code_sha256"],
             "training_code_version": code, "code_differs_from_pack": changed, "code_changes": []}
    provenance.write_json(path, ident)
    return ident


def train_from_pack(pack: TrainingPack, cfg: Config, out_root: Path, folds=None, models=None, seeds=None,
                    ablation: bool = False, force: bool = False, resume: bool = True, device=None,
                    allow_code_change: bool = False, log=print) -> list[dict]:
    """Train the selected runs of every selected fold from the pack into out_root."""
    out_root = Path(out_root)
    if cfg.training_fingerprint() != pack.manifest["training_fingerprint"]:
        raise PackError("the training settings differ from the pack's frozen configuration")
    if models and "pca" in models:
        raise PackError("PCA is fitted on the laptop (python -m occupancy train --models pca), not from a pack")
    real_work, syn_work = cfg.work_dir(False), cfg.work_dir(True)
    if pack.synthetic and _inside(out_root, real_work):
        raise PackError("runs trained on a SYNTHETIC pack may not be written into the real work directory")
    if not pack.synthetic and _inside(out_root, syn_work):
        raise PackError("runs trained on real data may not be written into the synthetic work directory")
    if pack.synthetic:
        log("*** SYNTHETIC pack: outputs are for testing only ***")
    out_root.mkdir(parents=True, exist_ok=True)
    check_output_dir(out_root, pack, cfg, allow_code_change, log)
    meta = {"synthetic_data": pack.synthetic, "pack": pack.identity()}
    results = []
    for fold in pack.folds():
        if folds is not None and fold.k not in folds:
            continue
        tr, va, norm, fold = pack.fold_frames(fold.k, cfg)
        results += train_runs(tr, va, norm, fold, cfg, out_root, models, seeds, ablation, force, resume, device,
                              meta, allow_code_change, log)
    return results
