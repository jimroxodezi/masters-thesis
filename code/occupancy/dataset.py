"""
dataset.py
----------
Capture index, frames and the session-level splits (design sections 3-5).

* ``index_captures`` reads every sidecar under ``<data_dir>/raw`` and refuses
  synthetic captures unless ``allow_synthetic=True`` (and always refuses a mix
  of real and synthetic captures). Captures whose status is not "ok" (aborted,
  retaken) are skipped.
* A frame is N = 16384 consecutive samples, non-overlapping, starting at the
  first sample of the capture; DC removal subtracts the complex frame mean.
* Folds: sessions in ascending (natural) order are cut into 4 contiguous test
  groups whose sizes differ by at most one. For fold k: test = group k,
  validation = the first 2 sessions of group k+1 (cyclic), training = the rest.
  The design fixes the rotation but not how sessions are assigned to groups;
  contiguous blocks are used so that each test group covers consecutive sessions.
"""

from __future__ import annotations

import datetime as _dt
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import provenance
from .config import CLASSES, GAIN_ROLES, Config
from .iqio import META_SUFFIX, open_raw, u8_to_complex

LABEL = {"A": 0, "B": 0, "C_centre": 1, "C_offset": 1}


class SyntheticDataError(RuntimeError):
    """Synthetic captures reached code that produces results without permission."""


class DatasetError(RuntimeError):
    pass


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


@dataclass(frozen=True)
class CaptureRecord:
    capture_id: str
    session_id: str
    position_id: str
    cls: str
    gain_role: str
    gain_db: float
    centre_freq_hz: float
    sample_rate_hz: float
    n_samples: int
    utc_start: str
    raw_path: str
    meta_path: str
    synthetic: bool
    clipping_fraction: float
    clipping_flagged: bool

    def n_frames(self, frame_len: int) -> int:
        return self.n_samples // frame_len

    @property
    def label(self) -> int:
        return LABEL[self.cls]

    def frame_time(self, frame_index: int, frame_len: int) -> _dt.datetime:
        return provenance.parse_utc(self.utc_start) + _dt.timedelta(
            seconds=frame_index * frame_len / self.sample_rate_hz)


class CaptureIndex:
    def __init__(self, records: list[CaptureRecord], synthetic: bool, skipped: list | None = None):
        self.records = sorted(records, key=lambda r: (natural_key(r.session_id),
                                                      CLASSES.index(r.cls), r.gain_role))
        self.synthetic = synthetic
        self.skipped = skipped or []
        self._by_key = {(r.session_id, r.cls, r.gain_role): r for r in self.records}

    def __len__(self):
        return len(self.records)

    def sessions(self) -> list[str]:
        return sorted({r.session_id for r in self.records}, key=natural_key)

    def get(self, session_id: str, cls: str, gain_role: str) -> CaptureRecord | None:
        return self._by_key.get((session_id, cls, gain_role))

    def session_records(self, session_id: str) -> list[CaptureRecord]:
        return [r for r in self.records if r.session_id == session_id]

    def position_of(self, session_id: str) -> str:
        pos = {r.position_id for r in self.session_records(session_id)}
        return sorted(pos)[0] if pos else ""

    def by_id(self, capture_id: str) -> CaptureRecord:
        for r in self.records:
            if r.capture_id == capture_id:
                return r
        raise KeyError(capture_id)

    def summary(self) -> dict:
        return {"synthetic": self.synthetic, "n_captures": len(self.records),
                "sessions": self.sessions(),
                "records": [asdict(r) for r in self.records], "skipped": self.skipped}


def index_captures(data_dir: Path, cfg: Config, allow_synthetic: bool = False,
                   log=print) -> CaptureIndex:
    """Index all complete captures under data_dir/raw (sidecars with kind == 'capture')."""
    root = Path(data_dir) / "raw"
    if not root.is_dir():
        raise DatasetError(f"no captures: {root} does not exist")
    records, skipped = [], []
    for meta_path in sorted(root.rglob(f"*{META_SUFFIX}")):
        try:
            meta = provenance.read_json(meta_path)
        except (OSError, ValueError) as exc:
            skipped.append({"file": str(meta_path), "reason": f"unreadable: {exc}"})
            continue
        if not isinstance(meta, dict) or meta.get("kind") != "capture":
            continue
        if meta.get("status") != "ok":
            skipped.append({"file": str(meta_path), "reason": f"status {meta.get('status')}"})
            continue
        raw_path = meta_path.with_name(meta["file"])
        n = int(meta["measurements"]["n_samples"])
        if not raw_path.exists() or raw_path.stat().st_size != 2 * n:
            skipped.append({"file": str(meta_path), "reason": "raw file missing or wrong size"})
            continue
        if meta.get("class") not in CLASSES or meta.get("gain_role") not in GAIN_ROLES:
            raise DatasetError(f"{meta_path}: bad class or gain role")
        fs = float(meta["radio"]["sample_rate_hz"])
        if abs(fs - cfg.capture.sample_rate_hz) > 1e-6:
            raise DatasetError(f"{meta_path}: sample rate {fs} differs from the fixed "
                               f"{cfg.capture.sample_rate_hz}")
        records.append(CaptureRecord(
            capture_id=meta["capture_id"], session_id=meta["session_id"],
            position_id=meta.get("position_id", ""), cls=meta["class"],
            gain_role=meta["gain_role"], gain_db=float(meta["radio"]["gain_db"]),
            centre_freq_hz=float(meta["radio"]["centre_freq_hz"]), sample_rate_hz=fs,
            n_samples=n, utc_start=meta["utc_start"], raw_path=str(raw_path),
            meta_path=str(meta_path), synthetic=bool(meta.get("synthetic", False)),
            clipping_fraction=float(meta["measurements"].get("clipping_fraction", 0.0)),
            clipping_flagged=bool(meta["measurements"].get("clipping_flagged", False))))
    if not records:
        raise DatasetError(f"no usable captures under {root}")
    n_syn = sum(r.synthetic for r in records)
    if n_syn and n_syn != len(records):
        raise SyntheticDataError(f"{n_syn} synthetic and {len(records) - n_syn} real captures under "
                                 f"{root}: real and synthetic data are never mixed")
    synthetic = n_syn > 0
    if synthetic and not allow_synthetic:
        raise SyntheticDataError(f"{root} holds synthetic captures. They are refused unless "
                                 "--allow-synthetic is given (tests only); real results never come "
                                 "from synthetic data.")
    keys = {}
    for r in records:
        k = (r.session_id, r.cls, r.gain_role)
        if k in keys:
            raise DatasetError(f"two usable captures for {k}: {keys[k]} and {r.capture_id}. "
                               "Mark one as retaken (status) in its sidecar.")
        keys[k] = r.capture_id
    flagged = [r.capture_id for r in records if r.clipping_flagged]
    if flagged:
        log(f"note: {len(flagged)} capture(s) flagged for clipping: {', '.join(flagged)}")
    if skipped:
        log(f"note: {len(skipped)} sidecar(s) skipped (see index summary)")
    return CaptureIndex(records, synthetic, skipped)


def dc_remove(frames: np.ndarray) -> np.ndarray:
    """Subtract the complex mean of each frame (design section 4)."""
    frames = np.asarray(frames)
    return frames - frames.mean(axis=-1, keepdims=True)


def read_frames(record: CaptureRecord, frame_len: int, start: int = 0, stop: int | None = None,
                dc: bool = True) -> np.ndarray:
    """Frames [start, stop) of a capture as complex64 (n, N), DC removed by default."""
    n_total = record.n_frames(frame_len)
    stop = n_total if stop is None else min(stop, n_total)
    if start >= stop:
        return np.zeros((0, frame_len), dtype=np.complex64)
    raw = open_raw(record.raw_path)[2 * frame_len * start: 2 * frame_len * stop]
    y = u8_to_complex(np.asarray(raw)).reshape(stop - start, frame_len)
    return dc_remove(y).astype(np.complex64) if dc else y


def read_frames_at(record: CaptureRecord, frame_len: int, idx: np.ndarray) -> np.ndarray:
    """Arbitrary frames (any order, repeats allowed), DC removed."""
    idx = np.asarray(idx, dtype=np.int64)
    raw = open_raw(record.raw_path)
    out = np.empty((idx.size, frame_len), dtype=np.complex64)
    for j, i in enumerate(idx):
        out[j] = u8_to_complex(np.asarray(raw[2 * frame_len * i: 2 * frame_len * (i + 1)]))
    return dc_remove(out).astype(np.complex64)


# ── session-level cross-validation (design section 5) ─────────────────────────

@dataclass(frozen=True)
class Fold:
    k: int
    test: tuple
    val: tuple
    train: tuple

    def to_dict(self) -> dict:
        return {"fold": self.k, "test": list(self.test), "val": list(self.val), "train": list(self.train)}


def make_folds(sessions, cfg: Config, log=print) -> list[Fold]:
    s = cfg.split
    sessions = sorted(set(sessions), key=natural_key)
    n = len(sessions)
    if n < s.min_sessions:
        raise DatasetError(f"{n} sessions; the {s.n_folds}-fold rotation needs at least {s.min_sessions}")
    if n < s.design_min_sessions:
        msg = f"only {n} sessions; the design asks for at least {s.design_min_sessions} (D4)"
        warnings.warn(msg)
        log("WARNING: " + msg)
    sizes = [len(g) for g in np.array_split(np.arange(n), s.n_folds)]
    groups, i = [], 0
    for size in sizes:
        groups.append(tuple(sessions[i:i + size]))
        i += size
    if min(sizes) < s.n_val_sessions:
        raise DatasetError("each test group needs at least as many sessions as the validation set")
    folds = []
    for k in range(s.n_folds):
        test = groups[k]
        nxt = sorted(groups[(k + 1) % s.n_folds], key=natural_key)
        val = tuple(nxt[:s.n_val_sessions])
        train = tuple(x for x in sessions if x not in test and x not in val)
        folds.append(Fold(k, test, val, train))
    return folds


def fold_records(index: CaptureIndex, sessions, classes, gains) -> list[CaptureRecord]:
    out = []
    for sid in sessions:
        for cls in classes:
            for g in gains:
                r = index.get(sid, cls, g)
                if r is not None:
                    out.append(r)
    return out


def train_records(index: CaptureIndex, fold: Fold, cfg: Config) -> list[CaptureRecord]:
    """Training data [D3]: Class B at g1 of the training sessions."""
    recs = fold_records(index, fold.train, (cfg.split.train_class,), (cfg.split.primary_gain,))
    if not recs:
        raise DatasetError(f"fold {fold.k}: no training captures")
    return recs


def val_records(index: CaptureIndex, fold: Fold, cfg: Config) -> list[CaptureRecord]:
    """Validation data: Class B at g1 of the validation sessions."""
    recs = fold_records(index, fold.val, ("B",), (cfg.split.primary_gain,))
    if not recs:
        raise DatasetError(f"fold {fold.k}: no validation captures (Class B at g1)")
    return recs
