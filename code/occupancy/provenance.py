"""
provenance.py
-------------
What produced a file: git state of code/, library versions, host, power mode.
Every capture sidecar and every result file embeds this record.

The code version is the SHA-256 of the source files of the occupancy package
(and one hash per file). Unlike the git hash, it is also known where the code
runs from a copy without git (a Google Colab runtime), and it changes with any
uncommitted edit. It ties trained models to the code that trained them.

Also here: SHA-256 checksums of files and atomic writes (a temporary file in
the same directory, then a rename), so that an interrupted write never leaves
a damaged file under the final name.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from .config import CODE_DIR

SYNTHETIC_BANNER = "SYNTHETIC DATA - test output only, not a thesis result"
PACKAGE_DIR = Path(__file__).resolve().parent


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def utc_iso(t: _dt.datetime | None = None) -> str:
    t = t or utc_now()
    return t.astimezone(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_utc(s: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(CODE_DIR), *args], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


@functools.lru_cache(maxsize=1)
def _git_info_cached() -> tuple:
    commit = _git("rev-parse", "HEAD")
    prefix = _git("rev-parse", "--show-prefix")          # e.g. "code/"
    tree = _git("rev-parse", f"HEAD:{prefix.rstrip('/')}") if commit and prefix else None
    status = _git("status", "--porcelain", "--", ".")
    return commit, tree, status


def git_info() -> dict:
    """Commit, tree hash of code/ and whether code/ has uncommitted changes.

    Looked up once per process (a run does not change the code it runs).
    """
    commit, tree, status = _git_info_cached()
    return {
        "commit": commit or "unknown",
        "code_tree": tree or "unknown",
        "dirty": None if status is None else bool(status),
    }


def software_version() -> str:
    """Short form for capture metadata: commit[+dirty]."""
    g = git_info()
    s = g["commit"][:12] if g["commit"] != "unknown" else "unknown"
    return s + ("+dirty" if g["dirty"] else "")


@functools.lru_cache(maxsize=1)
def _code_hashes() -> tuple:
    files = {}
    total = hashlib.sha256()
    for p in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(PACKAGE_DIR).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        files[rel] = h
        total.update(f"{rel}\0{h}\n".encode())
    return total.hexdigest(), tuple(files.items())


def code_version(with_files: bool = True) -> dict:
    """SHA-256 over the occupancy source files, the package version and the git state.

    with_files=True adds one SHA-256 per source file, so that a mismatch can be
    reported file by file (changed_code_files).
    """
    from . import __version__
    total, files = _code_hashes()
    out = {"code_sha256": total, "package_version": __version__, "git": git_info()}
    if with_files:
        out["files"] = dict(files)
    return out


def changed_code_files(a: dict, b: dict) -> list[str]:
    """Source files that differ between two code_version() records (added, removed or edited)."""
    fa, fb = a.get("files") or {}, b.get("files") or {}
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def sha256_file(path: str | Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_record(path: str | Path) -> dict:
    """Size and SHA-256 of a file, as stored in manifests and completion markers."""
    return {"bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}


def replace_file(tmp: str | Path, path: str | Path) -> None:
    """Rename tmp over path. Where a file system cannot replace an existing file
    in one step, the old file is removed first (callers keep a fallback copy)."""
    try:
        os.replace(tmp, path)
    except OSError:
        if not Path(tmp).exists():
            raise
        Path(path).unlink(missing_ok=True)
        os.rename(tmp, path)


def atomic_write(path: str | Path, writer: Callable, binary: bool = True, durable: bool = False) -> None:
    """Write a file atomically: writer(f) fills a temporary file in the same
    directory, which is then renamed to path. durable=True also forces the bytes
    to disk before the rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb" if binary else "w", **({} if binary else {"newline": ""})) as f:
        writer(f)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    replace_file(tmp, path)


@functools.lru_cache(maxsize=1)
def _library_versions_cached() -> tuple:
    out = {"python": sys.version.split()[0]}
    for dist in ("numpy", "scipy", "scikit-learn", "torch", "matplotlib", "pyrtlsdr"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return tuple(out.items())


def library_versions() -> dict:
    return dict(_library_versions_cached())


def cpu_info() -> dict:
    model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu_model": model, "logical_cpus": os.cpu_count(),
            "platform": platform.platform()}


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def power_mode() -> dict:
    """Power profile, CPU governor and AC state, where the OS exposes them."""
    info = {
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "cpufreq_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "energy_performance_preference":
            _read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "ac_online": None,
        "power_profiles_daemon": None,
    }
    psu = Path("/sys/class/power_supply")
    if psu.is_dir():
        for d in psu.iterdir():
            if _read(str(d / "type")) == "Mains":
                info["ac_online"] = _read(str(d / "online")) == "1"
    try:
        out = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            info["power_profiles_daemon"] = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def power_condition(info: dict | None = None) -> str:
    """'performance' (mains power and the Performance profile: the condition of
    every timing, design section 10), 'other', or 'unknown' if the OS does not
    expose the power state."""
    info = power_mode() if info is None else info
    ac = info.get("ac_online")
    profile = info.get("power_profiles_daemon") or {
        "performance": "performance", "balanced": "balanced", "quiet": "power-saver",
        "low-power": "power-saver"}.get(info.get("platform_profile") or "")
    if ac is None or profile is None:
        return "unknown"
    return "performance" if ac and profile == "performance" else "other"


POWER_HINT = ("plug in the charger and choose the Performance power profile "
              "(Settings > Power, or: powerprofilesctl set performance)")


def run_record(cfg=None, synthetic: bool = False, **extra: Any) -> dict:
    """Provenance block written into every result file."""
    rec = {
        "created_utc": utc_iso(),
        "argv": sys.argv,
        "git": git_info(),
        "versions": library_versions(),
        "host": cpu_info(),
        "synthetic_data": bool(synthetic),
    }
    if synthetic:
        rec["WARNING"] = SYNTHETIC_BANNER
    if cfg is not None:
        rec["config"] = cfg.to_dict()
        rec["config_fingerprint"] = cfg.fingerprint()
    rec.update(extra)
    return rec


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN/inf -> None, numpy scalars/arrays -> Python."""
    try:
        import numpy as np
    except ImportError:          # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, obj: Any, durable: bool = False) -> None:
    """Atomic JSON write (temporary file in the same directory, then rename).

    durable=True also forces the bytes to disk before the rename (capture
    sidecars and session manifests, which must survive a power cut in the field).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=2)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    replace_file(tmp, path)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
