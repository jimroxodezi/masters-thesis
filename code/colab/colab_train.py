#!/usr/bin/env python3
"""
colab_train.py
--------------
Training on a Google Colab GPU runtime: the logic behind train_on_colab.ipynb.

The script mounts nothing; every path is an argument, so it runs anywhere
(the notebook mounts Google Drive and passes paths under
/content/drive/MyDrive/...). It never needs pyrtlsdr or the receiver.

    check-env    which required packages are missing (numpy, scipy, torch),
                 their versions, and the GPU PyTorch sees
    setup        copy the training pack .zip from Google Drive to the local
                 disk of the runtime, unzip it and verify every checksum
                 (skipped when the same pack is already there and intact)
    time-epoch   python -m occupancy train --pack P --out O --time-epoch:
                 seconds per epoch of each model on this GPU and the estimated
                 hours (and compute units) of the whole design
    train        python -m occupancy train --pack P --out O: first the main
                 models, then (with --ablation) the latent-size ablations.
                 Finished runs are skipped and an unfinished run continues
                 from its last checkpoint, so after a disconnect the same
                 command simply carries on. A failed attempt (for example a
                 Google Drive input/output error) is retried up to
                 --max-attempts times; refusals (wrong pack, settings or code)
                 are not retried. The output is also appended to O/logs/.
    status       complete, unfinished and not started runs in the output directory
    zip-runs     zip the output directory, without checkpoints, for download

The training itself runs in a separate process (python -m occupancy train),
started with CUBLAS_WORKSPACE_CONFIG=:4096:8 so that cuBLAS can work
deterministically (see occupancy/device.py). time-epoch and train use
--device cuda by default: on a runtime without a GPU they stop at once
instead of training on its CPU.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]          # holds occupancy/
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before anything starts CUDA

REQUIRED = ("numpy", "scipy", "torch")       # import names = pip names; scikit-learn is laptop-only (PCA)
EXIT_REFUSED, EXIT_INTERRUPTED = 2, 130
SKIP_IN_ZIP = ("checkpoint.pt", "checkpoint_prev.pt")


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def check_env() -> dict:
    """Missing packages, versions and the GPU (without starting CUDA in this process)."""
    out = {"python": sys.version.split()[0], "missing": [], "versions": {}, "gpu": None}
    for name in REQUIRED:
        if importlib.util.find_spec(name) is None:
            out["missing"].append(name)
        else:
            out["versions"][name] = importlib.metadata.version(name)
    if "torch" not in out["missing"]:
        import torch
        out["gpu"] = {"cuda_available": torch.cuda.is_available(), "torch_cuda_build": torch.version.cuda,
                      "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            if smi.returncode == 0:
                out["gpu"]["nvidia_smi"] = smi.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def setup(pack_zip: Path, local: Path) -> Path:
    """Copy the pack zip to local disk, unzip and verify it; return the pack directory."""
    from occupancy.pack import PackError, fmt_bytes, open_pack, safe_extract, verify_pack
    pack_zip, local = Path(pack_zip), Path(local)
    if not pack_zip.is_file():
        raise PackError(f"{pack_zip} not found: upload the pack written by 'python -m occupancy pack'")
    try:
        with zipfile.ZipFile(pack_zip) as z:
            roots = {n.split("/")[0] for n in z.namelist()}
            if len(roots) != 1:
                raise PackError(f"{pack_zip}: expected one top-level directory, found {sorted(roots)}")
            root = roots.pop()
            wanted = json.loads(z.read(f"{root}/manifest.json"))["pack_id"]
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise PackError(f"{pack_zip} is not a training pack zip ({exc})") from exc
    target = local / root
    if (target / "manifest.json").exists():
        try:
            if verify_pack(target)["pack_id"] == wanted:
                print(f"pack {wanted} already set up and verified in {target}")
                return target
        except PackError as exc:
            print(f"local copy is not usable ({exc}); copying again")
        shutil.rmtree(target)
    local.mkdir(parents=True, exist_ok=True)
    tmp_zip = local / (pack_zip.name + ".partial")
    print(f"copying {pack_zip} ({fmt_bytes(pack_zip.stat().st_size)}) to {local} ...", flush=True)
    t0 = time.perf_counter()
    shutil.copyfile(pack_zip, tmp_zip)
    print(f"  {time.perf_counter() - t0:.0f} s; unzipping ...", flush=True)
    try:
        safe_extract(tmp_zip, local)
    finally:
        tmp_zip.unlink(missing_ok=True)
    open_pack(target)                                       # verifies every checksum
    return target


def _run_occupancy(args: list[str], log_path: Path | None) -> int:
    """python -m occupancy <args> in a child process; output to the console and the log."""
    cmd = [sys.executable, "-u", "-m", "occupancy", *map(str, args)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(CODE_ROOT), os.environ.get("PYTHONPATH", "")]),
           "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a")
        log.write(f"\n$ {' '.join(cmd)}   [{_utc_stamp()}]\n")
    try:
        proc = subprocess.Popen(cmd, cwd=CODE_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if log is not None:
                    log.write(line)
                    log.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.wait()
            return EXIT_INTERRUPTED
    finally:
        if log is not None:
            log.close()


def _common(a) -> list[str]:
    out = []
    if a.threads:
        out += ["--threads", str(a.threads)]
    if a.device:
        out += ["--device", a.device]
    if a.allow_synthetic:
        out.append("--allow-synthetic")
    if a.allow_code_change:
        out.append("--allow-code-change")
    for flag, vals in (("--folds", a.folds), ("--models", a.models), ("--seeds", a.seeds)):
        if vals:
            out += [flag, *map(str, vals)]
    return out


def time_epoch(a) -> int:
    args = ["train", "--pack", a.pack, "--out", a.out, "--time-epoch", *_common(a)]
    if a.units_per_hour:
        args += ["--units-per-hour", str(a.units_per_hour)]
    if a.time_batches:
        args += ["--time-batches", str(a.time_batches)]
    return _run_occupancy(args, Path(a.out) / "logs" / f"time_epoch_{_utc_stamp()}.log")


def train(a) -> int:
    """Main models, then the ablations; each phase retried after failures (resume is automatic)."""
    log = Path(a.out) / "logs" / f"train_{_utc_stamp()}.log"
    phases = [("main models", [])] + ([("latent-size ablations", ["--ablation"])] if a.ablation else [])
    for name, extra in phases:
        for attempt in range(1, max(1, a.max_attempts) + 1):
            print(f"=== {name}: attempt {attempt} of {a.max_attempts} ===", flush=True)
            code = _run_occupancy(["train", "--pack", a.pack, "--out", a.out, *extra, *_common(a)], log)
            if code == 0:
                break
            if code in (EXIT_REFUSED, EXIT_INTERRUPTED):
                print(f"stopped (exit code {code}): " + ("refused, see the error above" if code == EXIT_REFUSED
                                                         else "interrupted"))
                return code
            if attempt < a.max_attempts:
                print(f"attempt failed (exit code {code}); resuming in {a.wait} s ...", flush=True)
                time.sleep(a.wait)
        else:
            print(f"{name}: still failing after {a.max_attempts} attempts (log: {log}). If Google Drive was "
                  "disconnected, mount it again and re-run this step.")
            return code
    print(f"training finished (log: {log})")
    return 0


def status(pack_dir: Path, out: Path, ablation: bool) -> dict:
    """Counts of complete, unfinished and not started runs; lists the unfinished ones."""
    from occupancy.pack import open_pack
    from occupancy.train import expected_runs, model_dir, run_status
    pack = open_pack(Path(pack_dir), verify=False, log=lambda *x: None)
    cfg = pack.config()
    runs = expected_runs(cfg, [f.k for f in pack.folds()], ablation=ablation)
    counts, unfinished = {}, []
    for k, spec, seed in runs:
        st = run_status(model_dir(out, k, spec, seed))
        counts[st["state"]] = counts.get(st["state"], 0) + 1
        if st["state"] == "unfinished":
            unfinished.append(f"fold{k}/{spec.name}/seed{seed} ({st.get('epochs_done', 0)} epochs done)")
    print(f"{len(runs)} runs" + (" (with ablations)" if ablation else "") + ": "
          + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    for u in unfinished:
        print("  unfinished:", u)
    return {"total": len(runs), "counts": counts, "unfinished": unfinished}


def zip_runs(out: Path, zip_path: Path) -> Path:
    """runs/<...> without checkpoints, temporary files and the timing scratch directories."""
    out, zip_path = Path(out), Path(zip_path)
    tmp = zip_path.with_name(zip_path.name + ".partial")
    n, size = 0, 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in sorted(out.rglob("*")):
            rel = p.relative_to(out)
            if (not p.is_file() or p.name in SKIP_IN_ZIP or p.name.endswith((".tmp", ".partial"))
                    or any(part.startswith(".time_epoch") for part in rel.parts)
                    or p.resolve() in (zip_path.resolve(), tmp.resolve())):
                continue
            z.write(p, f"runs/{rel.as_posix()}")
            n, size = n + 1, size + p.stat().st_size
    os.replace(tmp, zip_path)
    print(f"{zip_path}: {n} files, {size / 1e6:.1f} MB before compression, "
          f"{zip_path.stat().st_size / 1e6:.1f} MB zipped")
    return zip_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colab_train.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("check-env", help="missing packages, versions, GPU")
    e.add_argument("--json", action="store_true")
    s = sub.add_parser("setup", help="copy, unzip and verify the training pack")
    s.add_argument("--pack-zip", required=True)
    s.add_argument("--local", required=True, help="local directory, e.g. /content/msc-thesis")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--pack", required=True, help="the verified pack directory (from setup)")
    run.add_argument("--out", required=True, help="output directory on Drive, e.g. .../MyDrive/msc-thesis/runs")
    run.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1),
                     help="torch CPU threads (default: min(4, CPUs of the runtime))")
    run.add_argument("--device", default="cuda",
                     help="cuda (default: a Colab GPU runtime is expected, so a runtime without a GPU fails at "
                          "once instead of training for weeks on its CPU); auto or cpu for a rehearsal")
    run.add_argument("--allow-synthetic", action="store_true", help="accept a SYNTHETIC pack (rehearsal)")
    run.add_argument("--allow-code-change", action="store_true")
    run.add_argument("--folds", nargs="+", type=int)
    run.add_argument("--models", nargs="+")
    run.add_argument("--seeds", nargs="+")
    t = sub.add_parser("time-epoch", parents=[run], help="time one epoch per model, estimate GPU-hours")
    t.add_argument("--units-per-hour", type=float)
    t.add_argument("--time-batches", type=int)
    tr = sub.add_parser("train", parents=[run], help="train (resumes automatically), with retries")
    tr.add_argument("--ablation", action="store_true", help="then also the latent-size ablations")
    tr.add_argument("--max-attempts", type=int, default=3)
    tr.add_argument("--wait", type=float, default=30.0, help="seconds between attempts")
    st = sub.add_parser("status", help="progress of the runs")
    st.add_argument("--pack", required=True)
    st.add_argument("--out", required=True)
    st.add_argument("--ablation", action="store_true")
    z = sub.add_parser("zip-runs", help="zip the outputs (no checkpoints) for download")
    z.add_argument("--out", required=True)
    z.add_argument("--zip", required=True)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "check-env":
            env = check_env()
            if a.json:
                print(json.dumps(env))
            else:
                print(f"python {env['python']}; " + ", ".join(f"{k} {v}" for k, v in env["versions"].items()))
                print("missing:", ", ".join(env["missing"]) or "nothing")
                print("GPU:", env["gpu"])
            return 0
        if a.cmd == "setup":
            print(setup(Path(a.pack_zip), Path(a.local)))
            return 0
        if a.cmd == "time-epoch":
            return time_epoch(a)
        if a.cmd == "train":
            return train(a)
        if a.cmd == "status":
            status(Path(a.pack), Path(a.out), a.ablation)
            return 0
        if a.cmd == "zip-runs":
            zip_runs(Path(a.out), Path(a.zip))
            return 0
    except Exception as exc:  # noqa: BLE001 - one clear line in the notebook instead of a traceback
        from occupancy.pack import PackError
        if isinstance(exc, PackError):
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        raise
    return 1


if __name__ == "__main__":
    sys.exit(main())
