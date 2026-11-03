"""
capture.py
----------
RTL-SDR Blog V4 capture to an interleaved uint8 I/Q file plus a JSON sidecar
(design section 1).

Rules enforced here
* Manual gain only: "auto" (or anything that is not one of the tuner's gain
  steps) is refused; the RTL2832 digital AGC is switched off explicitly.
* Bias tee forced off through librtlsdr (``rtlsdr_set_bias_tee``) when the
  library provides it; the control used is logged.
* 1.92 MS/s; the centre frequency must lie inside the R828D tuning range.
* The receiver must identify itself as an R828D-based "Blog V4" (EEPROM
  product string), unless the operator explicitly accepts another receiver.
* Samples are streamed in blocks of 131072 samples (Linux usbfs limits a
  single transfer to 16 MB) straight to disk; the file is written as
  ``.partial`` and renamed only when complete, and the sidecar is written last.

Nothing in this module runs at import time; pyrtlsdr is imported only when a
receiver is opened.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import provenance
from .config import CLASSES, GAIN_ROLES, Config
from .iqio import (PARTIAL_SUFFIX, CaptureStats, capture_metadata, capture_paths,
                   make_capture_id)

TUNER_NAMES = {0: "unknown", 1: "E4000", 2: "FC0012", 3: "FC0013", 4: "FC2580",
               5: "R820T", 6: "R828D"}


class GainError(ValueError):
    pass


class ReceiverError(RuntimeError):
    pass


# ── parsing and validation ────────────────────────────────────────────────────

def parse_freq(value) -> float:
    """'796M', '796.5e6', '1.2G', '500k' or a plain number of Hz -> Hz."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("_", "")
    mult = 1.0
    if s and s[-1] in "kKmMgG":
        mult = {"k": 1e3, "m": 1e6, "g": 1e9}[s[-1].lower()]
        s = s[:-1]
    if s.lower().endswith("hz"):
        s = s[:-2]
    return float(s) * mult


def validate_gain(gain) -> float:
    """Manual gain in dB; automatic gain control is never allowed (design section 1)."""
    if gain is None:
        raise GainError("a manual gain in dB is required (g1 or g2 from the survey)")
    if isinstance(gain, str):
        if gain.strip().lower() in ("auto", "agc", "automatic"):
            raise GainError("automatic gain is not allowed: give a fixed gain in dB")
        try:
            gain = float(gain)
        except ValueError as exc:
            raise GainError(f"gain must be a number in dB, got {gain!r}") from exc
    gain = float(gain)
    if not np.isfinite(gain) or gain < 0 or gain > 60:
        raise GainError(f"gain {gain} dB is outside the tuner's range")
    return gain


def check_centre_freq(freq_hz: float, cfg: Config) -> float:
    c = cfg.capture
    half = c.sample_rate_hz / 2
    if freq_hz + half > c.max_centre_freq_hz:
        raise ValueError(f"{freq_hz / 1e6:.3f} MHz: the window would exceed the tuner limit "
                         f"({c.max_centre_freq_hz / 1e6:.0f} MHz). LTE Band 3 (1805 MHz and above) "
                         "cannot be captured with the RTL-SDR.")
    if freq_hz - half < c.min_centre_freq_hz:
        raise ValueError(f"{freq_hz / 1e6:.3f} MHz is below the normal tuning range")
    return freq_hz


def n_rb_for(cfg: Config, channel_bw_mhz=None, n_rb=None) -> int | None:
    if n_rb is not None:
        return int(n_rb)
    if channel_bw_mhz is None:
        return None
    key = f"{float(channel_bw_mhz):g}"
    table = cfg.capture.nrb_by_channel_bw_mhz
    if key not in table:
        raise ValueError(f"unknown LTE channel bandwidth {channel_bw_mhz} MHz; "
                         f"expected one of {sorted(table, key=float)}")
    return int(table[key])


def c_offset_plan(cfg: Config, carrier_centre_hz: float, channel_bw_mhz=None, n_rb=None,
                  side: int = +1) -> dict:
    """Window placement for Class C_offset (design section 3).

    f_off = (0.54 MHz + W_occ/2) / 2 with W_occ = N_RB x 0.18 MHz, i.e. midway
    between the edge of the central 6 RBs and the occupied edge. Feasible only
    if W_occ/2 - 0.54 MHz >= 1.92 MHz (channel bandwidth >= 10 MHz).
    """
    if side not in (-1, +1):
        raise ValueError("side must be +1 or -1")
    c = cfg.capture
    nrb = n_rb_for(cfg, channel_bw_mhz, n_rb)
    plan = {"rule": "f_off = (0.54 MHz + W_occ/2)/2, W_occ = N_RB x 0.18 MHz (design section 3)",
            "carrier_centre_hz": float(carrier_centre_hz), "channel_bw_mhz": channel_bw_mhz,
            "n_rb": nrb, "side": side}
    if nrb is None:
        plan.update(feasible=False, reason="carrier bandwidth unknown: C_offset cannot be placed")
        return plan
    w_occ = nrb * c.rb_width_hz
    room = w_occ / 2 - c.central_rbs_half_width_hz
    f_off = (c.central_rbs_half_width_hz + w_occ / 2) / 2
    half = c.sample_rate_hz / 2
    plan.update(occupied_width_hz=w_occ, f_off_hz=f_off,
                window_from_centre_hz=(f_off - half, f_off + half),
                room_hz=room, feasible=bool(room >= c.sample_rate_hz))
    if plan["feasible"]:
        plan["centre_freq_hz"] = float(carrier_centre_hz + side * f_off)
    else:
        plan["reason"] = (f"carrier too narrow: W_occ/2 - 0.54 MHz = {room / 1e6:.2f} MHz "
                          f"< {c.sample_rate_hz / 1e6:.2f} MHz (needs a channel bandwidth of 10 MHz or more)")
    return plan


# ── librtlsdr access (ctypes) ─────────────────────────────────────────────────

def _librtlsdr():
    from rtlsdr.librtlsdr import librtlsdr   # loads the shared library
    return librtlsdr


def librtlsdr_info() -> dict:
    """Which librtlsdr is loaded and whether it is the RTL-SDR Blog build."""
    lib = _librtlsdr()
    path = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "librtlsdr" in line:
                    path = line.split()[-1]
                    break
    except OSError:
        pass
    return {"path": path,
            "blog_build": hasattr(lib, "rtlsdr_check_dongle_model"),
            "has_bias_tee": hasattr(lib, "rtlsdr_set_bias_tee")}


def usb_strings(sdr) -> dict:
    """Manufacturer, product and serial EEPROM strings of the open device."""
    lib = _librtlsdr()
    f = lib.rtlsdr_get_usb_strings
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    m, p, s = (ctypes.create_string_buffer(256) for _ in range(3))
    rc = f(sdr.dev_p, m, p, s)
    if rc != 0:
        return {"manufacturer": None, "product": None, "serial": None, "error": rc}
    dec = lambda b: b.value.decode("ascii", errors="replace")
    return {"manufacturer": dec(m), "product": dec(p), "serial": dec(s)}


def set_bias_tee(sdr, on: bool) -> str:
    lib = _librtlsdr()
    if not hasattr(lib, "rtlsdr_set_bias_tee"):
        if on:
            raise ReceiverError("this librtlsdr cannot switch the bias tee")
        return "not controllable with this librtlsdr; assumed off (power-on default)"
    f = lib.rtlsdr_set_bias_tee
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_int]
    rc = f(sdr.dev_p, 1 if on else 0)
    if rc != 0:
        raise ReceiverError(f"rtlsdr_set_bias_tee failed with code {rc}")
    return f"rtlsdr_set_bias_tee({1 if on else 0}) returned 0"


def open_receiver(device_index: int = 0, serial: str | None = None):
    try:
        from rtlsdr import RtlSdr
    except ImportError as exc:
        raise ReceiverError(f"cannot load pyrtlsdr/librtlsdr: {exc}. Install the RTL-SDR Blog "
                            "driver build of librtlsdr (see README).") from exc
    try:
        return RtlSdr(device_index=device_index, serial_number=serial)
    except Exception as exc:  # pyrtlsdr raises IOError/LibUSBError
        raise ReceiverError(f"cannot open the RTL-SDR: {exc}. Is it plugged in, and is the "
                            "dvb_usb_rtl28xxu kernel module blacklisted?") from exc


def receiver_identity(sdr, cfg: Config) -> dict:
    ident = usb_strings(sdr)
    try:
        ident["tuner"] = TUNER_NAMES.get(int(sdr.get_tuner_type()), "unknown")
    except Exception:  # noqa: BLE001
        ident["tuner"] = "unknown"
    ident["librtlsdr"] = librtlsdr_info()
    product_ok = bool(ident.get("product")) and cfg.capture.expected_product_substring in ident["product"]
    ident["matches_expected"] = bool(product_ok and ident["tuner"] == cfg.capture.expected_tuner)
    return ident


def check_receiver(ident: dict, cfg: Config, allow_other_receiver: bool) -> list[str]:
    warnings = []
    if not ident["librtlsdr"]["blog_build"]:
        warnings.append("the loaded librtlsdr is not the RTL-SDR Blog build; the V4 needs it")
    if not ident["matches_expected"]:
        msg = (f"receiver is not an {cfg.capture.receiver_model}: product={ident.get('product')!r}, "
               f"tuner={ident.get('tuner')!r}")
        if not allow_other_receiver:
            raise ReceiverError(msg + ". Use the V4, or pass --allow-other-receiver to record anyway "
                                "(the mismatch is then logged).")
        warnings.append(msg + " (accepted by the operator)")
    return warnings


def configure_receiver(sdr, cfg: Config, centre_freq_hz: float, gain_db: float) -> dict:
    """Apply the design's radio settings and read back what the driver reports."""
    c = cfg.capture
    gain_db = validate_gain(gain_db)
    check_centre_freq(centre_freq_hz, cfg)
    if c.ppm:
        # librtlsdr returns an error if the correction is unchanged, so only set non-zero values
        sdr.set_freq_correction(int(c.ppm))
    sdr.sample_rate = c.sample_rate_hz
    sdr.center_freq = centre_freq_hz
    valid = [float(g) for g in sdr.valid_gains_db]
    if not any(abs(g - gain_db) < 0.05 for g in valid):
        raise GainError(f"{gain_db} dB is not a gain step of this tuner; valid steps: {valid}")
    sdr.set_manual_gain_enabled(True)
    sdr.gain = gain_db
    sdr.set_agc_mode(False)          # RTL2832 digital AGC off
    bias = set_bias_tee(sdr, c.bias_tee_on)
    return {
        "centre_freq_hz_reported": float(sdr.get_center_freq()),
        "sample_rate_hz_actual": float(sdr.get_sample_rate()),
        "gain_db_reported": float(sdr.get_gain()),
        "valid_gains_db": valid,
        "ppm_reported": int(sdr.get_freq_correction()),
        "rtl_agc": "off",
        "bias_tee_control": bias,
    }


# ── streaming ─────────────────────────────────────────────────────────────────

def stream_to_file(sdr, path: Path, n_samples: int, block_samples: int, settle_samples: int,
                   stats: CaptureStats, stall_timeout_s: float = 5.0) -> dict:
    """Stream n_samples complex samples (2 bytes each) from sdr into path.

    ``sdr`` needs pyrtlsdr's ``read_bytes_async(callback, num_bytes)`` and
    ``cancel_read_async()``. The first ``settle_samples`` are discarded. A
    watchdog cancels the read if no block arrives for ``stall_timeout_s``.
    """
    need = 2 * n_samples
    skip = 2 * settle_samples
    state = {"written": 0, "skipped": 0, "error": None, "t_first": None, "t_last": None,
             "last_cb": time.monotonic(), "done": False, "blocks": 0}
    lock = threading.Lock()

    def cancel():
        try:
            sdr.cancel_read_async()
        except Exception:  # noqa: BLE001 - cancelling twice or late must not fail a capture
            pass

    with open(path, "wb") as f:
        def callback(values, _ctx):
            if state["done"]:
                return
            try:
                buf = np.ctypeslib.as_array(values) if not isinstance(values, np.ndarray) else values
                now = time.time()
                with lock:
                    state["last_cb"] = time.monotonic()
                    state["blocks"] += 1
                start = 0
                if state["skipped"] < skip:
                    start = min(skip - state["skipped"], buf.size)
                    state["skipped"] += start
                take = min(buf.size - start, need - state["written"])
                if take > 0:
                    chunk = np.array(buf[start:start + take], dtype=np.uint8, copy=True)
                    f.write(chunk.tobytes())
                    stats.update(chunk)
                    if state["t_first"] is None:
                        # arrival time of the first kept block (its samples end here)
                        state["t_first"] = now
                        state["t_first_block_samples"] = take // 2
                    state["written"] += take
                    state["t_last"] = now
            except BaseException as exc:  # noqa: BLE001 - incl. Ctrl-C: nothing may escape a C callback
                state["error"] = exc
                state["done"] = True
                cancel()
                return
            if state["written"] >= need:
                state["done"] = True
                cancel()

        def watchdog():
            while not state["done"]:
                time.sleep(0.5)
                with lock:
                    idle = time.monotonic() - state["last_cb"]
                if idle > stall_timeout_s and not state["done"]:
                    state["error"] = ReceiverError(f"no samples for {idle:.1f} s (USB stalled?)")
                    state["done"] = True
                    cancel()

        wd = threading.Thread(target=watchdog, daemon=True)
        t0 = time.perf_counter()
        wd.start()
        try:
            sdr.read_bytes_async(callback, num_bytes=2 * block_samples)
        finally:
            state["done"] = True
        elapsed = time.perf_counter() - t0
        f.flush()
        os.fsync(f.fileno())               # the raw file must survive a power cut in the field
    if state["error"] is not None:
        raise state["error"]
    if state["written"] < need:
        raise ReceiverError(f"capture stopped early: {state['written'] // 2:,} of {n_samples:,} samples")
    return {"elapsed_s": elapsed, "t_first": state["t_first"], "t_last": state["t_last"],
            "first_block_samples": state.get("t_first_block_samples", 0), "blocks": state["blocks"]}


def _utc_from_epoch(t: float) -> str:
    import datetime as _dt
    return provenance.utc_iso(_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc))


def record_capture(cfg: Config, out_dir: Path, *, session_id: str, position_id: str, cls: str,
                   gain_role: str, gain_db, centre_freq_hz: float, duration_s: float | None = None,
                   antenna: str = "", temperature_c=None, cell_search: dict | None = None,
                   notes: str = "", extra_context: dict | None = None, sdr=None,
                   device_index: int = 0, allow_other_receiver: bool = False,
                   log=print) -> dict:
    """Capture one class at one gain and write <capture_id>.cu8 and .json into out_dir."""
    if cls not in CLASSES:
        raise ValueError(f"class must be one of {CLASSES}")
    if gain_role not in GAIN_ROLES:
        raise ValueError(f"gain role must be one of {GAIN_ROLES}")
    gain_db = validate_gain(gain_db)
    check_centre_freq(centre_freq_hz, cfg)
    c = cfg.capture
    duration_s = float(duration_s or c.duration_s)
    n_samples = int(round(duration_s * c.sample_rate_hz))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    if free < c.min_free_disk_factor * 2 * n_samples:
        raise OSError(f"only {free / 1e9:.2f} GB free in {out_dir}; need "
                      f"{c.min_free_disk_factor * 2 * n_samples / 1e9:.2f} GB")
    if cls == "A" and not antenna:
        antenna = c.terminator_label

    t_req = provenance.utc_now()
    capture_id = make_capture_id(session_id, cls, gain_role, t_req)
    raw_path, meta_path = capture_paths(out_dir, capture_id)
    partial = raw_path.with_name(raw_path.name + PARTIAL_SUFFIX)

    own = sdr is None
    if own:
        sdr = open_receiver(device_index)
    try:
        ident = receiver_identity(sdr, cfg)
        warnings = check_receiver(ident, cfg, allow_other_receiver)
        for w in warnings:
            log(f"WARNING: {w}")
        radio = configure_receiver(sdr, cfg, centre_freq_hz, gain_db)
        stats = CaptureStats()
        log(f"capturing {capture_id}: {centre_freq_hz / 1e6:.4f} MHz, {gain_db:g} dB, {duration_s:g} s ...")
        timing = stream_to_file(sdr, partial, n_samples, c.read_block_samples,
                                c.settle_discard_samples, stats)
    finally:
        if own:
            sdr.close()

    partial.replace(raw_path)
    block_s = timing["first_block_samples"] / c.sample_rate_hz
    radio.update({"receiver_manufacturer": ident.get("manufacturer"),
                  "receiver_product": ident.get("product"),
                  "receiver_serial": ident.get("serial"),
                  "tuner": ident.get("tuner"),
                  "librtlsdr": ident.get("librtlsdr"),
                  "receiver_check": "ok" if ident["matches_expected"] else "MISMATCH accepted by operator"})
    meas = stats.result(c.clip_flag_fraction)
    meas["settle_discarded_samples"] = c.settle_discard_samples
    meas["usb_blocks"] = timing["blocks"]
    meta = capture_metadata(
        cfg=cfg, capture_id=capture_id, session_id=session_id, position_id=position_id, cls=cls,
        gain_role=gain_role, gain_db=gain_db, centre_freq_hz=centre_freq_hz,
        # start = arrival of the first kept block minus its duration (about +-70 ms)
        utc_start=_utc_from_epoch(timing["t_first"] - block_s),
        utc_end=_utc_from_epoch(timing["t_last"]),
        raw_file=raw_path.name, stats=meas, radio=radio, antenna=antenna,
        temperature_c=temperature_c, cell_search=cell_search, notes=notes,
        extra_context={**(extra_context or {}), "warnings": warnings},
        synthetic=False, wall_clock_s=timing["elapsed_s"])
    provenance.write_json(meta_path, meta, durable=True)
    return meta


def receiver_info(cfg: Config, device_index: int = 0) -> dict:
    """Open the receiver, report identity, gain steps and bias-tee control; no streaming."""
    sdr = open_receiver(device_index)
    try:
        ident = receiver_identity(sdr, cfg)
        ident["valid_gains_db"] = [float(g) for g in sdr.valid_gains_db]
        ident["bias_tee"] = set_bias_tee(sdr, False)
        return ident
    finally:
        sdr.close()
