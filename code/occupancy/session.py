"""
session.py
----------
Guided capture session (design section 3): the 8 captures of one session in the
fixed order A, B, C_centre, C_offset, each at g1 then g2 (this order needs the
50 ohm terminator to be fitted and removed only once per session).

The helper asks for the session details and the frequencies (from the day's
survey and cell search), plans the C_offset window with the design's rule
(f_off = (0.54 MHz + W_occ/2)/2; not captured, and recorded, for carriers
narrower than 10 MHz), checks the Class B window against the carrier, shows
the plan, then walks through the captures. After each capture it prints the
clipping fraction and mean power and writes a quick-look PNG; the operator can
continue, retake (the earlier capture is kept, marked "retaken") or quit.
A session manifest (session.json) is rewritten after every step, so an
interrupted session leaves a consistent record.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import provenance
from .capture import c_offset_plan, check_centre_freq, parse_freq, record_capture, validate_gain
from .config import Config
from .dataset import natural_key, time_of_day
from .quicklook import capture_quicklook


class SessionAbort(Exception):
    pass


@dataclass
class PlannedCapture:
    cls: str
    gain_role: str
    gain_db: float
    centre_freq_hz: float
    instruction: str
    cell_search: dict = field(default_factory=dict)
    extra_context: dict = field(default_factory=dict)


def _ask(input_fn, text, default=None, cast=str, choices=None, allow_empty=False):
    shown = f" [{default}]" if default not in (None, "") else ""
    while True:
        ans = input_fn(f"{text}{shown}: ").strip()
        if ans == "" and default is not None:
            ans = str(default)
        if ans == "" and allow_empty:
            return None
        if ans == "":
            continue
        if choices and ans not in choices:
            print(f"  please answer one of {', '.join(choices)}")
            continue
        try:
            return cast(ans)
        except (ValueError, TypeError) as exc:
            print(f"  not understood ({exc}); try again")


def _yes(input_fn, text, default=False) -> bool:
    ans = input_fn(f"{text} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if ans == "" else ans in ("y", "yes")


def suggest_session_id(raw_root: Path) -> str:
    existing = sorted((d.name for d in Path(raw_root).glob("S*") if d.is_dir()), key=natural_key)
    nums = [int(n[1:]) for n in existing if n[1:].isdigit()]
    return f"S{(max(nums) + 1) if nums else 1:02d}"


def _defaults_path(data_dir: Path) -> Path:
    return Path(data_dir) / "field_defaults.json"


def load_field_defaults(data_dir: Path) -> dict:
    p = _defaults_path(data_dir)
    return provenance.read_json(p) if p.exists() else {}


def plan_session(cfg: Config, info: dict) -> tuple[list[PlannedCapture], list[dict]]:
    """Captures in the fixed order; C_offset is left out (and recorded) when infeasible."""
    gains = {"g1": info["g1_db"], "g2": info["g2_db"]}
    freq = {"A": info["f_a_hz"], "B": info["f_b_hz"], "C_centre": info["f_c_hz"],
            "C_offset": info.get("f_c_offset_hz")}
    cs_c = {"tool": info.get("cell_tool"), "pci": info.get("pci"), "carrier_centre_hz": info["f_c_hz"],
            "bandwidth_mhz": info.get("bandwidth_mhz"), "result": "cell confirmed"}
    cs_b = {"tool": info.get("cell_tool"), "result": info.get("b_cell_result", "no cell found")}
    instr = {
        "A": f"Fit the {cfg.capture.terminator_label} to the antenna input (bias tee is forced off).",
        "B": "Remove the terminator and connect the antenna.",
        "C_centre": "Antenna connected; window centred on the carrier.",
        "C_offset": "Antenna connected; window inside the carrier, away from its central 6 RBs.",
    }
    plan, skipped = [], []
    for cls in cfg.capture.session_order:
        for role in cfg.capture.gain_order:
            if cls == "C_offset" and not info["c_offset_plan"].get("feasible"):
                skipped.append({"class": cls, "gain_role": role, "reason": info["c_offset_plan"].get("reason")})
                continue
            cs = cs_c if cls.startswith("C") else (cs_b if cls == "B" else {})
            extra = {"c_offset_plan": info["c_offset_plan"]} if cls == "C_offset" else {}
            if cls == "B":
                extra["b_window_check"] = info.get("b_window_check")
            plan.append(PlannedCapture(cls, role, gains[role], freq[cls], instr[cls], cs, extra))
    return plan, skipped


def b_window_check(cfg: Config, f_b: float, f_c: float, bandwidth_mhz) -> dict:
    """Distance of the Class B window from the carrier; overlap with its channel if known."""
    half = cfg.capture.sample_rate_hz / 2
    out = {"distance_to_carrier_hz": abs(f_b - f_c)}
    if bandwidth_mhz is not None:
        edge = float(bandwidth_mhz) * 1e6 / 2
        out["clear_of_carrier_channel"] = bool(abs(f_b - f_c) >= edge + half)
    return out


def _gather_info(cfg: Config, data_dir: Path, input_fn, log) -> dict:
    d = load_field_defaults(data_dir)
    raw_root = Path(data_dir) / "raw"
    info = {}
    log("\n=== Capture session (design section 3) ===")
    info["session_id"] = _ask(input_fn, "Session ID", suggest_session_id(raw_root))
    if (raw_root / info["session_id"] / "session.json").exists():
        if not _yes(input_fn, f"{info['session_id']} already has a manifest. Add to it?"):
            raise SessionAbort("session exists")
    info["position_id"] = _ask(input_fn, "Position ID", d.get("position_id", "P1"))
    info["g1_db"] = _ask(input_fn, "g1, primary gain (dB)", d.get("g1_db", cfg.capture.g1_db), validate_gain)
    info["g2_db"] = _ask(input_fn, "g2, gain-change gain (dB)", d.get("g2_db", cfg.capture.g2_db), validate_gain)
    if info["g1_db"] == info["g2_db"]:
        log("WARNING: g1 and g2 are equal; the gain-change test needs two different gains")
    info["ppm"] = _ask(input_fn, "Frequency correction (ppm)", d.get("ppm", cfg.capture.ppm), int)
    info["antenna"] = _ask(input_fn, "Antenna", d.get("antenna", cfg.capture.antenna or "passive antenna"))
    info["temperature_c"] = _ask(input_fn, "Temperature (deg C, blank if unknown)", None, float,
                                 allow_empty=True)
    info["notes"] = _ask(input_fn, "Operator notes (place, weather, ...)", "", allow_empty=True) or ""
    log("\n-- Frequencies (from today's survey and cell search) --")
    info["cell_tool"] = _ask(input_fn, "Cell-search tool", d.get("cell_tool", "LTE-Cell-Scanner"))
    info["f_c_hz"] = _ask(input_fn, "LTE carrier centre frequency (e.g. 796M)", d.get("f_c"), parse_freq)
    check_centre_freq(info["f_c_hz"], cfg)
    info["pci"] = _ask(input_fn, "PCI (blank if unknown)", d.get("pci"), int, allow_empty=True)
    info["bandwidth_mhz"] = _ask(input_fn, "Channel bandwidth in MHz (1.4/3/5/10/15/20, blank if unknown)",
                                 d.get("bandwidth_mhz"), float, allow_empty=True)
    side = _ask(input_fn, "Side of the carrier for C_offset (+ or -)", d.get("side", "+"), str,
                choices=("+", "-"))
    plan = c_offset_plan(cfg, info["f_c_hz"], info["bandwidth_mhz"], side=+1 if side == "+" else -1)
    if plan["feasible"]:
        lo, hi = plan["window_from_centre_hz"]
        log(f"C_offset: f_off = {plan['f_off_hz'] / 1e6:.3f} MHz; window {lo / 1e6:.2f}-{hi / 1e6:.2f} MHz "
            f"from the centre, inside the occupied band and clear of the central 6 RBs")
        f_co = _ask(input_fn, "C_offset centre frequency", f"{plan['centre_freq_hz'] / 1e6:.4f}M", parse_freq)
        check_centre_freq(f_co, cfg)
        if abs(f_co - plan["centre_freq_hz"]) > 1.0:
            plan["operator_override_hz"] = f_co
            log("note: C_offset frequency differs from the rule; the override is recorded")
        info["f_c_offset_hz"] = f_co
    else:
        log(f"C_offset will NOT be captured: {plan['reason']}. This is recorded in the manifest.")
    info["c_offset_plan"] = plan
    info["f_b_hz"] = _ask(input_fn, "Class B centre frequency (idle stretch >= 1.92 MHz, close to the carrier)",
                          d.get("f_b"), parse_freq)
    check_centre_freq(info["f_b_hz"], cfg)
    info["b_window_check"] = b_window_check(cfg, info["f_b_hz"], info["f_c_hz"], info["bandwidth_mhz"])
    if info["b_window_check"].get("clear_of_carrier_channel") is False:
        log("WARNING: the Class B window overlaps the carrier's channel")
        if not _yes(input_fn, "Use this Class B frequency anyway?"):
            raise SessionAbort("Class B window overlaps the carrier")
    if _yes(input_fn, "Did cell search at the Class B frequency find no cell?", False):
        info["b_cell_result"] = "no cell found"
    else:
        log("WARNING: Class B must be confirmed free by survey and cell search at capture time")
        info["b_cell_result"] = "NOT confirmed by cell search"
    info["f_a_hz"] = _ask(input_fn, "Class A centre frequency", f"{info['f_b_hz'] / 1e6:.4f}M", parse_freq)
    check_centre_freq(info["f_a_hz"], cfg)
    defaults = {"position_id": info["position_id"], "g1_db": info["g1_db"], "g2_db": info["g2_db"],
                "ppm": info["ppm"], "antenna": info["antenna"], "cell_tool": info["cell_tool"],
                "f_c": f"{info['f_c_hz'] / 1e6:.4f}M", "pci": info["pci"],
                "bandwidth_mhz": info["bandwidth_mhz"], "side": side, "f_b": f"{info['f_b_hz'] / 1e6:.4f}M"}
    provenance.write_json(_defaults_path(data_dir), defaults)
    return info


def run_session(cfg: Config, data_dir: Path, *, synthetic: bool = False, input_fn=input, log=print,
                capture_fn=None, duration_s: float | None = None, device_index: int = 0,
                allow_other_receiver: bool = False, quicklook: bool = True) -> dict:
    data_dir = Path(data_dir)
    info = _gather_info(cfg, data_dir, input_fn, log)
    cfg.capture.ppm = int(info["ppm"])
    duration_s = float(duration_s or cfg.capture.duration_s)
    plan, skipped = plan_session(cfg, info)
    out_dir = data_dir / "raw" / info["session_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    need = 2 * duration_s * cfg.capture.sample_rate_hz * len(plan)
    free = shutil.disk_usage(out_dir).free
    log(f"\nPlan for {info['session_id']} ({'SYNTHETIC' if synthetic else 'RTL-SDR'}):")
    for i, pc in enumerate(plan, 1):
        log(f"  {i}. {pc.cls:8s} {pc.gain_role} = {pc.gain_db:5.1f} dB  {pc.centre_freq_hz / 1e6:10.4f} MHz  "
            f"{duration_s:g} s")
    for s in skipped:
        log(f"  -  {s['class']:8s} {s['gain_role']}: not captured ({s['reason']})")
    log(f"Disk: {need / 1e6:.0f} MB needed, {free / 1e9:.1f} GB free")
    if free < need * cfg.capture.min_free_disk_factor:
        raise SessionAbort("not enough free disk space")
    tod = time_of_day(provenance.utc_now(), cfg)
    if tod == "off-schedule":
        log(f"WARNING: sessions start at {cfg.capture.day_start_local} (day) or "
            f"{cfg.capture.night_start_local} (night) local time; this one will be labelled off-schedule")
    else:
        log(f"Time of day: {tod}")
    if not _yes(input_fn, "Start?", True):
        raise SessionAbort("cancelled before the first capture")

    manifest_path = out_dir / "session.json"
    manifest = provenance.read_json(manifest_path) if manifest_path.exists() else {
        "kind": "session", "session_id": info["session_id"], "captures": [], "started_utc": provenance.utc_iso()}
    manifest.update({"synthetic": synthetic, "position_id": info["position_id"], "time_of_day": tod,
                     "g1_db": info["g1_db"], "g2_db": info["g2_db"], "ppm": info["ppm"],
                     "antenna": info["antenna"], "temperature_c": info["temperature_c"], "notes": info["notes"],
                     "frequencies_hz": {"A": info["f_a_hz"], "B": info["f_b_hz"], "C_centre": info["f_c_hz"],
                                        "C_offset": info.get("f_c_offset_hz")},
                     "cell_search": {"tool": info["cell_tool"], "pci": info["pci"],
                                     "bandwidth_mhz": info["bandwidth_mhz"],
                                     "class_b_result": info["b_cell_result"]},
                     "c_offset": info["c_offset_plan"], "b_window_check": info["b_window_check"],
                     "not_captured": skipped, "software_version": provenance.software_version()})
    provenance.write_json(manifest_path, manifest, durable=True)

    if capture_fn is None:
        capture_fn = _synthetic_capture_fn(cfg, info) if synthetic else record_capture
    done = {(c["class"], c["gain_role"]) for c in manifest["captures"] if c["status"] == "ok"}
    i = 0
    while i < len(plan):
        pc = plan[i]
        if (pc.cls, pc.gain_role) in done:
            log(f"\n[{i + 1}/{len(plan)}] Class {pc.cls}, {pc.gain_role}: already captured in this session")
            i += 1
            continue
        log(f"\n[{i + 1}/{len(plan)}] Class {pc.cls}, {pc.gain_role} = {pc.gain_db:g} dB, "
            f"{pc.centre_freq_hz / 1e6:.4f} MHz, {duration_s:g} s")
        log(f"      {pc.instruction}")
        ans = input_fn("      Enter = start, s = skip, q = quit: ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            manifest["not_captured"].append({"class": pc.cls, "gain_role": pc.gain_role, "reason": "skipped by operator"})
            provenance.write_json(manifest_path, manifest, durable=True)
            i += 1
            continue
        try:
            meta = capture_fn(cfg, out_dir, session_id=info["session_id"], position_id=info["position_id"],
                              cls=pc.cls, gain_role=pc.gain_role, gain_db=pc.gain_db,
                              centre_freq_hz=pc.centre_freq_hz, duration_s=duration_s,
                              antenna=info["antenna"] if pc.cls != "A" else cfg.capture.terminator_label,
                              temperature_c=info["temperature_c"], cell_search=pc.cell_search,
                              notes=info["notes"], extra_context=pc.extra_context,
                              **({} if synthetic else {"device_index": device_index,
                                                       "allow_other_receiver": allow_other_receiver}))
        except KeyboardInterrupt:
            log("capture interrupted; the partial file is kept as .partial and is never indexed")
            ans = input_fn("      r = retry, q = quit: ").strip().lower()
            if ans == "q":
                break
            continue
        except Exception as exc:  # noqa: BLE001 - report and let the operator decide (USB, disk, gain)
            log(f"capture FAILED: {type(exc).__name__}: {exc}")
            log("      nothing usable was written for this capture (a .partial file may remain)")
            ans = input_fn("      r = retry, q = quit: ").strip().lower()
            if ans == "q":
                break
            continue
        m = meta["measurements"]
        flag = "  ** CLIPPING FLAG: above 0.1 % **" if m["clipping_flagged"] else ""
        log(f"      done {meta['capture_id']}: {m['n_samples']:,} samples, mean power "
            f"{m['mean_power_db']:.2f} dB, clipping {100 * m['clipping_fraction']:.4f} %{flag}")
        if quicklook:
            try:
                png = capture_quicklook(out_dir / meta["file"], meta, cfg,
                                        out_dir / f"{meta['capture_id']}_quicklook.png")
                log(f"      quick look: {png}")
            except Exception as exc:  # noqa: BLE001 - a plot must never stop a session
                log(f"      (quick look failed: {exc})")
        manifest["captures"].append({"capture_id": meta["capture_id"], "class": pc.cls,
                                     "gain_role": pc.gain_role, "status": "ok",
                                     "clipping_flagged": m["clipping_flagged"]})
        provenance.write_json(manifest_path, manifest, durable=True)
        ans = input_fn("      Enter = next, r = retake, q = quit: ").strip().lower()
        if ans == "r":
            _mark_retaken(out_dir, meta["capture_id"], manifest)
            provenance.write_json(manifest_path, manifest, durable=True)
            continue
        i += 1
        if ans == "q":
            break
    manifest["ended_utc"] = provenance.utc_iso()
    provenance.write_json(manifest_path, manifest, durable=True)
    log(f"\nSession {info['session_id']}: {sum(c['status'] == 'ok' for c in manifest['captures'])} captures. "
        "Remember the labelling check on a later day (python -m occupancy inspect).")
    return manifest


def _mark_retaken(out_dir: Path, capture_id: str, manifest: dict) -> None:
    side = Path(out_dir) / f"{capture_id}.json"
    meta = provenance.read_json(side)
    meta["status"] = "retaken"
    meta["status_note"] = "operator chose to retake this capture; kept for the record, never indexed"
    provenance.write_json(side, meta, durable=True)
    for c in manifest["captures"]:
        if c["capture_id"] == capture_id:
            c["status"] = "retaken"


def _synthetic_capture_fn(cfg: Config, info: dict):
    """Capture function for rehearsing a session without a dongle (synthetic data only)."""
    from .synthetic import SessionSignalParams, stable_seed, write_synthetic_capture
    params = SessionSignalParams.draw(np.random.default_rng(stable_seed(info["session_id"], "params")))

    def fn(cfg_, out_dir, *, session_id, position_id, cls, gain_role, gain_db, centre_freq_hz, duration_s,
           antenna, temperature_c, cell_search, notes, extra_context):
        t = provenance.utc_now().replace(microsecond=0)
        while any(Path(out_dir).glob(f"{session_id}_{cls}_{gain_role}_{t.strftime('%Y%m%dT%H%M%SZ')}.*")):
            t += _dt.timedelta(seconds=1)          # a quick retake within the same second
        n = int(round(duration_s * cfg_.capture.sample_rate_hz))
        return write_synthetic_capture(
            cfg_, out_dir, session_id=session_id, position_id=position_id, cls=cls, gain_role=gain_role,
            gain_db=gain_db, centre_freq_hz=centre_freq_hz, n_samples=n, params=params, t_start=t,
            seed=stable_seed(session_id, cls, gain_role, t.isoformat()), notes=notes,
            cell_search=cell_search, extra_context=extra_context,
            antenna=antenna)
    return fn


__all__ = ["run_session", "plan_session", "suggest_session_id", "b_window_check", "SessionAbort"]
