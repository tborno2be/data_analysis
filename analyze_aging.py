#!/usr/bin/env python3
"""
analyze_aging.py  —  Electrode-aging tolerance tracker for a timed CV/CA session.

WHAT THIS ANSWERS
-----------------
"How far can the electrode age before the CV result changes by an amount the
user will not tolerate?"  We repeat the SAME CV (3 scans, 100 mV/s) and the
SAME CA (0.5 V) on a fixed schedule and watch a small set of benchmarks drift.

BENCHMARKS (标杆)
-----------------
  gap  = ΔEp = Epa - Epc          peak-to-peak separation (kinetics / fouling)
  Epa, Epc                         anodic / cathodic peak potentials (drift)
  ipa, ipc                         anodic / cathodic peak currents (sensitivity)
  |ipa/ipc|                        reversibility ratio
  tail = right/left half-width     peak asymmetry (Pearson-IV-style tailing)
  Ep MAD (across 3 scans)          within-run repeatability of Ep
  ip CV%  (across 3 scans)         within-run repeatability of ip
  beta = exponent in i(t)=A*t^-b+C  CA diffusional decay (0.5 = ideal Cottrell)

Everything is BLANK-FREE and DETERMINISTIC so the same electrode measured twice
gives the same number.  A benchmark is only reported when the peak clears a
signal-to-noise gate (SNR_MIN); otherwise it is flagged "below detection" and
plotted as an open marker, never as a fake trend line.

USAGE
-----
  python analyze_aging.py <session_dir>            # e.g. analysis_07172026
  python analyze_aging.py <session_dir> --out figs # figure/CSV output folder

Outputs (written into <session_dir>/analysis_result/ by default):
  benchmarks_cv.csv   one row per (cv_file, scan) + per-file medians
  benchmarks_ca.csv   one row per ca_file (beta fit)
  fig_diagnostic.png  raw CV overlays across the session + CA decays (reality check)
  fig_gap.png         ΔEp / Epa / Epc vs session, with tolerance bands
  fig_current.png     ipa / ipc / ratio + within-run repeatability
  fig_tail.png        tail asymmetry vs session
  fig_beta.png        CA beta vs session
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# -------- headless matplotlib --------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scipy is optional; degrade gracefully if missing
try:
    from scipy.signal import savgol_filter
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:            # pragma: no cover
    _HAVE_SCIPY = False


# ======================================================================
#  TOLERANCE CONFIG  —  edit these to your acceptance limits.
#  "User cannot tolerate" is defined here.  A run FAILS a criterion when the
#  metric leaves the band relative to the run-start reference (first CV/CA).
# ======================================================================
TOL = {
    "SNR_MIN":        3.0,   # peak must exceed 3x local noise to be "found"
    "dEpa_drift_mV":  10.0,  # |Epa - Epa_ref| sustained > this  -> 峰电位持续偏移
    "dEpc_drift_mV":  10.0,
    "dGap_incr_mV":   15.0,  # (gap - gap_ref) > this            -> 氧化/还原峰间距异常增大
    "ip_change_pct":  15.0,  # |ip/ip_ref - 1| > this            -> 峰电流明显下降/升高
    "Ep_MAD_mV":       5.0,  # within-run Ep MAD across scans    -> 峰电位标准差/MAD增大
    "ip_CVpct":        5.0,  # within-run ip CV% across scans    -> 峰电流多scan漂移
    "beta_lo":        0.40,  # CA beta acceptable window (ideal Cottrell = 0.5)
    "beta_hi":        0.60,
    # sensitivity/slope calibration limit is applied only if a calibration
    # slope column is supplied externally; tracked as ip drift here.
}

# Okabe-Ito colour-blind-safe palette (matches your plot.py)
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
       "#56B4E9", "#F0E442", "#000000"]


# ----------------------------------------------------------------------
#  primitives
# ----------------------------------------------------------------------
def noise_mad(x: np.ndarray) -> float:
    """Successive-difference MAD noise estimate (robust, matches core CV code)."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def _smooth(y: np.ndarray) -> np.ndarray:
    n = len(y)
    if _HAVE_SCIPY and n >= 9:
        w = min(11, n if n % 2 else n - 1)
        if w >= 5:
            return savgol_filter(y, w, 2)
    # moving-average fallback
    k = min(7, n)
    if k < 3:
        return y.astype(float)
    return np.convolve(y, np.ones(k) / k, "same")


@dataclass
class PeakPick:
    E: float           # peak potential (V)
    ip: float          # peak height above local baseline (A, signed)
    snr: float         # ip / branch noise
    tail: float        # right/left half-width ratio (>1 = tailing), NaN if unresolved
    found: bool


def _pick_branch(E: np.ndarray, i_s: np.ndarray, noise: float, polarity: int) -> PeakPick:
    """Pick the faradaic extremum on one half-sweep, blank-free.

    Baseline = straight line joining the two ends of the branch (foot-to-foot).
    Peak = extremum of (smoothed - baseline) in the branch interior.
    polarity +1 = anodic (look for max), -1 = cathodic (look for min).
    """
    n = len(E)
    if n < 12:
        return PeakPick(np.nan, np.nan, np.nan, np.nan, False)
    base = np.linspace(i_s[0], i_s[-1], n)
    resid = (i_s - base) * polarity          # so peak is always a positive bump
    # ignore the very edges (guard) where foot-line forces resid≈0
    g = max(3, n // 20)
    interior = np.arange(g, n - g)
    if len(interior) < 3:
        return PeakPick(np.nan, np.nan, np.nan, np.nan, False)
    k = interior[int(np.argmax(resid[interior]))]
    height = resid[k]
    snr = float(height / noise) if noise and np.isfinite(noise) else np.nan
    found = bool(np.isfinite(snr) and snr >= TOL["SNR_MIN"])
    Ep = float(E[k]); ip = float(height * polarity)
    # tail asymmetry: half-widths at half-max on each side of k (in E units)
    tail = np.nan
    if found:
        half = height / 2.0
        # left
        li = k
        while li > interior[0] and resid[li] > half:
            li -= 1
        ri = k
        while ri < interior[-1] and resid[ri] > half:
            ri += 1
        wl = abs(E[k] - E[li]); wr = abs(E[ri] - E[k])
        if wl > 0:
            tail = float(wr / wl)
    return PeakPick(Ep, ip, snr, tail, found)


# ----------------------------------------------------------------------
#  CV
# ----------------------------------------------------------------------
def read_cv(path: Path):
    """Return {scan:int -> (E, i, t)} dropping the CA preamble rows (scan blank)."""
    scans: dict[int, tuple[list, list, list]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("scan"):
                continue
            s = int(float(row["scan"]))
            E, i, t = scans.setdefault(s, ([], [], []))
            E.append(float(row["potential"])); i.append(float(row["current"]))
            t.append(float(row["time"]) if row.get("time") else np.nan)
    return {s: (np.array(E), np.array(i), np.array(t)) for s, (E, i, t) in scans.items()}


def analyze_cv_scan(E, i):
    """One CV scan -> dict of benchmarks (Epa/Epc/ipa/ipc/gap/tail/SNR)."""
    noise = noise_mad(i)
    i_s = _smooth(i)
    v = int(np.argmax(E))                    # forward/reverse vertex
    Ef, If = E[:v + 1], i_s[:v + 1]          # anodic (E rising)
    Er, Ir = E[v:], i_s[v:]                  # cathodic (E falling)
    an = _pick_branch(Ef, If, noise, +1)
    ca = _pick_branch(Er, Ir, noise, -1)
    gap = (an.E - ca.E) if (an.found and ca.found) else np.nan
    ratio = abs(an.ip / ca.ip) if (an.found and ca.found and ca.ip != 0) else np.nan
    return {
        "noise_nA": noise * 1e9,
        "Epa": an.E, "ipa_nA": an.ip * 1e9, "snr_a": an.snr, "tail_a": an.tail, "found_a": an.found,
        "Epc": ca.E, "ipc_nA": ca.ip * 1e9, "snr_c": ca.snr, "tail_c": ca.tail, "found_c": ca.found,
        "gap_mV": gap * 1e3 if np.isfinite(gap) else np.nan,
        "ratio_a_c": ratio,
    }


# ----------------------------------------------------------------------
#  CA  (beta = Cottrell-ish decay exponent)
# ----------------------------------------------------------------------
def read_ca(path: Path):
    t, i = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.lower().startswith("time"):
            continue
        parts = line.replace("\t", ",").split(",")
        if len(parts) < 2:
            continue
        try:
            t.append(float(parts[0])); i.append(float(parts[1]))
        except ValueError:
            continue
    t, i = np.array(t), np.array(i)
    if len(t):
        t = t - t.min()
    return t, i


def _model_beta(t, A, beta, C):
    return A * np.power(t, -beta) + C


def fit_ca_beta(t, i, start_s=0.5):
    m = np.isfinite(t) & np.isfinite(i) & (t > 0)
    t, i = t[m], i[m]
    sel = t >= start_s
    t, i = t[sel], i[sel]
    out = {"beta": np.nan, "beta_se": np.nan, "A": np.nan, "C": np.nan,
           "r2": np.nan, "n_points": int(len(t)),
           "i_first_nA": (i[0] * 1e9 if len(i) else np.nan),
           "i_last_nA": (i[-1] * 1e9 if len(i) else np.nan),
           "amp_nA": ((i.max() - i.min()) * 1e9 if len(i) else np.nan)}
    if not _HAVE_SCIPY or len(t) < 20:
        return out
    c0 = float(np.median(i[-max(3, len(i) // 10):]))
    a0 = float(i[0] - c0) or float(np.std(i) or 1e-12)
    try:
        popt, pcov = curve_fit(_model_beta, t, i, p0=[a0, 0.5, c0],
                               bounds=([-np.inf, 0.0, -np.inf], [np.inf, 2.0, np.inf]),
                               maxfev=50000)
        yh = _model_beta(t, *popt)
        ss_res = float(np.sum((i - yh) ** 2)); ss_tot = float(np.sum((i - i.mean()) ** 2))
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 3
        out.update(A=float(popt[0]), beta=float(popt[1]), C=float(popt[2]),
                   beta_se=float(perr[1]),
                   r2=(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------
#  session
# ----------------------------------------------------------------------
def load_session(session_dir: Path):
    """Parse session.csv -> ordered lists of CV and CA records with timestamps."""
    sc = session_dir / "session.csv"
    cv, ca, t0 = [], [], None
    if sc.exists():
        with open(sc, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ev = r["event"]; od = r.get("out_dir", "")
                ts = r.get("t_start") or ""
                try:
                    dt = datetime.fromisoformat(ts) if ts else None
                except ValueError:
                    dt = None
                if dt and t0 is None:
                    t0 = dt
                mins = ((dt - t0).total_seconds() / 60.0) if (dt and t0) else np.nan
                if ev == "CV" and od:
                    cv.append({"dir": od, "t_min": mins})
                elif ev == "CA" and od:
                    ca.append({"dir": od, "t_min": mins})
    # fallback: glob folders if session.csv absent/empty
    if not cv:
        cv = [{"dir": p.name, "t_min": np.nan} for p in sorted(session_dir.glob("cv_*"))]
    if not ca:
        ca = [{"dir": p.name, "t_min": np.nan} for p in sorted(session_dir.glob("ca_*"))]
    return cv, ca


def run(session_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_recs, ca_recs = load_session(session_dir)

    # ---- CV: per scan + per file ----
    cv_rows = []; cv_files = []
    for idx, rec in enumerate(cv_recs, 1):
        d = session_dir / rec["dir"] / "data.csv"
        if not d.exists():
            continue
        scans = read_cv(d)
        per_scan = []
        for s in sorted(scans):
            E, i, _ = scans[s]
            m = analyze_cv_scan(E, i)
            m.update(cv_index=idx, t_min=rec["t_min"], file=rec["dir"], scan=s)
            cv_rows.append(m); per_scan.append(m)
        # per-file medians / repeatability
        def col(key, need_found=None):
            vals = []
            for r in per_scan:
                if need_found and not r[need_found]:
                    continue
                v = r[key]
                if np.isfinite(v):
                    vals.append(v)
            return np.array(vals)
        Epa = col("Epa", "found_a"); Epc = col("Epc", "found_c")
        ipa = col("ipa_nA", "found_a"); ipc = col("ipc_nA", "found_c")
        gap = col("gap_mV")
        def med(a): return float(np.median(a)) if len(a) else np.nan
        def mad_mV(a):   # a already in volts->pass Epa in V
            return float(np.median(np.abs(a - np.median(a))) * 1e3) if len(a) else np.nan
        def cvpct(a):
            return float(100 * np.std(a) / abs(np.mean(a))) if len(a) and np.mean(a) else np.nan
        cv_files.append({
            "cv_index": idx, "t_min": rec["t_min"], "file": rec["dir"],
            "n_scan_found_a": int(len(Epa)), "n_scan_found_c": int(len(Epc)),
            "Epa_med": med(Epa), "Epc_med": med(Epc), "gap_med_mV": med(gap),
            "ipa_med_nA": med(ipa), "ipc_med_nA": med(ipc),
            "ratio_med": med(col("ratio_a_c")),
            "tail_a_med": med(col("tail_a", "found_a")),
            "Epa_MAD_mV": mad_mV(Epa), "Epc_MAD_mV": mad_mV(Epc),
            "ipa_CVpct": cvpct(ipa), "ipc_CVpct": cvpct(ipc),
            "noise_med_nA": med(col("noise_nA")),
        })

    # ---- CA: beta per file ----
    ca_files = []
    for idx, rec in enumerate(ca_recs, 1):
        d = session_dir / rec["dir"] / "data.csv"
        if not d.exists():
            continue
        t, i = read_ca(d)
        fit = fit_ca_beta(t, i)
        fit.update(ca_index=idx, t_min=rec["t_min"], file=rec["dir"])
        ca_files.append(fit)

    _write_csv(out_dir / "benchmarks_cv.csv", cv_files)
    _write_csv(out_dir / "benchmarks_ca.csv", ca_files)

    # ---- figures ----
    _fig_diagnostic(session_dir, cv_recs, ca_recs, out_dir / "fig_diagnostic.png")
    _fig_gap(cv_files, out_dir / "fig_gap.png")
    _fig_current(cv_files, cv_rows, out_dir / "fig_current.png")
    _fig_tail(cv_files, out_dir / "fig_tail.png")
    _fig_beta(ca_files, out_dir / "fig_beta.png")

    _print_summary(cv_files, ca_files)
    return cv_files, ca_files


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------------------------------------------------
#  figures
# ----------------------------------------------------------------------
def _style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9, colors="#333333")
    ax.grid(True, color="#eeeeee", lw=0.8, zorder=0)


def _xvals(rows):
    """Prefer elapsed minutes; fall back to index."""
    tm = np.array([r["t_min"] for r in rows], float)
    if np.all(np.isfinite(tm)):
        return tm, "Elapsed time (min)"
    return np.array([r["cv_index"] for r in rows], float), "CV index"


def _fig_diagnostic(session_dir, cv_recs, ca_recs, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=200)
    n = len(cv_recs)
    cmap = plt.cm.viridis
    peak2peak = []
    for k, rec in enumerate(cv_recs):
        d = session_dir / rec["dir"] / "data.csv"
        if not d.exists():
            continue
        scans = read_cv(d)
        if 0 not in scans:
            continue
        E, i, _ = scans[0]
        i_s = _smooth(i) * 1e9
        peak2peak.append(i_s.max() - i_s.min())
        ax1.plot(E, i_s, color=cmap(k / max(n - 1, 1)), lw=1.0, alpha=0.9)
    ax1.set_xlabel("Potential  E  (V)"); ax1.set_ylabel("Current  i  (nA, smoothed)")
    ax1.set_title(f"CV scan-1 overlay, all {n} repeats (dark→light = early→late)")
    _style(ax1)
    if peak2peak:
        med_noise = np.median([noise_mad(read_cv(session_dir / r["dir"] / "data.csv")[0][1]) * 1e9
                               for r in cv_recs if (session_dir / r["dir"] / "data.csv").exists()
                               and 0 in read_cv(session_dir / r["dir"] / "data.csv")])
        ax1.axhspan(-med_noise, med_noise, color="#D55E00", alpha=0.12, zorder=0)
        ax1.text(0.02, 0.97, f"noise band ±{med_noise:.1f} nA\nsmoothed p-p ≈ {np.median(peak2peak):.1f} nA",
                 transform=ax1.transAxes, va="top", fontsize=9, color="#333333")

    for k, rec in enumerate(ca_recs):
        d = session_dir / rec["dir"] / "data.csv"
        if not d.exists():
            continue
        t, i = read_ca(d)
        ax2.plot(t, i * 1e9, lw=1.3, color=PAL[k % len(PAL)],
                 label=f"{rec['dir']}  (t={rec['t_min']:.0f} min)" if np.isfinite(rec["t_min"]) else rec["dir"])
    ax2.set_xlabel("Time  t  (s)"); ax2.set_ylabel("Current  i  (nA)")
    ax2.set_title("CA decays across the session")
    ax2.legend(frameon=False, fontsize=9)
    _style(ax2)
    fig.suptitle("DIAGNOSTIC — is there a measurable signal?", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _scatter_found(ax, x, rows, key, found_key, color, label):
    """Filled marker when a benchmark row has enough found scans; open when not."""
    yv = np.array([r[key] for r in rows], float)
    nf = np.array([r.get(found_key, 1) for r in rows], float)
    solid = nf > 0
    ax.plot(x[solid], yv[solid], "-o", color=color, lw=1.6, ms=6, label=label, zorder=3)
    if np.any(~solid):
        ax.plot(x[~solid], yv[~solid], "o", mfc="white", mec=color, ms=6, zorder=3)


def _fig_gap(cv_files, out):
    if not cv_files:
        return
    x, xl = _xvals(cv_files)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), dpi=200, sharex=True)
    # gap
    ax = axes[0]
    _scatter_found(ax, x, cv_files, "gap_med_mV", "n_scan_found_a", PAL[0], "ΔEp = Epa−Epc")
    ref = _first_finite([r["gap_med_mV"] for r in cv_files])
    if np.isfinite(ref):
        ax.axhspan(ref, ref + TOL["dGap_incr_mV"], color="#009E73", alpha=0.10)
        ax.axhline(ref + TOL["dGap_incr_mV"], color="#D55E00", ls="--", lw=1.1,
                   label=f"tolerance +{TOL['dGap_incr_mV']:.0f} mV")
    ax.set_ylabel("gap  ΔEp  (mV)"); ax.set_title("Peak separation (gap) vs session"); _style(ax); ax.legend(frameon=False, fontsize=9)
    # Epa/Epc
    ax = axes[1]
    _scatter_found(ax, x, cv_files, "Epa_med", "n_scan_found_a", PAL[1], "Epa")
    _scatter_found(ax, x, cv_files, "Epc_med", "n_scan_found_c", PAL[0], "Epc")
    ax.set_ylabel("Ep  (V)"); ax.set_title("Peak potentials (drift)"); _style(ax); ax.legend(frameon=False, fontsize=9)
    # Ep MAD (repeatability)
    ax = axes[2]
    ax.plot(x, [r["Epa_MAD_mV"] for r in cv_files], "-o", color=PAL[1], lw=1.6, ms=6, label="Epa MAD (3 scans)")
    ax.plot(x, [r["Epc_MAD_mV"] for r in cv_files], "-o", color=PAL[0], lw=1.6, ms=6, label="Epc MAD (3 scans)")
    ax.axhline(TOL["Ep_MAD_mV"], color="#D55E00", ls="--", lw=1.1, label=f"tolerance {TOL['Ep_MAD_mV']:.0f} mV")
    ax.set_ylabel("Ep MAD  (mV)"); ax.set_xlabel(xl); ax.set_title("Within-run Ep repeatability"); _style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _fig_current(cv_files, cv_rows, out):
    if not cv_files:
        return
    x, xl = _xvals(cv_files)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), dpi=200, sharex=True)
    ax = axes[0]
    _scatter_found(ax, x, cv_files, "ipa_med_nA", "n_scan_found_a", PAL[1], "ipa")
    _scatter_found(ax, x, cv_files, "ipc_med_nA", "n_scan_found_c", PAL[0], "ipc")
    ax.set_ylabel("ip  (nA)"); ax.set_title("Peak currents (sensitivity)"); _style(ax); ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    refa = _first_finite([r["ipa_med_nA"] for r in cv_files])
    if np.isfinite(refa) and refa != 0:
        pct = [100 * (r["ipa_med_nA"] / refa - 1) for r in cv_files]
        ax.plot(x, pct, "-o", color=PAL[1], lw=1.6, ms=6, label="ipa change %")
        ax.axhspan(-TOL["ip_change_pct"], TOL["ip_change_pct"], color="#009E73", alpha=0.10)
        ax.axhline(TOL["ip_change_pct"], color="#D55E00", ls="--", lw=1.1)
        ax.axhline(-TOL["ip_change_pct"], color="#D55E00", ls="--", lw=1.1, label=f"±{TOL['ip_change_pct']:.0f}%")
    ax.set_ylabel("ipa change (%)"); ax.set_title("Peak-current drift vs run start"); _style(ax); ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    ax.plot(x, [r["ipa_CVpct"] for r in cv_files], "-o", color=PAL[1], lw=1.6, ms=6, label="ipa CV% (3 scans)")
    ax.axhline(TOL["ip_CVpct"], color="#D55E00", ls="--", lw=1.1, label=f"tolerance {TOL['ip_CVpct']:.0f}%")
    ax.set_ylabel("ip CV%"); ax.set_xlabel(xl); ax.set_title("Within-run ip repeatability"); _style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _fig_tail(cv_files, out):
    if not cv_files:
        return
    x, xl = _xvals(cv_files)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
    _scatter_found(ax, x, cv_files, "tail_a_med", "n_scan_found_a", PAL[3], "anodic tail (right/left width)")
    ax.axhline(1.0, color="#999999", ls=":", lw=1.0, label="symmetric (1.0)")
    ax.set_ylabel("tail asymmetry"); ax.set_xlabel(xl); ax.set_title("Peak tailing vs session"); _style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _fig_beta(ca_files, out):
    if not ca_files:
        return
    x = np.array([r["t_min"] for r in ca_files], float)
    xl = "Elapsed time (min)"
    if not np.all(np.isfinite(x)):
        x = np.arange(1, len(ca_files) + 1); xl = "CA index"
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=200)
    beta = np.array([r["beta"] for r in ca_files], float)
    se = np.array([r["beta_se"] for r in ca_files], float)
    ax.errorbar(x, beta, yerr=np.where(np.isfinite(se), se, 0), fmt="-o", color=PAL[2], lw=1.6, ms=7, capsize=3, label="CA beta")
    ax.axhspan(TOL["beta_lo"], TOL["beta_hi"], color="#009E73", alpha=0.10)
    ax.axhline(0.5, color="#999999", ls=":", lw=1.0, label="ideal Cottrell 0.5")
    ax.set_ylabel("beta  (i = A·t^−β + C)"); ax.set_xlabel(xl); ax.set_title("CA diffusional decay exponent vs session")
    _style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _first_finite(seq):
    for v in seq:
        if np.isfinite(v):
            return v
    return np.nan


def _print_summary(cv_files, ca_files):
    print("\n===== CV benchmark summary =====")
    nf = sum(f["n_scan_found_a"] for f in cv_files)
    print(f"CV files: {len(cv_files)}   scans with a FOUND anodic peak (SNR≥{TOL['SNR_MIN']}): {nf}")
    for f in cv_files:
        print(f"  {f['file']:<9} t={f['t_min']:>5.0f}min  found_a={f['n_scan_found_a']}/3 "
              f"gap={f['gap_med_mV'] if np.isfinite(f['gap_med_mV']) else float('nan'):>7.1f}mV "
              f"ipa={f['ipa_med_nA']:>6.1f}nA  noise={f['noise_med_nA']:.1f}nA")
    print("\n===== CA beta summary =====")
    for f in ca_files:
        print(f"  {f['file']:<9} t={f['t_min']:>5.0f}min  beta={f['beta']:.3f}±{f['beta_se']:.3f}  "
              f"r2={f['r2']:.3f}  amp={f['amp_nA']:.1f}nA")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="session folder, e.g. analysis_07172026")
    ap.add_argument("--out", default=None, help="output folder (default: <session_dir>/analysis_result)")
    args = ap.parse_args()
    sdir = Path(args.session_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else (sdir / "analysis_result")
    run(sdir, out)
    print(f"\nWrote CSVs + figures to: {out}")


if __name__ == "__main__":
    main()
